"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from networks_mixed import FewShotGen, GPPatchMcResDis
import time
from utils import calc_recall_at_k, calc_normalized_mutual_information, cluster_by_kmeans, assign_by_euclidian_at_k
import logging
from tqdm import tqdm

def recon_criterion(predict, target):
    return torch.mean(torch.abs(predict - target))

class FUNITModel(nn.Module):

    def __init__(self, hp):
        super(FUNITModel, self).__init__()
        self.gen = FewShotGen(hp['gen']) # generater
        self.dis = GPPatchMcResDis(hp['dis']) # discriminator
        self.gen_test = copy.deepcopy(self.gen)
        self.dis_test = copy.deepcopy(self.dis)
        self.proxies = torch.nn.Parameter(torch.randn(hp['dis']['num_classes'], hp['dis']['sz_embed']) / 8)
        self.scale = 3.

    def forward(self, co_data, cl_data, hp, mode):

        xa = co_data[0].cuda() # content image
        la = co_data[1].cuda() # content image's class label

        xb = cl_data[0].cuda() # class images
        lb = cl_data[1].cuda() # class images' class labels

        if mode == 'gen_update':
            c_xa = self.gen.enc_content(xa) # call content encoder
            s_xa = self.gen.enc_class_model(xa) # also feed itself into class encoder
            s_xb = self.gen.enc_class_model(xb) # feed target class images into class encoder

            ###### Equation 4 Generator Loss #######
            xr = self.gen.decode(c_xa, s_xa)  # reconstruct by original image xa and its mapped image (by class encoder) s_xa
            l_adv_r, gacc_r, xr_gan_feat = self.dis.calc_gen_loss(xr, la) # loss of generator (fool discriminator), generator success rate, xr's feature after discriminator

            xt = self.gen.decode(c_xa, s_xb)  # xbar, reconstruct by original image xa and encoded target class image s_xb
            l_adv_t, gacc_t, xt_gan_feat = self.dis.calc_gen_loss(xt, lb) # _, _, xt's feature after discriminator

            _, xb_gan_feat = self.dis(xb, lb)
            _, xa_gan_feat = self.dis(xa, la)

            ######## Equation 6 Feature Matching Loss ##################
            l_c_rec = recon_criterion(xr_gan_feat, # reconstructed xa to target xa
                                      xa_gan_feat) # xa

            l_m_rec = recon_criterion(xt_gan_feat, # reconstructed xa to target xb
                                      (xb_gan_feat + xa_gan_feat)/2) # xb

            ######## Equation 5 Reconstruction Loss ##################
            l_x_rec = recon_criterion(xr, xa)

            l_adv = 0.5 * (l_adv_t + l_adv_r) # loss of generator
            acc = 0.5 * (gacc_t + gacc_r)
            l_total = (hp['gan_w'] * l_adv + hp['r_w'] * l_x_rec + hp['fm_w'] * (l_c_rec + l_m_rec) )
            l_total.backward()
            return l_total, l_adv, l_x_rec, l_c_rec, l_m_rec, acc

        elif mode == 'dis_update':
            xb.requires_grad_()
            l_real_pre, acc_r, resp_r = self.dis.calc_dis_real_loss(xb, lb) # loss of discriminator (distinguish that xb is real, and classify xb as correct label)

            ##### Regularization on gradient in discriminator on REAL images
            l_reg_pre = self.dis.calc_grad2(resp_r, xb)
            l_reg = 10 * l_reg_pre
            l_reg.backward(retain_graph=True)

            ###### Equation 4 Discriminator Loss #######
            l_real = hp['gan_w'] * l_real_pre # D loss on real image
            l_real.backward()

            with torch.no_grad(): # dont update generator now
                c_xa = self.gen.enc_content(xa)
                s_xb = self.gen.enc_class_model(xb)
                xt = self.gen.decode(c_xa, s_xb)

            l_fake_p, acc_f, resp_f = self.dis.calc_dis_fake_loss(xt.detach(), lb)
            l_fake = hp['gan_w'] * l_fake_p # D loss on fake image
            l_fake.backward()

            ######## Metric Learning Loss ########
             # FIXME: detach xt or not
            l_metric = self.dis.calc_metric_learning_loss(xa, xb, xt.detach(), la, lb,
                                                          self.proxies, self.scale)
            l_metric = 100 * hp['gan_w'] * l_metric
            l_metric.backward()

            l_total = l_fake + l_real + l_reg + l_metric
            acc = 0.5 * (acc_f + acc_r)
            return l_total, l_fake_p, l_real_pre, l_reg_pre, l_metric, acc

        else:
            assert 0, 'Not support operation'

    @torch.no_grad()
    def test(self, co_data, cl_data):
        self.eval()
        self.gen.eval()
        self.gen_test.eval()
        self.dis_test.eval()
        xa = co_data[0].cuda()
        xb = cl_data[0].cuda()
        la = co_data[1].cuda()
        lb = cl_data[1].cuda()

        ## The following are produced by the current trained generator
        c_xa_current, z_xa_current = self.gen.enc_content(xa)
        s_xa_current = self.gen.enc_class_model(xa)
        s_xb_current = self.gen.enc_class_model(xb)
        xt_current = self.gen.decode(c_xa_current, s_xb_current)
        xr_current = self.gen.decode(c_xa_current, s_xa_current)

        ## The following are produced by the historical average generator before training
        c_xa, z_xa = self.gen_test.enc_content(xa)
        s_xa = self.gen_test.enc_class_model(xa)
        s_xb = self.gen_test.enc_class_model(xb)

        xt = self.gen_test.decode(c_xa, s_xb)
        xr = self.gen_test.decode(c_xa, s_xa)

        # The following are produced by current discriminator
        _, xa_feat_current = self.dis(xa, la)
        _, xb_feat_current = self.dis(xb, lb)
        # The following are produced by the historical average discriminator
        _, xa_feat = self.dis_test(xa, la)
        _, xb_feat = self.dis_test(xb, lb)

        self.train()
        return (xa, xr_current, xt_current, xb, xr, xt), (xa_feat_current, xb_feat_current, xa_feat, xb_feat)

    @torch.no_grad()
    def predict_batchwise(self, dataloader):
        self.dis_test.eval()
        X = torch.tensor([])
        T = torch.tensor([])
        # extract batches (A becomes list of samples)
        for batch in tqdm(dataloader, desc="Batch-wise prediction"):
            x, y, _ = batch
            # move images to device of model (approximate device)
            x, y = x.cuda(), y.cuda()
            # predict model output for image
            feat = self.dis_test.forward_partial(x, y)
            X = torch.cat((X, feat.detach().cpu()), dim=0)
            T = torch.cat((T, y.detach().cpu()), dim=0)
        self.dis_test.train()
        return X, T

    @torch.no_grad()
    def evaluate(self, dataloader, eval_nmi=True, recall_list=[1, 2, 4, 8]):
        eval_time = time.time()
        nb_classes = len(dataloader.dataset.classes)
        # calculate embeddings with model and get targets
        X, T = self.predict_batchwise(dataloader)
        print('done collecting prediction')

        if eval_nmi:
            # calculate NMI with kmeans clustering
            nmi = calc_normalized_mutual_information(
                T,
                cluster_by_kmeans(X, nb_classes)
            )
        else:
            nmi = 1

        print("NMI: {:.3f}".format(nmi * 100))

        # get predictions by assigning nearest 8 neighbors with euclidian
        max_dist = max(recall_list)
        Y = assign_by_euclidian_at_k(X, T, max_dist)
        Y = torch.from_numpy(Y)

        # calculate recall @ 1, 2, 4, 8
        recall = []
        for k in recall_list:
            r_at_k = calc_recall_at_k(T, Y, k)
            recall.append(r_at_k)
            print("R@{} : {:.3f}".format(k, 100 * r_at_k))

        chmean = (2 * nmi * recall[0]) / (nmi + recall[0])
        print("hmean: %s", str(chmean))

        eval_time = time.time() - eval_time
        logging.info('Eval time: %.2f' % eval_time)
        return nmi, recall

    def translate_k_shot(self, co_data, cl_data, k):
        '''
            Use initial Generator to generate xbar
        '''
        self.eval()
        xa = co_data[0].cuda()
        xb = cl_data[0].cuda()
        c_xa_current, z_xa_current = self.gen_test.enc_content(xa)
        if k == 1:
            c_xa_current, z_xa_current = self.gen_test.enc_content(xa)
            s_xb_current = self.gen_test.enc_class_model(xb)
            xt_current = self.gen_test.decode(c_xa_current, s_xb_current)
        else:
            s_xb_current_before = self.gen_test.enc_class_model(xb)
            s_xb_current_after = s_xb_current_before.squeeze(-1).permute(1, 2, 0)
            s_xb_current_pool = F.avg_pool1d(s_xb_current_after, k)
            s_xb_current = s_xb_current_pool.permute(2, 0, 1).unsqueeze(-1)
            xt_current = self.gen_test.decode(c_xa_current, s_xb_current)
        return xt_current

    def compute_k_style(self, style_batch, k):
        # Use initial Generator to encode K images into single feature representation
        self.eval()
        style_batch = style_batch.cuda()
        s_xb_before = self.gen_test.enc_class_model(style_batch)
        s_xb_after = s_xb_before.squeeze(-1).permute(1, 2, 0)
        s_xb_pool = F.avg_pool1d(s_xb_after, k)
        s_xb = s_xb_pool.permute(2, 0, 1).unsqueeze(-1)
        return s_xb

    def translate_simple(self, content_image, class_code):
        self.eval()
        xa = content_image.cuda()
        s_xb_current = class_code.cuda()
        c_xa_current, z_xa_current = self.gen_test.enc_content(xa)
        xt_current = self.gen_test.decode(c_xa_current, s_xb_current)
        return xt_current
