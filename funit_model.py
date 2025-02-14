"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from deprecated.FUNIT.networks import FewShotGen, GPPatchMcResDis


def recon_criterion(predict, target):
    '''
        Use L1 reconstruction loss
    '''
    return torch.mean(torch.abs(predict - target))

class FUNITModel(nn.Module):
    def __init__(self, hp):
        super(FUNITModel, self).__init__()
        self.gen = FewShotGen(hp['gen']) # generater
        self.dis = GPPatchMcResDis(hp['dis']) # discriminator
        self.gen_test = copy.deepcopy(self.gen)

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
            l_c_rec = recon_criterion(xr_gan_feat.mean(3).mean(2), # reconstructed xa to target xa
                                      xa_gan_feat.mean(3).mean(2)) # xa
            l_m_rec = recon_criterion(xt_gan_feat.mean(3).mean(2), # reconstructed xa to target xb
                                      xb_gan_feat.mean(3).mean(2)) # xb

            ######## Equation 5 Reconstruction Loss ##################
            l_x_rec = recon_criterion(xr, xa)

            l_adv = 0.5 * (l_adv_t + l_adv_r) # loss of generator
            acc = 0.5 * (gacc_t + gacc_r)
            l_total = (hp['gan_w'] * l_adv + hp['r_w'] * l_x_rec + hp['fm_w'] * (l_c_rec + l_m_rec))
            l_total.backward()
            return l_total, l_adv, l_x_rec, l_c_rec, l_m_rec, acc

        elif mode == 'dis_update':
            xb.requires_grad_()
            l_real_pre, acc_r, resp_r = self.dis.calc_dis_real_loss(xb, lb)
            # loss of discriminator (distinguish that xb is real, and classify xb as correct label)
            # discrimination accuracy
            # xb intermediate feature map from discriminator of shape (?, nf_out, H, W)

            ##### Regularization on over-large gradient in discriminator
            l_reg_pre = self.dis.calc_grad2(resp_r, xb)
            l_reg = 10 * l_reg_pre
            l_reg.backward(retain_graph=True)

            ###### Equation 4 Discriminator Loss #######
            l_real = hp['gan_w'] * l_real_pre # D loss on real image
            l_real.backward()

            with torch.no_grad():
                c_xa = self.gen.enc_content(xa)
                s_xb = self.gen.enc_class_model(xb)
                xt = self.gen.decode(c_xa, s_xb)

            l_fake_p, acc_f, resp_f = self.dis.calc_dis_fake_loss(xt.detach(), lb)
            l_fake = hp['gan_w'] * l_fake_p # D loss on fake image
            l_fake.backward()

            l_total = l_fake + l_real + l_reg
            acc = 0.5 * (acc_f + acc_r)
            return l_total, l_fake_p, l_real_pre, l_reg_pre, acc

        else:
            assert 0, 'Not support operation'

    @torch.no_grad()
    def test(self, co_data, cl_data):
        self.eval()
        self.gen.eval()
        self.gen_test.eval()
        xa = co_data[0].cuda()
        xb = cl_data[0].cuda()
        la = co_data[1]
        # print(la)
        lb = cl_data[1]
        # print(lb)

        ## The following are produced by the current trained generator
        c_xa_current = self.gen.enc_content(xa)
        # print(c_xa_current.shape)
        s_xa_current = self.gen.enc_class_model(xa)
        # print(s_xa_current.shape)
        s_xb_current = self.gen.enc_class_model(xb)
        # print(s_xb_current.shape)

        xt_current = self.gen.decode(c_xa_current, s_xb_current)
        # print(xt_current.shape)

        xr_current = self.gen.decode(c_xa_current, s_xa_current)
        # print(xr_current.shape)

        ## The following are produced by the initial generator before training
        c_xa = self.gen_test.enc_content(xa)
        s_xa = self.gen_test.enc_class_model(xa)
        s_xb = self.gen_test.enc_class_model(xb)

        xt = self.gen_test.decode(c_xa, s_xb)
        xr = self.gen_test.decode(c_xa, s_xa)

        self.train()
        return xa, xr_current, xt_current, xb, xr, xt

    def translate_k_shot(self, co_data, cl_data, k):
        '''
            Use initial Generator to generate xbar
        '''
        self.eval()
        xa = co_data[0].cuda()
        xb = cl_data[0].cuda()
        c_xa_current = self.gen_test.enc_content(xa)
        if k == 1:
            c_xa_current = self.gen_test.enc_content(xa)
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
        c_xa_current = self.gen_test.enc_content(xa)
        xt_current = self.gen_test.decode(c_xa_current, s_xb_current)
        return xt_current
