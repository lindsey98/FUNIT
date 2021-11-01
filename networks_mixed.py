"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import numpy as np

import torch
from torch import nn
from torch import autograd
import torch.nn.functional as F
from blocks import LinearBlock, Conv2dBlock, ResBlocks, ActFirstResBlock
import sklearn.preprocessing
import torchvision

def pairwise_distance(a, squared=False):
    '''
        Computes the pairwise distance matrix with numerical stability
        :param a: torch.Tensor (M, sz_embedding)
        :param squared: if True, will compute (euclidean_dist)^2
        :return pairwise_distances: distance torch.Tensor (M, M)
    '''

    a_norm = F.normalize(a, p=2, dim=-1)
    inner_prod = torch.mm(a_norm, torch.t(a_norm))

    pairwise_distances_squared = torch.add(
        a.pow(2).sum(dim=1, keepdim=True).expand(a.size(0), -1),
        torch.t(a).pow(2).sum(dim=0, keepdim=True).expand(a.size(0), -1)
    ) - 2 * (torch.mm(a, torch.t(a))) # compute euclidean distance in dot-product way, since ||x-y||^2 = x'x - 2x'y + y'y

    # Deal with numerical inaccuracies. Set small negatives to zero.
    pairwise_distances_squared = torch.clamp(
        pairwise_distances_squared, min=0.0
    )

    # Get the mask where the zero distances are at.
    error_mask = torch.le(pairwise_distances_squared, 0.0)

    # Optionally take the sqrt.
    if squared:
        pairwise_distances = pairwise_distances_squared
    else:
        pairwise_distances = torch.sqrt(
            pairwise_distances_squared + error_mask.float() * 1e-16
        )

    # Undo conditionally adding 1e-16.
    pairwise_distances = torch.mul(
        pairwise_distances,
        (error_mask == False).float()
    )

    # Explicitly set diagonals to zero since it is the distance to itself
    mask_offdiagonals = 1 - torch.eye(
        *pairwise_distances.size(),
        device=pairwise_distances.device
    )
    pairwise_distances = torch.mul(pairwise_distances, mask_offdiagonals)

    return pairwise_distances, inner_prod

def binarize_and_smooth_labels(T, nb_classes, smoothing_const=0):
    '''
        Create smoother gt class labels
        :param T: torch.Tensor of shape (N,), gt class labels
        :param nb_classes: number of classes
        :param smoothing_const: smoothing factor, when =0, no smoothing is applied, just one-hot label
        :return T: torch.Tensor of shape (N, C)
    '''
    T = T.cpu().numpy()
    T = sklearn.preprocessing.label_binarize(
        T, classes=range(0, nb_classes)
    )
    T = T * (1 - smoothing_const)
    T[T == 0] = smoothing_const / (nb_classes - 1)
    T = torch.FloatTensor(T).cuda()

    return T

def assign_adain_params(adain_params, model):
    '''
        Assign the adain_params to the AdaIN layers in model
        Like a feature alignment, align zx to target class (domain)'s distribution
    '''
    for m in model.modules():
        if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
            mean = adain_params[:, :m.num_features]
            std = adain_params[:, m.num_features:2*m.num_features]
            m.bias = mean.contiguous().view(-1)
            m.weight = std.contiguous().view(-1)
            if adain_params.size(1) > 2*m.num_features:
                adain_params = adain_params[:, 2*m.num_features:]

def get_num_adain_params(model):
    # return the number of AdaIN parameters needed by the model
    num_adain_params = 0
    for m in model.modules():
        if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
            num_adain_params += 2*m.num_features
    return num_adain_params


class GPPatchMcResDis(nn.Module):
    '''
        Discriminator which takes image as input, output the class activations and the intermediate activations
    '''
    def __init__(self, hp):
        super(GPPatchMcResDis, self).__init__()
        assert hp['n_res_blks'] % 2 == 0, 'n_res_blk must be multiples of 2'
        self.n_layers = hp['n_res_blks'] // 2
        nf = hp['nf']
        cnn_f = [Conv2dBlock(3, nf, 7, 1, 3,
                             pad_type='reflect',
                             norm='none',
                             activation='none')] # image (B,3,H,W) --> (B,nf,H,W)

        for i in range(self.n_layers - 1):
            nf_out = np.min([nf * 2, 1024])
            cnn_f += [ActFirstResBlock(nf, nf, None, 'lrelu', 'none')]
            cnn_f += [ActFirstResBlock(nf, nf_out, None, 'lrelu', 'none')]
            cnn_f += [nn.ReflectionPad2d(1)]
            cnn_f += [nn.AvgPool2d(kernel_size=3, stride=2)] # (B,nf,H,W) --> (B,nf*2,H/2,W/2) everytime channel depth grows twice upper bound is 1024
            nf = np.min([nf * 2, 1024])

        nf_out = np.min([nf, 1024])
        cnn_f += [ActFirstResBlock(nf, nf, None, 'lrelu', 'none')]
        cnn_f += [ActFirstResBlock(nf, nf_out, None, 'lrelu', 'none')]
        cnn_f += [Conv2dBlock(nf_out, hp['sz_embed'], 1, 1,
                             norm='none',
                             activation='lrelu', # leaky relu
                             activation_first=True)]
        self.cnn_f = nn.Sequential(*cnn_f) # (B,sz_embed,H,W)

        cnn_c = [Conv2dBlock(hp['sz_embed'], hp['num_classes'], 1, 1,
                             norm='none',
                             activation='lrelu', # leaky relu
                             activation_first=True)]
        self.cnn_c = nn.Sequential(*cnn_c) # (B,sz_embed,H,W) -> (B,C,H,W)

        self.scale = 3.

    def forward(self, x, y):
        assert(x.size(0) == y.size(0))
        feat = self.cnn_f(x) # (B,sz_embed,H,W)
        out = self.cnn_c(feat) # (B,C,H,W) for discriminating fake/real
        out = out[torch.arange(out.size()[0]), y, :, :] # (B,H,W) take corresponding channel

        feat = F.adaptive_avg_pool2d(feat, 1) # pooled over H, W
        feat = feat.squeeze()
        return out, feat

    def calc_metric_learning_loss(self, inputxa, inputxb, inputxt, la, lb, proxies):

        _, xb_gan_feat = self.forward(inputxb, lb)
        _, xa_gan_feat = self.forward(inputxa, la)
        _, xt_gan_feat = self.forward(inputxt.detach(), la)

        P = proxies
        P = self.scale * F.normalize(P, p=2, dim=-1)

        X = torch.cat((xa_gan_feat, xb_gan_feat, xt_gan_feat), dim=0)
        X = self.scale * F.normalize(X, p=2, dim=-1)

        D = pairwise_distance(
            torch.cat(
                [X, P]
            ),
            squared=True
        )[0][:X.size()[0], X.size()[0]:]

        TX = binarize_and_smooth_labels(T=la, nb_classes=len(P), smoothing_const=0)
        TY = binarize_and_smooth_labels(T=lb, nb_classes=len(P), smoothing_const=0)
        TXhat = 0.5*TX + 0.5*TY # label interpolation half-half
        Tall = torch.cat((TX, TY, TXhat), dim=0)

        loss = torch.sum(- Tall * F.log_softmax(-D, -1), -1)
        loss = loss.mean()
        return loss

    def calc_dis_fake_loss(self, input_fake, input_label):
        '''
            Activation functions such as ReLU are used to address the vanishing gradient problem in deep convolutional neural networks and promote sparse activations (e.g. lots of zero values).
            ReLU is recommended for the generator, but not for the discriminator model. Instead, a variation of ReLU that allows values less than zero, called Leaky ReLU, is preferred in the discriminator.
        '''
        resp_fake, gan_feat = self.forward(input_fake, input_label)
        total_count = torch.tensor(np.prod(resp_fake.size()), dtype=torch.float).cuda() # total is B

        fake_loss = torch.nn.ReLU()(1.0 + resp_fake).mean() # fake loss = [1 + out]+, then average over batch, W, H
        correct_count = (resp_fake < 0).sum() # fake image should be classified as fake, i.e. < 0
        fake_accuracy = correct_count.type_as(fake_loss) / total_count
        return fake_loss, fake_accuracy, resp_fake

    def calc_dis_real_loss(self, input_real, input_label):
        resp_real, gan_feat = self.forward(input_real, input_label)
        total_count = torch.tensor(np.prod(resp_real.size()), dtype=torch.float).cuda() # total is B

        real_loss = torch.nn.ReLU()(1.0 - resp_real).mean() # real loss = [1 - out]+, then average over batch, W, H
        correct_count = (resp_real >= 0).sum() # real image should be classified as real, i.e. >0
        real_accuracy = correct_count.type_as(real_loss) / total_count
        return real_loss, real_accuracy, resp_real

    def calc_gen_loss(self, input_fake, input_fake_label):
        resp_fake, gan_feat = self.forward(input_fake, input_fake_label)
        total_count = torch.tensor(np.prod(resp_fake.size()), dtype=torch.float).cuda()

        loss = -resp_fake.mean() # fake loss = -out, then average over batch, W, H
        correct_count = (resp_fake >= 0).sum() # generator would like to fool the discriminator into believing fake is real
        accuracy = correct_count.type_as(loss) / total_count
        return loss, accuracy, gan_feat

    def calc_grad2(self, d_out, x_in):
        batch_size = x_in.size(0)
        grad_dout = autograd.grad(outputs=d_out.mean(),
                                  inputs=x_in,
                                  create_graph=True,
                                  retain_graph=True,
                                  only_inputs=True)[0]
        grad_dout2 = grad_dout.pow(2) # square the gradient
        assert (grad_dout2.size() == x_in.size())
        reg = grad_dout2.sum() / batch_size # take mean over batch
        return reg


class FewShotGen(nn.Module):
    '''
        Generator which includes content encoder, class encoder, decoder
        Takes images as input, and output as reconstructed image
    '''
    def __init__(self, hp):
        super(FewShotGen, self).__init__()
        nf = hp['nf']
        nf_mlp = hp['nf_mlp']
        down_class = hp['n_downs_class']
        down_content = hp['n_downs_content']
        n_mlp_blks = hp['n_mlp_blks']
        n_res_blks = hp['n_res_blks']
        latent_dim = hp['latent_dim']

        self.enc_class_model = ClassModelEncoder(down_class,
                                                 3,
                                                 nf,
                                                 latent_dim,
                                                 norm='none',
                                                 activ='relu',
                                                 pad_type='reflect')

        self.enc_content = ContentEncoder(down_content,
                                          n_res_blks,
                                          3,
                                          nf,
                                          'in',
                                          activ='relu',
                                          pad_type='reflect')

        self.dec = Decoder(down_content,
                           n_res_blks,
                           self.enc_content.output_dim,
                           3,
                           res_norm='adain',
                           activ='relu',
                           pad_type='reflect')

        self.mlp = MLP(latent_dim,
                       get_num_adain_params(self.dec),
                       nf_mlp,
                       n_mlp_blks,
                       norm='none',
                       activ='relu')

    def forward(self, one_image, model_set):
        # reconstruct an image
        content, model_codes = self.encode(one_image, model_set)
        model_code = torch.mean(model_codes, dim=0).unsqueeze(0) # (1, nf, 1, 1)
        images_trans = self.decode(content, model_code)
        return images_trans

    def encode(self, one_image, model_set):
        # extract content code from the input image
        content = self.enc_content(one_image) # (1, nf, H, W)
        # extract model code from the images in the model set
        class_codes = self.enc_class_model(model_set)
        class_code = torch.mean(class_codes, dim=0).unsqueeze(0) # (1, nf, 1, 1) first dimension is 1 because we take the mean over all K images
        return content, class_code

    def decode(self, content, model_code):
        # decode content and style codes to an image
        adain_params = self.mlp(model_code)
        assign_adain_params(adain_params, self.dec)
        images = self.dec(content)
        return images


class ContentEncoder(nn.Module):
    def __init__(self, downs, n_res, input_dim, dim, norm, activ, pad_type):
        super(ContentEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3,
                                   norm=norm,
                                   activation=activ,
                                   pad_type=pad_type)]
        for i in range(downs):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1,
                                       norm=norm,
                                       activation=activ,
                                       pad_type=pad_type)]
            dim *= 2

        self.model += [ResBlocks(n_res, dim,
                                 norm=norm,
                                 activation=activ,
                                 pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)

class ClassModelEncoder(nn.Module):
    def __init__(self, downs, ind_im, dim, latent_dim, norm, activ, pad_type):
        super(ClassModelEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(ind_im, dim, 7, 1, 3,
                                   norm=norm,
                                   activation=activ,
                                   pad_type=pad_type)]
        for i in range(2):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1,
                                       norm=norm,
                                       activation=activ,
                                       pad_type=pad_type)]
            dim *= 2
        for i in range(downs - 2):
            self.model += [Conv2dBlock(dim, dim, 4, 2, 1,
                                       norm=norm,
                                       activation=activ,
                                       pad_type=pad_type)]
        self.model += [nn.AdaptiveAvgPool2d(1)] # W and H become 1
        self.model += [nn.Conv2d(dim, latent_dim, 1, 1, 0)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, ups, n_res, dim, out_dim, res_norm, activ, pad_type):
        super(Decoder, self).__init__()

        self.model = []
        self.model += [ResBlocks(n_res, dim, res_norm,
                                 activ, pad_type=pad_type)]
        for i in range(ups):
            self.model += [nn.Upsample(scale_factor=2),
                           Conv2dBlock(dim, dim // 2, 5, 1, 2,
                                       norm='in',
                                       activation=activ,
                                       pad_type=pad_type)]
            dim //= 2 # half the channel size but upsample H, W

        self.model += [Conv2dBlock(dim, out_dim, 7, 1, 3,
                                   norm='none',
                                   activation='tanh', # last layer of Generator should use tanh
                                   pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, dim, n_blk, norm, activ):

        super(MLP, self).__init__()
        self.model = []
        self.model += [LinearBlock(in_dim, dim, norm=norm, activation=activ)]
        for i in range(n_blk - 2):
            self.model += [LinearBlock(dim, dim, norm=norm, activation=activ)]
        self.model += [LinearBlock(dim, out_dim,
                                   norm='none', activation='none')]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))
