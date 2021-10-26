import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def weights_init_normal(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class IdentifierSep(nn.Module):
    def __init__(self, cir_len, num_classes, env_dim=16, filters=16, net_type=1):
        super(IdentifierSep, self).__init__()

        self.net_type = net_type
        if net_type == 1:  # Linear
            code_shape = (cir_len)
            self.identifier = IdentifierLinearSep(code_shape, num_classes)  # env_dim
        elif net_type == 2:  # Conv1d
            code_shape = (1, cir_len)
            self.identifier = IdentifierConv1dSep(code_shape, num_classes, filters)  # env_dim
        elif net_type == 3:  # Conv2d
            code_shape = (1, cir_len, cir_len)
            self.identifier = IdentifierConv2dSep(code_shape, num_classes, filters)  # env_dim
        else:
            raise ValueError("Unknown network type for Identifier.")

    def forward(self, inputs):
        # if self.net_type == 1:
        #     inputs_flat  = inputs.view(inputs.size(0), -1)
        #     label, env_code = self.identifier(inputs_flat)
        # elif self.net_type == 2:
        #     inputs = inputs.view(inputs.size(0), 1, inputs.size(1))
        #     label, env_code = self.identifier(inputs)
        # elif self.net_type == 3:
        #     inputs = inputs.view(inputs.size(0), 1, inputs.size(1), inputs.size(1))
        #     label, env_code = self.identifier(inputs)
        if self.net_type == 3:  # expand for conv2d
            inputs  = inputs.view(inputs.size(0), 1, inputs.size(1), 1).expand((inputs.size(0), 1, inputs.size(1), inputs.size(1)))

        label = self.identifier(inputs)  # env_code
        return label


class EstimatorSep(nn.Module):
    def __init__(self, cir_len, num_classes, env_dim=16, filters=16, net_type=1):
        super(EstimatorSep, self).__init__()

        self.net_type = net_type
        if net_type == 1:  # Linear
            code_shape = (cir_len)
            self.estimator = EstimatorLinearSep(num_classes, code_shape, env_dim)
        elif net_type == 2:  # Conv1d
            code_shape = (1, cir_len)
            self.estimator = EstimatorConv1dSep(num_classes, code_shape, filters, env_dim)
        elif net_type == 3:  # Conv2d
            code_shape = (1, cir_len, cir_len)
            self.estimator = EstimatorConv2dSep(num_classes, code_shape, filters, env_dim)
        else:
            raise ValueError("Unknown network type for Estimator.")

    def forward(self, label, inputs):  # env_code
        
        # if self.net_type == 1:
        #     inputs_flat = inputs.view(inputs.size(0), -1)
        #     outputs = self.Estimator(label, inputs)
        # elif self.net_type == 2:
        #     inputs = inputs.view(inputs.size(0), 1, inputs.size(1))
        #     outputs = self.Estimator(label, inputs)
        # elif self.net_type == 3:
        #     inputs = inputs.view(inputs.size(0), 1, inputs.size(1), 1)
        #     outputs = self.Estimator(code, inputs)
        if self.net_type == 3:  # expand for conv2d
            inputs = inputs.view(inputs.size(0), 1, inputs.size(1), 1).expand((inputs.size(0), 1, inputs.size(1), inputs.size(1)))
        
        # outputs = self.Estimator(label, inputs)
        mu, log_sigma, sri = self.estimator(label, inputs)

        return mu, log_sigma, sri


# ------------- Identifier Modules ---------------


class IdentifierLinearSep(nn.Module):
    def __init__(self, code_shape, num_classes):
        super(IdentifierLinearSep, self).__init__()

        # self.env_dim = env_dim
        self.num_classes = num_classes

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.linear_blocks = nn.Sequential(
            *block(int(np.prod(code_shape)), 512, normalize=False),
            *block(512, 256),
            *block(256, 128),
            *block(128, 64),
            nn.Tanh()
        )  # (157) -> (512) -> (256) -> (128) -> (64)

        # self.logit_layer = nn.Linear(64, num_classes)
        self.code_layer = nn.Linear(64, 16)
        self.logit_layer = nn.Sequential(
            nn.Linear(16, num_classes),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        inputs_flat = inputs.view(inputs.size(0), -1)
        feature = self.linear_blocks(inputs_flat)  # (b, cir_len) -> (b, 64)
        code_est = self.code_layer(feature)  # (b, env_dim)
        logit_est = self.logit_layer(code_est)  # (b, num_classes)

        return logit_est  # code_est


class IdentifierConv1dSep(nn.Module):
    def __init__(self, code_shape, num_classes, filters=16):  # env_dim
        super(IdentifierConv1dSep, self).__init__()

        # self.env_dim = env_dim
        self.num_classes = num_classes

        self.init_layer = nn.Sequential(
            nn.Linear(int(np.prod(code_shape)), 128)
        )  # (157/152) -> (128)

        def conv_block(in_filters, out_filters, bn=True):
            block = [
                nn.Conv1d(in_filters, out_filters, 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.25)
            ]
            if bn:
                block.append(nn.BatchNorm1d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *conv_block(1, filters, bn=False),
            *conv_block(filters, filters*2),
            *conv_block(filters*2, filters*4),
            *conv_block(filters*4, filters*8),
        )  # (1, 128) -> (16, 64) -> (32, 32) -> (64, 16) -> (128, 8)

        out_shape = (filters * 8, 8)
        self.code_layer = nn.Linear(int(np.prod(out_shape)), 16)  # env_dim
        # self.logit_layer = nn.Sequential(nn.Linear(int(np.prod(out_shape)), num_classes))
        self.logit_layer = nn.Sequential(
            nn.Linear(16, num_classes),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        # inputs = inputs.view(inputs.size(0), 1, inputs.size(1))
        inputs_flat = inputs.view(inputs.shape[0], -1)  # (b, cir_len)
        feature = self.init_layer(inputs_flat)  # (b, cir_len) -> (b, 128)
        feature_1d = feature.view(feature.shape[0], 1, 128)
        outputs = self.conv_blocks(feature_1d)  # (b, 1, 128) -> (b, 128, 8)

        outputs_flat = outputs.view(outputs.shape[0], -1)  # (b, 128*8)
        code_est = self.code_layer(outputs_flat)  # (b, env_dim)
        logit_est = self.logit_layer(code_est)  # (b, num_classes)

        return logit_est  # code_est


class IdentifierConv2dSep(nn.Module):
    def __init__(self, code_shape, num_classes, filters=16):  # env_dim
        super(IdentifierConv2dSep, self).__init__()

        # self.env_dim = env_dim
        self.num_classes = num_classes

        self.init_layer = nn.Sequential(
            nn.Linear(int(np.prod(code_shape)), 128*128)
        )  # (cir_len*cir_len) -> (128*128)

        def conv_block(in_filters, out_filters, bn=True):
            block = [
                nn.Conv2d(in_filters, out_filters, 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.25)
            ]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *conv_block(code_shape[0], filters, bn=False),
            *conv_block(filters, filters*2),
            *conv_block(filters*2, filters*4),
            *conv_block(filters*4, filters*8),
        )  # (1, 128, 128) -> ... -> (128, 8, 8)

        out_shape = (filters * 8, 8, 8)
        self.code_layer = nn.Linear(int(np.prod(out_shape)), 16)  # env_dim
        # self.logit_layer = nn.Linear(int(np.prod(out_shape)), env_dim)
        self.logit_layer = nn.Sequential(
            nn.Linear(16, num_classes),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        # inputs = inputs.view(inputs.size(0), 1, inputs.size(1), inputs.size(1))
        inputs_flat = inputs.reshape(inputs.size(0), -1)  # (b, cir_len*cir_len)
        feature = self.init_layer(inputs_flat)  # (b, 128*128)
        feature_2d = feature.view(feature.shape[0], 1, 128, 128)
        outputs = self.conv_blocks(feature_2d)  # (b, 1, 128, 128) -> (b, 128, 8, 8)

        outputs_flat = outputs.view(outputs.shape[0], -1)  # (b, 128*8*8)
        code_est = self.code_layer(outputs_flat)  # (b, env_dim)
        logit_est = self.logit_layer(code_est)  # (b, num_classes)

        return logit_est  # code_est


# ------------- Estimator Modules ---------------


class EstimatorLinearSep(nn.Module):
    def __init__(self, num_classes, code_shape, env_dim):
        super(EstimatorLinearSep, self).__init__()

        self.label_embedding = nn.Embedding(num_classes, env_dim)  # nn.Embedding(a, b) with shape (a, b)
        self.label_flatten = nn.Linear(num_classes * env_dim, env_dim)
        
        self.layers = nn.Sequential(
            nn.Linear(env_dim + int(np.prod(code_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 2),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, label, inputs):  # env_code
        label_emb = self.label_embedding(label)  # (b, num_classes) -> (b, env_dim)
        label_emb = self.label_flatten(label_emb.view(label_emb.size(0), -1))
        inputs_cat = torch.cat((inputs.view(inputs.size(0), -1), label_emb), -1)
        outputs = self.layers(inputs_cat)  # (b, env_dim+cir_len) -> (b, 2)
        mu, log_sigma = torch.split(outputs, 1, dim=1)
        noise = torch.randn_like(mu)
        sri = noise * log_sigma.exp() + mu
        # kl_div = self._loss(mu, log_sigma)

        return mu, log_sigma, sri

    # def _loss(self, mu, log_sigma, d_gt, epsilon):  # drop since not suitable
    #     kl_div = 0.5 * torch.sum(((2 * log_sigma).exp() + (mu - d_gt) ** 2)/(epsilon ** 2) - 1 - 2 * log_sigma, dim=1)
    #     kl_div = kl_div.mean()  # account for batch size b

    #     return kl_div


class EstimatorConv1dSep(nn.Module):
    def __init__(self, num_classes, code_shape, env_dim, filters=16):
        super(EstimatorConv1dSep, self).__init__()

        self.label_embedding = nn.Embedding(num_classes, env_dim)  # nn.Embedding(a, b) with shape (a, b)
        self.label_flatten = nn.Linear(num_classes * env_dim, env_dim)
        
        self.init_layer = nn.Sequential(
            nn.Linear(env_dim + int(np.prod(code_shape)), 128),
            nn.LeakyReLU(0.2)
        )  # (cir_len+env_dim) -> (128)

        def conv_block(in_filters, out_filters, bn=True):
            block = [
                nn.Conv1d(in_filters, out_filters, 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.25)
            ]
            if bn:
                block.append(nn.BatchNorm1d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *conv_block(1, filters, bn=False),
            *conv_block(filters, filters*2),
            *conv_block(filters*2, filters*4),
            *conv_block(filters*4, filters*8),
        )  # (1, 128) -> (16, 64) -> (32, 32) -> (64, 16) -> (128, 8)

        out_shape = (filters * 8, 8)
        self.output_layer = nn.Sequential(
            nn.Linear(int(np.prod(out_shape)), 2),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, label, inputs):  # code
        label_emb = self.label_embedding(label)  # (b, num_classes) -> (b, env_dim)
        label_emb = self.label_flatten(label_emb.view(label_emb.size(0), -1))
        inputs_cat = torch.cat((inputs.view(inputs.size(0), -1), label_emb), -1)
        feature = self.init_layer(inputs_cat)  # (b, env_dim+cir_len) -> (b, 128)
        feature_1d = feature.view(feature.shape[0], 1, 128)
        feature_out = self.conv_blocks(feature_1d)  # (b, 1, 128) -> (b, 128, 8)
        feature_flat = feature_out.view(feature_out.size(0), -1)
        outputs = self.output_layer(feature_flat)
        mu, log_sigma = torch.split(outputs, 1, dim=1)
        noise = torch.randn_like(mu)
        sri = noise * log_sigma.exp() + mu

        return mu, log_sigma, sri

    # def _loss(self, mu, log_sigma):
    #     kl_div = 0.5 * torch.sum((2 * log_sigma).exp() + mu ** 2 - 1 - 2 * log_sigma, dim=1)
    #     kl_div = kl_div.mean()

    #     return kl_div


class EstimatorConv2dSep(nn.Module):
    def __init__(self, num_classes, code_shape, env_dim, filters=16):
        super(EstimatorConv2dSep, self).__init__()

        self.label_embedding = nn.Embedding(num_classes, env_dim)  # nn.Embedding(a, b) with shape (a, b)
        self.label_flatten = nn.Linear(num_classes * env_dim, env_dim)

        self.init_layer = nn.Sequential(
            nn.Linear(env_dim + int(np.prod(code_shape)), 128*128),
            nn.LeakyReLU(0.2)
        )  # (cir_len*cir_len_env_dim) -> (128*128)

        def conv_block(in_filters, out_filters, bn=True):
            block = [
                nn.Conv2d(in_filters, out_filters, 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.25)
            ]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *conv_block(1, filters, bn=False),
            *conv_block(filters, filters*2),
            *conv_block(filters*2, filters*4),
            *conv_block(filters*4, filters*8),
        )  # (1, 128, 128) -> (16, 64, 64) -> ... -> (128, 8, 8)

        out_shape = (filters * 8, 8, 8)
        self.output_layer = nn.Sequential(
            nn.Linear(int(np.prod(out_shape)), 2),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, label, inputs):  # code
        label_emb = self.label_embedding(label)  # (b, num_classes) -> (b, env_dim)
        label_emb = self.label_flatten(label_emb.reshape(label_emb.size(0), -1))
        inputs_cat = torch.cat((inputs.reshape(inputs.size(0), -1), label_emb), -1)
        feature = self.init_layer(inputs_cat)  # (b, 128*128)
        feature_2d = feature.view(feature.size(0), 1, 128, 128)
        feature_out = self.conv_blocks(feature_2d)  # (b, 128, 8, 8)
        feature_flat = feature_out.view(feature_out.size(0), -1)
        outputs = self.output_layer(feature_flat)  # (b, 128*8*8) -> (b, 1)
        mu, log_sigma = torch.split(outputs, 1, dim=1)
        noise = torch.randn_like(mu)
        sri = noise * log_sigma.exp() + mu
        # kl_div = self._loss(mu, log_sigma)

        return mu, log_sigma, sri

    # def _loss(self, mu, log_sigma):  # unsupervised term
    #     kl_div = 0.5 * torch.sum((2 * log_sigma).exp() + mu ** 2 - 1 - 2 * log_sigma, dim=1)
    #     kl_div = kl_div.mean()

    #     return kl_div
