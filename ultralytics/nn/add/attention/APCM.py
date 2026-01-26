import torch
from torch import nn
import torch.nn.functional as F

__all__ = ['APCM']


class APCM(torch.nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(APCM, self).__init__()
        self.act = nn.Sigmoid()
        self.e_lambda = e_lambda
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.adaptavgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1

        x_pool = self.avgpool(x)
        x_minus_mu_square = (x_pool - x_pool.mean(dim=[2, 3], keepdim=True)).pow(2)
        y_pool = x_minus_mu_square / (
                    4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / (n / 4) + self.e_lambda)) + 0.5
        y = self.act(y_pool)
        y_upsampled = F.interpolate(y, size=(h, w), mode='bilinear', align_corners=False)

        x_global_pool = self.adaptavgpool(x)
        x_global_pool_repeated = x_global_pool.repeat(1, 1, h, w)
        x_minus_mu_global_square = (x - x_global_pool_repeated).pow(2)
        y_global = x_minus_mu_global_square / (
                    4 * (x_minus_mu_global_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        y_global = self.act(y_global)

        y_final = x * (y_upsampled + y_global) / 2
        return y_final