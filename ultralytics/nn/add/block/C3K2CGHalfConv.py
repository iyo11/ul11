import torch
import torch.nn as nn
from torch import Tensor
from ultralytics.nn.modules.block import C3k
from ultralytics.nn.modules import Bottleneck, Conv, C3k2

__all__ = ['C3k2_CGHalfConv']


class HalfConv(nn.Module):
    def __init__(self, dim, n_div=2):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        return x


class CGHalfConv(nn.Module):
    def __init__(self, dim):
        super(CGHalfConv, self).__init__()
        self.div_dim = int(dim / 3)
        self.remainder_dim = dim % 3
        self.p1 = HalfConv(self.div_dim, 2)
        self.p2 = HalfConv(self.div_dim, 2)
        self.p3 = HalfConv(self.div_dim + self.remainder_dim, 2)

    def forward(self, x):
        # 保留输入用于残差连接
        y = x
        # 将输入在通道维度上拆分为三部分
        x1, x2, x3 = torch.split(x, [self.div_dim, self.div_dim, self.div_dim + self.remainder_dim], dim=1)
        # 分别送入对应的 HalfConv 模块
        x1 = self.p1(x1)
        x2 = self.p2(x2)
        x3 = self.p3(x3)
        # 拼接处理后的三个部分
        x = torch.cat((x1, x2, x3), 1)
        # 加上残差，增强训练稳定性和特征表达能力
        return x + y


class Bottleneck_CGHalfConv(Bottleneck):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = CGHalfConv(c_)


class C3k_CGHalfConv(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_CGHalfConv(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class C3k2_CGHalfConv(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(
            C3k_CGHalfConv(self.c, self.c, 2, shortcut, g, e=1) if c3k else Bottleneck_CGHalfConv(self.c, self.c,
                                                                                                  shortcut, g, e=1) for
            _ in range(n))