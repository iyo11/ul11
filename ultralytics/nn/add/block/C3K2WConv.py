import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from ultralytics.nn.modules.conv import autopad
from ultralytics.nn.modules.block import C3k
from ultralytics.nn.modules import Bottleneck, C3k2

__all__ = ['C3k2_WConv']


class WConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, den, stride=1, padding=1, groups=1, dilation=1,
                 bias=False):
        super(WConv2d, self).__init__()
        self.stride = _pair(stride)
        self.kernel_size = _pair(kernel_size)
        self.padding = autopad(self.kernel_size, d=dilation)
        self.groups = groups
        self.dilation = _pair(dilation)
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *self.kernel_size))
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        device = torch.device('cpu')
        self.register_buffer('alfa', torch.cat([torch.tensor(den, device=device), torch.tensor([1.0], device=device),
                                                torch.flip(torch.tensor(den, device=device), dims=[0])]))
        self.register_buffer('Phi', torch.outer(self.alfa, self.alfa))

        if self.Phi.shape != self.kernel_size:
            raise ValueError(f"Phi shape {self.Phi.shape} must match kernel size {self.kernel_size}")

    def forward(self, x):
        Phi = self.Phi.to(x.device)
        weight_Phi = self.weight * Phi
        return F.conv2d(x, weight_Phi, bias=self.bias, stride=self.stride, padding=self.padding, groups=self.groups,
                        dilation=self.dilation)


class Bottleneck_WConv(Bottleneck):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), den=None, e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = WConv2d(c1, c_, k[0], den, padding=k[0] // 2)
        self.cv2 = WConv2d(c_, c2, k[1], den, padding=k[0] // 2)


class C3k_WConv(C3k):
    def __init__(self, c1, c2, n=1, den=None, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_WConv(c_, c_, shortcut, g, k=(k, k), den=den, e=1.0) for _ in range(n)))


class C3k2_WConv(C3k2):
    def __init__(self, c1, c2, n=1, den=None, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(
            C3k_WConv(self.c, self.c, 2, den, shortcut, g) if c3k else Bottleneck_WConv(self.c, self.c, shortcut, g,
                                                                                        den=den) for _ in range(n))