import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt

def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    # 只需要分解滤波器 (dec_hi, dec_lo)，不需要重构滤波器
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)
    # [4, 1, h, w] -> [4*in_channels, 1, h, w]
    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)
    return dec_filters

class ScaleModule(nn.Module):
    def __init__(self, channels, init_scale=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1) * init_scale)

    def forward(self, x):
        return x * self.weight


class WTDConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, wt_type='bior1.3'):
        super().__init__()
        self.c1 = c1

        # 1. 初始化小波核
        filters = create_wavelet_filter(wt_type, c1, c1, torch.float)
        self.register_buffer('wt_filter', filters)

        # --- 核心修复：动态计算 Padding ---
        # 获取生成的小波核大小 (bior1.3 是 6, db1 是 2, db2 是 4)
        k_wt = filters.shape[-1]
        # 计算 padding 以保证 H_out = H_in / 2
        self.wt_pad = (k_wt - 2) // 2

        # 2. Scale 模块
        self.scale = ScaleModule(c1 * 4)

        # 3. Attention 模块
        self.attn = ECA(c1 * 4)

        # 4. 融合卷积 (处理 p=None 的情况)
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(c1 * 4, c2, k, s, p, g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        y = F.conv2d(x, self.wt_filter, stride=2, groups=self.c1, padding=self.wt_pad)

        y = self.scale(y)
        y = self.attn(y)
        return self.act(self.bn(self.conv(y)))


# 简单的 ECA 模块 (保持不变)
class ECA(nn.Module):
    def __init__(self, c, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y)