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
        # 1. Base Branch (不变)
        self.base_conv = nn.Conv2d(c1, c2, 3, 2, 1, bias=False)
        self.base_bn = nn.BatchNorm2d(c2)

        # 2. Wavelet Branch
        filters = create_wavelet_filter(wt_type, c1, c1, torch.float)
        self.register_buffer('wt_filter', filters)
        k_wt = filters.shape[-1]
        self.wt_pad = (k_wt - 2) // 2

        # --- 修改点：使用软阈值代替简单的 Scale ---
        # 专门针对高频分量 (LH, HL, HH) 进行去噪
        self.threshold = nn.Parameter(torch.rand(1, c1 * 4, 1, 1) * 0.05)  # 初始化一个小阈值

        # Attention
        self.attn = ECA(c1 * 4)

        # Fusion Conv
        if p is None: p = k // 2
        self.wt_process = nn.Sequential(
            nn.Conv2d(c1 * 4, c2, k, s, p, g, bias=False),
            nn.BatchNorm2d(c2)
        )
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        # Branch A
        x_base = self.base_bn(self.base_conv(x))

        # Branch B: WT -> Soft Thresholding -> Attn -> Conv
        x_wt = F.conv2d(x, self.wt_filter, stride=2, groups=x.shape[1], padding=self.wt_pad)

        # === 核心修改：软阈值去噪 ===
        # 自动学习把绝对值小的系数变成0 (去噪)
        # 类似于: x = sign(x) * max(|x| - threshold, 0)
        x_wt = torch.sign(x_wt) * torch.relu(torch.abs(x_wt) - self.threshold)

        x_wt = self.attn(x_wt)
        x_wt = self.wt_process(x_wt)

        return self.act(x_base + x_wt)

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