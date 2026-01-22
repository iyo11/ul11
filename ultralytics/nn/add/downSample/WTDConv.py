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
    """
    混合增强版：并行使用普通卷积和小波卷积。
    既保留了普通卷积的纹理捕捉能力（保Recall），又利用小波增强边缘（提Precision）。
    """

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, wt_type='bior1.3'):
        super().__init__()
        # 1. 正常的下采样卷积 (保底，负责纹理和语义)
        # 注意：这里用 k=3, s=2 的标准卷积
        self.base_conv = nn.Conv2d(c1, c2, 3, 2, 1, bias=False)
        self.base_bn = nn.BatchNorm2d(c2)

        # 2. 小波分支 (负责提取细微边缘和去噪)
        filters = create_wavelet_filter(wt_type, c1, c1, torch.float)
        self.register_buffer('wt_filter', filters)
        k_wt = filters.shape[-1]
        self.wt_pad = (k_wt - 2) // 2

        self.scale = ScaleModule(c1 * 4, init_scale=1.5)  # 初始化稍微大点
        self.attn = ECA(c1 * 4)

        # 融合层 (降维)
        if p is None: p = k // 2
        self.wt_process = nn.Sequential(
            nn.Conv2d(c1 * 4, c2, k, s, p, g, bias=False),
            nn.BatchNorm2d(c2)
        )

        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        # 分支 A: 传统卷积 (稳)
        x_base = self.base_bn(self.base_conv(x))

        # 分支 B: 小波增强 (细)
        x_wt = F.conv2d(x, self.wt_filter, stride=2, groups=x.shape[1], padding=self.wt_pad)
        x_wt = self.scale(x_wt)
        x_wt = self.attn(x_wt)
        x_wt = self.wt_process(x_wt)

        # 融合：直接相加
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