import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt


def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)
    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)
    return dec_filters


class FrequencyGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        attn = self.fc(self.avg_pool(x))
        return x * attn


class WTDConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, wt_type='bior1.3'):
        super().__init__()
        self.c1 = c1

        # 1. 语义流 (Base Branch)
        self.base_conv = nn.Conv2d(c1, c2, 3, 2, 1, bias=False)
        self.base_bn = nn.BatchNorm2d(c2)

        # 2. 小波流 (Wavelet Branch)
        filters = create_wavelet_filter(wt_type, c1, c1, torch.float)
        self.register_buffer('wt_filter', filters)
        self.wt_pad = (filters.shape[-1] - 2) // 2

        # 3. 关键修改：先降维融合，再做门控
        # 以前是 4C -> Attention -> Conv(降维)
        # 现在是 4C -> Conv(降维) -> Gate(筛选) -> 加回
        # 这样 Attention 处理的是浓缩后的特征，抗噪性更好！

        self.wt_reduce = nn.Sequential(
            nn.Conv2d(c1 * 4, c2, 1, 1, 0, bias=False),  # 先把 4C 降到 C2
            nn.BatchNorm2d(c2),
            nn.SiLU()
        )

        # 频率门控 (作用在降维后的 C2 通道上)
        self.gate = FrequencyGate(c2)

        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        # A. 基础语义
        x_base = self.base_bn(self.base_conv(x))

        # B. 小波提取
        x_wt = F.conv2d(x, self.wt_filter, stride=2, groups=self.c1, padding=self.wt_pad)

        # C. 降维融合 (先把 4个频带的信息揉在一起)
        x_wt = self.wt_reduce(x_wt)

        # D. 门控筛选 (在融合后的特征中，筛选出对当前任务重要的通道)
        x_wt = self.gate(x_wt)

        # E. 残差相加
        return self.act(x_base + x_wt)