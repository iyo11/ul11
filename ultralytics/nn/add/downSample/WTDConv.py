import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt


# 1. 保持原有的滤波器生成函数 (不需要动)
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


# 2. 引入 SimAM 注意力 (全维度能量函数) - 这是核心升级
class SimAM(nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(SimAM, self).__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.activaton(y)


# 3. 终极版下采样模块
class WTDConv(nn.Module):
    """
    WT_SimAM_Down:
    1. 小波变换提供丰富的频域信息 (无损下采样)。
    2. 普通卷积提供基础语义。
    3. SimAM 提供基于能量的 3D 注意力，精准定位小目标。
    """

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, wt_type='bior1.3'):
        super().__init__()
        self.c1 = c1

        # --- 分支 1: 基础卷积 (Base) ---
        # 负责提取常规语义，保底作用
        self.base_conv = nn.Conv2d(c1, c2, 3, 2, 1, bias=False)
        self.base_bn = nn.BatchNorm2d(c2)

        # --- 分支 2: 小波变换 (Detail) ---
        filters = create_wavelet_filter(wt_type, c1, c1, torch.float)
        self.register_buffer('wt_filter', filters)
        k_wt = filters.shape[-1]
        self.wt_pad = (k_wt - 2) // 2

        # 降维层：把小波的 4C 通道 映射回 C2
        # 这里不加 BN/Act，保持原始频率特征进入 Attention
        self.wt_reduce = nn.Conv2d(c1 * 4, c2, 1, 1, 0, bias=False)

        # --- 核心: SimAM 注意力 ---
        # 作用于融合后的特征，寻找"能量"最高的像素(即目标)
        self.simam = SimAM()

        # --- 最终融合 ---
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        # 1. 语义流 (Base Semantic)
        x_base = self.base_bn(self.base_conv(x))

        # 2. 频率流 (Frequency Detail)
        x_wt = F.conv2d(x, self.wt_filter, stride=2, groups=self.c1, padding=self.wt_pad)
        x_wt = self.wt_reduce(x_wt)  # [B, 4C, H/2, W/2] -> [B, C2, H/2, W/2]

        # 3. 融合 (Add) - 信息互补
        # 此时 x_fuse 既有语义又有高频边缘
        x_fuse = x_base + x_wt

        # 4. 激活 (SimAM Attention)
        # SimAM 会自动评估每个像素的能量，抑制背景(低能量)，激活目标(高能量)
        # 它比 ReLU/SiLU 更适合做特征筛选
        out = self.simam(x_fuse)

        return self.act(out)