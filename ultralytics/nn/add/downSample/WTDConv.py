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


class WTDConv(nn.Module):
    """
    WT_SimAM_Down (640p 优化版)
    针对低分辨率，增加了 Learnable ResScale，防止小波特征在深层被稀释。
    """

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, wt_type='bior1.3'):
        super().__init__()
        self.c1 = c1

        # 1. 基础语义分支 (保持语义连贯)
        self.base_conv = nn.Conv2d(c1, c2, 3, 2, 1, bias=False)
        self.base_bn = nn.BatchNorm2d(c2)

        # 2. 小波细节分支
        filters = create_wavelet_filter(wt_type, c1, c1, torch.float)
        self.register_buffer('wt_filter', filters)
        k_wt = filters.shape[-1]
        self.wt_pad = (k_wt - 2) // 2

        # 降维：将 4倍通道 降维到 C2
        self.wt_reduce = nn.Conv2d(c1 * 4, c2, 1, 1, 0, bias=False)
        self.wt_bn = nn.BatchNorm2d(c2)  # 加个BN稳一点

        # 3. SimAM 注意力
        self.simam = SimAM()

        # 4. [关键修改] 可学习的融合权重
        # 在640p下，让网络自己决定听"语义"的还是听"细节"的
        self.alpha = nn.Parameter(torch.ones(1) * 0.5)

        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        # 分支 A: 语义 (Semantic)
        x_base = self.base_bn(self.base_conv(x))

        # 分支 B: 细节 (Detail)
        x_wt = F.conv2d(x, self.wt_filter, stride=2, groups=self.c1, padding=self.wt_pad)
        x_wt = self.wt_bn(self.wt_reduce(x_wt))

        # SimAM 激活：先对细节做一次能量筛选
        x_wt = self.simam(x_wt)

        # 融合：带权重的残差连接
        # 如果 alpha 学出来比较大，说明小波细节很重要
        out = x_base + self.alpha * x_wt

        return self.act(out)