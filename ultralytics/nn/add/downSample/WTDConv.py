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
    V2改进版：频域分离去噪。
    策略：保护 LL (低频语义)，仅对 LH/HL/HH (高频边缘/噪声) 进行软阈值处理。
    """

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, wt_type='bior1.3'):
        super().__init__()
        self.c1 = c1

        # 1. Base Branch (保留语义底色)
        self.base_conv = nn.Conv2d(c1, c2, 3, 2, 1, bias=False)
        self.base_bn = nn.BatchNorm2d(c2)

        # 2. Wavelet Branch
        filters = create_wavelet_filter(wt_type, c1, c1, torch.float)
        self.register_buffer('wt_filter', filters)
        k_wt = filters.shape[-1]
        self.wt_pad = (k_wt - 2) // 2

        # === 修改点 1: 通道级独立阈值 ===
        # 原来是 1 个全局阈值，现在改为 c1 * 3 (对应 LH, HL, HH)
        # 初始化非常重要！设小一点 (0.01) 防止开局就杀死了特征
        self.threshold = nn.Parameter(torch.rand(1, c1 * 3, 1, 1) * 0.01)

        # Attention (涵盖所有频带)
        self.attn = ECA(c1 * 4)

        # Fusion
        if p is None: p = k // 2
        self.wt_process = nn.Sequential(
            nn.Conv2d(c1 * 4, c2, k, s, p, g, bias=False),
            nn.BatchNorm2d(c2)
        )
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        # Branch A: 基础卷积
        x_base = self.base_bn(self.base_conv(x))

        # Branch B: 小波变换
        # x_wt shape: [B, 4*C, H/2, W/2]
        # 通道排列通常是: [LL_c1, LL_c2..., LH_c1..., HL..., HH...] 或者是交错的
        # 为了安全起见，我们需要明确 PyTorch Conv2d with Groups 的输出顺序
        # 这里的 create_wavelet_filter 生成顺序是 LL, LH, HL, HH
        # 所以输出 tensor 在通道维度的排列是:
        # [Img1_LL, Img1_LH, Img1_HL, Img1_HH, Img2_LL, ...]
        # 我们需要把它 reshape 才能分离

        x_wt = F.conv2d(x, self.wt_filter, stride=2, groups=self.c1, padding=self.wt_pad)
        B, C4, H, W = x_wt.shape
        C = C4 // 4

        # Reshape 成 [B, C, 4, H, W] 以便分离频带
        x_wt = x_wt.view(B, C, 4, H, W)

        # === 修改点 2: 频带拆分 (Split) ===
        x_LL = x_wt[:, :, 0, :, :]  # 低频 (纹理/颜色) -> 不去噪！
        x_High = x_wt[:, :, 1:, :, :]  # 高频 (LH, HL, HH) -> [B, C, 3, H, W]

        # === 修改点 3: 仅对高频做软阈值 (Soft Thresholding) ===
        # 阈值 shape 需要对齐: [1, 3*C, 1, 1] -> view -> [1, C, 3, 1, 1]
        thres = self.threshold.view(1, C, 3, 1, 1)

        # 软阈值公式: sign(x) * max(|x| - T, 0)
        x_High_denoised = torch.sign(x_High) * torch.relu(torch.abs(x_High) - thres)

        # 拼回去: [B, C, 4, H, W] -> Flatten -> [B, 4*C, H, W]
        x_out = torch.cat([x_LL.unsqueeze(2), x_High_denoised], dim=2)
        x_out = x_out.view(B, C4, H, W)

        # 后续处理: Attention -> Conv
        x_out = self.attn(x_out)
        x_out = self.wt_process(x_out)

        return self.act(x_base + x_out)

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