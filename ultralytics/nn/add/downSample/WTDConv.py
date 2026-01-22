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


class ScaleModule(nn.Module):
    def __init__(self, channels, init_scale=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1) * init_scale)

    def forward(self, x):
        return x * self.weight


# --- 核心组件：坐标注意力 (Coordinate Attention) ---
class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        # 降维比例，针对小模型可以适当减小 reduction 以保留更多信息
        mip = max(8, inp // reduction)

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # Y轴池化 -> 捕捉水平特征
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # X轴池化 -> 捕捉垂直特征

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.SiLU()  # 使用 SiLU 激活

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        # 1. 分别沿 H 和 W 方向池化
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        # 2. 拼接并降维处理 (捕捉跨方向的相关性)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # 3. 拆分并恢复通道
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        # 4. 生成注意力权重
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        # 5. 双向加权
        out = identity * a_w * a_h
        return out


# --- 结合体：WTDConv + Residual + CA ---
class WTDConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, wt_type='bior1.3'):
        super().__init__()
        # 1. 基础卷积分支 (负责语义 + Recall)
        self.base_conv = nn.Conv2d(c1, c2, 3, 2, 1, bias=False)
        self.base_bn = nn.BatchNorm2d(c2)

        # 2. 小波分支 (负责边缘 + Precision)
        filters = create_wavelet_filter(wt_type, c1, c1, torch.float)
        self.register_buffer('wt_filter', filters)
        k_wt = filters.shape[-1]
        self.wt_pad = (k_wt - 2) // 2

        # Scale层：依然保留，用于软性调节频带权重
        self.scale = ScaleModule(c1 * 4, init_scale=1.5)

        # === 核心升级：使用 CoordAtt 替换 ECA ===
        # 小波后的通道数是 4*c1，reduction设为16保证计算量不大
        self.attn = CoordAtt(c1 * 4, c1 * 4, reduction=16)

        # 融合层
        if p is None: p = k // 2
        self.wt_process = nn.Sequential(
            nn.Conv2d(c1 * 4, c2, k, s, p, g, bias=False),
            nn.BatchNorm2d(c2)
        )

        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        # A. 稳健的语义特征
        x_base = self.base_bn(self.base_conv(x))

        # B. 精细的频域特征
        x_wt = F.conv2d(x, self.wt_filter, stride=2, groups=x.shape[1], padding=self.wt_pad)
        x_wt = self.scale(x_wt)  # 频带加权

        # C. 坐标注意力增强 (精准定位小目标位置)
        # 这里会分别强化 X轴 和 Y轴 上有目标的区域，非常适合极小目标
        x_wt = self.attn(x_wt)

        # D. 降维
        x_wt = self.wt_process(x_wt)

        # E. 残差融合
        return self.act(x_base + x_wt)