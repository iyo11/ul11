import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt

__all = ['C3k2_WTSCConv']

# --- 1. ScConv 核心组件 (SRU & CRU) ---

class SRU(nn.Module):
    def __init__(self, oup_channels, group_num=16, gate_treshold=0.5):
        super().__init__()
        self.gn = nn.GroupNorm(num_channels=oup_channels, num_groups=group_num)
        self.gate_treshold = gate_treshold
        self.sigmoid = nn.Sigmoid()  # Fixed typo: sigomid -> sigmoid

    def forward(self, x):
        gn_x = self.gn(x)
        w_gamma = self.gn.weight / (sum(self.gn.weight) + 1e-10)
        w_gamma = w_gamma.view(1, -1, 1, 1)
        reweigts = self.sigmoid(gn_x * w_gamma)

        w1 = torch.where(reweigts > self.gate_treshold, torch.ones_like(reweigts), reweigts)
        w2 = torch.where(reweigts > self.gate_treshold, torch.zeros_like(reweigts), reweigts)

        x_1, x_2 = w1 * x, w2 * x
        # Reconstruct
        x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)
        x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)
        return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)


class CRU(nn.Module):
    def __init__(self, op_channel, alpha=0.5, squeeze_ratio=2, group_size=2, group_kernel_size=3):
        super().__init__()
        self.up_channel = int(alpha * op_channel)
        self.low_channel = op_channel - self.up_channel

        # Fixed typo: squeeze_radio -> squeeze_ratio
        self.squeeze1 = nn.Conv2d(self.up_channel, self.up_channel // squeeze_ratio, kernel_size=1, bias=False)
        self.squeeze2 = nn.Conv2d(self.low_channel, self.low_channel // squeeze_ratio, kernel_size=1, bias=False)

        self.GWC = nn.Conv2d(self.up_channel // squeeze_ratio, op_channel, kernel_size=group_kernel_size,
                             stride=1, padding=group_kernel_size // 2, groups=group_size)
        self.PWC1 = nn.Conv2d(self.up_channel // squeeze_ratio, op_channel, kernel_size=1, bias=False)

        # --- FIX: Added 'self.' to low_channel ---
        self.PWC2 = nn.Conv2d(self.low_channel // squeeze_ratio, op_channel - self.low_channel // squeeze_ratio,
                              kernel_size=1, bias=False)

        self.advavg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)
        up, low = self.squeeze1(up), self.squeeze2(low)
        Y1 = self.GWC(up) + self.PWC1(up)
        Y2 = torch.cat([self.PWC2(low), low], dim=1)
        out = torch.cat([Y1, Y2], dim=1)
        out = F.softmax(self.advavg(out), dim=1) * out
        out1, out2 = torch.split(out, out.size(1) // 2, dim=1)
        return out1 + out2


# --- 2. WTConv 辅助函数 ---

def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)
    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)
    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)
    return dec_filters, rec_filters


def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x


def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x


# --- 3. 融合模块: WT_ScConv2d ---

class WT_ScConv2d(nn.Module):
    """
    以 WTConv 为主干，融合 ScConv 的 SRU 和 CRU 单元。
    1. 使用 SRU 替换输入端的特征增强。
    2. 使用小波变换进行频域多尺度特征提取。
    3. 使用 CRU 的思想进行最终的特征融合。
    """

    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1'):
        super().__init__()
        assert in_channels == out_channels
        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride

        # 融合 A: 使用 SRU 进行空间维度的特征预处理
        self.sru = SRU(in_channels, group_num=min(16, in_channels))

        # 小波过滤器定义
        wt_filter, iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.register_buffer('wt_filter', wt_filter)
        self.register_buffer('iwt_filter', iwt_filter)

        # 频域分支
        self.wavelet_convs = nn.ModuleList([
            nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size, padding='same', groups=in_channels * 4, bias=False)
            for _ in range(self.wt_levels)
        ])

        # 融合 B: 使用 CRU 思想替换原有的 base_conv 和融合逻辑
        self.cru_fusion = CRU(in_channels, alpha=0.5)

    def forward(self, x):
        # 1. ScConv-SRU 空间重构增强
        x = self.sru(x)

        # 2. WTConv 多尺度处理路径
        curr_x_ll = x
        x_ll_in_levels, x_h_in_levels, shapes_in_levels = [], [], []

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_x_ll = F.pad(curr_x_ll, (0, curr_shape[3] % 2, 0, curr_shape[2] % 2))

            # 执行级联小波变换
            curr_x = wavelet_transform(curr_x_ll, self.wt_filter)
            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_convs[i](curr_x_tag)
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])
            curr_x_ll = curr_x[:, :, 0, :, :]

        # 3. 反小波变换重构
        next_x_ll = 0
        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop() + next_x_ll
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()
            next_x_ll = inverse_wavelet_transform(torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2), self.iwt_filter)
            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        # 4. ScConv-CRU 最终特征融合
        # 将原有的残差连接改为 CRU 结构的深度特征融合
        return self.cru_fusion(x + next_x_ll)


# --- 4. YOLO 适配层 (Conv, Bottleneck, C3k2) ---

def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck_WT_ScConv(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        # 将 WT_ScConv 放在 Bottleneck 的第二层
        self.cv2 = WT_ScConv2d(c_, c2) if c_ == c2 else Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3k_WT_ScConv(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck_WT_ScConv(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3k2_WTSCConv(nn.Module):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            C3k_WT_ScConv(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_WT_ScConv(self.c, self.c, shortcut, g)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# --- 5. 测试代码 ---

if __name__ == "__main__":
    # 输入 B, C, H, W
    input_tensor = torch.randn(1, 64, 128, 128)

    # 实例化模型
    # n=1: 包含一个 Bottleneck_WT_ScConv
    model = C3k2_WTSCConv(64, 64, n=1, c3k=True)

    # 前向传播
    output = model(input_tensor)

    print(f"Input Shape: {input_tensor.shape}")
    print(f"Output Shape: {output.shape}")