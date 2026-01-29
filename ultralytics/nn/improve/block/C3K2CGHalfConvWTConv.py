import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import pywt
import pywt.data

__all__ = ['C3k2_CGHalfWTConv']

# =========================================================
# 1) CGHalfConv (更保守版：n_div 默认=4，避免过抑制)
# =========================================================

class HalfConv(nn.Module):
    """
    将通道分成 [conv_part, untouched_part]，只对 conv_part 做 DWConv
    n_div 越大：参与卷积的通道越少 → 越保守 → 更利于 mAP50/召回
    """
    def __init__(self, dim, n_div=4, k=5):
        super().__init__()
        assert n_div >= 2
        self.dim_conv = dim // n_div
        self.dim_untouched = dim - self.dim_conv
        p = k // 2
        self.partial_conv = nn.Conv2d(self.dim_conv, self.dim_conv, k, 1, p, groups=self.dim_conv, bias=False)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.dim_conv, self.dim_untouched], dim=1)
        x1 = self.partial_conv(x1)
        return torch.cat((x1, x2), dim=1)


class CGHalfConv(nn.Module):
    """
    3-way split + each part HalfConv + residual
    """
    def __init__(self, dim, k=5, n_div=4):
        super().__init__()
        self.div_dim = dim // 3
        self.rem_dim = dim - 2 * self.div_dim

        self.p1 = HalfConv(self.div_dim, n_div=n_div, k=k)
        self.p2 = HalfConv(self.div_dim, n_div=n_div, k=k)
        self.p3 = HalfConv(self.rem_dim, n_div=n_div, k=k)

    def forward(self, x):
        shortcut = x
        x1, x2, x3 = torch.split(x, [self.div_dim, self.div_dim, self.rem_dim], dim=1)
        x1 = self.p1(x1)
        x2 = self.p2(x2)
        x3 = self.p3(x3)
        x = torch.cat((x1, x2, x3), dim=1)
        return x + shortcut


# =========================================================
# 2) Wavelet utils (保持你原逻辑)
# =========================================================

def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([
        dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
        dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
        dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
        dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)
    ], dim=0)
    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack([
        rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
        rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
        rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
        rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)
    ], dim=0)
    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)
    return dec_filters, rec_filters


def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    return x.reshape(b, c, 4, h // 2, w // 2)


def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    return F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)


class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)

    def forward(self, x):
        return self.weight * x


# =========================================================
# 3) WTConv2d Vehicle-friendly 版本（核心：alpha/beta 残差融合）
# =========================================================

class WTConv2d(nn.Module):
    """
    Vehicle-friendly WTConv2d:
      - base 分支改为 CGHalfConv（更保守可调 n_div）
      - 引入 identity residual 混合：x_out = (1-a)*x + a*base + b*x_tag
        其中 a,b 为可学习标量（每通道/全局都行，这里用全局标量最稳）
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=5,
        stride=1,
        bias=True,
        wt_levels=1,
        wt_type='db1',
        # 新增参数
        base_k=5,
        base_n_div=4,          # 默认更保守：只卷 1/4 通道
        alpha_init=0.50,       # identity 与 base 的混合系数
        beta_init=0.25,        # wavelet 重构的强度（建议小一点更稳）
        wavelet_scale_init=0.05 # 原来 0.1 偏激进，这里默认 0.05
    ):
        super().__init__()
        assert in_channels == out_channels
        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride

        # wavelet filters
        wt_filter, iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(iwt_filter, requires_grad=False)

        self.wt_function = partial(wavelet_transform, filters=self.wt_filter)
        self.iwt_function = partial(inverse_wavelet_transform, filters=self.iwt_filter)

        # base conv: CGHalfConv (DW-partial) + scale
        self.base_conv = CGHalfConv(in_channels, k=base_k, n_div=base_n_div)
        self.base_scale = _ScaleModule([1, in_channels, 1, 1], init_scale=1.0)

        # wavelet convs
        self.wavelet_convs = nn.ModuleList([
            nn.Conv2d(
                in_channels * 4, in_channels * 4,
                kernel_size, stride=1, padding='same', dilation=1,
                groups=in_channels * 4, bias=False
            ) for _ in range(self.wt_levels)
        ])
        self.wavelet_scale = nn.ModuleList([
            _ScaleModule([1, in_channels * 4, 1, 1], init_scale=wavelet_scale_init)
            for _ in range(self.wt_levels)
        ])

        # stride downsample (same as you)
        if self.stride > 1:
            self.stride_filter = nn.Parameter(torch.ones(in_channels, 1, 1, 1), requires_grad=False)
            self.do_stride = lambda x_in: F.conv2d(x_in, self.stride_filter, bias=None, stride=self.stride,
                                                  groups=in_channels)
        else:
            self.do_stride = None

        # -------- 新增：可学习融合系数（标量，最稳、最不容易炸） --------
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))  # base 与 identity 的混合
        self.beta  = nn.Parameter(torch.tensor(float(beta_init)))   # wavelet 分支强度

    def forward(self, x):
        # --- wavelet branch ---
        x_ll_in_levels, x_h_in_levels, shapes_in_levels = [], [], []
        curr_x_ll = x

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)

            # pad if odd
            if (curr_shape[2] % 2) or (curr_shape[3] % 2):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = self.wt_function(curr_x_ll)      # (b,c,4,h/2,w/2)
            curr_x_ll = curr_x[:, :, 0, :, :]         # LL

            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])

        next_x_ll = 0
        for _ in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h  = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll
            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = self.iwt_function(curr_x)
            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        x_tag = next_x_ll  # wavelet reconstructed feature

        # --- base branch (CGHalf) ---
        base = self.base_scale(self.base_conv(x))

        # --- fusion (关键) ---
        # clamp 防止 alpha/beta 训练到奇怪范围导致不稳定
        a = torch.clamp(self.alpha, 0.0, 1.0)
        b = torch.clamp(self.beta,  0.0, 1.0)

        out = (1.0 - a) * x + a * base + b * x_tag

        if self.do_stride is not None:
            out = self.do_stride(out)

        return out


# ==========================================
# 4. YOLO/Common 模块 (C3k2 的依赖)
# ==========================================

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class Bottleneck_WTConv(nn.Module):
    """
    使用修改后的 WTConv2d 的 Bottleneck
    """

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        if c_ == c2:
            # 这里调用的是修改后的 WTConv2d
            self.cv2 = WTConv2d(c_, c2, 5, 1)
        else:
            self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3k(C3):
    """C3k 模块，内部使用 Bottleneck_WTConv"""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # 使用 WTConv (含 CGHalfConv)
        self.m = nn.Sequential(*(Bottleneck_WTConv(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


# ==========================================
# 5. C3k2_WTConv (目标模块)
# ==========================================

class C3k2_CGHalfWTConv(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_WTConv(self.c, self.c, shortcut, g) for _ in
            range(n)
        )


# ==========================================
# 6. 测试运行
# ==========================================

if __name__ == "__main__":
    # Generating Sample image
    image_size = (1, 64, 240, 240)
    image = torch.rand(*image_size)

    # Model
    print("Building Model...")
    # 实例化 C3k2_WTConv
    model = C3k2_CGHalfWTConv(64, 64, n=1, c3k=True)  # 可以尝试 c3k=True 或 False

    print("Testing Forward Pass...")
    out = model(image)
    print(f"Output shape: {out.size()}")
    print("Success! WTConv2d inside C3k2 is now using CGHalfConv.")