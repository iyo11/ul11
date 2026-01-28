import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import pywt

__all = ['C3k2_SnakeWT']

# =========================
# wavelet helpers (保持你原来的也行)
# =========================

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


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
    # 更通用一点的 padding（对 db2/db4 之类更稳）
    pad_h = (filters.shape[2] - 2) // 2
    pad_w = (filters.shape[3] - 2) // 2
    x = F.conv2d(x, filters, stride=2, groups=c, padding=(pad_h, pad_w))
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x


def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad_h = (filters.shape[2] - 2) // 2
    pad_w = (filters.shape[3] - 2) // 2
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=(pad_h, pad_w))
    return x


class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)

    def forward(self, x):
        return x * self.weight


# =========================
# 改进版 DySnakeConv
# =========================

class DySnakeConv(nn.Module):
    """
    更稳健的 snake-style deform:
    - offset 直接输出 2 通道 (dx, dy)
    - tanh 限幅，避免漂移爆掉
    - base grid cache（按 H,W 缓存）
    """
    def __init__(self, channels, kernel_size=3, act=nn.ReLU, max_offset=None):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        # 限幅：默认给 (k//2) 像素
        self.max_offset = float(max_offset if max_offset is not None else (kernel_size // 2))

        # 直接输出 2 通道 (dx, dy)
        self.offset_conv = nn.Conv2d(
            channels, 2, kernel_size=kernel_size, stride=1,
            padding=kernel_size // 2, groups=1, bias=True
        )
        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        self.dw_conv = nn.Conv2d(
            channels, channels, kernel_size=kernel_size,
            stride=1, padding=kernel_size // 2, groups=channels, bias=False
        )
        self.act = act()

        # grid cache: key=(H,W,device)
        self._grid_cache = {}

    @torch.no_grad()
    def _get_base_grid(self, b, h, w, device, dtype):
        key = (h, w, device)
        grid = self._grid_cache.get(key, None)
        if grid is None or grid.dtype != dtype:
            yy = torch.arange(h, device=device, dtype=dtype)
            xx = torch.arange(w, device=device, dtype=dtype)
            y, x = torch.meshgrid(yy, xx, indexing='ij')
            # [1,2,H,W]
            grid = torch.stack([x, y], dim=0).unsqueeze(0)  # x first, y second
            self._grid_cache[key] = grid
        return grid.repeat(b, 1, 1, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        dtype = x.dtype

        offset = self.offset_conv(x)               # [B,2,H,W]
        offset = torch.tanh(offset) * self.max_offset

        base_grid = self._get_base_grid(b, h, w, x.device, dtype)  # [B,2,H,W]
        final_grid = base_grid + offset

        # normalize to [-1,1]
        final_grid[:, 0] = 2.0 * final_grid[:, 0] / max(w - 1, 1) - 1.0
        final_grid[:, 1] = 2.0 * final_grid[:, 1] / max(h - 1, 1) - 1.0
        final_grid = final_grid.permute(0, 2, 3, 1)  # [B,H,W,2]

        # align_corners=False 通常更稳，边缘伪影更少
        x_deformed = F.grid_sample(
            x, final_grid,
            mode='bilinear',
            padding_mode='reflection',
            align_corners=False
        )
        out = self.dw_conv(x_deformed)
        return self.act(out)


# =========================
# 改进版 Snake_WTConv2d
# =========================

class Snake_WTConv2d(nn.Module):
    """
    改进点：
    - filters 自动转到 x 的 dtype/device（兼容 AMP）
    - base_conv 使用 kernel_size 参数（不误导）
    - wavelet 分支加一个可学习 gamma，默认 0（更稳的“安全起步”）
    - stride 用 avg_pool2d（更合理）
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, bias=True,
                 wt_levels=1, wt_type='db1'):
        super().__init__()
        assert in_channels == out_channels
        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride

        wt_filter, iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float32)
        self.register_buffer("wt_filter", wt_filter, persistent=False)
        self.register_buffer("iwt_filter", iwt_filter, persistent=False)

        self.base_conv = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            padding=kernel_size // 2, stride=1,
            groups=in_channels, bias=bias
        )
        self.base_scale = _ScaleModule([1, in_channels, 1, 1], init_scale=1.0)

        self.wavelet_convs = nn.ModuleList()
        self.wavelet_scale = nn.ModuleList()
        for _ in range(self.wt_levels):
            self.wavelet_convs.append(nn.Sequential(
                DySnakeConv(in_channels * 4, kernel_size=3),
                nn.Conv2d(in_channels * 4, in_channels * 4, 1, groups=in_channels, bias=False)
            ))
            self.wavelet_scale.append(_ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1))

        # 安全起步：wavelet 分支默认不干扰（训练更稳）
        self.gamma = nn.Parameter(torch.zeros(1))

        self.do_stride = (stride > 1)

    def forward(self, x):
        # 兼容 AMP：filters 跟着 x 的 dtype/device
        wt_f = self.wt_filter.to(device=x.device, dtype=x.dtype)
        iwt_f = self.iwt_filter.to(device=x.device, dtype=x.dtype)

        wt_fn = partial(wavelet_transform, filters=wt_f)
        iwt_fn = partial(inverse_wavelet_transform, filters=iwt_f)

        x_ll_in_levels, x_h_in_levels, shapes_in_levels = [], [], []
        curr_x_ll = x

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)

            # odd size pad
            if (curr_shape[2] % 2) or (curr_shape[3] % 2):
                curr_x_ll = F.pad(curr_x_ll, (0, curr_shape[3] % 2, 0, curr_shape[2] % 2))

            curr_x = wt_fn(curr_x_ll)                # [B,C,4,H/2,W/2]
            curr_x_ll = curr_x[:, :, 0, :, :]

            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])

        next_x_ll = 0
        for _ in range(self.wt_levels):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll
            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = iwt_fn(curr_x)
            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        x_wt = next_x_ll                           # wavelet branch
        x_base = self.base_scale(self.base_conv(x)) # base branch

        x_out = x_base + self.gamma * x_wt

        if self.do_stride:
            # 比 ones depthwise stride 更稳（不会放大幅值）
            x_out = F.avg_pool2d(x_out, kernel_size=self.stride, stride=self.stride)

        return x_out

# ==============================================================================
# 4. YOLO 架构组件
# ==============================================================================

class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck_SnakeWT(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        if c_ == c2:
            self.cv2 = Snake_WTConv2d(c_, c2, kernel_size=k[1])
        else:
            self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3k(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_SnakeWT(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.m(x)


class C3k2_SnakeWT(nn.Module):
    def __init__(self, c1, c2, n=1, c3k=True, e=0.5, g=1, shortcut=True):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)

        if c3k:
            # [Fix 2] 修复通道不匹配问题: 显式传入 e=1.0
            # C3k 默认 e=0.5 会把通道数减半 (32->16)，但 C3k2 这里的输入实际上是完整的 self.c (32)
            # 所以必须设 e=1.0 保持通道数不变
            self.m = nn.ModuleList(C3k(self.c, self.c, 2, shortcut, g, e=1.0) for _ in range(n))
        else:
            self.m = nn.ModuleList(Bottleneck_SnakeWT(self.c, self.c, shortcut, g) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


if __name__ == "__main__":
    x = torch.randn(2, 64, 64, 64)
    model = C3k2_SnakeWT(c1=64, c2=64, n=2, c3k=True, e=0.5)
    print("Model initialized. Testing forward pass...")
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print("Success! Snake-WTConv integrated into C3k2.")