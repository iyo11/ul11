import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import pywt
import math

__all__ = ['C3k2_SnakeWT']


# ==============================================================================
# 1. 基础工具函数
# ==============================================================================

def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


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


class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)

    def forward(self, x):
        return torch.mul(self.weight, x)


# ==============================================================================
# 2. 核心模块: Dynamic Snake Conv (DSConv)
# ==============================================================================

class DySnakeConv(nn.Module):
    def __init__(self, channels, kernel_size=3, act=nn.ReLU):
        super().__init__()
        self.kernel_size = kernel_size
        self.channels = channels

        # [Fix 1] 修复 groups 问题: 将 groups 改为 1
        # 即使输入 channel 很大，offset 只需要 2*K 个通道，必须用标准卷积生成
        self.offset_conv = nn.Conv2d(channels, 2 * kernel_size, kernel_size=kernel_size,
                                     stride=1, padding=kernel_size // 2, groups=1, bias=True)

        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)

        self.dw_conv = nn.Conv2d(channels, channels, kernel_size=kernel_size,
                                 stride=1, padding=kernel_size // 2, groups=channels, bias=False)
        self.act = act()

    def forward(self, x):
        b, c, h, w = x.shape
        offset = self.offset_conv(x)

        y_iter, x_iter = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        base_grid = torch.stack([x_iter, y_iter], dim=0).to(x.device).float()
        base_grid = base_grid.unsqueeze(0).repeat(b, 1, 1, 1)

        offset_x = offset[:, :self.kernel_size, :, :].mean(dim=1, keepdim=True)
        offset_y = offset[:, self.kernel_size:, :, :].mean(dim=1, keepdim=True)
        grid_offset = torch.cat([offset_x, offset_y], dim=1)

        final_grid = base_grid + grid_offset
        final_grid[:, 0] = 2.0 * final_grid[:, 0] / max(w - 1, 1) - 1.0
        final_grid[:, 1] = 2.0 * final_grid[:, 1] / max(h - 1, 1) - 1.0
        final_grid = final_grid.permute(0, 2, 3, 1)

        x_deformed = F.grid_sample(x, final_grid, mode='bilinear', padding_mode='reflection', align_corners=True)
        out = self.dw_conv(x_deformed)
        return self.act(out)


# ==============================================================================
# 3. 核心模块: Snake-WTConv2d
# ==============================================================================

class Snake_WTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1'):
        super(Snake_WTConv2d, self).__init__()
        assert in_channels == out_channels
        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride

        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.wt_function = partial(wavelet_transform, filters=self.wt_filter)
        self.iwt_function = partial(inverse_wavelet_transform, filters=self.iwt_filter)

        self.base_conv = nn.Conv2d(in_channels, in_channels, 3, padding=1, stride=1,
                                   groups=in_channels, bias=bias)
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])

        self.wavelet_convs = nn.ModuleList()
        for _ in range(self.wt_levels):
            self.wavelet_convs.append(
                nn.Sequential(
                    DySnakeConv(in_channels * 4, kernel_size=3),
                    nn.Conv2d(in_channels * 4, in_channels * 4, 1, groups=in_channels, bias=False)
                )
            )
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1) for _ in range(self.wt_levels)]
        )

        if self.stride > 1:
            self.stride_filter = nn.Parameter(torch.ones(in_channels, 1, 1, 1), requires_grad=False)
            self.do_stride = lambda x_in: F.conv2d(x_in, self.stride_filter, bias=None, stride=self.stride,
                                                   groups=in_channels)
        else:
            self.do_stride = None

    def forward(self, x):
        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []
        curr_x_ll = x

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = self.wt_function(curr_x_ll)
            curr_x_ll = curr_x[:, :, 0, :, :]

            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)
            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])

        next_x_ll = 0
        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()
            curr_x_ll = curr_x_ll + next_x_ll
            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = self.iwt_function(curr_x)
            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        x_tag = next_x_ll
        x = self.base_scale(self.base_conv(x))
        x = x + x_tag

        if self.do_stride is not None:
            x = self.do_stride(x)
        return x


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