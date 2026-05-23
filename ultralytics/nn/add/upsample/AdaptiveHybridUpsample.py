import torch
import torch.nn as nn
import torch.nn.functional as F


def _normal_init(module, std=0.001):
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.normal_(module.weight, std=std)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, 0)


class DySample(nn.Module):
    """
    Dynamic Sample upsample module.
    From: Learning to Upsample by Learning to Sample.

    Input : [B, C, H, W]
    Output: [B, C, 2H, 2W]
    """

    def __init__(self, c1, scale=2, style="lp", groups=4):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups

        assert style in ["lp", "pl"]
        if style == "pl":
            assert c1 >= scale ** 2 and c1 % scale ** 2 == 0

        # 避免 YOLOv11n 小通道时 groups 不整除
        if c1 < groups or c1 % groups != 0:
            groups = 1
            self.groups = 1

        offset_channels = 2 * groups * scale ** 2 if style == "lp" else 2 * groups
        in_ch = c1 if style == "lp" else c1 // scale ** 2

        self.offset = nn.Conv2d(in_ch, offset_channels, 1)
        _normal_init(self.offset)

        self.register_buffer("init_pos", self._init_pos())

    def _init_pos(self):
        h = torch.arange(
            (-self.scale + 1) / 2,
            (self.scale - 1) / 2 + 1
        ) / self.scale

        # indexing='ij' 兼容新版 PyTorch
        grid = torch.meshgrid(h, h, indexing="ij")
        return (
            torch.stack(grid)
            .transpose(1, 2)
            .repeat(1, self.groups, 1)
            .reshape(1, -1, 1, 1)
        )

    def _sample(self, x, offset):
        B, _, H, W = offset.shape

        offset = offset.view(B, 2, -1, H, W)

        coords_h = torch.arange(H, dtype=x.dtype, device=x.device) + 0.5
        coords_w = torch.arange(W, dtype=x.dtype, device=x.device) + 0.5

        coords = torch.stack(
            torch.meshgrid(coords_w, coords_h, indexing="ij")
        ).transpose(1, 2)

        coords = coords.unsqueeze(1).unsqueeze(0)

        normalizer = torch.tensor(
            [W, H], dtype=x.dtype, device=x.device
        ).view(1, 2, 1, 1, 1)

        coords = 2 * (coords + offset) / normalizer - 1

        coords = F.pixel_shuffle(
            coords.view(B, -1, H, W),
            self.scale
        )

        coords = (
            coords.view(B, 2, -1, self.scale * H, self.scale * W)
            .permute(0, 2, 3, 4, 1)
            .contiguous()
            .flatten(0, 1)
        )

        x = x.reshape(B * self.groups, -1, H, W)

        out = F.grid_sample(
            x,
            coords,
            mode="bilinear",
            align_corners=False,
            padding_mode="border",
        )

        return out.view(B, -1, self.scale * H, self.scale * W)

    def forward(self, x):
        if self.style == "pl":
            x_ = F.pixel_shuffle(x, self.scale)
            offset = F.pixel_unshuffle(
                self.offset(x_), self.scale
            ) * 0.25 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos

        return self._sample(x, offset)


class CARAFE(nn.Module):
    """
    CARAFE upsample module.

    Input : [B, C, H, W]
    Output: [B, C, 2H, 2W]
    """

    def __init__(self, c1, k_enc=3, k_up=5, c_mid=64, scale=2):
        super().__init__()
        self.scale = scale
        self.k_up = k_up

        c_mid = min(c_mid, c1)

        self.comp = nn.Sequential(
            nn.Conv2d(c1, c_mid, 1, bias=False),
            nn.BatchNorm2d(c_mid),
            nn.SiLU(inplace=True),
        )

        self.enc = nn.Conv2d(
            c_mid,
            (scale * k_up) ** 2,
            k_enc,
            padding=k_enc // 2,
            bias=False,
        )

        self.pix_shf = nn.PixelShuffle(scale)
        self.upsmp = nn.Upsample(scale_factor=scale, mode="nearest")

        self.unfold = nn.Unfold(
            kernel_size=k_up,
            dilation=scale,
            padding=k_up // 2 * scale,
        )

    def forward(self, x):
        b, c, h, w = x.shape
        h_, w_ = h * self.scale, w * self.scale

        W = self.comp(x)
        W = self.enc(W)
        W = self.pix_shf(W)
        W = torch.softmax(W, dim=1)

        x = self.upsmp(x)
        x = self.unfold(x)
        x = x.reshape(b, c, -1, h_, w_)

        out = (W.unsqueeze(1) * x).sum(dim=2)
        return out


class AdaptiveHybridUpsample(nn.Module):
    """
    Improved Lite AdaptiveHybridUpsample for Ultralytics YOLO.

    改进点:
    1. 先用 1x1 Conv 做通道混合，避免硬切通道导致信息隔离
    2. DySample + CARAFE 仍然各走半通道，保持速度
    3. 加 nearest residual，保证不会比原始 Upsample 退化太多
    4. 加轻量 DWConv refine，增强局部细节
    5. YAML 不需要改:
       - [-1, 1, AdaptiveHybridUpsample, []]

    Input : [B, C, H, W]
    Output: [B, C, 2H, 2W]
    """

    def __init__(self, c1, scale=2, groups=4, carafe_mid=32):
        super().__init__()
        assert scale == 2, "AdaptiveHybridUpsample is designed for 2x upsampling."

        self.scale = scale

        # 先混合通道，避免直接 split 原始通道
        self.pre = nn.Sequential(
            nn.Conv2d(c1, c1, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        c_dy = c1 // 2
        c_ca = c1 - c_dy

        self.c_dy = c_dy
        self.c_ca = c_ca

        # 几何对齐分支
        self.dy = DySample(
            c_dy,
            scale=scale,
            groups=groups,
        )

        # 内容感知重组分支
        # 如果你想要更强效果，可以把 k_up=3 改成 k_up=5
        self.ca = CARAFE(
            c_ca,
            k_enc=3,
            k_up=3,
            c_mid=min(carafe_mid, c_ca),
            scale=scale,
        )

        # 双分支融合
        self.fuse = nn.Sequential(
            nn.Conv2d(c1, c1, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        # 最近邻残差兜底，提升稳定性
        self.shortcut = nn.Upsample(scale_factor=scale, mode="nearest")

        # 轻量局部细化
        self.refine = nn.Sequential(
            nn.Conv2d(c1, c1, 3, 1, 1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c1, c1, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        # 残差融合权重，初始偏向稳定的 nearest
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        shortcut = self.shortcut(x)

        x_mix = self.pre(x)

        x_dy, x_ca = torch.split(
            x_mix,
            [self.c_dy, self.c_ca],
            dim=1,
        )

        y_dy = self.dy(x_dy)
        y_ca = self.ca(x_ca)

        y = torch.cat([y_dy, y_ca], dim=1)
        y = self.fuse(y)

        # alpha 限制在 0~1，防止训练初期不稳定
        a = torch.clamp(self.alpha, 0.0, 1.0)

        # 动态分支 + baseline 上采样残差
        y = a * y + (1.0 - a) * shortcut

        y = self.refine(y)

        return y