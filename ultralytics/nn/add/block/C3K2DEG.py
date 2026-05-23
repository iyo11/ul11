######################################## CVPR2026 DEGConv by AI Little monster start  ########################################
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import Bottleneck, C2f, C3k2, C3k


def image2patches(x):
    """
    将图像分块为 2x2 的patch，兼容任意输入尺寸（内部已处理pad）
    """
    b, c, h, w = x.shape
    # 计算需要pad的尺寸，确保能被2整除
    pad_h = (2 - h % 2) % 2
    pad_w = (2 - w % 2) % 2

    # 仅在需要时pad（避免多余操作）
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')  # replicate pad避免边界失真

    # 分块
    x = einops.rearrange(x, "b c (hg h) (wg w) -> (hg wg b) c h w", hg=2, wg=2)
    return x, (pad_h, pad_w)  # 返回pad信息，用于后续裁剪


def patches2image(x, pad_info):
    """
    将patch还原为图像，并裁剪回原始尺寸
    Args:
        x: 分块后的张量
        pad_info: (pad_h, pad_w) 分块前的pad信息
    """
    # 还原图像
    x = einops.rearrange(x, "(hg wg b) c h w -> b c (hg h) (wg w)", hg=2, wg=2)

    # 裁剪掉pad的部分
    pad_h, pad_w = pad_info
    if pad_h > 0 or pad_w > 0:
        x = x[:, :, :-pad_h, :-pad_w] if (pad_h > 0 and pad_w > 0) else \
            x[:, :, :-pad_h, :] if pad_h > 0 else \
                x[:, :, :, :-pad_w]
    return x


class EdgeConv(nn.Module):
    def __init__(
            self,
            in_channels,
            mid_channels,
            out_channels,
            kernel_size=3,
            bias=True,
    ):
        super().__init__()

        self.in_proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=1,
            bias=bias,
        )
        self.w_conv = nn.Conv2d(
            mid_channels,
            mid_channels,
            kernel_size=(1, kernel_size),
            stride=1,
            padding=(0, kernel_size // 2),
            groups=mid_channels,
        )

        self.h_conv = nn.Conv2d(
            mid_channels,
            mid_channels,
            kernel_size=(kernel_size, 1),
            stride=1,
            padding=(kernel_size // 2, 0),
            groups=mid_channels,
        )

        self.out_proj = nn.Conv2d(
            in_channels=mid_channels * 2,
            out_channels=out_channels,
            kernel_size=1,
            bias=True,
        )

    def forward(self, x):
        x = self.in_proj(x)
        x_w = self.w_conv(x)
        x_h = self.h_conv(x)
        x = torch.cat([x_w, x_h], dim=1)
        x = self.out_proj(x)
        return x


class DEGConv(nn.Module):
    def __init__(self, in_dim, out_dim, nbins=36, cell_size=(8, 8)):
        super().__init__()

        self.nbins = nbins
        self.cell_size = cell_size
        self.cell_area = cell_size[0] * cell_size[1]  # 替换硬编码的64

        self.hog_feat = nn.Sequential(
            nn.Conv2d(nbins, in_dim, kernel_size=1),
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, groups=in_dim, bias=False),
            nn.GroupNorm(in_dim // 8, in_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.weight = nn.Sequential(
            EdgeConv(in_channels=in_dim, mid_channels=in_dim // 2, out_channels=in_dim),
            nn.GroupNorm(in_dim // 8, in_dim),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1, stride=1),
            nn.GroupNorm(in_dim // 8, in_dim),
        )

        self.fuse_block = nn.Sequential(
            EdgeConv(in_channels=in_dim, mid_channels=in_dim // 2, out_channels=in_dim, kernel_size=3),
            nn.GroupNorm(in_dim // 8, in_dim),
        )

        self.sigmoid = nn.Sigmoid()

        self.conv_1x1 = Conv(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        # 记录原始尺寸和dtype/device
        input_dtype = x.dtype
        input_shape = x.shape  # (b, c, h, w)
        residual = x

        # 分块（兼容任意尺寸，返回pad信息）
        x, pad_info = image2patches(x)

        # 计算HOG特征（匹配输入dtype）
        x_hog = self.get_hog_feature(x, input_dtype)
        x_hog = self.hog_feat(x_hog)

        # 权重融合
        x1 = self.sigmoid(self.weight(x + x_hog))
        x2 = self.conv(x)
        x = x1 * x2

        # 还原图像并裁剪回原始尺寸
        x = patches2image(x, pad_info)

        # 残差连接（确保尺寸匹配）
        x = x + residual
        x = self.fuse_block(x)

        return self.conv_1x1(x)

    def get_hog_feature(self, x, input_dtype):
        x_mean = x.mean(dim=1, keepdim=True)
        b, _, h, w = x_mean.shape
        device = x_mean.device

        # 1. 处理特征图尺寸小于cell_size的情况
        cell_h, cell_w = self.cell_size
        cell_h = min(cell_h, h)
        cell_w = min(cell_w, w)
        h_cells = max(1, h // cell_h)
        w_cells = max(1, w // cell_w)

        # 2. 裁剪到整数倍的cell尺寸
        crop_h = h_cells * cell_h
        crop_w = w_cells * cell_w
        dirs_crop = x_mean[:, :, :crop_h, :crop_w].to(dtype=input_dtype)

        # 3. Sobel梯度计算（匹配输入dtype）
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=input_dtype, device=device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=input_dtype, device=device).view(1, 1, 3, 3)

        dx = F.conv2d(dirs_crop, sobel_x, padding=1)
        dy = F.conv2d(dirs_crop, sobel_y, padding=1)

        # 4. 梯度方向计算（添加epsilon避免除以0）
        gradient_dir = torch.atan2(dy, dx + 1e-8)
        gradient_dir = torch.abs(gradient_dir)

        # 5. 重塑为cell维度
        dirs = gradient_dir.reshape(b, h_cells, cell_h, w_cells, cell_w)
        dirs = dirs.permute(0, 1, 3, 2, 4).reshape(b, h_cells, w_cells, -1)

        # 6. 计算bin索引（避免越界）
        bin_width = torch.pi / self.nbins
        bin_indices = (dirs.to(torch.float32) / bin_width).floor().long()
        bin_indices = torch.clamp(bin_indices, 0, self.nbins - 1)

        # 7. 计算每个bin的计数（向量化实现）
        bin_indices_flat = bin_indices.reshape(-1, dirs.shape[-1])
        weight = torch.zeros(bin_indices_flat.shape[0], self.nbins,
                             dtype=input_dtype, device=device)
        weight.scatter_add_(1, bin_indices_flat,
                            torch.ones_like(bin_indices_flat, dtype=input_dtype))

        # 8. 归一化
        weight = weight.reshape(b, h_cells, w_cells, self.nbins) / self.cell_area

        # 9. 生成HOG特征（匹配输入dtype）
        start = torch.pi / (2 * self.nbins)
        hog_bins = torch.linspace(start, torch.pi - start, self.nbins,
                                  dtype=input_dtype, device=device)
        hog_feature = hog_bins[None, None, None, :] * weight

        # 10. 调整维度并上采样
        hog_feature = hog_feature.permute(0, 3, 1, 2)
        hog_feature = F.interpolate(hog_feature, size=(h, w), mode='nearest')

        return hog_feature


class Bottleneck_DEGConv(Bottleneck):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = DEGConv(c1, c1)
        self.cv2 = DEGConv(c2, c2)


class C3k_DEGConv(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_DEGConv(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class C3k2_DEGConv(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(
            C3k_DEGConv(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_DEGConv(self.c, self.c, shortcut) for _
            in range(n))

######################################## CVPR2026 DEGConv by AI Little monster end  ########################################