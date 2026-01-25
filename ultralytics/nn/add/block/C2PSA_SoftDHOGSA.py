import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import Conv, PSABlock

__all__ = ['C2PSA_DHOGSA_Soft']


class SoftDHOG(nn.Module):
    """
    Soft Dynamic HOG Attention.
    Instead of sorting patches, this module computes a spatial attention map
    based on learnable gradient features.
    """

    def __init__(self, c1, output_c, n_bins=9):
        super().__init__()
        # Sobel kernels (fixed)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

        # HOG Feature Processing
        # 输入是 Gx, Gy, Magnitude (3通道) -> 映射到 output_c
        self.hog_proj = nn.Sequential(
            nn.Conv2d(3, c1 // 2, kernel_size=1),
            nn.BatchNorm2d(c1 // 2),
            nn.ReLU(),
            nn.Conv2d(c1 // 2, output_c, kernel_size=1),
            nn.Sigmoid()  # 生成 0~1 的注意力权重
        )

    def forward(self, x):
        # x: [B, C, H, W]
        # 1. 降维计算梯度 (为了速度，在通道维度取平均，或者取最大值)
        # 使用 mean 能够兼顾所有通道的边缘信息
        x_gray = x.mean(dim=1, keepdim=True)  # [B, 1, H, W]

        # 2. 计算梯度 (Vectorized Sobel)
        gx = F.conv2d(x_gray, self.sobel_x, padding=1)
        gy = F.conv2d(x_gray, self.sobel_y, padding=1)

        # 3. 计算幅值 (Magnitude)
        mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)

        # 4. 拼接几何特征 [B, 3, H, W] (Gx, Gy 包含方向信息，Mag 包含强度信息)
        # 注意：这里直接使用 Gx, Gy 代替计算复杂的 atan2 和 binning
        # 卷积层会自动学习如何组合 Gx 和 Gy 来关注特定的方向（相当于学习到了 Bin）
        hog_feats = torch.cat([gx, gy, mag], dim=1)

        # 5. 生成空间注意力图
        attn_map = self.hog_proj(hog_feats)

        return attn_map


class PSABlock_DHOGSA_Soft(nn.Module):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        super().__init__()
        self.c = c
        # 原始特征的通道注意力 (保留原版 PSA 的一半能力)
        self.attn_norm = nn.LayerNorm(c)

        # HOG 空间注意力分支
        self.hog_attn = SoftDHOG(c, c)

        # FFN 部分
        self.ffn = nn.Sequential(
            Conv(c, c * 2, 1),
            Conv(c * 2, c, 1, act=False)
        )
        self.add = shortcut

    def forward(self, x):
        # x: [B, C, H, W]
        identity = x

        # --- 1. HOG 空间增强 (Spatial Awareness) ---
        # 计算 HOG 注意力掩码
        hog_mask = self.hog_attn(x)
        # 软加权：在原特征上施加几何纹理增强
        # x = x * hog_mask  <-- 这种是乘法门控
        # x = x + x * hog_mask <-- 这种是残差增强 (推荐，更稳定)
        x_spatial = x + (x * hog_mask)

        # --- 2. Feed Forward (Channel Mixing) ---
        x = self.ffn(x_spatial)

        return x + identity if self.add else x


class C2PSASoftDHOGSA(nn.Module):
    """
    C2PSA with Soft Dynamic HOG Spatial Attention.
    Replaces standard PSA with HOG-guided soft attention.
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(
            PSABlock_DHOGSA_Soft(self.c, attn_ratio=0.5, num_heads=self.c // 64)
            for _ in range(n)
        ))

    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))