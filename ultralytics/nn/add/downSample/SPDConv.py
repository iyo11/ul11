import torch
import torch.nn as nn
from ultralytics.nn.modules import Conv


class SPDConv(nn.Module):
    """
    Space-to-Depth Convolution (SPD-Conv)
    优化点：
    1. 使用 PyTorch 原生 C++ 实现的 PixelUnshuffle 替代手动切片+拼接，内存效率更高。
    2. 使用 YOLO 自带的 Conv 模块，支持推理阶段的 Conv+BN 融合，速度更快。
    """

    def __init__(self, c1, c2, dimension=1):
        super().__init__()
        # 1. Space-to-Depth (下采样因子=2)
        # 将 (B, C, H, W) -> (B, C*4, H/2, W/2)
        self.spd = nn.PixelUnshuffle(downscale_factor=2)

        # 2. 卷积处理
        # 输入通道变为原来的4倍
        # 使用 Ultralytics 的 Conv 模块而不是原生 nn.Conv2d
        self.conv = Conv(c1 * 4, c2, k=3, s=1)

    def forward(self, x):
        # 流程：无损下采样 -> 卷积特征提取
        return self.conv(self.spd(x))