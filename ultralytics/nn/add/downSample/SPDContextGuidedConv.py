import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['SPDContextGuidedConv']

class Hsigmoid(nn.Module):
    def forward(self, x):
        return F.relu6(x + 3, inplace=True) / 6


class CoordinateAttention(nn.Module):
    """
    保留之前的优化：用 CA 替代 GlobalAvgPool，锁住小目标的 X/Y 坐标
    """

    def __init__(self, inp, reduction=32):
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = Hsigmoid()
        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        return identity * a_h * a_w


class TinySafeDownsample(nn.Module):
    """
    【核心修改】：SPD 下采样
    不使用 stride=2 的卷积，而是使用 reshape 操作。
    输入: (B, C, H, W)
    输出: (B, 4C, H/2, W/2) -> 1x1 Conv -> (B, Out, H/2, W/2)
    保证没有任何一个像素被丢弃。
    """

    def __init__(self, nIn, nOut):
        super().__init__()
        # SPD 变换后通道数会变成 4倍 (nIn * 4)
        # 我们用 1x1 卷积把它融合降维到 nOut
        self.merge = nn.Sequential(
            nn.Conv2d(nIn * 4, nOut, kernel_size=1, bias=False),
            nn.BatchNorm2d(nOut),
            nn.PReLU(nOut)
        )

    def forward(self, x):
        # 假设 x 是 (B, C, H, W)
        # 1. 切片提取 (Space-to-Depth)
        # 相当于把 2x2 的方格拆开，分别放到 4 个通道里
        x0 = x[:, :, 0::2, 0::2]  # 左上
        x1 = x[:, :, 1::2, 0::2]  # 右上
        x2 = x[:, :, 0::2, 1::2]  # 左下
        x3 = x[:, :, 1::2, 1::2]  # 右下

        # 2. 拼接: (B, 4C, H/2, W/2)
        x_cat = torch.cat([x0, x1, x2, x3], dim=1)

        # 3. 融合特征
        return self.merge(x_cat)


class SPDContextGuidedConv(nn.Module):
    """
    针对极小目标优化的下采样模块
    输出尺寸依然是 H/2, W/2，可以直接替换原代码
    """

    def __init__(self, nIn, nOut, dilation_rate=2, reduction=16):
        super().__init__()

        # 1. 【改动】：使用 SPD 无损下采样替代原来的 Conv 3x3 stride=2
        # 这样小车的像素被移动了，而不是被卷积核“跳过”了
        self.downsample_block = TinySafeDownsample(nIn, nOut)

        # 2. 局部特征 (保持不变，提取周边信息)
        self.F_loc = nn.Conv2d(nOut, nOut, kernel_size=3, padding=1,
                               groups=nOut, bias=False)

        # 3. 【改动】：去掉了 dilation=2 (空洞卷积)
        # 极小目标很忌讳空洞，因为容易漏掉中心特征。
        # 改用实心卷积，groups=nOut 保持轻量化
        self.F_sur = nn.Conv2d(nOut, nOut, kernel_size=3, padding=1,
                               groups=nOut, bias=False)

        # 4. 融合后的归一化与激活
        self.bn = nn.BatchNorm2d(2 * nOut, eps=1e-3)
        self.act = nn.PReLU(2 * nOut)

        # 5. 通道降维
        self.reduce = nn.Conv2d(2 * nOut, nOut, kernel_size=1, bias=False)

        # 6. 【改动】：使用 Coordinate Attention
        # 能够精确定位小物体在哪个坐标
        self.F_glo = CoordinateAttention(nOut, reduction)

    def forward(self, input):
        # 1. 无损下采样 (H,W -> H/2,W/2)
        output = self.downsample_block(input)

        # 2. 分支提取
        loc = self.F_loc(output)
        sur = self.F_sur(output)  # 这里不再是空洞卷积，避免网格效应

        # 3. 拼接与融合
        joi_feat = torch.cat([loc, sur], 1)
        joi_feat = self.reduce(self.act(self.bn(joi_feat)))

        # 4. 坐标加权
        output = self.F_glo(joi_feat)

        return output