import torch
import torch.nn as nn
import torch.nn.functional as F

class CoordAtt(nn.Module):
    """
    Coordinate Attention 模块
    替代原有的 FGlo (Global Context)。
    作用：捕捉长距离依赖的同时，保留精确的位置信息（对 RSOD 这种密集小目标至关重要）。
    """

    def __init__(self, inp, reduction=32):
        super(CoordAtt, self).__init__()
        # 这里的 reduction 可以适当调小（如 16），以便在小通道数时保留更多信息
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # (H, W) -> (H, 1)
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # (H, W) -> (1, W)

        mip = max(8, inp // reduction)

        # 共享的 1x1 卷积，用于降低维度
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()  # Hardswish 比 ReLU 在注意力中效果通常更好，也可以改回 ReLU

        # 分别恢复维度
        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        # 1. 沿两个方向分别池化
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # 变为 (N, C, W, 1) 以便拼接

        # 2. 拼接并在通道维度降维融合
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # 3. 切分回两个方向
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        # 4. 生成注意力权重
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        # 5. 加权 (广播机制)
        out = identity * a_h * a_w
        return out


class LSCoordinateConv(nn.Module):
    def __init__(self, nIn, nOut, dilation_rate=2, reduction=16):
        super().__init__()

        # 1. 下采样 + 通道扩张 (保持原逻辑，Stride=2)
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(nIn, nOut, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nOut, eps=1e-3),
            nn.PReLU(nOut)
        )
        # 2. 局部特征提取
        self.F_loc = nn.Conv2d(nOut, nOut, kernel_size=3, padding=1,
                               groups=nOut, bias=False)
        # 3. 周围上下文提取
        self.F_sur = nn.Conv2d(nOut, nOut, kernel_size=3, padding=dilation_rate,
                               dilation=dilation_rate, groups=nOut, bias=False)
        # 4. 融合后的归一化与激活
        self.bn = nn.BatchNorm2d(2 * nOut, eps=1e-3)
        self.act = nn.PReLU(2 * nOut)
        # 5. 通道降维
        self.reduce = nn.Conv2d(2 * nOut, nOut, kernel_size=1, bias=False)
        # 6. 全局上下文优化 -> 替换为 Coordinate Attention
        # 注意：这里传入的是最终输出的通道数 nOut
        self.attn = CoordAtt(nOut, reduction=reduction)

    def forward(self, input):
        # 下采样
        output = self.conv1x1(input)
        # 分支提取
        loc = self.F_loc(output)
        sur = self.F_sur(output)
        # 拼接与融合
        joi_feat = torch.cat([loc, sur], 1)
        joi_feat = self.reduce(self.act(self.bn(joi_feat)))
        # 坐标注意力加权 (替代原来的 self.F_glo)
        output = self.attn(joi_feat)
        return output