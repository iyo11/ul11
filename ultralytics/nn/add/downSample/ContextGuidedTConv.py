import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['ContextGuidedTConv']

class CoordAtt(nn.Module):
    """
    坐标注意力 (Coordinate Attention)
    替代原有的 FGlo (SE-Block)。
    相比全局平均池化，它能保留空间位置信息，对小目标定位极有帮助。
    """

    def __init__(self, inp, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        # 分别沿 H 和 W 方向池化，保留位置感知
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        return identity * a_h * a_w


class ContextGuidedTConv(nn.Module):
    """
    针对极小目标优化的 CGC 模块
    (H,W,C) -> (H/2, W/2, 2C)
    """

    def __init__(self, nIn, nOut, dilation_rate=2, reduction=16):
        super().__init__()

        # --- 改动1: 混合下采样 (Mixed Downsampling) ---
        # 很多小目标是高频信号，Convstride=2容易平滑丢失，MaxPool能保留强响应
        self.down_conv = nn.Sequential(
            nn.Conv2d(nIn, nOut, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nOut),
            nn.PReLU(nOut)
        )
        self.down_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 降维融合层 (处理 MaxPool 出来的通道数 + Conv 出来的通道数)
        # MaxPool不改变通道数(nIn)，Conv变成nOut
        self.fusion_mix = nn.Sequential(
            nn.Conv2d(nIn + nOut, nOut, kernel_size=1, bias=False),
            nn.BatchNorm2d(nOut),
            nn.PReLU(nOut)
        )

        # --- 改动2: 三路特征提取 (增加 1x1 原生分支) ---
        # 1. 局部特征 (3x3 DW)
        self.F_loc = nn.Conv2d(nOut, nOut, kernel_size=3, padding=1,
                               groups=nOut, bias=False)

        # 2. 上下文特征 (Dilated 3x3 DW)
        # 注意：极小目标如果不使用 dilation=1，可能会导致网格效应。
        # 这里保留 dilation，但建议外部调用时 dilation_rate 设小一点（如2）
        self.F_sur = nn.Conv2d(nOut, nOut, kernel_size=3, padding=dilation_rate,
                               dilation=dilation_rate, groups=nOut, bias=False)

        # 3. [新增] 点特征 (1x1) - 没有任何邻域混合，保护极小点特征
        self.F_pnt = nn.Conv2d(nOut, nOut, kernel_size=1, bias=False)

        # 融合层 (输入是 3 * nOut)
        self.bn_joint = nn.BatchNorm2d(3 * nOut, eps=1e-3)
        self.act_joint = nn.PReLU(3 * nOut)
        self.reduce = nn.Conv2d(3 * nOut, nOut, kernel_size=1, bias=False)

        # --- 改动3: 使用 Coordinate Attention 替代 Global Context ---
        self.attention = CoordAtt(nOut, reduction=reduction)

    def forward(self, input):
        # 1. 混合下采样
        d_conv = self.down_conv(input)
        d_pool = self.down_pool(input)

        # 拼接并融合通道
        # MaxPool 出来是 nIn 通道，Conv 出来是 nOut 通道
        output = torch.cat([d_conv, d_pool], dim=1)
        output = self.fusion_mix(output)

        # 2. 分支提取
        loc = self.F_loc(output)  # 3x3
        sur = self.F_sur(output)  # 3x3 dilated
        pnt = self.F_pnt(output)  # 1x1 (新增，保留点特征)

        # 3. 拼接与融合
        joi_feat = torch.cat([loc, sur, pnt], 1)
        joi_feat = self.reduce(self.act_joint(self.bn_joint(joi_feat)))

        # 4. 坐标注意力加权
        output = self.attention(joi_feat)

        return output