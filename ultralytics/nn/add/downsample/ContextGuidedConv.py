import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv

__all__ = ['ContextGuidedConv']


class FGlo(nn.Module):
    """
    全局上下文模块 (Global Context Refinement)
    利用 1x1 卷积代替 Linear，避免 view/reshape 操作，且对 Tensor 连续性更友好。
    """

    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(self.avg_pool(x))


class ContextGuidedConv(nn.Module):
    """
    CGConv 模块：结合局部特征和周围上下文。
    输入通道 nIn -> 输出通道 nOut，包含 2x 下采样。
    """

    def __init__(self, nIn, nOut, dilation_rate=2, reduction=16):
        super().__init__()

        # 1. 初始下采样：使用 Ultralytics 的 Conv 模块 (默认包含 Conv+BN+SiLU)
        # 如果你必须用 PReLU，可以将 act=nn.PReLU(nOut) 传入，但通常 SiLU 性能更好
        self.conv1x1 = Conv(nIn, nOut, k=3, s=2, p=1)

        # 2. 局部特征提取: Depth-wise Convolution (groups=nOut)
        self.F_loc = nn.Conv2d(nOut, nOut, kernel_size=3, padding=1, groups=nOut, bias=False)

        # 3. 周围上下文提取: Depth-wise Dilated Convolution (groups=nOut)
        self.F_sur = nn.Conv2d(nOut, nOut, kernel_size=3, padding=dilation_rate,
                               dilation=dilation_rate, groups=nOut, bias=False)

        # 4. 融合阶段
        # 使用 Conv 模块处理拼接后的特征 (2*nOut -> nOut)
        # 这里包含 BN 和激活
        self.bn_act_reduce = Conv(2 * nOut, nOut, k=1)

        # 5. 全局上下文优化
        self.F_glo = FGlo(nOut, reduction)

    def forward(self, x):
        # 初始特征提取与下采样
        x = self.conv1x1(x)

        # 分支：局部 + 上下文
        loc = self.F_loc(x)
        sur = self.F_sur(x)

        # 拼接并降维 (Concat -> BN -> Act -> 1x1 Conv)
        joi_feat = torch.cat([loc, sur], 1)
        joi_feat = self.bn_act_reduce(joi_feat)

        # 全局加权
        return self.F_glo(joi_feat)