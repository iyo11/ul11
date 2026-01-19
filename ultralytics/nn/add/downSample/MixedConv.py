import torch
import torch.nn as nn

class MixedConv(nn.Module):
    """
    位置建议：替代 Backbone 中所有的 Conv(stride=2)。
    作用：一路 MaxPool 抓取高亮噪点（极小目标），一路 Conv 抓取语义。
    """

    def __init__(self, c1, c2, k=3, s=1):  # 这里的 k, s 是为了接收 YAML 传来的参数，不用动
        super().__init__()
        c_ = c2 // 2  # 输出通道对半分

        # 分支1: MaxPool (专门保住 3px 的高亮像素)
        self.branch1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            # [修复] 必须用 kernel_size=1, stride=1
            nn.Conv2d(c1, c_, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(c_),
            nn.SiLU()
        )

        # 分支2: 卷积下采样 (保留常规特征)
        self.branch2 = nn.Sequential(
            # [修复] 必须用 kernel_size=3
            nn.Conv2d(c1, c_, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c_),
            nn.SiLU()
        )

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x)], dim=1)