import torch
import torch.nn as nn

# ---------------------- 模块 1: 高频增强 (针对极小纹理) ----------------------
class HighFreqEnhance(nn.Module):
    def __init__(self, c1, c2):  # c1输入通道, c2输出通道
        super().__init__()
        self.c_mid = c1 // 2

        # [修复] 必须使用 kernel_size=1，不能写 k=1
        self.reduce = nn.Conv2d(c1, self.c_mid, kernel_size=1, bias=False)

        # 这里的 AvgPool 相当于“低通滤波器”，提取背景
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        # 3. 激励层，生成权重
        self.excite = nn.Sequential(
            # [修复] kernel_size=1
            nn.Conv2d(self.c_mid, c1, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.Sigmoid()
        )

        # 4. 如果输入输出通道不一致，需要对齐
        # [修复] kernel_size=1
        self.project = nn.Identity() if c1 == c2 else nn.Conv2d(c1, c2, kernel_size=1)

    def forward(self, x):
        feat = self.reduce(x)

        # 高频信号(小目标) = 原图 - 低频信号(背景)
        smooth = self.avg_pool(feat)
        high_freq = feat - smooth

        # 生成增强权重
        weight = self.excite(high_freq)

        # 原始特征 + (原始特征 * 增强权重)
        out = x + (x * weight)
        return self.project(out)