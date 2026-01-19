import torch
import torch.nn as nn

__all__ = ['SalientContextConv']


class SimAM(nn.Module):
    """
    SimAM: A Simple, Parameter-Free Attention Module
    通过计算能量函数来寻找显著神经元，无需全局池化，
    能够完美保留极小目标的点状特征。
    """

    def __init__(self, e_lambda=1e-4):
        super(SimAM, self).__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        # x: (B, C, H, W)
        b, c, h, w = x.size()
        n = w * h - 1

        # 1. 计算均值 (每个像素点与当前通道均值的差的平方)
        # 注意：这里并没有把空间维度压缩成 1x1，而是保留了 (H, W)
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)

        # 2. 计算能量 (Energy Function)
        # 能量越低，代表该点与背景差异越大（即显著点/小目标）
        # 公式中 y 是能量的倒数形式，所以 y 越大越重要
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        # 3. 加权
        return x * self.activaton(y)


class SalientContextConv(nn.Module):
    """
    SalientContextConv
    结构：Context-Guided (多尺度感知) + SimAM (显著性能量增强)
    适用：极小目标检测 (Tiny Object Detection, <5px)
    """

    def __init__(self, nIn, nOut, dilation_rate=2, stride=2):
        super().__init__()

        # 1. Input Project & Downsample
        # 即使是下采样，我们也要尽力保留特征
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(nIn, nOut, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(nOut, eps=1e-3),
            nn.PReLU(nOut)
        )

        # 2. Local Feature (捕捉物体本身形状)
        # 深度可分离卷积 (groups=nOut)
        self.F_loc = nn.Conv2d(nOut, nOut, kernel_size=3, padding=1,
                               groups=nOut, bias=False)

        # 3. Surrounding Context (捕捉物体背景对比)
        # 空洞卷积扩大感受野，对于判断"这是不是一个孤立的小点"很有用
        self.F_sur = nn.Conv2d(nOut, nOut, kernel_size=3, padding=dilation_rate,
                               dilation=dilation_rate, groups=nOut, bias=False)

        # 4. Feature Fusion
        self.bn = nn.BatchNorm2d(2 * nOut, eps=1e-3)
        self.act = nn.PReLU(2 * nOut)
        self.reduce = nn.Conv2d(2 * nOut, nOut, kernel_size=1, bias=False)

        # 5. Salient Refinement (SimAM)
        # 替换了原有的 FGlo，直接在空间上增强显著点
        self.salient_attn = SimAM(e_lambda=1e-4)

    def forward(self, input):
        # 基础特征
        output = self.conv1x1(input)

        # 上下文分支
        loc = self.F_loc(output)
        sur = self.F_sur(output)

        # 融合上下文
        joi_feat = torch.cat([loc, sur], 1)
        joi_feat = self.reduce(self.act(self.bn(joi_feat)))

        # 显著性增强 (关键步骤：高亮小目标)
        output = self.salient_attn(joi_feat)

        return output


# --- 验证输入输出 ---
if __name__ == "__main__":
    # 模拟一个包含微小特征的输入 (Batch=1, C=64, H=128, W=128)
    x = torch.randn(1, 64, 128, 128)

    # 假设 stride=1 保持分辨率，或者 stride=2 下采样
    block = SalientContextConv(64, 128, stride=2)

    out = block(x)
    print(f"SalientContextConv Output: {out.shape}")