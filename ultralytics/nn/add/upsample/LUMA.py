import torch
import torch.nn as nn
import torch.nn.functional as F

class StableDSU(nn.Module):
    def __init__(self, c1, c2, scale=2):
        super().__init__()
        self.scale = scale

        self.compress = nn.Conv2d(c1, c2, 1, bias=False)

        # gate: 生成“可增强可抑制”的mask（tanh）
        self.gate_conv = nn.Sequential(
            nn.Conv2d(c2, c2, 3, padding=1, groups=c2, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(),
            nn.Conv2d(c2, c2, 1, bias=False),
        )

        # 关键：稳定起步，避免一上来就把纹理噪声抬起来
        self.gamma = nn.Parameter(torch.zeros(1, c2, 1, 1))

        self.refine = nn.Conv2d(c2, c2, 3, padding=1, groups=c2, bias=False)

    def forward(self, x):
        x_low = self.compress(x)
        mask_low = torch.tanh(self.gate_conv(x_low))          # [-1, 1]
        out_low = x_low * (1 + self.gamma * mask_low)         # 可增强可抑制，且初始≈x_low

        out = F.interpolate(out_low, scale_factor=self.scale, mode='bilinear', align_corners=False)
        return self.refine(out)
