import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Coordinate Attention (CVPR'21)
# -------------------------
class CoordAtt(nn.Module):
    def __init__(self, inp, reduction=32):
        super().__init__()
        mip = max(8, inp // reduction)

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # (B,C,H,1)
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # (B,C,1,W)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.SiLU(inplace=True)

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        b, c, h, w = x.size()
        x_h = self.pool_h(x)                  # (B,C,H,1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # (B,C,W,1)

        y = torch.cat([x_h, x_w], dim=2)      # (B,C,H+W,1)
        y = self.act(self.bn1(self.conv1(y)))

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)         # (B,C,1,W)

        a_h = torch.sigmoid(self.conv_h(x_h)) # (B,C,H,1)
        a_w = torch.sigmoid(self.conv_w(x_w)) # (B,C,1,W)
        return x * a_h * a_w


# -------------------------
# LUMA + Coordinate Attention (recommended strategy: CA -> Gate)
# -------------------------
class LUMACA(nn.Module):
    def __init__(self, c1, c2, scale=2, ca_reduction=32, ca_on="low"):
        """
        ca_on:
          - "low":  在低分辨率特征上做 CA（更推荐，小目标更敏感，计算也省）
          - "up":   在上采样后做 CA（更贵，一般不如 low 稳）
        """
        super().__init__()
        self.scale = scale
        self.ca_on = ca_on

        self.compress = nn.Conv2d(c1, c2, 1, bias=False)

        self.ca = CoordAtt(c2, reduction=ca_reduction)

        self.gate_conv = nn.Sequential(
            nn.Conv2d(c2, c2, 3, padding=1, groups=c2, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
            nn.Conv2d(c2, c2, 1, bias=False),
        )

        self.gamma = nn.Parameter(torch.zeros(1, c2, 1, 1))
        self.refine = nn.Conv2d(c2, c2, 3, padding=1, groups=c2, bias=False)

    def forward(self, x):
        x_low = self.compress(x)

        if self.ca_on == "low":
            x_cond = self.ca(x_low)               # 坐标增强后的条件特征
            mask_low = torch.tanh(self.gate_conv(x_cond))
            out_low = x_low * (1 + self.gamma * mask_low)
            out = F.interpolate(out_low, scale_factor=self.scale, mode='bilinear', align_corners=False)
            return self.refine(out)

        else:  # "up"
            mask_low = torch.tanh(self.gate_conv(x_low))
            out_low = x_low * (1 + self.gamma * mask_low)
            out = F.interpolate(out_low, scale_factor=self.scale, mode='bilinear', align_corners=False)
            out = self.ca(out)
            return self.refine(out)
