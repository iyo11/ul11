import torch
import torch.nn as nn
import torch.nn.functional as F

_all__ = ['StepwiseContextGuide']

def _make_norm(c: int, use_gn: bool = True, gn_groups: int = 16):
    if not use_gn:
        return nn.BatchNorm2d(c)
    g = min(gn_groups, c)
    while g > 1 and (c % g != 0):
        g -= 1
    return nn.GroupNorm(g, c)

class StepwiseContextGuide(nn.Module):
    def __init__(self, c_local: int, c_guide: int, r: int = 4,
                 use_gn: bool = True, gn_groups: int = 16):
        super().__init__()

        # 1) align guide -> local channels
        self.align = nn.Conv2d(c_guide, c_local, 1, 1, 0, bias=False)
        self.norm_align = _make_norm(c_local, use_gn, gn_groups)

        # 2) channel gate (SE-like)
        hidden = max(1, c_local // r)
        self.gate_c = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c_local, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, c_local, 1, bias=True),
        )

        # 3) spatial gate (lightweight)
        self.gate_s = nn.Conv2d(c_local, c_local, 3, 1, 1, groups=c_local, bias=False)

        # 4) learnable scales (init 0 => stable start)
        self.alpha = nn.Parameter(torch.zeros(1, c_local, 1, 1))
        self.beta  = nn.Parameter(torch.zeros(1, c_local, 1, 1))

    def forward(self, x):
        # Ultralytics YAML 传入 list 时，x 是 [x_local, x_guide]
        if isinstance(x, (list, tuple)):
            x_local, x_guide = x
        else:
            raise TypeError("SCGMv2 expects [x_local, x_guide].")

        g = self.norm_align(self.align(x_guide))
        g_up = F.interpolate(g, size=x_local.shape[2:], mode="bilinear", align_corners=False)

        gate_c = torch.sigmoid(self.gate_c(g_up))          # (B,C,1,1)
        gate_s = torch.sigmoid(self.gate_s(g_up))          # (B,C,H,W)

        gate = 0.5 * (gate_s + gate_c)                     # (B,C,H,W)
        gate = 1.0 + self.alpha * (gate - 0.5)             # residual gate

        x_gated = x_local * gate
        out = x_gated + self.beta * g_up                   # controllable injection
        return out
