import torch
import torch.nn as nn

__all__ = ['SFEContextGuide']

import torch
import torch.nn as nn


class SFEContextGuide(nn.Module):
    def __init__(self, c1, c2, k=3, r=8):
        super().__init__()
        assert r >= 1
        hidden = max(1, c2 // r)

        self.pw_in = nn.Sequential(
            nn.Conv2d(c1, c2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True)
        )

        # Local Branch: 保持细节
        self.dw_local = nn.Conv2d(c2, c2, k, 1, k // 2, groups=c2, bias=False)
        self.bn_local = nn.BatchNorm2d(c2)

        # Context Branch: 微调空洞率
        # 之前的 d=1 和 d=3 跨度太大，可能导致中间尺度的罐子特征断裂
        # 改回更紧凑的组合：一半 d=1，一半 d=2 (接近 V1 的感受野，但更丰富)
        self.dw_context1 = nn.Conv2d(c2 // 2, c2 // 2, k, 1, padding=1, dilation=1, groups=c2 // 2, bias=False)
        self.dw_context2 = nn.Conv2d(c2 // 2, c2 // 2, k, 1, padding=2, dilation=2, groups=c2 // 2, bias=False)  # 改为2
        self.bn_context = nn.BatchNorm2d(c2)

        # Gate Generator
        self.gate_gen = nn.Sequential(
            nn.Conv2d(c2, hidden, 1, 1, 0, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, c2, 1, 1, 0, bias=True),
            # 这里去掉最后的 3x3 DW，减少平滑，让 Gate 更加锐利，找回 V1 的感觉
        )

        self.pw_out = nn.Conv2d(c2, c2, 1, 1, 0, bias=False)
        self.bn_out = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x_in = self.pw_in(x)

        local_feat = self.bn_local(self.dw_local(x_in))

        # Context 计算
        c_half = x_in.shape[1] // 2
        x_c1, x_c2 = torch.split(x_in, [c_half, x_in.shape[1] - c_half], dim=1)
        context_feat = torch.cat([self.dw_context1(x_c1), self.dw_context2(x_c2)], dim=1)
        context_feat = self.bn_context(context_feat)

        # 生成 Gate
        # 注意：这里我们让 Gate 只由 Context 生成（回归 V1 的逻辑，强调上下文的主导性）
        gate = torch.sigmoid(self.gate_gen(context_feat))

        # 核心修改：特征调制 (Feature Modulation)
        # 1. (1 + gate): 如果 gate 高，local 特征被放大（增强信号）。
        # 2. 如果 gate 低，local 特征保持原样（不被抑制，防止 V1 的误删问题）。
        # 3. + context_feat: 补充上下文信息。
        out = local_feat * (1 + gate) + context_feat

        return self.act(self.bn_out(self.pw_out(out)) + x_in)