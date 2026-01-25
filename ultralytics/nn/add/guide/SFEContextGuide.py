import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['SFEContextGuide']


class SFEContextGuide(nn.Module):
    def __init__(self, c1, c2, k=3, r=8):
        super().__init__()
        assert r >= 1
        hidden = max(1, c2 // r)

        # 1) 输入投影：提升通道数并进行初步特征提取
        self.pw_in = nn.Sequential(
            nn.Conv2d(c1, c2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True)
        )

        # 2) Local Branch: 保持局部特征捕捉
        self.dw_local = nn.Sequential(
            nn.Conv2d(c2, c2, k, 1, k // 2, groups=c2, bias=False),
            nn.BatchNorm2d(c2)
        )

        # 3) Context Branch (改进点：引入条形卷积增强对 Bridge 等长目标的感知)
        # 将通道分为两半，一半提取水平特征，一半提取垂直特征
        self.dw_h = nn.Sequential(
            nn.Conv2d(c2 // 2, c2 // 2, (1, 7), 1, (0, 3), groups=c2 // 2, bias=False),
            nn.BatchNorm2d(c2 // 2)
        )
        self.dw_v = nn.Sequential(
            nn.Conv2d(c2 // 2, c2 // 2, (7, 1), 1, (3, 0), groups=c2 // 2, bias=False),
            nn.BatchNorm2d(c2 // 2)
        )

        # 4) Global Context (改进点：引入全局平均池化，解决 Vehicle 这种小目标的背景误判)
        self.glb_pool = nn.AdaptiveAvgPool2d(1)

        # 5) Gate 生成器 (改进点：融合局部上下文与全局背景)
        self.gate_gen = nn.Sequential(
            # 输入为 context_feat (c2) + glb_feat (c2) = 2*c2
            nn.Conv2d(c2 * 2, hidden, 1, 1, 0, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, c2, 1, 1, 0, bias=True),
            nn.Sigmoid()
        )

        # 6) 输出投影
        self.pw_out = nn.Sequential(
            nn.Conv2d(c2, c2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c2)
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        # 基础特征提取
        x_in = self.pw_in(x)

        # --- Local Branch ---
        local_feat = self.dw_local(x_in)

        # --- Context Branch (Strip Convolution) ---
        c_half = x_in.shape[1] // 2
        x_c1, x_c2 = torch.split(x_in, [c_half, x_in.shape[1] - c_half], dim=1)
        context_feat = torch.cat([self.dw_h(x_c1), self.dw_v(x_c2)], dim=1)

        # --- Global Guidance ---
        # 提取全局统计信息并扩展到与特征图相同大小
        glb_feat = self.glb_pool(x_in).expand_as(x_in)

        # --- Gate-based Fusion ---
        # 结合上下文特征和全局特征生成注意力权重
        gate_input = torch.cat([context_feat, glb_feat], dim=1)
        gate = self.gate_gen(gate_input)

        # 扬长避短融合逻辑：
        # gate 决定了对 local 特征的关注度，同时保留 context 特征作为结构补充
        out = local_feat * gate + context_feat * (1 - gate)

        # 残差连接并输出
        return self.act(self.pw_out(out) + x_in)