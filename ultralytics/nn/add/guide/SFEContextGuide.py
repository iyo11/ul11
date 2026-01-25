import torch

import torch.nn as nn


class SFEContextGuide(nn.Module):

    def __init__(self, c1, c2, k=3, r=8):
        super().__init__()

        assert r >= 1

        hidden = max(1, c2 // r)

        # 1) 输入投影

        self.pw_in = nn.Sequential(

            nn.Conv2d(c1, c2, 1, 1, 0, bias=False),

            nn.BatchNorm2d(c2),

            nn.SiLU(inplace=True)

        )

        # 2) Local Branch

        self.dw_local = nn.Conv2d(c2, c2, k, 1, k // 2, groups=c2, bias=False)

        self.bn_local = nn.BatchNorm2d(c2)

        # 3) Multi-scale Context Branch (正交分解：1xN + Nx1)

        c_half = c2 // 2

        # 分支1: 1x3 + 3x1 (标准感受野)

        self.dw_context1 = nn.Sequential(

            nn.Conv2d(c_half, c_half, (1, k), 1, (0, k // 2), groups=c_half, bias=False),

            nn.Conv2d(c_half, c_half, (k, 1), 1, (k // 2, 0), groups=c_half, bias=False)

        )

        # 分支2: 1x7 + 7x1 (大感受野，通过增大k或空洞卷积实现，这里用更大的k=7模拟高感受野)

        # 分支2: 使用空洞卷积实现大感受野 (等效 7x7)
        # 感受野计算公式: R = (k-1) * d + 1
        # 使用 k=3, dilation=3 -> (3-1)*3 + 1 = 7
        k_small = 3
        d_rate = 3
        self.dw_context3 = nn.Sequential(
            # 1x3 卷积 + dilation 3
            nn.Conv2d(c_half, c_half, (1, k_small), 1, (0, d_rate),
                      dilation=(1, d_rate), groups=c_half, bias=False),
            # 3x1 卷积 + dilation 3
            nn.Conv2d(c_half, c_half, (k_small, 1), 1, (d_rate, 0),
                      dilation=(d_rate, 1), groups=c_half, bias=False)
        )

        self.bn_context = nn.BatchNorm2d(c2)

        # 4) Gate Generator

        self.gate_gen = nn.Sequential(

            nn.Conv2d(c2, hidden, 1, 1, 0, bias=True),

            nn.SiLU(inplace=True),

            nn.Conv2d(hidden, c2, 1, 1, 0, bias=True),

            nn.Conv2d(c2, c2, 3, 1, 1, groups=c2, bias=False),

        )

        # 5) 输出融合

        self.pw_out = nn.Conv2d(c2, c2, 1, 1, 0, bias=False)

        self.bn_out = nn.BatchNorm2d(c2)

        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x_in = self.pw_in(x)

        # Local Branch

        local_feat = self.bn_local(self.dw_local(x_in))

        # Context Branch (Split -> Orthogonal Conv -> Concat)

        c_half = x_in.shape[1] // 2

        x_c1, x_c3 = torch.split(x_in, [c_half, x_in.shape[1] - c_half], dim=1)

        # 经过水平和垂直正交分解后的特征提取

        feat_c1 = self.dw_context1(x_c1)

        feat_c3 = self.dw_context3(x_c3)

        context_feat = torch.cat([feat_c1, feat_c3], dim=1)

        context_feat = self.bn_context(context_feat)

        # Gate 生成与增强

        gate = torch.sigmoid(self.gate_gen(context_feat))

        out = local_feat + gate * context_feat

        return self.act(self.bn_out(self.pw_out(out)) + x_in)