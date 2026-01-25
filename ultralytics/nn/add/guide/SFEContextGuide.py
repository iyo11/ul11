import torch

import torch.nn as nn

class SFEContextGuide(nn.Module):
    def __init__(self, c1, c2, k=3, d=2, r=8):

        super().__init__()

        assert r >= 1

        hidden = max(1, c2 // r)



        # 1) 输入投影

        self.pw_in = nn.Sequential(

            nn.Conv2d(c1, c2, 1, 1, 0, bias=False),

            nn.BatchNorm2d(c2),

            nn.SiLU(inplace=True)

        )



        # 2) Local Branch: 3x3 DW

        self.dw_local = nn.Conv2d(c2, c2, k, 1, k // 2, groups=c2, bias=False)

        self.bn_local = nn.BatchNorm2d(c2)



        # 3) Context Branch: 3x3 DW Dilated

        self.dw_context = nn.Conv2d(c2, c2, k, 1, d, dilation=d, groups=c2, bias=False)

        self.bn_context = nn.BatchNorm2d(c2)



        # 4) Gate Generator (Scheme C)

        #    先“提炼”再“平滑”，最后 sigmoid 得到 (0,1) attention

        self.gate_gen = nn.Sequential(

            nn.Conv2d(c2, hidden, 1, 1, 0, bias=True),             # squeeze

            nn.SiLU(inplace=True),

            nn.Conv2d(hidden, c2, 1, 1, 0, bias=True),             # excite

            nn.Conv2d(c2, c2, 3, 1, 1, groups=c2, bias=False),     # spatial smooth (DW)

        )



        # 5) 输出融合

        self.pw_out = nn.Conv2d(c2, c2, 1, 1, 0, bias=False)

        self.bn_out = nn.BatchNorm2d(c2)

        self.act = nn.SiLU(inplace=True)



    def forward(self, x):

        x_in = self.pw_in(x)



        local_feat = self.bn_local(self.dw_local(x_in))

        context_feat = self.bn_context(self.dw_context(x_in))

        # Scheme C: context -> attention

        gate = torch.sigmoid(self.gate_gen(context_feat))   # (B,C,H,W) in (0,1)

        guided_feat = local_feat * gate

        out = guided_feat + context_feat


        return self.act(self.bn_out(self.pw_out(out)))