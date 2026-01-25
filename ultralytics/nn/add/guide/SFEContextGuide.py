import torch
import torch.nn as nn

__all__ = ['SFEContextGuide']

class SFEContextGuide(nn.Module):
    def __init__(self, c1, c2, k=3, r=8):
        super().__init__()
        assert r >= 1
        hidden = max(1, c2 // r)

        # 1) 输入投影：将通道数从 c1 调整到 c2
        self.pw_in = nn.Sequential(
            nn.Conv2d(c1, c2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True)
        )

        # 2) Local Branch: 标准 3x3 深度可分离卷积 (d=1)
        self.dw_local = nn.Conv2d(c2, c2, k, 1, k // 2, groups=c2, bias=False)
        self.bn_local = nn.BatchNorm2d(c2)

        # 3) Multi-scale Context Branch (方案二改进)
        # 将通道分为两组：一组 d=1 保持近距离拓扑，一组 d=3 捕捉大尺度上下文
        self.dw_context1 = nn.Conv2d(c2 // 2, c2 // 2, k, 1, padding=1, dilation=1, groups=c2 // 2, bias=False)
        self.dw_context3 = nn.Conv2d(c2 // 2, c2 // 2, k, 1, padding=3, dilation=3, groups=c2 // 2, bias=False)
        self.bn_context = nn.BatchNorm2d(c2)

        # 4) Gate Generator
        self.gate_gen = nn.Sequential(
            nn.Conv2d(c2, hidden, 1, 1, 0, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, c2, 1, 1, 0, bias=True),
            nn.Conv2d(c2, c2, 3, 1, 1, groups=c2, bias=False),  # 空间平滑
        )

        # 5) 输出融合
        self.pw_out = nn.Conv2d(c2, c2, 1, 1, 0, bias=False)
        self.bn_out = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x_in = self.pw_in(x)

        # Local Branch
        local_feat = self.bn_local(self.dw_local(x_in))

        # Multi-scale Context Branch (Split-Conv-Concat)
        c_half = x_in.shape[1] // 2
        x_c1, x_c3 = torch.split(x_in, [c_half, x_in.shape[1] - c_half], dim=1)
        context_feat = torch.cat([self.dw_context1(x_c1), self.dw_context3(x_c3)], dim=1)
        context_feat = self.bn_context(context_feat)

        # Gate 生成 (基于 Context 特征生成权重)
        gate = torch.sigmoid(self.gate_gen(context_feat))

        # 融合层面改进：由“乘法门控”改为“加法增强” (方案改进)
        # 这样即使 gate 预测较小，local_feat 的核心信息也能完整保留
        out = local_feat + gate * context_feat

        # 最终输出 + 残差连接 (保底手段)
        return self.act(self.bn_out(self.pw_out(out)) + x_in)