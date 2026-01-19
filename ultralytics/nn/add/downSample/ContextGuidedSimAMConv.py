import torch
import torch.nn as nn

__all__ = ['ContextGuidedConv_SimAM']


class FGlo(nn.Module):
    """
    全局上下文模块 (Global Context Refinement)
    """
    def __init__(self, channel, reduction=16):
        super().__init__()
        hidden = max(1, channel // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channel, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(self.avg_pool(x))


class SimAM(nn.Module):
    """
    SimAM: parameter-free attention (energy-based)s
    这里做一个工程友好版：
      y = x + alpha * (x * att - x)  # residual attention, alpha=0 init
    """
    def __init__(self, e_lambda=1e-4, init_alpha=0.0):
        super().__init__()
        self.e_lambda = e_lambda
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))

    def forward(self, x):
        # x: [B, C, H, W]
        # variance over spatial dims per channel
        mu = x.mean(dim=(2, 3), keepdim=True)
        x_mu = x - mu
        var = (x_mu * x_mu).mean(dim=(2, 3), keepdim=True)

        # SimAM attention map
        att = torch.sigmoid(x_mu * x_mu / (4 * (var + self.e_lambda)) + 0.5)

        # residual-style to preserve geometry
        y = x + self.alpha * (x * att - x)
        return y


class ContextGuidedSimAMConv(nn.Module):
    """
    下采样模块: (H,W,C) -> (H/2, W/2, nOut)
    保留 FGlo，同时在融合后加入 SimAM（残差式）
    """
    def __init__(self, nIn, nOut, dilation_rate=2, reduction=16,
                 simam=True, e_lambda=1e-4, simam_init_alpha=0.0):
        super().__init__()

        # 1) 下采样 + 通道扩张
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(nIn, nOut, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nOut, eps=1e-3),
            nn.PReLU(nOut)
        )

        # 2) 局部分支
        self.F_loc = nn.Conv2d(nOut, nOut, kernel_size=3, padding=1,
                               groups=nOut, bias=False)

        # 3) 上下文分支（空洞）
        self.F_sur = nn.Conv2d(nOut, nOut, kernel_size=3, padding=dilation_rate,
                               dilation=dilation_rate, groups=nOut, bias=False)

        # 4) 拼接后 BN+Act
        self.bn = nn.BatchNorm2d(2 * nOut, eps=1e-3)
        self.act = nn.PReLU(2 * nOut)

        # 5) 降维回 nOut
        self.reduce = nn.Conv2d(2 * nOut, nOut, kernel_size=1, bias=False)

        # 6) SimAM（可选）
        self.use_simam = simam
        self.simam = SimAM(e_lambda=e_lambda, init_alpha=simam_init_alpha) if simam else nn.Identity()

        # 7) FGlo（保留）
        self.F_glo = FGlo(nOut, reduction)

    def forward(self, input):
        x = self.conv1x1(input)

        loc = self.F_loc(x)
        sur = self.F_sur(x)

        joi_feat = torch.cat([loc, sur], dim=1)
        joi_feat = self.reduce(self.act(self.bn(joi_feat)))

        # ✅ 先 SimAM（局部能量对比，tiny 友好），再 FGlo（全局重标定）
        joi_feat = self.simam(joi_feat)
        out = self.F_glo(joi_feat)

        return out
