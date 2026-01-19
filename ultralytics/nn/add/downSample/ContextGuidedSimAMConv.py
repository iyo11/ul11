import torch
import torch.nn as nn

__all__ = ['ContextGuidedSimAMConv']


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


class ResidualSimAM(nn.Module):
    """
    Residual SimAM (energy-based), no learnable params, HARD on/off handled outside.

    attention:
      att = sigmoid( (x-mu)^2 / (4*(var+e_lambda)) + 0.5 )

    residual form:
      y = x + (x * att - x)

    IMPORTANT:
      - This module itself does NOT decide on/off.
      - Use `nn.Identity()` outside to fully disable (exactly y=x).
    """
    def __init__(self, e_lambda=1e-4):
        super().__init__()
        self.e_lambda = e_lambda

    def forward(self, x):
        # x: [B, C, H, W]
        mu = x.mean(dim=(2, 3), keepdim=True)
        x_mu = x - mu
        var = (x_mu * x_mu).mean(dim=(2, 3), keepdim=True)

        att = torch.sigmoid(x_mu * x_mu / (4.0 * (var + self.e_lambda)) + 0.5)

        # residual-style (geometry-friendly) reweighting
        return x + (x * att - x)


class ContextGuidedSimAMConv(nn.Module):
    """
    Context Guided Downsample Block + (optional) ResidualSimAM + FGlo

    Input : [B, nIn,  H,  W]
    Output: [B, nOut, H/2, W/2]

    YAML usage (Ultralytics):
      - [-1, 1, ContextGuidedSimAMConv, [128, 2, 16, True]]
        -> nOut=128, dilation_rate=2, reduction=16, simam=True
      - [-1, 1, ContextGuidedSimAMConv, [512, 2, 16, False]]
        -> simam=False => SimAM replaced by Identity (STRICT y=x)
    """
    def __init__(self, nIn, nOut, dilation_rate=2, reduction=16, simam=True, e_lambda=1e-4):
        super().__init__()

        # 1) Downsample + channel expand
        self.conv_down = nn.Sequential(
            nn.Conv2d(nIn, nOut, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nOut, eps=1e-3),
            nn.PReLU(nOut)
        )

        # 2) Local branch (DWConv 3x3)
        self.F_loc = nn.Conv2d(nOut, nOut, kernel_size=3, padding=1, groups=nOut, bias=False)

        # 3) Surrounding/context branch (Dilated DWConv 3x3)
        self.F_sur = nn.Conv2d(
            nOut, nOut,
            kernel_size=3,
            padding=dilation_rate,
            dilation=dilation_rate,
            groups=nOut,
            bias=False
        )

        # 4) Concat -> BN + Act
        self.bn = nn.BatchNorm2d(2 * nOut, eps=1e-3)
        self.act = nn.PReLU(2 * nOut)

        # 5) Reduce back to nOut
        self.reduce = nn.Conv2d(2 * nOut, nOut, kernel_size=1, bias=False)

        # 6) HARD on/off: True -> ResidualSimAM, False -> Identity (exact)
        self.simam = ResidualSimAM(e_lambda=e_lambda) if simam else nn.Identity()

        # 7) Global refinement
        self.F_glo = FGlo(nOut, reduction)

    def forward(self, x):
        x = self.conv_down(x)

        loc = self.F_loc(x)
        sur = self.F_sur(x)

        feat = torch.cat([loc, sur], dim=1)
        feat = self.reduce(self.act(self.bn(feat)))

        # True: ResidualSimAM / False: Identity (no-op)
        feat = self.simam(feat)

        out = self.F_glo(feat)
        return out


if __name__ == "__main__":
    # quick sanity check
    inp = torch.randn(2, 64, 256, 256)

    m_on = ContextGuidedSimAMConv(64, 128, dilation_rate=2, reduction=16, simam=True)
    m_off = ContextGuidedSimAMConv(64, 128, dilation_rate=2, reduction=16, simam=False)

    y_on = m_on(inp)
    y_off = m_off(inp)

    print("on :", y_on.shape)
    print("off:", y_off.shape)

    # strict no-op check for the simam block itself when disabled:
    # (Here it's inside the full module, so outputs differ due to different random weights,
    #  but within a single instantiated module, simam=False guarantees feat is unchanged at that step.)
