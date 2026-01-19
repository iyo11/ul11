import torch
import torch.nn as nn

__all__ = ["ContextGuidedSimAMConv"]


class FGlo(nn.Module):
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


class SimAMAlpha(nn.Module):
    """
    SimAM (energy-based) with learnable scalar alpha.
    - alpha=0 => exact identity (stable start)
    - alpha is learnable => model decides strength
    """
    def __init__(self, e_lambda=1e-4, init_alpha=0.0):
        super().__init__()
        self.e_lambda = e_lambda
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))

    def forward(self, x):
        mu = x.mean(dim=(2, 3), keepdim=True)
        x_mu = x - mu
        var = (x_mu * x_mu).mean(dim=(2, 3), keepdim=True)
        att = torch.sigmoid(x_mu * x_mu / (4.0 * (var + self.e_lambda)) + 0.5)

        # residual gating (geometry-friendly)
        return x + self.alpha * (x * att - x)


class ContextGuidedSimAMConv(nn.Module):
    """
    HARD switch outside:
      simam=False -> Identity (strictly no SimAM)
      simam=True  -> SimAMAlpha (learnable alpha, alpha=0 stable init)
    """
    def __init__(self, nIn, nOut, dilation_rate=2, reduction=16,
                 simam=True, e_lambda=1e-4, simam_init_alpha=0.0):
        super().__init__()

        self.conv_down = nn.Sequential(
            nn.Conv2d(nIn, nOut, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nOut, eps=1e-3),
            nn.PReLU(nOut)
        )

        self.F_loc = nn.Conv2d(nOut, nOut, kernel_size=3, padding=1, groups=nOut, bias=False)
        self.F_sur = nn.Conv2d(
            nOut, nOut,
            kernel_size=3,
            padding=dilation_rate,
            dilation=dilation_rate,
            groups=nOut,
            bias=False
        )

        self.bn = nn.BatchNorm2d(2 * nOut, eps=1e-3)
        self.act = nn.PReLU(2 * nOut)
        self.reduce = nn.Conv2d(2 * nOut, nOut, kernel_size=1, bias=False)

        # ✅ hard switch
        self.simam = SimAMAlpha(e_lambda=e_lambda, init_alpha=simam_init_alpha) if simam else nn.Identity()

        self.F_glo = FGlo(nOut, reduction)

        # (optional) for logging
        self.last_alpha = None

    def forward(self, x):
        x = self.conv_down(x)

        loc = self.F_loc(x)
        sur = self.F_sur(x)

        feat = torch.cat([loc, sur], dim=1)
        feat = self.reduce(self.act(self.bn(feat)))

        feat = self.simam(feat)

        # log alpha if SimAM is enabled
        if hasattr(self.simam, "alpha"):
            self.last_alpha = self.simam.alpha.detach()
        else:
            self.last_alpha = None

        out = self.F_glo(feat)
        return out
