import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.block import PSABlock
from ultralytics.nn.modules import C3, C2f, C2PSA

Conv2d = nn.Conv2d
__all__ = ['C3k2_DIFF']


###############################################################################
def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x


class ElementScale(nn.Module):
    """A learnable element-wise scaler."""

    def __init__(self, embed_dims, init_value=0., requires_grad=True):
        super(ElementScale, self).__init__()
        self.scale = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)),
            requires_grad=requires_grad
        )

    def forward(self, x):
        return x * self.scale


class FFN_DIFF(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2, bias=False):
        super(FFN_DIFF, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.sigma = ElementScale(
            hidden_features // 4, init_value=1e-5, requires_grad=True)
        self.decompose = nn.Conv2d(
            in_channels=hidden_features // 4,  # C -> 1
            out_channels=1, kernel_size=1,
        )
        self.decompose_act = nn.GELU()
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv_5 = nn.Conv2d(hidden_features // 4, hidden_features // 4, kernel_size=5,
                                  stride=1, padding=2, groups=hidden_features // 4, bias=bias)
        self.dwconv_dilated2_1 = nn.Conv2d(hidden_features // 4, hidden_features // 4, kernel_size=3,
                                           stride=1, padding=2, groups=hidden_features // 4,
                                           bias=bias, dilation=2)
        self.p_unshuffle = nn.PixelUnshuffle(2)
        self.p_shuffle = nn.PixelShuffle(2)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def feat_decompose(self, x):
        # x_d: [B, C, H, W] -> [B, 1, H, W]
        x = x + self.sigma(x - self.decompose_act(self.decompose(x)))
        return x

    def forward(self, x):
        x = self.project_in(x)
        x = self.p_shuffle(x)
        x = channel_shuffle(x, groups=1)
        x1, x2 = x.chunk(2, dim=1)
        x1 = self.dwconv_5(x1)
        x2 = self.dwconv_dilated2_1(x2)
        x = F.mish(x2) * x1
        x = self.feat_decompose(x)
        x = self.p_unshuffle(x)
        x = self.project_out(x)
        return x


class C3k_DIFF(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(FFN_DIFF(c_) for _ in range(n)))


class C3k2_DIFF(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k_DIFF(self.c, self.c, 2, shortcut, g) if c3k else FFN_DIFF(self.c) for _ in range(n)
        )


class PSABlock_DIFF(PSABlock):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        super().__init__(c, attn_ratio, num_heads, shortcut)

        self.ffn = FFN_DIFF(c)


class C2PSA_DIFF(C2PSA):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)

        self.m = nn.Sequential(*(PSABlock_DIFF(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))