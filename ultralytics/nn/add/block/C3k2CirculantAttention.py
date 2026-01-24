import torch
import torch.nn as nn
from timm.models.layers import Mlp, DropPath
from ultralytics.nn.modules import Conv, C3, C2f
from einops import rearrange

__all__ = ['C3k2_CirculantAttention']


class Linear(nn.Linear):
    r""" Linear layer for complex number inputs.
    """

    def __init__(self, in_features, out_features, device=None, dtype=None):
        super(Linear, self).__init__(in_features, out_features, False, device, dtype)

    def forward(self, x):
        x = torch.view_as_real(x).transpose(-2, -1)
        x = torch.nn.functional.linear(x, self.weight).transpose(-2, -1)
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        x = torch.view_as_complex(x.contiguous())
        return x


class CirculantAttention(nn.Module):
    r""" Circulant Attention
    https://arxiv.org/abs/2512.21542
    """

    def __init__(self, dim, proj_drop=0.):
        super().__init__()
        self.qkv = Linear(dim, dim * 3)
        self.gate = nn.Sequential(nn.Linear(dim, dim), nn.SiLU())
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w->b (h w) c')

        b, n, c = x.shape

        # Prepare Q, K, V, T
        #    (1) qkv=fc(x), qkv=fft(qkv) is mathematically equivalent to x=fft(x), qkv=fc(x)
        #    (2) The latter requires fewer FFT computations, delivering higher throughput
        t = self.gate(x)
        x = x.reshape(b, h, w, c)
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, chunks=3, dim=-1)

        # Equation 15 of the paper
        #    (1) We use d=1 in practice, as discussed in Table 5
        #    (2) The 1/N factor is implicitly achieved by norm='ortho' in calculating Q, K
        attn = torch.conj(q) * k
        attn = torch.fft.irfft2(attn, s=(h, w), dim=(1, 2), norm='ortho')

        # Equation 16 of the paper
        attn = attn.reshape(b, n, c).softmax(dim=1).reshape(b, h, w, c)
        attn = torch.fft.rfft2(attn, dim=(1, 2))
        x = torch.conj(attn) * v
        x = torch.fft.irfft2(x, s=(h, w), dim=(1, 2), norm='ortho')

        # Output
        x = x.reshape(b, n, c) * t
        x = self.proj(x)

        return rearrange(x, 'b (h w) c->b c h w', h=h)


class Block(nn.Module):

    def __init__(self, dim, num_heads=4, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.cpe = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = CirculantAttention(dim, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w->b (h w) c')
        b, n, c = x.shape
        x = x + self.cpe(x.reshape(b, h, w, c).permute(0, 3, 1, 2)).flatten(2).transpose(1, 2)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return rearrange(x, 'b (h w) c->b c h w', h=h)


class Bottleneck_CirculantAttention(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
        self.Attention = CirculantAttention(c2)

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.Attention(self.cv2(self.cv1(x))) if self.add else self.Attention(self.cv2(self.cv1(x)))


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(
            *(Bottleneck_CirculantAttention(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))


class C3k2_CirculantAttention(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_CirculantAttention(self.c, self.c, shortcut, g,
                                                                                          k=((3, 3), (3, 3)), e=1.0) for
            _ in range(n)
        )


if __name__ == '__main__':
    input = torch.randn(1, 32, 64, 64)  # x: (B, C, D,H, W) 3D图像维度
    model = C3k2_CirculantAttention(32, 32)
    output = model(input)
    print("DLKModule_input size:", input.size())
    print("DLKModule_Output size:", output.size())