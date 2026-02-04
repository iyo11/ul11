import torch
import torch.nn as nn
from torch import Tensor

__all__ = ['CGHalfConv']

class HalfConv(nn.Module):
    """
    Partial Convolution (PConv) style layer that only processes a portion
    of the input channels to save FLOPs.
    """

    def __init__(self, c1, n_div=4):  # YOLO-style uses c1 (input channels)
        super().__init__()
        self.dim_conv3 = c1 // n_div
        self.dim_untouched = c1 - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        # Optimized for inference: use split only if necessary
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        return torch.cat((x1, x2), 1)


class CGHalfConv(nn.Module):
    """
    Split-Transform-Merge module using multiple HalfConv blocks with a residual connection.
    """

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        # c1: input channels, c2: output channels (must be equal for residual)
        super().__init__()
        assert c1 == c2, "CGHalfConv requires input channels to equal output channels for the shortcut."

        self.div_dim = c1 // 3
        self.remainder_dim = c1 % 3

        # Define dimensions for each split
        d1 = self.div_dim
        d2 = self.div_dim
        d3 = self.div_dim + self.remainder_dim

        self.p1 = HalfConv(d1, n_div=2)
        self.p2 = HalfConv(d2, n_div=2)
        self.p3 = HalfConv(d3, n_div=2)

    def forward(self, x):
        # Split into 3 groups across the channel dimension
        x1, x2, x3 = torch.split(x, [self.div_dim, self.div_dim, self.div_dim + self.remainder_dim], dim=1)

        # Apply transforms and concatenate
        out = torch.cat((self.p1(x1), self.p2(x2), self.p3(x3)), dim=1)

        # Residual connection
        return out + x