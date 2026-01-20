import torch
import torch.nn as nn

__all__ = ["CoordinateAttention"]


class CoordinateAttention(nn.Module):
    """
    Coordinate Attention (CA)
    Ultralytics-v11 style init: __init__(c1, c2, reduction=32, use_ca=True)

    - c1: input channels
    - c2: output channels
    - reduction: channel reduction ratio for the CA bottleneck
    - use_ca: True -> enable CA, False -> exact identity (except optional c1->c2 align)
    """

    def __init__(self, c1, c2, reduction=32, use_ca=True):
        super().__init__()
        self.use_ca = use_ca
        self.c1 = c1
        self.c2 = c2

        # 1) 对齐通道（保证 c1 != c2 也能用）
        self.align = nn.Identity() if c1 == c2 else nn.Conv2d(c1, c2, 1, bias=False)

        # 2) CA 主体只在 c2 上做（对齐后）
        mip = max(8, c2 // reduction)

        # pool along H and W
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # (B,C,H,1)
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # (B,C,1,W)

        # shared transform
        self.conv1 = nn.Conv2d(c2, mip, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.SiLU(inplace=True)

        # split transforms
        self.conv_h = nn.Conv2d(mip, c2, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv_w = nn.Conv2d(mip, c2, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        x = self.align(x)

        if not self.use_ca:
            return x  # 完全关闭：精确 Identity（除非 c1!=c2 需要 align）

        b, c, h, w = x.size()

        x_h = self.pool_h(x)                        # (B,C,H,1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)    # (B,C,W,1)

        y = torch.cat([x_h, x_w], dim=2)            # (B,C,H+W,1)
        y = self.act(self.bn1(self.conv1(y)))       # (B,mip,H+W,1)

        y_h, y_w = torch.split(y, [h, w], dim=2)
        y_w = y_w.permute(0, 1, 3, 2)               # (B,mip,1,W)

        a_h = torch.sigmoid(self.conv_h(y_h))       # (B,C,H,1)
        a_w = torch.sigmoid(self.conv_w(y_w))       # (B,C,1,W)

        return x * a_h * a_w
