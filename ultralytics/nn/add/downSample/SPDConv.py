from torch import nn
from ultralytics.nn.modules import Conv


class SPDConv(nn.Module):
    def __init__(self, c1, c2, dimension=1):
        super().__init__()
        self.spd = nn.PixelUnshuffle(downscale_factor=2)
        self.conv = Conv(c1 * 4, c2, k=3, s=1)
    def forward(self, x):
        return self.conv(self.spd(x))