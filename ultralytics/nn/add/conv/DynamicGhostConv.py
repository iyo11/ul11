import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from timm.layers import CondConv2d
except Exception:
    from timm.models.layers import CondConv2d


class ConvBNAct(nn.Module):
    """
    Standard Conv-BN-Activation block.
    """
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()

        if p is None:
            p = k // 2

        self.conv = nn.Conv2d(
            in_channels=c1,
            out_channels=c2,
            kernel_size=k,
            stride=s,
            padding=p,
            groups=g,
            bias=False
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DynamicGhostConv(nn.Module):
    """
    Lightweight Dynamic Ghost Convolution.

    注意：
    这个类名虽然叫 DynamicGhostConv，
    但结构是轻量版 LiteDynamicGhostConv。

    Structure:
        input x
          |
          |-- primary branch:
          |      normal Conv-BN-SiLU
          |
          |-- cheap branch:
          |      dynamic depthwise CondConv-BN-SiLU
          |
          |-- concat
          |
          |-- slice to c2 channels

    Compared with the previous heavy DynamicGhostConv:
        Old:
            primary branch = CondConv2d(c1 -> c_mid, k=3, num_experts=4)

        New:
            primary branch = normal Conv(c1 -> c_mid, k=3)
            cheap branch   = dynamic depthwise CondConv2d(c_mid -> c_mid)

    This avoids parameter explosion caused by CondConv on the main 3x3 branch.
    """

    def __init__(
        self,
        c1,
        c2,
        k=1,
        s=1,
        ratio=2,
        cheap_k=3,
        num_experts=2,
        act=True
    ):
        super().__init__()

        self.c1 = c1
        self.c2 = c2
        self.ratio = ratio
        self.num_experts = num_experts

        # Primary feature channels.
        # Example: c2=128, ratio=2 -> c_mid=64
        c_mid = int((c2 + ratio - 1) // ratio)
        self.c_mid = c_mid

        # 1. Primary branch: normal convolution
        self.primary_conv = ConvBNAct(
            c1=c1,
            c2=c_mid,
            k=k,
            s=s,
            p=k // 2,
            g=1,
            act=act
        )

        # 2. Routing function for CondConv experts
        self.routing = nn.Linear(c_mid, num_experts)

        # 3. Cheap branch: dynamic depthwise convolution
        self.cheap_conv = CondConv2d(
            in_channels=c_mid,
            out_channels=c_mid,
            kernel_size=cheap_k,
            stride=1,
            padding=cheap_k // 2,
            dilation=1,
            groups=c_mid,
            bias=False,
            num_experts=num_experts
        )

        self.cheap_bn = nn.BatchNorm2d(c_mid)
        self.cheap_act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        # Main features
        y = self.primary_conv(x)

        # Generate routing weights
        pooled = F.adaptive_avg_pool2d(y, 1).flatten(1)
        routing_weights = torch.sigmoid(self.routing(pooled))

        # Dynamic ghost features
        ghost = self.cheap_conv(y, routing_weights)
        ghost = self.cheap_bn(ghost)
        ghost = self.cheap_act(ghost)

        # Concatenate primary and ghost features
        out = torch.cat([y, ghost], dim=1)

        # Ensure output channels = c2
        out = out[:, :self.c2, :, :]

        return out


if __name__ == "__main__":
    x = torch.randn(2, 64, 80, 80)

    model = DynamicGhostConv(
        c1=64,
        c2=128,
        k=3,
        s=2,
        ratio=2,
        cheap_k=3,
        num_experts=2,
        act=True
    )

    y = model(x)

    print("Input shape :", x.shape)
    print("Output shape:", y.shape)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Total params    :", total_params)
    print("Trainable params:", trainable_params)