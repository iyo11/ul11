import torch
import torch.nn as nn

__all__ = ['C3k2_Dual', 'C2f_Dual']


# --- 基础组件 ---

def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DualConv(nn.Module):
    """
    DualConv: 结合组卷积 (GC) 和 逐点卷积 (PWC)
    """

    def __init__(self, in_channels, out_channels, k=3, stride=1, g=4):
        super(DualConv, self).__init__()
        # Group Convolution - 捕捉空间特征
        self.gc = nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=stride, padding=autopad(k), groups=g,
                            bias=False)
        # Pointwise Convolution - 捕捉通道特征
        self.pwc = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        # 融合两条路径并进行归一化和激活
        return self.act(self.bn(self.gc(x) + self.pwc(x)))


# --- 核心 Bottleneck 修改 ---

class Bottleneck_Dual(nn.Module):
    """使用 DualConv 替换标准卷积的 Bottleneck"""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        # 将 cv2 替换为 DualConv
        self.cv2 = DualConv(c_, c2, k[1], 1, g=g if g > 1 else 4)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


# --- C3k2_Dual 实现 ---

class C3k2_Dual(nn.Module):
    """C3k2 模块，内部 Bottleneck 使用 DualConv"""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, self.c, 1, 1)
        self.cv2 = Conv(c1, self.c, 1, 1)
        self.cv3 = Conv(2 * self.c, c2, 1)

        # c3k 参数决定使用 k=(3,3) 还是 k=(3,3) 但带有特定的缩放逻辑
        # 在 YOLOv11 中，c3k=True 通常意味着更深的 block
        self.m = nn.ModuleList(
            Bottleneck_Dual(self.c, self.c, shortcut, g, k=(3, 3) if c3k else (3, 3), e=1.0)
            for _ in range(n)
        )

    def forward(self, x):
        """Forward pass through C3k2_Dual layer."""
        return self.cv3(torch.cat((self.m[i](self.cv1(x)) for i in range(len(self.m))), 1) + self.cv2(x))
        # 注意：上面的实现是基于典型的 C3 结构，但为了符合 YOLOv11 的 C3k2 特性：
        # 实际上 C3k2 是两个分支拼接。更准确的 YOLOv11 风格写法如下：

    def forward(self, x):
        x1 = self.cv1(x)
        for m in self.m:
            x1 = m(x1)
        return self.cv3(torch.cat((x1, self.cv2(x)), 1))


class C2f_Dual(nn.Module):
    # CSP Bottleneck with 2 convolutions
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# --- 测试代码 ---

if __name__ == "__main__":
    # 测试输入 (B, C, H, W)
    input_tensor = torch.randn(1, 64, 128, 128)

    # 初始化 C3k2_Dual
    # c1=64, c2=64, n=2 (包含2个 Bottleneck), c3k=True
    model = C3k2_Dual(64, 64, n=2, c3k=True)

    output = model(input_tensor)
    print(f"输入形状: {input_tensor.shape}")
    print(f"输出形状: {output.shape}")  # 应该保持 (1, 64, 128, 128)