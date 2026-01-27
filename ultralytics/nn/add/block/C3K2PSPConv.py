import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import C3, C2f

__all__ = ['C3k2_PSConv', 'C3k2_SPSConv']
# 论文地址：https://arxiv.org/pdf/2412.16986
'''
来自 AAAI 2025顶会
即插即用卷积模块: PSConv 风车形状卷积模块  Pinwheel-shaped Convolution（PConv）
小目标检测损失函数: SDIoU,作为YOLOv8v10v11小目标检测任务的损失函数改进点！
近年来，基于卷积神经网络 （CNN） 的红外小目标检测方法取得了出色的性能。
然而，这些方法通常采用标准卷积，而忽略了红外小目标像素分布的空间特性。
因此，我们提出了一种新的风车形卷积 （PConv） 来替代骨干网络下层的标准卷积。
PConv 更好地与暗淡小目标的像素高斯空间分布保持一致，增强了特征提取，显著增加了感受野，并且仅引入了最小的参数增加。
此外，虽然最近的损失函数结合了尺度和位置损失，但它们没有充分考虑这些损失在不同目标尺度上的不同灵敏度，从而限制了对暗小目标的检测性能。
为了克服这个问题，我们提出了一种基于尺度的动态 （SD） 损失，它根据目标大小动态调整尺度和位置损失的影响，从而提高网络检测不同尺度目标的能力。
我们构建了一个新的基准 SIRST-UAVB，这是迄今为止最大、最具挑战性的实拍单帧红外小目标检测数据集。
最后，通过将 PConv 和 SD Loss 集成到最新的小目标检测算法中，
我们在 IRSTD-1K 和 SIRST-UAVB 数据集上实现了显著的性能改进，验证了我们方法的有效性和通用性。
适用于：红外小目标检测，小目标检测任务，目标检测，图像分割，语义分割，图像增强等所有一切计算机视觉CV任务通用的即插即用卷积模块。
'''


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class PConv(nn.Module):
    ''' Pinwheel-shaped Convolution using the Asymmetric Padding method. '''

    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()

        # 定义4种非对称填充方式，用于风车形状卷积的实现
        p = [(k, 0, 1, 0), (0, k, 0, 1), (0, 1, k, 0), (1, 0, 0, k)]  # 每个元组表示 (左, 上, 右, 下) 填充
        self.pad = [nn.ZeroPad2d(padding=(p[g])) for g in range(4)]  # 创建4个填充层

        # 定义水平方向卷积操作，卷积核大小为 (1, k)，步幅为 s，输出通道数为 c2 // 4
        self.cw = Conv(c1, c2 // 4, (1, k), s=s, p=0)

        # 定义垂直方向卷积操作，卷积核大小为 (k, 1)，步幅为 s，输出通道数为 c2 // 4
        self.ch = Conv(c1, c2 // 4, (k, 1), s=s, p=0)

        # 最终合并卷积结果的卷积层，卷积核大小为 (2, 2)，输出通道数为 c2
        self.cat = Conv(c2, c2, 2, s=1, p=0)

    def forward(self, x):
        # 对输入 x 进行不同填充和卷积操作，得到四个方向的特征
        yw0 = self.cw(self.pad[0](x))  # 水平方向，第一个填充方式
        yw1 = self.cw(self.pad[1](x))  # 水平方向，第二个填充方式
        yh0 = self.ch(self.pad[2](x))  # 垂直方向，第一个填充方式
        yh1 = self.ch(self.pad[3](x))  # 垂直方向，第二个填充方式

        # 将四个卷积结果在通道维度拼接，并通过一个额外的卷积层处理，最终输出
        return self.cat(torch.cat([yw0, yw1, yh0, yh1], dim=1))  # 在通道维度拼接，并通过 cat 卷积层处理


class APBottleneck(nn.Module):
    """Asymmetric Padding bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()

        self.psconv = PConv(c1, c2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""

        return x + self.psconv(x) if self.add else self.psconv(x)


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(APBottleneck(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))


class C3k2_PSConv(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else APBottleneck(self.c, self.c, shortcut, g,
                                                                         k=((3, 3), (3, 3)), e=1.0) for _ in range(n)
        )


# SPConv二创新模块代码
# 二次创新模块 SPConv  移动风车形状卷积
# SPConv是利用SCM 基于位移操作的新型模块，在不显著增加计算开销的前提下提升AAAI2025PConv风车卷积模块的性能


class SPConv(nn.Module):
    ''' Pinwheel-shaped Convolution using the Asymmetric Padding method. '''

    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()

        # 定义4种非对称填充方式，用于风车形状卷积的实现
        p = [(k, 0, 1, 0), (0, k, 0, 1), (0, 1, k, 0), (1, 0, 0, k)]  # 每个元组表示 (左, 上, 右, 下) 填充
        self.pad = [nn.ZeroPad2d(padding=(p[g])) for g in range(4)]  # 创建4个填充层

        # 定义水平方向卷积操作，卷积核大小为 (1, k)，步幅为 s，输出通道数为 c2 // 4
        self.cw = Conv(c1, c2 // 4, (1, k), s=s, p=0)

        # 定义垂直方向卷积操作，卷积核大小为 (k, 1)，步幅为 s，输出通道数为 c2 // 4
        self.ch = Conv(c1, c2 // 4, (k, 1), s=s, p=0)

        # 最终合并卷积结果的卷积层，卷积核大小为 (2, 2)，输出通道数为 c2
        self.cat = Conv(c2, c2, 2, s=1, p=0)

        self.shift_size = 1

    def forward(self, x):
        # 对输入 x 进行不同填充和卷积操作，得到四个方向的特征
        yw0 = self.cw(self.pad[0](x))  # 水平方向，第一个填充方式
        yw1 = self.cw(self.pad[1](x))  # 水平方向，第二个填充方式
        yh0 = self.ch(self.pad[2](x))  # 垂直方向，第一个填充方式
        yh1 = self.ch(self.pad[3](x))  # 垂直方向，第二个填充方式

        x1 = torch.roll(yw0, self.shift_size, dims=2)  # [:,:,1:,:]

        x2 = torch.roll(yw1, -self.shift_size, dims=2)  # [:,:,:-1,:]

        x3 = torch.roll(yh0, self.shift_size, dims=3)  # [:,:,:,1:]

        x4 = torch.roll(yh1, -self.shift_size, dims=3)  # [:,:,:,:-1]

        out = torch.cat([x1, x2, x3, x4], 1)
        # 将四个卷积结果在通道维度拼接，并通过一个额外的卷积层处理，最终输出
        return self.cat(out)  # 在通道维度拼接，并通过 cat 卷积层处理


'''
来自CVPR 2025 顶会
即插即用模块： SCM 特征位移混合模块
# 论文地址;https://arxiv.org/abs/2503.02394
主要内容：
模型二值化在实现卷积神经网络（CNN）的实时高效计算方面取得了显著进展，为视觉Transformer（ViT）在边缘设备上的部署挑战提供了潜在解决方案。
然而，由于CNN和Transformer架构的结构差异，直接将二值化CNN策略应用于ViT模型会导致性能显著下降。
为解决这一问题，我们提出了BHViT——一种适合二值化的混合ViT架构及其全二值化模型，其设计基于以下三个重要观察：
1.局部信息交互与分层特征聚合：BHViT利用从粗到细的分层特征聚合技术，减少因冗余token带来的计算开销。
2.基于移位操作的新型模块：提出一种基于移位操作的模块（SCM），在不显著增加计算负担的情况下提升二值化多层感知机（MLP）的性能。
3.量化分解的注意力矩阵二值化方法：提出一种基于量化分解的创新方法，用于评估二值化注意力矩阵中各token的重要性。
该Shift_channel_mix（SCM）模块是论文中提出的一个轻量化模块，用于增强二进制多层感知器（MLP）在二进制视觉变换器（BViT）中的表现。
它通过对输入特征图进行不同的移位操作，帮助缓解信息丢失和梯度消失的问题，从而提高网络的性能，同时避免增加过多的计算开销。
SCM模块的主要操作包括：
1.水平移位（Horizontal Shift）：通过torch.roll函数将特征图的列按指定的大小进行右/左移操作。这种操作模拟了在处理二进制向量时的特征循环，增强了表示能力。
2.垂直移位（Vertical Shift）：类似于水平移位，垂直移位会使特征图的行发生上下移动。这有助于捕获跨行的信息，同时适应不同的特征维度。
在代码实现中，torch.chunk将输入特征图沿着通道维度分成四个部分，之后通过不同的移位操作处理每一部分，最后将处理后的四个部分通过torch.cat拼接起来，形成最终的输出。
SCM模块适合：目标检测，图像分割，语义分割，图像增强，图像去噪，遥感语义分割，图像分类等所有CV任务通用的即插即用模块
这个SCM轻量小巧模块，建议最好搭配其它模块一起使用！
'''


class Shift_channel_mix(nn.Module):
    def __init__(self, shift_size=1):
        super(Shift_channel_mix, self).__init__()
        self.shift_size = shift_size

    def forward(self, x):  # x的张量 [B,C,H,W]
        x1, x2, x3, x4 = x.chunk(4, dim=1)

        x1 = torch.roll(x1, self.shift_size, dims=2)  # [:,:,1:,:]

        x2 = torch.roll(x2, -self.shift_size, dims=2)  # [:,:,:-1,:]

        x3 = torch.roll(x3, self.shift_size, dims=3)  # [:,:,:,1:]

        x4 = torch.roll(x4, -self.shift_size, dims=3)  # [:,:,:,:-1]

        x = torch.cat([x1, x2, x3, x4], 1)

        return x


class APBottleneck_SPConv(nn.Module):
    """Asymmetric Padding bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()

        self.psconv = SPConv(c1, c2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""

        return x + self.psconv(x) if self.add else self.psconv(x)


class C3k_(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(APBottleneck_SPConv(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))


class C3k2_SPSConv(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k_(self.c, self.c, 2, shortcut, g) if c3k else APBottleneck_SPConv(self.c, self.c, shortcut, g,
                                                                                 k=((3, 3), (3, 3)), e=1.0) for _ in
            range(n)
        )


# 输入 B C H W, 输出 B C H W
if __name__ == "__main__":
    module = PConv(c1=64, c2=128, k=3, s=1)
    input_tensor = torch.randn(1, 64, 128, 128)
    output_tensor = module(input_tensor)
    print('Input size:', input_tensor.size())  # 打印输入张量的形状
    print('Output size:', output_tensor.size())  # 打印输出张量的形状