import torch
from torch import nn
import numpy as np
from ultralytics.nn.modules import C3, C2f

__all__ = ['MSGDC3k2', 'DSC3k2_MSGDC']


class Config:
    def __init__(self):
        self.norm_layer = nn.LayerNorm
        self.layer_norm_eps = 1e-6
        self.weight_bits = 1  # 初始化为1，使用BinaryQuantizer
        self.input_bits = 1  # 初始化为1，使用BinaryQuantizer
        self.clip_val = 1.0
        self.recu = False


config = Config()


class BinaryQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        input = input[0]
        indicate_leftmid = ((input >= -1.0) & (input <= 0)).float()
        indicate_rightmid = ((input > 0) & (input <= 1.0)).float()

        grad_input = (indicate_leftmid * (2 + 2 * input) + indicate_rightmid * (2 - 2 * input)) * grad_output.clone()
        return grad_input


class TwnQuantizer(torch.autograd.Function):
    """Ternary Weight Networks (TWN)
    Ref: https://arxiv.org/abs/1605.04711
    """

    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise, type=None):
        """
        :param input: tensor to be ternarized
        :return: quantized tensor
        """
        ctx.save_for_backward(input, clip_val)
        input = torch.where(input < clip_val[1], input, clip_val[1])
        input = torch.where(input > clip_val[0], input, clip_val[0])
        if layerwise:
            m = input.norm(p=1).div(input.nelement())
            thres = 0.7 * m
            pos = (input > thres).float()
            neg = (input < -thres).float()
            mask = (input.abs() > thres).float()
            alpha = (mask * input).abs().sum() / mask.sum()
            result = alpha * pos - alpha * neg
        else:  # row-wise only for embed / weight
            n = input[0].nelement()
            m = input.data.norm(p=1, dim=1).div(n)
            thres = (0.7 * m).view(-1, 1).expand_as(input)
            pos = (input > thres).float()
            neg = (input < -thres).float()
            mask = (input.abs() > thres).float()
            alpha = ((mask * input).abs().sum(dim=1) / mask.sum(dim=1)).view(-1, 1)
            result = alpha * pos - alpha * neg

        return result

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        input, clip_val = ctx.saved_tensors  # unclipped input
        grad_input = grad_output.clone()
        grad_input[input.ge(clip_val[1])] = 0
        grad_input[input.le(clip_val[0])] = 0
        return grad_input, None, None, None, None


class SymQuantizer(torch.autograd.Function):
    """
        uniform quantization
    """

    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise, type=None):
        """
        :param ctx:
        :param input: tensor to be quantized
        :param clip_val: clip the tensor before quantization
        :param quant_bits: number of bits
        :return: quantized tensor
        """
        ctx.save_for_backward(input, clip_val)
        input = torch.where(input < clip_val[1], input, clip_val[1])
        input = torch.where(input > clip_val[0], input, clip_val[0])
        if layerwise:
            max_input = torch.max(torch.abs(input)).expand_as(input)
        else:
            if input.ndimension() <= 3:
                max_input = torch.max(torch.abs(input), dim=-1, keepdim=True)[0].expand_as(input).detach()
            elif input.ndimension() == 4:
                tmp = input.view(input.shape[0], input.shape[1], -1)
                max_input = torch.max(torch.abs(tmp), dim=-1, keepdim=True)[0].unsqueeze(-1).expand_as(input).detach()
            else:
                raise ValueError
        s = (2 ** (num_bits - 1) - 1) / max_input
        output = torch.round(input * s).div(s)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        input, clip_val = ctx.saved_tensors  # unclipped input
        grad_input = grad_output.clone()
        grad_input[input.ge(clip_val[1])] = 0
        grad_input[input.le(clip_val[0])] = 0
        return grad_input, None, None, None, None


class QuantizeConv2d(nn.Conv2d):
    def __init__(self, *kargs, bias=True, config=None):
        super(QuantizeConv2d, self).__init__(*kargs, bias=bias)
        self.weight_bits = config.weight_bits
        self.input_bits = config.input_bits
        self.recu = config.recu
        if self.weight_bits == 1:
            self.weight_quantizer = BinaryQuantizer
        elif self.weight_bits == 2:
            self.weight_quantizer = TwnQuantizer
            self.register_buffer('weight_clip_val', torch.tensor([-config.clip_val, config.clip_val]))
        elif self.weight_bits < 32:
            self.weight_quantizer = SymQuantizer
            self.register_buffer('weight_clip_val', torch.tensor([-config.clip_val, config.clip_val]))

        if self.input_bits == 1:
            self.act_quantizer = BinaryQuantizer
        elif self.input_bits == 2:
            self.act_quantizer = TwnQuantizer
            self.register_buffer('act_clip_val', torch.tensor([-config.clip_val, config.clip_val]))
        elif self.input_bits < 32:
            self.act_quantizer = SymQuantizer
            self.register_buffer('act_clip_val', torch.tensor([-config.clip_val, config.clip_val]))

    def forward(self, input, recu=False):
        if self.weight_bits == 1:

            real_weights = self.weight
            scaling_factor = torch.mean(
                torch.mean(torch.mean(abs(real_weights), dim=3, keepdim=True), dim=2, keepdim=True), dim=1,
                keepdim=True)
            real_weights = real_weights - real_weights.mean([1, 2, 3], keepdim=True)

            if recu:

                real_weights = real_weights / (
                        torch.sqrt(real_weights.var([1, 2, 3], keepdim=True) + 1e-5) / 2 / np.sqrt(2))
                EW = torch.mean(torch.abs(real_weights))
                Q_tau = (- EW * np.log(2 - 2 * 0.92)).detach().cpu().item()
                scaling_factor = scaling_factor.detach()
                binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
                cliped_weights = torch.clamp(real_weights, -Q_tau, Q_tau)
                weight = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights

            else:
                scaling_factor = scaling_factor.detach()
                binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
                cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
                weight = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        elif self.weight_bits < 32:
            weight = self.weight_quantizer.apply(self.weight, self.weight_clip_val, self.weight_bits, True)
        else:
            weight = self.weight

        if self.input_bits == 1:
            input = self.act_quantizer.apply(input)

        out = nn.functional.conv2d(input, weight, stride=self.stride, padding=self.padding, dilation=self.dilation,
                                   groups=self.groups)

        if not self.bias is None:
            out = out + self.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        return out


class RPReLU(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.move1 = nn.Parameter(torch.zeros(hidden_size))
        self.prelu = nn.PReLU(hidden_size)
        self.move2 = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        out = self.prelu((x - self.move1).transpose(-1, -2)).transpose(-1, -2) + self.move2
        return out


class LearnableBiasnn(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBiasnn, self).__init__()
        self.bias = nn.Parameter(torch.zeros([1, out_chn, 1, 1]), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out


class MSGDC(nn.Module):
    def __init__(self, in_chn, out_chn, config=config, dilation1=1, dilation2=3, dilation3=5, kernel_size=3, stride=1,
                 padding='same'):
        super(MSGDC, self).__init__()
        self.inc = in_chn
        self.ouc = out_chn
        if in_chn != out_chn:
            self.stem_conv = Conv(in_chn, in_chn, k=1)

        self.move = LearnableBiasnn(out_chn)
        self.cov1 = QuantizeConv2d(out_chn, out_chn, kernel_size, stride, padding, dilation1, 4, bias=True,
                                   config=config)
        self.cov2 = QuantizeConv2d(out_chn, out_chn, kernel_size, stride, padding, dilation2, 4, bias=True,
                                   config=config)
        self.cov3 = QuantizeConv2d(out_chn, out_chn, kernel_size, stride, padding, dilation3, 4, bias=True,
                                   config=config)
        self.norm = config.norm_layer(out_chn, eps=config.layer_norm_eps)
        self.act1 = RPReLU(out_chn)
        self.act2 = RPReLU(out_chn)
        self.act3 = RPReLU(out_chn)

    def forward(self, x):
        if self.inc != self.ouc:
            x = self.stem_conv(x)
        B, C, H, W = x.shape
        x = self.move(x)
        x1 = self.cov1(x).permute(0, 2, 3, 1).flatten(1, 2)
        x1 = self.act1(x1)
        x2 = self.cov2(x).permute(0, 2, 3, 1).flatten(1, 2)
        x2 = self.act2(x2)
        x3 = self.cov3(x).permute(0, 2, 3, 1).flatten(1, 2)
        x3 = self.act3(x3)
        x = self.norm(x1 + x2 + x3)
        return x.permute(0, 2, 1).view(-1, C, H, W).contiguous()


class MSGDConv(nn.Module):
    def __init__(self, inc, ouc, k=3, s=1, p=1, act=True):
        super().__init__()
        self.inc = inc
        self.ouc = ouc

        self.stem_conv = Conv(inc, ouc, k=k, s=s, p=p)
        self.MSGDConv = MSGDC(ouc, ouc)
        self.bn = nn.BatchNorm2d(ouc)
        self.act = Conv.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        x = self.stem_conv(x)
        x = self.MSGDConv(x)
        x = self.act(self.bn(x))
        return x


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


class DSC3k_MSGDC(C3):
    def __init__(self, c1, c2, n=1, shortcut=True, stage=1, g=1, e=0.5, k1=3, k2=5, d2=1):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)

        self.m = nn.Sequential(
            *(
                MSGDC(c_, c_) for _ in range(n)
            )
        )


class DSC3k2_MSGDC(C2f):
    def __init__(self, c1, c2, n=1, dsc3k=False, stage=1, e=0.5, g=1, shortcut=True, k1=3, k2=7, d2=1):
        super().__init__(c1, c2, n, shortcut, g, e)
        if dsc3k:
            self.m = nn.ModuleList(
                DSC3k_MSGDC(self.c, self.c, n=2, shortcut=shortcut, stage=stage, g=g, e=1.0, k1=k1, k2=k2, d2=d2) for _
                in range(n))
        else:
            self.m = nn.ModuleList(MSGDC(self.c, self.c) for _ in range(n))


class DSBottleneck_MSGDC(nn.Module):
    def __init__(self, c1, c2, shortcut=True, e=0.5, k1=3, k2=5, d2=1):
        super().__init__()
        c_ = int(c2 * e)
        # self.cv1 = DSConv(c1, c_, k1, s=1, p=None, d=1)
        # self.cv2 = DSConv(c_, c2, k2, s=1, p=None, d=d2)
        self.cv1 = MSGDC(c1, c1)  # 在这里可以替换一种DSConv，也可以都替换。自己可以做一下消融实验
        self.cv2 = MSGDC(c2, c2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class MSGDC3(C3):

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k1=3, k2=5, d2=1):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)

        self.m = nn.Sequential(
            *(
                DSBottleneck_MSGDC(
                    c_, c_,
                    shortcut=shortcut,
                    e=1.0,
                    k1=k1,
                    k2=k2,
                    d2=d2
                )
                for _ in range(n)
            )
        )


# class DSC3k2(C2f):
class MSGDC3k2(C2f):
    def __init__(self, c1, c2, n=1, dsc3k=False, e=0.5, g=1, shortcut=True, k1=3, k2=7, d2=1):
        super().__init__(c1, c2, n, shortcut, g, e)
        if dsc3k:
            self.m = nn.ModuleList(
                MSGDC3(self.c, self.c, n=2, shortcut=shortcut, g=g, e=1.0, k1=k1, k2=k2, d2=d2) for _ in range(n))
        else:
            self.m = nn.ModuleList(
                DSBottleneck_MSGDC(self.c, self.c, shortcut=shortcut, e=1.0, k1=k1, k2=k2, d2=d2) for _ in range(n))