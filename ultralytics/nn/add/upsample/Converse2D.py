import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['Converse2D']


class LayerNorm(nn.Module):
    '''
    LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    '''

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class Converse2D(nn.Module):
    def __init__(self, in_channels, kernel_size=5, scale=2, padding=4, padding_mode='circular', eps=1e-5):
        super(Converse2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.kernel_size = kernel_size
        self.scale = scale
        self.padding = padding
        self.padding_mode = padding_mode
        self.eps = eps

        # ensure depthwise
        assert self.out_channels == self.in_channels
        self.weight = nn.Parameter(torch.randn(1, self.in_channels, self.kernel_size, self.kernel_size))
        self.bias = nn.Parameter(torch.zeros(1, self.in_channels, 1, 1))
        self.weight.data = nn.functional.softmax(self.weight.data.view(1, self.in_channels, -1), dim=-1).view(1,
                                                                                                              self.in_channels,
                                                                                                              self.kernel_size,
                                                                                                              self.kernel_size)

    def forward(self, x):

        if self.padding > 0:
            x = nn.functional.pad(x, pad=[self.padding, self.padding, self.padding, self.padding],
                                  mode=self.padding_mode, value=0)

        # self.biaseps = torch.sigmoid(self.bias - 9.0) + self.eps
        biaseps = torch.sigmoid(self.bias - 9.0) + self.eps
        _, _, h, w = x.shape
        STy = self.upsample(x, scale=self.scale)
        if self.scale != 1:
            x = nn.functional.interpolate(x, scale_factor=self.scale, mode='nearest')
            # x = nn.functional.interpolate(x, scale_factor=self.scale, mode='bilinear',align_corners=False)
        # x = torch.zeros_like(x)

        FB = self.p2o(self.weight, (h * self.scale, w * self.scale))
        FBC = torch.conj(FB)
        F2B = torch.pow(torch.abs(FB), 2)
        FBFy = FBC * torch.fft.fftn(STy, dim=(-2, -1))

        FR = FBFy + torch.fft.fftn(biaseps * x, dim=(-2, -1))
        x1 = FB.mul(FR)
        FBR = torch.mean(self.splits(x1, self.scale), dim=-1, keepdim=False)
        invW = torch.mean(self.splits(F2B, self.scale), dim=-1, keepdim=False)
        invWBR = FBR.div(invW + biaseps)
        FCBinvWBR = FBC * invWBR.repeat(1, 1, self.scale, self.scale)
        FX = (FR - FCBinvWBR) / biaseps
        out = torch.real(torch.fft.ifftn(FX, dim=(-2, -1)))

        if self.padding > 0:
            out = out[..., self.padding * self.scale:-self.padding * self.scale,
            self.padding * self.scale:-self.padding * self.scale]

        return out

    def splits(self, a, scale):
        '''
        Split tensor `a` into `scale x scale` distinct blocks.
        Args:
            a: Tensor of shape (..., W, H)
            scale: Split factor
        Returns:
            b: Tensor of shape (..., W/scale, H/scale, scale^2)
        '''
        *leading_dims, W, H = a.size()
        W_s, H_s = W // scale, H // scale

        # Reshape to separate the scale factors
        b = a.view(*leading_dims, scale, W_s, scale, H_s)

        # Generate the permutation order
        permute_order = list(range(len(leading_dims))) + [len(leading_dims) + 1, len(leading_dims) + 3,
                                                          len(leading_dims), len(leading_dims) + 2]
        b = b.permute(*permute_order).contiguous()

        # Combine the scale dimensions
        b = b.view(*leading_dims, W_s, H_s, scale * scale)
        return b

    def p2o(self, psf, shape):
        '''
        Convert point-spread function to optical transfer function.
        otf = p2o(psf) computes the Fast Fourier Transform (FFT) of the
        point-spread function (PSF) array and creates the optical transfer
        function (OTF) array that is not influenced by the PSF off-centering.
        Args:
            psf: NxCxhxw
            shape: [H, W]
        Returns:
            otf: NxCxHxWx2
        '''
        otf = torch.zeros(psf.shape[:-2] + shape).type_as(psf)
        otf[..., :psf.shape[-2], :psf.shape[-1]].copy_(psf)
        otf = torch.roll(otf, (-int(psf.shape[-2] / 2), -int(psf.shape[-1] / 2)), dims=(-2, -1))
        otf = torch.fft.fftn(otf, dim=(-2, -1))

        return otf

    def upsample(self, x, scale=3):
        '''s-fold upsampler
        Upsampling the spatial size by filling the new entries with zeros
        x: tensor image, NxCxWxH
        '''
        st = 0
        z = torch.zeros((x.shape[0], x.shape[1], x.shape[2] * scale, x.shape[3] * scale)).type_as(x)
        z[..., st::scale, st::scale].copy_(x)
        return z