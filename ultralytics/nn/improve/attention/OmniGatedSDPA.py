import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

__all__ = ['OmniGatedSDPA', 'MetaOmniBlock', 'MetaOmniBlock_DIFF']

import torch
import torch.nn as nn
from einops import rearrange
from timm.layers import DropPath  # 推荐使用 timm 的 DropPath，训练更稳定

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


class Mlp(nn.Module):
    """
    适用于 (B, C, H, W) 输入的 Channel Mixer
    使用 1x1 卷积代替 Linear，避免频繁的 permute/reshape
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MetaOmniBlock(nn.Module):
    def __init__(self, dim, n_heads=8, reduction_ratio=1, mlp_ratio=4.,
                 drop=0., drop_path=0., act_layer=nn.GELU):
        super().__init__()

        # 1. Norm layer (通常 MetaFormer 使用 GroupNorm 或 LayerNorm)
        # 这里使用 LayerNorm，需要处理一下维度，或者使用 timm 的 LayerNorm2d
        self.norm1 = nn.GroupNorm(1, dim)  # GroupNorm(1) 等价于 LayerNorm 但支持 (B, C, H, W)

        # 2. Token Mixer (你的 OmniGatedSDPA)
        self.token_mixer = OmniGatedSDPA(dim, n_heads=n_heads, reduction_ratio=reduction_ratio)

        # 3. Norm layer for MLP
        self.norm2 = nn.GroupNorm(1, dim)

        # 4. Channel Mixer (MLP)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # 5. DropPath (Stochastic Depth)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        # --- Token Mixing 部分 ---
        # x = x + drop_path(token_mixer(norm1(x)))
        shortcut = x
        x = self.norm1(x)

        # 注意：你的 OmniGatedSDPA 内部已经有 rearrange 和 LayerNorm
        # 为了配合标准 Block，建议将 OmniGatedSDPA 内部的 self.ln 去掉，
        # 或者在这里跳过外部的 norm1。这里展示标准结构：
        x = self.token_mixer(x)
        x = shortcut + self.drop_path(x)

        # --- Channel Mixing 部分 ---
        # x = x + drop_path(mlp(norm2(x)))
        shortcut = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = shortcut + self.drop_path(x)

        return x

class MetaOmniBlock_DIFF(nn.Module):
    def __init__(self, dim, n_heads=8, reduction_ratio=1, mlp_ratio=4.,
                 drop=0., drop_path=0., act_layer=nn.GELU):
        super().__init__()

        # 1. Norm layer (通常 MetaFormer 使用 GroupNorm 或 LayerNorm)
        # 这里使用 LayerNorm，需要处理一下维度，或者使用 timm 的 LayerNorm2d
        self.norm1 = nn.GroupNorm(1, dim)  # GroupNorm(1) 等价于 LayerNorm 但支持 (B, C, H, W)

        # 2. Token Mixer (你的 OmniGatedSDPA)
        self.token_mixer = OmniGatedSDPA(dim, n_heads=n_heads, reduction_ratio=reduction_ratio)

        # 3. Norm layer for MLP
        self.norm2 = nn.GroupNorm(1, dim)

        # 4. Channel Mixer (MLP)
        mlp_hidden_dim = int(dim * mlp_ratio)
        #self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp = FFN_DIFF(dim)

        # 5. DropPath (Stochastic Depth)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        # --- Token Mixing 部分 ---
        # x = x + drop_path(token_mixer(norm1(x)))
        shortcut = x
        x = self.norm1(x)

        # 注意：你的 OmniGatedSDPA 内部已经有 rearrange 和 LayerNorm
        # 为了配合标准 Block，建议将 OmniGatedSDPA 内部的 self.ln 去掉，
        # 或者在这里跳过外部的 norm1。这里展示标准结构：
        x = self.token_mixer(x)
        x = shortcut + self.drop_path(x)

        # --- Channel Mixing 部分 ---
        # x = x + drop_path(mlp(norm2(x)))
        shortcut = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = shortcut + self.drop_path(x)

        return x

class OmniGatedSDPA(nn.Module):
    def __init__(self, d_model, n_heads=8, reduction_ratio=1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.reduction_ratio = reduction_ratio

        # Q, K, V 映射
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

        self.gate_proj = nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, groups=d_model, bias=True)

        self.w_o = nn.Linear(d_model, d_model, bias=False)

        if reduction_ratio > 1:
            self.sr = nn.Conv2d(d_model, d_model, kernel_size=reduction_ratio, stride=reduction_ratio)
            self.sr_ln = nn.LayerNorm(d_model)
        else:
            self.sr = nn.Identity()
            self.sr_ln = nn.Identity()

        self.ln = nn.LayerNorm(d_model)
        self._init_weights()

    def _init_weights(self):
        if self.gate_proj.bias is not None:
            nn.init.constant_(self.gate_proj.bias, -1.0)
    def forward(self, x):
        b, c, h, w = x.shape
        x_flat = rearrange(x, 'b c h w -> b (h w) c')
        x_norm = self.ln(x_flat)
        q = self.w_q(x_norm).view(b, -1, self.n_heads, self.d_head).transpose(1, 2)
        if self.reduction_ratio > 1:
            x_sr = self.sr(x)
            x_sr = rearrange(x_sr, 'b c h w -> b (h w) c')
            x_sr = self.sr_ln(x_sr)
            k = self.w_k(x_sr).view(b, -1, self.n_heads, self.d_head).transpose(1, 2)
            v = self.w_v(x_sr).view(b, -1, self.n_heads, self.d_head).transpose(1, 2)
        else:
            k = self.w_k(x_norm).view(b, -1, self.n_heads, self.d_head).transpose(1, 2)
            v = self.w_v(x_norm).view(b, -1, self.n_heads, self.d_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v)
        gate_input = rearrange(x_norm, 'b (h w) c -> b c h w', h=h, w=w)
        gate = self.gate_proj(gate_input)
        gate = rearrange(gate, 'b c h w -> b (h w) c')
        gate = gate.view(b, -1, self.n_heads, self.d_head).transpose(1, 2)
        y = y * torch.sigmoid(gate)
        y = y.transpose(1, 2).contiguous().view(b, -1, self.d_model)
        out = self.w_o(y)
        return rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)


# class OmniGatedSDPAForDetection(nn.Module):
#     def __init__(self, d_model, n_heads=8, reduction_ratio=1):
#         super().__init__()
#         self.d_model = d_model
#         self.n_heads = n_heads
#         self.d_head = d_model // n_heads
#         self.reduction_ratio = reduction_ratio
#
#         # Q, K, V 映射
#         self.w_q = nn.Linear(d_model, d_model, bias=False)
#         self.w_k = nn.Linear(d_model, d_model, bias=False)
#         self.w_v = nn.Linear(d_model, d_model, bias=False)
#
#         # 针对检测优化：Conv 提取局部上下文 + GN 保证小 Batch 训练稳定
#         self.gate_proj = nn.Sequential(
#             nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, groups=d_model, bias=True),
#             nn.GroupNorm(1, d_model)
#         )
#
#         self.w_o = nn.Linear(d_model, d_model, bias=False)
#
#         # 空间压缩：如果用于检测器的深层，可以设置 reduction_ratio > 1 节省显存
#         if reduction_ratio > 1:
#             self.sr = nn.Conv2d(d_model, d_model, kernel_size=reduction_ratio, stride=reduction_ratio)
#             self.sr_ln = nn.LayerNorm(d_model)
#         else:
#             self.sr = nn.Identity()
#             self.sr_ln = nn.Identity()
#
#         self.ln = nn.LayerNorm(d_model)
#         self._init_weights()
#
#     def _init_weights(self):
#         # 核心 Trick：负偏置初始化，让模型初始阶段倾向于“只看全局，不加门控”
#         if isinstance(self.gate_proj[0], nn.Conv2d):
#             nn.init.constant_(self.gate_proj[0].bias, -1.0)
#
#     def forward(self, x):
#         b, c, h, w = x.shape
#
#         # 1. 预归一化
#         x_flat = rearrange(x, 'b c h w -> b (h w) c')
#         x_norm = self.ln(x_flat)
#
#         # 2. 生成 QKV
#         q = rearrange(self.w_q(x_norm), 'b l (h d) -> b h l d', h=self.n_heads)
#
#         if self.reduction_ratio > 1:
#             x_sr = self.sr(x)
#             x_sr = rearrange(x_sr, 'b c h w -> b (h w) c')
#             k = rearrange(self.w_k(self.sr_ln(x_sr)), 'b l (h d) -> b h l d', h=self.n_heads)
#             v = rearrange(self.w_v(self.sr_ln(x_sr)), 'b l (h d) -> b h l d', h=self.n_heads)
#         else:
#             k = rearrange(self.w_k(x_norm), 'b l (h d) -> b h l d', h=self.n_heads)
#             v = rearrange(self.w_v(x_norm), 'b l (h d) -> b h l d', h=self.n_heads)
#
#         # 3. Flash Attention
#         y = F.scaled_dot_product_attention(q, k, v)
#
#         # 4. 门控路径：使用 x_norm 还原后的 4D Tensor，增强对检测目标的空间感知
#         gate_input = rearrange(x_norm, 'b (h w) c -> b c h w', h=h, w=w)
#         gate = self.gate_proj(gate_input)
#         gate = rearrange(gate, 'b (h d) h_w w_w -> b h (h_w w_w) d', h=self.n_heads, d=self.d_head)
#
#         # 5. 调制与输出
#         y = y * torch.sigmoid(gate)
#         y = rearrange(y, 'b h (h w) d -> b (h w) (h d)', h=h, w=w)
#         out = self.w_o(y)
#
#         return rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)