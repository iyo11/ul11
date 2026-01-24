import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

__all__ = ['OmniGatedSDPA']


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