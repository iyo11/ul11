import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange
from ultralytics.nn.modules import Conv

Conv2d = nn.Conv2d
__all__ = ["C2PSA_DHOGSA"]


# -------------------------
# reshape helpers
# -------------------------
def to_2d(x):
    return rearrange(x, "b c h w -> b (h w c)")


def to_3d(x):
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x, h, w):
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


# -------------------------
# LayerNorm variants
# -------------------------
class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5)


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5)


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type="WithBias"):
        super().__init__()
        if LayerNorm_type == "BiasFree":
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


# -------------------------
# shuffle helpers
# -------------------------
def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x


def inverse_channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, channels_per_group, groups, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x


# -------------------------
# small utility for FFN_DIFF
# -------------------------
class ElementScale(nn.Module):
    """A learnable element-wise scaler."""

    def __init__(self, embed_dims, init_value=0.0, requires_grad=True):
        super().__init__()
        self.scale = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)),
            requires_grad=requires_grad,
        )

    def forward(self, x):
        return x * self.scale


class FFN_DIFF(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.667, bias=False):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.sigma = ElementScale(hidden_features // 4, init_value=1e-5, requires_grad=True)
        self.decompose = nn.Conv2d(
            in_channels=hidden_features // 4,
            out_channels=1,
            kernel_size=1,
        )
        self.decompose_act = nn.GELU()
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv_5 = nn.Conv2d(
            hidden_features // 4,
            hidden_features // 4,
            kernel_size=5,
            stride=1,
            padding=2,
            groups=hidden_features // 4,
            bias=bias,
        )
        self.dwconv_dilated2_1 = nn.Conv2d(
            hidden_features // 4,
            hidden_features // 4,
            kernel_size=3,
            stride=1,
            padding=2,
            groups=hidden_features // 4,
            bias=bias,
            dilation=2,
        )
        self.p_unshuffle = nn.PixelUnshuffle(2)
        self.p_shuffle = nn.PixelShuffle(2)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def feat_decompose(self, x):
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


# -------------------------
# Gradient confidence gate (深层专用：更保守 + 归一化置信度)
# -------------------------
class GradConfidenceGate(nn.Module):
    """
    alpha = sigmoid(k*(conf - tau) + bias)
    深层建议:
      - tau 更小（深层梯度弱，不然永远不开）
      - bias 更负（训练初期更像 baseline，避免一开始就压小目标）
      - k 略大（让开关更“干脆”）
    """
    def __init__(self, init_tau=0.05, init_k=12.0, init_bias=-4.0):
        super().__init__()
        self.tau = nn.Parameter(torch.tensor(float(init_tau)))
        self.k = nn.Parameter(torch.tensor(float(init_k)))
        self.bias = nn.Parameter(torch.tensor(float(init_bias)))

    def forward(self, grad_conf: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.k * (grad_conf - self.tau) + self.bias)


# -------------------------
# TransformerBlock (保留)
# -------------------------
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, num_experts=3):
        super().__init__()
        self.attn_g_spatial = DHOGSA(dim, num_heads, bias)
        self.norm_g = LayerNorm(dim, LayerNorm_type)
        self.ffn = FFN_DIFF(dim, ffn_expansion_factor, bias)
        self.norm_ff1 = LayerNorm(dim, LayerNorm_type)

    def forward(self, x):
        f_spatial = self.attn_g_spatial(self.norm_g(x))
        x = x + f_spatial
        x_out = x + self.ffn(self.norm_ff1(x))
        return x_out


# -------------------------
# DHOGSA (深层版门控：top-k / global mean 归一化)
# -------------------------
class DHOGSA(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=4,
        bias=False,
        ifBox=True,
        patch_size=8,
        clip_limit=1.0,
        n_bins=9,
        # 深层门控超参（推荐默认）
        gate_topk_ratio=1 / 256,  # 深层更保守
        init_tau=0.05,
        init_k=12.0,
        init_bias=-4.0,
    ):
        super().__init__()
        self.factor = num_heads
        self.ifBox = ifBox
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = Conv2d(dim, dim * 5, kernel_size=1, bias=bias)
        self.qkv_dwconv = Conv2d(dim * 5, dim * 5, kernel_size=3, stride=1, padding=1, groups=dim * 5, bias=bias)
        self.project_out = Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.bin_proj = Conv2d(n_bins, dim // 2, kernel_size=1, bias=bias)
        self.patch_size = patch_size
        self.n_bins = n_bins

        # Sobel
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x.repeat(dim, 1, 1, 1))
        self.register_buffer("sobel_y", sobel_y.repeat(dim, 1, 1, 1))

        # gate
        self.gate = GradConfidenceGate(init_tau=init_tau, init_k=init_k, init_bias=init_bias)
        self.gate_topk_ratio = float(gate_topk_ratio)

    def pad(self, x, factor):
        hw = x.shape[-1]
        t_pad = [0, 0] if hw % factor == 0 else [0, (hw // factor + 1) * factor - hw]
        x = F.pad(x, t_pad, "constant", 0)
        return x, t_pad

    def unpad(self, x, t_pad):
        *_, hw = x.shape
        return x[:, :, t_pad[0] : hw - t_pad[1]]

    def softmax_1(self, x, dim=-1):
        logit = x.exp()
        logit = logit / (logit.sum(dim, keepdim=True) + 1)
        return logit

    def reshape_attn(self, q, k, v, ifBox):
        b, c = q.shape[:2]
        q, t_pad = self.pad(q, self.factor)
        k, t_pad = self.pad(k, self.factor)
        v, t_pad = self.pad(v, self.factor)

        hw = q.shape[-1] // self.factor
        shape_ori = "b (head c) (factor hw)" if ifBox else "b (head c) (hw factor)"
        shape_tar = "b head (c factor) hw"

        q = rearrange(q, f"{shape_ori} -> {shape_tar}", factor=self.factor, hw=hw, head=self.num_heads)
        k = rearrange(k, f"{shape_ori} -> {shape_tar}", factor=self.factor, hw=hw, head=self.num_heads)
        v = rearrange(v, f"{shape_ori} -> {shape_tar}", factor=self.factor, hw=hw, head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = self.softmax_1(attn, dim=-1)
        out = attn @ v

        out = rearrange(out, f"{shape_tar} -> {shape_ori}", factor=self.factor, hw=hw, b=b, head=self.num_heads)
        out = self.unpad(out, t_pad)
        return out

    def split_into_patches(self, x):
        b, c, h, w = x.shape
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        patches = rearrange(
            x, "b c (h p1) (w p2) -> b (h w) c (p1 p2)", p1=self.patch_size, p2=self.patch_size
        )
        n_h, n_w = (h + pad_h) // self.patch_size, (w + pad_w) // self.patch_size
        return patches, (b, c, h, w, pad_h, pad_w, n_h, n_w)

    def merge_patches(self, patches, shape_info):
        b, c, h, w, pad_h, pad_w, n_h, n_w = shape_info
        patches = rearrange(
            patches, "b (h w) c (p1 p2) -> b c (h p1) (w p2)", h=n_h, w=n_w, p1=self.patch_size, p2=self.patch_size
        )
        if pad_h > 0 or pad_w > 0:
            patches = patches[:, :, :h, :w]
        return patches

    def apply_hog_to_patch(self, x_half):
        b, c, h, w = x_half.shape

        gx = F.conv2d(x_half, self.sobel_x[:c], padding=1, groups=c)
        gy = F.conv2d(x_half, self.sobel_y[:c], padding=1, groups=c)
        magnitude = torch.sqrt(gx**2 + gy**2 + 1e-6)
        orientation = torch.atan2(gy, gx)  # [-pi, pi]
        orientation_bin = ((orientation + torch.pi) / (2 * torch.pi) * self.n_bins).long() % self.n_bins

        patches_x, shape_info = self.split_into_patches(x_half)
        patches_mag, _ = self.split_into_patches(magnitude)
        patches_ori, _ = self.split_into_patches(orientation_bin.float())

        b, n_patches, c, patch_pixels = patches_x.shape
        sort_values = torch.zeros_like(patches_x)
        hog_features = torch.zeros(b, n_patches, self.n_bins, device=x_half.device)

        for i in range(self.n_bins):
            bin_mask = (patches_ori == i).float()
            bin_magnitude = patches_mag * bin_mask
            sort_values += bin_magnitude * (i + 1)
            hog_features[..., i] = bin_magnitude.mean(dim=[-1, -2])

        hog_features = hog_features / (hog_features.sum(dim=-1, keepdim=True) + 1e-8)

        _, sort_indices = sort_values.sum(dim=2, keepdim=True).expand_as(patches_x).sort(dim=-1)
        patches_x_sorted = torch.gather(patches_x, -1, sort_indices)

        x_half_processed = self.merge_patches(patches_x_sorted, shape_info)
        return x_half_processed, sort_indices, hog_features, shape_info

    @torch.no_grad()
    def _safe_topk_k(self, n_pix: int) -> int:
        k = int(max(1, round(n_pix * self.gate_topk_ratio)))
        return max(1, min(k, n_pix))

    def forward(self, x):
        b, c, h, w = x.shape
        half_c = c // 2

        # --- (A) HOG patch sorting + hog_map injection on first half channels ---
        x_half = x[:, :half_c]
        x_half_processed, idx_patch, hog_features, shape_info = self.apply_hog_to_patch(x_half)

        b2, n_patches, n_bins = hog_features.shape
        n_h = shape_info[-2]
        n_w = shape_info[-1]
        hog_map = rearrange(hog_features, "b (nh nw) bins -> b bins nh nw", nh=n_h, nw=n_w).contiguous()
        hog_map = self.bin_proj(hog_map)
        hog_map = F.interpolate(hog_map, size=(h, w), mode="bilinear", align_corners=False)

        x = torch.cat((x_half_processed + hog_map, x[:, half_c:]), dim=1)

        # --- (B) QKV (5-way) ---
        qkv = self.qkv_dwconv(self.qkv(x))
        q1, k1, q2, k2, v = qkv.chunk(5, dim=1)

        # --- (C) compute gradient magnitude/orientation on v ---
        gx = F.conv2d(v, self.sobel_x[:c], padding=1, groups=c)
        gy = F.conv2d(v, self.sobel_y[:c], padding=1, groups=c)
        magnitude = torch.sqrt(gx**2 + gy**2 + 1e-6)  # (B,C,H,W)

        # ===== 深层门控：top-k / global mean 归一化置信度 =====
        g = magnitude.mean(dim=1, keepdim=True)  # (B,1,H,W)

        n_pix = h * w
        k_top = self._safe_topk_k(n_pix)
        topk_mean = torch.topk(g.flatten(1), k=k_top, dim=1).values.mean(dim=1, keepdim=True)  # (B,1)
        global_mean = g.flatten(1).mean(dim=1, keepdim=True)  # (B,1)

        grad_conf = (topk_mean / (global_mean + 1e-6)).view(b, 1, 1, 1)  # 相对置信度
        alpha = self.gate(grad_conf)  # (B,1,1,1)

        # --- (D) gradient direction-weighted sorting (original logic) ---
        magnitude_flat = magnitude.view(b, c, -1)
        orientation = torch.atan2(gy, gx).view(b, c, -1)  # [-pi, pi]
        orientation_norm = (orientation + torch.pi) / (2 * torch.pi)

        weighted_magnitude = magnitude_flat * orientation_norm
        _, idx = weighted_magnitude.sum(dim=1).sort(dim=-1)
        idx = idx.unsqueeze(1).expand(b, c, -1)

        v = torch.gather(v.view(b, c, -1), dim=2, index=idx)
        q1 = torch.gather(q1.view(b, c, -1), dim=2, index=idx)
        k1 = torch.gather(k1.view(b, c, -1), dim=2, index=idx)
        q2 = torch.gather(q2.view(b, c, -1), dim=2, index=idx)
        k2 = torch.gather(k2.view(b, c, -1), dim=2, index=idx)

        out1 = self.reshape_attn(q1, k1, v, True)
        out2 = self.reshape_attn(q2, k2, v, False)

        out1 = torch.scatter(out1, 2, idx, out1).view(b, c, h, w)
        out2 = torch.scatter(out2, 2, idx, out2).view(b, c, h, w)

        out = out1 * out2
        out = self.project_out(out)

        # --- (E) inverse patch restore on first half channels ---
        out_replace = out[:, :half_c]
        patches_out, shape_info2 = self.split_into_patches(out_replace)
        patches_out = torch.scatter(patches_out, -1, idx_patch, patches_out)
        out_replace = self.merge_patches(patches_out, shape_info2)
        out[:, :half_c] = out_replace

        # ✅ 门控：只缩放“增量输出”，不要混回 x（外面 PSABlock 会 residual add）
        out = out * alpha
        return out


# -------------------------
# PSA block wrapper
# -------------------------
class PSABlock_DHOGSA(nn.Module):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        super().__init__()
        self.attn = DHOGSA(c)  # 深层门控默认已内置
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


# -------------------------
# C2PSA_DHOGSA
# -------------------------
class C2PSA_DHOGSA(nn.Module):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        # 防止 self.c//64 == 0
        heads = max(1, self.c // 64)
        self.m = nn.Sequential(*(PSABlock_DHOGSA(self.c, attn_ratio=0.5, num_heads=heads) for _ in range(n)))

    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))
