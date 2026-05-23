import torch
import torch.nn as nn

try:
    from ultralytics.nn.modules.conv import Conv
except Exception:
    from ultralytics.nn.modules import Conv


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for BCHW."""

    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(c, eps=eps)

    def forward(self, x):
        # BCHW -> BHWC -> BCHW
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class BinaryAttention(nn.Module):
    """
    FP16/AMP-safe Binary Attention.
    Input : [B, C, H, W]
    Output: [B, C, H, W]
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        binary=False,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = max(1, min(num_heads, dim))

        while dim % self.num_heads != 0 and self.num_heads > 1:
            self.num_heads -= 1

        self.head_dim = dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.binary = binary

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W

        dtype = x.dtype

        # BCHW -> BNC
        x_flat = x.flatten(2).transpose(1, 2).contiguous()

        qkv = self.qkv(x_flat)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()

        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        if self.binary:
            # 二值化 attention，但是保持 dtype，不要 .float()
            attn_bin = (attn > attn.mean(dim=-1, keepdim=True)).to(dtype=attn.dtype)
            attn = attn + (attn_bin - attn).detach()

        attn = self.attn_drop(attn)

        # 关键修复：AMP/half 下 attn 和 v 必须同 dtype
        if attn.dtype != v.dtype:
            attn = attn.to(dtype=v.dtype)

        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C).contiguous()

        x = self.proj(x)
        x = self.proj_drop(x)

        # BNC -> BCHW
        x = x.transpose(1, 2).reshape(B, C, H, W).contiguous()

        if x.dtype != dtype:
            x = x.to(dtype=dtype)

        return x


class PSABlock_BinaryAttention(nn.Module):
    """
    C2PSA 内部 block。
    """

    def __init__(
        self,
        c,
        attn_ratio=0.5,
        num_heads=None,
        shortcut=True,
        binary=False,
    ):
        super().__init__()
        self.shortcut = shortcut

        if num_heads is None:
            num_heads = max(1, c // 64)

        self.norm1 = LayerNorm2d(c)
        self.attn = BinaryAttention(
            dim=c,
            num_heads=num_heads,
            qkv_bias=True,
            attn_drop=0.0,
            proj_drop=0.0,
            binary=binary,
        )

        self.norm2 = LayerNorm2d(c)

        hidden = int(c * 2)
        self.ffn = nn.Sequential(
            Conv(c, hidden, 1, 1),
            Conv(hidden, c, 1, 1, act=False),
        )

    def forward(self, x):
        if self.shortcut:
            x = x + self.attn(self.norm1(x))
            x = x + self.ffn(self.norm2(x))
        else:
            x = self.attn(self.norm1(x))
            x = self.ffn(self.norm2(x))
        return x


class C2PSA_BinaryAttention(nn.Module):
    """
    Ultralytics 可用版 C2PSA_BinaryAttention。

    YAML 用法示例：
      - [-1, 1, C2PSA_BinaryAttention, [1024]]
      - [-1, 2, C2PSA_BinaryAttention, [1024]]
    """

    def __init__(
        self,
        c1,
        c2,
        n=1,
        e=0.5,
        shortcut=True,
        binary=False,
    ):
        super().__init__()

        assert c1 == c2, "C2PSA_BinaryAttention requires c1 == c2"

        self.c = int(c1 * e)

        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1, 1)

        self.m = nn.Sequential(
            *[
                PSABlock_BinaryAttention(
                    self.c,
                    shortcut=shortcut,
                    binary=binary,
                )
                for _ in range(n)
            ]
        )

    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)

        # 防止极端情况下 dtype 不一致
        if a.dtype != b.dtype:
            b = b.to(dtype=a.dtype)

        return self.cv2(torch.cat((a, b), dim=1))