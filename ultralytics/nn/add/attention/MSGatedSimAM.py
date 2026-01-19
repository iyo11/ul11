import torch
import torch.nn as nn
import torch.nn.functional as F


class MSGatedSimAM(nn.Module):
    """
    Multi-Scale Gated SimAM for Tiny Object Detection.
    Arguments:
        c1 (int): Input channels
        c2 (int): Output channels (should be same as c1)
        window_size (int): Size of the local window (default 8)
        e_lambda (float): Regularization term for SimAM (default 1e-4)
    """

    def __init__(self, c1, c2, window_size=8, e_lambda=1e-4):
        super().__init__()
        # Attention 模块通常不改变通道数，但为了兼容 YOLO 格式，我们接收 c2
        # 如果 c1 != c2，通常意味着这是个设计错误，或者你需要在这里加个卷积层调整
        assert c1 == c2, f"MSGatedSimAM expects c1==c2, but got c1={c1}, c2={c2}"
        self.c1 = c1
        self.ws = window_size
        self.e_lambda = e_lambda
        self.activaton = nn.Sigmoid()

        # --- 门控机制 (The Gate) ---
        # 输入: 2个通道 (Global 能量图 + Local 能量图) -> 输出: 1个通道 (权重图)
        # 这是一个极其轻量的空间门控，与输入通道数 c1 无关
        self.gate_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=1, stride=1, bias=True),
            nn.Sigmoid()
        )

    def _get_energy(self, x_in, dim_mean, n):
        """SimAM 能量函数计算"""
        mu = x_in.mean(dim=dim_mean, keepdim=True)
        var = (x_in - mu).pow(2).sum(dim=dim_mean, keepdim=True) / n
        x_minus_mu_square = (x_in - mu).pow(2)
        y = x_minus_mu_square / (4 * (var + self.e_lambda)) + 0.5
        return y

    def forward(self, x):
        B, C, H, W = x.size()

        # --- 1. Global Branch ---
        n_global = H * W - 1
        energy_global = self._get_energy(x, dim_mean=[2, 3], n=n_global)

        # --- 2. Local Branch ---
        # Padding
        pad_h = (self.ws - H % self.ws) % self.ws
        pad_w = (self.ws - W % self.ws) % self.ws
        x_padded = F.pad(x, (0, pad_w, 0, pad_h))
        Hp, Wp = x_padded.shape[2], x_padded.shape[3]

        # Reshape to windows
        x_windows = x_padded.view(B, C, Hp // self.ws, self.ws, Wp // self.ws, self.ws)
        x_windows = x_windows.permute(0, 1, 2, 4, 3, 5)  # (B, C, h_n, w_n, ws, ws)

        # Local Energy
        n_local = self.ws * self.ws - 1
        energy_local_windows = self._get_energy(x_windows, dim_mean=[4, 5], n=n_local)

        # Reshape back
        energy_local = energy_local_windows.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, C, Hp, Wp)
        if pad_h > 0 or pad_w > 0:
            energy_local = energy_local[:, :, :H, :W]

        # --- 3. Gated Fusion ---
        # 压缩通道以计算空间 Gate 权重
        map_global = energy_global.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        map_local = energy_local.mean(dim=1, keepdim=True)  # (B, 1, H, W)

        # 计算 Alpha (B, 1, H, W)
        alpha = self.gate_conv(torch.cat([map_local, map_global], dim=1))

        # 融合
        energy_fused = alpha * energy_local + (1 - alpha) * energy_global

        return x * self.activaton(energy_fused)