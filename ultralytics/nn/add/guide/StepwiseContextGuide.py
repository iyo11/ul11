import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['StepwiseContextGuide']

class StepwiseContextGuide(nn.Module):
    def __init__(self, c1, c2, r=4):
        super().__init__()
        # 解析输入 c1: [c_local, c_guide]
        if isinstance(c1, (list, tuple)):
            c_local, c_guide = c1
        else:
            c_local, c_guide = c1, c1

        # 1. 对齐通道 (Align) - 轻量化
        # 如果 guide 通道很大，先用 1x1 降维，不仅能对齐，还能减少后续 grid_sample 的计算量
        # 我们不再强制 align 输出等于 c_local，而是等于 c2 (目标输出)，这样能大幅省参数
        self.align = nn.Sequential(
            nn.Conv2d(c_guide, c2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c2)
        )

        # 2. 采样点生成器 - 保持不变 (本身很轻量)
        self.offset_gen = nn.Sequential(
            nn.Conv2d(c_local, c_local // 4, 3, 1, 1, groups=c_local // 4, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_local // 4, 2, 1, 1, 0, bias=True)
        )

        # 3. MLP 分支 - 加入 Bottleneck (瓶颈结构)
        # 原始代码是 c_local -> c_local -> c_local，这里改为 c_local -> c_mid -> c2
        # 既融合了特征，又将维度统一到了输出维度 c2
        mid_c = c2 // 2  # 设为输出的一半作为瓶颈
        self.mlp = nn.Sequential(
            nn.Conv2d(c_local, mid_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_c),
            nn.GELU(),
            nn.Conv2d(mid_c, c2, 1, 1, 0, bias=False) # 最终映射到 c2
        )

        # 4. 最终融合
        # 此时 align 分支输出是 c2, mlp 分支输出也是 c2，直接相加即可，不需要额外的 Final Conv
        # 这一步省掉了一个巨大的卷积层
        self.fusion_act = nn.SiLU(inplace=True)

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            x_local, x_guide = x
        else:
            x_local, x_guide = x, x

        B, C, H, W = x_local.shape

        # --- Upsample Branch (Aligned) ---
        # 先对齐 guide 的通道到目标维度 c2
        g = self.align(x_guide)

        # 生成偏移量
        offset = self.offset_gen(x_local)
        grid = self._get_grid(B, H, W, x_local.device)
        v_grid = grid + offset.permute(0, 2, 3, 1) * (2.0 / max(H, W))

        # Grid Sample (隐式上采样)
        # g 的尺寸小，v_grid 尺寸大，grid_sample 会自动插值
        g_aligned = F.grid_sample(g, v_grid, mode='bilinear', padding_mode='zeros', align_corners=False)

        # --- MLP Branch ---
        l_feat = self.mlp(x_local)

        # --- Fusion ---
        # 直接相加 (Element-wise Add)
        out = self.fusion_act(g_aligned + l_feat)

        return out

    def _get_grid(self, B, H, W, device):
        y, x = torch.meshgrid(torch.linspace(-1, 1, H, device=device),
                              torch.linspace(-1, 1, W, device=device), indexing='ij')
        grid = torch.stack((x, y), dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)
        return grid