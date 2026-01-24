from typing import Tuple, Optional, Dict
import torch.nn.functional as F
import torch
import torch.nn as nn
import math

__all__ = ['ESMoE']


def get_safe_groups(channels: int, desired_groups: int = 8) -> int:
    """Ensure num_groups divides channels"""
    groups = min(desired_groups, channels)
    while channels % groups != 0:
        groups -= 1
    return max(1, groups)


# ==========================================
# Optimized expert modules
# ==========================================
class OptimizedSimpleExpert(nn.Module):
    """Use GroupNorm instead of BatchNorm to improve stability for small batches."""

    def __init__(self, in_channels, out_channels, expand_ratio=2, num_groups=8):
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
            nn.GroupNorm(get_safe_groups(hidden_dim, num_groups), hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.GroupNorm(get_safe_groups(out_channels, num_groups), out_channels)
        )
        self.hidden_dim = hidden_dim

    def forward(self, x):
        return self.conv(x)


class FusedGhostExpert(nn.Module):
    """Fused Ghost expert that reduces memory traffic by combining operations."""

    def __init__(self, in_channels, out_channels, kernel_size=3, ratio=2, num_groups=8):
        super().__init__()
        self.out_channels = out_channels
        init_channels = math.ceil(out_channels / ratio)
        new_channels = init_channels * (ratio - 1)

        # Use GroupNorm to improve stability
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size, padding=kernel_size // 2, bias=False),
            nn.GroupNorm(min(num_groups, init_channels), init_channels),
            nn.SiLU(inplace=True)
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, 3, padding=1, groups=init_channels, bias=False),
            nn.GroupNorm(min(num_groups, new_channels), new_channels),
            nn.SiLU(inplace=True)
        )
        self.init_channels = init_channels

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.out_channels, :, :]


class SimpleExpert(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio=2):
        super().__init__()
        hidden_dim = int(in_channels * expand_ratio)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x): return self.conv(x)


class GhostExpert(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, ratio=2):
        super().__init__()
        self.out_channels = out_channels
        init_channels = math.ceil(out_channels / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.SiLU(inplace=True)
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, 3, padding=1, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        return torch.cat([x1, x2], dim=1)[:, :self.out_channels, :, :]


class InvertedResidualExpert(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, expand_ratio=2):
        super().__init__()
        hidden_dim = int(in_channels * expand_ratio)
        self.use_expand = expand_ratio != 1
        layers = []
        if self.use_expand:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(inplace=True)
            ])
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=(kernel_size - 1) // 2, groups=hidden_dim,
                      bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x): return self.conv(x)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        padding = (kernel_size - 1) // 2
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class EfficientExpertGroup(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(EfficientExpertGroup, self).__init__()
        self.conv = DepthwiseSeparableConv(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        if not hasattr(self, "conv"):
            out_c = x.shape[1]
            self.conv = DepthwiseSeparableConv(x.shape[1], out_c, 3, 1)
        return self.conv(x)


class MoELoss(nn.Module):
    """
    Computes auxiliary losses for Mixture of Experts (MoE) models,
    specifically Load Balancing Loss and Z-Loss.
    """

    def __init__(self, balance_loss_coeff=0.01, z_loss_coeff=1e-3, num_experts=4, top_k=2):
        super().__init__()
        self.balance_loss_coeff = balance_loss_coeff
        self.z_loss_coeff = z_loss_coeff
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(self, router_probs, router_logits, expert_indices):
        """
        Args:
            router_probs (torch.Tensor): Probability distribution predicted by router [B, E]
            router_logits (torch.Tensor): Logits from router [B, E]
            expert_indices (torch.Tensor): Selected expert indices [B, k]
        Returns:
            torch.Tensor: Combined auxiliary loss
        """
        # 1) Load Balancing Loss
        # importance: probability distribution predicted by router (differentiable)
        importance = router_probs.mean(dim=0)

        # usage: which experts were actually selected (non-differentiable, detached)
        usage_mask = torch.zeros_like(router_probs)
        # B, E = router_probs.shape
        # usage_mask = torch.zeros(B, E, dtype=router_probs.dtype, device=router_probs.device)
        for k in range(self.top_k):
            usage_mask.scatter_(1, expert_indices[:, k].unsqueeze(1), 1.0)
        usage = usage_mask.mean(dim=0)

        balance_loss = self.num_experts * torch.sum(importance * usage.detach())

        # 2) Z-Loss (numerical stability)
        # Penalize square of log(sum(exp(logits))) to prevent logits from exploding.
        z_loss = torch.mean(torch.logsumexp(router_logits, dim=1) ** 2)

        return (self.balance_loss_coeff * balance_loss) + (self.z_loss_coeff * z_loss)


def get_safe_groups(channels: int, desired_groups: int = 8) -> int:
    """Ensure num_groups divides channels"""
    groups = min(desired_groups, channels)
    while channels % groups != 0:
        groups -= 1
    return max(1, groups)


# ==========================================
# Ultra-lightweight Router (core optimization)
# ==========================================
class UltraEfficientRouter(nn.Module):
    """
    Ultra-efficient router:
    1) Depthwise-separable convolution instead of standard conv
    2) Aggressive downsampling (8x)
    3) Early channel compression
    4) Improved numerical stability
    Expected FLOPs reduction: ~95% vs a local router baseline.
    """

    def __init__(self, in_channels, num_experts, reduction=16, top_k=2,
                 noise_std=1.0, temperature: float = 1.0, pool_scale=8):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std
        self.temperature = max(float(temperature), 1e-3)
        self.pool_scale = pool_scale

        # More aggressive channel compression
        reduced_channels = max(in_channels // reduction, 4)

        # Depthwise-separable conv: compute ~ 1/(kernel_size^2) of standard conv
        self.router = nn.Sequential(
            # Depthwise
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
            nn.GroupNorm(get_safe_groups(in_channels, 8), in_channels),
            nn.SiLU(inplace=True),
            # Pointwise compression
            nn.Conv2d(in_channels, reduced_channels, 1, bias=False),
            nn.GroupNorm(get_safe_groups(reduced_channels, 4), reduced_channels),
            nn.SiLU(inplace=True),
            # Expert projection
            nn.Conv2d(reduced_channels, num_experts, 1, bias=True)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x) -> Tuple[
        torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        B, C, H, W = x.shape

        # 1) Aggressive downsampling (core optimization)
        if H > self.pool_scale and W > self.pool_scale:
            x_down = F.avg_pool2d(x, kernel_size=self.pool_scale, stride=self.pool_scale)
        else:
            x_down = x

        # 2) Lightweight convolutional routing
        logits = self.router(x_down)

        # 3) Z-loss computation (numerical stability)
        z_loss_metric = None
        if self.training:
            # Use clamp instead of tanh for better performance
            logits_safe = logits.clamp(-10.0, 10.0)
            z_loss_metric = torch.logsumexp(logits_safe, dim=1).pow(2).mean()

        # 4) Noise injection
        if self.training and self.noise_std > 0:
            logits.add_(torch.randn_like(logits).mul_(self.noise_std))

        # 5) Softmax + TopK (fused operation)
        weights = self.softmax(logits / self.temperature)
        pooled_weights = weights.mean(dim=[2, 3], keepdim=True)

        topk_vals, topk_indices = torch.topk(pooled_weights, self.top_k, dim=1)

        # In-place normalization
        topk_vals.div_(topk_vals.sum(dim=1, keepdim=True).add_(1e-9))

        if self.training:
            importance = pooled_weights.sum(dim=0).view(self.num_experts)

            # Optimization: use one_hot instead of scatter
            topk_indices_flat = topk_indices.view(B, self.top_k, 1, 1)[:, :, 0, 0]
            mask = F.one_hot(topk_indices_flat, num_classes=self.num_experts).float()
            usage_frequency = mask.sum(dim=[0, 1]) / (B * self.top_k)

            return topk_vals, topk_indices, usage_frequency, importance, z_loss_metric
        else:
            return topk_vals, topk_indices, None, None, None


class BaseRouter(nn.Module):
    def __init__(self, num_experts, top_k):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.softmax = nn.Softmax(dim=1)

    def _process_logits(self, logits: torch.Tensor, noise_std: float, training: bool) -> Tuple[
        torch.Tensor, torch.Tensor, Dict]:
        """Unified logic to process logits into Top-K selection."""
        B = logits.shape[0]

        # 1) Add noise during training (simplified Gumbel-Softmax trick)
        if training and noise_std > 0:
            logits = logits + torch.randn_like(logits) * noise_std

        # 2) Compute probabilities
        probs = self.softmax(logits)

        # 3) Select Top-K
        topk_vals, topk_indices = torch.topk(probs, self.top_k, dim=1)

        # 4) Normalize weights
        sum_vals = topk_vals.sum(dim=1, keepdim=True) + 1e-6
        topk_vals = topk_vals / sum_vals

        # 5) Collect loss-related info (train only)
        loss_dict = {}
        if training:
            loss_dict['router_logits'] = logits
            loss_dict['router_probs'] = probs
            loss_dict['topk_indices'] = topk_indices

        return topk_vals, topk_indices, loss_dict


class EfficientSpatialRouter(BaseRouter):
    def __init__(self, in_channels, num_experts, reduction=8, top_k=2, noise_std=1.0, pool_scale=4):
        super().__init__(num_experts, top_k)
        self.noise_std = noise_std
        self.pool_scale = pool_scale
        reduced_channels = max(in_channels // reduction, 8)

        self.router = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(reduced_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(reduced_channels, num_experts, 1, bias=False),
            nn.BatchNorm2d(num_experts)  # numerical stability
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # Pre-pooling optimization
        if H > self.pool_scale and W > self.pool_scale:
            x_in = F.avg_pool2d(x, kernel_size=self.pool_scale, stride=self.pool_scale)
        else:
            x_in = x

        out = self.router(x_in)  # [B, E, H', W']
        global_logits = torch.mean(out, dim=[2, 3])  # [B, E]

        return self._process_logits(global_logits, self.noise_std, self.training)


class AdaptiveRoutingLayer(BaseRouter):
    def __init__(self, in_channels, num_experts, reduction=8, top_k=2, noise_std=1.0):
        super().__init__(num_experts, top_k)
        self.noise_std = noise_std
        reduced_channels = max(in_channels // reduction, 8)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.router = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, 1, bias=False),
            nn.BatchNorm2d(reduced_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(reduced_channels, num_experts, 1, bias=False),
            nn.BatchNorm2d(num_experts)
        )

    def forward(self, x):
        pooled = self.avg_pool(x)
        logits = self.router(pooled).squeeze(-1).squeeze(-1)  # [B, E]
        return self._process_logits(logits, self.noise_std, self.training)


class LocalRoutingLayer(BaseRouter):
    def __init__(self, in_channels, num_experts, reduction=8, top_k=2, noise_std=1.0):
        super().__init__(num_experts, top_k)
        self.noise_std = noise_std
        # Even for local routing, default to 2x downsampling to save FLOPs with minimal texture loss
        self.pool_scale = 2

        reduced_channels = max(in_channels // reduction, 8)
        self.router = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(reduced_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(reduced_channels, num_experts, 1, bias=False),
            nn.BatchNorm2d(num_experts)
        )

    def forward(self, x):
        # Moderate downsampling to accelerate
        if x.shape[2] > self.pool_scale:
            x_in = F.avg_pool2d(x, kernel_size=self.pool_scale, stride=self.pool_scale)
        else:
            x_in = x

        out = self.router(x_in)
        global_logits = torch.mean(out, dim=[2, 3])
        return self._process_logits(global_logits, self.noise_std, self.training)


class AdvancedRoutingLayer(nn.Module):
    """Compatibility router used by some legacy checkpoints; behaves like a global average-pooling router."""

    def __init__(self, in_channels=64, num_experts=3, top_k=None):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = num_experts if top_k is None else min(top_k, num_experts)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if not hasattr(self, "router"):
            reduced = max(in_channels // 8, 8)
            self.router = nn.Sequential(
                nn.Conv2d(in_channels, reduced, 1, bias=False),
                nn.SiLU(inplace=True),
                nn.Conv2d(reduced, num_experts, 1, bias=True),
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        B, C, H, W = x.shape
        if not hasattr(self, "avg_pool"):
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if not hasattr(self, "softmax"):
            self.softmax = nn.Softmax(dim=1)
        if not hasattr(self, "router"):
            reduced = max(C // 8, 8)
            self.router = nn.Sequential(
                nn.Conv2d(C, reduced, 1, bias=False),
                nn.SiLU(inplace=True),
                nn.Conv2d(reduced, getattr(self, "num_experts", 3), 1, bias=True),
            )
        pooled = self.avg_pool(x)
        if hasattr(self, "router") and isinstance(self.router, nn.Sequential) and len(self.router) > 0 and isinstance(
                self.router[0], nn.Conv2d):
            expected_in = self.router[0].in_channels
            if expected_in != C:
                if not hasattr(self, "_proj") or not isinstance(self._proj,
                                                                nn.Conv2d) or self._proj.in_channels != C or self._proj.out_channels != expected_in:
                    self._proj = nn.Conv2d(C, expected_in, 1, bias=False)
                pooled = self._proj(pooled)
        logits = self.router(pooled)
        probs = self.softmax(logits)
        E = probs.shape[1]
        k = getattr(self, "top_k", E)
        k = max(1, min(k, E))
        if k < E:
            vals, idx = torch.topk(probs, k, dim=1)
            vals = vals / (vals.sum(dim=1, keepdim=True) + 1e-6)
            weights = torch.zeros_like(probs)
            weights.scatter_(1, idx, vals)
        else:
            weights = probs
        return weights.repeat(1, 1, H, W)


class DynamicRoutingLayer(nn.Module):
    def __init__(self, in_channels, num_experts=3, reduction=8, top_k=None):
        """
        Args:
            top_k: Number of active experts; if None uses all experts (Softmax)
        """
        super(DynamicRoutingLayer, self).__init__()
        reduced_channels = max(in_channels // reduction, 8)

        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts) if top_k is not None else num_experts
        self.use_top_k = (top_k is not None)  # whether to enable Top-K

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Remove Softmax and control manually
        self.routing_network = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(reduced_channels, num_experts, kernel_size=1),
        )

    def forward(self, x):
        pooled = self.global_pool(x)
        routing_logits = self.routing_network(pooled)  # [B, num_experts, 1, 1]

        # Choose strategy based on Top-K enablement and train/infer mode
        if not self.use_top_k:
            # No Top-K: direct Softmax
            routing_weights = F.softmax(routing_logits, dim=1)
        elif self.training:
            # Training: soft Top-K (keeps gradients flowing)
            routing_weights = self._soft_top_k(routing_logits)
        else:
            # Inference: hard Top-K (truly sparse)
            routing_weights = self._hard_top_k(routing_logits)

        return routing_weights.repeat(1, 1, x.size(2), x.size(3))

    def _soft_top_k(self, logits):
        """Soft Top-K during training to maintain gradient flow."""
        B, E, H, W = logits.shape
        logits_flat = logits.view(B, E, -1)

        # Compute softmax
        weights = F.softmax(logits_flat, dim=1)

        # Find Top-K and build mask
        _, topk_indices = torch.topk(weights, self.top_k, dim=1)
        idx = topk_indices.permute(0, 2, 1).contiguous()
        mask_one_hot = F.one_hot(idx, num_classes=E).sum(dim=2)
        mask_one_hot = mask_one_hot.permute(0, 2, 1).contiguous().to(weights.dtype)

        # Apply mask and re-normalize
        weights = weights * mask_one_hot
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)

        return weights.view(B, E, H, W)

    def _hard_top_k(self, logits):
        """Hard Top-K during inference for true sparsity."""
        B, E, H, W = logits.shape
        logits_flat = logits.view(B, E, -1)

        # Find Top-K
        topk_values, topk_indices = torch.topk(logits_flat, self.top_k, dim=1)

        # Apply softmax to Top-K logits
        topk_weights = F.softmax(topk_values, dim=1)

        # Construct sparse weights
        idx = topk_indices.permute(0, 2, 1).contiguous()
        oh = F.one_hot(idx, num_classes=E)
        tw = topk_weights.permute(0, 2, 1).contiguous()
        weighted = (oh.to(tw.dtype) * tw.unsqueeze(-1)).sum(dim=2)
        weights = weighted.permute(0, 2, 1).contiguous()

        return weights.view(B, E, H, W)


class ESMoE(nn.Module):
    """Improved MoE with pluggable routers/experts and a shared expert for stability."""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            num_experts: int = 4,
            top_k: int = 2,
            expert_type: str = 'simple',  # ['simple', 'ghost', 'inverted']
            router_type: str = 'efficient',  # ['efficient', 'local', 'adaptive']
            noise_std: float = 1.0,
            balance_loss_coeff: float = 0.01,
            router_z_loss_coeff: float = 1e-3
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_experts = num_experts
        self.top_k = top_k
        self.balance_loss_coeff = balance_loss_coeff
        self.router_z_loss_coeff = router_z_loss_coeff

        # 1) Instantiate Router
        if router_type == 'local':
            self.routing = LocalRoutingLayer(in_channels, num_experts, top_k=top_k, noise_std=noise_std)
        elif router_type == 'adaptive':
            self.routing = AdaptiveRoutingLayer(in_channels, num_experts, top_k=top_k, noise_std=noise_std)
        else:
            self.routing = EfficientSpatialRouter(in_channels, num_experts, top_k=top_k, noise_std=noise_std)

        # 2) Instantiate Experts
        self.experts = nn.ModuleList()
        if expert_type == 'ghost':
            expert_cls = GhostExpert
        elif expert_type == 'inverted':
            expert_cls = InvertedResidualExpert
        else:
            expert_cls = SimpleExpert

        for _ in range(num_experts):
            self.experts.append(expert_cls(in_channels, out_channels))

        # 3) Shared expert (Always active)
        self.shared_expert = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )

        self._init_weights()
        self.aux_loss = torch.tensor(0.0)
        self.moe_loss_fn = MoELoss(balance_loss_coeff, router_z_loss_coeff, num_experts, top_k)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Robust router init: find the last Conv layer to initialize
        # Keep initial expert probabilities nearly uniform
        for m in self.routing.router.modules():
            if isinstance(m, nn.Conv2d):
                last_conv = m
        if last_conv:
            nn.init.normal_(last_conv.weight, mean=0, std=0.01)
            if last_conv.bias is not None:
                nn.init.constant_(last_conv.bias, 0)

    def forward(self, x):
        B, C, H, W = x.shape

        # 1) Routing (standardized interface)
        # loss_dict contains training loss inputs; empty during inference
        routing_weights, routing_indices, loss_dict = self.routing(x)

        # 2) Shared expert compute
        shared_out = self.shared_expert(x)

        # 3) Sparse expert compute
        # Initialize outputs with zeros
        expert_output = torch.zeros(B, self.out_channels, H, W, device=x.device, dtype=x.dtype)

        indices_flat = routing_indices.view(B, self.top_k)
        weights_flat = routing_weights.view(B, self.top_k)

        for i in range(self.num_experts):
            # Find all samples assigned to expert i
            mask = (indices_flat == i)
            if mask.any():
                batch_idx, k_idx = torch.where(mask)

                # Select input and compute
                inp = x[batch_idx]
                out = self.experts[i](inp)

                # Select weights and broadcast
                w = weights_flat[batch_idx, k_idx].view(-1, 1, 1, 1)

                # Accumulate results
                expert_output.index_add_(0, batch_idx, out.to(expert_output.dtype) * w.to(expert_output.dtype))

        final_output = shared_out + expert_output

        return final_output