"""Flow matching heads for pair-state initialized coevolution maps."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .heads import ResidualSingleFiHead, symmetric_pair_features
from .stat_targets import (
    STATS_Q,
    center_stats_logits_fp32,
    centered_log_prob_target,
    stats_kl_loss,
    symmetrize_stats_logits,
    weighted_kl_from_logits_fp32,
)


def upper_pair_mask(pair_mask: torch.Tensor) -> torch.Tensor:
    """Return valid upper-triangle mask [B, L, L]."""
    if pair_mask.ndim != 3 or pair_mask.shape[1] != pair_mask.shape[2]:
        raise ValueError(f"pair_mask must have shape [B, L, L], got {tuple(pair_mask.shape)}")
    B, L, _ = pair_mask.shape
    upper = torch.triu(torch.ones(L, L, device=pair_mask.device, dtype=torch.bool), diagonal=1)
    sym_mask = pair_mask.bool() & pair_mask.bool().transpose(-1, -2)
    return upper[None].expand(B, L, L) & sym_mask


def symmetrize_scalar_map(
    x: torch.Tensor,
    pair_mask: torch.Tensor | None = None,
    zero_diagonal: bool = True,
) -> torch.Tensor:
    """Symmetrize scalar maps and apply optional pair mask.

    Accepts [B, L, L] or [B, 1, L, L]. Returns the same rank as input.
    """
    squeeze = False
    if x.ndim == 3:
        x = x[:, None]
        squeeze = True
    if x.ndim != 4 or x.shape[1] != 1 or x.shape[-1] != x.shape[-2]:
        raise ValueError(f"x must be [B, L, L] or [B, 1, L, L], got {tuple(x.shape)}")

    y = 0.5 * (x + x.transpose(-1, -2))
    if zero_diagonal:
        L = y.shape[-1]
        diag = torch.eye(L, device=y.device, dtype=torch.bool)
        y = y.masked_fill(diag[None, None], 0.0)
    if pair_mask is not None:
        if pair_mask.shape != y.shape[:1] + y.shape[-2:]:
            raise ValueError(
                f"pair_mask shape {tuple(pair_mask.shape)} does not match map {tuple(y.shape)}"
            )
        sym_mask = pair_mask.bool() & pair_mask.bool().transpose(-1, -2)
        y = y * sym_mask[:, None].float()
    return y[:, 0] if squeeze else y


def symmetric_gaussian_noise_like(
    x: torch.Tensor,
    pair_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Create symmetric Gaussian noise with zero diagonal and masked padding."""
    noise = torch.randn_like(x)
    return symmetrize_scalar_map(noise, pair_mask=pair_mask)


def masked_upper_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    pair_mask: torch.Tensor,
) -> torch.Tensor:
    """MSE on valid upper-triangle entries only."""
    if pred.ndim == 3:
        pred = pred[:, None]
    if target.ndim == 3:
        target = target[:, None]
    if pred.shape != target.shape:
        raise ValueError(f"pred/target shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
    valid = upper_pair_mask(pair_mask)[:, None]
    if not valid.any():
        return (pred - target).sum() * 0.0
    return F.mse_loss(pred[valid], target[valid])


def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding for scalar timesteps in [0, 1]."""
    if t.ndim == 0:
        t = t[None]
    t = t.float().view(-1)
    half = dim // 2
    if half == 0:
        return t[:, None]
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / max(half - 1, 1)
    )
    args = t[:, None] * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


def _group_count(channels: int) -> int:
    groups = min(8, channels)
    while channels % groups != 0:
        groups -= 1
    return groups


class PairToInitialMapHead(nn.Module):
    """Project directed pair representation to a symmetric scalar initial map."""

    def __init__(self, d_z: int, hidden_dim: int | None = None, use_tanh: bool = True):
        super().__init__()
        hidden_dim = hidden_dim or d_z
        self.use_tanh = use_tanh
        self.net = nn.Sequential(
            nn.LayerNorm(2 * d_z),
            nn.Linear(2 * d_z, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, pair: torch.Tensor, pair_mask: torch.Tensor | None = None) -> torch.Tensor:
        if pair.ndim != 4 or pair.shape[1] != pair.shape[2]:
            raise ValueError(f"pair must have shape [B, L, L, C], got {tuple(pair.shape)}")
        pair_t = pair.transpose(1, 2)
        feat = torch.cat([pair + pair_t, torch.abs(pair - pair_t)], dim=-1)
        raw = self.net(feat).permute(0, 3, 1, 2).contiguous()
        if self.use_tanh:
            raw = torch.tanh(raw)
        return symmetrize_scalar_map(raw, pair_mask=pair_mask)


class TimeFiLMResBlock(nn.Module):
    """2D residual block with sinusoidal-time FiLM modulation."""

    def __init__(self, in_channels: int, out_channels: int, time_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, 2 * out_channels)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.time_proj(time_emb).chunk(2, dim=-1)
        h = h * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return self.skip(x) + h


class FlowUNet2D(nn.Module):
    """Small 2D U-Net for scalar map flow prediction."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        levels: int = 3,
        time_dim: int = 128,
    ):
        super().__init__()
        if levels < 1:
            raise ValueError("levels must be at least 1")
        self.levels = levels
        self.time_dim = time_dim
        channels = [base_channels * (2**idx) for idx in range(levels)]

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )
        self.in_block = TimeFiLMResBlock(in_channels, channels[0], time_dim)
        self.down_blocks = nn.ModuleList(
            TimeFiLMResBlock(channels[idx - 1], channels[idx], time_dim)
            for idx in range(1, levels)
        )
        self.bottleneck = TimeFiLMResBlock(channels[-1], channels[-1], time_dim)
        self.up_blocks = nn.ModuleList(
            TimeFiLMResBlock(channels[idx] + channels[idx - 1], channels[idx - 1], time_dim)
            for idx in range(levels - 1, 0, -1)
        )
        self.out_norm = nn.GroupNorm(_group_count(channels[0]), channels[0])
        self.out_conv = nn.Conv2d(channels[0], out_channels, kernel_size=3, padding=1)

    def _pad_to_multiple(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        multiple = 2 ** max(self.levels - 1, 0)
        H, W = x.shape[-2:]
        pad_h = (multiple - H % multiple) % multiple
        pad_w = (multiple - W % multiple) % multiple
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        return x, (pad_h, pad_w)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        pair_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"x must have shape [B, C, L, L], got {tuple(x.shape)}")
        orig_h, orig_w = x.shape[-2:]
        x, _ = self._pad_to_multiple(x)
        time_emb = self.time_mlp(sinusoidal_time_embedding(t, self.time_dim).to(x.dtype))

        h = self.in_block(x, time_emb)
        skips = [h]
        for idx, block in enumerate(self.down_blocks):
            h = F.avg_pool2d(h, kernel_size=2)
            h = block(h, time_emb)
            if idx < len(self.down_blocks) - 1:
                skips.append(h)

        h = self.bottleneck(h, time_emb)
        for block in self.up_blocks:
            skip = skips.pop()
            h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = block(torch.cat([h, skip], dim=1), time_emb)

        h = self.out_conv(F.silu(self.out_norm(h)))
        h = h[..., :orig_h, :orig_w]
        if pair_mask is not None:
            h = symmetrize_scalar_map(h, pair_mask=pair_mask)
        return h


class FlowMatchingHead(nn.Module):
    """Pair-state initialized flow matching head for coevolution maps."""

    def __init__(
        self,
        d_z: int,
        prior_hidden_dim: int | None = None,
        unet_base_channels: int = 64,
        unet_levels: int = 3,
        time_dim: int = 128,
        objective: str = "endpoint",
    ):
        super().__init__()
        if objective not in {"endpoint", "velocity"}:
            raise ValueError("objective must be 'endpoint' or 'velocity'")
        self.objective = objective
        self.prior = PairToInitialMapHead(d_z=d_z, hidden_dim=prior_hidden_dim)
        self.net = FlowUNet2D(
            in_channels=1,
            out_channels=1,
            base_channels=unet_base_channels,
            levels=unet_levels,
            time_dim=time_dim,
        )

    def forward(
        self,
        pair: torch.Tensor,
        target: torch.Tensor,
        pair_mask: torch.Tensor,
        objective: str | None = None,
        x0_noise_scale: float = 0.05,
        t: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        objective = objective or self.objective
        if objective not in {"endpoint", "velocity"}:
            raise ValueError("objective must be 'endpoint' or 'velocity'")
        if target.ndim == 3:
            target = target[:, None]

        target = symmetrize_scalar_map(target, pair_mask=pair_mask)
        x0_mean = self.prior(pair, pair_mask=pair_mask)
        noise = symmetric_gaussian_noise_like(x0_mean, pair_mask=pair_mask)
        x0 = symmetrize_scalar_map(x0_mean + x0_noise_scale * noise, pair_mask=pair_mask)

        B = pair.shape[0]
        if t is None:
            t = torch.rand(B, device=pair.device, dtype=pair.dtype)
        else:
            t = t.to(device=pair.device, dtype=pair.dtype).view(B)
        t_view = t[:, None, None, None]
        xt = symmetrize_scalar_map((1.0 - t_view) * x0 + t_view * target, pair_mask=pair_mask)
        pred = symmetrize_scalar_map(self.net(xt, t, pair_mask=pair_mask), pair_mask=pair_mask)

        if objective == "endpoint":
            flow_loss = masked_upper_mse(pred, target, pair_mask)
            pred_endpoint = pred
            pred_velocity = symmetrize_scalar_map((pred - xt) / (1.0 - t_view).clamp_min(1e-4), pair_mask)
        else:
            velocity_target = target - x0
            flow_loss = masked_upper_mse(pred, velocity_target, pair_mask)
            pred_velocity = pred
            pred_endpoint = symmetrize_scalar_map(xt + (1.0 - t_view) * pred_velocity, pair_mask)
        prior_loss = masked_upper_mse(x0_mean, target, pair_mask)

        return {
            "flow_loss": flow_loss,
            "prior_loss": prior_loss,
            "x0_mean": x0_mean,
            "x0": x0,
            "xt": xt,
            "t": t,
            "pred": pred,
            "pred_endpoint": pred_endpoint,
            "pred_velocity": pred_velocity,
        }

    @torch.no_grad()
    def sample(
        self,
        pair: torch.Tensor,
        pair_mask: torch.Tensor,
        objective: str | None = None,
        steps: int = 16,
        x0_noise_scale: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        if steps < 1:
            raise ValueError("steps must be >= 1")
        objective = objective or self.objective
        x0_mean = self.prior(pair, pair_mask=pair_mask)
        noise = symmetric_gaussian_noise_like(x0_mean, pair_mask=pair_mask)
        x = symmetrize_scalar_map(x0_mean + x0_noise_scale * noise, pair_mask=pair_mask)
        B = pair.shape[0]
        for idx in range(steps):
            t0 = idx / steps
            t1 = (idx + 1) / steps
            dt = t1 - t0
            t = torch.full((B,), t0, device=pair.device, dtype=pair.dtype)
            pred = symmetrize_scalar_map(self.net(x, t, pair_mask=pair_mask), pair_mask=pair_mask)
            if objective == "endpoint":
                v = (pred - x) / max(1.0 - t0, 1e-4)
            elif objective == "velocity":
                v = pred
            else:
                raise ValueError("objective must be 'endpoint' or 'velocity'")
            x = symmetrize_scalar_map(x + dt * v, pair_mask=pair_mask)
        return {"sample": x, "x0_mean": x0_mean}


def _stats_logits_to_channels(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 4 or x.shape[-1] != STATS_Q * STATS_Q:
        raise ValueError(f"x must have shape [B, L, L, 25], got {tuple(x.shape)}")
    return x.permute(0, 3, 1, 2).contiguous()


def _channels_to_stats_logits(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 4 or x.shape[1] != STATS_Q * STATS_Q:
        raise ValueError(f"x must have shape [B, 25, L, L], got {tuple(x.shape)}")
    return x.permute(0, 2, 3, 1).contiguous()


def _center_stats_logits(logits: torch.Tensor) -> torch.Tensor:
    return logits - logits.mean(dim=-1, keepdim=True)


def symmetric_stats_gaussian_noise_like(
    x: torch.Tensor,
    pair_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    return symmetrize_stats_logits(torch.randn_like(x), pair_mask=pair_mask)


class PairToInitialStatsHead(nn.Module):
    """Project directed pair representation to symmetric 25-channel initial logits."""

    def __init__(self, d_z: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or d_z
        self.net = nn.Sequential(
            nn.LayerNorm(2 * d_z),
            nn.Linear(2 * d_z, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, STATS_Q * STATS_Q),
        )

    def forward(self, pair: torch.Tensor, pair_mask: torch.Tensor | None = None) -> torch.Tensor:
        logits = self.net(symmetric_pair_features(pair))
        return _center_stats_logits(symmetrize_stats_logits(logits, pair_mask=pair_mask))


class PairToStatsConditionProjector(nn.Module):
    """Project directed pair state into symmetric conditioning channels."""

    def __init__(self, d_z: int, condition_dim: int = 64, dropout: float = 0.0):
        super().__init__()
        if condition_dim < 1:
            raise ValueError("condition_dim must be >= 1")
        self.net = nn.Sequential(
            nn.LayerNorm(2 * d_z),
            nn.Linear(2 * d_z, condition_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(condition_dim, condition_dim),
        )

    def forward(self, pair: torch.Tensor, pair_mask: torch.Tensor | None = None) -> torch.Tensor:
        cond = self.net(symmetric_pair_features(pair))
        cond = 0.5 * (cond + cond.transpose(1, 2))
        if pair_mask is not None:
            cond = cond * pair_mask[..., None].to(cond.dtype)
        return cond


def _weighted_stats_smooth_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    pair_mask: torch.Tensor,
    pair_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError(f"pred/target mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
    if pred.ndim != 4 or pred.shape[-1] != STATS_Q * STATS_Q:
        raise ValueError(f"pred must be [B, L, L, 25], got {tuple(pred.shape)}")
    valid = upper_pair_mask(pair_mask)
    weights = valid.float()
    if pair_weights is not None:
        if pair_weights.shape != pair_mask.shape:
            raise ValueError(f"pair_weights must be [B, L, L], got {tuple(pair_weights.shape)}")
        weights = weights * pair_weights.float().clamp_min(0.0)
    if not (weights > 0).any():
        return (pred - target).sum() * 0.0
    per_pair = F.smooth_l1_loss(pred.float(), target.float(), reduction="none").mean(dim=-1)
    return (per_pair * weights).sum() / weights.sum().clamp_min(1.0)


class ConditionalStatsVelocityFlowHead(nn.Module):
    """Conditional vector-field flow head for off-diagonal ``fij`` logits."""

    def __init__(
        self,
        d_z: int,
        prior_hidden_dim: int | None = None,
        condition_dim: int = 64,
        unet_base_channels: int = 64,
        unet_levels: int = 3,
        time_dim: int = 128,
        objective: str = "velocity",
        dropout: float = 0.0,
        eps: float = 1e-6,
        logit_clip: float = 8.0,
    ):
        super().__init__()
        if objective not in {"velocity", "endpoint"}:
            raise ValueError("objective must be 'velocity' or 'endpoint'")
        self.objective = objective
        self.eps = eps
        self.logit_clip = logit_clip
        self.prior = PairToInitialStatsHead(d_z=d_z, hidden_dim=prior_hidden_dim)
        self.cond = PairToStatsConditionProjector(d_z=d_z, condition_dim=condition_dim, dropout=dropout)
        self.net = FlowUNet2D(
            in_channels=STATS_Q * STATS_Q + condition_dim,
            out_channels=STATS_Q * STATS_Q,
            base_channels=unet_base_channels,
            levels=unet_levels,
            time_dim=time_dim,
        )

    def _condition_channels(
        self,
        pair: torch.Tensor,
        pair_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        cond = self.cond(pair, pair_mask=pair_mask)
        return cond.permute(0, 3, 1, 2).contiguous()

    def forward(
        self,
        pair: torch.Tensor,
        fij_target: torch.Tensor,
        pair_mask: torch.Tensor,
        objective: str | None = None,
        x0_noise_scale: float = 0.05,
        t: torch.Tensor | None = None,
        pair_weights: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        objective = objective or self.objective
        if objective not in {"velocity", "endpoint"}:
            raise ValueError("objective must be 'velocity' or 'endpoint'")
        if fij_target.ndim != 4 or fij_target.shape[-1] != STATS_Q * STATS_Q:
            raise ValueError(f"fij_target must be [B, L, L, 25], got {tuple(fij_target.shape)}")

        B = pair.shape[0]
        x0_mean = center_stats_logits_fp32(self.prior(pair, pair_mask=pair_mask), pair_mask=pair_mask)
        x1 = centered_log_prob_target(fij_target, eps=self.eps, clip_value=self.logit_clip)
        x1 = center_stats_logits_fp32(symmetrize_stats_logits(x1, pair_mask=pair_mask), pair_mask=pair_mask)
        x0 = x0_mean
        if self.training and x0_noise_scale > 0.0:
            noise = symmetric_stats_gaussian_noise_like(x0_mean, pair_mask=pair_mask).float()
            x0 = center_stats_logits_fp32(x0_mean + float(x0_noise_scale) * noise, pair_mask=pair_mask)

        if t is None:
            t = torch.rand(B, device=pair.device, dtype=pair.dtype)
        else:
            t = t.to(device=pair.device, dtype=pair.dtype).view(B)
        t_view = t.float()[:, None, None, None]
        xt_float = center_stats_logits_fp32(
            symmetrize_stats_logits((1.0 - t_view) * x0 + t_view * x1, pair_mask=pair_mask),
            pair_mask=pair_mask,
        )
        cond_ch = self._condition_channels(pair, pair_mask=pair_mask).to(pair.dtype)
        net_in = torch.cat([_stats_logits_to_channels(xt_float.to(pair.dtype)), cond_ch], dim=1)
        raw = _channels_to_stats_logits(self.net(net_in, t))

        if objective == "velocity":
            v_pred = center_stats_logits_fp32(symmetrize_stats_logits(raw, pair_mask=pair_mask), pair_mask=pair_mask)
            v_target = center_stats_logits_fp32(x1 - x0, pair_mask=pair_mask)
            endpoint_logits = center_stats_logits_fp32(
                symmetrize_stats_logits(xt_float + (1.0 - t_view) * v_pred, pair_mask=pair_mask),
                pair_mask=pair_mask,
            )
            velocity_loss = _weighted_stats_smooth_l1(v_pred, v_target, pair_mask, pair_weights=pair_weights)
            flow_loss = velocity_loss
        else:
            endpoint_logits = center_stats_logits_fp32(
                symmetrize_stats_logits(raw, pair_mask=pair_mask),
                pair_mask=pair_mask,
            )
            v_pred = center_stats_logits_fp32(endpoint_logits - xt_float, pair_mask=pair_mask)
            v_target = center_stats_logits_fp32(x1 - x0, pair_mask=pair_mask)
            velocity_loss = endpoint_logits.sum() * 0.0
            flow_loss = weighted_kl_from_logits_fp32(
                endpoint_logits,
                fij_target,
                pair_mask=pair_mask,
                pair_weights=pair_weights,
                eps=self.eps,
            )

        endpoint_kl_loss = weighted_kl_from_logits_fp32(
            endpoint_logits,
            fij_target,
            pair_mask=pair_mask,
            pair_weights=pair_weights,
            eps=self.eps,
        )
        prior_kl_loss = weighted_kl_from_logits_fp32(
            x0_mean,
            fij_target,
            pair_mask=pair_mask,
            pair_weights=pair_weights,
            eps=self.eps,
        )

        return {
            "flow_loss": flow_loss,
            "velocity_loss": velocity_loss,
            "endpoint_kl_loss": endpoint_kl_loss,
            "prior_kl_loss": prior_kl_loss,
            "x0_mean": x0_mean,
            "x0": x0,
            "x1": x1,
            "xt": xt_float,
            "t": t,
            "v_pred": v_pred,
            "v_target": v_target,
            "endpoint_logits": endpoint_logits,
            "pred_endpoint": endpoint_logits,
            "pred_velocity": v_pred,
        }

    @torch.no_grad()
    def sample(
        self,
        pair: torch.Tensor,
        pair_mask: torch.Tensor,
        objective: str | None = None,
        steps: int = 4,
        x0_noise_scale: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        objective = objective or self.objective
        if objective not in {"velocity", "endpoint"}:
            raise ValueError("objective must be 'velocity' or 'endpoint'")
        if steps < 1:
            raise ValueError("steps must be >= 1")
        if objective == "endpoint" and steps != 1:
            raise ValueError("endpoint stats flow requires sample steps == 1")

        x0_mean = center_stats_logits_fp32(self.prior(pair, pair_mask=pair_mask), pair_mask=pair_mask)
        x = x0_mean
        if x0_noise_scale > 0.0:
            noise = symmetric_stats_gaussian_noise_like(x0_mean, pair_mask=pair_mask).float()
            x = center_stats_logits_fp32(x + float(x0_noise_scale) * noise, pair_mask=pair_mask)
        cond_ch = self._condition_channels(pair, pair_mask=pair_mask).to(pair.dtype)
        B = pair.shape[0]

        if objective == "endpoint":
            t = torch.ones(B, device=pair.device, dtype=pair.dtype)
            net_in = torch.cat([_stats_logits_to_channels(x.to(pair.dtype)), cond_ch], dim=1)
            raw = _channels_to_stats_logits(self.net(net_in, t))
            x = center_stats_logits_fp32(symmetrize_stats_logits(raw, pair_mask=pair_mask), pair_mask=pair_mask)
        else:
            dt = 1.0 / float(steps)
            for idx in range(steps):
                t = torch.full((B,), idx / float(steps), device=pair.device, dtype=pair.dtype)
                net_in = torch.cat([_stats_logits_to_channels(x.to(pair.dtype)), cond_ch], dim=1)
                raw = _channels_to_stats_logits(self.net(net_in, t))
                v = center_stats_logits_fp32(symmetrize_stats_logits(raw, pair_mask=pair_mask), pair_mask=pair_mask)
                x = center_stats_logits_fp32(
                    symmetrize_stats_logits(x + dt * v, pair_mask=pair_mask),
                    pair_mask=pair_mask,
                )

        probs = F.softmax(x.float(), dim=-1)
        if pair_mask is not None:
            probs = probs * pair_mask[..., None].float()
        return {"sample_logits": x, "sample_probs": probs, "x0_mean": x0_mean}


class FactorizedConditionalFlowStatsHead(nn.Module):
    """Factorized stats head: single-site fi plus conditional-flow fij."""

    def __init__(
        self,
        d_s: int,
        d_z: int,
        fi_hidden_dim: int | None = None,
        fi_layers: int = 2,
        prior_hidden_dim: int | None = None,
        condition_dim: int = 64,
        unet_base_channels: int = 64,
        unet_levels: int = 3,
        time_dim: int = 128,
        objective: str = "velocity",
        dropout: float = 0.0,
        eps: float = 1e-6,
        logit_clip: float = 8.0,
    ):
        super().__init__()
        self.fi_head = ResidualSingleFiHead(
            d_s=d_s,
            hidden_dim=fi_hidden_dim,
            num_layers=fi_layers,
            dropout=dropout,
        )
        self.fij_flow = ConditionalStatsVelocityFlowHead(
            d_z=d_z,
            prior_hidden_dim=prior_hidden_dim,
            condition_dim=condition_dim,
            unet_base_channels=unet_base_channels,
            unet_levels=unet_levels,
            time_dim=time_dim,
            objective=objective,
            dropout=dropout,
            eps=eps,
            logit_clip=logit_clip,
        )

    def forward(
        self,
        single: torch.Tensor,
        pair: torch.Tensor,
        mask: torch.Tensor,
        pair_mask: torch.Tensor,
        fij_target: torch.Tensor,
        objective: str | None = None,
        x0_noise_scale: float = 0.05,
        t: torch.Tensor | None = None,
        pair_weights: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        fi_logits = self.fi_head(single, mask=mask)
        out = self.fij_flow(
            pair,
            fij_target,
            pair_mask,
            objective=objective,
            x0_noise_scale=x0_noise_scale,
            t=t,
            pair_weights=pair_weights,
        )
        out["fi_logits"] = fi_logits
        return out

    @torch.no_grad()
    def sample(
        self,
        pair: torch.Tensor,
        pair_mask: torch.Tensor,
        objective: str | None = None,
        steps: int = 4,
        x0_noise_scale: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        return self.fij_flow.sample(
            pair,
            pair_mask=pair_mask,
            objective=objective,
            steps=steps,
            x0_noise_scale=x0_noise_scale,
        )


class StatsFlowMatchingHead(nn.Module):
    """Pair-state initialized 25-channel flow head in pair-stat logit space."""

    def __init__(
        self,
        d_z: int,
        prior_hidden_dim: int | None = None,
        unet_base_channels: int = 64,
        unet_levels: int = 3,
        time_dim: int = 128,
        objective: str = "endpoint",
    ):
        super().__init__()
        if objective not in {"endpoint", "velocity"}:
            raise ValueError("objective must be 'endpoint' or 'velocity'")
        self.objective = objective
        self.prior = PairToInitialStatsHead(d_z=d_z, hidden_dim=prior_hidden_dim)
        self.net = FlowUNet2D(
            in_channels=STATS_Q * STATS_Q,
            out_channels=STATS_Q * STATS_Q,
            base_channels=unet_base_channels,
            levels=unet_levels,
            time_dim=time_dim,
        )

    def forward(
        self,
        pair: torch.Tensor,
        target: torch.Tensor,
        pair_mask: torch.Tensor,
        objective: str | None = None,
        x0_noise_scale: float = 0.05,
        t: torch.Tensor | None = None,
        pair_weights: torch.Tensor | None = None,
        diag_weight: float = 0.2,
    ) -> dict[str, torch.Tensor]:
        objective = objective or self.objective
        if objective not in {"endpoint", "velocity"}:
            raise ValueError("objective must be 'endpoint' or 'velocity'")
        if target.ndim != 4 or target.shape[-1] != STATS_Q * STATS_Q:
            raise ValueError(f"target must be [B, L, L, 25], got {tuple(target.shape)}")

        target_prob = target.clamp_min(1e-8)
        target_prob = target_prob / target_prob.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        target_logits = _center_stats_logits(symmetrize_stats_logits(target_prob.log(), pair_mask=pair_mask))
        x0_mean = self.prior(pair, pair_mask=pair_mask)
        noise = symmetric_stats_gaussian_noise_like(x0_mean, pair_mask=pair_mask)
        x0 = _center_stats_logits(symmetrize_stats_logits(x0_mean + x0_noise_scale * noise, pair_mask=pair_mask))

        B = pair.shape[0]
        if t is None:
            t = torch.rand(B, device=pair.device, dtype=pair.dtype)
        else:
            t = t.to(device=pair.device, dtype=pair.dtype).view(B)
        t_view = t[:, None, None, None]
        xt = _center_stats_logits(
            symmetrize_stats_logits((1.0 - t_view) * x0 + t_view * target_logits, pair_mask=pair_mask)
        )

        pred_ch = self.net(_stats_logits_to_channels(xt), t)
        pred = _center_stats_logits(symmetrize_stats_logits(_channels_to_stats_logits(pred_ch), pair_mask=pair_mask))

        if objective == "endpoint":
            flow_loss = stats_kl_loss(
                pred,
                target_prob,
                pair_mask,
                diag_weight=diag_weight,
                pair_weights=pair_weights,
            )
            pred_endpoint = pred
            pred_velocity = _center_stats_logits(
                symmetrize_stats_logits((pred - xt) / (1.0 - t_view).clamp_min(1e-4), pair_mask=pair_mask)
            )
        else:
            velocity_target = target_logits - x0
            valid = pair_mask[..., None].float()
            flow_loss = F.mse_loss(pred * valid, velocity_target * valid)
            pred_velocity = pred
            pred_endpoint = _center_stats_logits(
                symmetrize_stats_logits(xt + (1.0 - t_view) * pred_velocity, pair_mask=pair_mask)
            )
        prior_kl_loss = stats_kl_loss(
            x0_mean,
            target_prob,
            pair_mask,
            diag_weight=diag_weight,
            pair_weights=pair_weights,
        )
        return {
            "flow_loss": flow_loss,
            "prior_kl_loss": prior_kl_loss,
            "x0_mean": x0_mean,
            "x0": x0,
            "xt": xt,
            "t": t,
            "pred": pred,
            "pred_endpoint": pred_endpoint,
            "pred_velocity": pred_velocity,
        }

    @torch.no_grad()
    def sample(
        self,
        pair: torch.Tensor,
        pair_mask: torch.Tensor,
        objective: str | None = None,
        steps: int = 16,
        x0_noise_scale: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        objective = objective or self.objective
        if steps < 1:
            raise ValueError("steps must be >= 1")
        x0_mean = self.prior(pair, pair_mask=pair_mask)
        noise = symmetric_stats_gaussian_noise_like(x0_mean, pair_mask=pair_mask)
        x = _center_stats_logits(symmetrize_stats_logits(x0_mean + x0_noise_scale * noise, pair_mask=pair_mask))
        B = pair.shape[0]
        for idx in range(steps):
            t0 = idx / steps
            t1 = (idx + 1) / steps
            dt = t1 - t0
            t = torch.full((B,), t0, device=pair.device, dtype=pair.dtype)
            pred_ch = self.net(_stats_logits_to_channels(x), t)
            pred = _center_stats_logits(symmetrize_stats_logits(_channels_to_stats_logits(pred_ch), pair_mask=pair_mask))
            if objective == "endpoint":
                v = (pred - x) / max(1.0 - t0, 1e-4)
            elif objective == "velocity":
                v = pred
            else:
                raise ValueError("objective must be 'endpoint' or 'velocity'")
            x = _center_stats_logits(symmetrize_stats_logits(x + dt * v, pair_mask=pair_mask))
        return {"sample_logits": x, "sample_probs": torch.softmax(x, dim=-1), "x0_mean": x0_mean}
