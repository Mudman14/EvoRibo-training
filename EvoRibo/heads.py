"""Readout heads for EvoRibo pair representations."""

from __future__ import annotations

import torch
from torch import nn

from .stat_targets import STATS_Q, symmetrize_stats_logits


def symmetric_pair_features(pair: torch.Tensor) -> torch.Tensor:
    """Build symmetric features from both directions of a directed pair state."""
    if pair.ndim != 4:
        raise ValueError(f"pair must have shape [B, L, L, C], got {tuple(pair.shape)}")
    if pair.shape[1] != pair.shape[2]:
        raise ValueError(f"pair must be square on axes 1/2, got {tuple(pair.shape)}")
    pair_t = pair.transpose(1, 2)
    return torch.cat([pair + pair_t, torch.abs(pair - pair_t)], dim=-1)


def upper_or_full_symmetric_map(
    raw: torch.Tensor,
    pair_mask: torch.Tensor | None = None,
    return_full: bool = True,
) -> torch.Tensor:
    """Keep valid upper triangle or mirror it into a full symmetric scalar map."""
    if raw.ndim != 3 or raw.shape[1] != raw.shape[2]:
        raise ValueError(f"raw must have shape [B, L, L], got {tuple(raw.shape)}")

    B, L, _ = raw.shape
    upper_mask = torch.triu(
        torch.ones(L, L, device=raw.device, dtype=torch.bool),
        diagonal=1,
    )
    valid_upper = upper_mask[None].expand(B, L, L)

    sym_pair_mask = None
    if pair_mask is not None:
        if pair_mask.shape != (B, L, L):
            raise ValueError(
                "pair_mask must have shape [B, L, L], "
                f"got {tuple(pair_mask.shape)}"
            )
        sym_pair_mask = pair_mask.bool() & pair_mask.bool().transpose(-1, -2)
        valid_upper = valid_upper & sym_pair_mask

    upper = raw.masked_fill(~valid_upper, 0.0)
    if not return_full:
        return upper

    full = upper + upper.transpose(-1, -2)
    if sym_pair_mask is not None:
        full = full * sym_pair_mask.float()
    return full[:, None]


def _group_count(channels: int) -> int:
    groups = min(8, channels)
    while channels % groups != 0:
        groups -= 1
    return groups


def _symmetrize_channels(x: torch.Tensor, pair_mask: torch.Tensor | None = None) -> torch.Tensor:
    x = 0.5 * (x + x.transpose(-1, -2))
    if pair_mask is not None:
        sym_mask = pair_mask.bool() & pair_mask.bool().transpose(-1, -2)
        x = x * sym_mask[:, None].float()
    return x


class DirectCoevoHead(nn.Module):
    """Read a symmetric scalar coevolution map from a directed pair state."""

    def __init__(self, d_z: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or d_z
        self.net = nn.Sequential(
            nn.LayerNorm(d_z),
            nn.Linear(d_z, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        pair: torch.Tensor,
        pair_mask: torch.Tensor | None = None,
        symmetrize: bool = True,
        zero_diagonal: bool = True,
    ) -> torch.Tensor:
        if pair.ndim != 4:
            raise ValueError(f"pair must have shape [B, L, L, C], got {tuple(pair.shape)}")

        y = self.net(pair).permute(0, 3, 1, 2).contiguous()

        if symmetrize:
            y = 0.5 * (y + y.transpose(-1, -2))

        if zero_diagonal:
            L = y.shape[-1]
            diag = torch.eye(L, device=y.device, dtype=torch.bool)
            y = y.masked_fill(diag[None, None], 0.0)

        if pair_mask is not None:
            y = y * pair_mask[:, None].float()

        return y


class SymmetricUpperTriangleCoevoHead(nn.Module):
    """Read a strictly symmetric scalar map from directed pair states.

    The head keeps the encoder's directed pair representation intact, but
    predicts each unordered residue pair only once. It builds a symmetric
    feature from both directions:

        [z_ij + z_ji, abs(z_ij - z_ji)]

    The scalar prediction is retained only on the upper triangle. Full-map
    output is reconstructed by mirroring that upper triangle.
    """

    def __init__(self, d_z: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or d_z
        self.net = nn.Sequential(
            nn.LayerNorm(2 * d_z),
            nn.Linear(2 * d_z, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        pair: torch.Tensor,
        pair_mask: torch.Tensor | None = None,
        return_full: bool = True,
    ) -> torch.Tensor:
        feat = symmetric_pair_features(pair)
        raw = self.net(feat).squeeze(-1)  # [B, L, L]
        return upper_or_full_symmetric_map(raw, pair_mask=pair_mask, return_full=return_full)


class ResidualMLPBlock(nn.Module):
    """Small residual MLP block over the last dimension."""

    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        nn.init.zeros_(self.net[-2].weight)
        nn.init.zeros_(self.net[-2].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class SymmetricResidualMLPCoevoHead(nn.Module):
    """Lightweight residual readout from directed pair states to a symmetric map."""

    def __init__(
        self,
        d_z: int,
        hidden_dim: int | None = None,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        hidden_dim = hidden_dim or d_z
        self.input = nn.Sequential(
            nn.LayerNorm(2 * d_z),
            nn.Linear(2 * d_z, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList(
            ResidualMLPBlock(hidden_dim, dropout) for _ in range(max(0, num_layers - 1))
        )
        self.out = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        pair: torch.Tensor,
        pair_mask: torch.Tensor | None = None,
        return_full: bool = True,
    ) -> torch.Tensor:
        h = self.input(symmetric_pair_features(pair))
        for block in self.blocks:
            h = block(h)
        raw = self.out(h).squeeze(-1)
        return upper_or_full_symmetric_map(raw, pair_mask=pair_mask, return_full=return_full)


class SymmetricResidualMLPStatsHead(nn.Module):
    """Residual readout from directed pair states to reversible 25-channel pair stats.

    The head predicts a full directed tensor, then explicitly enforces pair
    symmetry in the flattened 5x5 channels:

        logits[i, j, a, b] == logits[j, i, b, a]
    """

    def __init__(
        self,
        d_z: int,
        hidden_dim: int | None = None,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        hidden_dim = hidden_dim or d_z
        self.input = nn.Sequential(
            nn.LayerNorm(2 * d_z),
            nn.Linear(2 * d_z, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList(
            ResidualMLPBlock(hidden_dim, dropout) for _ in range(max(0, num_layers - 1))
        )
        self.out = nn.Linear(hidden_dim, STATS_Q * STATS_Q)

    def forward(
        self,
        pair: torch.Tensor,
        pair_mask: torch.Tensor | None = None,
        return_logits: bool = True,
    ) -> torch.Tensor:
        h = self.input(symmetric_pair_features(pair))
        for block in self.blocks:
            h = block(h)
        logits = symmetrize_stats_logits(self.out(h), pair_mask=pair_mask)
        if return_logits:
            return logits
        return torch.softmax(logits, dim=-1)


class ResidualSingleFiHead(nn.Module):
    """Residual readout from trunk single states to single-site ``fi`` logits."""

    def __init__(
        self,
        d_s: int,
        hidden_dim: int | None = None,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        hidden_dim = hidden_dim or d_s
        self.input = nn.Sequential(
            nn.LayerNorm(d_s),
            nn.Linear(d_s, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList(
            ResidualMLPBlock(hidden_dim, dropout) for _ in range(max(0, num_layers - 1))
        )
        self.out = nn.Linear(hidden_dim, STATS_Q)

    def forward(
        self,
        single: torch.Tensor,
        mask: torch.Tensor | None = None,
        return_logits: bool = True,
    ) -> torch.Tensor:
        if single.ndim != 3:
            raise ValueError(f"single must have shape [B, L, C], got {tuple(single.shape)}")
        h = self.input(single)
        for block in self.blocks:
            h = block(h)
        logits = self.out(h)
        if mask is not None:
            if mask.shape != logits.shape[:2]:
                raise ValueError(f"mask must be [B, L], got {tuple(mask.shape)}")
            logits = logits * mask[..., None].float()
        if return_logits:
            return logits
        return torch.softmax(logits, dim=-1)


class SymmetricResidualMLPFijHead(nn.Module):
    """Residual readout from directed pair states to symmetric off-diagonal ``fij`` logits."""

    def __init__(
        self,
        d_z: int,
        hidden_dim: int | None = None,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        hidden_dim = hidden_dim or d_z
        self.input = nn.Sequential(
            nn.LayerNorm(2 * d_z),
            nn.Linear(2 * d_z, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList(
            ResidualMLPBlock(hidden_dim, dropout) for _ in range(max(0, num_layers - 1))
        )
        self.out = nn.Linear(hidden_dim, STATS_Q * STATS_Q)

    def forward(
        self,
        pair: torch.Tensor,
        pair_mask: torch.Tensor | None = None,
        return_logits: bool = True,
    ) -> torch.Tensor:
        h = self.input(symmetric_pair_features(pair))
        for block in self.blocks:
            h = block(h)
        logits = symmetrize_stats_logits(self.out(h), pair_mask=pair_mask)
        if return_logits:
            return logits
        return torch.softmax(logits, dim=-1)


class FactorizedStatsHead(nn.Module):
    """Two-head stats readout: ``single -> fi`` and ``pair -> fij``."""

    def __init__(
        self,
        d_s: int,
        d_z: int,
        hidden_dim: int | None = None,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.fi_head = ResidualSingleFiHead(
            d_s=d_s,
            hidden_dim=hidden_dim or d_s,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.fij_head = SymmetricResidualMLPFijHead(
            d_z=d_z,
            hidden_dim=hidden_dim or d_z,
            num_layers=num_layers,
            dropout=dropout,
        )

    def forward(
        self,
        single: torch.Tensor,
        pair: torch.Tensor,
        mask: torch.Tensor | None = None,
        pair_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        fi_logits = self.fi_head(single, mask=mask)
        fij_logits = self.fij_head(pair, pair_mask=pair_mask)
        return {
            "fi_logits": fi_logits,
            "fij_logits": fij_logits,
            "fi_probs": torch.softmax(fi_logits, dim=-1),
            "fij_probs": torch.softmax(fij_logits, dim=-1),
        }


class SymmetricConv2DRefinementBlock(nn.Module):
    """Residual 2D conv block with explicit spatial symmetrization."""

    def __init__(self, channels: int, kernel_size: int, dropout: float):
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.GroupNorm(_group_count(channels), channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding),
            nn.Dropout2d(dropout),
            nn.GroupNorm(_group_count(channels), channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding),
            nn.Dropout2d(dropout),
        )
        nn.init.zeros_(self.net[-2].weight)
        nn.init.zeros_(self.net[-2].bias)

    def forward(self, x: torch.Tensor, pair_mask: torch.Tensor | None = None) -> torch.Tensor:
        return _symmetrize_channels(x + self.net(x), pair_mask=pair_mask)


class SymmetricConv2DRefinementHead(nn.Module):
    """2D refinement readout for ablations, with symmetry enforced after every block."""

    def __init__(
        self,
        d_z: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        self.input_norm = nn.LayerNorm(2 * d_z)
        self.input_proj = nn.Linear(2 * d_z, hidden_dim)
        self.blocks = nn.ModuleList(
            SymmetricConv2DRefinementBlock(hidden_dim, kernel_size, dropout)
            for _ in range(num_layers)
        )
        self.out_norm = nn.GroupNorm(_group_count(hidden_dim), hidden_dim)
        self.out = nn.Conv2d(hidden_dim, 1, kernel_size=1)

    def forward(
        self,
        pair: torch.Tensor,
        pair_mask: torch.Tensor | None = None,
        return_full: bool = True,
    ) -> torch.Tensor:
        feat = self.input_norm(symmetric_pair_features(pair))
        h = self.input_proj(feat).permute(0, 3, 1, 2).contiguous()
        h = _symmetrize_channels(h, pair_mask=pair_mask)
        for block in self.blocks:
            h = block(h, pair_mask=pair_mask)
        raw = self.out(torch.nn.functional.silu(self.out_norm(h))).squeeze(1)
        raw = 0.5 * (raw + raw.transpose(-1, -2))
        return upper_or_full_symmetric_map(raw, pair_mask=pair_mask, return_full=return_full)
