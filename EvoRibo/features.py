"""Feature helpers for RNA sequence, BPP, and SS-aware adapters."""

from __future__ import annotations

import torch

A_IDX = 0
U_IDX = 1
C_IDX = 2
G_IDX = 3
GAP_IDX = 4
PAD_IDX = 5

SS_FEATURE_DIM = 7


def canonical_pair_features(tokens: torch.Tensor) -> torch.Tensor:
    """Build AU/GC/GU canonical-pair features from query tokens.

    Args:
        tokens: Long tensor shaped [B, L].

    Returns:
        Float tensor shaped [B, L, L, 4] with channels:
        AU/UA, GC/CG, GU/UG, any canonical or wobble pair.
    """
    if tokens.ndim != 2:
        raise ValueError(f"tokens must have shape [B, L], got {tuple(tokens.shape)}")

    t_i = tokens.long()[:, :, None]
    t_j = tokens.long()[:, None, :]

    au = ((t_i == A_IDX) & (t_j == U_IDX)) | ((t_i == U_IDX) & (t_j == A_IDX))
    gc = ((t_i == G_IDX) & (t_j == C_IDX)) | ((t_i == C_IDX) & (t_j == G_IDX))
    gu = ((t_i == G_IDX) & (t_j == U_IDX)) | ((t_i == U_IDX) & (t_j == G_IDX))
    any_pair = au | gc | gu

    return torch.stack([au, gc, gu, any_pair], dim=-1).float()


def normalize_bpp(bpp: torch.Tensor, eps: float = 1e-4) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetrize BPP and return probability plus logit features."""
    if bpp.ndim != 3:
        raise ValueError(f"bpp must have shape [B, L, L], got {tuple(bpp.shape)}")
    if bpp.shape[-1] != bpp.shape[-2]:
        raise ValueError(f"bpp must be square, got {tuple(bpp.shape)}")

    bpp = bpp.float()
    bpp = 0.5 * (bpp + bpp.transpose(-1, -2))
    bpp = bpp.clamp(0.0, 1.0)
    bpp_logit = torch.logit(bpp.clamp(eps, 1.0 - eps))
    return bpp, bpp_logit


def build_ss_features(
    bpp: torch.Tensor,
    canonical: torch.Tensor,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Build SS adapter features.

    Channels are:
    bpp, logit_bpp, AU, GC, GU, canonical_any, bpp * canonical_any.
    """
    if canonical.ndim != 4 or canonical.shape[-1] != 4:
        raise ValueError(
            "canonical must have shape [B, L, L, 4], "
            f"got {tuple(canonical.shape)}"
        )

    bpp, bpp_logit = normalize_bpp(bpp, eps)
    if canonical.shape[:3] != bpp.shape:
        raise ValueError(
            "canonical and bpp shapes are incompatible: "
            f"{tuple(canonical.shape)} vs {tuple(bpp.shape)}"
        )

    canonical_any = canonical[..., 3:4]
    return torch.cat(
        [
            bpp[..., None],
            bpp_logit[..., None],
            canonical[..., 0:1],
            canonical[..., 1:2],
            canonical[..., 2:3],
            canonical_any,
            bpp[..., None] * canonical_any,
        ],
        dim=-1,
    )
