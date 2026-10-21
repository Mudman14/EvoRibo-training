"""ESMFold-style RESM/BPP-conditioned pair encoder."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .config import SSPairEncoderConfig
from .features import (
    SS_FEATURE_DIM,
    build_ss_features,
    canonical_pair_features,
    normalize_bpp,
)
from .modules import (
    GatedSelfAttention,
    PairToSequence,
    RelativePosition,
    ResidueMLP,
    SSBiasAdapter,
    SSPairGateAdapter,
    SequenceToPair,
    SharedDropout,
    TriangleAttention,
    TriangleMultiplication,
)


class RESMStateEmbedder(nn.Module):
    """Project RESM single/pair priors and BPP features into trunk states."""

    def __init__(self, cfg: SSPairEncoderConfig):
        super().__init__()
        self.cfg = cfg
        pair_in_dim = cfg.resm_pair_dim + (SS_FEATURE_DIM if cfg.use_bpp_level1 else 0)

        self.single_mlp = nn.Sequential(
            nn.LayerNorm(cfg.resm_single_dim),
            nn.Linear(cfg.resm_single_dim, cfg.d_s),
            nn.GELU(),
            nn.Linear(cfg.d_s, cfg.d_s),
        )
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.d_s, padding_idx=cfg.pad_idx)
        self.pair_mlp = nn.Sequential(
            nn.LayerNorm(pair_in_dim),
            nn.Linear(pair_in_dim, cfg.d_z),
            nn.GELU(),
            nn.Linear(cfg.d_z, cfg.d_z),
        )

    def forward(
        self,
        resm_single: torch.Tensor,
        resm_pair: torch.Tensor,
        bpp: torch.Tensor,
        tokens: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._validate_inputs(resm_single, resm_pair, bpp, tokens, mask)

        bpp, _ = normalize_bpp(bpp, self.cfg.bpp_eps)
        canonical = canonical_pair_features(tokens)
        ss_feat = build_ss_features(bpp, canonical, self.cfg.bpp_eps)

        token_ids = tokens.long().clamp(0, self.cfg.pad_idx)
        s = self.single_mlp(resm_single.float()) + self.token_embed(token_ids)

        if self.cfg.use_bpp_level1:
            pair_input = torch.cat([resm_pair.float(), ss_feat], dim=-1)
        else:
            pair_input = resm_pair.float()
        z = self.pair_mlp(pair_input)

        pair_mask = mask[:, :, None] & mask[:, None, :]
        s = s * mask[..., None].float()
        z = z * pair_mask[..., None].float()

        return s, z, bpp, canonical, ss_feat

    def _validate_inputs(self, resm_single, resm_pair, bpp, tokens, mask) -> None:
        if resm_single.ndim != 3:
            raise ValueError("resm_single must have shape [B, L, C]")
        if resm_pair.ndim != 4:
            raise ValueError("resm_pair must have shape [B, L, L, C]")
        if bpp.ndim != 3:
            raise ValueError("bpp must have shape [B, L, L]")
        if tokens.ndim != 2 or mask.ndim != 2:
            raise ValueError("tokens and mask must have shape [B, L]")

        B, L, C_s = resm_single.shape
        if C_s != self.cfg.resm_single_dim:
            raise ValueError(f"expected resm_single dim {self.cfg.resm_single_dim}, got {C_s}")
        if resm_pair.shape != (B, L, L, self.cfg.resm_pair_dim):
            raise ValueError(
                "resm_pair shape mismatch: expected "
                f"{(B, L, L, self.cfg.resm_pair_dim)}, got {tuple(resm_pair.shape)}"
            )
        if bpp.shape != (B, L, L):
            raise ValueError(f"bpp shape mismatch: expected {(B, L, L)}, got {tuple(bpp.shape)}")
        if tokens.shape != (B, L) or mask.shape != (B, L):
            raise ValueError("tokens/mask shape mismatch with resm_single")


class SSAwareFoldingBlock(nn.Module):
    """Single/pair trunk block with learnable BPP attention bias and gates."""

    def __init__(self, cfg: SSPairEncoderConfig):
        super().__init__()
        if cfg.d_s % cfg.sequence_head_width != 0:
            raise ValueError("d_s must be divisible by sequence_head_width")
        if cfg.d_z % cfg.pairwise_head_width != 0:
            raise ValueError("d_z must be divisible by pairwise_head_width")

        self.cfg = cfg
        num_seq_heads = cfg.d_s // cfg.sequence_head_width

        self.seq_norm = nn.LayerNorm(cfg.d_s)
        self.pair_to_sequence = PairToSequence(cfg.d_z, num_seq_heads)
        self.seq_attention = GatedSelfAttention(
            embed_dim=cfg.d_s,
            num_heads=num_seq_heads,
            head_width=cfg.sequence_head_width,
            gated=True,
        )
        self.seq_mlp = ResidueMLP(cfg.d_s, 4 * cfg.d_s, cfg.dropout)
        self.sequence_to_pair = SequenceToPair(cfg.d_s, cfg.d_z // 2, cfg.d_z)

        self.tri_mul_out = TriangleMultiplication(cfg.d_z, direction="outgoing")
        self.tri_mul_in = TriangleMultiplication(cfg.d_z, direction="incoming")
        self.tri_att_row = TriangleAttention(
            cfg.d_z,
            cfg.pairwise_head_width,
            wise="row",
            chunk_size=cfg.triangle_attention_chunk_size,
        )
        self.tri_att_col = TriangleAttention(
            cfg.d_z,
            cfg.pairwise_head_width,
            wise="col",
            chunk_size=cfg.triangle_attention_chunk_size,
        )
        self.pair_mlp = ResidueMLP(cfg.d_z, 4 * cfg.d_z, cfg.dropout)

        self.ss_bias_adapter = SSBiasAdapter(
            ss_feat_dim=SS_FEATURE_DIM,
            num_heads=num_seq_heads,
            hidden_dim=cfg.ss_bias_hidden,
            scale_init=cfg.ss_bias_scale_init,
        )
        self.ss_gate_adapter = SSPairGateAdapter(
            d_z=cfg.d_z,
            ss_feat_dim=SS_FEATURE_DIM,
            hidden_dim=cfg.ss_gate_hidden,
            scale_init=cfg.ss_gate_scale_init,
            amplitude=cfg.ss_gate_amplitude,
        )

        self.drop = nn.Dropout(cfg.dropout)
        self.row_drop = SharedDropout(cfg.dropout * 2.0, share_dims=2)
        self.col_drop = SharedDropout(cfg.dropout * 2.0, share_dims=1)

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        ss_feat: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pair_mask = mask[:, :, None] & mask[:, None, :]

        pair_bias = self.pair_to_sequence(z)
        if self.cfg.use_bpp_level2_bias:
            pair_bias = pair_bias + self.ss_bias_adapter(ss_feat, pair_mask)

        y = self.seq_norm(s)
        y, _ = self.seq_attention(y, mask=mask, bias=pair_bias)
        s = s + self.drop(y)
        s = self.seq_mlp(s)
        s = s * mask[..., None].float()

        z = z + self.sequence_to_pair(s)
        z = z * pair_mask[..., None].float()

        if self.cfg.use_bpp_level3_gate:
            gates = self.ss_gate_adapter(z, ss_feat, pair_mask)
        else:
            gates = z.new_ones(*z.shape[:3], 4)

        z = z + self.row_drop(gates[..., 0:1] * self.tri_mul_out(z, pair_mask))
        z = z + self.col_drop(gates[..., 1:2] * self.tri_mul_in(z, pair_mask))
        z = z + self.row_drop(gates[..., 2:3] * self.tri_att_row(z, pair_mask))
        z = z + self.col_drop(gates[..., 3:4] * self.tri_att_col(z, pair_mask))
        z = self.pair_mlp(z)
        z = z * pair_mask[..., None].float()

        return s, z


class EvoNASSPairEncoder(nn.Module):
    """Independent RESM + BPP encoder producing a directed pair representation."""

    def __init__(self, cfg: SSPairEncoderConfig | None = None):
        super().__init__()
        self.cfg = cfg or SSPairEncoderConfig()
        if self.cfg.num_recycles < 1:
            raise ValueError("num_recycles must be >= 1")

        self.embedder = RESMStateEmbedder(self.cfg)
        self.relpos = RelativePosition(self.cfg.position_bins, self.cfg.d_z)
        self.recycle_s_norm = nn.LayerNorm(self.cfg.d_s)
        self.recycle_z_norm = nn.LayerNorm(self.cfg.d_z)
        self.blocks = nn.ModuleList(
            [SSAwareFoldingBlock(self.cfg) for _ in range(self.cfg.num_blocks)]
        )

    def forward(
        self,
        resm_single: torch.Tensor,
        resm_pair: torch.Tensor,
        bpp: torch.Tensor,
        tokens: torch.Tensor,
        mask: torch.Tensor | None = None,
        residx: torch.Tensor | None = None,
        num_recycles: int | None = None,
        return_pair_sym: bool | None = None,
        return_intermediates: bool = False,
    ) -> dict[str, torch.Tensor | list[dict[str, torch.Tensor]]]:
        B, L, _ = resm_single.shape
        device = resm_single.device

        if mask is None:
            mask = torch.ones(B, L, device=device, dtype=torch.bool)
        else:
            mask = mask.to(device=device).bool()

        tokens = tokens.to(device=device)
        bpp = bpp.to(device=device)
        resm_pair = resm_pair.to(device=device)

        if residx is None:
            residx = torch.arange(L, device=device, dtype=torch.long).expand(B, L)
        else:
            residx = residx.to(device=device, dtype=torch.long)

        recycles = self.cfg.num_recycles if num_recycles is None else int(num_recycles)
        if recycles < 1:
            raise ValueError("num_recycles must be >= 1")

        s0, z0, bpp, canonical, ss_feat = self.embedder(
            resm_single, resm_pair, bpp, tokens, mask
        )

        pair_mask = mask[:, :, None] & mask[:, None, :]
        recycle_s = torch.zeros_like(s0)
        recycle_z = torch.zeros_like(z0)
        intermediates: list[dict[str, torch.Tensor]] = []

        s = s0
        z = z0
        for _ in range(recycles):
            s = s0 + self.recycle_s_norm(recycle_s.detach())
            z = z0 + self.recycle_z_norm(recycle_z.detach())
            z = z + self.relpos(residx, mask)
            z = z * pair_mask[..., None].float()

            use_checkpoint = (
                self.training
                and self.cfg.use_gradient_checkpointing
                and torch.is_grad_enabled()
                and not return_intermediates
            )
            for block in self.blocks:
                if use_checkpoint:
                    s, z = checkpoint(
                        block,
                        s,
                        z,
                        ss_feat,
                        mask,
                        use_reentrant=False,
                    )
                else:
                    s, z = block(s, z, ss_feat, mask)

            recycle_s = s
            recycle_z = z

            if return_intermediates:
                intermediates.append({"single": s, "pair": z})

        out: dict[str, torch.Tensor | list[dict[str, torch.Tensor]]] = {
            "single": s,
            "pair": z,
            "pair_mask": pair_mask,
            "bpp": bpp,
            "canonical": canonical,
        }

        want_pair_sym = self.cfg.return_pair_sym if return_pair_sym is None else return_pair_sym
        if want_pair_sym:
            out["pair_sym"] = 0.5 * (z + z.transpose(1, 2))

        if return_intermediates:
            out["intermediates"] = intermediates

        return out
