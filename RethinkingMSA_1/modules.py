"""Self-contained trunk modules for the RethinkingMSA encoder."""

from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F


class SharedDropout(nn.Module):
    """Dropout with a mask shared along selected dimensions."""

    def __init__(self, p: float, share_dims: int | Iterable[int]):
        super().__init__()
        self.p = float(p)
        if isinstance(share_dims, int):
            self.share_dims = [share_dims]
        else:
            self.share_dims = list(share_dims)
        self.dropout = nn.Dropout(self.p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        shape = list(x.shape)
        for dim in self.share_dims:
            shape[dim] = 1
        mask = x.new_ones(shape)
        return x * self.dropout(mask)


class RelativePosition(nn.Module):
    """ESMFold-style clipped relative positional embedding for pair states."""

    def __init__(self, bins: int, d_z: int):
        super().__init__()
        self.bins = int(bins)
        self.embedding = nn.Embedding(2 * self.bins + 2, d_z)

    def forward(
        self,
        residx: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if residx.ndim != 2 or residx.dtype != torch.long:
            raise ValueError("residx must be a long tensor with shape [B, L]")

        diff = residx[:, None, :] - residx[:, :, None]
        diff = diff.clamp(-self.bins, self.bins)
        diff = diff + self.bins + 1

        if mask is not None:
            pair_mask = mask.bool()[:, :, None] & mask.bool()[:, None, :]
            diff = diff.masked_fill(~pair_mask, 0)

        return self.embedding(diff)


class GatedSelfAttention(nn.Module):
    """Multi-head self-attention with optional pairwise attention bias."""

    def __init__(self, embed_dim: int, num_heads: int, head_width: int, gated: bool = True):
        super().__init__()
        if embed_dim != num_heads * head_width:
            raise ValueError("embed_dim must equal num_heads * head_width")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_width = head_width
        self.scale = head_width ** -0.5

        self.proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        self.gated = gated

        if gated:
            self.g_proj = nn.Linear(embed_dim, embed_dim)
            nn.init.zeros_(self.g_proj.weight)
            nn.init.ones_(self.g_proj.bias)

        nn.init.zeros_(self.o_proj.weight)
        nn.init.zeros_(self.o_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"x must have shape [B, L, C], got {tuple(x.shape)}")

        B, L, C = x.shape
        qkv = self.proj(x).view(B, L, 3, self.num_heads, self.head_width)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        logits = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale

        if bias is not None:
            if bias.shape != (B, L, L, self.num_heads):
                raise ValueError(
                    "bias must have shape [B, L, L, H], "
                    f"got {tuple(bias.shape)}"
                )
            logits = logits + bias.permute(0, 3, 1, 2)

        if mask is not None:
            key_mask = mask.bool()[:, None, None, :]
            logits = logits.masked_fill(~key_mask, -1e4)

        attn = F.softmax(logits, dim=-1)
        y = torch.einsum("bhij,bhjd->bhid", attn, v)
        y = y.transpose(1, 2).contiguous().view(B, L, C)

        if self.gated:
            y = torch.sigmoid(self.g_proj(x)) * y

        y = self.o_proj(y)

        if mask is not None:
            y = y * mask[..., None].float()

        return y, attn.permute(0, 2, 3, 1)


class SequenceToPair(nn.Module):
    """ESMFold-style single-to-pair update using product and difference."""

    def __init__(self, d_s: int, inner_dim: int, d_z: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_s)
        self.proj = nn.Linear(d_s, 2 * inner_dim)
        self.out = nn.Linear(2 * inner_dim, d_z)

        nn.init.zeros_(self.proj.bias)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        if s.ndim != 3:
            raise ValueError(f"s must have shape [B, L, C], got {tuple(s.shape)}")
        q, k = self.proj(self.norm(s)).chunk(2, dim=-1)
        prod = q[:, None, :, :] * k[:, :, None, :]
        diff = q[:, None, :, :] - k[:, :, None, :]
        return self.out(torch.cat([prod, diff], dim=-1))


class PairToSequence(nn.Module):
    """Project pair state to per-head sequence-attention bias."""

    def __init__(self, d_z: int, num_heads: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_z)
        self.linear = nn.Linear(d_z, num_heads, bias=False)
        nn.init.zeros_(self.linear.weight)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 4:
            raise ValueError(f"z must have shape [B, L, L, C], got {tuple(z.shape)}")
        return self.linear(self.norm(z))


class ResidueMLP(nn.Module):
    """Residual feed-forward network over the last dimension."""

    def __init__(self, embed_dim: int, inner_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, embed_dim),
            nn.Dropout(dropout),
        )
        nn.init.zeros_(self.net[-2].weight)
        nn.init.zeros_(self.net[-2].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class TriangleMultiplication(nn.Module):
    """trRosettaRNA2-style outgoing/incoming triangle multiplication."""

    def __init__(self, d_z: int, hidden_dim: int | None = None, direction: str = "outgoing"):
        super().__init__()
        if direction not in {"outgoing", "incoming"}:
            raise ValueError("direction must be 'outgoing' or 'incoming'")

        hidden_dim = hidden_dim or d_z
        self.direction = direction
        self.norm = nn.LayerNorm(d_z)
        self.linear_ab = nn.Linear(d_z, 2 * hidden_dim)
        self.linear_gate_ab = nn.Linear(d_z, 2 * hidden_dim)
        self.output_gate = nn.Linear(d_z, d_z)
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, d_z)

        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, z: torch.Tensor, pair_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.norm(z)
        a, b = (self.linear_ab(x) * torch.sigmoid(self.linear_gate_ab(x))).chunk(2, dim=-1)

        if pair_mask is not None:
            mask = pair_mask[..., None].float()
            a = a * mask
            b = b * mask

        if self.direction == "outgoing":
            update = torch.einsum("bikd,bjkd->bijd", a, b)
        else:
            update = torch.einsum("bkid,bkjd->bijd", a, b)

        update = torch.sigmoid(self.output_gate(x)) * self.out(self.out_norm(update))

        if pair_mask is not None:
            update = update * pair_mask[..., None].float()

        return update


class TriangleAttention(nn.Module):
    """Row/column triangle attention without OpenFold dependencies."""

    def __init__(self, d_z: int, head_width: int, wise: str, chunk_size: int = 0):
        super().__init__()
        if wise not in {"row", "col"}:
            raise ValueError("wise must be 'row' or 'col'")
        if d_z % head_width != 0:
            raise ValueError("d_z must be divisible by head_width")

        self.wise = wise
        self.chunk_size = max(0, int(chunk_size))
        self.num_heads = d_z // head_width
        self.head_width = head_width
        self.norm = nn.LayerNorm(d_z)
        self.to_qkv = nn.Linear(d_z, 3 * d_z, bias=False)
        self.pair_bias = nn.Linear(d_z, self.num_heads, bias=False)
        self.gate = nn.Linear(d_z, d_z)
        self.out = nn.Linear(d_z, d_z)

        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def _row_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bias: torch.Tensor,
        pair_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        B, L, _, H, D = q.shape
        if self.chunk_size <= 0 or self.chunk_size >= L:
            logits = torch.einsum("brihd,brjhd->brijh", q, k) / math.sqrt(D)
            logits = logits + bias[:, None, :, :, :]
            if pair_mask is not None:
                key_mask = pair_mask[:, :, None, :]
                logits = logits.masked_fill(~key_mask[..., None], -1e4)
            attn = F.softmax(logits.float(), dim=3).to(v.dtype)
            return torch.einsum("brijh,brjhd->brihd", attn, v)

        out = q.new_empty(B, L, L, H, D)
        for start in range(0, L, self.chunk_size):
            end = min(start + self.chunk_size, L)
            q_chunk = q[:, :, start:end, :, :]
            logits = torch.einsum("brihd,brjhd->brijh", q_chunk, k) / math.sqrt(D)
            logits = logits + bias[:, None, start:end, :, :]
            if pair_mask is not None:
                key_mask = pair_mask[:, :, None, :]
                logits = logits.masked_fill(~key_mask[..., None], -1e4)
            attn = F.softmax(logits.float(), dim=3).to(v.dtype)
            out[:, :, start:end, :, :] = torch.einsum("brijh,brjhd->brihd", attn, v)
        return out

    def _col_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bias: torch.Tensor,
        pair_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        B, L, _, H, D = q.shape
        if self.chunk_size <= 0 or self.chunk_size >= L:
            logits = torch.einsum("bilhd,bjlhd->bijlh", q, k) / math.sqrt(D)
            logits = logits + bias[:, :, :, None, :]
            if pair_mask is not None:
                key_mask = pair_mask[:, None, :, :]
                logits = logits.masked_fill(~key_mask[..., None], -1e4)
            attn = F.softmax(logits.float(), dim=2).to(v.dtype)
            return torch.einsum("bijlh,bjlhd->bilhd", attn, v)

        out = q.new_empty(B, L, L, H, D)
        for start in range(0, L, self.chunk_size):
            end = min(start + self.chunk_size, L)
            q_chunk = q[:, start:end, :, :, :]
            logits = torch.einsum("bilhd,bjlhd->bijlh", q_chunk, k) / math.sqrt(D)
            logits = logits + bias[:, start:end, :, None, :]
            if pair_mask is not None:
                key_mask = pair_mask[:, None, :, :]
                logits = logits.masked_fill(~key_mask[..., None], -1e4)
            attn = F.softmax(logits.float(), dim=2).to(v.dtype)
            out[:, start:end, :, :, :] = torch.einsum("bijlh,bjlhd->bilhd", attn, v)
        return out

    def forward(self, z: torch.Tensor, pair_mask: torch.Tensor | None = None) -> torch.Tensor:
        if z.ndim != 4:
            raise ValueError(f"z must have shape [B, L, L, C], got {tuple(z.shape)}")

        x = self.norm(z)
        B, L, _, C = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q = q.view(B, L, L, self.num_heads, self.head_width)
        k = k.view(B, L, L, self.num_heads, self.head_width)
        v = v.view(B, L, L, self.num_heads, self.head_width)
        bias = self.pair_bias(x)

        if self.wise == "row":
            out = self._row_attention(q, k, v, bias, pair_mask)
        else:
            out = self._col_attention(q, k, v, bias, pair_mask)

        out = out.contiguous().view(B, L, L, C)
        out = torch.sigmoid(self.gate(x)) * out
        out = self.out(out)

        if pair_mask is not None:
            out = out * pair_mask[..., None].float()

        return out


class SSBiasAdapter(nn.Module):
    """Learnable BPP/query bias added to sequence attention logits."""

    def __init__(self, ss_feat_dim: int, num_heads: int, hidden_dim: int, scale_init: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(ss_feat_dim),
            nn.Linear(ss_feat_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_heads),
        )
        self.scale_logit = nn.Parameter(torch.tensor(float(scale_init)))

        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        ss_feat: torch.Tensor,
        pair_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bias = torch.sigmoid(self.scale_logit) * self.net(ss_feat)
        if pair_mask is not None:
            bias = bias * pair_mask[..., None].float()
        return bias


class SSPairGateAdapter(nn.Module):
    """Learnable BPP/query gates for the four triangle pair updates."""

    def __init__(
        self,
        d_z: int,
        ss_feat_dim: int,
        hidden_dim: int,
        scale_init: float,
        amplitude: float,
    ):
        super().__init__()
        self.z_proj = nn.Sequential(
            nn.LayerNorm(d_z),
            nn.Linear(d_z, 8),
            nn.SiLU(),
        )
        self.net = nn.Sequential(
            nn.LayerNorm(ss_feat_dim + 8),
            nn.Linear(ss_feat_dim + 8, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 4),
        )
        self.scale_logit = nn.Parameter(torch.tensor(float(scale_init)))
        self.amplitude = float(amplitude)

        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        z: torch.Tensor,
        ss_feat: torch.Tensor,
        pair_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raw = self.net(torch.cat([ss_feat, self.z_proj(z)], dim=-1))
        scale = torch.sigmoid(self.scale_logit)
        gate = 1.0 + self.amplitude * scale * torch.tanh(raw)

        if pair_mask is not None:
            gate = gate * pair_mask[..., None].float()

        return gate
