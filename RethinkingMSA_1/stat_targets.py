"""adabmDCA-style RNA statistics targets for RethinkingMSA.

The training target convention in this module is intentionally solver-facing:

* alphabet order is ``-ACGU``;
* off-diagonal entries store ``fij[i, a, j, b]`` flattened as ``a * 5 + b``;
* diagonal entries store ``diag(fi[i])`` flattened the same way.

Those rules make the target reversible into eaDCA/adabmDCA-style ``fi/fij`` and
mi3-GPU upper-triangle bivariate marginals.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


STATS_TOKENS = "-ACGU"
STATS_Q = len(STATS_TOKENS)
STATS_TOKEN_TO_IDX = {token: idx for idx, token in enumerate(STATS_TOKENS)}
VALID_STATS_TOKENS = set(STATS_TOKENS)
EPS = 1e-8


@dataclass(frozen=True)
class StatsTargetResult:
    fi: np.ndarray
    fij: np.ndarray
    stats_map: np.ndarray
    scalar_map: np.ndarray
    scalar_scale: float
    meff: float
    pseudocount: float
    raw_depth: int
    equal_length_depth: int
    valid_depth: int
    used_depth: int
    unique_depth: int
    length: int
    tokens: str = STATS_TOKENS

    def meta(self) -> dict[str, object]:
        data = asdict(self)
        for key in ("fi", "fij", "stats_map", "scalar_map"):
            data.pop(key)
        return data


def clean_alignment_sequence(seq: str) -> tuple[str, bool]:
    """Remove A3M lowercase insertions and validate uppercase RNA tokens."""
    chars: list[str] = []
    valid = True
    for ch in seq:
        if ch.islower():
            continue
        c = ch.upper()
        if c == "T":
            c = "U"
        if c in VALID_STATS_TOKENS:
            chars.append(c)
        else:
            valid = False
    return "".join(chars), valid


def read_clean_a3m_sequences(
    msa_path: str | Path,
    expected_length: int | None = None,
    max_seqs: int = 0,
) -> tuple[list[str], dict[str, int]]:
    """Read valid equal-length RNA alignment rows from FASTA/A3M."""
    msa_path = Path(msa_path)
    records: list[tuple[str, bool]] = []
    current: list[str] = []
    seen_header = False
    with msa_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seen_header:
                    records.append(clean_alignment_sequence("".join(current)))
                    current = []
                seen_header = True
            elif seen_header:
                current.append(line)
        if seen_header:
            records.append(clean_alignment_sequence("".join(current)))
    if not records:
        raise ValueError(f"No FASTA/A3M records found in {msa_path}")

    query_length = expected_length if expected_length is not None else len(records[0][0])
    if query_length <= 0:
        raise ValueError(f"{msa_path}: query alignment length is empty after cleaning")

    equal_length = [seq for seq, _ in records if len(seq) == query_length]
    valid_equal = [seq for seq, ok in records if ok and len(seq) == query_length]
    used = valid_equal[:max_seqs] if max_seqs and max_seqs > 0 else valid_equal
    return used, {
        "raw_depth": len(records),
        "equal_length_depth": len(equal_length),
        "valid_depth": len(valid_equal),
        "used_depth": len(used),
        "length": int(query_length),
    }


def deduplicate_preserve_order(seqs: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for seq in seqs:
        if seq in seen:
            continue
        seen.add(seq)
        unique.append(seq)
    return unique


def sequences_to_numeric(seqs: list[str]) -> np.ndarray:
    if not seqs:
        raise ValueError("No sequences provided")
    length = len(seqs[0])
    arr = np.empty((len(seqs), length), dtype=np.int16)
    for row, seq in enumerate(seqs):
        if len(seq) != length:
            raise ValueError(f"Sequence length mismatch: expected {length}, got {len(seq)}")
        try:
            arr[row] = [STATS_TOKEN_TO_IDX[ch] for ch in seq]
        except KeyError as exc:
            raise ValueError(f"Invalid RNA token {exc} in alignment row") from exc
    return arr


def compute_adabm_weights(
    msa_num: np.ndarray,
    clustering_seqid: float = 0.8,
    chunk_size: int = 512,
) -> np.ndarray:
    """Return adabmDCA-style sequence weights ``1 / cluster_size``."""
    if msa_num.ndim != 2:
        raise ValueError(f"msa_num must be [N, L], got {msa_num.shape}")
    nseq, length = msa_num.shape
    if nseq == 0:
        raise ValueError("Cannot compute weights for an empty MSA")
    if not 0.0 <= clustering_seqid <= 1.0:
        raise ValueError("clustering_seqid must be in [0, 1]")
    counts = np.zeros(nseq, dtype=np.float64)
    for start in range(0, nseq, chunk_size):
        block = msa_num[start : start + chunk_size]
        identity = (block[:, None, :] == msa_num[None, :, :]).mean(axis=2)
        counts[start : start + chunk_size] = np.sum(identity > clustering_seqid, axis=1)
    counts = np.maximum(counts, 1.0)
    return (1.0 / counts).astype(np.float64)


def compute_fi_fij(
    msa_num: np.ndarray,
    weights: np.ndarray | None = None,
    pseudocount: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Compute weighted adabmDCA-style ``fi`` and ``fij`` with flat pseudocount."""
    if msa_num.ndim != 2:
        raise ValueError(f"msa_num must be [N, L], got {msa_num.shape}")
    nseq, length = msa_num.shape
    if nseq < 1:
        raise ValueError("Need at least one sequence to compute statistics")

    if weights is None:
        weights = np.ones(nseq, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (nseq,):
        raise ValueError(f"weights must have shape [{nseq}], got {weights.shape}")
    meff = float(weights.sum())
    if not np.isfinite(meff) or meff <= 0:
        raise ValueError("Meff must be positive and finite")

    onehot = np.eye(STATS_Q, dtype=np.float64)[msa_num.astype(np.int64)]
    weighted = onehot * weights[:, None, None]
    fi = weighted.sum(axis=0) / meff
    flat = onehot.reshape(nseq, length * STATS_Q)
    weighted_flat = flat * weights[:, None]
    fij = (weighted_flat.T @ flat) / meff
    fij = fij.reshape(length, STATS_Q, length, STATS_Q)

    pc = float(1.0 / meff if pseudocount is None else pseudocount)
    pc = min(max(pc, 0.0), 1.0)
    fi = (1.0 - pc) * fi + pc / STATS_Q
    fij = (1.0 - pc) * fij + pc / (STATS_Q * STATS_Q)
    idx = np.arange(length)
    fij[idx, :, idx, :] = 0.0
    for i in range(length):
        fij[i, :, i, :] = np.diag(fi[i])
    return fi.astype(np.float32), fij.astype(np.float32), meff, pc


def fij_to_stats_map(fi: np.ndarray, fij: np.ndarray) -> np.ndarray:
    fi = np.asarray(fi, dtype=np.float32)
    fij = np.asarray(fij, dtype=np.float32)
    if fi.ndim != 2 or fi.shape[1] != STATS_Q:
        raise ValueError(f"fi must be [L, 5], got {fi.shape}")
    length = fi.shape[0]
    if fij.shape != (length, STATS_Q, length, STATS_Q):
        raise ValueError(f"fij shape {fij.shape} is incompatible with fi {fi.shape}")
    stats = np.empty((length, length, STATS_Q * STATS_Q), dtype=np.float32)
    for i in range(length):
        for j in range(length):
            if i == j:
                stats[i, j] = np.diag(fi[i]).reshape(-1)
            else:
                stats[i, j] = fij[i, :, j, :].reshape(-1)
    return stats


def fij_to_pair_stats_map(fij: np.ndarray) -> np.ndarray:
    """Return off-diagonal ``fij`` as ``[L, L, 25]`` without encoding ``fi`` on diagonal."""
    fij = np.asarray(fij, dtype=np.float32)
    if fij.ndim != 4 or fij.shape[0] != fij.shape[2] or fij.shape[1] != STATS_Q or fij.shape[3] != STATS_Q:
        raise ValueError(f"fij must be [L, 5, L, 5], got {fij.shape}")
    length = fij.shape[0]
    stats = fij.transpose(0, 2, 1, 3).reshape(length, length, STATS_Q * STATS_Q).copy()
    idx = np.arange(length)
    stats[idx, idx] = 0.0
    return stats.astype(np.float32)


def stats_map_to_fi_fij(stats_map: np.ndarray, eps: float = EPS) -> tuple[np.ndarray, np.ndarray]:
    stats = np.asarray(stats_map, dtype=np.float32)
    if stats.ndim != 3 or stats.shape[0] != stats.shape[1] or stats.shape[2] != STATS_Q * STATS_Q:
        raise ValueError(f"stats_map must be [L, L, 25], got {stats.shape}")
    length = stats.shape[0]
    blocks = stats.reshape(length, length, STATS_Q, STATS_Q)
    fi = np.empty((length, STATS_Q), dtype=np.float32)
    for i in range(length):
        diag = np.diag(blocks[i, i]).astype(np.float64)
        diag = np.maximum(diag, eps)
        fi[i] = (diag / diag.sum()).astype(np.float32)

    fij = blocks.transpose(0, 2, 1, 3).astype(np.float32, copy=True)
    for i in range(length):
        fij[i, :, i, :] = np.diag(fi[i])
    return fi, fij


def export_mi3_bimarg(stats_map: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Export upper-triangle bivariate marginals as mi3-compatible ``[P, 25]``."""
    stats = np.asarray(stats_map, dtype=np.float64)
    if stats.ndim != 3 or stats.shape[0] != stats.shape[1] or stats.shape[2] != STATS_Q * STATS_Q:
        raise ValueError(f"stats_map must be [L, L, 25], got {stats.shape}")
    idx = np.triu_indices(stats.shape[0], k=1)
    bimarg = np.maximum(stats[idx[0], idx[1]], eps)
    bimarg /= bimarg.sum(axis=1, keepdims=True)
    return bimarg.astype("<f4", copy=False)


def normalize_target_map(target: np.ndarray, eps: float = EPS) -> tuple[np.ndarray, float]:
    target = np.asarray(target, dtype=np.float32)
    if target.ndim != 2 or target.shape[0] != target.shape[1]:
        raise ValueError(f"target map must be [L, L], got {target.shape}")
    upper = np.triu(np.ones(target.shape, dtype=bool), k=1)
    scale = float(np.max(np.abs(target[upper]))) if upper.any() else 0.0
    if not np.isfinite(scale) or scale < eps:
        scale = 1.0
    return (target / scale).astype(np.float32), scale


def mi_apc_from_fij(fij: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    fij = np.asarray(fij, dtype=np.float64)
    if fij.ndim != 4 or fij.shape[0] != fij.shape[2] or fij.shape[1] != STATS_Q or fij.shape[3] != STATS_Q:
        raise ValueError(f"fij must be [L, 5, L, 5], got {fij.shape}")
    length = fij.shape[0]
    mi = np.zeros((length, length), dtype=np.float64)
    for i in range(length):
        for j in range(length):
            if i == j:
                continue
            joint = np.maximum(fij[i, :, j, :], 0.0)
            total = float(joint.sum())
            if total <= eps:
                continue
            joint = joint / total
            fi = joint.sum(axis=1, keepdims=True)
            fj = joint.sum(axis=0, keepdims=True)
            denom = fi @ fj
            valid = joint > eps
            mi[i, j] = float(np.sum(joint[valid] * np.log2(joint[valid] / np.maximum(denom[valid], eps))))
    mi = np.maximum(mi, 0.0)
    np.fill_diagonal(mi, 0.0)
    if length <= 1:
        return mi.astype(np.float32)
    total = float(mi.sum() / max(length * (length - 1), 1))
    if total <= eps:
        return np.zeros_like(mi, dtype=np.float32)
    row = mi.sum(axis=1) / max(length - 1, 1)
    apc = mi - np.outer(row, row) / total
    apc = np.maximum(apc, 0.0)
    np.fill_diagonal(apc, 0.0)
    return apc.astype(np.float32)


def compute_adabm_stats_target(
    msa_path: str | Path,
    expected_length: int | None = None,
    target_max_seqs: int = 0,
    clustering_seqid: float = 0.8,
    remove_duplicates: bool = True,
    no_reweighting: bool = False,
    pseudocount: float | None = None,
    normalize_scalar: bool = True,
) -> StatsTargetResult:
    seqs, report = read_clean_a3m_sequences(msa_path, expected_length=expected_length, max_seqs=target_max_seqs)
    if remove_duplicates:
        seqs = deduplicate_preserve_order(seqs)
    if len(seqs) < 1:
        length = int(expected_length or report["length"])
        fi = np.full((length, STATS_Q), 1.0 / STATS_Q, dtype=np.float32)
        fij = np.full((length, STATS_Q, length, STATS_Q), 1.0 / (STATS_Q * STATS_Q), dtype=np.float32)
        for i in range(length):
            fij[i, :, i, :] = np.diag(fi[i])
        meff = 1.0
        pc = 1.0
    else:
        msa_num = sequences_to_numeric(seqs)
        weights = None if no_reweighting else compute_adabm_weights(msa_num, clustering_seqid=clustering_seqid)
        fi, fij, meff, pc = compute_fi_fij(msa_num, weights=weights, pseudocount=pseudocount)
        length = int(msa_num.shape[1])

    stats_map = fij_to_stats_map(fi, fij)
    scalar_map = mi_apc_from_fij(fij)
    scalar_scale = 1.0
    if normalize_scalar:
        scalar_map, scalar_scale = normalize_target_map(scalar_map)
    return StatsTargetResult(
        fi=fi,
        fij=fij,
        stats_map=stats_map,
        scalar_map=scalar_map.astype(np.float32),
        scalar_scale=float(scalar_scale),
        meff=float(meff),
        pseudocount=float(pc),
        raw_depth=int(report["raw_depth"]),
        equal_length_depth=int(report["equal_length_depth"]),
        valid_depth=int(report["valid_depth"]),
        used_depth=int(report["used_depth"]),
        unique_depth=int(len(seqs)),
        length=int(length),
    )


def load_or_compute_adabm_stats_target(
    rna_id: str,
    msa_path: str | Path,
    expected_length: int,
    target_max_seqs: int,
    target_cache_dir: str | Path | None,
    clustering_seqid: float = 0.8,
    remove_duplicates: bool = True,
    no_reweighting: bool = False,
    pseudocount: float | None = None,
    normalize_scalar: bool = True,
) -> StatsTargetResult:
    if target_cache_dir is None:
        return compute_adabm_stats_target(
            msa_path,
            expected_length=expected_length,
            target_max_seqs=target_max_seqs,
            clustering_seqid=clustering_seqid,
            remove_duplicates=remove_duplicates,
            no_reweighting=no_reweighting,
            pseudocount=pseudocount,
            normalize_scalar=normalize_scalar,
        )
    target_cache_dir = Path(target_cache_dir)
    target_cache_dir.mkdir(parents=True, exist_ok=True)
    pc_tag = "auto" if pseudocount is None else f"{pseudocount:g}"
    cache_name = (
        f"{rna_id}_L{expected_length}_max{target_max_seqs}_adabm"
        f"_seqid{clustering_seqid:g}_dedup{int(remove_duplicates)}"
        f"_rew{int(not no_reweighting)}_pc{pc_tag}.npz"
    )
    cache_path = target_cache_dir / cache_name
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=False)
        meta = json.loads(str(data["meta_json"]))
        return StatsTargetResult(
            fi=data["fi"].astype(np.float32),
            fij=data["fij"].astype(np.float32),
            stats_map=data["stats_map"].astype(np.float32),
            scalar_map=data["scalar_map"].astype(np.float32),
            scalar_scale=float(meta["scalar_scale"]),
            meff=float(meta["meff"]),
            pseudocount=float(meta["pseudocount"]),
            raw_depth=int(meta["raw_depth"]),
            equal_length_depth=int(meta["equal_length_depth"]),
            valid_depth=int(meta["valid_depth"]),
            used_depth=int(meta["used_depth"]),
            unique_depth=int(meta["unique_depth"]),
            length=int(meta["length"]),
            tokens=str(meta.get("tokens", STATS_TOKENS)),
        )

    result = compute_adabm_stats_target(
        msa_path,
        expected_length=expected_length,
        target_max_seqs=target_max_seqs,
        clustering_seqid=clustering_seqid,
        remove_duplicates=remove_duplicates,
        no_reweighting=no_reweighting,
        pseudocount=pseudocount,
        normalize_scalar=normalize_scalar,
    )
    np.savez_compressed(
        cache_path,
        fi=result.fi,
        fij=result.fij,
        stats_map=result.stats_map,
        scalar_map=result.scalar_map,
        meta_json=json.dumps(result.meta()),
    )
    return result


def legalize_stats_map(stats_map: np.ndarray, eps: float = EPS, ipf_iters: int = 16) -> np.ndarray:
    """Project model-like stats onto positive normalized pair blocks with transpose symmetry."""
    stats = np.asarray(stats_map, dtype=np.float64)
    if stats.ndim != 3 or stats.shape[0] != stats.shape[1] or stats.shape[2] != STATS_Q * STATS_Q:
        raise ValueError(f"stats_map must be [L, L, 25], got {stats.shape}")
    length = stats.shape[0]
    blocks = np.maximum(stats.reshape(length, length, STATS_Q, STATS_Q), eps)
    blocks = 0.5 * (blocks + blocks.transpose(1, 0, 3, 2))

    fi = np.empty((length, STATS_Q), dtype=np.float64)
    for i in range(length):
        diag = np.maximum(np.diag(blocks[i, i]), eps)
        fi[i] = diag / diag.sum()

    for i in range(length):
        blocks[i, i] = np.diag(fi[i])
    for i in range(length):
        for j in range(i + 1, length):
            block = np.maximum(blocks[i, j], eps)
            for _ in range(max(0, ipf_iters)):
                block *= fi[i][:, None] / np.maximum(block.sum(axis=1, keepdims=True), eps)
                block *= fi[j][None, :] / np.maximum(block.sum(axis=0, keepdims=True), eps)
            block = np.maximum(block, eps)
            block /= block.sum()
            blocks[i, j] = block
            blocks[j, i] = block.T
    return blocks.reshape(length, length, STATS_Q * STATS_Q).astype(np.float32)


def transpose_flat25_torch(x: torch.Tensor) -> torch.Tensor:
    """Transpose the two 5-state axes encoded in a flat 25-channel tensor."""
    if x.shape[-1] != STATS_Q * STATS_Q:
        raise ValueError(f"last dimension must be 25, got {tuple(x.shape)}")
    return x.reshape(*x.shape[:-1], STATS_Q, STATS_Q).transpose(-1, -2).reshape_as(x)


def symmetrize_stats_logits(logits: torch.Tensor, pair_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Mirror upper-triangle 25-channel logits with ``a,b -> b,a`` channel transpose."""
    if logits.ndim != 4 or logits.shape[1] != logits.shape[2] or logits.shape[-1] != STATS_Q * STATS_Q:
        raise ValueError(f"logits must be [B, L, L, 25], got {tuple(logits.shape)}")
    mirrored = transpose_flat25_torch(logits.transpose(1, 2))
    out = 0.5 * (logits + mirrored)
    if pair_mask is not None:
        out = out * pair_mask[..., None].float()
    return out


def stats_kl_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    pair_mask: torch.Tensor,
    diag_weight: float = 0.2,
    pair_weights: torch.Tensor | None = None,
    eps: float = EPS,
) -> torch.Tensor:
    """Masked KL(target || softmax(logits)) over weighted upper-triangle pairs plus diagonal fi."""
    if logits.shape != target.shape:
        raise ValueError(f"logits/target mismatch: {tuple(logits.shape)} vs {tuple(target.shape)}")
    if logits.ndim != 4 or logits.shape[-1] != STATS_Q * STATS_Q:
        raise ValueError(f"logits must be [B, L, L, 25], got {tuple(logits.shape)}")
    B, L, _, _ = logits.shape
    upper = torch.triu(torch.ones(L, L, device=logits.device, dtype=torch.bool), diagonal=1)
    diag = torch.eye(L, device=logits.device, dtype=torch.bool)
    valid_pair = pair_mask.bool() & pair_mask.bool().transpose(-1, -2)
    weights = upper[None].float() * valid_pair.float()
    if pair_weights is not None:
        if pair_weights.shape != (B, L, L):
            raise ValueError(f"pair_weights must be [B, L, L], got {tuple(pair_weights.shape)}")
        weights = weights * pair_weights.float().clamp_min(0.0)
    if diag_weight > 0:
        weights = weights + float(diag_weight) * diag[None].float() * valid_pair.float()
    if not (weights > 0).any():
        return logits.sum() * 0.0
    target_prob = target.clamp_min(eps)
    target_prob = target_prob / target_prob.sum(dim=-1, keepdim=True).clamp_min(eps)
    log_prob = F.log_softmax(logits, dim=-1)
    ce = -(target_prob * log_prob).sum(dim=-1)
    entropy = -(target_prob * target_prob.log()).sum(dim=-1)
    kl = ce - entropy
    return (kl * weights).sum() / weights.sum().clamp_min(1.0)


def upper_valid_pair_mask(pair_mask: torch.Tensor) -> torch.Tensor:
    """Return valid upper-triangle pair mask ``[B,L,L]``."""
    if pair_mask.ndim != 3 or pair_mask.shape[1] != pair_mask.shape[2]:
        raise ValueError(f"pair_mask must be [B, L, L], got {tuple(pair_mask.shape)}")
    B, L, _ = pair_mask.shape
    upper = torch.triu(torch.ones(L, L, device=pair_mask.device, dtype=torch.bool), diagonal=1)
    valid_pair = pair_mask.bool() & pair_mask.bool().transpose(-1, -2)
    return upper[None].expand(B, L, L) & valid_pair


def stats_pair_weights_from_scalar_target(
    scalar_target: torch.Tensor,
    pair_mask: torch.Tensor,
    alpha: float = 2.0,
    top_factor: float = 2.0,
    top_boost: float = 2.0,
    eps: float = EPS,
) -> torch.Tensor:
    """Build pair weights that emphasize high coevolution edges in stats losses.

    The scalar MI/APC target remains the authority for which pairs matter most.
    We use a smooth per-sample normalized weight plus an optional top-k boost.
    """
    if scalar_target.ndim == 4:
        if scalar_target.shape[1] != 1:
            raise ValueError(f"scalar_target channel dim must be 1, got {tuple(scalar_target.shape)}")
        scalar = scalar_target[:, 0]
    elif scalar_target.ndim == 3:
        scalar = scalar_target
    else:
        raise ValueError(f"scalar_target must be [B,1,L,L] or [B,L,L], got {tuple(scalar_target.shape)}")
    if scalar.shape != pair_mask.shape:
        raise ValueError(f"scalar_target/pair_mask mismatch: {tuple(scalar.shape)} vs {tuple(pair_mask.shape)}")

    valid = upper_valid_pair_mask(pair_mask)
    scores = scalar.detach().abs().float() * valid.float()
    scale = scores.amax(dim=(-1, -2), keepdim=True).clamp_min(eps)
    normalized = (scores / scale).clamp(0.0, 1.0)
    weights = (1.0 + float(alpha) * normalized) * valid.float()

    if top_factor > 0 and top_boost > 0:
        B, L, _ = scores.shape
        diag_valid = torch.diagonal(pair_mask.bool(), dim1=-2, dim2=-1).sum(dim=-1)
        boosted = torch.zeros_like(weights)
        for b in range(B):
            edge_count = int(valid[b].sum().item())
            if edge_count <= 0:
                continue
            k = min(edge_count, max(1, int(np.ceil(float(top_factor) * float(diag_valid[b].item())))))
            values = scores[b][valid[b]]
            if values.numel() == 0:
                continue
            if float(values.max().item()) <= eps:
                continue
            top_idx = torch.topk(values, k=k, largest=True).indices
            coords = valid[b].nonzero(as_tuple=False)
            selected = coords[top_idx]
            boosted[b, selected[:, 0], selected[:, 1]] = float(top_boost)
        weights = weights + boosted
    return weights * valid.float()


def normalize_prob_target_fp32(target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalize a categorical probability target in fp32 for stable loss math."""
    target_prob = target.float().clamp_min(eps)
    return target_prob / target_prob.sum(dim=-1, keepdim=True).clamp_min(eps)


def center_stats_logits_fp32(
    logits: torch.Tensor,
    pair_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Center 25-channel pair logits in fp32 and optionally apply a pair mask."""
    centered = logits.float() - logits.float().mean(dim=-1, keepdim=True)
    if pair_mask is not None:
        centered = centered * pair_mask[..., None].float()
    return centered


def centered_log_prob_target(
    target: torch.Tensor,
    eps: float = 1e-6,
    clip_value: float = 8.0,
) -> torch.Tensor:
    """Convert a probability target to centered, clipped log-prob logits."""
    logits = torch.log(normalize_prob_target_fp32(target, eps=eps))
    logits = logits - logits.mean(dim=-1, keepdim=True)
    return logits.clamp(-float(clip_value), float(clip_value))


def _normalize_stats_target(target: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    return normalize_prob_target_fp32(target, eps=eps)


def _target_fi_from_stats(target_prob: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    if target_prob.ndim != 4 or target_prob.shape[1] != target_prob.shape[2] or target_prob.shape[-1] != STATS_Q * STATS_Q:
        raise ValueError(f"target_prob must be [B, L, L, 25], got {tuple(target_prob.shape)}")
    B, L, _, _ = target_prob.shape
    blocks = target_prob.reshape(B, L, L, STATS_Q, STATS_Q)
    diag_blocks = blocks[:, torch.arange(L, device=target_prob.device), torch.arange(L, device=target_prob.device)]
    fi = torch.diagonal(diag_blocks, dim1=-2, dim2=-1).clamp_min(eps)
    return fi / fi.sum(dim=-1, keepdim=True).clamp_min(eps)


def _pair_weight_average(pair_loss: torch.Tensor, pair_mask: torch.Tensor, pair_weights: torch.Tensor | None) -> torch.Tensor:
    valid = upper_valid_pair_mask(pair_mask)
    weights = valid.float()
    if pair_weights is not None:
        if pair_weights.shape != pair_mask.shape:
            raise ValueError(f"pair_weights must be [B, L, L], got {tuple(pair_weights.shape)}")
        weights = weights * pair_weights.float().clamp_min(0.0)
    if not (weights > 0).any():
        return pair_loss.sum() * 0.0
    return (pair_loss * weights).sum() / weights.sum().clamp_min(1.0)


def weighted_kl_from_logits_fp32(
    logits: torch.Tensor,
    target: torch.Tensor,
    pair_mask: torch.Tensor | None = None,
    pair_weights: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Weighted KL(target || softmax(logits)) in fp32 using log-softmax."""
    if logits.shape != target.shape:
        raise ValueError(f"logits/target mismatch: {tuple(logits.shape)} vs {tuple(target.shape)}")
    target_prob = normalize_prob_target_fp32(target, eps=eps)
    log_prob = F.log_softmax(logits.float(), dim=-1)
    kl_per_pair = (target_prob * (torch.log(target_prob) - log_prob)).sum(dim=-1)
    if pair_mask is None:
        weights = torch.ones_like(kl_per_pair, dtype=kl_per_pair.dtype)
        if pair_weights is not None:
            if pair_weights.shape != kl_per_pair.shape:
                raise ValueError(f"pair_weights must be {tuple(kl_per_pair.shape)}, got {tuple(pair_weights.shape)}")
            weights = weights * pair_weights.float().clamp_min(0.0)
        return (kl_per_pair * weights).sum() / weights.sum().clamp_min(1.0)
    return _pair_weight_average(kl_per_pair, pair_mask, pair_weights)


def stats_marginal_consistency_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    pair_mask: torch.Tensor,
    pair_weights: torch.Tensor | None = None,
    eps: float = EPS,
) -> torch.Tensor:
    """Force predicted off-diagonal pair blocks to recover target single-site marginals."""
    if logits.shape != target.shape:
        raise ValueError(f"logits/target mismatch: {tuple(logits.shape)} vs {tuple(target.shape)}")
    probs = F.softmax(logits, dim=-1).reshape(*logits.shape[:3], STATS_Q, STATS_Q)
    target_prob = _normalize_stats_target(target, eps=eps)
    fi = _target_fi_from_stats(target_prob, eps=eps)
    fi_i = fi[:, :, None, :]
    fi_j = fi[:, None, :, :]
    row_marg = probs.sum(dim=-1)
    col_marg = probs.sum(dim=-2)
    row_loss = F.smooth_l1_loss(row_marg, fi_i.expand_as(row_marg), reduction="none").mean(dim=-1)
    col_loss = F.smooth_l1_loss(col_marg, fi_j.expand_as(col_marg), reduction="none").mean(dim=-1)
    return _pair_weight_average(0.5 * (row_loss + col_loss), pair_mask, pair_weights)


def stats_log_odds_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    pair_mask: torch.Tensor,
    pair_weights: torch.Tensor | None = None,
    eps: float = EPS,
    clip_value: float = 8.0,
) -> torch.Tensor:
    """Match pair log-odds tensors relative to the target single-site frequencies."""
    if logits.shape != target.shape:
        raise ValueError(f"logits/target mismatch: {tuple(logits.shape)} vs {tuple(target.shape)}")
    pred = F.softmax(logits, dim=-1).reshape(*logits.shape[:3], STATS_Q, STATS_Q).clamp_min(eps)
    target_prob = _normalize_stats_target(target, eps=eps)
    target_joint = target_prob.reshape(*target.shape[:3], STATS_Q, STATS_Q).clamp_min(eps)
    fi = _target_fi_from_stats(target_prob, eps=eps)
    denom = (fi[:, :, None, :, None] * fi[:, None, :, None, :]).clamp_min(eps)
    pred_log_odds = (pred / denom).log().clamp(-clip_value, clip_value)
    target_log_odds = (target_joint / denom).log().clamp(-clip_value, clip_value)
    pair_loss = F.smooth_l1_loss(pred_log_odds, target_log_odds, reduction="none").mean(dim=(-1, -2))
    return _pair_weight_average(pair_loss, pair_mask, pair_weights)


def stats_covariance_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    pair_mask: torch.Tensor,
    pair_weights: torch.Tensor | None = None,
    eps: float = EPS,
) -> torch.Tensor:
    """Match connected pair statistics ``Pij - Pi Pj``."""
    if logits.shape != target.shape:
        raise ValueError(f"logits/target mismatch: {tuple(logits.shape)} vs {tuple(target.shape)}")
    pred = F.softmax(logits, dim=-1).reshape(*logits.shape[:3], STATS_Q, STATS_Q)
    target_prob = _normalize_stats_target(target, eps=eps)
    target_joint = target_prob.reshape(*target.shape[:3], STATS_Q, STATS_Q)
    fi = _target_fi_from_stats(target_prob, eps=eps)
    independent = fi[:, :, None, :, None] * fi[:, None, :, None, :]
    pred_cov = pred - independent
    target_cov = target_joint - independent
    pair_loss = F.smooth_l1_loss(pred_cov, target_cov, reduction="none").mean(dim=(-1, -2))
    return _pair_weight_average(pair_loss, pair_mask, pair_weights)


def stats_probs_to_mi_apc_torch(probs: torch.Tensor, pair_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Differentiable MI+APC from normalized ``[B,L,L,25]`` pair probabilities."""
    if probs.ndim != 4 or probs.shape[1] != probs.shape[2] or probs.shape[-1] != STATS_Q * STATS_Q:
        raise ValueError(f"probs must be [B, L, L, 25], got {tuple(probs.shape)}")
    B, L, _, _ = probs.shape
    joint = probs.reshape(B, L, L, STATS_Q, STATS_Q).clamp_min(EPS)
    joint = joint / joint.sum(dim=(-1, -2), keepdim=True).clamp_min(EPS)
    pi = joint.sum(dim=-1, keepdim=True)
    pj = joint.sum(dim=-2, keepdim=True)
    denom = (pi * pj).clamp_min(EPS)
    mi = (joint * (joint / denom).log()).sum(dim=(-1, -2)) / np.log(2.0)
    mi = mi.clamp_min(0.0)
    diag = torch.eye(L, device=probs.device, dtype=torch.bool)
    mi = mi.masked_fill(diag[None], 0.0)
    if pair_mask is not None:
        valid = pair_mask.bool() & pair_mask.bool().transpose(-1, -2)
        mi = mi * valid.float()
        offdiag_valid = valid & ~diag[None]
        row_count = offdiag_valid.float().sum(dim=-1).clamp_min(1.0)
        row = (mi * offdiag_valid.float()).sum(dim=-1) / row_count
        total_count = offdiag_valid.float().sum(dim=(-1, -2)).clamp_min(1.0)
        total = (mi * offdiag_valid.float()).sum(dim=(-1, -2)) / total_count
    else:
        row = mi.sum(dim=-1) / max(L - 1, 1)
        total = mi.sum(dim=(-1, -2)) / max(L * (L - 1), 1)
    if L <= 1:
        return mi
    apc = mi - row[:, :, None] * row[:, None, :] / total[:, None, None].clamp_min(EPS)
    apc = apc.clamp_min(0.0).masked_fill(diag[None], 0.0)
    if pair_mask is not None:
        apc = apc * valid.float()
    return apc


def stats_scalar_consistency_loss(
    logits: torch.Tensor,
    scalar_target: torch.Tensor,
    pair_mask: torch.Tensor,
) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    pred_scalar = stats_probs_to_mi_apc_torch(probs, pair_mask=pair_mask)
    if scalar_target.ndim == 4:
        scalar_target = scalar_target[:, 0]
    valid = torch.triu(torch.ones(pred_scalar.shape[-1], pred_scalar.shape[-1], device=logits.device, dtype=torch.bool), diagonal=1)
    valid = valid[None] & pair_mask.bool() & pair_mask.bool().transpose(-1, -2)
    if not valid.any():
        return pred_scalar.sum() * 0.0
    return F.mse_loss(pred_scalar[valid], scalar_target[valid])


def fi_mse_loss(
    fi_logits: torch.Tensor,
    fi_target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = EPS,
) -> torch.Tensor:
    """MSE between predicted single-site frequencies and target ``fi``."""
    if fi_logits.shape != fi_target.shape:
        raise ValueError(f"fi_logits/fi_target mismatch: {tuple(fi_logits.shape)} vs {tuple(fi_target.shape)}")
    if fi_logits.ndim != 3 or fi_logits.shape[-1] != STATS_Q:
        raise ValueError(f"fi_logits must be [B, L, 5], got {tuple(fi_logits.shape)}")
    if mask.shape != fi_logits.shape[:2]:
        raise ValueError(f"mask must be [B, L], got {tuple(mask.shape)}")
    valid = mask.bool()
    if not valid.any():
        return fi_logits.sum() * 0.0
    target = fi_target.clamp_min(eps)
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(eps)
    pred = torch.softmax(fi_logits.float(), dim=-1)
    return F.mse_loss(pred[valid], target[valid])


def fij_kl_loss(
    fij_logits: torch.Tensor,
    fij_target: torch.Tensor,
    pair_mask: torch.Tensor,
    pair_weights: torch.Tensor | None = None,
    eps: float = EPS,
) -> torch.Tensor:
    """Weighted off-diagonal KL(target fij || predicted fij)."""
    if fij_logits.shape != fij_target.shape:
        raise ValueError(f"fij_logits/fij_target mismatch: {tuple(fij_logits.shape)} vs {tuple(fij_target.shape)}")
    if fij_logits.ndim != 4 or fij_logits.shape[-1] != STATS_Q * STATS_Q:
        raise ValueError(f"fij_logits must be [B, L, L, 25], got {tuple(fij_logits.shape)}")
    target = fij_target.clamp_min(eps)
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(eps)
    log_prob = F.log_softmax(fij_logits.float(), dim=-1)
    ce = -(target * log_prob).sum(dim=-1)
    entropy = -(target * target.log()).sum(dim=-1)
    return _pair_weight_average(ce - entropy, pair_mask, pair_weights)


def _normalize_fi_target(fi_target: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    if fi_target.ndim != 3 or fi_target.shape[-1] != STATS_Q:
        raise ValueError(f"fi_target must be [B, L, 5], got {tuple(fi_target.shape)}")
    target = fi_target.clamp_min(eps)
    return target / target.sum(dim=-1, keepdim=True).clamp_min(eps)


def fij_marginal_target_loss(
    fij_logits: torch.Tensor,
    fi_target: torch.Tensor,
    pair_mask: torch.Tensor,
    pair_weights: torch.Tensor | None = None,
    eps: float = EPS,
) -> torch.Tensor:
    """Force predicted ``fij`` marginals to match target ``fi``."""
    if fij_logits.ndim != 4 or fij_logits.shape[-1] != STATS_Q * STATS_Q:
        raise ValueError(f"fij_logits must be [B, L, L, 25], got {tuple(fij_logits.shape)}")
    pred = torch.softmax(fij_logits.float(), dim=-1).reshape(*fij_logits.shape[:3], STATS_Q, STATS_Q)
    fi = _normalize_fi_target(fi_target, eps=eps)
    row_marg = pred.sum(dim=-1)
    col_marg = pred.sum(dim=-2)
    row_loss = F.smooth_l1_loss(row_marg, fi[:, :, None, :].expand_as(row_marg), reduction="none").mean(dim=-1)
    col_loss = F.smooth_l1_loss(col_marg, fi[:, None, :, :].expand_as(col_marg), reduction="none").mean(dim=-1)
    return _pair_weight_average(0.5 * (row_loss + col_loss), pair_mask, pair_weights)


def fij_marginal_pred_loss(
    fij_logits: torch.Tensor,
    fi_logits: torch.Tensor,
    pair_mask: torch.Tensor,
    pair_weights: torch.Tensor | None = None,
    detach_fi: bool = True,
) -> torch.Tensor:
    """Force predicted ``fij`` marginals to agree with the predicted ``fi`` head."""
    if fij_logits.ndim != 4 or fij_logits.shape[-1] != STATS_Q * STATS_Q:
        raise ValueError(f"fij_logits must be [B, L, L, 25], got {tuple(fij_logits.shape)}")
    if fi_logits.shape[:2] != fij_logits.shape[:2] or fi_logits.shape[-1] != STATS_Q:
        raise ValueError(f"fi_logits must be [B, L, 5], got {tuple(fi_logits.shape)}")
    pred = torch.softmax(fij_logits.float(), dim=-1).reshape(*fij_logits.shape[:3], STATS_Q, STATS_Q)
    fi = torch.softmax(fi_logits.float(), dim=-1)
    if detach_fi:
        fi = fi.detach()
    row_marg = pred.sum(dim=-1)
    col_marg = pred.sum(dim=-2)
    row_loss = F.smooth_l1_loss(row_marg, fi[:, :, None, :].expand_as(row_marg), reduction="none").mean(dim=-1)
    col_loss = F.smooth_l1_loss(col_marg, fi[:, None, :, :].expand_as(col_marg), reduction="none").mean(dim=-1)
    return _pair_weight_average(0.5 * (row_loss + col_loss), pair_mask, pair_weights)


def fij_log_odds_loss(
    fij_logits: torch.Tensor,
    fij_target: torch.Tensor,
    fi_target: torch.Tensor,
    pair_mask: torch.Tensor,
    pair_weights: torch.Tensor | None = None,
    eps: float = EPS,
    clip_value: float = 8.0,
) -> torch.Tensor:
    """Match off-diagonal pair log-odds using target ``fi`` as the denominator."""
    return fij_log_odds_loss_fp32(
        fij_logits=fij_logits,
        fij_target=fij_target,
        fi_target=fi_target,
        pair_mask=pair_mask,
        pair_weights=pair_weights,
        eps=eps,
        clip_value=clip_value,
    )


def fij_log_odds_loss_fp32(
    fij_logits: torch.Tensor,
    fij_target: torch.Tensor,
    fi_target: torch.Tensor,
    pair_mask: torch.Tensor,
    pair_weights: torch.Tensor | None = None,
    eps: float = EPS,
    clip_value: float = 6.0,
) -> torch.Tensor:
    """Stable off-diagonal pair log-odds loss using fp32 log-softmax logits."""
    if fij_logits.shape != fij_target.shape:
        raise ValueError(f"fij_logits/fij_target mismatch: {tuple(fij_logits.shape)} vs {tuple(fij_target.shape)}")
    if fij_logits.ndim != 4 or fij_logits.shape[-1] != STATS_Q * STATS_Q:
        raise ValueError(f"fij_logits must be [B, L, L, 25], got {tuple(fij_logits.shape)}")
    log_pred = F.log_softmax(fij_logits.float(), dim=-1).reshape(*fij_logits.shape[:3], STATS_Q, STATS_Q)
    target = normalize_prob_target_fp32(fij_target, eps=eps)
    log_target = torch.log(target).reshape(*fij_target.shape[:3], STATS_Q, STATS_Q)
    fi = _normalize_fi_target(fi_target, eps=eps)
    log_pi = torch.log(fi[:, :, None, :, None].clamp_min(eps))
    log_pj = torch.log(fi[:, None, :, None, :].clamp_min(eps))
    pred_log_odds = (log_pred - log_pi - log_pj).clamp(-clip_value, clip_value)
    target_log_odds = (log_target - log_pi - log_pj).clamp(-clip_value, clip_value)
    pair_loss = F.smooth_l1_loss(pred_log_odds, target_log_odds, reduction="none").mean(dim=(-1, -2))
    return _pair_weight_average(pair_loss, pair_mask, pair_weights)


def fij_covariance_loss(
    fij_logits: torch.Tensor,
    fij_target: torch.Tensor,
    fi_target: torch.Tensor,
    pair_mask: torch.Tensor,
    pair_weights: torch.Tensor | None = None,
    eps: float = EPS,
) -> torch.Tensor:
    """Match connected off-diagonal statistics ``fij - fi*fj``."""
    if fij_logits.shape != fij_target.shape:
        raise ValueError(f"fij_logits/fij_target mismatch: {tuple(fij_logits.shape)} vs {tuple(fij_target.shape)}")
    pred = torch.softmax(fij_logits.float(), dim=-1).reshape(*fij_logits.shape[:3], STATS_Q, STATS_Q)
    target = fij_target.clamp_min(eps)
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(eps)
    target_joint = target.reshape(*fij_target.shape[:3], STATS_Q, STATS_Q)
    fi = _normalize_fi_target(fi_target, eps=eps)
    independent = fi[:, :, None, :, None] * fi[:, None, :, None, :]
    pred_cov = pred - independent
    target_cov = target_joint - independent
    pair_loss = F.smooth_l1_loss(pred_cov, target_cov, reduction="none").mean(dim=(-1, -2))
    return _pair_weight_average(pair_loss, pair_mask, pair_weights)


def fij_scalar_consistency_loss(
    fij_logits: torch.Tensor,
    scalar_target: torch.Tensor,
    pair_mask: torch.Tensor,
) -> torch.Tensor:
    """MSE between MI+APC implied by predicted ``fij`` and scalar target."""
    return stats_scalar_consistency_loss(fij_logits, scalar_target, pair_mask)
