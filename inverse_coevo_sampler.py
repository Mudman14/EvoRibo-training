"""
No-training coevolution-map-to-MSA inverse sampler.

This script takes raw RNA A3M files, computes an MI+APC coevolution map,
then generates a synthetic aligned MSA with a query-conditioned Potts sampler
and per-MSA iterative coupling calibration. It does not train a model and does
not require dependencies beyond numpy and matplotlib.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ALPHABET = "AUCG-"
GAP_IDX = 4
EPS = 1e-12

AUTO_RNA_IDS = [
    "2VPL_B",
    "2OIH_B",
    "3EGZ_B",
    "1JBT_C",
    "5AOX_C",
    "6BJX_A",
    "1ASY_R",
    "7A0R_Y",
    "1E8O_E",
    "1G59_B",
]

CANONICAL_PAIRS = {("A", "U"), ("U", "A"), ("G", "C"), ("C", "G")}
WOBBLE_PAIRS = {("G", "U"), ("U", "G")}


@dataclass
class ParsedMSA:
    headers: list[str]
    seqs: list[str]

    @property
    def query(self) -> str:
        return self.seqs[0] if self.seqs else ""

    @property
    def depth(self) -> int:
        return len(self.seqs)

    @property
    def length(self) -> int:
        return len(self.query)


@dataclass
class EdgeSet:
    i: np.ndarray
    j: np.ndarray
    target_norm: np.ndarray
    scale: float

    @property
    def size(self) -> int:
        return int(self.i.size)


@dataclass
class SourceStats:
    seqs: list[str]
    profile: np.ndarray
    identity_values: np.ndarray
    gap_fraction_values: np.ndarray | None = None
    global_prior: "GlobalInversePrior | None" = None
    alphabet_prior: "AlphabetInversePrior | None" = None
    bpp: np.ndarray | None = None

    @property
    def depth(self) -> int:
        return len(self.seqs)


@dataclass
class GlobalInversePrior:
    path: Path
    arrays: dict[str, np.ndarray]
    last_pair_fallback_counts: dict[str, int]

    def array(self, key: str) -> np.ndarray:
        if key not in self.arrays:
            raise KeyError(f"Global inverse prior {self.path} is missing array '{key}'")
        return self.arrays[key]


@dataclass
class AlphabetInversePrior:
    path: Path
    probs: np.ndarray
    mode: str


def normalize_char(ch: str) -> str:
    c = ch.upper()
    if c == "T":
        c = "U"
    if c in ALPHABET:
        return c
    return "-"


def clean_a3m_sequence(seq: str) -> str:
    """Remove A3M lowercase insertions and normalize to A/U/C/G/gap."""
    return "".join(normalize_char(ch) for ch in seq if ch.isupper() or ch == "-")


def parse_a3m(path: Path) -> ParsedMSA:
    headers: list[str] = []
    seqs: list[str] = []
    header: str | None = None
    current: list[str] = []

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n\r")
            if line.startswith(">"):
                if header is not None:
                    headers.append(header)
                    seqs.append(clean_a3m_sequence("".join(current)))
                header = line
                current = []
            else:
                current.append(line.strip())

    if header is not None:
        headers.append(header)
        seqs.append(clean_a3m_sequence("".join(current)))

    seqs = align_sequences(seqs)
    return ParsedMSA(headers=headers, seqs=seqs)


def align_sequences(seqs: list[str]) -> list[str]:
    if not seqs:
        return seqs
    length = len(seqs[0])
    aligned = []
    for seq in seqs:
        if len(seq) >= length:
            aligned.append(seq[:length])
        else:
            aligned.append(seq + "-" * (length - len(seq)))
    return aligned


def seqs_to_numeric(seqs: list[str]) -> np.ndarray:
    idx = {c: i for i, c in enumerate(ALPHABET)}
    return np.asarray([[idx.get(ch, GAP_IDX) for ch in seq] for seq in seqs], dtype=np.int8)


def numeric_to_seqs(msa_num: np.ndarray) -> list[str]:
    chars = np.asarray(list(ALPHABET))
    return ["".join(chars[row].tolist()) for row in msa_num]


def compute_mi_apc(seqs: list[str], max_seqs: int | None = None) -> np.ndarray:
    """Compute MI+APC using vectorized 5-state joint counts."""
    if max_seqs is not None:
        seqs = seqs[:max_seqs]
    seqs = align_sequences(seqs)
    if len(seqs) < 2:
        length = len(seqs[0]) if seqs else 1
        return np.zeros((length, length), dtype=np.float64)

    msa_num = seqs_to_numeric(seqs)
    n_seq, length = msa_num.shape
    n_states = len(ALPHABET)
    pseudo = 1.0 / max(n_seq, 1)

    single_counts = np.zeros((length, n_states), dtype=np.float64)
    state_masks: list[np.ndarray] = []
    for state in range(n_states):
        mask = (msa_num == state).astype(np.float64)
        state_masks.append(mask)
        single_counts[:, state] = mask.sum(axis=0)

    fi = (single_counts + pseudo) / (n_seq + n_states * pseudo)
    hi = -np.sum(fi * np.log2(fi + EPS), axis=1)

    joint_counts = np.empty((length, length, n_states, n_states), dtype=np.float64)
    for a in range(n_states):
        ma = state_masks[a]
        for b in range(n_states):
            joint_counts[:, :, a, b] = ma.T @ state_masks[b]

    fij = (joint_counts + pseudo) / (n_seq + n_states * n_states * pseudo)
    hij = -np.sum(fij * np.log2(fij + EPS), axis=(2, 3))
    mi = hi[:, None] + hi[None, :] - hij
    mi = np.maximum(mi, 0.0)
    np.fill_diagonal(mi, 0.0)

    denom = max(length - 1, 1)
    row_mean = mi.sum(axis=1) / denom
    total_mean = mi.sum() / max(length * (length - 1), 1)
    if total_mean <= EPS:
        return np.zeros_like(mi)

    apc = mi - np.outer(row_mean, row_mean) / total_mean
    apc = np.maximum(apc, 0.0)
    np.fill_diagonal(apc, 0.0)
    return apc


def upper_values(matrix: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    idx = np.triu_indices(matrix.shape[0], k=1)
    return matrix[idx], idx


def target_scale(matrix: np.ndarray) -> float:
    values, _ = upper_values(matrix)
    positive = values[np.isfinite(values) & (values > 0)]
    if positive.size == 0:
        return 1.0
    return float(max(np.percentile(positive, 95), positive.max() * 0.25, EPS))


def select_edges(target_mi: np.ndarray, edge_factor: float) -> EdgeSet:
    length = target_mi.shape[0]
    values, idx = upper_values(target_mi)
    top_k = min(max(1, int(round(edge_factor * length))), values.size)
    order = np.argsort(values)[::-1][:top_k]
    scale = target_scale(target_mi)
    edge_i = idx[0][order].astype(np.int32)
    edge_j = idx[1][order].astype(np.int32)
    target_norm = target_mi[edge_i, edge_j] / scale
    return EdgeSet(i=edge_i, j=edge_j, target_norm=target_norm, scale=scale)


def build_query_fields(
    query: str,
    same_prob: float,
    mutation_prob_each: float,
    gap_prob: float,
) -> np.ndarray:
    fields = np.empty((len(query), len(ALPHABET)), dtype=np.float64)
    base_states = [0, 1, 2, 3]
    for pos, base in enumerate(query):
        probs = np.full(len(ALPHABET), mutation_prob_each, dtype=np.float64)
        probs[GAP_IDX] = gap_prob
        base_idx = ALPHABET.find(base)
        if base_idx in base_states:
            probs[base_idx] = same_prob
        else:
            probs[:4] = (1.0 - gap_prob) / 4.0
            probs[GAP_IDX] = gap_prob
        probs = probs / probs.sum()
        fields[pos] = np.log(probs + EPS)
    return fields


def estimate_profile(seqs: list[str], pseudo: float = 1.0) -> np.ndarray:
    seqs = align_sequences(seqs)
    if not seqs:
        raise ValueError("Cannot estimate profile from empty MSA.")
    msa_num = seqs_to_numeric(seqs)
    length = msa_num.shape[1]
    counts = np.zeros((length, len(ALPHABET)), dtype=np.float64)
    for state in range(len(ALPHABET)):
        counts[:, state] = (msa_num == state).sum(axis=0)
    return (counts + pseudo) / (msa_num.shape[0] + len(ALPHABET) * pseudo)


def identity_values_to_query(seqs: list[str]) -> np.ndarray:
    seqs = align_sequences(seqs)
    if len(seqs) <= 1:
        return np.asarray([], dtype=np.float64)
    query = seqs[0]
    identities = []
    for seq in seqs[1:]:
        matches = 0
        total = 0
        for q_char, s_char in zip(query, seq):
            if q_char == "-":
                continue
            total += 1
            matches += int(q_char == s_char)
        if total:
            identities.append(matches / total)
    return np.asarray(identities, dtype=np.float64)


def gap_fraction_values(seqs: list[str]) -> np.ndarray:
    seqs = align_sequences(seqs)
    if len(seqs) <= 1:
        return np.asarray([], dtype=np.float64)
    values = []
    for seq in seqs[1:]:
        values.append(seq.count("-") / max(len(seq), 1))
    return np.asarray(values, dtype=np.float64)


def parse_dense_or_sparse_bpp(path: Path, expected_length: int) -> np.ndarray:
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rows.append([float(part) for part in line.split()])
            except ValueError:
                continue
    if not rows:
        return np.zeros((expected_length, expected_length), dtype=np.float64)

    widths = {len(row) for row in rows}
    dense = len(widths) == 1 and next(iter(widths)) == len(rows)
    if dense and len(rows) == 3:
        dense = not all(
            float(row[0]).is_integer()
            and float(row[1]).is_integer()
            and 1 <= int(row[0]) <= 3
            and 1 <= int(row[1]) <= 3
            for row in rows
        )
    if dense:
        mat = np.asarray(rows, dtype=np.float64)
    else:
        mat = np.zeros((expected_length, expected_length), dtype=np.float64)
        for row in rows:
            if len(row) < 3:
                continue
            i, j, p = int(row[0]) - 1, int(row[1]) - 1, float(row[2])
            if 0 <= i < expected_length and 0 <= j < expected_length:
                mat[i, j] = p
                mat[j, i] = p

    if mat.shape != (expected_length, expected_length):
        raise ValueError(f"{path} shape {mat.shape} != {(expected_length, expected_length)}")
    mat = np.maximum(mat, mat.T)
    np.fill_diagonal(mat, 0.0)
    return mat


def resolve_bpp_path(args: argparse.Namespace, rna_id: str) -> Path | None:
    bpp_path = getattr(args, "bpp_path", None)
    if bpp_path:
        return Path(bpp_path)
    bpp_dir = getattr(args, "bpp_dir", None)
    if bpp_dir:
        candidate = Path(bpp_dir) / f"{rna_id}.prob"
        if candidate.exists():
            return candidate
    return None


def load_case_bpp(args: argparse.Namespace, rna_id: str, expected_length: int) -> np.ndarray | None:
    path = resolve_bpp_path(args, rna_id)
    if path is None or not path.exists():
        return None
    return parse_dense_or_sparse_bpp(path, expected_length)


def load_global_inverse_prior(path: str | Path) -> GlobalInversePrior:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing global inverse prior: {path}")
    with np.load(path, allow_pickle=False) as loaded:
        arrays = {key: np.asarray(loaded[key]) for key in loaded.files}
    return GlobalInversePrior(path=path, arrays=arrays, last_pair_fallback_counts={})


def load_alphabet_inverse_prior(path: str | Path, mode: str = "family_balanced") -> AlphabetInversePrior:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing alphabet inverse prior: {path}")
    key = "probs_family_balanced" if mode == "family_balanced" else "probs_raw_token"
    with np.load(path, allow_pickle=False) as loaded:
        if key not in loaded.files:
            raise KeyError(f"Alphabet inverse prior {path} is missing '{key}'")
        probs = np.asarray(loaded[key], dtype=np.float64)
    if probs.shape != (len(ALPHABET),):
        raise ValueError(f"Alphabet prior probabilities must have shape ({len(ALPHABET)},), got {probs.shape}")
    if not np.isfinite(probs).all() or probs.sum() <= EPS:
        raise ValueError(f"Alphabet prior probabilities are invalid in {path}")
    probs = np.maximum(probs, EPS)
    probs = probs / probs.sum()
    return AlphabetInversePrior(path=path, probs=probs, mode=mode)


def _bin_index(value: float, edges: np.ndarray) -> int:
    edges = np.asarray(edges, dtype=np.float64)
    return int(np.searchsorted(edges, float(value), side="right"))


def _query_base_index(ch: str) -> int:
    idx = ALPHABET.find(ch)
    if idx < 0:
        return GAP_IDX
    return idx


def _query_pair_code(qi: int, qj: int) -> int:
    if 0 <= qi < 4 and 0 <= qj < 4:
        return qi * 4 + qj
    return 16


def _length_bin(length: int, edges: np.ndarray) -> int:
    return min(_bin_index(float(length), edges), len(edges))


def values_from_histogram(hist: np.ndarray, edges: np.ndarray, max_values: int = 2000) -> np.ndarray:
    hist = np.asarray(hist, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.float64)
    if hist.size == 0 or hist.sum() <= EPS:
        return np.asarray([], dtype=np.float64)
    probs = hist / hist.sum()
    mids = 0.5 * (edges[:-1] + edges[1:])
    counts = np.floor(probs * max_values).astype(np.int32)
    for idx in np.argsort(probs)[::-1]:
        if counts.sum() >= max_values:
            break
        counts[idx] += 1
    values = np.repeat(mids, counts)
    return values.astype(np.float64)


def global_prior_identity_gap_values(
    prior: GlobalInversePrior,
    length: int,
    min_count: float,
) -> tuple[np.ndarray, np.ndarray]:
    value_edges = prior.array("value_bin_edges")
    length_edges = prior.array("length_bin_edges")
    length_idx = _length_bin(length, length_edges)

    identity_hist = prior.array("identity_hist_global")
    gap_hist = prior.array("gap_hist_global")
    if "identity_hist_by_length" in prior.arrays:
        by_length = prior.array("identity_hist_by_length")
        if length_idx < by_length.shape[0] and by_length[length_idx].sum() >= min_count:
            identity_hist = by_length[length_idx]
    if "gap_hist_by_length" in prior.arrays:
        by_length = prior.array("gap_hist_by_length")
        if length_idx < by_length.shape[0] and by_length[length_idx].sum() >= min_count:
            gap_hist = by_length[length_idx]

    return (
        values_from_histogram(identity_hist, value_edges),
        values_from_histogram(gap_hist, value_edges),
    )


def build_global_prior_profile(
    query: str,
    prior: GlobalInversePrior,
    bpp: np.ndarray | None,
    args: argparse.Namespace,
) -> np.ndarray:
    single_base = prior.array("single_probs_base")
    single_base_counts = prior.array("single_counts_base")
    single_paired = prior.array("single_probs_paired")
    single_paired_counts = prior.array("single_counts_paired")
    paired_edges = prior.array("pairedness_bin_edges")
    mode = getattr(args, "global_prior_mode", "auto")
    min_count = float(getattr(args, "global_prior_min_count", 5.0))

    query_fields = build_query_fields(query, args.same_prob, args.mutation_prob_each, args.gap_prob)
    fallback = np.exp(query_fields)
    fallback = fallback / fallback.sum(axis=1, keepdims=True)
    profile = np.empty((len(query), len(ALPHABET)), dtype=np.float64)
    pairedness = bpp.max(axis=1) if bpp is not None and bpp.shape[0] == len(query) else None

    for pos, base in enumerate(query):
        qi = _query_base_index(base)
        chosen: np.ndarray | None = None
        if mode != "global" and pairedness is not None:
            paired_bin = _bin_index(float(pairedness[pos]), paired_edges)
            count = float(single_paired_counts[paired_bin, qi])
            if count > 0 and (mode == "conditioned" or count >= min_count):
                chosen = single_paired[paired_bin, qi]
        if chosen is None and float(single_base_counts[qi]) > 0:
            chosen = single_base[qi]
        if chosen is None:
            chosen = fallback[pos]
        chosen = np.asarray(chosen, dtype=np.float64)
        profile[pos] = chosen / max(chosen.sum(), EPS)
    return profile


def build_alphabet_prior_profile(query: str, prior: AlphabetInversePrior) -> np.ndarray:
    probs = prior.probs / max(float(prior.probs.sum()), EPS)
    return np.repeat(probs[None, :], len(query), axis=0)


def select_source_seqs(parsed: ParsedMSA, args: argparse.Namespace) -> list[str]:
    if args.stats_source == "oracle":
        return parsed.seqs[: args.stats_max_seqs]
    if args.stats_source == "shallow":
        return parsed.seqs[: min(args.stats_max_seqs, args.shallow_depth)]
    if args.stats_source in {"global_prior", "alphabet_prior"}:
        return []
    return [parsed.query]


def build_source_stats(
    parsed: ParsedMSA,
    args: argparse.Namespace,
    bpp: np.ndarray | None = None,
) -> SourceStats:
    if args.stats_source == "global_prior":
        global_prior_path = getattr(args, "global_prior_path", None)
        if not global_prior_path:
            raise ValueError("--global_prior_path is required when --stats_source global_prior")
        prior = load_global_inverse_prior(global_prior_path)
        profile = build_global_prior_profile(parsed.query, prior, bpp, args)
        identity_values, gap_values = global_prior_identity_gap_values(
            prior,
            parsed.length,
            float(getattr(args, "global_prior_min_count", 5.0)),
        )
        return SourceStats(
            seqs=[],
            profile=profile,
            identity_values=identity_values,
            gap_fraction_values=gap_values,
            global_prior=prior,
            bpp=bpp,
        )
    if args.stats_source == "alphabet_prior":
        alphabet_prior_path = getattr(args, "alphabet_prior_path", None)
        if not alphabet_prior_path:
            raise ValueError("--alphabet_prior_path is required when --stats_source alphabet_prior")
        prior = load_alphabet_inverse_prior(
            alphabet_prior_path,
            getattr(args, "alphabet_prior_mode", "family_balanced"),
        )
        return SourceStats(
            seqs=[],
            profile=build_alphabet_prior_profile(parsed.query, prior),
            identity_values=np.zeros(0, dtype=np.float64),
            gap_fraction_values=np.zeros(0, dtype=np.float64),
            alphabet_prior=prior,
            bpp=bpp,
        )

    source_seqs = select_source_seqs(parsed, args)
    profile = estimate_profile(source_seqs, pseudo=args.profile_pseudocount)
    identity_values = identity_values_to_query(source_seqs)
    gaps = gap_fraction_values(source_seqs)
    return SourceStats(seqs=source_seqs, profile=profile, identity_values=identity_values, gap_fraction_values=gaps)


def build_stat_fields(query: str, source_stats: SourceStats, args: argparse.Namespace) -> np.ndarray:
    query_fields = build_query_fields(query, args.same_prob, args.mutation_prob_each, args.gap_prob)
    query_probs = np.exp(query_fields)
    query_probs = query_probs / query_probs.sum(axis=1, keepdims=True)
    source_probs = source_stats.profile
    fields = args.profile_weight * np.log(source_probs + EPS) + args.query_weight * np.log(query_probs + EPS)
    fields -= fields.max(axis=1, keepdims=True)
    return fields


def is_canonical(a: str, b: str) -> bool:
    return (a, b) in CANONICAL_PAIRS


def is_wobble(a: str, b: str) -> bool:
    return (a, b) in WOBBLE_PAIRS


def is_valid_pair(a: str, b: str) -> bool:
    return is_canonical(a, b) or is_wobble(a, b)


def compatibility_score(a_idx: int, b_idx: int, qi_idx: int, qj_idx: int) -> float:
    a = ALPHABET[a_idx]
    b = ALPHABET[b_idx]
    qi = ALPHABET[qi_idx]
    qj = ALPHABET[qj_idx]

    if a_idx == GAP_IDX or b_idx == GAP_IDX:
        return -1.35

    a_same = a_idx == qi_idx
    b_same = b_idx == qj_idx
    one_changed = a_same != b_same
    both_changed = (not a_same) and (not b_same)
    query_pair = is_valid_pair(qi, qj)

    if query_pair:
        if is_canonical(a, b):
            score = 1.25
        elif is_wobble(a, b):
            score = 0.90
        else:
            score = -1.05

        if both_changed and is_valid_pair(a, b):
            score += 0.40
        elif one_changed:
            score -= 0.65
        elif a_same and b_same:
            score += 0.20
        return score

    if a_same and b_same:
        return 0.35
    if both_changed:
        return 0.75 if is_valid_pair(a, b) else 0.40
    if one_changed:
        return -0.55
    return -0.25


def build_compatibilities(query: str, edges: EdgeSet) -> np.ndarray:
    query_idx = np.asarray([ALPHABET.find(ch) if ch in ALPHABET else GAP_IDX for ch in query], dtype=np.int8)
    compat = np.empty((edges.size, len(ALPHABET), len(ALPHABET)), dtype=np.float64)
    for edge_idx, (i, j) in enumerate(zip(edges.i, edges.j)):
        qi = int(query_idx[i])
        qj = int(query_idx[j])
        for a in range(len(ALPHABET)):
            for b in range(len(ALPHABET)):
                compat[edge_idx, a, b] = compatibility_score(a, b, qi, qj)
        compat[edge_idx] -= compat[edge_idx].mean()
    return compat


def compute_edge_log_odds(
    source_seqs: list[str],
    profile: np.ndarray,
    edges: EdgeSet,
    pseudo: float,
    clip_value: float,
) -> np.ndarray:
    msa_num = seqs_to_numeric(align_sequences(source_seqs))
    n_states = len(ALPHABET)
    compat = np.zeros((edges.size, n_states, n_states), dtype=np.float64)
    denom = msa_num.shape[0] + n_states * n_states * pseudo
    for edge_idx, (i, j) in enumerate(zip(edges.i, edges.j)):
        pair_index = msa_num[:, i].astype(np.int16) * n_states + msa_num[:, j].astype(np.int16)
        counts = np.bincount(pair_index, minlength=n_states * n_states).astype(np.float64).reshape(n_states, n_states)
        joint = (counts + pseudo) / denom
        expected = np.outer(profile[i], profile[j])
        log_odds = np.log((joint + EPS) / (expected + EPS))
        log_odds = np.clip(log_odds, -clip_value, clip_value)
        compat[edge_idx] = log_odds - log_odds.mean()
    return compat


def build_global_prior_compatibilities(
    query: str,
    edges: EdgeSet,
    source_stats: SourceStats,
    args: argparse.Namespace,
) -> np.ndarray:
    prior = source_stats.global_prior
    if prior is None:
        raise ValueError("Global prior compatibilities require SourceStats.global_prior")

    mode = getattr(args, "global_prior_mode", "auto")
    min_count = float(getattr(args, "global_prior_min_count", 5.0))
    bpp = source_stats.bpp
    sep_edges = prior.array("separation_bin_edges")
    bpp_edges = prior.array("bpp_pair_bin_edges")
    strength_edges = prior.array("edge_strength_bin_edges")

    cond = prior.array("pair_log_odds_cond")
    cond_counts = prior.array("pair_counts_cond")
    no_edge = prior.array("pair_log_odds_no_edge")
    no_edge_counts = prior.array("pair_counts_no_edge")
    no_bpp = prior.array("pair_log_odds_no_bpp")
    no_bpp_counts = prior.array("pair_counts_no_bpp")
    qpair_only = prior.array("pair_log_odds_qpair")
    qpair_counts = prior.array("pair_counts_qpair")
    global_tensor = prior.array("pair_log_odds_global")

    compat = np.empty((edges.size, len(ALPHABET), len(ALPHABET)), dtype=np.float64)
    fallback_counts = {"conditioned": 0, "no_edge": 0, "no_bpp": 0, "qpair": 0, "global": 0}
    query_idx = np.asarray([_query_base_index(ch) for ch in query], dtype=np.int8)

    for edge_idx, (i, j) in enumerate(zip(edges.i, edges.j)):
        qi = int(query_idx[i])
        qj = int(query_idx[j])
        qpair = _query_pair_code(qi, qj)
        sep_bin = _bin_index(abs(int(j) - int(i)), sep_edges)
        if bpp is not None and bpp.shape[0] == len(query):
            bpp_bin = _bin_index(float(bpp[int(i), int(j)]), bpp_edges)
        else:
            bpp_bin = 0
        strength_bin = _bin_index(float(edges.target_norm[edge_idx]), strength_edges)

        if mode == "global":
            tensor = global_tensor
            fallback_counts["global"] += 1
        elif float(cond_counts[qpair, sep_bin, bpp_bin, strength_bin]) >= min_count or (
            mode == "conditioned" and float(cond_counts[qpair, sep_bin, bpp_bin, strength_bin]) > 0
        ):
            tensor = cond[qpair, sep_bin, bpp_bin, strength_bin]
            fallback_counts["conditioned"] += 1
        elif float(no_edge_counts[qpair, sep_bin, bpp_bin]) >= min_count:
            tensor = no_edge[qpair, sep_bin, bpp_bin]
            fallback_counts["no_edge"] += 1
        elif float(no_bpp_counts[qpair, sep_bin]) >= min_count:
            tensor = no_bpp[qpair, sep_bin]
            fallback_counts["no_bpp"] += 1
        elif float(qpair_counts[qpair]) >= min_count:
            tensor = qpair_only[qpair]
            fallback_counts["qpair"] += 1
        else:
            tensor = global_tensor
            fallback_counts["global"] += 1
        compat[edge_idx] = tensor

    prior.last_pair_fallback_counts = fallback_counts
    compat -= compat.mean(axis=(1, 2), keepdims=True)
    return compat


def build_stat_compatibilities(
    query: str,
    edges: EdgeSet,
    source_stats: SourceStats,
    args: argparse.Namespace,
) -> np.ndarray:
    if args.stat_variant == "profile":
        return np.zeros((edges.size, len(ALPHABET), len(ALPHABET)), dtype=np.float64)
    if args.stats_source == "global_prior":
        log_odds = build_global_prior_compatibilities(query, edges, source_stats, args)
        rule = build_compatibilities(query, edges)
        compat = args.pair_tensor_weight * log_odds + args.rule_mix * rule
        compat -= compat.mean(axis=(1, 2), keepdims=True)
        return compat
    if args.stats_source == "alphabet_prior":
        return build_compatibilities(query, edges)

    log_odds = compute_edge_log_odds(
        source_stats.seqs,
        source_stats.profile,
        edges,
        args.pair_pseudocount,
        args.pair_log_odds_clip,
    )
    rule = build_compatibilities(query, edges)
    compat = args.pair_tensor_weight * log_odds + args.rule_mix * rule
    compat -= compat.mean(axis=(1, 2), keepdims=True)
    return compat


def build_adjacency(length: int, edges: EdgeSet) -> list[list[tuple[int, int, int]]]:
    adjacency: list[list[tuple[int, int, int]]] = [[] for _ in range(length)]
    for edge_idx, (i, j) in enumerate(zip(edges.i, edges.j)):
        adjacency[int(i)].append((int(edge_idx), int(j), 0))
        adjacency[int(j)].append((int(edge_idx), int(i), 1))
    return adjacency


def sample_from_logits(logits: np.ndarray, rng: np.random.Generator, temperature: float) -> np.ndarray:
    scaled = logits / max(temperature, EPS)
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    probs = np.exp(scaled)
    probs = probs / probs.sum(axis=1, keepdims=True)
    cdf = np.cumsum(probs, axis=1)
    draws = rng.random(logits.shape[0])
    return (draws[:, None] > cdf).sum(axis=1).astype(np.int8)


def sample_query_prior(depth: int, fields: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    probs = np.exp(fields)
    probs = probs / probs.sum(axis=1, keepdims=True)
    length = fields.shape[0]
    msa = np.empty((depth, length), dtype=np.int8)
    for pos in range(length):
        cdf = np.cumsum(probs[pos])
        msa[:, pos] = (rng.random(depth)[:, None] > cdf).sum(axis=1)
    return msa


def sample_potts(
    depth: int,
    fields: np.ndarray,
    edges: EdgeSet,
    compat: np.ndarray,
    adjacency: list[list[tuple[int, int, int]]],
    weights: np.ndarray,
    rng: np.random.Generator,
    sweeps: int,
    gamma: float,
    temperature: float,
) -> np.ndarray:
    states = sample_query_prior(depth, fields, rng)
    length = fields.shape[0]

    for _ in range(sweeps):
        for pos in rng.permutation(length):
            logits = np.tile(fields[pos], (depth, 1))
            for edge_idx, neighbor, orientation in adjacency[pos]:
                strength = gamma * weights[edge_idx]
                if strength <= 0:
                    continue
                neighbor_states = states[:, neighbor]
                k_mat = compat[edge_idx]
                if orientation == 0:
                    logits += strength * k_mat[:, neighbor_states].T
                else:
                    logits += strength * k_mat[neighbor_states, :]
            states[:, pos] = sample_from_logits(logits, rng, temperature)

    return states


def numeric_identity_to_query(msa_num: np.ndarray, query: str) -> np.ndarray:
    query_idx = np.asarray([ALPHABET.find(ch) if ch in ALPHABET else GAP_IDX for ch in query], dtype=np.int8)
    mask = query_idx != GAP_IDX
    if not np.any(mask):
        return np.zeros(msa_num.shape[0], dtype=np.float64)
    return (msa_num[:, mask] == query_idx[mask]).mean(axis=1)


def numeric_gap_fraction(msa_num: np.ndarray) -> np.ndarray:
    if msa_num.size == 0:
        return np.asarray([], dtype=np.float64)
    return (msa_num == GAP_IDX).mean(axis=1)


def select_by_identity_distribution(
    candidates: np.ndarray,
    source_stats: SourceStats | None,
    query: str,
    target_depth: int,
    rng: np.random.Generator,
    match_gap_distribution: bool = False,
) -> np.ndarray:
    if source_stats is None or source_stats.identity_values.size == 0 or candidates.shape[0] <= target_depth:
        return candidates[:target_depth]

    candidate_identity = numeric_identity_to_query(candidates, query)
    target_identity = rng.choice(source_stats.identity_values, size=target_depth, replace=True)
    candidate_gap = numeric_gap_fraction(candidates)
    target_gap = None
    if (
        match_gap_distribution
        and source_stats.gap_fraction_values is not None
        and source_stats.gap_fraction_values.size > 0
    ):
        target_gap = rng.choice(source_stats.gap_fraction_values, size=target_depth, replace=True)

    candidate_order = np.argsort(candidate_identity)
    candidate_sorted = candidate_identity[candidate_order]
    target_order = np.argsort(target_identity)
    selected_positions: list[int] = []
    used: set[int] = set()
    for target_rank_pos, target in enumerate(target_identity[target_order]):
        gap_target = None if target_gap is None else float(target_gap[target_order[target_rank_pos]])
        center = int(np.searchsorted(candidate_sorted, target))
        best_pos = None
        best_dist = float("inf")
        for radius in range(len(candidate_order)):
            left = center - radius
            right = center + radius
            for pos in (left, right):
                if pos < 0 or pos >= len(candidate_order) or pos in used:
                    continue
                candidate_idx = int(candidate_order[pos])
                dist = abs(float(candidate_sorted[pos] - target))
                if gap_target is not None:
                    dist += 0.5 * abs(float(candidate_gap[candidate_idx] - gap_target))
                if dist < best_dist:
                    best_dist = dist
                    best_pos = pos
            if best_pos is not None:
                break
        if best_pos is None:
            break
        used.add(best_pos)
        selected_positions.append(best_pos)
    if len(selected_positions) < target_depth:
        leftovers = [pos for pos in range(len(candidate_order)) if pos not in used]
        selected_positions.extend(leftovers[: target_depth - len(selected_positions)])
    selected = candidate_order[np.asarray(selected_positions[:target_depth], dtype=np.int32)]

    # Preserve the sampled target identity ordering enough to avoid deterministic row sorting.
    reordered = np.empty_like(selected)
    reordered[target_order] = selected
    return candidates[reordered]


def sample_inverse_states(
    depth: int,
    fields: np.ndarray,
    edges: EdgeSet,
    compat: np.ndarray,
    adjacency: list[list[tuple[int, int, int]]],
    weights: np.ndarray,
    rng: np.random.Generator,
    args: argparse.Namespace,
    query: str,
    source_stats: SourceStats | None = None,
) -> np.ndarray:
    needs_identity_selection = args.sampler_mode == "stat_field" and args.stat_variant == "profile_pair_identity"
    sample_depth = depth * args.identity_candidate_multiplier if needs_identity_selection else depth
    if args.sampler_mode == "stat_field" and args.stat_variant == "profile":
        candidates = sample_query_prior(sample_depth, fields, rng)
    else:
        candidates = sample_potts(
            depth=sample_depth,
            fields=fields,
            edges=edges,
            compat=compat,
            adjacency=adjacency,
            weights=weights,
            rng=rng,
            sweeps=args.gibbs_sweeps,
            gamma=args.gamma,
            temperature=args.temperature,
        )
    if needs_identity_selection:
        return select_by_identity_distribution(
            candidates,
            source_stats,
            query,
            depth,
            rng,
            match_gap_distribution=bool(getattr(args, "match_row_gap_distribution", False)),
        )
    return candidates[:depth]


def distribution_quantile_mae(values: np.ndarray, target_values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    target_values = np.asarray(target_values, dtype=np.float64)
    if values.size == 0 or target_values.size == 0:
        return 0.0
    quantiles = [0.1, 0.5, 0.9]
    return float(np.mean(np.abs(np.quantile(values, quantiles) - np.quantile(target_values, quantiles))))


def duplicate_row_fraction(msa_num: np.ndarray) -> float:
    if msa_num.shape[0] <= 1:
        return 0.0
    unique = np.unique(msa_num, axis=0).shape[0]
    return float(1.0 - unique / max(msa_num.shape[0], 1))


def upper_weighted_mse(target_mi: np.ndarray, generated_mi: np.ndarray, edges: EdgeSet) -> float:
    length = min(target_mi.shape[0], generated_mi.shape[0])
    if length < 2:
        return 0.0
    scale = max(float(edges.scale), EPS)
    target_norm = target_mi[:length, :length] / scale
    generated_norm = generated_mi[:length, :length] / scale
    diff2 = (target_norm - generated_norm) ** 2
    upper = np.triu(np.ones((length, length), dtype=bool), k=1)
    weights = np.ones((length, length), dtype=np.float64)
    valid_edge = (edges.i < length) & (edges.j < length)
    weights[edges.i[valid_edge], edges.j[valid_edge]] = 3.0
    weighted = diff2[upper] * weights[upper]
    return float(weighted.sum() / max(weights[upper].sum(), EPS))


def inverse_msa_fitness(
    msa_num: np.ndarray,
    query: str,
    target_mi: np.ndarray,
    edges: EdgeSet,
    source_stats: SourceStats | None,
    args: argparse.Namespace,
) -> tuple[float, dict[str, float], np.ndarray]:
    seqs = [query] + numeric_to_seqs(msa_num)
    generated_mi = compute_mi_apc(seqs)
    map_mse = upper_weighted_mse(target_mi, generated_mi, edges)
    top_overlap = top_l_overlap(target_mi, generated_mi)

    identity_penalty = 0.0
    gap_penalty = 0.0
    if source_stats is not None:
        if source_stats.identity_values.size:
            identity_penalty = distribution_quantile_mae(
                numeric_identity_to_query(msa_num, query),
                source_stats.identity_values,
            )
        if source_stats.gap_fraction_values is not None and source_stats.gap_fraction_values.size:
            gap_penalty = distribution_quantile_mae(numeric_gap_fraction(msa_num), source_stats.gap_fraction_values)
    prior_penalty = 0.5 * (identity_penalty + gap_penalty)
    duplicate_penalty = duplicate_row_fraction(msa_num)

    loss = (
        float(getattr(args, "ea_map_weight", 1.0)) * map_mse
        + float(getattr(args, "ea_top_weight", 0.25)) * (1.0 - top_overlap)
        + float(getattr(args, "ea_prior_weight", 0.10)) * prior_penalty
        + float(getattr(args, "ea_duplicate_weight", 0.05)) * duplicate_penalty
    )
    components = {
        "loss": float(loss),
        "map_mse": float(map_mse),
        "top_l_overlap": float(top_overlap),
        "top_loss": float(1.0 - top_overlap),
        "prior_penalty": float(prior_penalty),
        "identity_penalty": float(identity_penalty),
        "gap_penalty": float(gap_penalty),
        "duplicate_penalty": float(duplicate_penalty),
    }
    return float(loss), components, generated_mi


def numeric_profile_from_msa(msa_num: np.ndarray, query: str, pseudo: float = 1.0) -> np.ndarray:
    query_idx = np.asarray([ALPHABET.find(ch) if ch in ALPHABET else GAP_IDX for ch in query], dtype=np.int8)
    combined = np.vstack([query_idx[None, :], msa_num.astype(np.int8, copy=False)])
    length = combined.shape[1]
    profile = np.zeros((length, len(ALPHABET)), dtype=np.float64)
    for state in range(len(ALPHABET)):
        profile[:, state] = (combined == state).sum(axis=0)
    profile = (profile + pseudo) / (combined.shape[0] + len(ALPHABET) * pseudo)
    return profile


def repair_profile_penalties(msa_num: np.ndarray, query: str, source_stats: SourceStats | None) -> tuple[float, float]:
    if source_stats is None or source_stats.profile.size == 0:
        return 0.0, 0.0
    profile = numeric_profile_from_msa(msa_num, query)
    length = min(profile.shape[0], source_stats.profile.shape[0])
    if length == 0:
        return 0.0, 0.0
    diff = np.abs(profile[:length] - source_stats.profile[:length])
    profile_penalty = float(np.mean(diff.sum(axis=1)))
    gap_penalty = float(np.mean(diff[:, GAP_IDX]))
    return profile_penalty, gap_penalty


def numeric_alphabet_probs(msa_num: np.ndarray, query: str) -> np.ndarray:
    query_idx = np.asarray([ALPHABET.find(ch) if ch in ALPHABET else GAP_IDX for ch in query], dtype=np.int8)
    combined = np.concatenate([query_idx, msa_num.reshape(-1).astype(np.int8, copy=False)])
    counts = np.bincount(combined.astype(np.int16), minlength=len(ALPHABET)).astype(np.float64)
    return counts / max(float(counts.sum()), EPS)


def repair_msa_fitness(
    msa_num: np.ndarray,
    query: str,
    target_mi: np.ndarray,
    edges: EdgeSet,
    source_stats: SourceStats | None,
    args: argparse.Namespace,
) -> tuple[float, dict[str, float], np.ndarray]:
    generated_mi = compute_mi_apc([query] + numeric_to_seqs(msa_num))
    pearson = matrix_pearson(target_mi, generated_mi)
    spearman = matrix_spearman(target_mi, generated_mi)
    top_overlap = top_l_overlap(target_mi, generated_mi)
    edge_mae = selected_edge_mae(target_mi, generated_mi, edges)

    profile_penalty, gap_profile_penalty = repair_profile_penalties(msa_num, query, source_stats)
    alphabet_penalty = 0.0
    if source_stats is not None and source_stats.alphabet_prior is not None:
        alphabet_penalty = float(
            np.mean(np.abs(numeric_alphabet_probs(msa_num, query) - source_stats.alphabet_prior.probs))
        )
        profile_penalty = alphabet_penalty
        gap_profile_penalty = 0.0
    identity_penalty = 0.0
    row_gap_penalty = 0.0
    if source_stats is not None:
        if source_stats.alphabet_prior is None and source_stats.identity_values.size:
            identity_penalty = distribution_quantile_mae(
                numeric_identity_to_query(msa_num, query),
                source_stats.identity_values,
            )
        if (
            source_stats.alphabet_prior is None
            and source_stats.gap_fraction_values is not None
            and source_stats.gap_fraction_values.size
        ):
            row_gap_penalty = distribution_quantile_mae(numeric_gap_fraction(msa_num), source_stats.gap_fraction_values)
    gap_penalty = 0.5 * (gap_profile_penalty + row_gap_penalty)
    duplicate_penalty = duplicate_row_fraction(msa_num)

    loss = (
        float(getattr(args, "repair_corr_weight", 2.0)) * (1.0 - pearson)
        + float(getattr(args, "repair_edge_weight", 1.0)) * edge_mae
        + float(getattr(args, "repair_top_weight", 0.5)) * (1.0 - top_overlap)
        + float(getattr(args, "repair_profile_weight", 0.05)) * profile_penalty
        + float(getattr(args, "repair_gap_weight", 0.05)) * gap_penalty
        + float(getattr(args, "repair_identity_weight", 0.03)) * identity_penalty
        + float(getattr(args, "repair_duplicate_weight", 0.02)) * duplicate_penalty
    )
    components = {
        "loss": float(loss),
        "pearson": float(pearson),
        "spearman": float(spearman),
        "selected_edge_mae": float(edge_mae),
        "top_l_overlap": float(top_overlap),
        "profile_penalty": float(profile_penalty),
        "alphabet_penalty": float(alphabet_penalty),
        "gap_penalty": float(gap_penalty),
        "identity_penalty": float(identity_penalty),
        "duplicate_penalty": float(duplicate_penalty),
    }
    return float(loss), components, generated_mi


def make_edge_set_from_pairs(target_mi: np.ndarray, pairs: list[tuple[int, int]], scale: float | None = None) -> EdgeSet:
    length = target_mi.shape[0]
    seen: set[tuple[int, int]] = set()
    clean: list[tuple[int, int]] = []
    for i, j in pairs:
        i = int(i)
        j = int(j)
        if i == j or i < 0 or j < 0 or i >= length or j >= length:
            continue
        if i > j:
            i, j = j, i
        pair = (i, j)
        if pair not in seen:
            seen.add(pair)
            clean.append(pair)
    if not clean:
        return EdgeSet(
            i=np.zeros(0, dtype=np.int32),
            j=np.zeros(0, dtype=np.int32),
            target_norm=np.zeros(0, dtype=np.float64),
            scale=float(scale or target_scale(target_mi)),
        )
    edge_scale = float(scale or target_scale(target_mi))
    edge_i = np.asarray([p[0] for p in clean], dtype=np.int32)
    edge_j = np.asarray([p[1] for p in clean], dtype=np.int32)
    return EdgeSet(i=edge_i, j=edge_j, target_norm=target_mi[edge_i, edge_j] / max(edge_scale, EPS), scale=edge_scale)


def select_repair_edges(
    target_mi: np.ndarray,
    generated_mi: np.ndarray | None,
    repair_edge_factor: float,
    control_edge_factor: float,
) -> EdgeSet:
    length = target_mi.shape[0]
    scale = target_scale(target_mi)
    target_edges = select_edges(target_mi, repair_edge_factor)
    pairs = [(int(i), int(j)) for i, j in zip(target_edges.i, target_edges.j)]
    if generated_mi is not None and control_edge_factor > 0:
        target_top = set(pairs)
        values, idx = upper_values(generated_mi[:length, :length])
        control_k = min(max(1, int(round(control_edge_factor * length))), values.size)
        order = np.argsort(values)[::-1]
        added = 0
        target_values, _ = upper_values(target_mi)
        low_target_cutoff = float(np.percentile(target_values, 70)) if target_values.size else 0.0
        for n in order:
            i = int(idx[0][n])
            j = int(idx[1][n])
            pair = (i, j) if i < j else (j, i)
            if pair in target_top:
                continue
            if target_mi[pair] > low_target_cutoff:
                continue
            pairs.append(pair)
            target_top.add(pair)
            added += 1
            if added >= control_k:
                break
    return make_edge_set_from_pairs(target_mi, pairs, scale)


def apc_cache_from_mi(mi: np.ndarray) -> tuple[np.ndarray, float]:
    length = mi.shape[0]
    denom = max(length - 1, 1)
    row_mean = mi.sum(axis=1) / denom
    total_mean = float(mi.sum() / max(length * (length - 1), 1))
    return row_mean, total_mean


def pair_raw_mi(col_i: np.ndarray, col_j: np.ndarray) -> float:
    n_seq = int(col_i.size)
    n_states = len(ALPHABET)
    if n_seq < 2:
        return 0.0
    pseudo = 1.0 / max(n_seq, 1)
    counts_i = np.bincount(col_i.astype(np.int16), minlength=n_states).astype(np.float64)
    counts_j = np.bincount(col_j.astype(np.int16), minlength=n_states).astype(np.float64)
    fi = (counts_i + pseudo) / (n_seq + n_states * pseudo)
    fj = (counts_j + pseudo) / (n_seq + n_states * pseudo)
    hi = -np.sum(fi * np.log2(fi + EPS))
    hj = -np.sum(fj * np.log2(fj + EPS))
    pair_index = col_i.astype(np.int16) * n_states + col_j.astype(np.int16)
    joint = np.bincount(pair_index, minlength=n_states * n_states).astype(np.float64).reshape(n_states, n_states)
    fij = (joint + pseudo) / (n_seq + n_states * n_states * pseudo)
    hij = -np.sum(fij * np.log2(fij + EPS))
    return float(max(hi + hj - hij, 0.0))


def compute_pair_mi_apc(
    msa_num: np.ndarray,
    query_idx: np.ndarray,
    i: int,
    j: int,
    row_mean: np.ndarray | None = None,
    total_mean: float | None = None,
) -> float:
    col_i = np.concatenate([query_idx[[i]], msa_num[:, i].astype(np.int8, copy=False)])
    col_j = np.concatenate([query_idx[[j]], msa_num[:, j].astype(np.int8, copy=False)])
    mi = pair_raw_mi(col_i, col_j)
    if row_mean is None or total_mean is None or total_mean <= EPS:
        return mi
    apc = mi - float(row_mean[i] * row_mean[j] / total_mean)
    return float(max(apc, 0.0))


def compute_selected_edge_mi(msa_num: np.ndarray, query: str, edges: EdgeSet) -> np.ndarray:
    query_idx = np.asarray([ALPHABET.find(ch) if ch in ALPHABET else GAP_IDX for ch in query], dtype=np.int8)
    values = np.zeros(edges.size, dtype=np.float64)
    for edge_idx, (i, j) in enumerate(zip(edges.i, edges.j)):
        col_i = np.concatenate([query_idx[[int(i)]], msa_num[:, int(i)].astype(np.int8, copy=False)])
        col_j = np.concatenate([query_idx[[int(j)]], msa_num[:, int(j)].astype(np.int8, copy=False)])
        values[edge_idx] = pair_raw_mi(col_i, col_j)
    return values


def compute_selected_edge_apc_approx(
    raw_mi: np.ndarray,
    edges: EdgeSet,
    row_mean: np.ndarray,
    total_mean: float,
) -> np.ndarray:
    if total_mean <= EPS or raw_mi.size == 0:
        return raw_mi.copy()
    apc = raw_mi - (row_mean[edges.i] * row_mean[edges.j]) / total_mean
    return np.maximum(apc, 0.0)


def softmax_probs(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - np.max(logits)
    probs = np.exp(logits)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= EPS:
        return np.full(logits.size, 1.0 / max(logits.size, 1), dtype=np.float64)
    return probs / total


def sample_pair_states_for_edge(
    field_i: np.ndarray,
    field_j: np.ndarray,
    compat_edge: np.ndarray,
    rng: np.random.Generator,
    n_rows: int,
    beta: float,
) -> tuple[np.ndarray, np.ndarray]:
    logits = field_i[:, None] + field_j[None, :] + float(beta) * compat_edge
    probs = softmax_probs(logits.reshape(-1))
    flat = rng.choice(len(ALPHABET) * len(ALPHABET), size=n_rows, replace=True, p=probs)
    return (flat // len(ALPHABET)).astype(np.int8), (flat % len(ALPHABET)).astype(np.int8)


def sample_single_states(field: np.ndarray, rng: np.random.Generator, n_rows: int) -> np.ndarray:
    probs = softmax_probs(field)
    return rng.choice(len(ALPHABET), size=n_rows, replace=True, p=probs).astype(np.int8)


def repair_row_count(depth: int, args: argparse.Namespace) -> int:
    if depth <= 0:
        return 0
    n_rows = int(round(depth * float(getattr(args, "repair_row_fraction", 0.06))))
    n_rows = max(int(getattr(args, "repair_min_rows", 16)), n_rows)
    n_rows = min(int(getattr(args, "repair_max_rows", 128)), n_rows, depth)
    return max(1, n_rows)


def column_pair_repair_move(
    msa_num: np.ndarray,
    query_idx: np.ndarray,
    edge_index: int,
    edges: EdgeSet,
    fields: np.ndarray,
    compat: np.ndarray,
    current_norm: float,
    row_mean: np.ndarray,
    total_mean: float,
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> tuple[np.ndarray, float]:
    candidate = msa_num.copy()
    depth = candidate.shape[0]
    n_rows = repair_row_count(depth, args)
    rows = rng.choice(depth, size=n_rows, replace=False)
    i = int(edges.i[edge_index])
    j = int(edges.j[edge_index])
    target_norm = float(edges.target_norm[edge_index])
    if target_norm >= current_norm:
        states_i, states_j = sample_pair_states_for_edge(
            fields[i],
            fields[j],
            compat[edge_index],
            rng,
            n_rows,
            float(getattr(args, "repair_pair_beta", 2.0)),
        )
        candidate[rows, i] = states_i
        candidate[rows, j] = states_j
    elif rng.random() < float(getattr(args, "repair_state_replace_prob", 0.6)):
        if rng.random() < 0.5:
            candidate[rows, i] = sample_single_states(fields[i], rng, n_rows)
        else:
            candidate[rows, j] = sample_single_states(fields[j], rng, n_rows)
    elif n_rows > 1:
        candidate[rows, j] = candidate[rows, j][rng.permutation(n_rows)]
    candidate_apc = compute_pair_mi_apc(candidate, query_idx, i, j, row_mean, total_mean)
    return candidate, float(candidate_apc / max(edges.scale, EPS))


def column_pair_sa_optimize(
    baseline_num: np.ndarray,
    target_mi: np.ndarray,
    fields: np.ndarray,
    args: argparse.Namespace,
    query: str,
    source_stats: SourceStats | None = None,
    rng: np.random.Generator | None = None,
) -> dict[str, object]:
    rng = rng or np.random.default_rng()
    query_idx = np.asarray([ALPHABET.find(ch) if ch in ALPHABET else GAP_IDX for ch in query], dtype=np.int8)
    fitness_edges = select_edges(target_mi, float(getattr(args, "repair_edge_factor", 8.0)))
    repair_edges = select_repair_edges(
        target_mi,
        None,
        float(getattr(args, "repair_edge_factor", 8.0)),
        0.0,
    )
    repair_compat = (
        build_stat_compatibilities(query, repair_edges, source_stats, args)
        if source_stats is not None and getattr(args, "sampler_mode", "mi_only") == "stat_field"
        else build_compatibilities(query, repair_edges)
    )
    baseline_loss, baseline_components, baseline_mi = repair_msa_fitness(
        baseline_num,
        query,
        target_mi,
        fitness_edges,
        source_stats,
        args,
    )
    current_num = baseline_num.copy()
    current_mi = baseline_mi
    current_loss = baseline_loss
    current_components = dict(baseline_components)
    best_num = current_num.copy()
    best_mi = current_mi
    best_loss = current_loss
    best_components = dict(current_components)
    row_mean, total_mean = apc_cache_from_mi(current_mi)
    selected_raw = compute_selected_edge_mi(current_num, query, repair_edges)
    selected_apc = compute_selected_edge_apc_approx(selected_raw, repair_edges, row_mean, total_mean)
    current_selected_norm = selected_apc / max(repair_edges.scale, EPS)

    steps = max(0, int(getattr(args, "repair_steps", 3000)))
    candidates_per_step = max(1, int(getattr(args, "repair_candidates_per_step", 4)))
    full_interval = max(1, int(getattr(args, "repair_full_eval_interval", 100)))
    start_temp = float(getattr(args, "repair_start_temperature", 0.05))
    end_temp = float(getattr(args, "repair_end_temperature", 0.005))
    control_factor = float(getattr(args, "repair_control_edge_factor", 2.0))
    history: list[dict[str, float]] = [
        {
            "step": 0.0,
            "accepted": 0.0,
            "current_loss": float(current_loss),
            "best_loss": float(best_loss),
            **best_components,
            "repair_edges": float(repair_edges.size),
        }
    ]
    accepted = 0

    for step in range(1, steps + 1):
        if repair_edges.size == 0:
            break
        progress = step / max(steps, 1)
        temperature = start_temp * ((end_temp / max(start_temp, EPS)) ** progress) if start_temp > 0 else 0.0
        errors = repair_edges.target_norm - current_selected_norm
        focus_count = min(max(1, int(getattr(args, "repair_focus_edges", 32))), repair_edges.size)
        focus_order = np.argsort(np.abs(errors))[::-1][:focus_count]
        edge_index = int(rng.choice(focus_order))
        current_local_loss = float(abs(errors[edge_index]))
        best_candidate: np.ndarray | None = None
        best_candidate_norm = float(current_selected_norm[edge_index])
        best_candidate_local_loss = current_local_loss
        for _ in range(candidates_per_step):
            candidate, candidate_norm = column_pair_repair_move(
                current_num,
                query_idx,
                edge_index,
                repair_edges,
                fields,
                repair_compat,
                current_selected_norm[edge_index],
                row_mean,
                total_mean,
                rng,
                args,
            )
            candidate_local_loss = float(abs(repair_edges.target_norm[edge_index] - candidate_norm))
            if candidate_local_loss < best_candidate_local_loss:
                best_candidate = candidate
                best_candidate_norm = candidate_norm
                best_candidate_local_loss = candidate_local_loss
        delta = best_candidate_local_loss - current_local_loss
        if best_candidate is not None and (delta <= 0.0 or (temperature > 0 and rng.random() < math.exp(-delta / max(temperature, EPS)))):
            current_num = best_candidate
            current_selected_norm[edge_index] = best_candidate_norm
            accepted += 1

        if step % full_interval == 0 or step == steps:
            current_loss, current_components, current_mi = repair_msa_fitness(
                current_num,
                query,
                target_mi,
                fitness_edges,
                source_stats,
                args,
            )
            if current_loss < best_loss:
                best_num = current_num.copy()
                best_mi = current_mi
                best_loss = current_loss
                best_components = dict(current_components)

            repair_edges = select_repair_edges(
                target_mi,
                current_mi,
                float(getattr(args, "repair_edge_factor", 8.0)),
                control_factor,
            )
            repair_compat = (
                build_stat_compatibilities(query, repair_edges, source_stats, args)
                if source_stats is not None and getattr(args, "sampler_mode", "mi_only") == "stat_field"
                else build_compatibilities(query, repair_edges)
            )
            row_mean, total_mean = apc_cache_from_mi(current_mi)
            selected_raw = compute_selected_edge_mi(current_num, query, repair_edges)
            selected_apc = compute_selected_edge_apc_approx(selected_raw, repair_edges, row_mean, total_mean)
            current_selected_norm = selected_apc / max(repair_edges.scale, EPS)
            history.append(
                {
                    "step": float(step),
                    "accepted": float(accepted),
                    "acceptance_rate": float(accepted / max(step, 1)),
                    "current_loss": float(current_loss),
                    "best_loss": float(best_loss),
                    **best_components,
                    "repair_edges": float(repair_edges.size),
                }
            )

    return {
        "msa_num": best_num,
        "generated_mi": best_mi,
        "baseline_generated_mi": baseline_mi,
        "history": history,
        "baseline_fitness": baseline_components,
        "best_fitness": best_components,
        "pool_size": int(baseline_num.shape[0]),
        "optimizer": "column_pair_sa",
    }


def random_unique_indices(pool_size: int, depth: int, rng: np.random.Generator) -> np.ndarray:
    if pool_size < depth:
        raise ValueError(f"EA row pool size {pool_size} is smaller than requested depth {depth}")
    return rng.choice(pool_size, size=depth, replace=False).astype(np.int32)


def crossover_indices(parent_a: np.ndarray, parent_b: np.ndarray, pool_size: int, depth: int, rng: np.random.Generator) -> np.ndarray:
    merged = np.unique(np.concatenate([parent_a, parent_b]).astype(np.int32))
    if merged.size >= depth:
        return rng.choice(merged, size=depth, replace=False).astype(np.int32)
    used = set(int(x) for x in merged)
    remaining = np.asarray([idx for idx in range(pool_size) if idx not in used], dtype=np.int32)
    fill = rng.choice(remaining, size=depth - merged.size, replace=False).astype(np.int32)
    return np.concatenate([merged, fill]).astype(np.int32)


def mutate_indices(indices: np.ndarray, pool_size: int, mutation_fraction: float, rng: np.random.Generator) -> np.ndarray:
    child = indices.copy().astype(np.int32)
    depth = child.size
    if depth == 0:
        return child
    if mutation_fraction <= 0.0:
        return child
    n_mut = min(depth, max(1, int(round(depth * mutation_fraction))))
    replace_pos = rng.choice(depth, size=n_mut, replace=False)
    used = set(int(x) for x in child)
    for pos in replace_pos:
        used.discard(int(child[pos]))
    available = np.asarray([idx for idx in range(pool_size) if idx not in used], dtype=np.int32)
    if available.size < n_mut:
        return child
    replacement = rng.choice(available, size=n_mut, replace=False).astype(np.int32)
    child[replace_pos] = replacement
    return child


def tournament_select(population: list[np.ndarray], losses: list[float], tournament_size: int, rng: np.random.Generator) -> np.ndarray:
    size = len(population)
    n_pick = min(max(1, tournament_size), size)
    choices = rng.choice(size, size=n_pick, replace=False)
    best = int(min(choices, key=lambda idx: losses[int(idx)]))
    return population[best]


def optimize_inverse_msa(
    baseline_num: np.ndarray,
    target_mi: np.ndarray,
    fields: np.ndarray,
    edges: EdgeSet,
    compat: np.ndarray,
    adjacency: list[list[tuple[int, int, int]]],
    weights: np.ndarray,
    rng: np.random.Generator,
    args: argparse.Namespace,
    query: str,
    source_stats: SourceStats | None = None,
) -> dict[str, object]:
    mode = getattr(args, "inverse_optimizer", "none")
    baseline_loss, baseline_components, baseline_mi = inverse_msa_fitness(
        baseline_num,
        query,
        target_mi,
        edges,
        source_stats,
        args,
    )
    if mode == "none":
        return {
            "msa_num": baseline_num,
            "generated_mi": baseline_mi,
            "baseline_generated_mi": baseline_mi,
            "history": [],
            "baseline_fitness": baseline_components,
            "best_fitness": baseline_components,
            "pool_size": int(baseline_num.shape[0]),
            "optimizer": "none",
        }

    depth = int(baseline_num.shape[0])
    best_num = baseline_num.copy()
    best_mi = baseline_mi
    best_loss = baseline_loss
    best_components = dict(baseline_components)
    history: list[dict[str, float]] = [
        {"generation": 0.0, "best_loss": float(best_loss), "mean_loss": float(best_loss), **best_components}
    ]

    if mode == "best_of":
        population_size = max(1, int(getattr(args, "ea_population", 8)))
        for candidate_idx in range(1, population_size):
            candidate = sample_inverse_states(
                depth=depth,
                fields=fields,
                edges=edges,
                compat=compat,
                adjacency=adjacency,
                weights=weights,
                rng=rng,
                args=args,
                query=query,
                source_stats=source_stats,
            )
            loss, components, generated_mi = inverse_msa_fitness(candidate, query, target_mi, edges, source_stats, args)
            if loss < best_loss:
                best_num = candidate.copy()
                best_mi = generated_mi
                best_loss = loss
                best_components = dict(components)
            history.append({"generation": float(candidate_idx), "best_loss": float(best_loss), **best_components})
        return {
            "msa_num": best_num,
            "generated_mi": best_mi,
            "baseline_generated_mi": baseline_mi,
            "history": history,
            "baseline_fitness": baseline_components,
            "best_fitness": best_components,
            "pool_size": int(depth),
            "optimizer": "best_of",
        }

    if mode == "column_pair_sa":
        return column_pair_sa_optimize(
            baseline_num=baseline_num,
            target_mi=target_mi,
            fields=fields,
            args=args,
            query=query,
            source_stats=source_stats,
            rng=rng,
        )

    if mode == "mi3_potts":
        from EvoRibo.mi3_potts_inverse import fit_mi3_potts_inverse

        if source_stats is not None and source_stats.alphabet_prior is not None:
            alphabet_probs = source_stats.alphabet_prior.probs
        elif source_stats is not None and source_stats.profile.size:
            alphabet_probs = source_stats.profile.mean(axis=0)
        else:
            alphabet_probs = np.full(len(ALPHABET), 1.0 / len(ALPHABET), dtype=np.float64)
        mi3_result = fit_mi3_potts_inverse(
            target_map=target_mi,
            query=query,
            alphabet_probs=alphabet_probs,
            synthetic_depth=depth,
            rng=rng,
            args=args,
        )
        generated_mi = compute_mi_apc([query] + numeric_to_seqs(mi3_result.msa_num))
        return {
            "msa_num": mi3_result.msa_num,
            "generated_mi": generated_mi,
            "baseline_generated_mi": baseline_mi,
            "history": mi3_result.history,
            "baseline_fitness": baseline_components,
            "best_fitness": mi3_result.best_fitness,
            "pool_size": int(depth),
            "optimizer": "mi3_potts",
            "target_bimarg": mi3_result.target_bimarg,
            "final_model_bimarg": mi3_result.final_model_bimarg,
            "final_J": mi3_result.final_J,
            "mi3_baseline_fitness": mi3_result.baseline_fitness,
        }

    if mode != "row_pool_ea":
        raise ValueError(f"Unknown inverse optimizer: {mode}")

    pool_multiplier = max(1.0, float(getattr(args, "ea_pool_multiplier", 5.0)))
    target_pool_size = max(depth, int(math.ceil(depth * pool_multiplier)))
    extra_depth = max(0, target_pool_size - depth)
    if extra_depth:
        extra_pool = sample_inverse_states(
            depth=extra_depth,
            fields=fields,
            edges=edges,
            compat=compat,
            adjacency=adjacency,
            weights=weights,
            rng=rng,
            args=args,
            query=query,
            source_stats=source_stats,
        )
        row_pool = np.concatenate([baseline_num, extra_pool], axis=0)
    else:
        row_pool = baseline_num.copy()
    pool_size = int(row_pool.shape[0])

    population_size = max(2, int(getattr(args, "ea_population", 8)))
    generations = max(0, int(getattr(args, "ea_generations", 10)))
    elite_count = min(max(1, int(getattr(args, "ea_elite", 2))), population_size)
    tournament_size = max(1, int(getattr(args, "ea_tournament", 3)))
    mutation_fraction = float(getattr(args, "ea_mutation_fraction", 0.12))

    population: list[np.ndarray] = [np.arange(depth, dtype=np.int32)]
    while len(population) < population_size:
        population.append(random_unique_indices(pool_size, depth, rng))

    for generation in range(1, generations + 1):
        evaluated: list[tuple[float, dict[str, float], np.ndarray, np.ndarray]] = []
        for individual in population:
            msa_num = row_pool[individual]
            loss, components, generated_mi = inverse_msa_fitness(msa_num, query, target_mi, edges, source_stats, args)
            evaluated.append((loss, components, generated_mi, individual))
            if loss < best_loss:
                best_num = msa_num.copy()
                best_mi = generated_mi
                best_loss = loss
                best_components = dict(components)

        evaluated.sort(key=lambda item: item[0])
        losses = [float(item[0]) for item in evaluated]
        history.append(
            {
                "generation": float(generation),
                "best_loss": float(best_loss),
                "generation_best_loss": float(evaluated[0][0]),
                "mean_loss": float(np.mean(losses)),
                **best_components,
            }
        )

        elites = [item[3].copy() for item in evaluated[:elite_count]]
        next_population = elites[:]
        sorted_population = [item[3] for item in evaluated]
        sorted_losses = [float(item[0]) for item in evaluated]
        while len(next_population) < population_size:
            parent_a = tournament_select(sorted_population, sorted_losses, tournament_size, rng)
            parent_b = tournament_select(sorted_population, sorted_losses, tournament_size, rng)
            child = crossover_indices(parent_a, parent_b, pool_size, depth, rng)
            child = mutate_indices(child, pool_size, mutation_fraction, rng)
            next_population.append(child)
        population = next_population

    return {
        "msa_num": best_num,
        "generated_mi": best_mi,
        "baseline_generated_mi": baseline_mi,
        "history": history,
        "baseline_fitness": baseline_components,
        "best_fitness": best_components,
        "pool_size": pool_size,
        "optimizer": "row_pool_ea",
    }


def rankdata_average(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 2:
        return 0.0
    x = x[finite]
    y = y[finite]
    x_std = float(x.std())
    y_std = float(y.std())
    if x_std <= EPS or y_std <= EPS:
        return 1.0 if np.allclose(x, y) else 0.0
    return float(np.corrcoef(x, y)[0, 1])


def matrix_pearson(a: np.ndarray, b: np.ndarray) -> float:
    length = min(a.shape[0], b.shape[0])
    idx = np.triu_indices(length, k=1)
    return safe_pearson(a[:length, :length][idx], b[:length, :length][idx])


def matrix_spearman(a: np.ndarray, b: np.ndarray) -> float:
    length = min(a.shape[0], b.shape[0])
    idx = np.triu_indices(length, k=1)
    av = a[:length, :length][idx]
    bv = b[:length, :length][idx]
    return safe_pearson(rankdata_average(av), rankdata_average(bv))


def top_edge_set(matrix: np.ndarray, top_k: int) -> set[tuple[int, int]]:
    values, idx = upper_values(matrix)
    top_k = min(max(top_k, 0), values.size)
    if top_k == 0:
        return set()
    order = np.argsort(values)[::-1][:top_k]
    return {(int(idx[0][n]), int(idx[1][n])) for n in order}


def top_l_overlap(target: np.ndarray, generated: np.ndarray) -> float:
    length = min(target.shape[0], generated.shape[0])
    top_k = min(length, length * (length - 1) // 2)
    if top_k <= 0:
        return 0.0
    target_set = top_edge_set(target[:length, :length], top_k)
    generated_set = top_edge_set(generated[:length, :length], top_k)
    if not target_set:
        return 0.0
    return float(len(target_set & generated_set) / len(target_set))


def selected_edge_mae(target_mi: np.ndarray, generated_mi: np.ndarray, edges: EdgeSet) -> float:
    target = target_mi[edges.i, edges.j] / edges.scale
    generated = generated_mi[edges.i, edges.j] / edges.scale
    return float(np.mean(np.abs(target - generated))) if target.size else 0.0


def calculate_neff(seqs: list[str], identity_threshold: float = 0.8, max_seqs: int = 500) -> float:
    seqs = align_sequences(seqs[:max_seqs])
    if len(seqs) <= 1:
        return float(len(seqs))
    msa_num = seqs_to_numeric(seqs)
    n_seq = msa_num.shape[0]
    similar_counts = np.ones(n_seq, dtype=np.float64)
    block = 128
    for start in range(0, n_seq, block):
        stop = min(start + block, n_seq)
        sim = (msa_num[start:stop, None, :] == msa_num[None, :, :]).mean(axis=2)
        similar_counts[start:stop] = np.maximum((sim >= identity_threshold).sum(axis=1), 1)
    return float(np.sum(1.0 / similar_counts))


def mean_identity_to_query(seqs: list[str]) -> float:
    seqs = align_sequences(seqs)
    if len(seqs) <= 1:
        return 1.0 if seqs else 0.0
    query = seqs[0]
    identities = []
    for seq in seqs[1:]:
        matches = 0
        total = 0
        for q_char, s_char in zip(query, seq):
            if q_char == "-":
                continue
            total += 1
            if q_char == s_char:
                matches += 1
        if total:
            identities.append(matches / total)
    return float(np.mean(identities)) if identities else 0.0


def gap_fraction(seqs: list[str]) -> float:
    total = sum(len(seq) for seq in seqs)
    if total == 0:
        return 0.0
    return float(sum(seq.count("-") for seq in seqs) / total)


def identity_quantiles(seqs: list[str]) -> dict[str, float]:
    values = identity_values_to_query(seqs)
    if values.size == 0:
        return {"identity_q10": 0.0, "identity_q50": 0.0, "identity_q90": 0.0}
    q10, q50, q90 = np.quantile(values, [0.1, 0.5, 0.9])
    return {"identity_q10": float(q10), "identity_q50": float(q50), "identity_q90": float(q90)}


def profile_l1(raw_seqs: list[str], generated_seqs: list[str]) -> float:
    raw_profile = estimate_profile(raw_seqs)
    generated_profile = estimate_profile(generated_seqs)
    return float(np.mean(np.abs(raw_profile - generated_profile).sum(axis=1)))


def profile_kl(raw_seqs: list[str], generated_seqs: list[str]) -> float:
    raw_profile = estimate_profile(raw_seqs)
    generated_profile = estimate_profile(generated_seqs)
    return float(np.mean(np.sum(raw_profile * np.log((raw_profile + EPS) / (generated_profile + EPS)), axis=1)))


def gap_profile_mae(raw_seqs: list[str], generated_seqs: list[str]) -> float:
    raw_profile = estimate_profile(raw_seqs)
    generated_profile = estimate_profile(generated_seqs)
    return float(np.mean(np.abs(raw_profile[:, GAP_IDX] - generated_profile[:, GAP_IDX])))


def joint_frequency_mae(raw_seqs: list[str], generated_seqs: list[str], edges: EdgeSet) -> float:
    raw_num = seqs_to_numeric(align_sequences(raw_seqs))
    gen_num = seqs_to_numeric(align_sequences(generated_seqs))
    n_states = len(ALPHABET)
    errors = []
    for i, j in zip(edges.i, edges.j):
        raw_pair = raw_num[:, i].astype(np.int16) * n_states + raw_num[:, j].astype(np.int16)
        gen_pair = gen_num[:, i].astype(np.int16) * n_states + gen_num[:, j].astype(np.int16)
        raw_counts = np.bincount(raw_pair, minlength=n_states * n_states).astype(np.float64)
        gen_counts = np.bincount(gen_pair, minlength=n_states * n_states).astype(np.float64)
        raw_freq = (raw_counts + 1.0) / (raw_num.shape[0] + n_states * n_states)
        gen_freq = (gen_counts + 1.0) / (gen_num.shape[0] + n_states * n_states)
        errors.append(np.mean(np.abs(raw_freq - gen_freq)))
    return float(np.mean(errors)) if errors else 0.0


def compare_to_target(target_mi: np.ndarray, generated_mi: np.ndarray, edges: EdgeSet) -> dict[str, float]:
    return {
        "pearson": matrix_pearson(target_mi, generated_mi),
        "spearman": matrix_spearman(target_mi, generated_mi),
        "top_l_overlap": top_l_overlap(target_mi, generated_mi),
        "selected_edge_mae": selected_edge_mae(target_mi, generated_mi, edges),
    }


def write_a3m(path: Path, query: str, synthetic: list[str], label: str) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(">query\n")
        handle.write(f"{query}\n")
        for idx, seq in enumerate(synthetic, start=1):
            handle.write(f">{label}_{idx:06d}\n")
            handle.write(f"{seq}\n")


def copy_standard_files(source_dir: Path, out_dir: Path, rna_id: str, calibrated_a3m: Path) -> None:
    """Write a NuFold-style nested case folder: <id>.a3m/.fasta/.ipknot.ss."""
    shutil.copyfile(calibrated_a3m, out_dir / f"{rna_id}.a3m")
    for suffix in (".fasta", ".ipknot.ss"):
        src = source_dir / f"{rna_id}{suffix}"
        if src.exists():
            shutil.copyfile(src, out_dir / f"{rna_id}{suffix}")


def save_json(path: Path, data: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def plot_comparison(
    target_mi: np.ndarray,
    random_mi: np.ndarray,
    calibrated_mi: np.ndarray,
    metrics: dict[str, object],
    out_path: Path,
) -> None:
    vmax = max(
        float(np.percentile(target_mi, 99)),
        float(np.percentile(random_mi, 99)),
        float(np.percentile(calibrated_mi, 99)),
        0.01,
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    panels = [
        ("Target raw MI+APC", target_mi),
        (
            "Random baseline\n"
            f"r={metrics['random']['pearson']:.3f}, top-L={metrics['random']['top_l_overlap']:.3f}",
            random_mi,
        ),
        (
            "Calibrated inverse\n"
            f"r={metrics['calibrated']['pearson']:.3f}, top-L={metrics['calibrated']['top_l_overlap']:.3f}",
            calibrated_mi,
        ),
    ]
    for ax, (title, matrix) in zip(axes, panels):
        im = ax.imshow(matrix, cmap="YlOrRd", vmin=0.0, vmax=vmax, aspect="equal")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Position")
        ax.set_ylabel("Position")
        fig.colorbar(im, ax=ax, shrink=0.75)
    fig.suptitle(str(metrics["rna_id"]), fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_summary(summary: list[dict[str, object]], out_path: Path) -> None:
    ids = [str(row["rna_id"]) for row in summary]
    x = np.arange(len(ids))
    width = 0.38

    rand_p = np.array([row["random"]["pearson"] for row in summary], dtype=float)
    cal_p = np.array([row["calibrated"]["pearson"] for row in summary], dtype=float)
    rand_mae = np.array([row["random"]["selected_edge_mae"] for row in summary], dtype=float)
    cal_mae = np.array([row["calibrated"]["selected_edge_mae"] for row in summary], dtype=float)
    rand_top = np.array([row["random"]["top_l_overlap"] for row in summary], dtype=float)
    cal_top = np.array([row["calibrated"]["top_l_overlap"] for row in summary], dtype=float)
    neff = np.array([row["calibrated_stats"]["neff"] for row in summary], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    ax = axes[0, 0]
    ax.bar(x - width / 2, rand_p, width, label="random", color="#9aa0a6")
    ax.bar(x + width / 2, cal_p, width, label="calibrated", color="#d95f02")
    ax.set_title("MI+APC Pearson")
    ax.set_ylim(-0.1, 1.0)
    ax.legend()

    ax = axes[0, 1]
    ax.bar(x - width / 2, rand_top, width, label="random", color="#9aa0a6")
    ax.bar(x + width / 2, cal_top, width, label="calibrated", color="#1b9e77")
    ax.set_title("Top-L edge overlap")
    ax.set_ylim(0.0, 1.0)
    ax.legend()

    ax = axes[1, 0]
    ax.bar(x - width / 2, rand_mae, width, label="random", color="#9aa0a6")
    ax.bar(x + width / 2, cal_mae, width, label="calibrated", color="#7570b3")
    ax.set_title("Selected-edge MAE, normalized")
    ax.legend()

    ax = axes[1, 1]
    ax.bar(x, neff, color="#66a61e")
    ax.set_title("Calibrated synthetic Neff")

    for ax in axes.ravel():
        ax.set_xticks(x)
        ax.set_xticklabels(ids, rotation=45, ha="right")
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def flatten_summary_row(row: dict[str, object]) -> dict[str, object]:
    flat: dict[str, object] = {
        "rna_id": row["rna_id"],
        "raw_depth": row["raw_depth"],
        "length": row["length"],
        "selected_edges": row["selected_edges"],
        "target_scale": row["target_scale"],
    }
    for prefix in ("random", "calibrated"):
        metrics = row[prefix]
        for key, value in metrics.items():
            flat[f"{prefix}_{key}"] = value
    for prefix in ("random_stats", "calibrated_stats"):
        stats = row[prefix]
        for key, value in stats.items():
            flat[f"{prefix}_{key}"] = value
    for prefix in ("random_raw_similarity", "calibrated_raw_similarity"):
        stats = row.get(prefix, {})
        for key, value in stats.items():
            flat[f"{prefix}_{key}"] = value
    flat["mae_improved"] = row["calibrated"]["selected_edge_mae"] < row["random"]["selected_edge_mae"]
    flat["pearson_improved"] = row["calibrated"]["pearson"] > row["random"]["pearson"]
    flat["top_l_improved"] = row["calibrated"]["top_l_overlap"] > row["random"]["top_l_overlap"]
    return flat


def write_summary_csv(path: Path, summary: list[dict[str, object]]) -> None:
    rows = [flatten_summary_row(row) for row in summary]
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_rna_ids(value: str) -> list[str]:
    if value.strip().lower() == "auto":
        return AUTO_RNA_IDS
    return [part.strip() for part in value.split(",") if part.strip()]


def has_nested_cases(msa_dir: Path) -> bool:
    return any((child / f"{child.name}.a3m").exists() for child in msa_dir.iterdir() if child.is_dir())


def effective_input_layout(args: argparse.Namespace) -> str:
    if args.input_layout != "auto":
        return args.input_layout
    msa_dir = Path(args.msa_dir)
    return "nested" if msa_dir.exists() and has_nested_cases(msa_dir) else "flat"


def discover_rna_ids(args: argparse.Namespace, layout: str) -> list[str]:
    if args.rna_ids.strip().lower() != "auto":
        return parse_rna_ids(args.rna_ids)
    if layout == "nested":
        msa_dir = Path(args.msa_dir)
        return sorted(child.name for child in msa_dir.iterdir() if child.is_dir() and (child / f"{child.name}.a3m").exists())
    return AUTO_RNA_IDS


def resolve_msa_path(msa_dir: Path, rna_id: str, layout: str) -> Path:
    if layout == "nested":
        return msa_dir / rna_id / f"{rna_id}.a3m"
    return msa_dir / f"{rna_id}.a3m"


def msa_stats(seqs: list[str]) -> dict[str, float]:
    stats = {
        "depth": float(len(seqs)),
        "neff": calculate_neff(seqs),
        "mean_identity_to_query": mean_identity_to_query(seqs),
        "gap_fraction": gap_fraction(seqs),
    }
    stats.update(identity_quantiles(seqs))
    return stats


def raw_similarity_stats(raw_seqs: list[str], generated_seqs: list[str], edges: EdgeSet) -> dict[str, float]:
    return {
        "profile_l1": profile_l1(raw_seqs, generated_seqs),
        "profile_kl": profile_kl(raw_seqs, generated_seqs),
        "gap_profile_mae": gap_profile_mae(raw_seqs, generated_seqs),
        "joint_frequency_mae": joint_frequency_mae(raw_seqs, generated_seqs, edges),
        "identity_q50_abs_error": abs(
            identity_quantiles(raw_seqs)["identity_q50"] - identity_quantiles(generated_seqs)["identity_q50"]
        ),
    }


def process_one(args: argparse.Namespace, rna_id: str, rng: np.random.Generator) -> dict[str, object]:
    layout = effective_input_layout(args)
    msa_path = resolve_msa_path(Path(args.msa_dir), rna_id, layout)
    if not msa_path.exists():
        raise FileNotFoundError(f"Missing MSA: {msa_path}")

    parsed = parse_a3m(msa_path)
    if parsed.depth == 0 or parsed.length == 0:
        raise ValueError(f"Empty or invalid MSA: {msa_path}")

    out_dir = Path(args.out_dir) / rna_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[{rna_id}] raw_depth={parsed.depth}, L={parsed.length}")
    print("  computing target MI+APC...")
    target_mi = compute_mi_apc(parsed.seqs, max_seqs=args.target_max_seqs)
    np.save(out_dir / "target_mi_apc.npy", target_mi)

    edges = select_edges(target_mi, args.edge_factor)
    source_stats: SourceStats | None = None
    if args.sampler_mode == "stat_field":
        bpp = load_case_bpp(args, rna_id, parsed.length) if args.stats_source == "global_prior" else None
        source_stats = build_source_stats(parsed, args, bpp=bpp)
        fields = build_stat_fields(parsed.query, source_stats, args)
        compat = build_stat_compatibilities(parsed.query, edges, source_stats, args)
    else:
        fields = build_query_fields(parsed.query, args.same_prob, args.mutation_prob_each, args.gap_prob)
        compat = build_compatibilities(parsed.query, edges)
    adjacency = build_adjacency(parsed.length, edges)

    print(f"  selected_edges={edges.size}, target_scale={edges.scale:.6g}")
    if source_stats is not None:
        print(f"  stat_field source={args.stats_source}, source_depth={source_stats.depth}, variant={args.stat_variant}")
        if source_stats.global_prior is not None:
            print(f"  global_prior={source_stats.global_prior.path}")
    print("  generating independent field baseline...")
    random_num = sample_query_prior(args.synthetic_depth, fields, rng)
    random_synthetic = numeric_to_seqs(random_num)
    random_seqs = [parsed.query] + random_synthetic
    random_mi = compute_mi_apc(random_seqs)
    np.save(out_dir / "random_mi_apc.npy", random_mi)
    write_a3m(out_dir / "synthetic_random_baseline.a3m", parsed.query, random_synthetic, "synthetic_random")

    target_norm = edges.target_norm
    if args.inverse_optimizer == "mi3_potts":
        weights = np.zeros_like(target_norm)
        calibration_rounds = 0
        calibration_skip_reason = "mi3_potts optimizer fits its own Potts couplings"
    elif args.sampler_mode == "stat_field" and args.stat_variant == "profile":
        weights = np.zeros_like(target_norm)
        calibration_rounds = 0
        calibration_skip_reason = "profile-only variant"
    else:
        weights = np.clip(target_norm.copy(), 0.0, args.w_max)
        calibration_rounds = args.calibration_rounds
        calibration_skip_reason = "requested zero calibration rounds"
    history: list[dict[str, float]] = []
    print(
        "  calibrating Potts couplings..."
        if calibration_rounds
        else f"  skipping pair calibration for {calibration_skip_reason}..."
    )
    for round_idx in range(calibration_rounds):
        sampled_num = sample_inverse_states(
            depth=args.synthetic_depth,
            fields=fields,
            edges=edges,
            compat=compat,
            adjacency=adjacency,
            weights=weights,
            rng=rng,
            args=args,
            query=parsed.query,
            source_stats=source_stats,
        )
        sampled_seqs = [parsed.query] + numeric_to_seqs(sampled_num)
        sampled_mi = compute_mi_apc(sampled_seqs)
        sampled_norm = sampled_mi[edges.i, edges.j] / edges.scale
        mae_before = float(np.mean(np.abs(target_norm - sampled_norm)))
        weights = np.clip(weights + args.eta * (target_norm - sampled_norm), 0.0, args.w_max)
        history.append(
            {
                "round": float(round_idx + 1),
                "selected_edge_mae": mae_before,
                "mean_weight": float(weights.mean()) if weights.size else 0.0,
                "max_weight": float(weights.max()) if weights.size else 0.0,
            }
        )
        print(
            f"    round {round_idx + 1}/{args.calibration_rounds}: "
            f"edge_mae={mae_before:.4f}, mean_w={history[-1]['mean_weight']:.3f}"
        )

    print("  sampling calibrated MSA...")
    calibrated_num = sample_inverse_states(
        depth=args.synthetic_depth,
        fields=fields,
        edges=edges,
        compat=compat,
        adjacency=adjacency,
        weights=weights,
        rng=rng,
        args=args,
        query=parsed.query,
        source_stats=source_stats,
    )
    opt_result = optimize_inverse_msa(
        baseline_num=calibrated_num,
        target_mi=target_mi,
        fields=fields,
        edges=edges,
        compat=compat,
        adjacency=adjacency,
        weights=weights,
        rng=rng,
        args=args,
        query=parsed.query,
        source_stats=source_stats,
    )
    baseline_mi = opt_result["baseline_generated_mi"]
    if args.inverse_optimizer != "none":
        is_column_pair = opt_result["optimizer"] == "column_pair_sa"
        is_mi3 = opt_result["optimizer"] == "mi3_potts"
        baseline_synthetic = numeric_to_seqs(calibrated_num)
        baseline_a3m = out_dir / "synthetic_baseline.a3m"
        write_a3m(baseline_a3m, parsed.query, baseline_synthetic, "synthetic_baseline")
        np.save(out_dir / "baseline_generated_mi_apc.npy", baseline_mi)
        history_name = "mi3_history.json" if is_mi3 else ("repair_history.json" if is_column_pair else "ea_history.json")
        save_json(
            out_dir / history_name,
            {
                "optimizer": opt_result["optimizer"],
                "pool_size": opt_result["pool_size"],
                "baseline_fitness": opt_result["baseline_fitness"],
                "best_fitness": opt_result["best_fitness"],
                "mi3_baseline_fitness": opt_result.get("mi3_baseline_fitness"),
                "history": opt_result["history"],
            },
        )
        if is_mi3:
            np.save(out_dir / "target_bimarg.npy", np.asarray(opt_result["target_bimarg"], dtype=np.float32))
            np.save(out_dir / "final_model_bimarg.npy", np.asarray(opt_result["final_model_bimarg"], dtype=np.float32))
            np.save(out_dir / "final_J.npy", np.asarray(opt_result["final_J"], dtype=np.float32))
        print(
            f"  optimizer={opt_result['optimizer']}: "
            f"baseline_loss={opt_result['baseline_fitness']['loss']:.4f}, "
            f"best_loss={opt_result['best_fitness']['loss']:.4f}"
        )
    calibrated_num = opt_result["msa_num"]
    calibrated_synthetic = numeric_to_seqs(calibrated_num)
    calibrated_seqs = [parsed.query] + calibrated_synthetic
    calibrated_mi = opt_result["generated_mi"]
    np.save(out_dir / "calibrated_mi_apc.npy", calibrated_mi)
    if args.inverse_optimizer != "none":
        if opt_result["optimizer"] == "column_pair_sa":
            np.save(out_dir / "column_pair_sa_generated_mi_apc.npy", calibrated_mi)
            write_a3m(
                out_dir / "synthetic_column_pair_sa_best.a3m",
                parsed.query,
                calibrated_synthetic,
                "synthetic_column_pair_sa_best",
            )
        elif opt_result["optimizer"] == "mi3_potts":
            np.save(out_dir / "mi3_potts_generated_mi_apc.npy", calibrated_mi)
            write_a3m(out_dir / "synthetic_mi3_potts.a3m", parsed.query, calibrated_synthetic, "synthetic_mi3_potts")
        else:
            np.save(out_dir / "ea_generated_mi_apc.npy", calibrated_mi)
            write_a3m(out_dir / "synthetic_ea_best.a3m", parsed.query, calibrated_synthetic, "synthetic_ea_best")
    calibrated_a3m = out_dir / "synthetic_calibrated.a3m"
    write_a3m(calibrated_a3m, parsed.query, calibrated_synthetic, "synthetic_calibrated")
    if args.standard_output:
        copy_standard_files(msa_path.parent, out_dir, rna_id, calibrated_a3m)

    metrics: dict[str, object] = {
        "rna_id": rna_id,
        "raw_depth": parsed.depth,
        "length": parsed.length,
        "selected_edges": edges.size,
        "target_scale": edges.scale,
        "random": compare_to_target(target_mi, random_mi, edges),
        "calibrated": compare_to_target(target_mi, calibrated_mi, edges),
        "random_stats": msa_stats(random_seqs),
        "calibrated_stats": msa_stats(calibrated_seqs),
        "random_raw_similarity": raw_similarity_stats(parsed.seqs[: args.target_max_seqs], random_seqs, edges),
        "calibrated_raw_similarity": raw_similarity_stats(parsed.seqs[: args.target_max_seqs], calibrated_seqs, edges),
        "calibration_history": history,
        "params": {
            "sampler_mode": args.sampler_mode,
            "stats_source": args.stats_source,
            "stat_variant": args.stat_variant,
            "alphabet_prior_path": args.alphabet_prior_path,
            "alphabet_prior_mode": args.alphabet_prior_mode,
            "synthetic_depth": args.synthetic_depth,
            "target_max_seqs": args.target_max_seqs,
            "stats_max_seqs": args.stats_max_seqs,
            "calibration_rounds": args.calibration_rounds,
            "edge_factor": args.edge_factor,
            "gamma": args.gamma,
            "eta": args.eta,
            "w_max": args.w_max,
            "gibbs_sweeps": args.gibbs_sweeps,
            "temperature": args.temperature,
            "profile_weight": args.profile_weight,
            "query_weight": args.query_weight,
            "pair_tensor_weight": args.pair_tensor_weight,
            "rule_mix": args.rule_mix,
            "inverse_optimizer": args.inverse_optimizer,
            "ea_pool_multiplier": args.ea_pool_multiplier,
            "ea_population": args.ea_population,
            "ea_generations": args.ea_generations,
            "ea_elite": args.ea_elite,
            "ea_tournament": args.ea_tournament,
            "ea_mutation_fraction": args.ea_mutation_fraction,
            "repair_edge_factor": args.repair_edge_factor,
            "repair_control_edge_factor": args.repair_control_edge_factor,
            "repair_steps": args.repair_steps,
            "repair_candidates_per_step": args.repair_candidates_per_step,
            "repair_focus_edges": args.repair_focus_edges,
            "repair_row_fraction": args.repair_row_fraction,
            "repair_min_rows": args.repair_min_rows,
            "repair_max_rows": args.repair_max_rows,
            "repair_start_temperature": args.repair_start_temperature,
            "repair_end_temperature": args.repair_end_temperature,
            "repair_full_eval_interval": args.repair_full_eval_interval,
            "repair_pair_beta": args.repair_pair_beta,
            "repair_state_replace_prob": args.repair_state_replace_prob,
            "repair_corr_weight": args.repair_corr_weight,
            "repair_edge_weight": args.repair_edge_weight,
            "repair_top_weight": args.repair_top_weight,
            "repair_profile_weight": args.repair_profile_weight,
            "repair_gap_weight": args.repair_gap_weight,
            "repair_identity_weight": args.repair_identity_weight,
            "repair_duplicate_weight": args.repair_duplicate_weight,
            "mi3_mcmc_rounds": args.mi3_mcmc_rounds,
            "mi3_newton_steps": args.mi3_newton_steps,
            "mi3_mcmc_sweeps": args.mi3_mcmc_sweeps,
            "mi3_gamma": args.mi3_gamma,
            "mi3_damping": args.mi3_damping,
            "mi3_frac_neff": args.mi3_frac_neff,
            "mi3_pair_beta_max": args.mi3_pair_beta_max,
            "mi3_query_prior_mix": args.mi3_query_prior_mix,
            "mi3_target_pseudocount": args.mi3_target_pseudocount,
            "mi3_temperature": args.mi3_temperature,
            "mi3_init": args.mi3_init,
        },
        "inverse_optimizer": {
            "optimizer": opt_result["optimizer"],
            "pool_size": opt_result["pool_size"],
            "baseline_fitness": opt_result["baseline_fitness"],
            "best_fitness": opt_result["best_fitness"],
            "mi3_baseline_fitness": opt_result.get("mi3_baseline_fitness"),
        },
    }
    if source_stats is not None and source_stats.global_prior is not None:
        metrics["global_prior_fallback_counts"] = dict(source_stats.global_prior.last_pair_fallback_counts)
    if source_stats is not None and source_stats.alphabet_prior is not None:
        metrics["alphabet_prior"] = {
            "path": str(source_stats.alphabet_prior.path),
            "mode": source_stats.alphabet_prior.mode,
            "probs": source_stats.alphabet_prior.probs.tolist(),
        }

    save_json(out_dir / "metrics.json", metrics)
    plot_comparison(target_mi, random_mi, calibrated_mi, metrics, out_dir / "comparison.png")
    print(
        "  result: "
        f"random_r={metrics['random']['pearson']:.3f}, "
        f"calibrated_r={metrics['calibrated']['pearson']:.3f}, "
        f"random_mae={metrics['random']['selected_edge_mae']:.3f}, "
        f"calibrated_mae={metrics['calibrated']['selected_edge_mae']:.3f}"
    )
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="No-training coevolution map to raw MSA inverse sampler.")
    parser.add_argument("--msa_dir", type=str, default="data/msa")
    parser.add_argument("--out_dir", type=str, default="inverse_coevo_results")
    parser.add_argument("--rna_ids", type=str, default="auto")
    parser.add_argument("--input_layout", choices=["auto", "flat", "nested"], default="auto")
    parser.add_argument(
        "--standard_output",
        action="store_true",
        help="Also write <out_dir>/<rna_id>/<rna_id>.a3m and copy matching .fasta/.ipknot.ss files.",
    )
    parser.add_argument("--synthetic_depth", type=int, default=500)
    parser.add_argument("--target_max_seqs", type=int, default=500)
    parser.add_argument("--stats_max_seqs", type=int, default=500)
    parser.add_argument("--calibration_rounds", type=int, default=5)
    parser.add_argument("--edge_factor", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument("--sampler_mode", choices=["mi_only", "stat_field"], default="mi_only")
    parser.add_argument(
        "--stats_source",
        choices=["oracle", "shallow", "query", "global_prior", "alphabet_prior"],
        default="oracle",
    )
    parser.add_argument(
        "--stat_variant",
        choices=["profile", "profile_pair", "profile_pair_identity"],
        default="profile_pair_identity",
    )
    parser.add_argument("--shallow_depth", type=int, default=50)
    parser.add_argument("--profile_pseudocount", type=float, default=1.0)
    parser.add_argument("--pair_pseudocount", type=float, default=1.0)
    parser.add_argument("--pair_log_odds_clip", type=float, default=2.0)
    parser.add_argument("--profile_weight", type=float, default=1.0)
    parser.add_argument("--query_weight", type=float, default=0.0)
    parser.add_argument("--pair_tensor_weight", type=float, default=1.0)
    parser.add_argument("--rule_mix", type=float, default=0.15)
    parser.add_argument("--identity_candidate_multiplier", type=int, default=5)
    parser.add_argument("--global_prior_path", type=str, default="")
    parser.add_argument("--global_prior_mode", choices=["auto", "conditioned", "global"], default="auto")
    parser.add_argument("--global_prior_min_count", type=float, default=5.0)
    parser.add_argument("--alphabet_prior_path", type=str, default="")
    parser.add_argument("--alphabet_prior_mode", choices=["family_balanced", "raw_token"], default="family_balanced")
    parser.add_argument("--bpp_dir", type=str, default="")
    parser.add_argument("--bpp_path", type=str, default="")
    parser.add_argument("--match_row_gap_distribution", action="store_true")

    parser.add_argument("--inverse_optimizer", choices=["none", "best_of", "row_pool_ea", "column_pair_sa", "mi3_potts"], default="none")
    parser.add_argument("--ea_pool_multiplier", type=float, default=5.0)
    parser.add_argument("--ea_population", type=int, default=8)
    parser.add_argument("--ea_generations", type=int, default=10)
    parser.add_argument("--ea_elite", type=int, default=2)
    parser.add_argument("--ea_tournament", type=int, default=3)
    parser.add_argument("--ea_mutation_fraction", type=float, default=0.12)
    parser.add_argument("--ea_map_weight", type=float, default=1.0)
    parser.add_argument("--ea_top_weight", type=float, default=0.25)
    parser.add_argument("--ea_prior_weight", type=float, default=0.10)
    parser.add_argument("--ea_duplicate_weight", type=float, default=0.05)
    parser.add_argument("--repair_edge_factor", type=float, default=8.0)
    parser.add_argument("--repair_control_edge_factor", type=float, default=2.0)
    parser.add_argument("--repair_steps", type=int, default=3000)
    parser.add_argument("--repair_candidates_per_step", type=int, default=4)
    parser.add_argument("--repair_focus_edges", type=int, default=32)
    parser.add_argument("--repair_row_fraction", type=float, default=0.06)
    parser.add_argument("--repair_min_rows", type=int, default=16)
    parser.add_argument("--repair_max_rows", type=int, default=128)
    parser.add_argument("--repair_start_temperature", type=float, default=0.05)
    parser.add_argument("--repair_end_temperature", type=float, default=0.005)
    parser.add_argument("--repair_full_eval_interval", type=int, default=100)
    parser.add_argument("--repair_pair_beta", type=float, default=2.0)
    parser.add_argument("--repair_state_replace_prob", type=float, default=0.6)
    parser.add_argument("--repair_corr_weight", type=float, default=2.0)
    parser.add_argument("--repair_edge_weight", type=float, default=1.0)
    parser.add_argument("--repair_top_weight", type=float, default=0.5)
    parser.add_argument("--repair_profile_weight", type=float, default=0.05)
    parser.add_argument("--repair_gap_weight", type=float, default=0.05)
    parser.add_argument("--repair_identity_weight", type=float, default=0.03)
    parser.add_argument("--repair_duplicate_weight", type=float, default=0.02)
    parser.add_argument("--mi3_mcmc_rounds", type=int, default=15)
    parser.add_argument("--mi3_newton_steps", type=int, default=64)
    parser.add_argument("--mi3_mcmc_sweeps", type=int, default=10)
    parser.add_argument("--mi3_gamma", type=float, default=0.02)
    parser.add_argument("--mi3_damping", type=float, default=0.001)
    parser.add_argument("--mi3_frac_neff", type=float, default=0.90)
    parser.add_argument("--mi3_pair_beta_max", type=float, default=8.0)
    parser.add_argument("--mi3_query_prior_mix", type=float, default=0.35)
    parser.add_argument("--mi3_target_pseudocount", type=float, default=0.5)
    parser.add_argument("--mi3_temperature", type=float, default=1.0)
    parser.add_argument("--mi3_init", choices=["independent", "zero"], default="independent")

    parser.add_argument("--same_prob", type=float, default=0.70)
    parser.add_argument("--mutation_prob_each", type=float, default=0.09)
    parser.add_argument("--gap_prob", type=float, default=0.03)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--eta", type=float, default=0.5)
    parser.add_argument("--w_max", type=float, default=3.0)
    parser.add_argument("--gibbs_sweeps", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
    return parser


def assert_finite_summary(summary: Iterable[dict[str, object]]) -> None:
    for row in summary:
        for group in (
            "random",
            "calibrated",
            "random_stats",
            "calibrated_stats",
            "random_raw_similarity",
            "calibrated_raw_similarity",
        ):
            for key, value in row[group].items():
                if not np.isfinite(float(value)):
                    raise ValueError(f"Non-finite metric for {row['rna_id']} {group}.{key}: {value}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.synthetic_depth < 2:
        raise ValueError("--synthetic_depth must be at least 2")
    if args.target_max_seqs < 2:
        raise ValueError("--target_max_seqs must be at least 2")
    if args.stats_max_seqs < 1:
        raise ValueError("--stats_max_seqs must be at least 1")
    if args.identity_candidate_multiplier < 1:
        raise ValueError("--identity_candidate_multiplier must be at least 1")
    if args.stats_source == "global_prior" and not args.global_prior_path:
        raise ValueError("--global_prior_path is required when --stats_source global_prior")
    if args.stats_source == "alphabet_prior" and not args.alphabet_prior_path:
        raise ValueError("--alphabet_prior_path is required when --stats_source alphabet_prior")
    if args.ea_population < 1:
        raise ValueError("--ea_population must be at least 1")
    if args.ea_generations < 0:
        raise ValueError("--ea_generations must be non-negative")
    if args.ea_elite < 1:
        raise ValueError("--ea_elite must be at least 1")
    if args.ea_pool_multiplier < 1.0:
        raise ValueError("--ea_pool_multiplier must be at least 1.0")
    if args.ea_mutation_fraction < 0.0:
        raise ValueError("--ea_mutation_fraction must be non-negative")
    if args.repair_edge_factor <= 0.0:
        raise ValueError("--repair_edge_factor must be positive")
    if args.repair_control_edge_factor < 0.0:
        raise ValueError("--repair_control_edge_factor must be non-negative")
    if args.repair_steps < 0:
        raise ValueError("--repair_steps must be non-negative")
    if args.repair_candidates_per_step < 1:
        raise ValueError("--repair_candidates_per_step must be at least 1")
    if args.mi3_mcmc_rounds < 0:
        raise ValueError("--mi3_mcmc_rounds must be non-negative")
    if args.mi3_newton_steps < 0:
        raise ValueError("--mi3_newton_steps must be non-negative")
    if args.mi3_mcmc_sweeps < 0:
        raise ValueError("--mi3_mcmc_sweeps must be non-negative")
    if args.mi3_gamma <= 0.0:
        raise ValueError("--mi3_gamma must be positive")
    if args.mi3_damping < 0.0:
        raise ValueError("--mi3_damping must be non-negative")
    if not (0.0 < args.mi3_frac_neff <= 1.0):
        raise ValueError("--mi3_frac_neff must be in (0, 1]")
    if args.mi3_pair_beta_max < 0.0:
        raise ValueError("--mi3_pair_beta_max must be non-negative")
    if not (0.0 <= args.mi3_query_prior_mix <= 1.0):
        raise ValueError("--mi3_query_prior_mix must be in [0, 1]")
    if args.mi3_target_pseudocount < 0.0:
        raise ValueError("--mi3_target_pseudocount must be non-negative")
    if args.mi3_temperature <= 0.0:
        raise ValueError("--mi3_temperature must be positive")
    if args.repair_focus_edges < 1:
        raise ValueError("--repair_focus_edges must be at least 1")
    if args.repair_min_rows < 1 or args.repair_max_rows < 1:
        raise ValueError("--repair_min_rows and --repair_max_rows must be positive")
    if args.repair_min_rows > args.repair_max_rows:
        raise ValueError("--repair_min_rows cannot exceed --repair_max_rows")
    if args.repair_row_fraction <= 0.0:
        raise ValueError("--repair_row_fraction must be positive")
    if args.repair_start_temperature < 0.0 or args.repair_end_temperature < 0.0:
        raise ValueError("--repair temperatures must be non-negative")
    if args.repair_full_eval_interval < 1:
        raise ValueError("--repair_full_eval_interval must be at least 1")
    if not (0.0 <= args.repair_state_replace_prob <= 1.0):
        raise ValueError("--repair_state_replace_prob must be between 0 and 1")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    layout = effective_input_layout(args)
    rna_ids = discover_rna_ids(args, layout)
    print(f"RNA IDs: {', '.join(rna_ids)}")
    print(f"Input layout: {layout}")
    print(f"Output: {out_dir.resolve()}")

    summary: list[dict[str, object]] = []
    for rna_id in rna_ids:
        summary.append(process_one(args, rna_id, rng))

    assert_finite_summary(summary)
    save_json(out_dir / "summary.json", summary)
    write_summary_csv(out_dir / "summary.csv", summary)
    plot_summary(summary, out_dir / "summary.png")

    mae_wins = sum(
        row["calibrated"]["selected_edge_mae"] < row["random"]["selected_edge_mae"] for row in summary
    )
    pearson_wins = sum(row["calibrated"]["pearson"] > row["random"]["pearson"] for row in summary)
    top_l_wins = sum(row["calibrated"]["top_l_overlap"] > row["random"]["top_l_overlap"] for row in summary)
    print("\nDone.")
    print(f"  MAE wins:     {mae_wins}/{len(summary)}")
    print(f"  Pearson wins: {pearson_wins}/{len(summary)}")
    print(f"  Top-L wins:   {top_l_wins}/{len(summary)}")
    print(f"  Summary:      {out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
