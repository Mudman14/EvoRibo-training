"""Dataset adapter for RESM embedding/attention smoke experiments."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from .data_utils import NUC_TO_IDX, parse_prob_file, read_fasta_sequence, tokenize_sequence
from .stat_targets import fij_to_pair_stats_map, load_or_compute_adabm_stats_target


EMBED_SUFFIXES = ("_emb", "_embedding")
ATTN_SUFFIXES = ("_atp", "_attention", "_attention_map", "_attn")


@dataclass(frozen=True)
class RESMSamplePaths:
    rna_id: str
    embedding: Path
    attention: Path
    query: Path
    bpp: Path
    msa: Path


def _strip_known_suffix(stem: str, suffixes: Iterable[str]) -> str:
    for suffix in suffixes:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def resolve_feature_dirs(
    resm_dir: str | Path | None = None,
    embedding_dir: str | Path | None = None,
    attention_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    if embedding_dir is None or attention_dir is None:
        if resm_dir is None:
            raise ValueError("Either --resm_dir or both embedding_dir/attention_dir are required")
        resm_dir = Path(resm_dir)
        embedding_dir = embedding_dir or (resm_dir / "embeddings")
        attention_dir = attention_dir or (resm_dir / "attention_map")
    return Path(embedding_dir), Path(attention_dir)


def discover_resm_ids(
    resm_dir: str | Path | None = None,
    embedding_dir: str | Path | None = None,
    attention_dir: str | Path | None = None,
) -> list[str]:
    emb_dir, attn_dir = resolve_feature_dirs(resm_dir, embedding_dir, attention_dir)
    emb_ids = {
        _strip_known_suffix(path.stem, EMBED_SUFFIXES)
        for path in emb_dir.glob("*.npy")
    }
    attn_ids = {
        _strip_known_suffix(path.stem, ATTN_SUFFIXES)
        for path in attn_dir.glob("*.npy")
    }
    return sorted(emb_ids | attn_ids)


def _first_existing(base_dir: Path, rna_id: str, suffixes: Iterable[str]) -> Path:
    candidates = [base_dir / f"{rna_id}{suffix}" for suffix in suffixes]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def resolve_sample_paths(
    rna_id: str,
    resm_dir: str | Path | None,
    query_dir: str | Path,
    bpp_dir: str | Path,
    msa_dir: str | Path,
    embedding_dir: str | Path | None = None,
    attention_dir: str | Path | None = None,
) -> RESMSamplePaths:
    emb_dir, attn_dir = resolve_feature_dirs(resm_dir, embedding_dir, attention_dir)
    query_dir = Path(query_dir)
    bpp_dir = Path(bpp_dir)
    msa_dir = Path(msa_dir)
    embedding = _first_existing(emb_dir, rna_id, ("_emb.npy", "_embedding.npy", ".npy"))
    attention = _first_existing(
        attn_dir,
        rna_id,
        ("_atp.npy", "_attention.npy", "_attention_map.npy", "_attn.npy", ".npy"),
    )
    query = _first_existing(query_dir, rna_id, (".a2m", ".fasta", ".fa", ".a3m"))
    bpp = bpp_dir / f"{rna_id}.prob"
    msa = _first_existing(msa_dir, rna_id, (".a3m", ".a2m"))
    if not msa.exists():
        for suffix in (".a3m", ".a2m"):
            nested_msa = msa_dir / rna_id / f"{rna_id}{suffix}"
            if nested_msa.exists():
                msa = nested_msa
                break
    if not msa.exists():
        bucket_matches: list[Path] = []
        for suffix in (".a3m", ".a2m"):
            bucket_matches.extend(msa_dir.glob(f"*/{rna_id}{suffix}"))
        if bucket_matches:
            msa = sorted(bucket_matches)[0]
    return RESMSamplePaths(rna_id, embedding, attention, query, bpp, msa)


def coerce_resm_single(array: np.ndarray) -> np.ndarray:
    """Return RESM single embedding as [L, C]."""
    if array.ndim == 2:
        return np.asarray(array)
    if array.ndim == 3 and array.shape[0] == 1:
        return np.asarray(array[0])
    raise ValueError(f"RESM single embedding must be [L, C] or [1, L, C], got {array.shape}")


def coerce_resm_pair(
    array: np.ndarray,
    expected_length: int | None = None,
    materialize: bool = True,
) -> np.ndarray:
    """Return RESM attention map as [L, L, C]."""
    if array.ndim != 3:
        raise ValueError(f"RESM attention map must be rank 3, got {array.shape}")
    if array.shape[0] == array.shape[1]:
        pair = array
    elif array.shape[1] == array.shape[2]:
        pair = np.moveaxis(array, 0, -1)
    else:
        raise ValueError(f"Cannot infer RESM attention axes from shape {array.shape}")

    if pair.shape[0] != pair.shape[1]:
        raise ValueError(f"Coerced RESM pair is not square: {pair.shape}")
    if expected_length is not None and pair.shape[:2] != (expected_length, expected_length):
        raise ValueError(
            f"RESM pair length mismatch: expected {(expected_length, expected_length)}, "
            f"got {pair.shape[:2]}"
        )
    if materialize:
        return np.ascontiguousarray(pair)
    return pair


def normalize_target_map(target: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, float]:
    target = np.asarray(target, dtype=np.float32)
    if target.ndim != 2 or target.shape[0] != target.shape[1]:
        raise ValueError(f"target map must have shape [L, L], got {target.shape}")
    upper = np.triu(np.ones(target.shape, dtype=bool), k=1)
    if upper.any():
        scale = float(np.max(np.abs(target[upper])))
    else:
        scale = 0.0
    if not np.isfinite(scale) or scale < eps:
        scale = 1.0
    return (target / scale).astype(np.float32), scale


def _clean_a3m_sequence(lines: list[str]) -> str:
    chars: list[str] = []
    for ch in "".join(lines):
        if ch.islower():
            continue
        c = ch.upper()
        if c == "T":
            c = "U"
        if c in {"A", "U", "C", "G", "-"}:
            chars.append(c)
    return "".join(chars)


def _read_clean_a3m_sequences(msa_path: Path) -> list[str]:
    seqs: list[str] = []
    current: list[str] = []
    seen_header = False
    with msa_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seen_header:
                    seqs.append(_clean_a3m_sequence(current))
                    current = []
                seen_header = True
            elif seen_header:
                current.append(line)
        if seen_header:
            seqs.append(_clean_a3m_sequence(current))
    return seqs


def _load_target_mi_apc(
    msa_path: Path,
    target_max_seqs: int,
    expected_length: int | None = None,
) -> np.ndarray:
    from inverse_coevo_sampler import compute_mi_apc, parse_a3m

    if expected_length is None:
        parsed = parse_a3m(msa_path)
        return compute_mi_apc(parsed.seqs, max_seqs=target_max_seqs)

    seqs = _read_clean_a3m_sequences(msa_path)
    seqs = [seq for seq in seqs if len(seq) == expected_length]
    if len(seqs) < 2:
        return np.zeros((expected_length, expected_length), dtype=np.float64)
    return compute_mi_apc(seqs, max_seqs=target_max_seqs)


def _load_or_compute_target_mi_apc(
    rna_id: str,
    msa_path: Path,
    target_max_seqs: int,
    target_cache_dir: Path | None,
    expected_length: int | None = None,
) -> np.ndarray:
    if target_cache_dir is None:
        return _load_target_mi_apc(msa_path, target_max_seqs, expected_length=expected_length)
    target_cache_dir.mkdir(parents=True, exist_ok=True)
    length_tag = f"_L{expected_length}" if expected_length is not None else ""
    cache_path = target_cache_dir / f"{rna_id}{length_tag}_max{target_max_seqs}_mi_apc.npy"
    if cache_path.exists():
        return np.load(cache_path)
    target = _load_target_mi_apc(msa_path, target_max_seqs, expected_length=expected_length)
    np.save(cache_path, target)
    return target


def _a3m_depth_and_lengths(msa_path: Path) -> tuple[int, list[int]]:
    depth = 0
    lengths: set[int] = set()
    current: list[str] = []
    seen_header = False
    with msa_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seen_header:
                    lengths.add(_clean_a3m_len(current))
                    depth += 1
                    current = []
                seen_header = True
            elif seen_header:
                current.append(line)
        if seen_header:
            lengths.add(_clean_a3m_len(current))
            depth += 1
    return depth, sorted(lengths)


def _clean_a3m_len(lines: list[str]) -> int:
    length = 0
    for ch in "".join(lines):
        if ch.islower():
            continue
        if ch.upper() in {"A", "U", "C", "G", "T", "-"}:
            length += 1
    return length


class RESMTestDataset(Dataset):
    """Load RESM650M features, BPP, query tokens, and scalar/statistics targets."""

    def __init__(
        self,
        resm_dir: str | Path,
        query_dir: str | Path,
        bpp_dir: str | Path,
        msa_dir: str | Path,
        ids: list[str] | None = None,
        target_max_seqs: int = 64,
        normalize_target: bool = True,
        cache_targets: bool = False,
        embedding_dir: str | Path | None = None,
        attention_dir: str | Path | None = None,
        target_cache_dir: str | Path | None = None,
        target_builder: str = "adabm_stats",
        enable_stats_target: bool = False,
        include_stats_target_map: bool = True,
        target_clustering_seqid: float = 0.8,
        target_remove_duplicates: bool = True,
        target_no_reweighting: bool = False,
        target_pseudocount: float | None = None,
    ) -> None:
        self.resm_dir = Path(resm_dir) if resm_dir is not None else None
        self.embedding_dir, self.attention_dir = resolve_feature_dirs(
            self.resm_dir,
            embedding_dir,
            attention_dir,
        )
        self.query_dir = Path(query_dir)
        self.bpp_dir = Path(bpp_dir)
        self.msa_dir = Path(msa_dir)
        self.ids = ids or discover_resm_ids(
            self.resm_dir,
            self.embedding_dir,
            self.attention_dir,
        )
        self.target_max_seqs = target_max_seqs
        self.normalize_target = normalize_target
        self.cache_targets = cache_targets
        self.target_cache_dir = Path(target_cache_dir) if target_cache_dir is not None else None
        if target_builder not in {"legacy_mi_apc", "adabm_stats"}:
            raise ValueError("target_builder must be 'legacy_mi_apc' or 'adabm_stats'")
        self.target_builder = target_builder
        self.enable_stats_target = enable_stats_target
        self.include_stats_target_map = include_stats_target_map
        self.target_clustering_seqid = target_clustering_seqid
        self.target_remove_duplicates = target_remove_duplicates
        self.target_no_reweighting = target_no_reweighting
        self.target_pseudocount = target_pseudocount
        self._target_cache: dict[str, dict[str, torch.Tensor | float]] = {}

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | float | int]:
        rna_id = self.ids[index]
        paths = resolve_sample_paths(
            rna_id,
            self.resm_dir,
            self.query_dir,
            self.bpp_dir,
            self.msa_dir,
            self.embedding_dir,
            self.attention_dir,
        )
        for path in (paths.embedding, paths.attention, paths.query, paths.bpp, paths.msa):
            if not path.exists():
                raise FileNotFoundError(path)

        single_np = coerce_resm_single(np.load(paths.embedding, mmap_mode="r"))
        seq = read_fasta_sequence(paths.query)
        L, single_dim = single_np.shape
        if len(seq) != L:
            raise ValueError(f"{rna_id}: query length {len(seq)} != embedding length {L}")

        pair_np = coerce_resm_pair(np.load(paths.attention, mmap_mode="r"), expected_length=L)
        bpp = parse_prob_file(paths.bpp, expected_length=L)
        target_info = self._target_for(paths, expected_length=L)
        target = target_info["target"]
        target_scale = float(target_info["target_scale"])
        if target.shape != (L, L):
            raise ValueError(f"{rna_id}: target shape {tuple(target.shape)} != {(L, L)}")

        tokens = tokenize_sequence(seq)
        mask = torch.ones(L, dtype=torch.bool)
        item: dict[str, torch.Tensor | str | float | int] = {
            "id": rna_id,
            "length": L,
            "resm_single_dim": single_dim,
            "resm_pair_dim": pair_np.shape[-1],
            "resm_single": torch.from_numpy(np.array(single_np, dtype=np.float32, copy=True, order="C")),
            "resm_pair": torch.from_numpy(np.array(pair_np, dtype=np.float32, copy=True, order="C")),
            "bpp": bpp.float(),
            "tokens": tokens,
            "mask": mask,
            "target": target,
            "target_scale": target_scale,
        }
        if self.enable_stats_target:
            if self.include_stats_target_map:
                stats_target = target_info.get("stats_target")
                if stats_target is None:
                    raise ValueError("stats_target requested but target_builder does not provide it")
                if stats_target.shape != (L, L, 25):
                    raise ValueError(f"{rna_id}: stats target shape {tuple(stats_target.shape)} != {(L, L, 25)}")
                item["stats_target"] = stats_target
            fi_target = target_info.get("fi_target")
            fij_target = target_info.get("fij_target")
            if fi_target is not None:
                if fi_target.shape != (L, 5):
                    raise ValueError(f"{rna_id}: fi target shape {tuple(fi_target.shape)} != {(L, 5)}")
                item["fi_target"] = fi_target
            if fij_target is not None:
                if fij_target.shape != (L, L, 25):
                    raise ValueError(f"{rna_id}: fij target shape {tuple(fij_target.shape)} != {(L, L, 25)}")
                item["fij_target"] = fij_target
        if "target_meff" in target_info:
            item["target_meff"] = float(target_info["target_meff"])
        if "target_pseudocount" in target_info:
            item["target_pseudocount"] = float(target_info["target_pseudocount"])
        return item

    def _target_for(self, paths: RESMSamplePaths, expected_length: int) -> dict[str, torch.Tensor | float]:
        cache_key = (
            f"{paths.rna_id}_L{expected_length}_{self.target_builder}"
            f"_stats{int(self.enable_stats_target)}_map{int(self.include_stats_target_map)}"
        )
        if self.cache_targets and cache_key in self._target_cache:
            return self._target_cache[cache_key]

        if self.target_builder == "legacy_mi_apc":
            target_np = _load_or_compute_target_mi_apc(
                paths.rna_id,
                paths.msa,
                self.target_max_seqs,
                self.target_cache_dir,
                expected_length=expected_length,
            )
            scale = 1.0
            if self.normalize_target:
                target_np, scale = normalize_target_map(target_np)
            result: dict[str, torch.Tensor | float] = {
                "target": torch.from_numpy(np.ascontiguousarray(target_np)).float(),
                "target_scale": float(scale),
            }
        else:
            stats = load_or_compute_adabm_stats_target(
                paths.rna_id,
                paths.msa,
                expected_length=expected_length,
                target_max_seqs=self.target_max_seqs,
                target_cache_dir=self.target_cache_dir,
                clustering_seqid=self.target_clustering_seqid,
                remove_duplicates=self.target_remove_duplicates,
                no_reweighting=self.target_no_reweighting,
                pseudocount=self.target_pseudocount,
                normalize_scalar=self.normalize_target,
            )
            result = {
                "target": torch.from_numpy(np.ascontiguousarray(stats.scalar_map)).float(),
                "target_scale": float(stats.scalar_scale),
                "fi_target": torch.from_numpy(np.ascontiguousarray(stats.fi)).float(),
                "fij_target": torch.from_numpy(np.ascontiguousarray(fij_to_pair_stats_map(stats.fij))).float(),
                "target_meff": float(stats.meff),
                "target_pseudocount": float(stats.pseudocount),
            }
            if self.include_stats_target_map:
                result["stats_target"] = torch.from_numpy(np.ascontiguousarray(stats.stats_map)).float()
        if self.cache_targets:
            self._target_cache[cache_key] = result
        return result


def build_shape_report(
    resm_dir: str | Path | None,
    query_dir: str | Path,
    bpp_dir: str | Path,
    msa_dir: str | Path,
    ids: list[str] | None = None,
    embedding_dir: str | Path | None = None,
    attention_dir: str | Path | None = None,
) -> list[dict[str, object]]:
    emb_dir, attn_dir = resolve_feature_dirs(resm_dir, embedding_dir, attention_dir)
    ids = ids or discover_resm_ids(resm_dir, emb_dir, attn_dir)
    rows: list[dict[str, object]] = []
    for rna_id in ids:
        paths = resolve_sample_paths(rna_id, resm_dir, query_dir, bpp_dir, msa_dir, emb_dir, attn_dir)
        row: dict[str, object] = {"id": rna_id, "status": "OK"}
        issues: list[str] = []
        try:
            emb = coerce_resm_single(np.load(paths.embedding, mmap_mode="r"))
            row["embedding_shape"] = list(emb.shape)
            row["L_embedding"] = int(emb.shape[0])
            row["resm_single_dim"] = int(emb.shape[-1])
        except Exception as exc:  # noqa: BLE001 - report should keep scanning samples.
            issues.append(f"embedding:{exc}")
            row["embedding_shape"] = None
            row["L_embedding"] = None

        try:
            attn_raw = np.load(paths.attention, mmap_mode="r")
            row["attention_raw_shape"] = list(attn_raw.shape)
            attn = coerce_resm_pair(
                attn_raw,
                expected_length=row["L_embedding"],
                materialize=False,
            )
            row["attention_shape"] = list(attn.shape)
            row["L_attention"] = int(attn.shape[0])
            row["resm_pair_dim"] = int(attn.shape[-1])
        except Exception as exc:  # noqa: BLE001
            issues.append(f"attention:{exc}")
            row["attention_raw_shape"] = None
            row["attention_shape"] = None
            row["L_attention"] = None

        try:
            seq = read_fasta_sequence(paths.query)
            row["query_length"] = len(seq)
            if row.get("L_embedding") is not None and len(seq) != row["L_embedding"]:
                issues.append(f"query_length:{len(seq)}!={row['L_embedding']}")
        except Exception as exc:  # noqa: BLE001
            issues.append(f"query:{exc}")
            row["query_length"] = None

        try:
            expected = row.get("L_embedding")
            bpp = parse_prob_file(
                paths.bpp,
                expected_length=int(expected) if expected is not None else None,
            )
            row["bpp_shape"] = list(bpp.shape)
            row["bpp_max"] = float(bpp.max().item()) if bpp.numel() else 0.0
        except Exception as exc:  # noqa: BLE001
            issues.append(f"bpp:{exc}")
            row["bpp_shape"] = None
            row["bpp_max"] = None

        try:
            depth, lengths = _a3m_depth_and_lengths(paths.msa)
            row["msa_depth"] = depth
            row["msa_lengths"] = lengths
            if row.get("L_embedding") is not None and lengths != [row["L_embedding"]]:
                issues.append(f"msa_lengths:{lengths}!={[row['L_embedding']]}")
        except Exception as exc:  # noqa: BLE001
            issues.append(f"msa:{exc}")
            row["msa_depth"] = None
            row["msa_lengths"] = None

        if issues:
            row["status"] = ";".join(issues)
        rows.append(row)
    return rows


def write_shape_report(rows: list[dict[str, object]], out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "shape_report.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    fieldnames = sorted({key for row in rows for key in row})
    with (out_dir / "shape_report.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_fasta_ids(fasta_path: str | Path | None) -> set[str]:
    if fasta_path is None:
        return set()
    fasta_path = Path(fasta_path)
    if not fasta_path.is_file():
        raise FileNotFoundError(f"Heldout FASTA does not exist: {fasta_path}")
    ids: set[str] = set()
    with fasta_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if line.startswith(">"):
                ids.add(line[1:].split()[0])
    return ids


def read_test_folder_ids(test_input_dir: str | Path | None) -> set[str]:
    if test_input_dir is None:
        return set()
    test_input_dir = Path(test_input_dir)
    if not test_input_dir.is_dir():
        raise FileNotFoundError(f"Heldout input directory does not exist: {test_input_dir}")
    return {path.name for path in test_input_dir.iterdir() if path.is_dir()}


def split_train_val_test_ids(
    all_ids: list[str],
    val_fasta: str | Path | None = None,
    test_input_dir: str | Path | None = None,
) -> tuple[list[str], list[str], list[str]]:
    all_set = set(all_ids)
    val_ids = sorted(read_fasta_ids(val_fasta) & all_set)
    test_ids = sorted(read_test_folder_ids(test_input_dir) & all_set)
    heldout = set(val_ids) | set(test_ids)
    train_ids = [rna_id for rna_id in all_ids if rna_id not in heldout]
    return train_ids, val_ids, test_ids


def collate_resm_samples(samples: list[dict[str, torch.Tensor | str | float | int]]) -> dict[str, object]:
    if not samples:
        raise ValueError("Cannot collate an empty batch")

    batch_size = len(samples)
    lengths = [int(sample["length"]) for sample in samples]
    max_len = max(lengths)
    single_dim = int(samples[0]["resm_single_dim"])
    pair_dim = int(samples[0]["resm_pair_dim"])
    pad_idx = NUC_TO_IDX["N"]

    resm_single = torch.zeros(batch_size, max_len, single_dim, dtype=torch.float32)
    resm_pair = torch.zeros(batch_size, max_len, max_len, pair_dim, dtype=torch.float32)
    bpp = torch.zeros(batch_size, max_len, max_len, dtype=torch.float32)
    target = torch.zeros(batch_size, 1, max_len, max_len, dtype=torch.float32)
    has_stats_target = any("stats_target" in sample for sample in samples)
    if has_stats_target and not all("stats_target" in sample for sample in samples):
        raise ValueError("Either all samples or no samples must contain stats_target")
    has_fi_target = any("fi_target" in sample for sample in samples)
    if has_fi_target and not all("fi_target" in sample for sample in samples):
        raise ValueError("Either all samples or no samples must contain fi_target")
    has_fij_target = any("fij_target" in sample for sample in samples)
    if has_fij_target and not all("fij_target" in sample for sample in samples):
        raise ValueError("Either all samples or no samples must contain fij_target")
    stats_target = (
        torch.zeros(batch_size, max_len, max_len, 25, dtype=torch.float32)
        if has_stats_target
        else None
    )
    fi_target = torch.zeros(batch_size, max_len, 5, dtype=torch.float32) if has_fi_target else None
    fij_target = (
        torch.zeros(batch_size, max_len, max_len, 25, dtype=torch.float32)
        if has_fij_target
        else None
    )
    tokens = torch.full((batch_size, max_len), pad_idx, dtype=torch.long)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    target_scale = torch.zeros(batch_size, dtype=torch.float32)
    ids: list[str] = []

    for idx, sample in enumerate(samples):
        L = int(sample["length"])
        if int(sample["resm_single_dim"]) != single_dim or int(sample["resm_pair_dim"]) != pair_dim:
            raise ValueError("All samples in a batch must share RESM feature dimensions")
        resm_single[idx, :L] = sample["resm_single"]
        resm_pair[idx, :L, :L] = sample["resm_pair"]
        bpp[idx, :L, :L] = sample["bpp"]
        target[idx, 0, :L, :L] = sample["target"]
        if stats_target is not None:
            stats_target[idx, :L, :L] = sample["stats_target"]
        if fi_target is not None:
            fi_target[idx, :L] = sample["fi_target"]
        if fij_target is not None:
            fij_target[idx, :L, :L] = sample["fij_target"]
        tokens[idx, :L] = sample["tokens"]
        mask[idx, :L] = True
        target_scale[idx] = float(sample["target_scale"])
        ids.append(str(sample["id"]))

    pair_mask = mask[:, :, None] & mask[:, None, :]
    batch: dict[str, object] = {
        "id": ids,
        "length": torch.tensor(lengths, dtype=torch.long),
        "resm_single_dim": single_dim,
        "resm_pair_dim": pair_dim,
        "resm_single": resm_single,
        "resm_pair": resm_pair,
        "bpp": bpp,
        "tokens": tokens,
        "mask": mask,
        "pair_mask": pair_mask,
        "target": target,
        "target_scale": target_scale,
    }
    if stats_target is not None:
        batch["stats_target"] = stats_target
    if fi_target is not None:
        batch["fi_target"] = fi_target
    if fij_target is not None:
        batch["fij_target"] = fij_target
    return batch
