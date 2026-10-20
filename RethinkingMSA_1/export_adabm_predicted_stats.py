"""Export RethinkingMSA predicted fi/fij targets for adabmDCA inverse sampling.

The exported per-family ``adabm_stats.pt`` files are intentionally MSA-free:
they contain the predicted single-site frequencies ``fi`` and bivariate
frequencies ``fij`` in adabmDCA RNA order ``-ACGU``.  They can be consumed by
``run_adabm_inverse_from_predicted_stats.py`` without reading the original MSA.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from .config import SSPairEncoderConfig
from .encoder import EvoNASSPairEncoder
from .resm_dataset import RESMTestDataset, discover_resm_ids, resolve_sample_paths
from .stat_targets import (
    STATS_Q,
    STATS_TOKENS,
    export_mi3_bimarg,
    fij_to_stats_map,
    legalize_stats_map,
    mi_apc_from_fij,
    stats_map_to_fi_fij,
    stats_probs_to_mi_apc_torch,
)
from .train_direct_trunk import build_head, build_stats_head
from .train_resm_smoke import autocast_context, dataset_batch, map_metrics, move_batch_to_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--resm_dir", type=Path, default=Path("data/resm"))
    parser.add_argument("--embedding_dir", type=Path, default=None)
    parser.add_argument("--attention_dir", type=Path, default=None)
    parser.add_argument("--query_dir", type=Path, default=Path("data/query"))
    parser.add_argument("--bpp_dir", type=Path, default=Path("data/bpp"))
    parser.add_argument("--msa_dir", type=Path, default=Path("data/msa"))
    parser.add_argument(
        "--test_input_dir",
        type=Path,
        default=None,
        help="Directory containing one subdirectory per test ID. Used when --ids auto.",
    )
    parser.add_argument("--ids", default="auto", help="'auto' or comma-separated test IDs")
    parser.add_argument("--target_max_seqs", type=int, default=2001)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp_dtype", choices=("none", "bf16"), default="bf16")
    parser.add_argument("--stats_flow_sample_steps", type=int, default=0, help="0 uses checkpoint args")
    parser.add_argument("--stats_flow_x0_noise_scale", type=float, default=0.0)
    parser.add_argument("--legalize_eps", type=float, default=1e-8)
    parser.add_argument("--legalize_ipf_iters", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    return parser


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def checkpoint_args_namespace(ckpt_args: dict[str, Any], overrides: argparse.Namespace) -> SimpleNamespace:
    args = dict(ckpt_args)
    args["amp_dtype"] = overrides.amp_dtype
    args["device"] = str(overrides.device)
    if overrides.stats_flow_sample_steps > 0:
        args["stats_flow_sample_steps"] = int(overrides.stats_flow_sample_steps)
    return SimpleNamespace(**args)


def config_from_checkpoint(config: dict[str, Any]) -> SSPairEncoderConfig:
    valid = {field.name for field in fields(SSPairEncoderConfig)}
    kwargs = {key: value for key, value in config.items() if key in valid}
    cfg = SSPairEncoderConfig(**kwargs)
    cfg.use_gradient_checkpointing = False
    return cfg


def resolve_ids(args: argparse.Namespace) -> list[str]:
    if args.ids.lower() != "auto":
        return [item.strip() for item in args.ids.split(",") if item.strip()]
    discovered = set(discover_resm_ids(args.resm_dir, args.embedding_dir, args.attention_dir))
    if args.test_input_dir and args.test_input_dir.exists():
        ids = sorted(path.name for path in args.test_input_dir.iterdir() if path.is_dir() and path.name in discovered)
    else:
        ids = sorted(discovered)
    if not ids:
        raise ValueError("No test IDs found")
    return ids


def validate_fi_fij(fi: np.ndarray, fij: np.ndarray, eps: float = 1e-6) -> dict[str, float | int | bool]:
    if fi.ndim != 2 or fi.shape[1] != STATS_Q:
        raise ValueError(f"fi must be [L,5], got {fi.shape}")
    length = fi.shape[0]
    if fij.shape != (length, STATS_Q, length, STATS_Q):
        raise ValueError(f"fij must be [L,5,L,5], got {fij.shape}")
    off_sums = []
    diag_errors = []
    marginal_errors = []
    for i in range(length):
        diag_errors.append(float(np.max(np.abs(fij[i, :, i, :] - np.diag(fi[i])))))
        for j in range(length):
            if i == j:
                continue
            block = fij[i, :, j, :]
            off_sums.append(float(block.sum()))
            marginal_errors.append(float(np.max(np.abs(block.sum(axis=1) - fi[i]))))
            marginal_errors.append(float(np.max(np.abs(block.sum(axis=0) - fi[j]))))
    off = np.asarray(off_sums, dtype=np.float64) if off_sums else np.asarray([1.0])
    marg = np.asarray(marginal_errors, dtype=np.float64) if marginal_errors else np.asarray([0.0])
    return {
        "length": int(length),
        "tokens": STATS_TOKENS,
        "finite": bool(np.isfinite(fi).all() and np.isfinite(fij).all()),
        "nonnegative": bool((fi >= -eps).all() and (fij >= -eps).all()),
        "fi_sum_min": float(fi.sum(axis=1).min()),
        "fi_sum_max": float(fi.sum(axis=1).max()),
        "offdiag_pair_sum_min": float(off.min()),
        "offdiag_pair_sum_max": float(off.max()),
        "diag_max_abs_error": float(max(diag_errors) if diag_errors else 0.0),
        "marginal_max_abs_error": float(marg.max()),
        "fi_min": float(fi.min()),
        "fi_max": float(fi.max()),
        "fij_min": float(fij.min()),
        "fij_max": float(fij.max()),
    }


def write_query_fasta(path: Path, rna_id: str, seq: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f">{rna_id}\n")
        for start in range(0, len(seq), 80):
            handle.write(seq[start : start + 80] + "\n")


def load_query_sequence(args: argparse.Namespace, rna_id: str) -> str:
    from .data_utils import read_fasta_sequence

    paths = resolve_sample_paths(
        rna_id,
        args.resm_dir,
        args.query_dir,
        args.bpp_dir,
        args.msa_dir,
        args.embedding_dir,
        args.attention_dir,
    )
    return read_fasta_sequence(paths.query)


def export_sample(
    args: argparse.Namespace,
    ckpt_args: SimpleNamespace,
    encoder: EvoNASSPairEncoder,
    scalar_head: torch.nn.Module,
    stats_head: torch.nn.Module,
    dataset: RESMTestDataset,
    rna_id: str,
    device: torch.device,
) -> dict[str, object]:
    id_to_index = {sample_id: idx for idx, sample_id in enumerate(dataset.ids)}
    batch = move_batch_to_device(dataset_batch(dataset, [rna_id], id_to_index), device)
    with torch.no_grad(), autocast_context(ckpt_args, device):
        enc = encoder(
            batch["resm_single"],
            batch["resm_pair"],
            batch["bpp"],
            batch["tokens"],
            batch["mask"],
        )
        pred_scalar = scalar_head(enc["pair"], enc["pair_mask"], return_full=True)
        fi_logits = stats_head.fi_head(enc["single"], mask=batch["mask"])
        sample = stats_head.sample(
            enc["pair"],
            enc["pair_mask"],
            objective=getattr(ckpt_args, "stats_flow_objective", "velocity"),
            steps=int(getattr(ckpt_args, "stats_flow_sample_steps", 4)),
            x0_noise_scale=float(args.stats_flow_x0_noise_scale),
        )
        fi_probs_t = torch.softmax(fi_logits.float(), dim=-1)
        fij_probs_t = sample["sample_probs"].float()
        fij_mi_t = stats_probs_to_mi_apc_torch(fij_probs_t, pair_mask=enc["pair_mask"])

    length = int(batch["lengths"][0]) if "lengths" in batch else int(batch["mask"][0].sum().item())
    out_dir = args.out_dir / rna_id
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_fi = fi_probs_t[0, :length].detach().cpu().numpy().astype(np.float32)
    raw_pair_stats = fij_probs_t[0, :length, :length].detach().cpu().numpy().astype(np.float32)
    pred_scalar_np = pred_scalar[0, 0, :length, :length].detach().cpu().numpy().astype(np.float32)
    pred_fij_mi_raw = fij_mi_t[0, :length, :length].detach().cpu().numpy().astype(np.float32)
    target_scalar = batch["target"][0, 0, :length, :length].detach().cpu().numpy().astype(np.float32)

    stats_map_seed = np.maximum(raw_pair_stats, float(args.legalize_eps))
    for i in range(length):
        stats_map_seed[i, i] = np.diag(pred_fi[i]).reshape(-1)
    stats_map = legalize_stats_map(stats_map_seed, eps=args.legalize_eps, ipf_iters=args.legalize_ipf_iters)
    pred_fi_legal, pred_fij = stats_map_to_fi_fij(stats_map)
    pred_fij_mi = mi_apc_from_fij(pred_fij)
    mi3_bimarg = export_mi3_bimarg(stats_map)
    validation = validate_fi_fij(pred_fi_legal, pred_fij, eps=args.legalize_eps)

    query = load_query_sequence(args, rna_id)
    write_query_fasta(out_dir / "query.fasta", rna_id, query)
    np.save(out_dir / "predicted_fi.npy", pred_fi_legal)
    np.save(out_dir / "predicted_fij.npy", pred_fij)
    np.save(out_dir / "predicted_stats_map.npy", stats_map)
    np.save(out_dir / "predicted_mi3_bimarg.npy", mi3_bimarg)
    np.save(out_dir / "predicted_scalar_map.npy", pred_scalar_np)
    np.save(out_dir / "predicted_fij_mi_apc.npy", pred_fij_mi)
    np.save(out_dir / "raw_flow_pair_stats.npy", raw_pair_stats)
    np.save(out_dir / "raw_flow_fij_mi_apc.npy", pred_fij_mi_raw)
    np.save(out_dir / "target_scalar_map.npy", target_scalar)

    stats_payload = {
        "fi": torch.from_numpy(pred_fi_legal),
        "fij": torch.from_numpy(pred_fij),
        "stats_map": torch.from_numpy(stats_map),
        "scalar_map_direct": torch.from_numpy(pred_scalar_np),
        "scalar_map_from_fij": torch.from_numpy(pred_fij_mi.astype(np.float32)),
        "tokens": STATS_TOKENS,
        "L": int(length),
        "q": STATS_Q,
        "id": rna_id,
        "query_sequence": query,
        "source": "RethinkingMSA_1 factorized_conditional_flow checkpoint",
        "checkpoint": str(args.checkpoint),
        "stats_flow_sample_steps": int(getattr(ckpt_args, "stats_flow_sample_steps", 4)),
        "stats_flow_objective": str(getattr(ckpt_args, "stats_flow_objective", "velocity")),
        "legalize_ipf_iters": int(args.legalize_ipf_iters),
        "validation": validation,
    }
    torch.save(stats_payload, out_dir / "adabm_stats.pt")

    metrics_direct = map_metrics(pred_scalar_np, target_scalar, length)
    metrics_fij = map_metrics(pred_fij_mi, target_scalar, length)
    row: dict[str, object] = {
        "id": rna_id,
        "length": int(length),
        "stats_path": str(out_dir / "adabm_stats.pt"),
        "direct_scalar_pearson": float(metrics_direct["pearson"]),
        "direct_scalar_spearman": float(metrics_direct["spearman"]),
        "direct_scalar_top_l_overlap": float(metrics_direct["top_l_overlap"]),
        "direct_scalar_mae": float(metrics_direct["mae"]),
        "fij_scalar_pearson": float(metrics_fij["pearson"]),
        "fij_scalar_spearman": float(metrics_fij["spearman"]),
        "fij_scalar_top_l_overlap": float(metrics_fij["top_l_overlap"]),
        "fij_scalar_mae": float(metrics_fij["mae"]),
        **{f"validation_{key}": value for key, value in validation.items() if key != "tokens"},
    }
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(row | {"query_sequence": query, "tokens": STATS_TOKENS}, handle, indent=2)
    return row


def main() -> None:
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    ckpt_args = checkpoint_args_namespace(dict(ckpt.get("args", {})), args)
    cfg = config_from_checkpoint(dict(ckpt["config"]))
    encoder = EvoNASSPairEncoder(cfg).to(device)
    scalar_head = build_head(ckpt_args, cfg, device)
    stats_head = build_stats_head(ckpt_args, cfg, device)
    if stats_head is None or getattr(ckpt_args, "stats_head_type", "") != "factorized_conditional_flow":
        raise ValueError("Checkpoint must use stats_head_type=factorized_conditional_flow")
    encoder.load_state_dict(ckpt["encoder"])
    scalar_head.load_state_dict(ckpt["head"])
    stats_head.load_state_dict(ckpt["stats_head"])
    encoder.eval()
    scalar_head.eval()
    stats_head.eval()

    ids = resolve_ids(args)
    dataset = RESMTestDataset(
        resm_dir=args.resm_dir,
        embedding_dir=args.embedding_dir,
        attention_dir=args.attention_dir,
        query_dir=args.query_dir,
        bpp_dir=args.bpp_dir,
        msa_dir=args.msa_dir,
        ids=ids,
        target_max_seqs=args.target_max_seqs,
        normalize_target=True,
        cache_targets=False,
        target_builder="adabm_stats",
        enable_stats_target=False,
    )

    rows: list[dict[str, object]] = []
    for rna_id in ids:
        print(f"[export] {rna_id}", flush=True)
        rows.append(export_sample(args, ckpt_args, encoder, scalar_head, stats_head, dataset, rna_id, device))

    fieldnames = sorted({key for row in rows for key in row})
    with (args.out_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    with (args.out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "checkpoint": str(args.checkpoint),
                "ids": ids,
                "n": len(ids),
                "device": str(device),
                "stats_flow_sample_steps": int(getattr(ckpt_args, "stats_flow_sample_steps", 4)),
                "rows": rows,
            },
            handle,
            indent=2,
        )
    with (args.out_dir / "stats_paths.txt").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(str(row["stats_path"]) + "\n")


if __name__ == "__main__":
    main()
