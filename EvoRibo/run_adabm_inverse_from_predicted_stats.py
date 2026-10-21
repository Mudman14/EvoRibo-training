"""Fit adabmDCA Potts models from predicted fi/fij stats and sample RNA MSAs.

Input is one or more ``adabm_stats.pt`` files produced by
``export_adabm_predicted_stats.py``.  The original MSA is not read during this
stage; adabmDCA receives only predicted ``fi`` and ``fij`` tensors.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import types
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


if "wandb" not in sys.modules:
    sys.modules["wandb"] = types.SimpleNamespace(
        init=lambda *args, **kwargs: None,
        log=lambda *args, **kwargs: None,
    )

from adabmDCA.fasta import decode_sequence
from adabmDCA.io import save_chains, save_params
from adabmDCA.sampling import get_sampler
from adabmDCA.stats import get_correlation_two_points, get_freq_single_point, get_freq_two_points
from adabmDCA.training import train_graph
from adabmDCA.utils import get_mask_save, init_chains, init_parameters

from .stat_targets import STATS_Q, STATS_TOKENS, mi_apc_from_fij


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats_root", type=Path, default=None)
    parser.add_argument("--stats_path", type=Path, default=None, help="Run one explicit adabm_stats.pt file")
    parser.add_argument("--ids", default="auto", help="'auto' or comma-separated IDs under --stats_root")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--nchains", type=int, default=10000)
    parser.add_argument("--ngen", type=int, default=1000)
    parser.add_argument("--nsweeps", type=int, default=5)
    parser.add_argument("--sample_nsweeps", type=int, default=50)
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--target_pearson", type=float, default=0.90)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--l2_reg", type=float, default=0.0)
    parser.add_argument("--sampler", choices=("gibbs", "metropolis"), default="gibbs")
    parser.add_argument("--progress_bar", action="store_true")
    parser.add_argument(
        "--resume_training",
        action="store_true",
        help="Continue from an existing adabm_inverse_result.pt; max_epochs is the additional budget.",
    )
    parser.add_argument("--force", action="store_true")
    return parser


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def choose_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError(f"unsupported dtype: {name}")


def resolve_stats_paths(args: argparse.Namespace) -> list[tuple[str, Path]]:
    if args.stats_path is not None:
        path = args.stats_path
        rna_id = path.parent.name if path.name == "adabm_stats.pt" else path.stem
        return [(rna_id, path)]
    if args.stats_root is None:
        raise ValueError("Either --stats_root or --stats_path is required")
    if args.ids.lower() == "auto":
        pairs = [(path.parent.name, path) for path in sorted(args.stats_root.glob("*/adabm_stats.pt"))]
    else:
        pairs = [(rna_id, args.stats_root / rna_id / "adabm_stats.pt") for rna_id in args.ids.split(",") if rna_id.strip()]
    missing = [str(path) for _, path in pairs if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing stats files:\n" + "\n".join(missing[:20]))
    if not pairs:
        raise ValueError(f"No adabm_stats.pt files found under {args.stats_root}")
    return pairs


def validate_stats(fi: torch.Tensor, fij: torch.Tensor, tokens: str) -> dict[str, float | int | bool | str]:
    if tokens != STATS_TOKENS:
        raise ValueError(f"Expected tokens {STATS_TOKENS!r}, got {tokens!r}")
    if fi.ndim != 2 or fi.shape[1] != STATS_Q:
        raise ValueError(f"fi must be [L,5], got {tuple(fi.shape)}")
    length = int(fi.shape[0])
    if tuple(fij.shape) != (length, STATS_Q, length, STATS_Q):
        raise ValueError(f"fij must be [L,5,L,5], got {tuple(fij.shape)}")
    off_sums = []
    diag_errors = []
    for i in range(length):
        diag_errors.append(float((fij[i, :, i, :] - torch.diag(fi[i])).abs().max().item()))
        for j in range(length):
            if i != j:
                off_sums.append(float(fij[i, :, j, :].sum().item()))
    return {
        "tokens": tokens,
        "length": length,
        "finite": bool(torch.isfinite(fi).all().item() and torch.isfinite(fij).all().item()),
        "nonnegative": bool((fi >= 0).all().item() and (fij >= 0).all().item()),
        "fi_sum_min": float(fi.sum(dim=-1).min().item()),
        "fi_sum_max": float(fi.sum(dim=-1).max().item()),
        "offdiag_pair_sum_min": float(min(off_sums) if off_sums else 1.0),
        "offdiag_pair_sum_max": float(max(off_sums) if off_sums else 1.0),
        "diag_max_abs_error": float(max(diag_errors) if diag_errors else 0.0),
    }


def decoded_samples_from_chains(chains: torch.Tensor, tokens: str) -> list[str]:
    decoded = decode_sequence(chains.detach().cpu(), tokens)
    if isinstance(decoded, str):
        return [decoded]
    return [str(item) for item in decoded.tolist()]


def write_fasta(path: Path, records: Iterable[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for header, seq in records:
            handle.write(f">{header}\n")
            for start in range(0, len(seq), 80):
                handle.write(seq[start : start + 80] + "\n")


def upper_pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.shape != b.shape or a.ndim != 2:
        return None
    idx = np.triu_indices(a.shape[0], k=1)
    x = a[idx].astype(np.float64)
    y = b[idx].astype(np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2:
        return None
    x = x[ok]
    y = y[ok]
    if x.std() <= 1e-12 or y.std() <= 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def run_one(args: argparse.Namespace, rna_id: str, stats_path: Path, device: torch.device, dtype: torch.dtype) -> dict[str, object]:
    out_dir = args.out_dir / rna_id
    metrics_path = out_dir / "metrics.json"
    result_path = out_dir / "adabm_inverse_result.pt"
    existing_metrics = None
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as handle:
            existing_metrics = json.load(handle)
        if args.resume_training and float(existing_metrics.get("final_pearson", -math.inf)) >= args.target_pearson:
            return existing_metrics
        if not args.force and not args.resume_training:
            return existing_metrics
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = torch.load(stats_path, map_location="cpu")
    tokens = str(stats.get("tokens", STATS_TOKENS))
    fi = stats["fi"].to(device=device, dtype=dtype).clamp_min(1e-8)
    fi = fi / fi.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    fij = stats["fij"].to(device=device, dtype=dtype).clamp_min(1e-8)
    L, q = fi.shape
    idx = torch.arange(L, device=device)
    fij[idx, :, idx, :] = torch.diag_embed(fi).to(dtype)
    validation = validate_stats(fi, fij, tokens)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    mask = torch.ones((L, q, L, q), dtype=torch.bool, device=device)
    mask[idx, :, idx, :] = False
    previous_history: dict[str, list] = {}
    resumed_from_epochs = 0
    if args.resume_training:
        if not result_path.exists():
            raise FileNotFoundError(f"Cannot resume {rna_id}: missing {result_path}")
        previous = torch.load(result_path, map_location="cpu", weights_only=False)
        if str(previous.get("tokens")) != tokens:
            raise ValueError(f"Cannot resume {rna_id}: token alphabet changed")
        params = {key: value.to(device=device, dtype=dtype) for key, value in previous["params"].items()}
        chains = previous["final_chains"].to(device=device, dtype=dtype)
        log_weights = previous["log_weights"].to(device=device, dtype=dtype)
        if tuple(chains.shape) != (args.nchains, L, q):
            raise ValueError(
                f"Cannot resume {rna_id}: chains shape {tuple(chains.shape)} != {(args.nchains, L, q)}"
            )
        if tuple(log_weights.shape) != (args.nchains,):
            raise ValueError(f"Cannot resume {rna_id}: log_weights shape {tuple(log_weights.shape)}")
        previous_history = {key: list(value) for key, value in previous.get("history", {}).items()}
        resumed_from_epochs = len(previous_history.get("Epochs", []))
    else:
        params = init_parameters(fi=fi)
        chains = init_chains(num_chains=args.nchains, L=L, q=q, fi=fi, device=device, dtype=dtype)
        log_weights = torch.zeros(args.nchains, device=device, dtype=dtype)
    sampler = torch.jit.script(get_sampler(args.sampler))

    with torch.no_grad():
        pi0 = get_freq_single_point(data=chains)
        pij0 = get_freq_two_points(data=chains)
        initial_pearson, initial_slope = get_correlation_two_points(fij=fij, pij=pij0, fi=fi, pi=pi0)

    chains, params, log_weights, current_history = train_graph(
        sampler=sampler,
        chains=chains,
        mask=mask,
        fi_target=fi,
        fij_target=fij,
        params=params,
        nsweeps=args.nsweeps,
        lr=args.lr,
        max_epochs=args.max_epochs,
        target_pearson=args.target_pearson,
        log_weights=log_weights,
        progress_bar=args.progress_bar,
        l2_reg=args.l2_reg,
    )
    history = previous_history
    if history:
        epoch_offset = int(history.get("Epochs", [0])[-1]) if history.get("Epochs") else resumed_from_epochs
        time_offset = float(history.get("Time", [0.0])[-1]) if history.get("Time") else 0.0
        for key, values in current_history.items():
            adjusted = list(values)
            if key == "Epochs":
                adjusted = [int(value) + epoch_offset for value in adjusted]
            elif key == "Time":
                adjusted = [float(value) + time_offset for value in adjusted]
            history.setdefault(key, []).extend(adjusted)
    else:
        history = {key: list(value) for key, value in current_history.items()}

    with torch.no_grad():
        pi_final = get_freq_single_point(data=chains)
        pij_final = get_freq_two_points(data=chains)
        final_pearson, final_slope = get_correlation_two_points(fij=fij, pij=pij_final, fi=fi, pi=pi_final)

        if args.ngen <= len(chains):
            selected = torch.randperm(len(chains), device=device)[: args.ngen]
            sample_chains = chains[selected].clone()
            generation_source = "trained_chains_subset"
        else:
            sample_chains = init_chains(num_chains=args.ngen, L=L, q=q, fi=fi, device=device, dtype=dtype)
            sample_chains = sampler(chains=sample_chains, params=params, nsweeps=args.sample_nsweeps)
            generation_source = "fresh_chains"
        if args.sample_nsweeps > 0:
            sample_chains = sampler(chains=sample_chains, params=params, nsweeps=args.sample_nsweeps)
            generation_source += "_resampled"
        pi_sample = get_freq_single_point(data=sample_chains)
        pij_sample = get_freq_two_points(data=sample_chains)
        sample_pearson, sample_slope = get_correlation_two_points(fij=fij, pij=pij_sample, fi=fi, pi=pi_sample)

    seqs = decoded_samples_from_chains(sample_chains, tokens)
    fasta_records = [(f"{rna_id}_sample_{idx + 1}", seq) for idx, seq in enumerate(seqs)]
    write_fasta(out_dir / "samples.fasta", fasta_records)
    write_fasta(out_dir / "samples.a3m", fasta_records)
    save_chains(str(out_dir / "final_chains.fasta"), chains.argmax(dim=-1), tokens=tokens, log_weights=log_weights)
    save_params(str(out_dir / "params.dat"), params=params, tokens=tokens, mask=get_mask_save(L, q, device=device))

    target_apc = mi_apc_from_fij(fij.detach().cpu().numpy())
    sample_apc = mi_apc_from_fij(pij_sample.detach().cpu().numpy())
    np.save(out_dir / "target_mi_apc_from_predicted_fij.npy", target_apc)
    np.save(out_dir / "sample_mi_apc.npy", sample_apc)
    torch.save(
        {
            "params": {key: value.detach().cpu() for key, value in params.items()},
            "final_chains": chains.detach().cpu(),
            "sample_chains": sample_chains.detach().cpu(),
            "log_weights": log_weights.detach().cpu(),
            "history": history,
            "tokens": tokens,
            "source_stats_path": str(stats_path),
        },
        result_path,
    )
    metrics = {
        "id": rna_id,
        "stats_path": str(stats_path),
        "out_dir": str(out_dir),
        "tokens": tokens,
        "L": int(L),
        "q": int(q),
        "nchains": int(args.nchains),
        "ngen": int(args.ngen),
        "nsweeps": int(args.nsweeps),
        "sample_nsweeps": int(args.sample_nsweeps),
        "max_epochs": int(args.max_epochs),
        "target_pearson": float(args.target_pearson),
        "lr": float(args.lr),
        "resume_training": bool(args.resume_training),
        "resumed_from_epochs": int(resumed_from_epochs),
        "initial_pearson": float(initial_pearson),
        "initial_slope": float(initial_slope),
        "final_pearson": float(final_pearson),
        "final_slope": float(final_slope),
        "sample_pearson": float(sample_pearson),
        "sample_slope": float(sample_slope),
        "scalar_mi_apc_pearson": upper_pearson(target_apc, sample_apc),
        "train_epochs": int(len(history.get("Epochs", []))),
        "train_epochs_this_run": int(len(current_history.get("Epochs", []))),
        "history_last_epoch": int(history["Epochs"][-1]) if history.get("Epochs") else 0,
        "generation_source": generation_source,
        "finite_metrics": bool(
            np.isfinite([initial_pearson, initial_slope, final_pearson, final_slope, sample_pearson, sample_slope]).all()
        ),
        "stats_validation": validation,
    }
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    return metrics


def main() -> None:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    dtype = choose_dtype(args.dtype)
    pairs = resolve_stats_paths(args)
    rows = []
    for rna_id, stats_path in pairs:
        print(f"[adabm inverse] {rna_id} {stats_path}", flush=True)
        rows.append(run_one(args, rna_id, stats_path, device=device, dtype=dtype))

    fieldnames = sorted({key for row in rows for key in row if not isinstance(row.get(key), dict)})
    with (args.out_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: value for key, value in row.items() if key in fieldnames})
    with (args.out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"n": len(rows), "device": str(device), "rows": rows}, handle, indent=2)


if __name__ == "__main__":
    main()
