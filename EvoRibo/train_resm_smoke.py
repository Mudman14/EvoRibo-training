"""RESM650M -> coevolution diagnostic/flow training entrypoint."""

from __future__ import annotations

import argparse
import json
import math
import random
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .config import SSPairEncoderConfig
from .encoder import EvoNASSPairEncoder
from .flow import FlowMatchingHead, masked_upper_mse
from .heads import SymmetricUpperTriangleCoevoHead
from .resm_dataset import (
    RESMTestDataset,
    build_shape_report,
    collate_resm_samples,
    discover_resm_ids,
    split_train_val_test_ids,
    write_shape_report,
)


def build_parser() -> argparse.ArgumentParser:
    default_out = Path(__file__).resolve().parent / "smoke_runs" / "resm650m_smoke"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resm_dir", type=Path, default=Path("data/resm"))
    parser.add_argument("--embedding_dir", type=Path, default=None)
    parser.add_argument("--attention_dir", type=Path, default=None)
    parser.add_argument("--query_dir", type=Path, default=Path("data/query"))
    parser.add_argument("--bpp_dir", type=Path, default=Path("data/bpp"))
    parser.add_argument("--msa_dir", type=Path, default=Path("data/msa"))
    parser.add_argument("--out_dir", type=Path, default=default_out)
    parser.add_argument("--target_cache_dir", type=Path, default=None)
    parser.add_argument("--ids", default="auto", help="'auto' or comma-separated RNA IDs")
    parser.add_argument("--val_fasta", type=Path, default=None)
    parser.add_argument("--test_input_dir", type=Path, default=None)
    parser.add_argument("--audit_only", action="store_true")
    parser.add_argument("--skip_bad_audit", action="store_true")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--max_length",
        "--lmax",
        dest="max_length",
        type=int,
        default=0,
        help="Drop samples with RESM/query length above this value. 0 disables filtering.",
    )

    parser.add_argument("--head_type", choices=("diagnostic", "flow"), default="flow")
    parser.add_argument("--flow_objective", choices=("endpoint", "velocity"), default="endpoint")
    parser.add_argument("--x0_noise_scale", type=float, default=0.05)
    parser.add_argument("--prior_loss_weight", type=float, default=0.2)
    parser.add_argument("--flow_steps", type=int, default=16)
    parser.add_argument("--prior_hidden_dim", type=int, default=0, help="0 means 2 * d_z")
    parser.add_argument("--unet_base_channels", type=int, default=64)
    parser.add_argument("--unet_levels", type=int, default=3)
    parser.add_argument("--time_dim", type=int, default=128)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps", type=int, default=0, help="Optional optimizer-step limit for smoke runs")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--validate_every", type=int, default=5)
    parser.add_argument("--target_max_seqs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp_dtype", choices=("none", "bf16"), default="bf16")
    parser.add_argument("--no_gradient_checkpointing", action="store_true")
    parser.add_argument("--triangle_attention_chunk_size", type=int, default=64)

    parser.add_argument("--resm_single_dim", type=int, default=0, help="0 means infer from data")
    parser.add_argument("--resm_pair_dim", type=int, default=0, help="0 means infer from data")
    parser.add_argument("--d_s", type=int, default=32)
    parser.add_argument("--d_z", type=int, default=16)
    parser.add_argument("--num_blocks", type=int, default=1)
    parser.add_argument("--num_recycles", type=int, default=1)
    parser.add_argument("--sequence_head_width", type=int, default=16)
    parser.add_argument("--pairwise_head_width", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.0)
    return parser


def parse_ids(args: argparse.Namespace) -> list[str]:
    if args.ids.lower() == "auto":
        ids = discover_resm_ids(args.resm_dir, args.embedding_dir, args.attention_dir)
    else:
        ids = [item.strip() for item in args.ids.split(",") if item.strip()]
    if not ids:
        raise ValueError("No RESM samples found")
    return ids


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def pathlike_to_str(value):
    if isinstance(value, Path):
        return str(value)
    return value


def serializable_args(args: argparse.Namespace) -> dict[str, object]:
    return {key: pathlike_to_str(value) for key, value in vars(args).items()}


def infer_feature_dims(rows: list[dict[str, object]], args: argparse.Namespace) -> tuple[int, int]:
    ok_rows = [row for row in rows if row["status"] == "OK"]
    if not ok_rows:
        raise ValueError("Cannot infer feature dims because no samples passed audit")
    single_dims = {int(row["resm_single_dim"]) for row in ok_rows}
    pair_dims = {int(row["resm_pair_dim"]) for row in ok_rows}
    if len(single_dims) != 1:
        raise ValueError(f"Mixed RESM single dims: {sorted(single_dims)}")
    if len(pair_dims) != 1:
        raise ValueError(f"Mixed RESM pair dims: {sorted(pair_dims)}")
    single_dim = args.resm_single_dim or next(iter(single_dims))
    pair_dim = args.resm_pair_dim or next(iter(pair_dims))
    return single_dim, pair_dim


def assert_audit_ok(rows: list[dict[str, object]]) -> None:
    bad = [row for row in rows if row["status"] != "OK"]
    if bad:
        preview = "; ".join(f"{row['id']}={row['status']}" for row in bad[:5])
        raise ValueError(f"Shape audit failed for {len(bad)} samples: {preview}")


def filter_rows_by_max_length(
    rows: list[dict[str, object]],
    ids: list[str],
    max_length: int,
) -> tuple[list[dict[str, object]], list[str], list[str]]:
    """Keep only audited samples whose embedding/query length is <= max_length."""
    if max_length <= 0:
        return rows, ids, []

    kept_rows: list[dict[str, object]] = []
    removed: list[str] = []
    for row in rows:
        rna_id = str(row["id"])
        if row["status"] == "OK" and int(row["L_embedding"]) > max_length:
            removed.append(rna_id)
        else:
            kept_rows.append(row)

    kept_ids = {str(row["id"]) for row in kept_rows}
    return kept_rows, [rna_id for rna_id in ids if rna_id in kept_ids], removed


def move_batch_to_device(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    moved: dict[str, object] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return moved


def autocast_context(args: argparse.Namespace, device: torch.device):
    if args.amp_dtype == "bf16" and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def finite_gradients(parameters) -> bool:
    return all(param.grad is None or torch.isfinite(param.grad).all().item() for param in parameters)


def grad_norm(parameters) -> float:
    total = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        total += float(param.grad.detach().pow(2).sum().item())
    return math.sqrt(total)


def make_epoch_batches(
    ids: list[str],
    length_by_id: dict[str, int],
    batch_size: int,
    seed: int,
    epoch: int,
    shuffle: bool,
) -> list[list[str]]:
    ordered = sorted(ids, key=lambda rna_id: length_by_id[rna_id])
    if not shuffle:
        return [ordered[idx : idx + batch_size] for idx in range(0, len(ordered), batch_size)]

    rng = random.Random(seed + epoch)
    bucket_size = max(batch_size * 20, batch_size)
    buckets = [ordered[idx : idx + bucket_size] for idx in range(0, len(ordered), bucket_size)]
    batches: list[list[str]] = []
    for bucket in buckets:
        rng.shuffle(bucket)
        batches.extend(bucket[idx : idx + batch_size] for idx in range(0, len(bucket), batch_size))
    rng.shuffle(batches)
    return batches


def dataset_batch(dataset: RESMTestDataset, id_batch: list[str], id_to_index: dict[str, int]) -> dict[str, object]:
    samples = [dataset[id_to_index[rna_id]] for rna_id in id_batch]
    return collate_resm_samples(samples)


def build_encoder_config(args: argparse.Namespace, rows: list[dict[str, object]]) -> SSPairEncoderConfig:
    single_dim, pair_dim = infer_feature_dims(rows, args)
    return SSPairEncoderConfig(
        resm_single_dim=single_dim,
        resm_pair_dim=pair_dim,
        d_s=args.d_s,
        d_z=args.d_z,
        num_blocks=args.num_blocks,
        num_recycles=args.num_recycles,
        sequence_head_width=args.sequence_head_width,
        pairwise_head_width=args.pairwise_head_width,
        dropout=args.dropout,
        use_gradient_checkpointing=not args.no_gradient_checkpointing,
        triangle_attention_chunk_size=args.triangle_attention_chunk_size,
    )


def build_models(args: argparse.Namespace, cfg: SSPairEncoderConfig, device: torch.device):
    encoder = EvoNASSPairEncoder(cfg).to(device)
    if args.head_type == "diagnostic":
        head = SymmetricUpperTriangleCoevoHead(d_z=cfg.d_z, hidden_dim=max(32, cfg.d_z * 2)).to(device)
    else:
        prior_hidden_dim = args.prior_hidden_dim or max(32, cfg.d_z * 2)
        head = FlowMatchingHead(
            d_z=cfg.d_z,
            prior_hidden_dim=prior_hidden_dim,
            unet_base_channels=args.unet_base_channels,
            unet_levels=args.unet_levels,
            time_dim=args.time_dim,
            objective=args.flow_objective,
        ).to(device)
    return encoder, head


def compute_train_loss(
    args: argparse.Namespace,
    encoder: EvoNASSPairEncoder,
    head: torch.nn.Module,
    batch: dict[str, object],
) -> tuple[torch.Tensor, dict[str, float]]:
    device = batch["resm_single"].device
    with autocast_context(args, device):
        out = encoder(
            batch["resm_single"],
            batch["resm_pair"],
            batch["bpp"],
            batch["tokens"],
            batch["mask"],
        )
        target = batch["target"]
        pair_mask = out["pair_mask"]
        if args.head_type == "diagnostic":
            pred_upper = head(out["pair"], pair_mask, return_full=False)
            loss = masked_upper_mse(pred_upper[:, None], target, pair_mask)
            return loss, {"loss": float(loss.detach().cpu().item())}

        flow_out = head(
            out["pair"],
            target,
            pair_mask,
            objective=args.flow_objective,
            x0_noise_scale=args.x0_noise_scale,
        )
        total = flow_out["flow_loss"] + args.prior_loss_weight * flow_out["prior_loss"]
    return total, {
        "loss": float(total.detach().cpu().item()),
        "flow_loss": float(flow_out["flow_loss"].detach().cpu().item()),
        "prior_loss": float(flow_out["prior_loss"].detach().cpu().item()),
    }


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    return ranks


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return 0.0
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt(np.sum(x * x) * np.sum(y * y))
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(x * y) / denom)


def map_metrics(pred: np.ndarray, target: np.ndarray, length: int) -> dict[str, float]:
    valid = np.triu(np.ones((length, length), dtype=bool), k=1)
    pred_v = pred[:length, :length][valid]
    target_v = target[:length, :length][valid]
    mae = float(np.mean(np.abs(pred_v - target_v))) if pred_v.size else 0.0
    pearson = _corr(pred_v, target_v)
    spearman = _corr(_rankdata(pred_v), _rankdata(target_v)) if pred_v.size else 0.0
    k = min(length, pred_v.size)
    if k > 0:
        pred_top = set(np.argpartition(pred_v, -k)[-k:].tolist())
        target_top = set(np.argpartition(target_v, -k)[-k:].tolist())
        top_l_overlap = len(pred_top & target_top) / float(k)
    else:
        top_l_overlap = 0.0
    return {
        "pearson": pearson,
        "spearman": spearman,
        "mae": mae,
        "top_l_overlap": top_l_overlap,
    }


@torch.no_grad()
def validate(
    args: argparse.Namespace,
    encoder: EvoNASSPairEncoder,
    head: torch.nn.Module,
    dataset: RESMTestDataset,
    length_by_id: dict[str, int],
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    if len(dataset) == 0:
        return {}

    encoder.eval()
    head.eval()
    id_to_index = {rna_id: idx for idx, rna_id in enumerate(dataset.ids)}
    batches = make_epoch_batches(dataset.ids, length_by_id, args.batch_size, args.seed, epoch, shuffle=False)
    val_dir = args.out_dir / "validation" / f"epoch_{epoch:04d}"
    val_dir.mkdir(parents=True, exist_ok=True)

    per_sample: list[dict[str, float]] = []
    torch.manual_seed(args.seed + epoch)
    for id_batch in batches:
        batch = move_batch_to_device(dataset_batch(dataset, id_batch, id_to_index), device)
        with autocast_context(args, device):
            out = encoder(
                batch["resm_single"],
                batch["resm_pair"],
                batch["bpp"],
                batch["tokens"],
                batch["mask"],
            )
            if args.head_type == "diagnostic":
                pred = head(out["pair"], out["pair_mask"], return_full=True)
                prior = pred
            else:
                sample_out = head.sample(
                    out["pair"],
                    out["pair_mask"],
                    objective=args.flow_objective,
                    steps=args.flow_steps,
                    x0_noise_scale=args.x0_noise_scale,
                )
                pred = sample_out["sample"]
                prior = sample_out["x0_mean"]

        target = batch["target"]
        for idx, rna_id in enumerate(batch["id"]):
            L = int(batch["length"][idx].item())
            pred_np = pred[idx, 0, :L, :L].detach().cpu().numpy()
            target_np = target[idx, 0, :L, :L].detach().cpu().numpy()
            prior_np = prior[idx, 0, :L, :L].detach().cpu().numpy()
            metrics = map_metrics(pred_np, target_np, L)
            metrics["length"] = float(L)
            metrics["target_scale"] = float(batch["target_scale"][idx].detach().cpu().item())
            per_sample.append({"id": rna_id, **metrics})
            np.save(val_dir / f"{rna_id}_sampled_map.npy", pred_np)
            np.save(val_dir / f"{rna_id}_prior_map.npy", prior_np)
            np.save(val_dir / f"{rna_id}_target_map.npy", target_np)

    with (val_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(per_sample, handle, indent=2)
    summary: dict[str, float] = {}
    for key in ("pearson", "spearman", "mae", "top_l_overlap"):
        summary[key] = float(np.mean([row[key] for row in per_sample])) if per_sample else 0.0
    return summary


def save_checkpoint(
    path: Path,
    encoder: EvoNASSPairEncoder,
    head: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: SSPairEncoderConfig,
    args: argparse.Namespace,
    epoch: int,
    global_step: int,
    best_metric: float | None,
    train_logs: list[dict[str, object]],
    val_logs: list[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "head": head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": asdict(cfg),
            "args": serializable_args(args),
            "epoch": epoch,
            "global_step": global_step,
            "best_metric": best_metric,
            "train_logs": train_logs,
            "val_logs": val_logs,
        },
        path,
    )


def load_checkpoint(
    resume_path: Path,
    encoder: EvoNASSPairEncoder,
    head: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, int, float | None, list[dict[str, object]], list[dict[str, object]]]:
    ckpt = torch.load(resume_path, map_location=device)
    encoder.load_state_dict(ckpt["encoder"])
    head.load_state_dict(ckpt["head"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return (
        int(ckpt.get("epoch", 0)) + 1,
        int(ckpt.get("global_step", 0)),
        ckpt.get("best_metric"),
        list(ckpt.get("train_logs", [])),
        list(ckpt.get("val_logs", [])),
    )


def write_metrics(args: argparse.Namespace, cfg: SSPairEncoderConfig, split: dict[str, list[str]], train_logs, val_logs) -> None:
    result = {
        "config": asdict(cfg),
        "args": serializable_args(args),
        "split": split,
        "train_logs": train_logs,
        "val_logs": val_logs,
    }
    with (args.out_dir / "train_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)


def run_training(
    args: argparse.Namespace,
    cfg: SSPairEncoderConfig,
    rows: list[dict[str, object]],
    train_ids: list[str],
    val_ids: list[str],
    split: dict[str, list[str]],
) -> None:
    if not train_ids:
        raise ValueError("No training IDs remain after val/test exclusion")

    device = choose_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.target_cache_dir is None:
        args.target_cache_dir = args.out_dir / "target_cache"

    length_by_id = {str(row["id"]): int(row["L_embedding"]) for row in rows if row["status"] == "OK"}
    train_dataset = RESMTestDataset(
        args.resm_dir,
        args.query_dir,
        args.bpp_dir,
        args.msa_dir,
        ids=train_ids,
        target_max_seqs=args.target_max_seqs,
        embedding_dir=args.embedding_dir,
        attention_dir=args.attention_dir,
        target_cache_dir=args.target_cache_dir,
    )
    val_dataset = RESMTestDataset(
        args.resm_dir,
        args.query_dir,
        args.bpp_dir,
        args.msa_dir,
        ids=val_ids,
        target_max_seqs=args.target_max_seqs,
        embedding_dir=args.embedding_dir,
        attention_dir=args.attention_dir,
        target_cache_dir=args.target_cache_dir,
    )
    train_index = {rna_id: idx for idx, rna_id in enumerate(train_dataset.ids)}

    encoder, head = build_models(args, cfg, device)
    params = list(encoder.parameters()) + list(head.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    start_epoch = 1
    global_step = 0
    best_metric: float | None = None
    train_logs: list[dict[str, object]] = []
    val_logs: list[dict[str, object]] = []
    if args.resume is not None:
        start_epoch, global_step, best_metric, train_logs, val_logs = load_checkpoint(
            args.resume,
            encoder,
            head,
            optimizer,
            device,
        )
        print(f"Resumed {args.resume} at epoch={start_epoch} global_step={global_step}")

    stop_training = False
    for epoch in range(start_epoch, args.epochs + 1):
        encoder.train()
        head.train()
        batches = make_epoch_batches(train_ids, length_by_id, args.batch_size, args.seed, epoch, shuffle=True)
        optimizer.zero_grad(set_to_none=True)
        epoch_losses: list[float] = []
        accum = 0
        for batch_num, id_batch in enumerate(batches, start=1):
            batch = move_batch_to_device(dataset_batch(train_dataset, id_batch, train_index), device)
            print(
                f"epoch {epoch:03d} batch {batch_num:05d} preforward "
                f"Lmax={int(batch['length'].max().item())} ids={','.join(batch['id'])}",
                flush=True,
            )
            loss, loss_parts = compute_train_loss(args, encoder, head, batch)
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite loss at epoch {epoch} batch {batch_num}: {loss.item()}")
            (loss / args.grad_accum).backward()
            accum += 1
            epoch_losses.append(loss_parts["loss"])

            should_step = accum >= args.grad_accum or batch_num == len(batches)
            if should_step:
                if not finite_gradients(params):
                    raise ValueError(f"Non-finite gradient at epoch {epoch} batch {batch_num}")
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
                norm = grad_norm(params)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                accum = 0
                log_row = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "batch": batch_num,
                    "ids": list(batch["id"]),
                    "length_max": int(batch["length"].max().item()),
                    "grad_norm": norm,
                    **loss_parts,
                }
                train_logs.append(log_row)
                print(
                    f"epoch {epoch:03d} step {global_step:05d} "
                    f"Lmax={log_row['length_max']} loss={loss_parts['loss']:.6f}"
                )
                if args.steps and global_step >= args.steps:
                    stop_training = True
                    break

        train_mean = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        val_summary = {}
        if val_ids and args.validate_every > 0 and epoch % args.validate_every == 0:
            val_summary = validate(args, encoder, head, val_dataset, length_by_id, device, epoch)
            val_log = {"epoch": epoch, "global_step": global_step, **val_summary}
            val_logs.append(val_log)
            print(
                f"validation epoch {epoch:03d}: "
                f"pearson={val_summary.get('pearson', 0.0):.4f} "
                f"mae={val_summary.get('mae', 0.0):.4f} "
                f"topL={val_summary.get('top_l_overlap', 0.0):.4f}"
            )

        monitor = val_summary.get("mae", train_mean) if val_summary else train_mean
        if best_metric is None or monitor < best_metric:
            best_metric = monitor
            save_checkpoint(
                args.out_dir / "best.pt",
                encoder,
                head,
                optimizer,
                cfg,
                args,
                epoch,
                global_step,
                best_metric,
                train_logs,
                val_logs,
            )
        save_checkpoint(
            args.out_dir / "last.pt",
            encoder,
            head,
            optimizer,
            cfg,
            args,
            epoch,
            global_step,
            best_metric,
            train_logs,
            val_logs,
        )
        write_metrics(args, cfg, split, train_logs, val_logs)

        if stop_training:
            break

    print(f"Training complete. Metrics written to {args.out_dir / 'train_metrics.json'}")


def main() -> None:
    args = build_parser().parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")
    if args.grad_accum < 1:
        raise ValueError("--grad_accum must be >= 1")
    if args.target_max_seqs < 2:
        raise ValueError("--target_max_seqs must be >= 2")
    if args.triangle_attention_chunk_size < 0:
        raise ValueError("--triangle_attention_chunk_size must be >= 0")
    if args.max_length < 0:
        raise ValueError("--max_length/--lmax must be >= 0")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_ids = parse_ids(args)
    rows = build_shape_report(
        args.resm_dir,
        args.query_dir,
        args.bpp_dir,
        args.msa_dir,
        ids=all_ids,
        embedding_dir=args.embedding_dir,
        attention_dir=args.attention_dir,
    )
    write_shape_report(rows, args.out_dir)
    if args.skip_bad_audit:
        bad = [row for row in rows if row["status"] != "OK"]
        if bad:
            preview = "; ".join(f"{row['id']}={row['status']}" for row in bad[:5])
            print(f"Skipping {len(bad)} samples that failed audit: {preview}")
            rows = [row for row in rows if row["status"] == "OK"]
            ok_ids = {str(row["id"]) for row in rows}
            all_ids = [rna_id for rna_id in all_ids if rna_id in ok_ids]
            if not all_ids:
                raise ValueError("All samples failed audit")
            write_shape_report(rows, args.out_dir)
    else:
        assert_audit_ok(rows)

    rows, all_ids, removed_long = filter_rows_by_max_length(rows, all_ids, args.max_length)
    if removed_long:
        preview = ",".join(removed_long[:8])
        print(
            f"Length filter L <= {args.max_length}: "
            f"removed={len(removed_long)} kept={len(all_ids)} preview={preview}"
        )
        if not all_ids:
            raise ValueError("All samples were removed by --max_length/--lmax")
        write_shape_report(rows, args.out_dir)

    train_ids, val_ids, test_ids = split_train_val_test_ids(all_ids, args.val_fasta, args.test_input_dir)
    split = {"train": train_ids, "val": val_ids, "test": test_ids}
    with (args.out_dir / "split.json").open("w", encoding="utf-8") as handle:
        json.dump(split, handle, indent=2)
    print(
        f"Shape audit OK for {len(rows)} samples. "
        f"Split: train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}"
    )

    cfg = build_encoder_config(args, rows)
    if args.audit_only:
        print(f"Reports written to {args.out_dir}")
        return

    run_training(args, cfg, rows, train_ids, val_ids, split)


if __name__ == "__main__":
    main()
