"""Single-node DDP training for the direct RESM/BPP trunk."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import random
import time
import traceback
from contextlib import ExitStack, contextmanager
from datetime import timedelta
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .config import SSPairEncoderConfig
from .encoder import EvoNASSPairEncoder
from .resm_dataset import (
    RESMTestDataset,
    build_shape_report,
    discover_resm_ids,
    read_fasta_ids,
    read_test_folder_ids,
    write_shape_report,
)
from .train_direct_trunk import (
    best_checkpoint_score,
    build_encoder_config,
    build_stats_head,
    build_models,
    build_parser as build_single_gpu_parser,
    compute_train_loss,
    load_checkpoint,
    save_checkpoint,
    validation_best_monitor,
    validate,
)
from .train_resm_smoke import (
    assert_audit_ok,
    dataset_batch,
    filter_rows_by_max_length,
    finite_gradients,
    grad_norm,
    make_epoch_batches,
    move_batch_to_device,
    pathlike_to_str,
)


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    distributed: bool
    device: torch.device
    backend: str


@dataclass(frozen=True)
class DataSource:
    name: str
    resm_dir: Path | None
    embedding_dir: Path | None
    attention_dir: Path | None
    query_dir: Path
    bpp_dir: Path
    msa_dir: Path
    ids: str = "auto"


@dataclass(frozen=True)
class SourceSample:
    source: DataSource
    source_id: str
    sample_id: str


def add_missing_argument(parser: argparse.ArgumentParser, *args, **kwargs) -> None:
    option = next((arg for arg in args if isinstance(arg, str) and arg.startswith("--")), None)
    if option is None:
        parser.add_argument(*args, **kwargs)
        return
    for action in parser._actions:
        if option in action.option_strings:
            return
    parser.add_argument(*args, **kwargs)


class MultiSourceRESMDataset:
    """Dataset facade that keeps file lookup source-local but exposes global sample IDs."""

    def __init__(
        self,
        samples_by_id: dict[str, SourceSample],
        ids: list[str],
        target_max_seqs: int,
        target_cache_dir: Path | None,
        target_builder: str = "adabm_stats",
        enable_stats_target: bool = False,
        include_stats_target_map: bool = True,
        target_clustering_seqid: float = 0.8,
        target_remove_duplicates: bool = True,
        target_no_reweighting: bool = False,
        target_pseudocount: float | None = None,
        cache_targets: bool = False,
    ) -> None:
        self.ids = ids
        self.samples_by_id = samples_by_id
        self.datasets: dict[str, RESMTestDataset] = {}
        self.local_index: dict[str, dict[str, int]] = {}
        for sample_id in ids:
            sample = samples_by_id[sample_id]
            if sample.source.name in self.datasets:
                continue
            source_ids = [
                item.source_id
                for item in samples_by_id.values()
                if item.source.name == sample.source.name
            ]
            source_cache_dir = None
            if target_cache_dir is not None:
                source_cache_dir = target_cache_dir / sample.source.name
            dataset = RESMTestDataset(
                sample.source.resm_dir,
                sample.source.query_dir,
                sample.source.bpp_dir,
                sample.source.msa_dir,
                ids=source_ids,
                target_max_seqs=target_max_seqs,
                embedding_dir=sample.source.embedding_dir,
                attention_dir=sample.source.attention_dir,
                target_cache_dir=source_cache_dir,
                target_builder=target_builder,
                enable_stats_target=enable_stats_target,
                include_stats_target_map=include_stats_target_map,
                cache_targets=cache_targets,
                target_clustering_seqid=target_clustering_seqid,
                target_remove_duplicates=target_remove_duplicates,
                target_no_reweighting=target_no_reweighting,
                target_pseudocount=target_pseudocount,
            )
            self.datasets[sample.source.name] = dataset
            self.local_index[sample.source.name] = {
                source_id: idx for idx, source_id in enumerate(dataset.ids)
            }

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample_id = self.ids[index]
        sample = self.samples_by_id[sample_id]
        dataset = self.datasets[sample.source.name]
        local_idx = self.local_index[sample.source.name][sample.source_id]
        item = dict(dataset[local_idx])
        item["id"] = sample_id
        item["source"] = sample.source.name
        item["source_id"] = sample.source_id
        return item


def build_parser() -> argparse.ArgumentParser:
    parser = build_single_gpu_parser()
    parser.description = __doc__
    parser.set_defaults(
        out_dir=Path(__file__).resolve().parent / "smoke_runs" / "direct_trunk_ddp",
        target_max_seqs=2001,
    )
    add_missing_argument(parser, "--stats_flow_eval_sample_steps", default="")
    add_missing_argument(parser, "--stats_scalar_consistency_warmup_epochs", type=int, default=0)
    add_missing_argument(parser, "--stats_flow_condition_dim", type=int, default=64)
    add_missing_argument(parser, "--stats_prior_loss_weight", type=float, default=0.2)
    add_missing_argument(parser, "--stats_prior_warmup_start", type=float, default=0.05)
    add_missing_argument(parser, "--stats_prior_warmup_epochs", type=int, default=10)
    add_missing_argument(parser, "--stats_fij_velocity_weight", type=float, default=1.0)
    add_missing_argument(parser, "--stats_fij_endpoint_weight", type=float, default=1.0)
    add_missing_argument(parser, "--stats_diag_weight", type=float, default=0.2)
    parser.add_argument(
        "--no_ddp_static_graph",
        action="store_true",
        help="Disable DDP static_graph even when the installed PyTorch supports it.",
    )
    parser.add_argument(
        "--shared_target_cache",
        action="store_true",
        help="Use one MI/APC target cache directory across ranks. Use only with a precomputed cache.",
    )
    parser.add_argument(
        "--target_cache_rank_offset",
        type=int,
        choices=(-1, 0, 1),
        default=-1,
        help=(
            "Rank subdirectory offset for DDP target caches. -1 auto-detects; 0 uses rank_000..; "
            "1 uses legacy/precomputed rank_001..rank_00N."
        ),
    )
    parser.add_argument(
        "--memory_target_cache",
        action="store_true",
        help=(
            "Keep per-sample target tensors in process memory. Disabled by default for large DDP runs "
            "because fij/stat targets can grow CPU RAM across epochs."
        ),
    )
    parser.add_argument(
        "--ddp_timeout_minutes",
        type=int,
        default=120,
        help=(
            "Process-group collective timeout. Large target-cache misses or long L batches can make "
            "one rank reach allreduce much later than others; keep this above worst-case single-batch time."
        ),
    )
    parser.add_argument(
        "--ddp_log_batch_timing",
        action="store_true",
        help="Print pre-load and timing diagnostics for every DDP local batch.",
    )
    parser.add_argument(
        "--ddp_slow_batch_seconds",
        type=float,
        default=60.0,
        help="Always print timing diagnostics for batches slower than this threshold.",
    )
    parser.add_argument("--dataset1_name", default="nufold")
    parser.add_argument("--dataset2_name", default="repa_round1")
    parser.add_argument("--dataset2_resm_dir", type=Path, default=None)
    parser.add_argument("--dataset2_embedding_dir", type=Path, default=None)
    parser.add_argument("--dataset2_attention_dir", type=Path, default=None)
    parser.add_argument("--dataset2_query_dir", type=Path, default=None)
    parser.add_argument("--dataset2_bpp_dir", type=Path, default=None)
    parser.add_argument("--dataset2_msa_dir", type=Path, default=None)
    parser.add_argument("--dataset2_ids", default="auto")
    parser.add_argument("--dataset3_name", default="nufolddb_round3")
    parser.add_argument("--dataset3_resm_dir", type=Path, default=None)
    parser.add_argument("--dataset3_embedding_dir", type=Path, default=None)
    parser.add_argument("--dataset3_attention_dir", type=Path, default=None)
    parser.add_argument("--dataset3_query_dir", type=Path, default=None)
    parser.add_argument("--dataset3_bpp_dir", type=Path, default=None)
    parser.add_argument("--dataset3_msa_dir", type=Path, default=None)
    parser.add_argument("--dataset3_ids", default="auto")
    parser.add_argument("--dataset4_name", default="nufolddb_relaxed")
    parser.add_argument("--dataset4_resm_dir", type=Path, default=None)
    parser.add_argument("--dataset4_embedding_dir", type=Path, default=None)
    parser.add_argument("--dataset4_attention_dir", type=Path, default=None)
    parser.add_argument("--dataset4_query_dir", type=Path, default=None)
    parser.add_argument("--dataset4_bpp_dir", type=Path, default=None)
    parser.add_argument("--dataset4_msa_dir", type=Path, default=None)
    parser.add_argument("--dataset4_ids", default="auto")
    parser.add_argument(
        "--test_fasta",
        type=Path,
        default=None,
        help="Optional test-set FASTA. Headers are excluded from training like --val_fasta.",
    )
    return parser


def parse_ids(args: argparse.Namespace) -> list[str]:
    if args.ids.lower() == "auto":
        ids = discover_resm_ids(args.resm_dir, args.embedding_dir, args.attention_dir)
    else:
        ids = [item.strip() for item in args.ids.split(",") if item.strip()]
    if not ids:
        raise ValueError("No RESM samples found")
    return ids


def sanitize_source_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in value.strip())
    if not cleaned:
        raise ValueError("Dataset source name cannot be empty")
    return cleaned


def active_sources(args: argparse.Namespace) -> list[DataSource]:
    sources = [
        DataSource(
            name=sanitize_source_name(args.dataset1_name),
            resm_dir=args.resm_dir,
            embedding_dir=args.embedding_dir,
            attention_dir=args.attention_dir,
            query_dir=args.query_dir,
            bpp_dir=args.bpp_dir,
            msa_dir=args.msa_dir,
            ids=args.ids,
        )
    ]
    dataset2_values = [
        args.dataset2_resm_dir,
        args.dataset2_embedding_dir,
        args.dataset2_attention_dir,
        args.dataset2_query_dir,
        args.dataset2_bpp_dir,
        args.dataset2_msa_dir,
    ]
    if any(value is not None for value in dataset2_values):
        required = {
            "--dataset2_embedding_dir": args.dataset2_embedding_dir,
            "--dataset2_attention_dir": args.dataset2_attention_dir,
            "--dataset2_query_dir": args.dataset2_query_dir,
            "--dataset2_bpp_dir": args.dataset2_bpp_dir,
            "--dataset2_msa_dir": args.dataset2_msa_dir,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(f"Dataset2 is partially configured; missing {', '.join(missing)}")
        sources.append(
            DataSource(
                name=sanitize_source_name(args.dataset2_name),
                resm_dir=args.dataset2_resm_dir,
                embedding_dir=args.dataset2_embedding_dir,
                attention_dir=args.dataset2_attention_dir,
                query_dir=args.dataset2_query_dir,
                bpp_dir=args.dataset2_bpp_dir,
                msa_dir=args.dataset2_msa_dir,
                ids=args.dataset2_ids,
            )
        )
    dataset3_values = [
        args.dataset3_resm_dir,
        args.dataset3_embedding_dir,
        args.dataset3_attention_dir,
        args.dataset3_query_dir,
        args.dataset3_bpp_dir,
        args.dataset3_msa_dir,
    ]
    if any(value is not None for value in dataset3_values):
        required = {
            "--dataset3_embedding_dir": args.dataset3_embedding_dir,
            "--dataset3_attention_dir": args.dataset3_attention_dir,
            "--dataset3_query_dir": args.dataset3_query_dir,
            "--dataset3_bpp_dir": args.dataset3_bpp_dir,
            "--dataset3_msa_dir": args.dataset3_msa_dir,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(f"Dataset3 is partially configured; missing {', '.join(missing)}")
        sources.append(
            DataSource(
                name=sanitize_source_name(args.dataset3_name),
                resm_dir=args.dataset3_resm_dir,
                embedding_dir=args.dataset3_embedding_dir,
                attention_dir=args.dataset3_attention_dir,
                query_dir=args.dataset3_query_dir,
                bpp_dir=args.dataset3_bpp_dir,
                msa_dir=args.dataset3_msa_dir,
                ids=args.dataset3_ids,
            )
        )
    dataset4_values = [
        getattr(args, "dataset4_resm_dir", None),
        getattr(args, "dataset4_embedding_dir", None),
        getattr(args, "dataset4_attention_dir", None),
        getattr(args, "dataset4_query_dir", None),
        getattr(args, "dataset4_bpp_dir", None),
        getattr(args, "dataset4_msa_dir", None),
    ]
    if any(value is not None for value in dataset4_values):
        required = {
            "--dataset4_embedding_dir": getattr(args, "dataset4_embedding_dir", None),
            "--dataset4_attention_dir": getattr(args, "dataset4_attention_dir", None),
            "--dataset4_query_dir": getattr(args, "dataset4_query_dir", None),
            "--dataset4_bpp_dir": getattr(args, "dataset4_bpp_dir", None),
            "--dataset4_msa_dir": getattr(args, "dataset4_msa_dir", None),
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(f"Dataset4 is partially configured; missing {', '.join(missing)}")
        sources.append(
            DataSource(
                name=sanitize_source_name(getattr(args, "dataset4_name", "nufolddb_relaxed")),
                resm_dir=getattr(args, "dataset4_resm_dir", None),
                embedding_dir=getattr(args, "dataset4_embedding_dir", None),
                attention_dir=getattr(args, "dataset4_attention_dir", None),
                query_dir=getattr(args, "dataset4_query_dir", None),
                bpp_dir=getattr(args, "dataset4_bpp_dir", None),
                msa_dir=getattr(args, "dataset4_msa_dir", None),
                ids=getattr(args, "dataset4_ids", "auto"),
            )
        )
    names = [source.name for source in sources]
    if len(names) != len(set(names)):
        raise ValueError(f"Dataset source names must be unique, got {names}")
    return sources


def parse_source_ids(source: DataSource) -> list[str]:
    if source.ids.lower() == "auto":
        ids = discover_resm_ids(source.resm_dir, source.embedding_dir, source.attention_dir)
    else:
        ids = [item.strip() for item in source.ids.split(",") if item.strip()]
    if not ids:
        raise ValueError(f"No RESM samples found for source {source.name}")
    return ids


def make_sample_id(source: DataSource, source_id: str, prefix_sources: bool) -> str:
    return f"{source.name}__{source_id}" if prefix_sources else source_id


def build_multi_source_shape_report(
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[str], dict[str, SourceSample], list[dict[str, object]]]:
    sources = active_sources(args)
    prefix_sources = len(sources) > 1
    rows: list[dict[str, object]] = []
    all_ids: list[str] = []
    samples_by_id: dict[str, SourceSample] = {}
    source_summary: list[dict[str, object]] = []
    for source in sources:
        source_ids = parse_source_ids(source)
        source_rows = build_shape_report(
            source.resm_dir,
            source.query_dir,
            source.bpp_dir,
            source.msa_dir,
            ids=source_ids,
            embedding_dir=source.embedding_dir,
            attention_dir=source.attention_dir,
        )
        source_summary.append({"name": source.name, "discovered": len(source_ids)})
        for row in source_rows:
            source_id = str(row["id"])
            sample_id = make_sample_id(source, source_id, prefix_sources)
            if sample_id in samples_by_id:
                raise ValueError(f"Duplicate sample ID after source prefixing: {sample_id}")
            row["source"] = source.name
            row["source_id"] = source_id
            row["id"] = sample_id
            row["embedding_dir"] = str(source.embedding_dir)
            row["attention_dir"] = str(source.attention_dir)
            row["query_dir"] = str(source.query_dir)
            row["bpp_dir"] = str(source.bpp_dir)
            row["msa_dir"] = str(source.msa_dir)
            rows.append(row)
            all_ids.append(sample_id)
            samples_by_id[sample_id] = SourceSample(source, source_id, sample_id)
    return rows, all_ids, samples_by_id, source_summary


def split_multi_source_train_val_test_ids(
    all_ids: list[str],
    rows: list[dict[str, object]],
    val_fasta: str | Path | None,
    test_fasta: str | Path | None,
    test_input_dir: str | Path | None,
) -> tuple[list[str], list[str], list[str]]:
    val_raw = read_fasta_ids(val_fasta)
    test_raw = read_fasta_ids(test_fasta) | read_test_folder_ids(test_input_dir)
    row_by_id = {str(row["id"]): row for row in rows}

    val_ids: list[str] = []
    test_ids: list[str] = []
    for sample_id in all_ids:
        row = row_by_id[sample_id]
        source_id = str(row.get("source_id", sample_id))
        if sample_id in val_raw or source_id in val_raw:
            val_ids.append(sample_id)
        if sample_id in test_raw or source_id in test_raw:
            test_ids.append(sample_id)

    heldout = set(val_ids) | set(test_ids)
    train_ids = [sample_id for sample_id in all_ids if sample_id not in heldout]
    return train_ids, val_ids, test_ids


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")
    if args.grad_accum < 1:
        raise ValueError("--grad_accum must be >= 1")
    if args.best_checkpoint_mae_ref <= 0:
        raise ValueError("--best_checkpoint_mae_ref must be > 0")
    if args.target_max_seqs < 2:
        raise ValueError("--target_max_seqs must be >= 2")
    if args.triangle_attention_chunk_size < 0:
        raise ValueError("--triangle_attention_chunk_size must be >= 0")
    if args.max_length < 0:
        raise ValueError("--max_length/--lmax must be >= 0")
    if args.head_layers < 1:
        raise ValueError("--head_layers must be >= 1")
    if args.head_hidden_dim < 1:
        raise ValueError("--head_hidden_dim must be >= 1")
    if args.conv_kernel_size < 1 or args.conv_kernel_size % 2 == 0:
        raise ValueError("--conv_kernel_size must be a positive odd integer")
    if args.stats_head_layers < 1:
        raise ValueError("--stats_head_layers must be >= 1")
    if args.stats_head_hidden_dim < 1:
        raise ValueError("--stats_head_hidden_dim must be >= 1")
    stats_flow_sample_steps = getattr(args, "stats_flow_sample_steps", 1)
    stats_flow_condition_dim = getattr(args, "stats_flow_condition_dim", 1)
    stats_head_type = getattr(args, "stats_head_type", "none")
    stats_flow_objective = getattr(args, "stats_flow_objective", "endpoint")

    if stats_flow_sample_steps < 1:
        raise ValueError("--stats_flow_sample_steps must be >= 1")
    if stats_flow_condition_dim < 1:
        raise ValueError("--stats_flow_condition_dim must be >= 1")
    if (
        stats_head_type == "factorized_conditional_flow"
        and stats_flow_objective == "endpoint"
        and stats_flow_sample_steps != 1
    ):
        raise ValueError("--stats_flow_objective endpoint requires --stats_flow_sample_steps 1")
    eval_sample_steps = parse_stats_flow_eval_sample_steps(getattr(args, "stats_flow_eval_sample_steps", ""))
    if (
        stats_head_type == "factorized_conditional_flow"
        and stats_flow_objective == "endpoint"
        and any(steps != 1 for steps in eval_sample_steps)
    ):
        raise ValueError("--stats_flow_objective endpoint requires validation sample steps to be 1")
    if getattr(args, "stats_prior_warmup_epochs", 0) < 0:
        raise ValueError("--stats_prior_warmup_epochs must be >= 0")
    if getattr(args, "stats_scalar_consistency_warmup_epochs", 0) < 0:
        raise ValueError("--stats_scalar_consistency_warmup_epochs must be >= 0")
    for name in (
        "scalar_loss_weight",
        "stats_loss_weight",
        "stats_fi_loss_weight",
        "stats_fij_loss_weight",
        "stats_diag_weight",
        "stats_pair_weight_alpha",
        "stats_pair_weight_top_factor",
        "stats_pair_weight_top_boost",
        "stats_marginal_consistency_weight",
        "stats_marginal_pred_weight",
        "stats_log_odds_weight",
        "stats_covariance_weight",
        "stats_scalar_consistency_weight",
        "stats_prior_warmup_start",
        "stats_prior_loss_weight",
    ):
        if getattr(args, name, 0.0) < 0:
            raise ValueError(f"--{name} must be >= 0")
    if getattr(args, "target_builder", "legacy_mi_apc") == "legacy_mi_apc" and stats_head_type != "none":
        raise ValueError("stats_head_type requires --target_builder adabm_stats")
    active_sources(args)


def parse_stats_flow_eval_sample_steps(value: object) -> list[int]:
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    steps: list[int] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        step = int(item)
        if step < 1:
            raise ValueError("--stats_flow_eval_sample_steps values must be >= 1")
        steps.append(step)
    return steps


def init_distributed(args: argparse.Namespace) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    use_cuda = args.device != "cpu" and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    distributed = world_size > 1
    if distributed and not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            timeout=timedelta(minutes=int(getattr(args, "ddp_timeout_minutes", 120))),
        )

    return DistributedContext(rank, local_rank, world_size, distributed, device, backend)


def cleanup_distributed(ctx: DistributedContext) -> None:
    if ctx.distributed and dist.is_initialized():
        dist.destroy_process_group()


def barrier(ctx: DistributedContext) -> None:
    if ctx.distributed and dist.is_initialized():
        if ctx.backend == "nccl" and ctx.device.type == "cuda":
            dist.barrier(device_ids=[ctx.local_rank])
        else:
            dist.barrier()


def is_rank_zero(rank: int) -> bool:
    return rank == 0


def rank0_call(rank: int, fn, *args, **kwargs):
    if is_rank_zero(rank):
        return fn(*args, **kwargs)
    return None


def unwrap_module(module: torch.nn.Module) -> torch.nn.Module:
    return module.module if hasattr(module, "module") else module


def effective_batch_metadata(args: argparse.Namespace, world_size: int) -> dict[str, int]:
    per_gpu = int(args.batch_size)
    grad_accum = int(args.grad_accum)
    return {
        "per_gpu_batch_size": per_gpu,
        "world_size": int(world_size),
        "grad_accum": grad_accum,
        "effective_global_batch_size": per_gpu * int(world_size) * grad_accum,
    }


def distributed_batch_plan(
    global_batches: list[list[str]],
    rank: int,
    world_size: int,
) -> list[dict[str, Any]]:
    """Assign global batches by rank while keeping equal local loop lengths."""
    if world_size < 1:
        raise ValueError("world_size must be >= 1")
    if rank < 0 or rank >= world_size:
        raise ValueError("rank must satisfy 0 <= rank < world_size")

    groups = math.ceil(len(global_batches) / world_size) if global_batches else 0
    plan: list[dict[str, Any]] = []
    for group_idx in range(groups):
        group_start = group_idx * world_size
        active_count = min(world_size, len(global_batches) - group_start)
        global_batch_idx = group_start + rank
        active = global_batch_idx < len(global_batches)
        plan.append(
            {
                "local_step": group_idx + 1,
                "global_batch_index": global_batch_idx + 1 if active else 0,
                "id_batch": global_batches[global_batch_idx] if active else [],
                "active": active,
                "active_ranks": active_count,
                "loss_scale": float(world_size / active_count) if active_count > 0 and active else 0.0,
            }
        )
    return plan


def verify_distributed_batch_plan(
    local_plan: list[dict[str, Any]],
    ctx: DistributedContext,
    epoch: int,
    grad_accum: int,
) -> None:
    """Fail fast if ranks would execute different local loop/sync schedules."""
    if not ctx.distributed:
        return
    local_summary = {
        "rank": ctx.rank,
        "num_local_steps": len(local_plan),
        "sync_steps": [
            idx + 1
            for idx in range(len(local_plan))
            if (idx % grad_accum) == (grad_accum - 1) or (idx + 1) == len(local_plan)
        ],
        "active_steps": sum(1 for entry in local_plan if bool(entry["active"])),
        "dummy_steps": sum(1 for entry in local_plan if not bool(entry["active"])),
    }
    gathered: list[dict[str, Any] | None] = [None for _ in range(ctx.world_size)]
    dist.all_gather_object(gathered, local_summary)
    num_steps = {int(item["num_local_steps"]) for item in gathered if item is not None}
    sync_schedules = {
        tuple(int(step) for step in item["sync_steps"])
        for item in gathered
        if item is not None
    }
    if len(num_steps) != 1 or len(sync_schedules) != 1:
        raise RuntimeError(
            f"DDP batch-plan mismatch at epoch {epoch}: {gathered}. "
            "All ranks must run the same number of local steps and allreduce steps."
        )
    if ctx.rank == 0:
        print(f"epoch {epoch:03d} DDP batch-plan check: {gathered}", flush=True)

def rank_target_cache_dir(args: argparse.Namespace, rank: int, world_size: int) -> Path:
    base = args.target_cache_dir or (args.out_dir / "target_cache")
    base = Path(base)
    if world_size <= 1 or args.shared_target_cache:
        return base
    rank_offset = int(getattr(args, "target_cache_rank_offset", -1))
    if rank_offset >= 0:
        return base / f"rank_{rank + rank_offset:03d}"
    has_zero_indexed_cache = (base / "rank_000").exists()
    has_one_indexed_cache = (base / "rank_001").exists() and (base / f"rank_{world_size:03d}").exists()
    if has_one_indexed_cache and not has_zero_indexed_cache:
        return base / f"rank_{rank + 1:03d}"
    return base / f"rank_{rank:03d}"


def wrap_ddp(module: torch.nn.Module, args: argparse.Namespace, ctx: DistributedContext) -> torch.nn.Module:
    if not ctx.distributed:
        return module

    kwargs: dict[str, Any] = {"find_unused_parameters": False}
    if ctx.device.type == "cuda":
        kwargs["device_ids"] = [ctx.local_rank]
        kwargs["output_device"] = ctx.local_rank
    if not args.no_ddp_static_graph and "static_graph" in inspect.signature(DistributedDataParallel).parameters:
        kwargs["static_graph"] = True
    return DistributedDataParallel(module, **kwargs)


@contextmanager
def ddp_no_sync_context(modules: list[torch.nn.Module] | tuple[torch.nn.Module, ...], sync: bool):
    if sync:
        yield
        return

    with ExitStack() as stack:
        for module in modules:
            no_sync = getattr(module, "no_sync", None)
            if no_sync is not None:
                stack.enter_context(no_sync())
        yield


def rank0_guarded_call(ctx: DistributedContext, label: str, fn):
    if not ctx.distributed:
        if ctx.rank == 0:
            return fn()
        return None

    result = None
    payload = [{"ok": True, "label": label, "error": "", "traceback": ""}]
    if ctx.rank == 0:
        try:
            result = fn()
        except Exception as exc:  # noqa: BLE001 - broadcast rank0 failures to every rank.
            payload[0] = {
                "ok": False,
                "label": label,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
    dist.broadcast_object_list(payload, src=0)
    status = payload[0]
    if not status["ok"]:
        raise RuntimeError(
            f"Rank 0 failed during {status['label']}: {status['error']}\n{status['traceback']}"
        )
    return result


def reduce_loss_mean(losses: list[float], ctx: DistributedContext) -> float:
    local = torch.tensor(
        [float(sum(losses)), float(len(losses))],
        dtype=torch.float64,
        device=ctx.device,
    )
    if ctx.distributed:
        dist.all_reduce(local, op=dist.ReduceOp.SUM)
    count = float(local[1].item())
    if count <= 0:
        return 0.0
    return float(local[0].item() / count)


def save_unwrapped_checkpoint(
    path: Path,
    encoder: torch.nn.Module,
    head: torch.nn.Module,
    stats_head: torch.nn.Module | None,
    optimizer: torch.optim.Optimizer,
    cfg: SSPairEncoderConfig,
    args: argparse.Namespace,
    epoch: int,
    global_step: int,
    best_metric: float | None,
    train_logs: list[dict[str, object]],
    val_logs: list[dict[str, object]],
) -> None:
    save_checkpoint(
        path,
        unwrap_module(encoder),
        unwrap_module(head),
        optimizer,
        cfg,
        args,
        epoch,
        global_step,
        best_metric,
        train_logs,
        val_logs,
        stats_head=unwrap_module(stats_head) if stats_head is not None else None,
    )


def write_metrics_ddp(
    args: argparse.Namespace,
    cfg: SSPairEncoderConfig,
    split: dict[str, list[str]],
    source_summary: list[dict[str, object]],
    train_logs: list[dict[str, object]],
    val_logs: list[dict[str, object]],
    ctx: DistributedContext,
    best_metric: float | None,
) -> None:
    result = {
        "config": asdict(cfg),
        "args": {key: pathlike_to_str(value) for key, value in vars(args).items()},
        "best_checkpoint": {
            "best_monitor": args.best_checkpoint_metric,
            "best_score": best_metric,
            "higher_is_better": True,
            "mae_ref": args.best_checkpoint_mae_ref,
        },
        "ddp": {
            **effective_batch_metadata(args, ctx.world_size),
            "backend": ctx.backend,
            "rank0_only_outputs": True,
            "batch_size_semantics": "per_gpu",
        },
        "sources": source_summary,
        "split": split,
        "train_logs": train_logs,
        "val_logs": val_logs,
    }
    with (args.out_dir / "train_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)


def prepare_inputs(
    args: argparse.Namespace,
    ctx: DistributedContext,
) -> tuple[
    SSPairEncoderConfig,
    list[dict[str, object]],
    list[str],
    list[str],
    dict[str, list[str]],
    dict[str, SourceSample],
    list[dict[str, object]],
]:
    if ctx.rank == 0:
        args.out_dir.mkdir(parents=True, exist_ok=True)
    barrier(ctx)

    rows, all_ids, samples_by_id, source_summary = build_multi_source_shape_report(args)
    if ctx.rank == 0:
        write_shape_report(rows, args.out_dir)
        print(f"Active data sources: {source_summary}", flush=True)

    if args.skip_bad_audit:
        bad = [row for row in rows if row["status"] != "OK"]
        if bad and ctx.rank == 0:
            preview = "; ".join(f"{row['id']}={row['status']}" for row in bad[:5])
            print(f"Skipping {len(bad)} samples that failed audit: {preview}", flush=True)
        rows = [row for row in rows if row["status"] == "OK"]
        ok_ids = {str(row["id"]) for row in rows}
        all_ids = [rna_id for rna_id in all_ids if rna_id in ok_ids]
        samples_by_id = {rna_id: samples_by_id[rna_id] for rna_id in all_ids}
        if not all_ids:
            raise ValueError("All samples failed audit")
        if bad and ctx.rank == 0:
            write_shape_report(rows, args.out_dir)
    else:
        assert_audit_ok(rows)

    rows, all_ids, removed_long = filter_rows_by_max_length(rows, all_ids, args.max_length)
    if removed_long:
        if not all_ids:
            raise ValueError("All samples were removed by --max_length/--lmax")
        samples_by_id = {rna_id: samples_by_id[rna_id] for rna_id in all_ids}
        if ctx.rank == 0:
            preview = ",".join(removed_long[:8])
            print(
                f"Length filter L <= {args.max_length}: "
                f"removed={len(removed_long)} kept={len(all_ids)} preview={preview}",
                flush=True,
            )
            write_shape_report(rows, args.out_dir)

    train_ids, val_ids, test_ids = split_multi_source_train_val_test_ids(
        all_ids,
        rows,
        args.val_fasta,
        args.test_fasta,
        args.test_input_dir,
    )
    split = {"train": train_ids, "val": val_ids, "test": test_ids}
    if ctx.rank == 0:
        with (args.out_dir / "split.json").open("w", encoding="utf-8") as handle:
            json.dump(split, handle, indent=2)
        print(
            f"Shape audit OK for {len(rows)} samples. "
            f"Split: train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}",
            flush=True,
        )

    cfg = build_encoder_config(args, rows)
    barrier(ctx)
    return cfg, rows, train_ids, val_ids, split, samples_by_id, source_summary


def run_training_ddp(
    args: argparse.Namespace,
    cfg: SSPairEncoderConfig,
    rows: list[dict[str, object]],
    train_ids: list[str],
    val_ids: list[str],
    split: dict[str, list[str]],
    samples_by_id: dict[str, SourceSample],
    source_summary: list[dict[str, object]],
    ctx: DistributedContext,
) -> None:
    if not train_ids:
        raise ValueError("No training IDs remain after val/test exclusion")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if ctx.device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    args.target_cache_dir = rank_target_cache_dir(args, ctx.rank, ctx.world_size)
    print(f"rank {ctx.rank} target_cache_dir={args.target_cache_dir}", flush=True)
    length_by_id = {str(row["id"]): int(row["L_embedding"]) for row in rows if row["status"] == "OK"}
    enable_stats_target = args.stats_head_type != "none"
    include_stats_target_map = args.stats_head_type not in {
        "factorized_conditional_flow",
        "factorized_residual_mlp",
    }
    train_dataset = MultiSourceRESMDataset(
        samples_by_id,
        ids=train_ids,
        target_max_seqs=args.target_max_seqs,
        target_cache_dir=args.target_cache_dir,
        target_builder=args.target_builder,
        enable_stats_target=enable_stats_target,
        include_stats_target_map=include_stats_target_map,
        target_clustering_seqid=args.target_clustering_seqid,
        target_remove_duplicates=not args.target_keep_duplicates,
        target_no_reweighting=args.target_no_reweighting,
        target_pseudocount=args.target_pseudocount,
        cache_targets=args.memory_target_cache,
    )
    val_dataset = None
    if ctx.rank == 0:
        val_dataset = MultiSourceRESMDataset(
            samples_by_id,
            ids=val_ids,
            target_max_seqs=args.target_max_seqs,
            target_cache_dir=args.target_cache_dir,
            target_builder=args.target_builder,
            enable_stats_target=enable_stats_target,
            include_stats_target_map=include_stats_target_map,
            target_clustering_seqid=args.target_clustering_seqid,
            target_remove_duplicates=not args.target_keep_duplicates,
            target_no_reweighting=args.target_no_reweighting,
            target_pseudocount=args.target_pseudocount,
            cache_targets=False,
        )
    train_index = {rna_id: idx for idx, rna_id in enumerate(train_dataset.ids)}

    encoder_raw, head_raw = build_models(args, cfg, ctx.device)
    stats_head_raw = build_stats_head(args, cfg, ctx.device)
    params = list(encoder_raw.parameters()) + list(head_raw.parameters())
    if stats_head_raw is not None:
        params += list(stats_head_raw.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    start_epoch = 1
    global_step = 0
    best_metric: float | None = None
    train_logs: list[dict[str, object]] = []
    val_logs: list[dict[str, object]] = []
    if args.resume is not None:
        start_epoch, global_step, best_metric, train_logs, val_logs = load_checkpoint(
            args.resume,
            encoder_raw,
            head_raw,
            optimizer,
            ctx.device,
            stats_head=stats_head_raw,
        )
        if ctx.rank == 0:
            print(f"Resumed {args.resume} at epoch={start_epoch} global_step={global_step}", flush=True)

    encoder = wrap_ddp(encoder_raw, args, ctx)
    head = wrap_ddp(head_raw, args, ctx)
    stats_head = wrap_ddp(stats_head_raw, args, ctx) if stats_head_raw is not None else None
    dummy_id = train_ids[0]
    stop_training = False
    if ctx.rank == 0:
        print(
            "DDP batch semantics: "
            f"per_gpu_batch_size={args.batch_size} world_size={ctx.world_size} "
            f"grad_accum={args.grad_accum} "
            f"effective_global_batch_size={args.batch_size * ctx.world_size * args.grad_accum}",
            flush=True,
        )

    for epoch in range(start_epoch, args.epochs + 1):
        encoder.train()
        head.train()
        if stats_head is not None:
            stats_head.train()
        global_batches = make_epoch_batches(
            train_ids,
            length_by_id,
            args.batch_size,
            args.seed,
            epoch,
            shuffle=True,
        )
        local_plan = distributed_batch_plan(global_batches, ctx.rank, ctx.world_size)
        verify_distributed_batch_plan(local_plan, ctx, epoch, args.grad_accum)
        if ctx.rank == 0:
            print(
                f"epoch {epoch:03d}: global_batches={len(global_batches)} "
                f"rank_local_steps={len(local_plan)}",
                flush=True,
            )
        optimizer.zero_grad(set_to_none=True)
        epoch_losses: list[float] = []
        accum = 0

        for local_step, entry in enumerate(local_plan, start=1):
            active = bool(entry["active"])
            id_batch = list(entry["id_batch"]) if active else [dummy_id]
            batch_start_time = time.perf_counter()
            if args.ddp_log_batch_timing:
                print(
                    f"rank {ctx.rank} epoch {epoch:03d} local_batch {local_step:05d} "
                    f"global_batch {int(entry['global_batch_index']):05d} active={active} "
                    f"pre_load ids={','.join(id_batch)}",
                    flush=True,
                )
            load_start_time = time.perf_counter()
            batch = move_batch_to_device(dataset_batch(train_dataset, id_batch, train_index), ctx.device)
            load_seconds = time.perf_counter() - load_start_time
            print(
                f"rank {ctx.rank} epoch {epoch:03d} local_batch {local_step:05d} "
                f"global_batch {int(entry['global_batch_index']):05d} active={active} "
                f"Lmax={int(batch['length'].max().item())} load_s={load_seconds:.3f} ids={','.join(batch['id'])}",
                flush=True,
            )
            forward_start_time = time.perf_counter()
            loss, loss_parts = compute_train_loss(
                args,
                encoder,
                head,
                batch,
                stats_head=stats_head,
                epoch=epoch,
            )
            forward_seconds = time.perf_counter() - forward_start_time
            if not torch.isfinite(loss):
                raise ValueError(
                    f"Non-finite loss at rank {ctx.rank} epoch {epoch} local batch {local_step}: "
                    f"{loss.item()}"
                )
            scaled_loss = loss * float(entry["loss_scale"]) if active else loss * 0.0
            should_step = (accum + 1) >= args.grad_accum or local_step == len(local_plan)
            sync_modules = (encoder, head) if stats_head is None else (encoder, head, stats_head)
            backward_start_time = time.perf_counter()
            with ddp_no_sync_context(sync_modules, sync=should_step):
                (scaled_loss / args.grad_accum).backward()
            backward_seconds = time.perf_counter() - backward_start_time
            batch_seconds = time.perf_counter() - batch_start_time
            if (
                args.ddp_log_batch_timing
                or batch_seconds >= args.ddp_slow_batch_seconds
                or load_seconds >= args.ddp_slow_batch_seconds
                or forward_seconds >= args.ddp_slow_batch_seconds
                or backward_seconds >= args.ddp_slow_batch_seconds
            ):
                print(
                    f"rank {ctx.rank} epoch {epoch:03d} local_batch {local_step:05d} "
                    f"timing load_s={load_seconds:.3f} forward_s={forward_seconds:.3f} "
                    f"backward_s={backward_seconds:.3f} total_s={batch_seconds:.3f} "
                    f"sync_step={should_step}",
                    flush=True,
                )
            accum += 1
            if active:
                epoch_losses.append(loss_parts["loss"])

            if should_step:
                if not finite_gradients(params):
                    raise ValueError(f"Non-finite gradient at rank {ctx.rank} epoch {epoch} batch {local_step}")
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
                norm = grad_norm(params)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                accum = 0
                if ctx.rank == 0:
                    log_row = {
                        "epoch": epoch,
                        "global_step": global_step,
                        "rank0_local_batch": local_step,
                        "rank0_global_batch": int(entry["global_batch_index"]),
                        "rank0_active": active,
                        "rank0_ids": list(batch["id"]) if active else [],
                        "rank0_length_max": int(batch["length"].max().item()) if active else 0,
                        "grad_norm": norm,
                        **effective_batch_metadata(args, ctx.world_size),
                        **loss_parts,
                    }
                    train_logs.append(log_row)
                    print(
                        f"epoch {epoch:03d} step {global_step:05d} "
                        f"rank0_active={active} loss={loss_parts['loss']:.6f} grad_norm={norm:.4f}",
                        flush=True,
                    )
                if args.steps and global_step >= args.steps:
                    stop_training = True
                    break

        train_mean = reduce_loss_mean(epoch_losses, ctx)

        def run_rank0_validation() -> dict[str, float]:
            if not (
                ctx.rank == 0
                and val_ids
                and val_dataset is not None
                and args.validate_every > 0
                and epoch % args.validate_every == 0
            ):
                return {}
            summary = validate(
                args,
                unwrap_module(encoder),
                unwrap_module(head),
                val_dataset,
                length_by_id,
                ctx.device,
                epoch,
                stats_head=unwrap_module(stats_head) if stats_head is not None else None,
            )
            _, best_fields = best_checkpoint_score(args, summary)
            summary = {**summary, **best_fields}
            val_log = {"epoch": epoch, "global_step": global_step, **summary}
            val_logs.append(val_log)
            print(
                f"validation epoch {epoch:03d}: "
                f"pearson={summary.get('pearson', 0.0):.4f} "
                f"mae={summary.get('mae', 0.0):.4f} "
                f"topL={summary.get('top_l_overlap', 0.0):.4f} "
                f"best_score={summary.get('best_score', 0.0):.4f}",
                flush=True,
            )
            return summary

        val_summary = rank0_guarded_call(ctx, f"validation epoch {epoch:03d}", run_rank0_validation) or {}

        def write_rank0_epoch_outputs() -> None:
            nonlocal best_metric
            if ctx.rank != 0:
                return
            monitor = validation_best_monitor(args, val_summary)
            if monitor is not None and (best_metric is None or monitor > best_metric):
                best_metric = monitor
                save_unwrapped_checkpoint(
                    args.out_dir / "best.pt",
                    encoder,
                    head,
                    stats_head,
                    optimizer,
                    cfg,
                    args,
                    epoch,
                    global_step,
                    best_metric,
                    train_logs,
                    val_logs,
                )
            save_unwrapped_checkpoint(
                args.out_dir / "last.pt",
                encoder,
                head,
                stats_head,
                optimizer,
                cfg,
                args,
                epoch,
                global_step,
                best_metric,
                train_logs,
                val_logs,
            )
            write_metrics_ddp(args, cfg, split, source_summary, train_logs, val_logs, ctx, best_metric)

        rank0_guarded_call(ctx, f"checkpoint epoch {epoch:03d}", write_rank0_epoch_outputs)

        if stop_training:
            break

    if ctx.rank == 0:
        print(f"Training complete. Metrics written to {args.out_dir / 'train_metrics.json'}", flush=True)


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    ctx = init_distributed(args)
    try:
        if ctx.rank == 0:
            print(
                f"Distributed context: rank={ctx.rank} local_rank={ctx.local_rank} "
                f"world_size={ctx.world_size} backend={ctx.backend} device={ctx.device}",
                flush=True,
            )
        cfg, rows, train_ids, val_ids, split, samples_by_id, source_summary = prepare_inputs(args, ctx)
        if args.audit_only:
            if ctx.rank == 0:
                print(f"Reports written to {args.out_dir}", flush=True)
            return
        run_training_ddp(args, cfg, rows, train_ids, val_ids, split, samples_by_id, source_summary, ctx)
    finally:
        cleanup_distributed(ctx)


if __name__ == "__main__":
    main()




