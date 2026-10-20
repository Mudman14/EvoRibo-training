"""Direct RESM/BPP trunk training for normalized coevolution map prediction."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .config import SSPairEncoderConfig
from .encoder import EvoNASSPairEncoder
from .flow import FactorizedConditionalFlowStatsHead, StatsFlowMatchingHead
from .heads import (
    FactorizedStatsHead,
    SymmetricConv2DRefinementHead,
    SymmetricResidualMLPCoevoHead,
    SymmetricResidualMLPStatsHead,
    SymmetricUpperTriangleCoevoHead,
)
from .resm_dataset import (
    RESMTestDataset,
    build_shape_report,
    discover_resm_ids,
    split_train_val_test_ids,
    write_shape_report,
)
from .stat_targets import (
    fi_mse_loss,
    fij_covariance_loss,
    fij_kl_loss,
    fij_log_odds_loss,
    fij_log_odds_loss_fp32,
    fij_marginal_pred_loss,
    fij_marginal_target_loss,
    fij_scalar_consistency_loss,
    stats_covariance_loss,
    stats_kl_loss,
    stats_log_odds_loss,
    stats_marginal_consistency_loss,
    stats_pair_weights_from_scalar_target,
    stats_probs_to_mi_apc_torch,
    stats_scalar_consistency_loss,
)
from .train_resm_smoke import (
    assert_audit_ok,
    autocast_context,
    choose_device,
    dataset_batch,
    filter_rows_by_max_length,
    finite_gradients,
    grad_norm,
    infer_feature_dims,
    make_epoch_batches,
    map_metrics,
    move_batch_to_device,
    pathlike_to_str,
    serializable_args,
)


def build_parser() -> argparse.ArgumentParser:
    default_out = Path(__file__).resolve().parent / "smoke_runs" / "direct_trunk"
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
        default=200,
        help="Drop samples with RESM/query length above this value. 0 disables filtering.",
    )

    parser.add_argument(
        "--readout_head",
        choices=("simple_mlp", "residual_mlp", "conv2d_refine"),
        default="residual_mlp",
    )
    parser.add_argument("--head_hidden_dim", type=int, default=256)
    parser.add_argument("--head_layers", type=int, default=3)
    parser.add_argument("--head_dropout", type=float, default=0.1)
    parser.add_argument("--conv_kernel_size", type=int, default=3)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps", type=int, default=0, help="Optional optimizer-step limit for smoke runs")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--validate_every", type=int, default=10)
    parser.add_argument(
        "--best_checkpoint_metric",
        choices=("composite_topl", "mae", "joint", "stats_kl", "stats_inverse"),
        default="stats_inverse",
        help="Validation metric used to select best.pt. composite_topl is higher-is-better; mae uses -MAE.",
    )
    parser.add_argument(
        "--best_checkpoint_mae_ref",
        type=float,
        default=0.05,
        help="MAE reference scale for composite_topl: mae_quality = 1 - min(mae / ref, 1).",
    )
    parser.add_argument("--target_max_seqs", type=int, default=200)
    parser.add_argument(
        "--target_builder",
        choices=("legacy_mi_apc", "adabm_stats"),
        default="adabm_stats",
        help="Scalar target source. adabm_stats also provides optional LxLx25 stats targets.",
    )
    parser.add_argument("--target_clustering_seqid", type=float, default=0.8)
    parser.add_argument("--target_no_reweighting", action="store_true")
    parser.add_argument("--target_keep_duplicates", action="store_true")
    parser.add_argument(
        "--target_pseudocount",
        type=float,
        default=None,
        help="Override adabm pseudocount. Default None uses 1/Meff.",
    )
    parser.add_argument(
        "--stats_head_type",
        choices=("none", "residual_mlp", "flow", "factorized_residual_mlp", "factorized_conditional_flow"),
        default="factorized_conditional_flow",
    )
    parser.add_argument("--stats_head_hidden_dim", type=int, default=256)
    parser.add_argument("--stats_head_layers", type=int, default=3)
    parser.add_argument("--stats_head_dropout", type=float, default=0.1)
    parser.add_argument("--scalar_loss_weight", type=float, default=1.0)
    parser.add_argument("--stats_loss_weight", type=float, default=1.0)
    parser.add_argument("--stats_fi_loss_weight", type=float, default=1.0)
    parser.add_argument("--stats_fij_loss_weight", type=float, default=1.0)
    parser.add_argument("--stats_diag_weight", type=float, default=0.2)
    parser.add_argument("--stats_pair_weight_alpha", type=float, default=2.0)
    parser.add_argument("--stats_pair_weight_top_factor", type=float, default=2.0)
    parser.add_argument("--stats_pair_weight_top_boost", type=float, default=3.0)
    parser.add_argument("--stats_marginal_consistency_weight", type=float, default=0.2)
    parser.add_argument("--stats_marginal_pred_weight", type=float, default=0.0)
    parser.add_argument("--stats_log_odds_weight", type=float, default=0.2)
    parser.add_argument("--stats_log_odds_clip", type=float, default=6.0)
    parser.add_argument("--stats_covariance_weight", type=float, default=0.5)
    parser.add_argument("--stats_scalar_consistency_weight", type=float, default=0.0)
    parser.add_argument(
        "--stats_scalar_consistency_warmup_epochs",
        type=int,
        default=0,
        help="Use zero scalar-consistency loss weight through this epoch, then enable the configured weight.",
    )
    parser.add_argument("--stats_prior_loss_weight", type=float, default=0.2)
    parser.add_argument("--stats_prior_warmup_start", type=float, default=0.05)
    parser.add_argument("--stats_prior_warmup_epochs", type=int, default=10)
    parser.add_argument("--stats_flow_objective", choices=("endpoint", "velocity"), default="velocity")
    parser.add_argument("--stats_flow_x0_noise_scale", type=float, default=0.05)
    parser.add_argument("--stats_flow_sample_steps", type=int, default=4)
    parser.add_argument(
        "--stats_flow_eval_sample_steps",
        default="",
        help="Optional comma-separated validation-only sample-step sweep, for example '1,2,4,8'.",
    )
    parser.add_argument("--stats_flow_condition_dim", type=int, default=64)
    parser.add_argument("--stats_fij_velocity_weight", type=float, default=1.0)
    parser.add_argument("--stats_fij_endpoint_weight", type=float, default=1.0)
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
    parser.add_argument("--d_s", type=int, default=512)
    parser.add_argument("--d_z", type=int, default=256)
    parser.add_argument("--num_blocks", type=int, default=12)
    parser.add_argument("--num_recycles", type=int, default=1)
    parser.add_argument("--sequence_head_width", type=int, default=32)
    parser.add_argument("--pairwise_head_width", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    return parser


def parse_ids(args: argparse.Namespace) -> list[str]:
    if args.ids.lower() == "auto":
        ids = discover_resm_ids(args.resm_dir, args.embedding_dir, args.attention_dir)
    else:
        ids = [item.strip() for item in args.ids.split(",") if item.strip()]
    if not ids:
        raise ValueError("No RESM samples found")
    return ids


def upper_pair_mask(pair_mask: torch.Tensor) -> torch.Tensor:
    if pair_mask.ndim != 3 or pair_mask.shape[1] != pair_mask.shape[2]:
        raise ValueError(f"pair_mask must have shape [B, L, L], got {tuple(pair_mask.shape)}")
    B, L, _ = pair_mask.shape
    upper = torch.triu(torch.ones(L, L, device=pair_mask.device, dtype=torch.bool), diagonal=1)
    sym_mask = pair_mask.bool() & pair_mask.bool().transpose(-1, -2)
    return upper[None].expand(B, L, L) & sym_mask


def masked_upper_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    pair_mask: torch.Tensor,
) -> torch.Tensor:
    if pred.ndim == 3:
        pred = pred[:, None]
    if target.ndim == 3:
        target = target[:, None]
    if pred.shape != target.shape:
        raise ValueError(f"pred/target shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
    valid = upper_pair_mask(pair_mask)[:, None]
    if not valid.any():
        return (pred - target).sum() * 0.0
    return F.mse_loss(pred[valid], target[valid])


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


def build_head(args: argparse.Namespace, cfg: SSPairEncoderConfig, device: torch.device) -> torch.nn.Module:
    hidden_dim = args.head_hidden_dim or cfg.d_z
    if args.readout_head == "simple_mlp":
        head = SymmetricUpperTriangleCoevoHead(d_z=cfg.d_z, hidden_dim=hidden_dim)
    elif args.readout_head == "residual_mlp":
        head = SymmetricResidualMLPCoevoHead(
            d_z=cfg.d_z,
            hidden_dim=hidden_dim,
            num_layers=args.head_layers,
            dropout=args.head_dropout,
        )
    elif args.readout_head == "conv2d_refine":
        head = SymmetricConv2DRefinementHead(
            d_z=cfg.d_z,
            hidden_dim=hidden_dim,
            num_layers=args.head_layers,
            kernel_size=args.conv_kernel_size,
            dropout=args.head_dropout,
        )
    else:
        raise ValueError(f"unknown readout head: {args.readout_head}")
    return head.to(device)


def build_models(args: argparse.Namespace, cfg: SSPairEncoderConfig, device: torch.device):
    encoder = EvoNASSPairEncoder(cfg).to(device)
    head = build_head(args, cfg, device)
    return encoder, head


def build_stats_head(
    args: argparse.Namespace,
    cfg: SSPairEncoderConfig,
    device: torch.device,
) -> torch.nn.Module | None:
    if args.stats_head_type == "none":
        return None
    hidden_dim = args.stats_head_hidden_dim or cfg.d_z
    if args.stats_head_type == "factorized_residual_mlp":
        return FactorizedStatsHead(
            d_s=cfg.d_s,
            d_z=cfg.d_z,
            hidden_dim=hidden_dim,
            num_layers=args.stats_head_layers,
            dropout=args.stats_head_dropout,
        ).to(device)
    if args.stats_head_type == "factorized_conditional_flow":
        return FactorizedConditionalFlowStatsHead(
            d_s=cfg.d_s,
            d_z=cfg.d_z,
            fi_hidden_dim=hidden_dim,
            fi_layers=max(2, args.stats_head_layers),
            prior_hidden_dim=hidden_dim,
            condition_dim=args.stats_flow_condition_dim,
            unet_base_channels=hidden_dim,
            unet_levels=3,
            time_dim=max(128, hidden_dim),
            objective=args.stats_flow_objective,
            dropout=args.stats_head_dropout,
        ).to(device)
    if args.stats_head_type == "residual_mlp":
        return SymmetricResidualMLPStatsHead(
            d_z=cfg.d_z,
            hidden_dim=hidden_dim,
            num_layers=args.stats_head_layers,
            dropout=args.stats_head_dropout,
        ).to(device)
    if args.stats_head_type == "flow":
        return StatsFlowMatchingHead(
            d_z=cfg.d_z,
            prior_hidden_dim=hidden_dim,
            unet_base_channels=hidden_dim,
            unet_levels=3,
            time_dim=max(128, hidden_dim),
            objective=args.stats_flow_objective,
        ).to(device)
    raise ValueError(f"unknown stats head type: {args.stats_head_type}")


def stats_prior_weight_for_epoch(args: argparse.Namespace, epoch: int | None) -> float:
    final_weight = float(getattr(args, "stats_prior_loss_weight", 0.0))
    start_weight = float(getattr(args, "stats_prior_warmup_start", final_weight))
    warmup_epochs = int(getattr(args, "stats_prior_warmup_epochs", 0))
    if epoch is None or warmup_epochs <= 0:
        return final_weight
    progress = min(max(float(epoch) / float(warmup_epochs), 0.0), 1.0)
    return start_weight + progress * (final_weight - start_weight)


def stats_scalar_consistency_weight_for_epoch(args: argparse.Namespace, epoch: int | None) -> float:
    final_weight = float(getattr(args, "stats_scalar_consistency_weight", 0.0))
    warmup_epochs = int(getattr(args, "stats_scalar_consistency_warmup_epochs", 0))
    if final_weight <= 0.0:
        return 0.0
    if epoch is not None and warmup_epochs > 0 and epoch <= warmup_epochs:
        return 0.0
    return final_weight


def parse_stats_flow_eval_sample_steps(value: object) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        raw_parts = value
    else:
        raw = str(value).strip()
        if not raw:
            return []
        raw_parts = raw.split(",")
    steps: list[int] = []
    seen: set[int] = set()
    for part in raw_parts:
        text = str(part).strip()
        if not text:
            continue
        step = int(text)
        if step < 1:
            raise ValueError("--stats_flow_eval_sample_steps values must be >= 1")
        if step not in seen:
            seen.add(step)
            steps.append(step)
    return steps


def compute_train_loss(
    args: argparse.Namespace,
    encoder: EvoNASSPairEncoder,
    head: torch.nn.Module,
    batch: dict[str, object],
    stats_head: torch.nn.Module | None = None,
    epoch: int | None = None,
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
        pred_upper = head(out["pair"], out["pair_mask"], return_full=False)
        scalar_loss = masked_upper_mse(pred_upper[:, None], batch["target"], out["pair_mask"])
        loss = float(getattr(args, "scalar_loss_weight", 1.0)) * scalar_loss
        parts_t: dict[str, torch.Tensor] = {"scalar_loss": scalar_loss}

        if stats_head is not None:
            stats_pair_weights = stats_pair_weights_from_scalar_target(
                batch["target"],
                out["pair_mask"],
                alpha=getattr(args, "stats_pair_weight_alpha", 2.0),
                top_factor=getattr(args, "stats_pair_weight_top_factor", 2.0),
                top_boost=getattr(args, "stats_pair_weight_top_boost", 2.0),
            )
            stats_head_type = getattr(args, "stats_head_type", "residual_mlp")
            if stats_head_type == "factorized_conditional_flow":
                if "fi_target" not in batch or "fij_target" not in batch:
                    raise ValueError("factorized conditional flow stats head requires fi_target and fij_target")
                stats_out = stats_head(
                    out["single"],
                    out["pair"],
                    mask=batch["mask"],
                    pair_mask=out["pair_mask"],
                    fij_target=batch["fij_target"],
                    objective=getattr(args, "stats_flow_objective", "velocity"),
                    x0_noise_scale=getattr(args, "stats_flow_x0_noise_scale", 0.05),
                    pair_weights=stats_pair_weights,
                )
                fi_logits = stats_out["fi_logits"]
                fij_logits = stats_out["endpoint_logits"]
                fi_loss = fi_mse_loss(fi_logits, batch["fi_target"], batch["mask"])
                velocity_loss = stats_out["velocity_loss"]
                endpoint_loss = stats_out["endpoint_kl_loss"]
                prior_loss = stats_out["prior_kl_loss"]
                marginal_target = fij_marginal_target_loss(
                    fij_logits,
                    batch["fi_target"],
                    out["pair_mask"],
                    pair_weights=stats_pair_weights,
                )
                log_odds = fij_log_odds_loss_fp32(
                    fij_logits,
                    batch["fij_target"],
                    batch["fi_target"],
                    out["pair_mask"],
                    pair_weights=stats_pair_weights,
                    clip_value=getattr(args, "stats_log_odds_clip", 6.0),
                )
                covariance = fij_covariance_loss(
                    fij_logits,
                    batch["fij_target"],
                    batch["fi_target"],
                    out["pair_mask"],
                    pair_weights=stats_pair_weights,
                )
                prior_weight = stats_prior_weight_for_epoch(args, epoch)
                loss = loss + float(getattr(args, "stats_fi_loss_weight", 1.0)) * fi_loss
                loss = loss + float(getattr(args, "stats_fij_velocity_weight", 1.0)) * velocity_loss
                loss = loss + float(getattr(args, "stats_fij_endpoint_weight", 1.0)) * endpoint_loss
                loss = loss + prior_weight * prior_loss
                loss = loss + float(getattr(args, "stats_marginal_consistency_weight", 0.2)) * marginal_target
                loss = loss + float(getattr(args, "stats_log_odds_weight", 0.2)) * log_odds
                loss = loss + float(getattr(args, "stats_covariance_weight", 0.5)) * covariance
                parts_t["fi_mse_loss"] = fi_loss
                parts_t["fij_velocity_loss"] = velocity_loss
                parts_t["fij_endpoint_kl_loss"] = endpoint_loss
                parts_t["fij_prior_kl_loss"] = prior_loss
                parts_t["fij_prior_weight"] = torch.as_tensor(prior_weight, device=loss.device)
                parts_t["fij_marginal_target_loss"] = marginal_target
                parts_t["fij_log_odds_loss"] = log_odds
                parts_t["fij_covariance_loss"] = covariance
                parts_t["stats_loss"] = endpoint_loss
                parts_t["stats_kl_loss"] = endpoint_loss

                marginal_pred_weight = float(getattr(args, "stats_marginal_pred_weight", 0.0))
                if marginal_pred_weight > 0.0:
                    marginal_pred = fij_marginal_pred_loss(
                        fij_logits,
                        fi_logits,
                        out["pair_mask"],
                        pair_weights=stats_pair_weights,
                        detach_fi=True,
                    )
                    loss = loss + marginal_pred_weight * marginal_pred
                    parts_t["fij_marginal_pred_loss"] = marginal_pred

                scalar_consistency_weight = stats_scalar_consistency_weight_for_epoch(args, epoch)
                parts_t["stats_scalar_consistency_weight_effective"] = torch.as_tensor(
                    scalar_consistency_weight,
                    device=loss.device,
                )
                if scalar_consistency_weight > 0.0:
                    consistency = fij_scalar_consistency_loss(
                        fij_logits,
                        batch["target"],
                        out["pair_mask"],
                    )
                    loss = loss + scalar_consistency_weight * consistency
                    parts_t["fij_scalar_consistency_loss"] = consistency
            elif stats_head_type == "factorized_residual_mlp":
                if "fi_target" not in batch or "fij_target" not in batch:
                    raise ValueError("factorized stats head requires fi_target and fij_target")
                stats_out = stats_head(
                    out["single"],
                    out["pair"],
                    mask=batch["mask"],
                    pair_mask=out["pair_mask"],
                )
                fi_logits = stats_out["fi_logits"]
                fij_logits = stats_out["fij_logits"]
                fi_loss = fi_mse_loss(fi_logits, batch["fi_target"], batch["mask"])
                fij_loss = fij_kl_loss(
                    fij_logits,
                    batch["fij_target"],
                    out["pair_mask"],
                    pair_weights=stats_pair_weights,
                )
                marginal_target = fij_marginal_target_loss(
                    fij_logits,
                    batch["fi_target"],
                    out["pair_mask"],
                    pair_weights=stats_pair_weights,
                )
                marginal_pred = fij_marginal_pred_loss(
                    fij_logits,
                    fi_logits,
                    out["pair_mask"],
                    pair_weights=stats_pair_weights,
                )
                log_odds = fij_log_odds_loss(
                    fij_logits,
                    batch["fij_target"],
                    batch["fi_target"],
                    out["pair_mask"],
                    pair_weights=stats_pair_weights,
                    clip_value=getattr(args, "stats_log_odds_clip", 6.0),
                )
                covariance = fij_covariance_loss(
                    fij_logits,
                    batch["fij_target"],
                    batch["fi_target"],
                    out["pair_mask"],
                    pair_weights=stats_pair_weights,
                )
                loss = loss + float(getattr(args, "stats_fi_loss_weight", 1.0)) * fi_loss
                loss = loss + float(getattr(args, "stats_fij_loss_weight", 1.0)) * fij_loss
                loss = loss + float(getattr(args, "stats_marginal_consistency_weight", 0.2)) * marginal_target
                loss = loss + float(getattr(args, "stats_marginal_pred_weight", 0.1)) * marginal_pred
                loss = loss + float(getattr(args, "stats_log_odds_weight", 0.1)) * log_odds
                loss = loss + float(getattr(args, "stats_covariance_weight", 0.5)) * covariance
                scalar_consistency_weight = stats_scalar_consistency_weight_for_epoch(args, epoch)
                parts_t["stats_scalar_consistency_weight_effective"] = torch.as_tensor(
                    scalar_consistency_weight,
                    device=loss.device,
                )
                if scalar_consistency_weight > 0.0:
                    consistency = fij_scalar_consistency_loss(
                        fij_logits,
                        batch["target"],
                        out["pair_mask"],
                    )
                    loss = loss + scalar_consistency_weight * consistency
                    parts_t["fij_scalar_consistency_loss"] = consistency
                parts_t["fi_mse_loss"] = fi_loss
                parts_t["fij_loss"] = fij_loss
                parts_t["fij_kl_loss"] = fij_loss
                parts_t["fij_marginal_target_loss"] = marginal_target
                parts_t["fij_marginal_pred_loss"] = marginal_pred
                parts_t["fij_log_odds_loss"] = log_odds
                parts_t["fij_covariance_loss"] = covariance
                parts_t["stats_loss"] = fij_loss
                parts_t["stats_kl_loss"] = fij_loss
            else:
                if "stats_target" not in batch:
                    raise ValueError("stats_head is enabled but batch does not contain stats_target")
                stats_target = batch["stats_target"]
                if stats_head_type == "flow":
                    flow_out = stats_head(
                        out["pair"],
                        stats_target,
                        out["pair_mask"],
                        objective=getattr(args, "stats_flow_objective", "endpoint"),
                        x0_noise_scale=getattr(args, "stats_flow_x0_noise_scale", 0.05),
                        pair_weights=stats_pair_weights,
                        diag_weight=getattr(args, "stats_diag_weight", 0.2),
                    )
                    stats_logits = flow_out["pred_endpoint"]
                    stats_prior_loss = flow_out["prior_kl_loss"]
                    loss = loss + float(getattr(args, "stats_prior_loss_weight", 1.0)) * stats_prior_loss
                    parts_t["stats_flow_loss"] = flow_out["flow_loss"]
                    parts_t["stats_prior_loss"] = stats_prior_loss
                else:
                    stats_logits = stats_head(out["pair"], out["pair_mask"])
                stats_loss = stats_kl_loss(
                    stats_logits,
                    stats_target,
                    out["pair_mask"],
                    diag_weight=getattr(args, "stats_diag_weight", 0.2),
                    pair_weights=stats_pair_weights,
                )
                stats_marginal_loss = stats_marginal_consistency_loss(
                    stats_logits,
                    stats_target,
                    out["pair_mask"],
                    pair_weights=stats_pair_weights,
                )
                stats_log_odds = stats_log_odds_loss(
                    stats_logits,
                    stats_target,
                    out["pair_mask"],
                    pair_weights=stats_pair_weights,
                )
                stats_covariance = stats_covariance_loss(
                    stats_logits,
                    stats_target,
                    out["pair_mask"],
                    pair_weights=stats_pair_weights,
                )
                loss = loss + float(getattr(args, "stats_loss_weight", 1.0)) * stats_loss
                loss = loss + float(getattr(args, "stats_marginal_consistency_weight", 0.2)) * stats_marginal_loss
                loss = loss + float(getattr(args, "stats_log_odds_weight", 0.1)) * stats_log_odds
                loss = loss + float(getattr(args, "stats_covariance_weight", 0.5)) * stats_covariance
                scalar_consistency_weight = stats_scalar_consistency_weight_for_epoch(args, epoch)
                parts_t["stats_scalar_consistency_weight_effective"] = torch.as_tensor(
                    scalar_consistency_weight,
                    device=loss.device,
                )
                if scalar_consistency_weight > 0.0:
                    stats_consistency = stats_scalar_consistency_loss(
                        stats_logits,
                        batch["target"],
                        out["pair_mask"],
                    )
                    loss = loss + scalar_consistency_weight * stats_consistency
                    parts_t["stats_scalar_consistency_loss"] = stats_consistency
                parts_t["stats_loss"] = stats_loss
                parts_t["stats_kl_loss"] = stats_loss
                parts_t["stats_marginal_consistency_loss"] = stats_marginal_loss
                parts_t["stats_log_odds_loss"] = stats_log_odds
                parts_t["stats_covariance_loss"] = stats_covariance
        parts_t["loss"] = loss
    return loss, {key: float(value.detach().float().cpu().item()) for key, value in parts_t.items()}


@torch.no_grad()
def validate(
    args: argparse.Namespace,
    encoder: EvoNASSPairEncoder,
    head: torch.nn.Module,
    dataset: RESMTestDataset,
    length_by_id: dict[str, int],
    device: torch.device,
    epoch: int,
    stats_head: torch.nn.Module | None = None,
) -> dict[str, float]:
    if len(dataset) == 0:
        return {}

    encoder.eval()
    head.eval()
    if stats_head is not None:
        stats_head.eval()
    id_to_index = {rna_id: idx for idx, rna_id in enumerate(dataset.ids)}
    batches = make_epoch_batches(dataset.ids, length_by_id, args.batch_size, args.seed, epoch, shuffle=False)
    val_dir = args.out_dir / "validation" / f"epoch_{epoch:04d}"
    val_dir.mkdir(parents=True, exist_ok=True)

    per_sample: list[dict[str, float | str]] = []
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
            pred = head(out["pair"], out["pair_mask"], return_full=True)
            stats_logits = None
            stats_probs = None
            stats_mi = None
            fi_logits = None
            fi_probs = None
            fij_logits = None
            fij_probs = None
            fij_mi = None
            fij_eval_by_steps: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
            stats_pair_weights = None
            stats_head_type = getattr(args, "stats_head_type", "residual_mlp")
            if stats_head is not None:
                stats_pair_weights = stats_pair_weights_from_scalar_target(
                    batch["target"],
                    out["pair_mask"],
                    alpha=getattr(args, "stats_pair_weight_alpha", 2.0),
                    top_factor=getattr(args, "stats_pair_weight_top_factor", 2.0),
                    top_boost=getattr(args, "stats_pair_weight_top_boost", 2.0),
                )
                if (
                    stats_head_type in {"factorized_residual_mlp", "factorized_conditional_flow"}
                    and "fi_target" in batch
                    and "fij_target" in batch
                ):
                    if stats_head_type == "factorized_conditional_flow":
                        fi_logits = stats_head.fi_head(out["single"], mask=batch["mask"])
                        default_steps = int(getattr(args, "stats_flow_sample_steps", 4))
                        sample = stats_head.sample(
                            out["pair"],
                            out["pair_mask"],
                            objective=getattr(args, "stats_flow_objective", "velocity"),
                            steps=default_steps,
                        )
                        fij_logits = sample["sample_logits"]
                    else:
                        stats_out = stats_head(
                            out["single"],
                            out["pair"],
                            mask=batch["mask"],
                            pair_mask=out["pair_mask"],
                        )
                        fi_logits = stats_out["fi_logits"]
                        fij_logits = stats_out["fij_logits"]
                    fi_probs = torch.softmax(fi_logits.float(), dim=-1)
                    fij_probs = torch.softmax(fij_logits.float(), dim=-1)
                    fij_mi = stats_probs_to_mi_apc_torch(fij_probs, pair_mask=out["pair_mask"])
                    if stats_head_type == "factorized_conditional_flow":
                        eval_steps = parse_stats_flow_eval_sample_steps(
                            getattr(args, "stats_flow_eval_sample_steps", "")
                        )
                        for steps in eval_steps:
                            if steps == int(getattr(args, "stats_flow_sample_steps", 4)):
                                step_logits = fij_logits
                                step_probs = fij_probs
                                step_mi = fij_mi
                            else:
                                step_sample = stats_head.sample(
                                    out["pair"],
                                    out["pair_mask"],
                                    objective=getattr(args, "stats_flow_objective", "velocity"),
                                    steps=steps,
                                )
                                step_logits = step_sample["sample_logits"]
                                step_probs = torch.softmax(step_logits.float(), dim=-1)
                                step_mi = stats_probs_to_mi_apc_torch(
                                    step_probs,
                                    pair_mask=out["pair_mask"],
                                )
                            fij_eval_by_steps[steps] = (step_logits, step_probs, step_mi)
                elif stats_head_type == "flow" and "stats_target" in batch:
                    sample = stats_head.sample(
                        out["pair"],
                        out["pair_mask"],
                        objective=getattr(args, "stats_flow_objective", "endpoint"),
                        steps=getattr(args, "stats_flow_sample_steps", 16),
                    )
                    stats_logits = sample["sample_logits"]
                elif "stats_target" in batch:
                    stats_logits = stats_head(out["pair"], out["pair_mask"])
                if stats_logits is not None:
                    stats_probs = torch.softmax(stats_logits.float(), dim=-1)
                    stats_mi = stats_probs_to_mi_apc_torch(stats_probs, pair_mask=out["pair_mask"])

        target = batch["target"]
        for idx, rna_id in enumerate(batch["id"]):
            L = int(batch["length"][idx].item())
            pred_np = pred[idx, 0, :L, :L].detach().float().cpu().numpy()
            target_np = target[idx, 0, :L, :L].detach().float().cpu().numpy()
            metrics = map_metrics(pred_np, target_np, L)
            metrics["length"] = float(L)
            metrics["target_scale"] = float(batch["target_scale"][idx].detach().cpu().item())
            if (
                stats_head is not None
                and getattr(args, "stats_head_type", "residual_mlp")
                in {"factorized_residual_mlp", "factorized_conditional_flow"}
                and fi_probs is not None
                and fij_probs is not None
                and "fi_target" in batch
                and "fij_target" in batch
            ):
                pred_fi_np = fi_probs[idx, :L].detach().float().cpu().numpy()
                target_fi_np = batch["fi_target"][idx, :L].detach().float().cpu().numpy()
                pred_fij_np = fij_probs[idx, :L, :L].detach().float().cpu().numpy()
                target_fij_np = batch["fij_target"][idx, :L, :L].detach().float().cpu().numpy()
                pred_fij_mi_np = fij_mi[idx, :L, :L].detach().float().cpu().numpy()
                local_pair_weights = (
                    stats_pair_weights[idx : idx + 1, :L, :L]
                    if stats_pair_weights is not None
                    else None
                )
                local_fi_logits = fi_logits[idx : idx + 1, :L]
                local_fij_logits = fij_logits[idx : idx + 1, :L, :L]
                local_fi_target = batch["fi_target"][idx : idx + 1, :L]
                local_fij_target = batch["fij_target"][idx : idx + 1, :L, :L]
                local_pair_mask = out["pair_mask"][idx : idx + 1, :L, :L]
                metrics["fi_mse"] = float(
                    fi_mse_loss(local_fi_logits, local_fi_target, batch["mask"][idx : idx + 1, :L])
                    .detach()
                    .float()
                    .cpu()
                    .item()
                )
                metrics["fi_mae"] = float(np.mean(np.abs(pred_fi_np - target_fi_np)))
                metrics["fij_kl"] = float(
                    fij_kl_loss(
                        local_fij_logits,
                        local_fij_target,
                        local_pair_mask,
                        pair_weights=local_pair_weights,
                    )
                    .detach()
                    .float()
                    .cpu()
                    .item()
                )
                upper = np.triu(np.ones((L, L), dtype=bool), k=1)
                metrics["fij_mae"] = float(np.mean(np.abs(pred_fij_np[upper] - target_fij_np[upper])))
                metrics["fij_marginal_target"] = float(
                    fij_marginal_target_loss(
                        local_fij_logits,
                        local_fi_target,
                        local_pair_mask,
                        pair_weights=local_pair_weights,
                    )
                    .detach()
                    .float()
                    .cpu()
                    .item()
                )
                metrics["fij_marginal_pred"] = float(
                    fij_marginal_pred_loss(
                        local_fij_logits,
                        local_fi_logits,
                        local_pair_mask,
                        pair_weights=local_pair_weights,
                    )
                    .detach()
                    .float()
                    .cpu()
                    .item()
                )
                metrics["fij_log_odds"] = float(
                    fij_log_odds_loss(
                        local_fij_logits,
                        local_fij_target,
                        local_fi_target,
                        local_pair_mask,
                        pair_weights=local_pair_weights,
                        clip_value=getattr(args, "stats_log_odds_clip", 6.0),
                    )
                    .detach()
                    .float()
                    .cpu()
                    .item()
                )
                metrics["fij_covariance"] = float(
                    fij_covariance_loss(
                        local_fij_logits,
                        local_fij_target,
                        local_fi_target,
                        local_pair_mask,
                        pair_weights=local_pair_weights,
                    )
                    .detach()
                    .float()
                    .cpu()
                    .item()
                )
                fij_map_metrics = map_metrics(pred_fij_mi_np, target_np, L)
                metrics["fij_scalar_pearson"] = float(fij_map_metrics["pearson"])
                metrics["fij_scalar_spearman"] = float(fij_map_metrics["spearman"])
                metrics["fij_scalar_top_l_overlap"] = float(fij_map_metrics["top_l_overlap"])
                metrics["stats_kl"] = metrics["fij_kl"]
                metrics["stats_mae"] = metrics["fij_mae"]
                metrics["stats_scalar_pearson"] = metrics["fij_scalar_pearson"]
                metrics["stats_scalar_spearman"] = metrics["fij_scalar_spearman"]
                metrics["stats_scalar_top_l_overlap"] = metrics["fij_scalar_top_l_overlap"]
                for steps, (step_logits, step_probs, step_mi) in fij_eval_by_steps.items():
                    local_step_logits = step_logits[idx : idx + 1, :L, :L]
                    local_step_probs = step_probs[idx, :L, :L].detach().float().cpu().numpy()
                    local_step_mi = step_mi[idx, :L, :L].detach().float().cpu().numpy()
                    step_metrics = map_metrics(local_step_mi, target_np, L)
                    metrics[f"stats_kl_steps_{steps}"] = float(
                        fij_kl_loss(
                            local_step_logits,
                            local_fij_target,
                            local_pair_mask,
                            pair_weights=local_pair_weights,
                        )
                        .detach()
                        .float()
                        .cpu()
                        .item()
                    )
                    metrics[f"stats_mae_steps_{steps}"] = float(
                        np.mean(np.abs(local_step_probs[upper] - target_fij_np[upper]))
                    )
                    metrics[f"stats_scalar_pearson_steps_{steps}"] = float(step_metrics["pearson"])
                    metrics[f"stats_scalar_spearman_steps_{steps}"] = float(step_metrics["spearman"])
                    metrics[f"stats_scalar_top_l_overlap_steps_{steps}"] = float(
                        step_metrics["top_l_overlap"]
                    )
                np.save(val_dir / f"{rna_id}_pred_fi.npy", pred_fi_np)
                np.save(val_dir / f"{rna_id}_target_fi.npy", target_fi_np)
                np.save(val_dir / f"{rna_id}_pred_fij.npy", pred_fij_np)
                np.save(val_dir / f"{rna_id}_target_fij.npy", target_fij_np)
                np.save(val_dir / f"{rna_id}_pred_fij_mi_apc.npy", pred_fij_mi_np)
            elif stats_head is not None and stats_probs is not None and "stats_target" in batch:
                pred_stats_np = stats_probs[idx, :L, :L].detach().float().cpu().numpy()
                target_stats_np = batch["stats_target"][idx, :L, :L].detach().float().cpu().numpy()
                pred_stats_mi_np = stats_mi[idx, :L, :L].detach().float().cpu().numpy()
                stats_kl = stats_kl_loss(
                    stats_logits[idx : idx + 1, :L, :L],
                    batch["stats_target"][idx : idx + 1, :L, :L],
                    out["pair_mask"][idx : idx + 1, :L, :L],
                    diag_weight=getattr(args, "stats_diag_weight", 0.2),
                    pair_weights=stats_pair_weights[idx : idx + 1, :L, :L] if stats_pair_weights is not None else None,
                )
                metrics["stats_kl"] = float(stats_kl.detach().float().cpu().item())
                metrics["stats_marginal_consistency"] = float(
                    stats_marginal_consistency_loss(
                        stats_logits[idx : idx + 1, :L, :L],
                        batch["stats_target"][idx : idx + 1, :L, :L],
                        out["pair_mask"][idx : idx + 1, :L, :L],
                        pair_weights=stats_pair_weights[idx : idx + 1, :L, :L]
                        if stats_pair_weights is not None
                        else None,
                    )
                    .detach()
                    .float()
                    .cpu()
                    .item()
                )
                metrics["stats_log_odds"] = float(
                    stats_log_odds_loss(
                        stats_logits[idx : idx + 1, :L, :L],
                        batch["stats_target"][idx : idx + 1, :L, :L],
                        out["pair_mask"][idx : idx + 1, :L, :L],
                        pair_weights=stats_pair_weights[idx : idx + 1, :L, :L]
                        if stats_pair_weights is not None
                        else None,
                    )
                    .detach()
                    .float()
                    .cpu()
                    .item()
                )
                metrics["stats_covariance"] = float(
                    stats_covariance_loss(
                        stats_logits[idx : idx + 1, :L, :L],
                        batch["stats_target"][idx : idx + 1, :L, :L],
                        out["pair_mask"][idx : idx + 1, :L, :L],
                        pair_weights=stats_pair_weights[idx : idx + 1, :L, :L]
                        if stats_pair_weights is not None
                        else None,
                    )
                    .detach()
                    .float()
                    .cpu()
                    .item()
                )
                metrics["stats_mae"] = float(np.mean(np.abs(pred_stats_np - target_stats_np)))
                stats_map_metrics = map_metrics(pred_stats_mi_np, target_np, L)
                metrics["stats_scalar_pearson"] = float(stats_map_metrics["pearson"])
                metrics["stats_scalar_spearman"] = float(stats_map_metrics["spearman"])
                metrics["stats_scalar_top_l_overlap"] = float(stats_map_metrics["top_l_overlap"])
                np.save(val_dir / f"{rna_id}_pred_stats.npy", pred_stats_np)
                np.save(val_dir / f"{rna_id}_target_stats.npy", target_stats_np)
                np.save(val_dir / f"{rna_id}_pred_stats_mi_apc.npy", pred_stats_mi_np)
            per_sample.append({"id": rna_id, **metrics})
            np.save(val_dir / f"{rna_id}_pred_map.npy", pred_np)
            np.save(val_dir / f"{rna_id}_target_map.npy", target_np)

    with (val_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(per_sample, handle, indent=2)
    summary: dict[str, float] = {}
    for key in ("pearson", "spearman", "mae", "top_l_overlap"):
        summary[key] = float(np.mean([float(row[key]) for row in per_sample])) if per_sample else 0.0
    for key in (
        "stats_kl",
        "stats_marginal_consistency",
        "stats_log_odds",
        "stats_covariance",
        "stats_mae",
        "stats_scalar_pearson",
        "stats_scalar_spearman",
        "stats_scalar_top_l_overlap",
        "fi_mse",
        "fi_mae",
        "fij_kl",
        "fij_mae",
        "fij_marginal_target",
        "fij_marginal_pred",
        "fij_log_odds",
        "fij_covariance",
        "fij_scalar_pearson",
        "fij_scalar_spearman",
        "fij_scalar_top_l_overlap",
    ):
        values = [float(row[key]) for row in per_sample if key in row and np.isfinite(float(row[key]))]
        if values:
            summary[key] = float(np.mean(values))
    step_metric_prefixes = (
        "stats_kl_steps_",
        "stats_mae_steps_",
        "stats_scalar_pearson_steps_",
        "stats_scalar_spearman_steps_",
        "stats_scalar_top_l_overlap_steps_",
    )
    step_keys = sorted(
        {
            key
            for row in per_sample
            for key in row
            if any(str(key).startswith(prefix) for prefix in step_metric_prefixes)
        }
    )
    for key in step_keys:
        values = [float(row[key]) for row in per_sample if key in row and np.isfinite(float(row[key]))]
        if values:
            summary[key] = float(np.mean(values))
    eval_steps = parse_stats_flow_eval_sample_steps(getattr(args, "stats_flow_eval_sample_steps", ""))
    best_step: int | None = None
    best_step_score: float | None = None
    for steps in eval_steps:
        pearson = summary.get(f"stats_scalar_pearson_steps_{steps}")
        top_l = summary.get(f"stats_scalar_top_l_overlap_steps_{steps}")
        kl_value = summary.get(f"stats_kl_steps_{steps}")
        if pearson is None or top_l is None or kl_value is None:
            continue
        score = 0.45 * float(pearson) + 0.35 * float(top_l) + 0.20 / (1.0 + max(float(kl_value), 0.0))
        summary[f"stats_flow_eval_score_steps_{steps}"] = float(score)
        if best_step_score is None or score > best_step_score:
            best_step = steps
            best_step_score = float(score)
    if best_step is not None and best_step_score is not None:
        summary["stats_flow_best_eval_sample_steps"] = float(best_step)
        summary["stats_flow_best_eval_score"] = float(best_step_score)
    return summary


def best_checkpoint_score(args: argparse.Namespace, val_summary: dict[str, float]) -> tuple[float | None, dict[str, object]]:
    if not val_summary:
        return None, {}

    metric = getattr(args, "best_checkpoint_metric", "composite_topl")
    if metric == "mae":
        mae = float(val_summary["mae"])
        score = -mae
        return score, {
            "best_monitor": "mae",
            "best_score": score,
            "best_score_components": {"mae": mae},
        }

    if metric == "stats_kl":
        if "stats_kl" not in val_summary:
            return None, {}
        stats_kl_value = float(val_summary["stats_kl"])
        score = -stats_kl_value
        return score, {
            "best_monitor": "stats_kl",
            "best_score": score,
            "best_score_components": {"stats_kl": stats_kl_value},
        }

    if metric == "stats_inverse":
        scalar_pearson = float(val_summary.get("pearson", 0.0))
        scalar_top_l = float(val_summary.get("top_l_overlap", 0.0))
        stats_scalar_pearson = float(val_summary.get("stats_scalar_pearson", scalar_pearson))
        stats_scalar_top_l = float(val_summary.get("stats_scalar_top_l_overlap", scalar_top_l))
        stats_kl_value = float(val_summary.get("stats_kl", 1e6))
        stats_quality = 1.0 / (1.0 + max(stats_kl_value, 0.0))
        components = {
            "scalar_pearson": scalar_pearson,
            "scalar_top_l_overlap": scalar_top_l,
            "stats_scalar_pearson": stats_scalar_pearson,
            "stats_scalar_top_l_overlap": stats_scalar_top_l,
            "stats_kl": stats_kl_value,
            "stats_quality": stats_quality,
            "weight_scalar_pearson": 0.25,
            "weight_scalar_top_l_overlap": 0.15,
            "weight_stats_scalar_pearson": 0.30,
            "weight_stats_scalar_top_l_overlap": 0.20,
            "weight_stats_quality": 0.10,
        }
        score = (
            0.25 * scalar_pearson
            + 0.15 * scalar_top_l
            + 0.30 * stats_scalar_pearson
            + 0.20 * stats_scalar_top_l
            + 0.10 * stats_quality
        )
        return float(score), {
            "best_monitor": "stats_inverse",
            "best_score": float(score),
            "best_score_components": components,
        }

    if metric not in {"composite_topl", "joint"}:
        raise ValueError(f"Unsupported best checkpoint metric: {metric}")

    mae_ref = float(getattr(args, "best_checkpoint_mae_ref", 0.05))
    if mae_ref <= 0:
        raise ValueError("--best_checkpoint_mae_ref must be > 0")
    top_l = float(val_summary["top_l_overlap"])
    spearman = float(val_summary["spearman"])
    pearson = float(val_summary["pearson"])
    mae = float(val_summary["mae"])
    mae_quality = 1.0 - min(max(mae, 0.0) / mae_ref, 1.0)
    components = {
        "top_l_overlap": top_l,
        "spearman": spearman,
        "pearson": pearson,
        "mae": mae,
        "mae_quality": mae_quality,
        "weight_top_l_overlap": 0.50,
        "weight_spearman": 0.30,
        "weight_pearson": 0.15,
        "weight_mae_quality": 0.05,
        "mae_ref": mae_ref,
    }
    score = 0.50 * top_l + 0.30 * spearman + 0.15 * pearson + 0.05 * mae_quality
    if metric == "joint":
        stats_top = float(val_summary.get("stats_scalar_top_l_overlap", top_l))
        stats_pearson = float(val_summary.get("stats_scalar_pearson", pearson))
        stats_kl_value = float(val_summary.get("stats_kl", 0.0))
        stats_quality = 1.0 / (1.0 + max(stats_kl_value, 0.0))
        components.update(
            {
                "stats_scalar_top_l_overlap": stats_top,
                "stats_scalar_pearson": stats_pearson,
                "stats_kl": stats_kl_value,
                "stats_quality": stats_quality,
                "weight_scalar_composite": 0.60,
                "weight_stats_top": 0.20,
                "weight_stats_pearson": 0.15,
                "weight_stats_quality": 0.05,
            }
        )
        score = 0.60 * score + 0.20 * stats_top + 0.15 * stats_pearson + 0.05 * stats_quality
    return float(score), {
        "best_monitor": metric,
        "best_score": float(score),
        "best_score_components": components,
    }


def validation_best_monitor(args: argparse.Namespace, val_summary: dict[str, float]) -> float | None:
    if not val_summary:
        return None
    if "best_score" in val_summary:
        return float(val_summary["best_score"])
    score, _ = best_checkpoint_score(args, val_summary)
    return score


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
    stats_head: torch.nn.Module | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
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
    }
    if stats_head is not None:
        payload["stats_head"] = stats_head.state_dict()
    torch.save(payload, path)


def load_checkpoint(
    resume_path: Path,
    encoder: EvoNASSPairEncoder,
    head: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    stats_head: torch.nn.Module | None = None,
) -> tuple[int, int, float | None, list[dict[str, object]], list[dict[str, object]]]:
    ckpt = torch.load(resume_path, map_location=device)
    encoder.load_state_dict(ckpt["encoder"])
    head.load_state_dict(ckpt["head"])
    if stats_head is not None:
        if "stats_head" not in ckpt:
            raise ValueError(f"{resume_path} does not contain stats_head weights")
        stats_head.load_state_dict(ckpt["stats_head"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return (
        int(ckpt.get("epoch", 0)) + 1,
        int(ckpt.get("global_step", 0)),
        ckpt.get("best_metric"),
        list(ckpt.get("train_logs", [])),
        list(ckpt.get("val_logs", [])),
    )


def write_metrics(
    args: argparse.Namespace,
    cfg: SSPairEncoderConfig,
    split: dict[str, list[str]],
    train_logs,
    val_logs,
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
    enable_stats_target = args.stats_head_type != "none"
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
        target_builder=args.target_builder,
        enable_stats_target=enable_stats_target,
        target_clustering_seqid=args.target_clustering_seqid,
        target_remove_duplicates=not args.target_keep_duplicates,
        target_no_reweighting=args.target_no_reweighting,
        target_pseudocount=args.target_pseudocount,
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
        target_builder=args.target_builder,
        enable_stats_target=enable_stats_target,
        target_clustering_seqid=args.target_clustering_seqid,
        target_remove_duplicates=not args.target_keep_duplicates,
        target_no_reweighting=args.target_no_reweighting,
        target_pseudocount=args.target_pseudocount,
    )
    train_index = {rna_id: idx for idx, rna_id in enumerate(train_dataset.ids)}

    encoder, head = build_models(args, cfg, device)
    stats_head = build_stats_head(args, cfg, device)
    params = list(encoder.parameters()) + list(head.parameters())
    if stats_head is not None:
        params += list(stats_head.parameters())
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
            stats_head=stats_head,
        )
        print(f"Resumed {args.resume} at epoch={start_epoch} global_step={global_step}")

    stop_training = False
    for epoch in range(start_epoch, args.epochs + 1):
        encoder.train()
        head.train()
        if stats_head is not None:
            stats_head.train()
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
            loss, loss_parts = compute_train_loss(args, encoder, head, batch, stats_head=stats_head, epoch=epoch)
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
            val_summary = validate(
                args,
                encoder,
                head,
                val_dataset,
                length_by_id,
                device,
                epoch,
                stats_head=stats_head,
            )
            _, best_fields = best_checkpoint_score(args, val_summary)
            val_summary = {**val_summary, **best_fields}
            val_log = {"epoch": epoch, "global_step": global_step, **val_summary}
            val_logs.append(val_log)
            print(
                f"validation epoch {epoch:03d}: "
                f"pearson={val_summary.get('pearson', 0.0):.4f} "
                f"mae={val_summary.get('mae', 0.0):.4f} "
                f"topL={val_summary.get('top_l_overlap', 0.0):.4f} "
                f"best_score={val_summary.get('best_score', 0.0):.4f}"
            )

        monitor = validation_best_monitor(args, val_summary)
        if monitor is not None:
            if best_metric is None or monitor > best_metric:
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
                    stats_head=stats_head,
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
            stats_head=stats_head,
        )
        write_metrics(args, cfg, split, train_logs, val_logs, best_metric)

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
    if args.stats_flow_sample_steps < 1:
        raise ValueError("--stats_flow_sample_steps must be >= 1")
    if args.stats_flow_condition_dim < 1:
        raise ValueError("--stats_flow_condition_dim must be >= 1")
    if (
        args.stats_head_type == "factorized_conditional_flow"
        and args.stats_flow_objective == "endpoint"
        and args.stats_flow_sample_steps != 1
    ):
        raise ValueError("--stats_flow_objective endpoint requires --stats_flow_sample_steps 1")
    eval_sample_steps = parse_stats_flow_eval_sample_steps(args.stats_flow_eval_sample_steps)
    if (
        args.stats_head_type == "factorized_conditional_flow"
        and args.stats_flow_objective == "endpoint"
        and any(steps != 1 for steps in eval_sample_steps)
    ):
        raise ValueError("--stats_flow_objective endpoint requires validation sample steps to be 1")
    if args.stats_prior_warmup_epochs < 0:
        raise ValueError("--stats_prior_warmup_epochs must be >= 0")
    if args.stats_scalar_consistency_warmup_epochs < 0:
        raise ValueError("--stats_scalar_consistency_warmup_epochs must be >= 0")
    for name in (
        "scalar_loss_weight",
        "stats_loss_weight",
        "stats_fi_loss_weight",
        "stats_fij_loss_weight",
        "stats_fij_velocity_weight",
        "stats_fij_endpoint_weight",
        "stats_diag_weight",
        "stats_pair_weight_alpha",
        "stats_pair_weight_top_factor",
        "stats_pair_weight_top_boost",
        "stats_marginal_consistency_weight",
        "stats_marginal_pred_weight",
        "stats_log_odds_weight",
        "stats_covariance_weight",
        "stats_scalar_consistency_weight",
        "stats_prior_loss_weight",
        "stats_prior_warmup_start",
    ):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name} must be >= 0")
    if args.target_builder == "legacy_mi_apc" and args.stats_head_type != "none":
        raise ValueError("stats_head_type requires --target_builder adabm_stats")

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
