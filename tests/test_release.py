"""Synthetic release checks: no original data or pretrained weights required."""

import ast
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch

from RethinkingMSA_1.resm_dataset import RESMTestDataset, _load_target_mi_apc
from RethinkingMSA_1.train_direct_trunk import build_encoder_config
from RethinkingMSA_1.train_direct_trunk_ddp import build_parser, rank_target_cache_dir
from train import profile_arguments


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def synthetic_data(tmp_path):
    rng = np.random.default_rng(7)
    for directory in ("embedding", "attention_map", "query", "bpp", "msa"):
        (tmp_path / directory).mkdir()
    for rna_id in ("tiny_train", "tiny_val", "tiny_test"):
        np.save(tmp_path / "embedding" / f"{rna_id}_emb.npy", rng.normal(size=(8, 16)).astype("float32"))
        np.save(tmp_path / "attention_map" / f"{rna_id}_atp.npy", rng.normal(size=(8, 8, 12)).astype("float32"))
        (tmp_path / "query" / f"{rna_id}.fasta").write_text(f">{rna_id}\nACGUACGU\n", encoding="utf-8")
        (tmp_path / "bpp" / f"{rna_id}.prob").write_text("1 8 0.7\n2 7 0.4\n", encoding="utf-8")
        (tmp_path / "msa" / f"{rna_id}.a3m").write_text(
            ">query\nACGUACGU\n>one\nACGUUCGU\n>two\nGCGUACGA\n>three\nAC-UACGU\n",
            encoding="utf-8",
        )
    (tmp_path / "val.fasta").write_text(">tiny_val\nACGUACGU\n", encoding="utf-8")
    (tmp_path / "test.fasta").write_text(">tiny_test\nACGUACGU\n", encoding="utf-8")
    return tmp_path


def run_cli(arguments, cwd=ROOT):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    for key in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(key, None)
    result = subprocess.run(
        [sys.executable, *arguments], cwd=cwd, env=env,
        text=True, capture_output=True, timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.mark.parametrize("module", [
    "RethinkingMSA_1", "RethinkingMSA_1.train_direct_trunk",
    "RethinkingMSA_1.train_direct_trunk_ddp", "RethinkingMSA_1.train_resm_smoke",
    "RethinkingMSA_1.mi3_potts_inverse", "inverse_coevo_sampler",
    "RethinkingMSA_1.export_adabm_predicted_stats",
])
def test_imports_resolve_inside_release(module):
    imported = importlib.import_module(module)
    assert Path(imported.__file__).resolve().is_relative_to(ROOT)


@pytest.mark.parametrize("entry", [
    ["-m", "RethinkingMSA_1.train_direct_trunk"],
    ["-m", "RethinkingMSA_1.train_direct_trunk_ddp"],
    ["-m", "RethinkingMSA_1.train_resm_smoke"],
    ["train.py", "--profile", "configs/primary_8k.json"],
])
def test_cli_help(entry):
    assert "--target_cache_dir" in run_cli([*entry, "--help"]).stdout


def test_profile_preserves_recorded_config_and_explicit_overrides():
    profile = json.loads((ROOT / "configs/primary_8k.json").read_text(encoding="utf-8"))
    parser = build_parser()
    args = parser.parse_args(profile_arguments(ROOT / "configs/primary_8k.json"))
    cfg = build_encoder_config(args, [{"status": "OK", "resm_single_dim": 1280, "resm_pair_dim": 660}])
    assert vars(cfg) == profile["encoder_config"]
    assert (args.batch_size, args.grad_accum, args.num_recycles) == (16, 2, 3)
    defaults = parser.parse_args([])
    assert (defaults.batch_size, defaults.grad_accum, defaults.num_recycles) == (2, 16, 1)
    assert not defaults.memory_target_cache
    assert not args.memory_target_cache
    overridden = parser.parse_args([*profile_arguments(ROOT / "configs/primary_8k.json"), "--batch_size", "1"])
    assert overridden.batch_size == 1


def test_recorded_membership_and_profile_agree():
    recorded = ROOT / "configs/recorded"
    record = json.loads((recorded / "primary_8k_epoch75_record.json").read_text(encoding="utf-8"))
    profile = json.loads((ROOT / "configs/primary_8k.json").read_text(encoding="utf-8"))
    for key, value in profile["args"].items():
        assert record["recorded_nonpath_args"][key] == value
    membership = json.loads((recorded / "split_membership.json").read_text(encoding="utf-8"))
    assert len(membership["train"]) == record["training_count"] == 8296
    assert len(membership["validation"]) == 48
    assert len(membership["test"]) == 36
    assert record["recorded_internal_split_counts"] == {"train": 8296, "validation": 46, "test": 36}
    train_ids = {row["id"] for row in membership["train"]}
    assert not train_ids.intersection(membership["validation"])
    assert not train_ids.intersection(membership["test"])
    validation = json.loads((recorded / "metadata_validation.json").read_text(encoding="utf-8"))
    for name, digest in validation["files"].items():
        assert hashlib.sha256((recorded / name).read_bytes()).hexdigest() == digest


def test_epoch45_profile_is_separate_from_primary_membership():
    profile = json.loads((ROOT / "configs/epoch45.json").read_text(encoding="utf-8"))
    record = json.loads((ROOT / "configs/recorded/epoch45_checkpoint_record.json").read_text(encoding="utf-8"))
    assert profile["args"] == record["recorded_nonpath_args"]
    args = build_parser().parse_args(profile_arguments(ROOT / "configs/epoch45.json"))
    cfg = build_encoder_config(args, [{"status": "OK", "resm_single_dim": 1280, "resm_pair_dim": 660}])
    assert vars(cfg) == profile["encoder_config"] == record["encoder_config"]
    assert (record["epoch"], record["global_step"]) == (45, 10791)
    assert args.grad_accum == 4
    assert args.dataset4_name == "nufolddb_relaxed_gap25"
    assert not args.memory_target_cache
    assert record["has_embedded_split"] is False
    assert profile["provenance"]["historical_world_size"] == record["historical_world_size"] == 4
    assert record["training_count"] == 50990
    assert record["recorded_split_counts"] == {"train": 50990, "validation": 46, "test": 36}
    membership = json.loads((ROOT / "configs/recorded/epoch45_split_membership.json").read_text(encoding="utf-8"))
    assert len(membership["train"]) == 50990
    train_keys = {(row["source"], row["id"]) for row in membership["train"]}
    assert len(train_keys) == 50990
    assert not train_keys.intersection((row["source"], row["id"]) for row in membership["validation"] + membership["test"])
    assert len(record["checkpoint_sha256"]) == 64


def test_rank_cache_is_separate(tmp_path):
    args = build_parser().parse_args(["--target_cache_dir", str(tmp_path)])
    rank0 = rank_target_cache_dir(args, rank=0, world_size=4)
    rank1 = rank_target_cache_dir(args, rank=1, world_size=4)
    assert rank0 != rank1
    assert rank0.parent == rank1.parent == tmp_path


def test_explicit_missing_holdouts_never_disable_exclusions(tmp_path):
    from RethinkingMSA_1.resm_dataset import read_fasta_ids, read_test_folder_ids
    from RethinkingMSA_1.train_direct_trunk_ddp import split_multi_source_train_val_test_ids

    assert read_fasta_ids(None) == read_test_folder_ids(None) == set()
    rows = [{"id": "source__heldout", "source_id": "heldout"}]
    missing = tmp_path / "missing.fasta"
    for val, test, folder in ((missing, None, None), (None, missing, None), (None, None, missing)):
        with pytest.raises(FileNotFoundError):
            split_multi_source_train_val_test_ids(["source__heldout"], rows, val, test, folder)

    valid = tmp_path / "val.fasta"
    valid.write_text(">heldout\nACGU\n", encoding="ascii")
    assert split_multi_source_train_val_test_ids(["source__heldout"], rows, valid, None, None) == (
        [], ["source__heldout"], [])


def test_disk_target_reused_without_memory_retention(synthetic_data, monkeypatch):
    data = synthetic_data
    dataset = RESMTestDataset(
        data, data / "query", data / "bpp", data / "msa", ids=["tiny_train"],
        embedding_dir=data / "embedding", target_cache_dir=data / "cache",
        enable_stats_target=True,
    )
    first = dataset[0]
    cache_files = list((data / "cache").glob("*.npz"))
    assert len(cache_files) == 1
    assert not dataset.cache_targets
    assert not dataset._target_cache

    def fail_compute(*args, **kwargs):
        raise AssertionError("Expected disk-cache reuse")

    monkeypatch.setattr("RethinkingMSA_1.stat_targets.compute_adabm_stats_target", fail_compute)
    second = dataset[0]
    assert torch.equal(first["fij_target"], second["fij_target"])
    assert not dataset._target_cache
    assert first["fi_target"].shape == (8, 5)
    assert first["fij_target"].shape == (8, 8, 25)


def test_legacy_lazy_helper_dependency(synthetic_data):
    target = _load_target_mi_apc(synthetic_data / "msa/tiny_train.a3m", 2001, expected_length=8)
    assert target.shape == (8, 8)
    assert np.isfinite(target).all()


def test_local_import_closure():
    external = set(sys.stdlib_module_names) | {"numpy", "torch", "matplotlib", "adabmDCA"}
    for source in [*(ROOT / "RethinkingMSA_1").glob("*.py"), ROOT / "inverse_coevo_sampler.py"]:
        for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom):
                if node.level:
                    assert (source.parent / f"{node.module}.py").is_file()
                    continue
                modules = [node.module]
            elif isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            else:
                continue
            for module in modules:
                if module.split(".")[0] not in external:
                    target = ROOT.joinpath(*module.split("."))
                    assert target.with_suffix(".py").is_file() or (target / "__init__.py").is_file(), module


def test_source_manifest():
    assert "* -text" in (ROOT / ".gitattributes").read_text(encoding="utf-8")
    for script in (ROOT / "examples/train.sh", ROOT / "examples/run_a800.sbatch"):
        assert b"\r" not in script.read_bytes()
    manifest = json.loads((ROOT / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    assert len(manifest["files"]) == 17
    for record in manifest["files"]:
        assert hashlib.sha256((ROOT / record["path"]).read_bytes()).hexdigest() == record["release_sha256"]


def test_isolated_tiny_cli_training(synthetic_data, tmp_path):
    # A separate source copy prevents accidental imports from the parent EvoNA checkout.
    isolated = tmp_path / "isolated"
    isolated.mkdir()
    shutil.copytree(ROOT / "RethinkingMSA_1", isolated / "RethinkingMSA_1", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / "configs", isolated / "configs")
    for name in ("train.py", "inverse_coevo_sampler.py"):
        shutil.copy2(ROOT / name, isolated / name)
    data = synthetic_data
    output = tmp_path / "output"
    training_args = [
        "train.py", "--profile", "configs/primary_8k.json",
        "--embedding_dir", str(data / "embedding"), "--attention_dir", str(data / "attention_map"),
        "--query_dir", str(data / "query"), "--bpp_dir", str(data / "bpp"), "--msa_dir", str(data / "msa"),
        "--val_fasta", str(data / "val.fasta"), "--test_fasta", str(data / "test.fasta"),
        "--out_dir", str(output), "--device", "cpu", "--amp_dtype", "none",
        "--epochs", "1", "--steps", "1", "--batch_size", "1", "--grad_accum", "1",
        "--resm_single_dim", "16", "--resm_pair_dim", "12", "--d_s", "32", "--d_z", "16",
        "--sequence_head_width", "8", "--pairwise_head_width", "8",
        "--num_blocks", "1", "--num_recycles", "1", "--head_hidden_dim", "32",
        "--stats_head_hidden_dim", "8", "--stats_flow_condition_dim", "8",
        "--validate_every", "1", "--stats_flow_sample_steps", "1",
        "--no_gradient_checkpointing", "--triangle_attention_chunk_size", "0",
    ]
    result = run_cli(training_args, cwd=isolated)
    assert "Training complete" in result.stdout
    metrics = json.loads((output / "train_metrics.json").read_text(encoding="utf-8"))
    assert len(metrics["train_logs"]) == 1
    log = metrics["train_logs"][0]
    assert log["global_step"] == 1
    assert np.isfinite(log["loss"])
    assert log["grad_norm"] > 0
    assert "fij_velocity_loss" in log
    assert metrics["args"]["memory_target_cache"] is False
    assert len(metrics["split"]["train"]) == len(metrics["split"]["val"]) == len(metrics["split"]["test"]) == 1
    assert len(metrics["val_logs"]) == 1
    assert (output / "last.pt").is_file()
    assert (output / "best.pt").is_file()
    assert list((output / "target_cache").rglob("*.npz"))
    checkpoint = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
    assert checkpoint["global_step"] == 1
    assert checkpoint["stats_head"]
    assert checkpoint["optimizer"]["state"]
    resumed = run_cli([
        *training_args, "--resume", str(output / "last.pt"), "--epochs", "2", "--steps", "2",
    ], cwd=isolated)
    assert "Resumed" in resumed.stdout
    second = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
    assert second["epoch"] == 2
    assert second["global_step"] == 2
    assert len(second["train_logs"]) == 2
    first_steps = [state["step"].item() for state in checkpoint["optimizer"]["state"].values()]
    second_steps = [state["step"].item() for state in second["optimizer"]["state"].values()]
    assert min(first_steps) == max(first_steps) == 1
    assert min(second_steps) == max(second_steps) == 2
    assert any(not torch.equal(checkpoint["encoder"][key], second["encoder"][key]) for key in checkpoint["encoder"])

    exported = tmp_path / "exported"
    run_cli([
        "-m", "RethinkingMSA_1.export_adabm_predicted_stats",
        "--checkpoint", str(output / "last.pt"), "--out_dir", str(exported),
        "--embedding_dir", str(data / "embedding"), "--attention_dir", str(data / "attention_map"),
        "--query_dir", str(data / "query"), "--bpp_dir", str(data / "bpp"), "--msa_dir", str(data / "msa"),
        "--ids", "tiny_test", "--device", "cpu", "--amp_dtype", "none", "--stats_flow_sample_steps", "1",
    ], cwd=isolated)
    stats_path = exported / "tiny_test/adabm_stats.pt"
    stats = torch.load(stats_path, map_location="cpu", weights_only=False)
    assert stats["tokens"] == "-ACGU"
    assert stats["fi"].shape == (8, 5)
    assert stats["fij"].shape == (8, 5, 8, 5)
    assert torch.isfinite(stats["fij"]).all()
    assert (exported / "tiny_test/target_scalar_map.npy").is_file()

    if importlib.util.find_spec("adabmDCA") is not None:
        inverse = tmp_path / "inverse"
        run_cli([
            "-m", "RethinkingMSA_1.run_adabm_inverse_from_predicted_stats",
            "--stats_path", str(stats_path), "--out_dir", str(inverse), "--device", "cpu",
            "--nchains", "16", "--ngen", "4", "--max_epochs", "1",
            "--nsweeps", "1", "--sample_nsweeps", "1",
        ], cwd=isolated)
        fasta = (inverse / "tiny_test/samples.fasta").read_text(encoding="utf-8")
        assert fasta.count(">") == 4
