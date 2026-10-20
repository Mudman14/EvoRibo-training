# EvoRibo Training

Standalone source snapshot of the RESM/BPP-conditioned pair trunk and factorized
conditional velocity-flow statistics head. The Python package remains
`RethinkingMSA_1` for import/checkpoint compatibility.

## Install And Check

Use Python >=3.10. For GPU runs, install a CUDA-compatible PyTorch build for your
machine first. Linux with CUDA/NCCL is the intended multi-GPU platform.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pip install --no-deps -e .
python -m pytest -q
python train.py --help
```

Dependency ranges are not the original training environment lock. See
[REPRODUCIBILITY.md](REPRODUCIBILITY.md) for provenance and verification limits.
No data, pretrained weights, checkpoints, private paths, or Git history are
included. Training consumes already extracted RESM features; RESM/SPOT-RNA
feature extraction and upstream models must be supplied separately.

## Inputs

For each of `source1`, `source2`, `source3`, use this layout under `data/`:

```text
source1/embedding/<id>_emb.npy      # [L,1280] or [1,L,1280]
source1/attention_map/<id>_atp.npy  # [L,L,660] or [660,L,L]
source1/query/<id>.fasta           # first record, ungapped query
source1/bpp/<id>.prob              # dense LxL or 1-based i j probability
source1/msa/<id>.a3m               # aligned target MSA, query first
val.fasta                         # held-out IDs in FASTA headers
test.fasta                        # held-out IDs in FASTA headers
```

Explicitly supplied held-out files/directories must exist; a typo raises an
error instead of disabling exclusions.

All features and MSA alignment columns must agree on query length; do not mix
RESM model variants. Query token order is `AUCG-N`; statistics use `-ACGU`.
Lowercase A3M insertions are removed and T is converted to U. Target rows with
invalid tokens or unequal cleaned lengths are excluded. Inspect the audit,
`split.json`, and source counts before a long run. Dataset sources are named
`nufold`, `repa_round1`, `nufolddb_round3`; split exclusion matches source-local
or qualified IDs, not sequence homology. Supply independently curated held-out
sets to control leakage. Recorded primary split membership is included under
`configs/recorded/`: 8296 train, 48 validation, 36 test. These are IDs, not data.
The validation48/test36 IDs are the official benchmark. Actual historical
training sidecars record 46 internal validation entries for both models;
the two additional benchmark targets are `1Y0Q_A` and `7O7Z_AH`.
Restrict each source's input files to its recorded membership for comparisons;
auto-discovery otherwise trains on every available non-held-out sample.

The delivered validation48/test36 generated-MSA set uses the **different**
`all_data_epoch45_best_1.pt`: select `configs/epoch45.json` (epoch45/step10791,
accumulation4, fourth source `nufolddb_relaxed_gap25`). Its checkpoint hash and
exact path-free args/config are recorded separately. An argument/log-matched
sidecar records 50,990 train, 46 internal validation and 36 test entries, with
four DDP ranks; its membership is saved separately from the primary split.
The external generated-MSA validation collection contains 48 entries.

Use the [sequence-overlap auditor](homology/README.md) with the original query
inputs to check training/validation/test homology at 80%, 85% and 90% identity.
Recorded ID exclusions alone do not certify homology independence.

## Train

From this release root, with all three input sources available:

```bash
# Audit first (no optimization).
bash examples/train.sh --audit_only
# One GPU, effective batch 2 * 1 * 32 = 64.
bash examples/train.sh
# Four GPUs via torchrun, effective batch 8 * 4 * 2 = 64.
NPROC=4 bash examples/train.sh
# Slurm: environment must already be activated; resources override directives.
sbatch --partition=gpua800 --qos=4a800 examples/run_a800.sbatch
# Epoch45 four-source setup (recorded effective batch 256).
PROFILE=configs/epoch45.json SOURCE_COUNT=4 NPROC=4 \
  BATCH_SIZE=16 GRAD_ACCUM=4 bash examples/train.sh
```

`PROFILE`, `SOURCE_COUNT`, `DATA_ROOT`, `OUT_DIR`, `TARGET_CACHE_DIR`, `BATCH_SIZE`, and `GRAD_ACCUM` are
environment overrides. Extra training flags are forwarded last. Reduce
micro-batch and increase accumulation when memory is limited; a global batch
match does not imply identical gradients or results. The launcher defaults to
bf16 and gradient checkpointing. Use `--amp_dtype none` for hardware without
bf16 support. The 4-GPU example is an adaptation, not the historical topology.

See [direct Python/torchrun and Slurm examples](examples/README.md) for launches
without the shell wrapper. Source4 uses `data/source4/` in the same layout as
above, or explicit `--dataset4_embedding_dir`, `--dataset4_attention_dir`,
`--dataset4_query_dir`, `--dataset4_bpp_dir`, `--dataset4_msa_dir` paths.

Targets are disk-cached automatically under the output directory unless an
explicit directory is given. DDP uses rank/source subdirectories. Do not enable
`--memory_target_cache` for large runs; `[L,L,25]` targets can exhaust CPU RAM.
Only use `--shared_target_cache` with a precomputed cache. Invalidate caches
after changing data or target settings; do not reuse unrelated experiment
caches. CPU target construction and attention copies still require substantial
RAM. Outputs include audits, splits, metrics, `last.pt` and validation-selected
`best.pt`. With no validation IDs, no validation-selected best is established.
Resume using `--resume runs/.../last.pt`; only load trusted pickle checkpoints.

## Export A New Checkpoint

No ONNX conversion or retraining is needed for the native PyTorch route:

```bash
python -m RethinkingMSA_1.export_adabm_predicted_stats \
  --checkpoint runs/primary_8k_1gpu/best.pt --out_dir runs/export \
  --embedding_dir data/source1/embedding --attention_dir data/source1/attention_map \
  --query_dir data/source1/query --bpp_dir data/source1/bpp \
  --msa_dir data/source1/msa --ids YOUR_ID
python -m pip install -r requirements-inverse.txt
python -m RethinkingMSA_1.run_adabm_inverse_from_predicted_stats \
  --stats_root runs/export --out_dir runs/inverse
```

The exporter reads an alignment to construct diagnostic targets. Predictions
condition on query/RESM/BPP, but this script is **not** an MSA-free input path.
For a novel query, a query-only `<id>.a3m` in `--msa_dir` satisfies the interface;
its target comparisons are not scientific evaluation. The downstream native
inverse stage reads only exported predicted statistics. Default inverse runs
are substantial (10000 chains, up to 1000 epochs); reduce budgets for checks.

This release has lightweight synthetic checks, not a verified full retraining.
Original project code is MIT; see [third-party boundaries](THIRD_PARTY.md).
