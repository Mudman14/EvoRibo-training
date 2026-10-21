# EvoRibo Training

Training code for predicting RNA coevolution maps and single/pair statistics
from RESM features, query sequence and user-provided BPP.

## Install

Use Python 3.10+ and install a CUDA-compatible PyTorch build first.

```bash
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

## Prepare Data

For each input source, use this layout:

```text
data/source1/embedding/<id>_emb.npy      # [L,1280] or [1,L,1280]
data/source1/attention_map/<id>_atp.npy  # [660,L,L] or [L,L,660]
data/source1/query/<id>.fasta
data/source1/bpp/<id>.prob              # dense LxL or i j probability
data/source1/msa/<id>.a3m               # aligned target MSA, query first
data/val.fasta                         # validation IDs in FASTA headers
data/test.fasta                        # test IDs in FASTA headers
```

All inputs for an ID must have the same query length. RESM features and BPP
must be prepared separately; data and pretrained weights are not included.
Additional sources use the same layout under `source2/`, `source3/`, etc.

## Train

```bash
# Check inputs first.
SOURCE_COUNT=1 bash examples/train.sh --audit_only

# Single GPU.
SOURCE_COUNT=1 bash examples/train.sh

# Four GPUs.
SOURCE_COUNT=1 NPROC=4 bash examples/train.sh

# Slurm: activate your environment and set your cluster partition/QOS.
sbatch --partition=YOUR_PARTITION --qos=YOUR_QOS examples/run_a800.sbatch
```

The Slurm example uses three sources by default; set `SOURCE_COUNT=1` for one.
Override paths and resources with `DATA_ROOT`, `OUT_DIR`, `SOURCE_COUNT`,
`BATCH_SIZE` and `GRAD_ACCUM`. Model/loss settings are in
`configs/primary_8k.json`; extra CLI flags override them.

Resume with `bash examples/train.sh --resume runs/YOUR_RUN/last.pt`
using the same data and source configuration. Training saves metrics,
`last.pt` and validation-selected `best.pt`.
Targets use a disk cache. Do not enable `--memory_target_cache` for large runs.
Use `--amp_dtype none` if your GPU does not support bf16.

## Generate MSAs

```bash
python -m RethinkingMSA_1.export_adabm_predicted_stats \
  --checkpoint runs/YOUR_RUN/best.pt --out_dir runs/export \
  --embedding_dir data/source1/embedding --attention_dir data/source1/attention_map \
  --query_dir data/source1/query --bpp_dir data/source1/bpp \
  --msa_dir data/source1/msa --ids YOUR_ID

python -m pip install -r requirements-inverse.txt
python -m RethinkingMSA_1.run_adabm_inverse_from_predicted_stats \
  --stats_root runs/export --out_dir runs/inverse
```

For a novel query, supply a query-only `<id>.a3m` for the export interface.
Inverse sampling uses the exported predicted statistics, not original-MSA
statistics. Generated alignments are written to `samples.a3m`.

Optional checks: `pip install -r requirements-dev.txt`, then `python -m pytest -q`.
See [THIRD_PARTY.md](THIRD_PARTY.md) for external software licenses.
