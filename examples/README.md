# Direct Launches

Run from the release root. These single-source examples exercise the real
training route, not the complete historical three/four-source data mixture.
Native module entry points `RethinkingMSA_1.train_direct_trunk` and
`RethinkingMSA_1.train_direct_trunk_ddp` also remain available without profiles.

```bash
python train.py --profile configs/primary_8k.json --resm_dir data/resm \
  --query_dir data/query --bpp_dir data/bpp --msa_dir data/msa \
  --val_fasta data/val.fasta --test_fasta data/test.fasta \
  --batch_size 2 --grad_accum 32 --out_dir runs/one \
  --target_cache_dir runs/one/target_cache
torchrun --standalone --nnodes=1 --nproc_per_node=4 train.py \
  --profile configs/primary_8k.json --resm_dir data/resm \
  --query_dir data/query --bpp_dir data/bpp --msa_dir data/msa \
  --val_fasta data/val.fasta --test_fasta data/test.fasta \
  --batch_size 8 --grad_accum 2 --out_dir runs/four \
  --target_cache_dir runs/four/target_cache
```

Add `--dataset2_*`, `--dataset3_*` paths for the primary mixture. For epoch45,
select `--profile configs/epoch45.json` and supply `--dataset4_embedding_dir`,
`--dataset4_attention_dir`, `--dataset4_query_dir`, `--dataset4_bpp_dir`, and
`--dataset4_msa_dir`, with name `nufolddb_relaxed_gap25` already in the profile.
The epoch45 recorded batch16/accumulation4 does not establish GPU topology or
global batch. The shell wrapper's smaller defaults are memory adaptations:
set `BATCH_SIZE` and `GRAD_ACCUM` explicitly for your intended experiment.

## Slurm

Activate the intended Python/CUDA environment before submitting. Override
resource directives through scheduler flags, not literal shell variables in
`#SBATCH` lines. For example:

```bash
sbatch --partition=gpua800 --qos=4a800 --gres=gpu:4 --mem=360G \
  --cpus-per-task=16 --time=48:00:00 examples/run_a800.sbatch
```

The script detects the release root from `SLURM_SUBMIT_DIR` (release root,
examples directory, or enclosing EvoNA workspace) or `RELEASE_ROOT`.
`slurm-%j.out` is written directly in the scheduler submit directory.
Set `NPROC=1` when requesting one GPU. Environment overrides for data/profile/
output/cache are forwarded. For example, `PROFILE=configs/epoch45.json
SOURCE_COUNT=4 BATCH_SIZE=16 GRAD_ACCUM=4 sbatch ...` selects the epoch45
four-source setup without asserting a historical world size.
