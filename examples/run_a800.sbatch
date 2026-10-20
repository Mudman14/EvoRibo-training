#!/usr/bin/env bash
#SBATCH --job-name=evoribo-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

# Override resources with sbatch flags, e.g. --partition=gpua800 --qos=4a800.
# SLURM copies this script into its spool: prefer the submit directory, not $0.
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT=""
for CANDIDATE in "${RELEASE_ROOT:-}" "$SUBMIT_DIR" "$SUBMIT_DIR/.." \
  "$SUBMIT_DIR/releases/EvoRibo-training" "$SCRIPT_DIR/.."; do
  if [[ -n "$CANDIDATE" && -f "$CANDIDATE/RethinkingMSA_1/train_direct_trunk_ddp.py" \
      && -f "$CANDIDATE/examples/train.sh" ]]; then
    ROOT="$(cd -- "$CANDIDATE" && pwd)"
    break
  fi
done
if [[ -z "$ROOT" ]]; then
  echo "Cannot locate release root; submit from the release or set RELEASE_ROOT." >&2
  exit 2
fi
cd "$ROOT"
export NPROC="${NPROC:-4}"
export DATA_ROOT="${DATA_ROOT:-data}"
export OUT_DIR="${OUT_DIR:-runs/slurm_${SLURM_JOB_ID:-manual}}"
export TARGET_CACHE_DIR="${TARGET_CACHE_DIR:-$OUT_DIR/target_cache}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
# Activate the desired Python/CUDA environment before sbatch submission.
bash examples/train.sh "$@"
