#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Data paths are relative to the release root unless DATA_ROOT is absolute.
DATA_ROOT="${DATA_ROOT:-data}"
NPROC="${NPROC:-1}"
PROFILE="${PROFILE:-configs/primary_8k.json}"
SOURCE_COUNT="${SOURCE_COUNT:-3}"
case "$SOURCE_COUNT" in
  1|2|3|4) ;;
  *) echo "SOURCE_COUNT must be 1, 2, 3, or 4" >&2; exit 2 ;;
esac
RUN_NAME="$(basename "$PROFILE" .json)"
if [[ "$NPROC" == "1" ]]; then
  LAUNCH=(python train.py)
  BATCH_SIZE="${BATCH_SIZE:-2}"
  GRAD_ACCUM="${GRAD_ACCUM:-32}"
elif [[ "$NPROC" == "4" ]]; then
  LAUNCH=(torchrun --standalone --nnodes=1 --nproc_per_node=4 train.py)
  BATCH_SIZE="${BATCH_SIZE:-8}"
  GRAD_ACCUM="${GRAD_ACCUM:-2}"
else
  echo "NPROC must be 1 or 4" >&2
  exit 2
fi

ARGS=(--profile "$PROFILE")
for HELDOUT in val.fasta test.fasta; do
  if [[ ! -f "$DATA_ROOT/$HELDOUT" ]]; then
    echo "Missing held-out membership FASTA: $DATA_ROOT/$HELDOUT" >&2
    exit 2
  fi
done
for ((SOURCE=1; SOURCE<=SOURCE_COUNT; SOURCE++)); do
  PREFIX="dataset${SOURCE}_"
  if [[ "$SOURCE" == "1" ]]; then PREFIX=""; fi
  BASE="$DATA_ROOT/source${SOURCE}"
  ARGS+=(
    "--${PREFIX}embedding_dir" "$BASE/embedding"
    "--${PREFIX}attention_dir" "$BASE/attention_map"
    "--${PREFIX}query_dir" "$BASE/query"
    "--${PREFIX}bpp_dir" "$BASE/bpp"
    "--${PREFIX}msa_dir" "$BASE/msa"
  )
done
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
"${LAUNCH[@]}" "${ARGS[@]}" \
  --val_fasta "$DATA_ROOT/val.fasta" \
  --test_fasta "$DATA_ROOT/test.fasta" \
  --batch_size "$BATCH_SIZE" --grad_accum "$GRAD_ACCUM" \
  --out_dir "${OUT_DIR:-runs/${RUN_NAME}_${NPROC}gpu}" \
  --target_cache_dir "${TARGET_CACHE_DIR:-runs/${RUN_NAME}_${NPROC}gpu/target_cache}" \
  "$@"
