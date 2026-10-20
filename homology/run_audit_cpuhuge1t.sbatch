#!/usr/bin/env bash
#SBATCH --job-name=evoribo-homology
#SBATCH --partition=cpuhuge1t
#SBATCH --qos=46cores
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=46
#SBATCH --mem=32G
#SBATCH --output=slurm-%j.out

set -euo pipefail

# Slurm may execute a spool copy: never resolve repository paths from $0.
base=${RELEASE_ROOT:-${SLURM_SUBMIT_DIR:-}}
if [[ -z "$base" ]]; then
    echo "ERROR: set RELEASE_ROOT or submit from the release, its homology directory, or the EvoNA workspace." >&2
    exit 2
fi
release_root=
for candidate in "$base" "$base/releases/EvoRibo-training"; do
    if [[ -d "$candidate/homology" && -f "$candidate/README.md" ]]; then
        release_root=$(cd "$candidate" && pwd -P)
        break
    fi
done
if [[ -z "$release_root" && "${base%/}" == */homology ]]; then
    candidate="${base%/}/.."
    if [[ -d "$candidate/homology" && -f "$candidate/README.md" ]]; then
        release_root=$(cd "$candidate" && pwd -P)
    fi
fi
if [[ -z "$release_root" ]]; then
    echo "ERROR: cannot locate EvoRibo-training from '$base'; set RELEASE_ROOT to the release checkout." >&2
    exit 2
fi
RELEASE_ROOT=$release_root
cd "$RELEASE_ROOT"

PYTHON=${PYTHON:-python}
WORKERS=${WORKERS:-46}
allocated_cpus=${SLURM_CPUS_PER_TASK:-46}
if [[ ! "$WORKERS" =~ ^[1-9][0-9]*$ || ! "$allocated_cpus" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: WORKERS and SLURM_CPUS_PER_TASK must be positive integers." >&2
    exit 2
fi
workers=$(( WORKERS < allocated_cpus ? WORKERS : allocated_cpus ))
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

auditor="$RELEASE_ROOT/homology/audit_homology_thresholds.py"
if [[ ! -f "$auditor" ]]; then
    echo "ERROR: missing portable auditor: $auditor. Use a complete release checkout." >&2
    exit 2
fi
if ! "$PYTHON" -c 'import Bio; from Bio import Align, SeqIO; print("Biopython", Bio.__version__)'; then
    echo "ERROR: PYTHON='$PYTHON' must name one executable in an environment with Biopython >=1.85,<2; activate that environment before submission." >&2
    exit 2
fi

PRIMARY8K_MEMBERSHIP=${PRIMARY8K_MEMBERSHIP:-$RELEASE_ROOT/homology/primary8k_training_membership.json}
EPOCH45_MEMBERSHIP=${EPOCH45_MEMBERSHIP:-$RELEASE_ROOT/homology/epoch45_training_membership.json}
BENCHMARK_MANIFEST=${BENCHMARK_MANIFEST:-$RELEASE_ROOT/homology/benchmark_manifest.csv}
QUERY_ROOT_NUFOLD=${QUERY_ROOT_NUFOLD:-$HOME/RESM_query/input/dsdata/msa}
QUERY_ROOT_REPA_ROUND1=${QUERY_ROOT_REPA_ROUND1:-$HOME/RESM_round2_query/input/dsdata/msa}
QUERY_ROOT_NUFOLDDB_ROUND3=${QUERY_ROOT_NUFOLDDB_ROUND3:-$HOME/RESM_round3_query/input/dsdata/msa}
QUERY_ROOT_NUFOLDDB_RELAXED_GAP25=${QUERY_ROOT_NUFOLDDB_RELAXED_GAP25:-$HOME/RESM_relaxed_gap25_exclude_existing_training_1449190/input/dsdata/msa}
# Empty FASTA overrides intentionally disable that input for per-ID-only runs.
HELDOUT_VAL_FASTA=${HELDOUT_VAL_FASTA-$HOME/NuFold/val.fasta}
HELDOUT_TEST_FASTA=${HELDOUT_TEST_FASTA-$HOME/NuFold/test.fasta}
HELDOUT_INPUT_DIR=${HELDOUT_INPUT_DIR:-}

OUT_DIR=${OUT_DIR:-$RELEASE_ROOT/homology/audit_outputs}
job_id=${SLURM_JOB_ID:-manual-$(date -u +%Y%m%dT%H%M%SZ)-$$}
run_dir="$OUT_DIR/job-$job_id"
mkdir -p "$run_dir"
run_dir=$(cd "$run_dir" && pwd -P)
common_args=(
    --benchmark_manifest "$BENCHMARK_MANIFEST"
    --workers "$workers"
)
if [[ -n "$HELDOUT_VAL_FASTA" ]]; then
    common_args+=(--heldout_fasta "$HELDOUT_VAL_FASTA")
fi
if [[ -n "$HELDOUT_TEST_FASTA" ]]; then
    common_args+=(--heldout_fasta "$HELDOUT_TEST_FASTA")
fi
if [[ -n "$HELDOUT_INPUT_DIR" ]]; then
    common_args+=(--heldout_input_dir "$HELDOUT_INPUT_DIR")
fi
printf 'Release: %s\nPython: %s\nWorkers: %s (allocated: %s)\nResults: %s\n' \
    "$RELEASE_ROOT" "$PYTHON" "$workers" "$allocated_cpus" "$run_dir"

failed=0
run_audit() {
    local name=$1 membership=$2 output="$run_dir/$1" audit_rc=0
    local sources source
    local -a root_args=()
    mkdir -p "$output"
    echo "Starting $name using $membership"
    if ! sources=$("$PYTHON" - "$membership" 2>"$output/audit.stderr.log" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
try:
    membership = json.loads(path.read_text(encoding="utf-8-sig"))
    members = membership.get("training_members") if isinstance(membership, dict) else None
    if not isinstance(members, list) or not members:
        raise ValueError("training_members must be a nonempty list")
    sources = set()
    for member in members:
        source = member.get("source") if isinstance(member, dict) else None
        if not isinstance(source, str) or not source or "\n" in source or "\r" in source:
            raise ValueError("each training member requires a nonempty single-line source")
        sources.add(source)
except (OSError, ValueError) as exc:
    print(f"ERROR: cannot read membership sources from {path}: {exc}", file=sys.stderr)
    sys.exit(1)
print("\n".join(sorted(sources)))
PY
    ); then
        echo "ERROR: $name membership is missing or malformed; inspect $output/audit.stderr.log and repair $membership. Continuing with the other membership." >&2
        failed=1
        return 0
    fi
    while IFS= read -r source; do
        source=${source%$'\r'}
        case "$source" in
            nufold) root_args+=(--query_root "nufold=$QUERY_ROOT_NUFOLD") ;;
            repa_round1) root_args+=(--query_root "repa_round1=$QUERY_ROOT_REPA_ROUND1") ;;
            nufolddb_round3) root_args+=(--query_root "nufolddb_round3=$QUERY_ROOT_NUFOLDDB_ROUND3") ;;
            nufolddb_relaxed_gap25) root_args+=(--query_root "nufolddb_relaxed_gap25=$QUERY_ROOT_NUFOLDDB_RELAXED_GAP25") ;;
            *) echo "WARNING: $name source '$source' has no configured query root; the auditor will report unresolved members and incomplete status if their queries cannot resolve." >&2 ;;
        esac
    done <<<"$sources"
    "$PYTHON" "$auditor" --checkpoint_audit "$membership" \
        "${common_args[@]}" "${root_args[@]}" --out_dir "$output" \
        >"$output/audit.stdout.log" 2>"$output/audit.stderr.log" || audit_rc=$?
    if (( audit_rc != 0 )); then
        echo "ERROR: $name auditor exited $audit_rc; inspect $output/audit.stderr.log (membership, query roots, and heldout FASTAs)." >&2
        failed=1
    fi
    if ! "$PYTHON" - "$output/summary.json" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
try:
    summary = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(summary, dict):
        raise ValueError("expected a JSON object")
except (OSError, ValueError) as exc:
    print(f"ERROR: cannot validate {path}: {exc}; inspect audit.stderr.log and rerun.", file=sys.stderr)
    sys.exit(1)
print(f"{path.parent.name}: status={summary.get('status')}, "
      f"resolved training={summary.get('resolved_training_queries')}/{summary.get('training_members')}, "
      f"missing heldout={len(summary.get('missing_heldout_queries', []))}")
if summary.get("status") != "complete":
    print(f"ERROR: {path} is not complete. Resolve missing_training_ids.csv and "
          "summary.json missing_heldout_queries using the configured roots/FASTAs. "
          "Retained counts are lower bounds, not evidence of independence.", file=sys.stderr)
    sys.exit(1)
PY
    then
        failed=1
    fi
    echo "Retained $name results/logs: $output"
}

# Do not short-circuit: retain evidence for both fixed memberships.
run_audit primary8k "$PRIMARY8K_MEMBERSHIP"
run_audit epoch45 "$EPOCH45_MEMBERSHIP"
if (( failed )); then
    echo "ERROR: at least one audit failed or is incomplete. Both run directories are retained under $run_dir; repair inputs and rerun." >&2
    exit 1
fi
echo "Both audits complete. Results retained under $run_dir"
