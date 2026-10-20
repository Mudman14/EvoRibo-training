# Training-query overlap audit

Compare each official validation/test query with the recorded training
membership at inclusive 80%, 85% and 90% identity. Counts report targets with
at least one hit and the number of training/held-out pairs.

The two memberships remain separate:

| Model | Train | Internal validation | Test |
| --- | ---: | ---: | ---: |
| Primary 8K | 8,296 | 46 | 36 |
| Epoch45 | 50,990 | 46 | 36 |

The audit evaluates the full official **48 validation + 36 test** queries.
`1Y0Q_A` and `7O7Z_AH` are additional official validation targets absent
from both recorded internal validation splits. Membership files contain IDs,
not query sequences.

## Run on the original HPC

Activate an environment with Python >=3.10, then install the audit dependency:

```bash
python -m pip install -r requirements-audit.txt
sbatch homology/run_audit_cpuhuge1t.sbatch
```

The wrapper uses `cpuhuge1t`, `46cores`, 46 CPUs and 32 GB RAM, and runs
both memberships. Submit from the checkout or set `RELEASE_ROOT`. Logs go to
`slurm-<jobid>.out`, avoiding the Slurm spool directory.

Default original query directories are:

- NuFold: `$HOME/RESM_query/input/dsdata/msa`
- REPA: `$HOME/RESM_round2_query/input/dsdata/msa`
- Round3: `$HOME/RESM_round3_query/input/dsdata/msa`
- Relaxed: `$HOME/RESM_relaxed_gap25_exclude_existing_training_1449190/input/dsdata/msa`

Held-out queries default to `$HOME/NuFold/val.fasta` and
`$HOME/NuFold/test.fasta`. Provide all 84 official queries; the two FASTAs
must have unique, non-conflicting IDs. The optional `HELDOUT_INPUT_DIR`
provides additional `<id>/<id>.fasta` queries. Empty FASTA environment
overrides disable that source when using per-ID files.

Paths/resources can be overridden with `QUERY_ROOT_NUFOLD`,
`QUERY_ROOT_REPA_ROUND1`, `QUERY_ROOT_NUFOLDDB_ROUND3`,
`QUERY_ROOT_NUFOLDDB_RELAXED_GAP25`, `HELDOUT_VAL_FASTA`,
`HELDOUT_TEST_FASTA`, `OUT_DIR`, `PYTHON` and `WORKERS`.
Results are retained in `homology/audit_outputs/job-<jobid>/` by default.
Worker count is capped at the allocated CPUs.

## Interpret the output

Each model writes `summary.json`, `threshold_hits.csv`,
`heldout_best_qualifying.csv`, `missing_training_ids.csv`, and
`resolved_training_queries.csv`. The last file contains query sequences
and local provenance: treat output folders as data, not source code.

Identity is canonical A/C/G/U matches divided by all global-alignment columns,
including gaps. Both sequences require >=80% residue coverage. Scores are
match 2, mismatch -1, gap-open -2 and gap-extension -0.5. The first optimal
alignment resolves ties; no reverse-complement or family annotation is used.
A safe length bound is the only prefilter.

Only best qualifying hits are reported, not nearest neighbors below 80%.
Missing query inputs produce **partial, lower-bound counts**. Zero observed
hits in a partial search do not establish zero overlap. The wrapper keeps both
runs' evidence and exits unsuccessfully if either audit is incomplete.

Identifier exclusion and sequence-homology exclusion are different checks.
This source release does not certify homology independence.
