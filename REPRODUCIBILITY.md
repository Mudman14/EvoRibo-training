# Provenance And Limits

## Primary Checkpoint Versus Defaults

`configs/primary_8k.json` records non-path arguments and encoder configuration
read from local `8k_epoch75_best.pt`: epoch 75, global optimizer step 9750.
Its training logs record **2 GPUs**, batch 16 per GPU, accumulation 2, effective
global batch 64. The epoch budget was 100; epoch 75 is the selected artifact,
not evidence that training was configured to stop at 75. No checkpoint is shipped.

| Setting | Primary checkpoint | Current training CLI default | Config dataclass default |
| --- | --- | --- | --- |
| RESM single/pair channels | 1280 / 660 | infer / infer | 640 / 600 |
| Trunk single/pair width | 512 / 256 | 512 / 256 | 256 / 128 |
| Blocks / recycles | 12 / 3 | 12 / 1 | 8 / 1 |
| Batch / accumulation | 16 / 2 per GPU | 2 / 16 per GPU | n/a |
| Validation interval | 5 | 10 | n/a |
| Target row cap | 2001 | DDP 2001, single 200 | n/a |
| Stats head | factorized conditional flow | same | n/a |
| Flow objective / sample steps | velocity / 4 | same | n/a |
| Gradient checkpointing / attention chunk | on / 64 | on / 64 | off / 0 |

The checkpoint records AdamW lr/weight decay 1e-4, seed 7, max L 200,
adabm-style targets, sequence reweighting at identity >0.8, duplicate removal,
and default pseudocount 1/Meff. Statistics alphabet is `-ACGU`. The JSON retains
all recorded non-path arguments, including options unused by this head; their
presence should not be interpreted as evidence every loss option is active.
The native loss implementation is authoritative. Options added since the
checkpoint (for example evaluation step sweeps and scalar-consistency warmup)
use the current source defaults, not reconstructed historical values.

The checkpoint contains no `split` field. The separately audited,
argument-matched metrics sidecar supplies training membership in `configs/recorded/`:
8296 train (2513 nufold, 3986 repa_round1, 1797 nufolddb_round3), 46 internal
validation and 36 test members, with no raw train-ID/held-out overlap. The official
benchmark validation48/test36 list is separate; `1Y0Q_A` and `7O7Z_AH` are absent
from both models' internal validation splits. The sanitized record includes
checkpoint and sidecar SHA-256 identifiers. Membership does not establish
homology independence. Actual data/features, historical environment, RNG
state, and code revision remain unbundled. Historical query resolution was
incomplete outside the NuFold source; consult the record's limitations before
claiming dataset equivalence. Current source may include later fixes;
metadata plus source is not a bitwise replay.
The one/four-GPU examples preserve nominal batch 64 but differ from the
two-GPU run in accumulation/grouping and can produce different results.

This primary configuration belongs specifically to `8k_epoch75_best.pt` and the
primary paper test36 setup. The latest sampled validation/test84 generated-MSA
evaluation instead uses `all_data_epoch45_best_1.pt`. Its separately extracted
`configs/epoch45.json` records epoch45, step10791, batch16, accumulation4, and
the fourth source `nufolddb_relaxed_gap25`. Exact args/config/hash are in
`configs/recorded/epoch45_checkpoint_record.json`. This artifact has no embedded
split. A separately saved sidecar matches its exact arguments and training-log
prefix and records 50,990 training members: 2,513 NuFold, 3,986 REPA, 1,797
NuFoldDB round3, and 42,694 relaxed-gap25. Its membership is preserved in
`configs/recorded/epoch45_split_membership.json`; the recorded DDP world size is
four, so batch16/accumulation4 gives effective batch256. The internal validation
split contains 46 IDs, versus 48 externally evaluated validation targets:
`1Y0Q_A` and `7O7Z_AH` are absent from the internal validation split.
ID exclusions do not establish homology separation; see `homology/`.
Do not transfer the primary split membership to epoch45. Both profiles use the
same recorded trunk widths/blocks/recycles and conditional velocity-flow head,
but matching architecture does not make their training provenance equivalent.
No weights for either artifact are included in this training release.

## Source Boundary

Copied core: `RethinkingMSA_1/{__init__,config,data_utils,features,modules,
encoder,heads,flow,stat_targets,resm_dataset,train_resm_smoke,
train_direct_trunk,train_direct_trunk_ddp}.py`.
Native checkpoint export/inverse entry points are also included:
`export_adabm_predicted_stats.py` and `run_adabm_inverse_from_predicted_stats.py`.
Their local closure is the same core package; inverse additionally requires
the separately installed adabmDCA package.

Lazy legacy target construction imports `inverse_coevo_sampler.py` from the
repository root. Its lazy mi3 import requires
`RethinkingMSA_1/mi3_potts_inverse.py`; both complete files are included under
their original relative paths rather than replacing helpers with approximations.
This is why matplotlib remains a dependency even though the primary target
builder itself needs only NumPy/PyTorch. No external source directories,
manuscripts, solver binaries, experiment outputs, or history were staged.

Release-only changes to copied source:

- `train_direct_trunk.py`, `train_resm_smoke.py`: replace four machine-local
  input defaults with `data/resm`, `data/query`, `data/bpp`, `data/msa`.
- `inverse_coevo_sampler.py`: replace the machine-local MSA CLI default.
- `resm_dataset.py`: default `cache_targets=False` instead of `True`, avoiding
  unbounded in-memory target retention for single-GPU/native callers as well
  as DDP. Explicit small-run opt-in remains available in Python/DDP.
  Explicitly supplied missing held-out FASTAs/directories also raise an error
  instead of silently disabling validation/test exclusions.
- `export_adabm_predicted_stats.py`: portable input defaults, no implicit test
  directory restriction, and disabled in-memory target retention. Diagnostics
  still read an alignment. No prediction/target-construction algorithm changed.

No tensor math, model architecture, target convention, loss, optimizer, or
sampling implementation was changed. The new `train.py` only expands recorded
profile arguments before the existing DDP main, including in world-size-one
mode. `SOURCE_MANIFEST.json` records copied-file hashes and release differences.
`.gitattributes` disables Git text normalization to preserve those byte hashes
and the recorded metadata checksums. Bash scripts are stored with LF endings.

## Verification Scope

The release tests check import resolution from an isolated copy, native CLI
help, profile/default distinctions, local dependency closure, legacy target
construction, disk-cache reuse without in-memory accumulation, and a tiny CPU
CLI training update with the real conditional-flow loss and checkpoint output,
resume/optimizer restoration, and native export. An optional tiny native inverse
check runs when adabmDCA is installed.
All fixtures are synthetic and generated at test time in temporary directories.
These checks do not establish scientific reproduction, training convergence,
exact checkpoint compatibility, multi-GPU collectives, performance, or downstream
structure-prediction improvements. See `VERIFICATION.md` for the actual run.
