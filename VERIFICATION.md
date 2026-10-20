# Release Verification

Validation performed in the requested DiffMSA environment. This is the release
validation environment, not a reconstructed historical training environment:
Python 3.14.2, PyTorch 2.11.0.dev20260125+cu128, NumPy 2.4.1,
matplotlib 3.10.8, pytest 9.0.3, optional adabmDCA 0.7.8.

## Passed

```text
conda activate DiffMSA
python -m pytest -q -p no:cacheprovider
21 passed (final run: 29.17s)

python -m compileall -q RethinkingMSA_1 inverse_coevo_sampler.py train.py tests
bash -n examples/train.sh examples/run_a800.sbatch
```

The suite verifies local imports, CLI help, both checkpoint profiles, primary
membership/checksums, source hash integrity, AST dependency closure, legacy
MI/APC target construction, and disk-target reuse without memory retention.
The missing-holdout regression was reproduced before fixing it; supplied
missing validation/test FASTAs or test directories now fail before splitting.

An isolated copy outside the original source tree executes real tiny CPU
training with the factorized conditional velocity-flow loss, validation,
`best.pt` and `last.pt` output. Resume advances epoch/global_step from 1/1 to
2/2, preserves optimizer step counters, and changes model weights. Native
export then reads that resumed checkpoint and emits finite `fi`/`fij` tensors
in `-ACGU` order. Optional inverse sampling was exercised here with 16 chains,
one fit epoch and four generated sequences using adabmDCA 0.7.8. No scientific
convergence is implied. All fixtures and outputs are temporary synthetic data.
Actual saved checkpoints load successfully under this PyTorch version's
`torch.load` defaults, including the native resume/export paths.

The workspace-only helper `releases/prepare_training_release.py` verifies all
17 copied source files against the source snapshot with only the enumerated
portable-path/cache-retention changes. The manifest stores both SHA-256 hashes.
Both Bash scripts pass syntax checks in Git Bash. A mocked launcher confirms
the Slurm script selects the release root from its submit directory and
forwards `--audit_only`; no Slurm job was submitted.

## Not Verified

- Full historical retraining, original-data equivalence, convergence or bitwise
  checkpoint reproduction. Neither full datasets nor weights are bundled.
- Original epoch45 training membership/GPU topology; none is inferred. Primary
  split membership is separate and must not be reused as epoch45 provenance.
- Real one/four-GPU optimization, NCCL collectives, A800 memory/performance,
  scheduler resource availability, or downstream RNA structure accuracy.
- A clean dependency installation across the declared version ranges. These
  ranges are compatibility intent, not a tested matrix or historical lockfile.
- Separate Ruff lint and mypy typecheck: both were attempted but are not
  installed in the validation environment. Compilation, AST closure checks,
  source-integrity checks and runtime tests passed; no lint/typecheck success
  is claimed.
