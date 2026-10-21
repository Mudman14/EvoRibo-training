# Training Examples

Run commands from the repository root after activating your Python environment.

```bash
SOURCE_COUNT=1 bash examples/train.sh --audit_only
SOURCE_COUNT=1 bash examples/train.sh
SOURCE_COUNT=1 NPROC=4 bash examples/train.sh
```

Set `SOURCE_COUNT=3` to use `data/source1/` through `data/source3/`.
Override `DATA_ROOT`, `OUT_DIR`, `BATCH_SIZE`, `GRAD_ACCUM` as needed.
The launcher defaults to one GPU. For Slurm, activate your environment, then:

```bash
SOURCE_COUNT=1 sbatch --partition=YOUR_PARTITION --qos=YOUR_QOS examples/run_a800.sbatch
```

The Slurm script requests four GPUs; adjust its resource directives for your cluster.
Submit from the repository root, or set `RELEASE_ROOT` to its absolute path.
