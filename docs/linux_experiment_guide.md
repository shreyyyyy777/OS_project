# Linux experiment guide

Run controlled experiments on a dedicated Linux host, not the macOS development
machine. Linux cgroup v2 is required for CPU-weight/CPU-limit actions; the
portable collector can still support development on macOS but cannot claim
equivalence to LAPS' BPF telemetry.

## First experiment

Use a disposable foreground probe and a disposable background compilation.
Start with the four baseline rows generated from `experiments/example.toml`:

1. `linux_default` - collect only.
2. `static_nice_plus_5` - one explicitly reviewed static adjustment.
3. `bgopt_heuristic` - BGOpt without a trained model, always dry-run first.
4. `bgopt_ml_assisted` - only after training on separate experiments.

Run at least five repetitions per baseline. Capture the same foreground probe,
background work unit, machine/kernel configuration, input size, and sample
interval every time. Never use a row-level train/test score as an experiment
result.

## Safe collection and control

```bash
bgopt collect --experiment-id fg_web_bg_compile_001 \
  --scenario web_browsing_plus_kernel_compile \
  --output data/raw/fg_web_bg_compile_001.jsonl --interval 2 --samples 120

bgopt control --config config/example.toml \
  --log logs/fg_web_bg_compile_001.jsonl --interval 2 --iterations 120
```

Both commands are observational/dry-run by default. The controller records
`NO_ACTION`, `NICE_PLUS_5`, `NICE_PLUS_10`, `REDUCE_CPU_WEIGHT`, `CPU_LIMIT`,
or a guarded `RESTORE_PRIORITY` proposal. It never kills a process or touches
`sched_ext`.

Real application is a later, one-PID-at-a-time Linux experiment. It requires
`--apply --allow-pid PID`; cgroup work also needs a pre-existing dedicated
cgroup, `--allow-cgroup`, and `--cgroup-path`. Restoring a priority additionally
requires `--allow-restore`. Read the audit log between every trial.
