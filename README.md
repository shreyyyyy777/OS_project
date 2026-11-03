# BGOpt: safe ML-assisted background-process optimizer

BGOpt is a user-space research prototype for Linux process optimisation.  It takes
the useful *architecture* of LAPS—observe, classify, then choose from a small
policy table—without changing the kernel or trying to run LAPS's scheduler
switcher.

The key design distinction is deliberate:

| LAPS | BGOpt |
| --- | --- |
| Classifies machine/workload scenes | Classifies per-process workload behaviour |
| Maps a scene to an expert **scheduler** | Maps a safe process state to a **user-space action** |
| Uses BPF/kernel scheduling infrastructure | Uses `psutil`; no kernel changes |

`background_cpu`, `background_io`, `background_network`, `background_mixed`,
`memory_heavy`, `idle`, and `interactive`
are workload labels. They are **not** actions.  BGOpt independently evaluates
pressure and process safety before proposing `no_op`, a lower-priority `nice`
value, or (on Linux) a CPU weight for an explicitly supplied, pre-existing
cgroup.

## Safety contract

* The default controller mode is **dry-run**. It writes a JSONL audit log of
  proposed actions and changes nothing.
* It never kills, suspends, or restarts a process.
* A real `nice` change needs all of `--apply`, `--allow-pid PID`, a same-user
  process, and an action that only *increases* nice (lowers priority).
* A cgroup action additionally needs `--allow-cgroup`, `--cgroup-path PATH`,
  Linux cgroup v2, and an existing writable cgroup supplied by you. The tool
  will never create cgroups or use a broad system cgroup by default.
* Terminal foreground / likely interactive processes are always no-op. On Linux, foreground status uses the controlling TTY's foreground process group, so a shell-launched background job remains eligible.
* Configured protected process names also protect matching processes' observed descendant tree for each telemetry snapshot when protect_descendants = true (the default).

## Project layout

```text
config/                    Conservative runtime configuration
experiments/               Review-only experiment matrices and scenario configs
data/raw/                  Local telemetry captures (ignored by git)
data/processed/            Prepared labelled feature sets (ignored by git)
models/                    Trained classifiers (ignored by git)
reports/                   Training/evaluation reports (ignored by git)
logs/                      Controller audit logs (ignored by git)
src/bgopt/                 Collector, LAPS adapter, ML, policy, controller
tests/                     Pure policy and artifact-adapter tests
```

## Quick start (dry-run only)

Use Python 3.10–3.13. The environment used to create this project has Python
3.14, while current binary builds of scikit-learn may lag that version; create
an environment with a supported Python if installation complains.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
# For the tested, reproducible Python 3.12 environment instead:
# python -m pip install -r requirements-lock.txt
python -m pip install -e .

# Collect 60 samples of current-user process telemetry, five seconds apart.
bgopt collect --output data/raw/telemetry.jsonl --interval 5 --samples 60

# Create a small labelling sheet. Fill workload_label with one of:
# interactive, background_cpu, background_io, background_network,
# background_mixed, memory_heavy, idle
bgopt label-template --telemetry data/raw/telemetry.jsonl \
  --output data/raw/labels.csv

# Prepare and train only after adding your own labels/outcomes.
bgopt preprocess --input data/raw/telemetry.jsonl --labels data/raw/labels.csv \
  --output data/processed/training.csv
bgopt train --dataset data/processed/training.csv \
  --model models/workload_classifier.joblib --report reports/training.json

# Observe, classify, estimate pressure, and log proposed actions. No changes.
bgopt control --config config/example.toml --model models/workload_classifier.joblib \
  --log logs/controller.jsonl --interval 5 --iterations 24

# Summarise the audit log and, when labels exist, evaluate the model.
bgopt benchmark --controller-log logs/controller.jsonl --report reports/benchmark.json
bgopt evaluate --dataset data/processed/training.csv \
  --model models/workload_classifier.joblib --report reports/evaluation.json
```

The collector itself is usable before training. In that case BGOpt uses a
clearly-labelled transparent fallback classifier so that the observation →
decision pipeline can be tested; it is not a trained result.

## Using the downloaded LAPS artifact

Copy or extract the LAPS archive somewhere local, then inspect it without
altering it:

```bash
bgopt inspect-laps --artifact-root /absolute/path/to/LAPS-Code\&Dataset \
  --output reports/laps_inventory.json

# The released features.pkl and metadata.pkl are pickle files. Profile them
# only after confirming you trust the downloaded artifact.
bgopt profile-laps-dataset \
  --dataset-dir /absolute/path/to/LAPS-Code\&Dataset/dataset \
  --feature-map /absolute/path/to/LAPS-Code\&Dataset/LAPS/config/feature_map.txt \
  --output reports/laps_dataset_profile.json --trust-pickle
```

The command inventories README/paper/config/model/data candidates and reads a
`feature_map.txt` if present. Its report includes the conservative feature
crosswalk used here. LAPS' full 150-feature BPF model is intentionally not
treated as a drop-in per-daemon action model: its labels describe workload
scenes, its model must match its feature ordering, and its policy table maps
to expert schedulers. Use it as a source of feature semantics and experiment
design, then collect BGOpt's own process-level labels and action outcomes.

## Labels and outcomes to collect

For each sampled PID, label workload behaviour (`workload_label`) and record
the outcome of a trial separately: foreground latency, background completion
time, CPU/memory pressure, and whether a proposed action was accepted. Do not
train on a proposed action as if it were the workload label. That separation
keeps model mistakes from being mistaken for scheduling truth.

Start with no-op versus nice in controlled, disposable workloads. Compare
matched baseline and treatment runs. Cgroup CPU weighting is a later Linux-only
experiment because it moves a process into a cgroup and needs system setup.

## Controlled experiment workflow

`bgopt plan-experiment` creates a reviewable matrix for Linux default,
static-nice, BGOpt heuristic, and BGOpt ML-assisted baselines. It never starts
a command. `outcome-template` creates blank measurement fields; you fill them
only with observed foreground latency/performance, background progress or
completion time, starvation duration, and controller overhead.

```bash
bgopt plan-experiment --config experiments/example.toml --output reports/plan.json
bgopt outcome-template --plan reports/plan.json --output data/processed/outcomes.csv
# After completed repetitions only:
bgopt summarise-outcomes --outcomes data/processed/outcomes.csv --report reports/outcome_summary.json
```

Use the grouped split automatically inferred from `experiment_id`, `group`, or
`scenario` when training. A row-level fallback is explicitly marked as a
leakage-risk development check, not a scientific result. See
`docs/linux_experiment_guide.md` and `docs/laps_feature_crosswalk.md`.

## Optional real application (only after review)

The controller remains dry-run unless `--apply` is given. A deliberately
narrow nice application looks like:

```bash
bgopt control --config config/example.toml --log logs/applied.jsonl \
  --iterations 1 --apply --allow-pid 12345
```

For cgroups, first create and verify a dedicated cgroup yourself. Then pass
both `--allow-cgroup` and its exact pre-existing path. Read every audit log
before widening a trial.

## LAPS material inspected

The upstream LAPS README describes an Observe–Infer–Switch design, BPF/system
feature extraction, a 150-feature/28-class classifier, and a Policy Mapping
Table which selects expert schedulers. This project was scaffolded before a
local artifact was available in this workspace; run `inspect-laps` after you
place the downloaded archive here to make the inventory reproducible.

## Experiment 001 evaluation harness

`run-evaluation` records one append-only JSONL result for the controlled foreground-probe plus disposable compile-workload scenario. The foreground metric is fixed-interval scheduler lag: a small SHA-256 probe is scheduled every 50 ms and its p50/p95/max deadline lag is recorded. This is deterministic enough for matched repetitions and does not rely on manual browser activity. The background metric is the workload's natural completion time, with run-normalized throughput also recorded. Each record includes host CPU, normalized load, memory, swap, duration, exit status, and BGOpt decision/intervention fields.

During measurement validation, only `linux_default` is executable. `static_nice_plus_5` and both BGOpt modes are explicitly refused by the runner before starting a workload; there is no `--apply` option and the runner never changes nice values, cgroups, scheduler settings, or process lifetimes. A single safe pilot is:

    .venv/bin/bgopt run-evaluation --mode linux_default --repetition 1

Results append to `data/processed/experiment_001_evaluation.jsonl`; the existing planned outcomes CSV is deliberately not overwritten.

## BGOpt-native ML data

LAPS is used only as an external workload-taxonomy and feature-design reference. Its BPF feature matrix and scene labels are not BGOpt action labels and are not loaded into the psutil-compatible runtime classifier. The deployable classifier is trained only on BGOpt-native labeled telemetry containing the exact runtime MODEL_FEATURES, workload_label, and independent group/experiment identifiers.

The relationship is: LAPS -> workload taxonomy/feature-design reference; BGOpt-native labeled telemetry -> runtime classifier; runtime classifier -> workload characterization; pressure + protection + classification -> policy; policy -> optimization action.

A small reproducible collection protocol is tools/collect_bgopt_native_training.py; it launches finite controlled idle, interactive, CPU, memory, and I/O workloads and records only the collector's runtime schema. Train with bgopt train --dataset data/processed/bgopt_native_training.csv --model models/bgopt_native_workload_classifier.joblib --report reports/bgopt_native_training_report.json --group-column group.

## Research dashboard

Launch the professor-facing, read-only dashboard from the project environment:

```bash
.venv/bin/python -m streamlit run dashboard/app.py
```

It presents Experiment 001 final artifacts and offers a one-shot live telemetry snapshot. The live page uses the existing collector, classifier, protection annotation, pressure estimator, and policy logic only to display suggestions. It never invokes BGOpt action application, modifies priorities or cgroups, starts an optimisation loop, or writes experiment artifacts.
