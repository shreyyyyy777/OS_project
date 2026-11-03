# BGOpt experiment protocol

## Components and their responsibilities

1. **Collector** (`bgopt collect`) samples portable host and process telemetry.
   Its output is observations, not labels or action history.
2. **LAPS inspector** (`bgopt inspect-laps`) inventories the downloaded artifact
   and documents semantic feature overlap. It does not reuse a LAPS model
   blindly, because its 150 inputs and workload classes are deployment-specific.
3. **Labelling and preprocessing** turn selected observations into a BGOpt
   workload dataset. `workload_label` describes observed process behaviour:
   `interactive`, `background_cpu`, `background_io`, `memory_heavy`, or `idle`.
4. **Classifier** predicts that workload behaviour from the fixed portable
   feature schema. It is only one input to a decision.
5. **Pressure estimator** separately scores host CPU/load, memory, swap, and
   disk activity. It never receives the model's label as an input.
6. **Action policy** combines workload prediction, measured pressure, and
   safety checks into a proposed `no_op`, `nice`, or cgroup CPU-weight action.
7. **Applier** enforces dry-run by default and the PID/cgroup guards for an
   actual, explicitly authorised trial.
8. **Audit/benchmark reporting** counts what was observed and proposed. It
   does not claim an improvement without externally collected outcomes.

## Minimum controlled trial

Use disposable processes only. For a given workload mix, repeat a baseline
trial (`no_op`) and a treatment trial (one explicitly authorised nice change).
Record the fields below for each trial. Keep the action and workload columns
separate.

| Field | Why it matters |
| --- | --- |
| `trial_id`, machine/kernel, timestamp | Reproducibility and environmental drift |
| workload command and input size | A comparable work unit |
| process PID/start time/executable | Guard against PID reuse |
| `workload_label` | What the classifier should learn |
| proposed/applied action, action parameters | The project-owned optimisation layer |
| foreground p50/p95 latency | Protect interactivity |
| background completion time/throughput | Quantify the cost to batch work |
| host CPU, memory, swap, I/O summary | Context and resource pressure |
| accepted/rolled-back, notes | Human review and anomalies |

Do not use `proposed_action` as `workload_label`. A process can be
`background_cpu` in one trial and still correctly receive `no_op` when the host
is not pressured. Likewise, an action outcome should not be treated as an
automatic correctness label for the classifier.

## Advancement gates

1. Keep dry-run until audit logs show foreground/terminal processes are
   consistently protected and candidate selection makes sense.
2. Train only after enough balanced, human-reviewed workload labels exist.
3. Evaluate on a held-out capture from a separate trial, not the training CSV.
4. Trial one exact same-user PID with a `nice` increase; compare it with a
   baseline and inspect the audit log.
5. Consider cgroup CPU weighting only on Linux cgroup v2 and only in a
   pre-created dedicated leaf cgroup. Keep the process's recovery procedure
   documented before moving it.

