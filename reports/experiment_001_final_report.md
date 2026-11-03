# Experiment 001 Final Report

## Objective
Evaluate the scheduler-responsiveness/background-throughput tradeoff under controlled CPU contention. Scheduler lag is a responsiveness proxy, not browser-rendering latency.

## Protocol
Ubuntu VM with 4 vCPUs; five fresh workload executions per mode; deterministic interleaved order; 50 ms scheduler-lag probe. Modes: Linux Default, Static Nice +5, BGOpt Heuristic, and BGOpt ML-Assisted.

## Aggregate results

| Mode | p95 lag mean / median (ms) | Completion mean / median (s) | Controller iterations mean | Applied NICE actions |
|---|---:|---:|---:|---:|
| Linux Default | 1.092 / 1.086 | 26.221 / 26.051 | 0.000 | 0.000 |
| Static Nice +5 | 1.119 / 1.117 | 25.711 / 25.802 | 0.000 | 1.000 |
| BGOpt Heuristic | 1.088 / 1.086 | 27.297 / 27.352 | 539.000 | 0.000 |
| BGOpt ML-Assisted | 0.389 / 0.382 | 28.528 / 28.534 | 2.000 | 0.000 |

## Linux-default baseline comparisons

| Mode | p95 mean change | p95 median change | Completion mean change | Completion median change |
|---|---:|---:|---:|---:|
| Static Nice +5 | 2.46% | 2.84% | -1.94% | -0.96% |
| BGOpt Heuristic | -0.38% | -0.03% | 4.10% | 4.99% |
| BGOpt ML-Assisted | -64.39% | -64.86% | 8.80% | 9.53% |

## Raw results

| Order | Mode | Rep | p95 lag (ms) | Completion (s) | Controller iterations | Applied actions |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Linux Default | 1 | 1.086 | 25.200 | 0 | 0 |
| 2 | Static Nice +5 | 1 | 1.115 | 25.201 | 0 | 1 |
| 3 | BGOpt Heuristic | 1 | 1.086 | 26.816 | 530 | 0 |
| 4 | BGOpt ML-Assisted | 1 | 0.416 | 28.534 | 2 | 0 |
| 5 | Static Nice +5 | 2 | 1.122 | 25.802 | 0 | 1 |
| 6 | BGOpt Heuristic | 2 | 1.085 | 27.352 | 540 | 0 |
| 7 | BGOpt ML-Assisted | 2 | 0.374 | 28.640 | 2 | 0 |
| 8 | Linux Default | 2 | 1.085 | 26.051 | 0 | 0 |
| 9 | BGOpt Heuristic | 3 | 1.090 | 27.414 | 541 | 0 |
| 10 | BGOpt ML-Assisted | 3 | 0.410 | 28.553 | 2 | 0 |
| 11 | Linux Default | 3 | 1.100 | 27.250 | 0 | 0 |
| 12 | Static Nice +5 | 3 | 1.117 | 25.900 | 0 | 1 |
| 13 | BGOpt ML-Assisted | 4 | 0.363 | 28.451 | 2 | 0 |
| 14 | Linux Default | 4 | 1.115 | 26.951 | 0 | 0 |
| 15 | Static Nice +5 | 4 | 1.136 | 26.351 | 0 | 1 |
| 16 | BGOpt Heuristic | 4 | 1.093 | 27.651 | 545 | 0 |
| 17 | Linux Default | 5 | 1.074 | 25.651 | 0 | 0 |
| 18 | BGOpt Heuristic | 5 | 1.084 | 27.252 | 539 | 0 |
| 19 | Static Nice +5 | 5 | 1.102 | 25.301 | 0 | 1 |
| 20 | BGOpt ML-Assisted | 5 | 0.382 | 28.464 | 2 | 0 |

## ML-assisted behavior
Frozen v4 ML predictions: {"background_cpu": 10, "interactive": 40, "io_heavy": 685}. Mean controller iterations: 2.000; applied actions: 0. The separate process preserves probe cadence, but model runtime remains costly and yields far fewer iterations than the heuristic controller.

## Safety outcome
All 20 workloads exited with code 0. All applied real actions matched the corresponding disposable workload identity. No process was killed; no cgroup or scheduler change occurred; controller workers were cleaned up.

## Threats to validity
n=5 per mode; Ubuntu 4-vCPU VM; host scheduling may affect results; controlled disposable workload; scheduler lag is not browser-rendering latency; small controlled ML corpus; ML domain-shift/generalization and runtime-cost limitations. Results are prototype evidence, not production scheduler benchmarks.

## Conclusion
Interpret differences descriptively only: n=5 does not establish statistical significance. The ML path demonstrates integration but its low cadence is a material runtime limitation.
