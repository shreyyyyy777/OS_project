# LAPS-to-BGOpt feature crosswalk

The trusted LAPS artifact contains 150 BPF-derived features, 459,481 feature
rows, 28 paired-scene labels, and ten VM identifiers. BGOpt uses it to define
feature families and experimental workload coverage; it does not load LAPS'
28-class model into the per-process controller.

| LAPS feature family | BGOpt status | BGOpt signal / reason |
| --- | --- | --- |
| `cpu_usage_*`, `cpu_total_*`, core count | Approximated | `host_cpu_percent`, `host_load_1`, `host_load_1_norm`; psutil does not expose every Linux CPU-mode counter on macOS. |
| `mem_*` | Directly reused / approximated | Host memory, available memory, swap percentage. |
| `process_top_*`, `focus_process_*`, `process_count` | Directly reused / extended | Per-PID CPU, memory, RSS, age, threads, open files, and process role. |
| disk queues, IOPS, read/write latency/size | Approximated | Process read/write bytes and rates; host disk throughput. Queue depth and latency need Linux eBPF or block tracing and are not silently fabricated. |
| network byte/packet/retransmit metrics | Partly approximated | Host network byte rate. Per-process network attribution and TCP retransmits are Linux instrumentation work items. |
| run queue, context-switch/wakeup latency, locks, migrations | Unavailable in portable collector | These require the LAPS BPF probes or equivalent Linux eBPF collection. BGOpt marks them unavailable rather than zero-equivalent features. |
| input events | New project label/input | Foreground role and measured foreground latency/performance come from experiment configuration/probes. |
| LAPS paired scene label | Reference only | Used to choose controlled combinations such as browsing + compilation. It is never converted into `nice`, cgroup, or kill labels. |
| action/outcome/starvation/cooldown fields | New in BGOpt | Project-specific labels and measurements for the optimisation contribution. |

## Exact boundary

LAPS maps its workload scene to an expert **scheduler** through a deployment
Policy Mapping Table. BGOpt maps its own process-behaviour estimate, foreground
demand, pressure, cooldown, and fairness state to a guarded **user-space
proposal**. A feature name overlap is not model compatibility: LAPS' feature
order, BPF semantics, 28-class vocabulary, and scheduler mapping must stay
with LAPS unless reproduced on a Linux test platform.
