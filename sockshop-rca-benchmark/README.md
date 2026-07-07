# Sock Shop RCA Benchmark

This package contains the reusable benchmark-side artifacts for the EviRCA microservice root cause analysis study.

The benchmark is built around an enhanced Sock Shop system and is designed for reproducible workload generation, controlled fault injection, synchronized telemetry collection, and RCA evaluation over metrics, logs, traces, topology, and structured fault metadata.

## Scope

Included in this anonymized package:

- Locust workload generation scripts and user/profile helpers
- normal and faulty run orchestration scripts
- Chaos Mesh YAML specifications for the injected fault cases
- telemetry collectors and parsers for Prometheus metrics, Loki logs, and Jaeger traces
- selected metric-name inventories covering the benchmark telemetry schema
- service-topology and benchmark metadata files
- reproduced baseline RCA runners for service-level evaluation
- compact reproduced result summaries

Intentionally not included:

- full enhanced Sock Shop service source repositories
- full raw telemetry archives
- full case-level ground-truth release files
- large per-case RCA outputs
- non-anonymous dataset download links

The full telemetry payloads and case-level labels are large and are omitted from this anonymized artifact. Evaluation scripts derive RCA targets from each case's `fault_metadata.json` structure when the full dataset is available.

## Benchmark Summary

- **Application:** enhanced Sock Shop microservice benchmark
- **Workload:** Locust-driven online user traffic
- **Fault injection:** Chaos Mesh
- **Telemetry:** Prometheus metrics, Loki logs, and Jaeger traces
- **Metric resolution:** predominantly 5 seconds
- **Metric coverage:** 1,390 selected Prometheus metric signals
- **Fault cases:** 320 cases across five experiment days
- **Normal baselines:** 100 normal runs
- **Fault actions:** network delay, network loss, network partition, CPU stress, memory stress, pod failure, JVM exception injection, and I/O fault injection
- **Evaluation targets:** service, pod, service-fault, and pod-fault rankings

## Repository Layout

- `benchmark/`
  - `workload/locust/`: Locust workload generator and user/profile helpers
  - `orchestration/`: normal/fault batch runners and telemetry collection entrypoints
  - `chaos/chaosmesh/`: declarative Chaos Mesh fault specifications
  - `chaos/tools/`: helper scripts for chaos experiment execution
  - `telemetry/collectors/metrics/`: Prometheus metric collection scripts
  - `telemetry/collectors/logs/`: Loki log collection and parsing scripts
  - `telemetry/collectors/traces/`: Jaeger trace collection and parsing scripts
  - `telemetry/data/metrics/types/`: selected metric-name inventories
  - `topology/`: service dependency topology
  - `metadata/`: benchmark metadata and version notes
- `methods/`
  - `baselines/`: reproduced baseline RCA method runners and compact outputs
- `evaluation/reference_results/`: compact reproduced result summaries
- `docs/`: benchmark packaging and layout notes

## Dataset Organization

When the full dataset is available, it follows the organization described in the paper:

```text
fault_run/<date>/<case_id>/
  fault_info/
    fault_metadata.json
  workload/
    workload_metadata.json
    locust_stats.json

normal_run/<date>/<case_id>/
  workload/
    workload_metadata.json
    locust_stats.json

telemetry/<date>/
  metrics/
  logs/
  traces/
```

`workload_metadata.json` records workload start/end timestamps, target host, users, spawn rate, and run duration. For fault cases, `fault_metadata.json` records the fault identity, injection tool, target service or pod, injection timestamps, and fault-specific parameters such as action, duration, mode, delay, loss ratio, or stress configuration.

RCA methods should use the workload and fault timestamps to select case-specific analysis windows and keep metrics, logs, and traces temporally aligned.

## Notes for Reviewers

This repository intentionally avoids dataset-hosting links in order to preserve anonymous-review constraints. The committed files are meant to document and reproduce the benchmark construction and evaluation pipeline once the full telemetry payloads are available in the corresponding artifact release.
