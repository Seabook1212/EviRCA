# EviRCA Artifact Repository

This repository contains the anonymized benchmark artifact for **EviRCA**, an evidence-aware, skill-based LLM agent and telemetry-rich microservice RCA benchmark.

The benchmark package is under [`sockshop-rca-benchmark/`](sockshop-rca-benchmark/). It provides the reusable experiment assets used to construct and evaluate the enhanced Sock Shop RCA benchmark, while omitting large telemetry payloads, full case-level ground-truth releases, and dataset download links for anonymous review.

## Artifact Scope

This repository focuses on benchmark-side materials:

- workload generation assets for the enhanced Sock Shop benchmark
- fault orchestration and Chaos Mesh fault specifications
- telemetry collection and parsing scripts for metrics, logs, and traces
- metric-type inventories and service-topology metadata
- baseline RCA runners and compact reproduced result summaries
- documentation describing the benchmark layout and dataset organization

The enhanced Sock Shop service implementations and full telemetry dataset are not duplicated in this package. Ground-truth labels are represented by the fault metadata schema and evaluation scripts, but the full case-level release is intentionally not bundled here.

## Paper Alignment

The artifact corresponds to the benchmark described in the paper:

- system: enhanced Sock Shop microservice benchmark
- telemetry: Prometheus metrics, Loki logs, and Jaeger traces
- scale: 320 fault cases and 100 normal baseline cases across five experiment days
- faults: network delay, network loss, network partition, CPU stress, memory stress, pod failure, JVM exception injection, and I/O fault injection
- metrics: 1,390 selected Prometheus metric signals across application, container, node, service-proxy, middleware, database, network, and KPI layers
- labels: service-, pod-, service-fault-, and pod-fault-level RCA evaluation targets derived from structured fault metadata

## Repository Layout

- [`sockshop-rca-benchmark/benchmark/`](sockshop-rca-benchmark/benchmark/)
  - workload generation, orchestration, chaos definitions, telemetry collection, topology, and metadata
- [`sockshop-rca-benchmark/methods/`](sockshop-rca-benchmark/methods/)
  - reproduced baseline RCA method runners and result summaries
- [`sockshop-rca-benchmark/evaluation/reference_results/`](sockshop-rca-benchmark/evaluation/reference_results/)
  - compact benchmark result summaries
- [`sockshop-rca-benchmark/docs/`](sockshop-rca-benchmark/docs/)
  - benchmark packaging and layout notes

For details, see [`sockshop-rca-benchmark/README.md`](sockshop-rca-benchmark/README.md).
