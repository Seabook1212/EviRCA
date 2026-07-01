# Sock Shop RCA Benchmark

This repository packages the benchmark-side artifacts for the Sock Shop RCA study.

It is intended to hold the reusable experiment assets needed to reproduce workload generation, fault injection, telemetry collection, and RCA benchmarking, without bundling the enhanced Sock Shop service source code or the full released telemetry dataset.

## Included

- workload generation scripts
- fault orchestration scripts
- Chaos Mesh YAML specifications
- telemetry collection and parsing scripts
- topology and metadata reference files
- baseline RCA runners and helper scripts
- LLM-based RCA implementations, configs, docs, and examples
- selected benchmark result summaries

## Not Included

- enhanced Sock Shop service source repositories
- Kubernetes deployment manifests for each service repo
- full `dataset_v2` telemetry payloads
- full case-level ground-truth releases
- large per-case RCA outputs

## Dataset

- released dataset folder: [Google Drive](https://drive.google.com/drive/folders/1dBNDc2YUYyjJch_BmhO-Hx-0YvdPHHl_?usp=drive_link)

## Repository Layout

- `benchmark/`
  - `workload/locust/`: Locust workload generator and user/profile helpers
  - `orchestration/`: normal/fault batch runners and telemetry collection entrypoints
  - `chaos/`: Chaos Mesh YAML files and chaos tooling scripts
  - `telemetry/collectors/`: metrics, logs, traces, and event collection/parsing scripts
  - `topology/`: service dependency topology
  - `metadata/`: dataset description, fault inventory, and version notes
- `methods/`
  - `baselines/`: CloudRanger, MicroRCA, Nezha, SBLD, and TraceAnomaly runners
  - `llm_rca/`: lightweight LLM RCA and the newer skill-based LLM RCA implementation
- `evaluation/reference_results/`: placeholder area for compact reproduced summaries
- `docs/`: benchmark packaging notes

## Notes

- Some copied metadata files come from earlier dataset packaging work and may need a light refresh before public release.
- The current benchmark repo is a packaging scaffold. A release-ready version should still add deployment instructions, environment pinning, dataset manifest files, and service commit references.
