# EviRCA

This repository currently hosts the Sock Shop RCA benchmark package under [`sockshop-rca-benchmark/`](sockshop-rca-benchmark/).

The detailed benchmark README remains here: [`sockshop-rca-benchmark/README.md`](sockshop-rca-benchmark/README.md)

## Dataset

- released dataset folder: [Google Drive](https://drive.google.com/drive/folders/1dBNDc2YUYyjJch_BmhO-Hx-0YvdPHHl_?usp=drive_link)

## Benchmark Package Summary

The benchmark package includes reusable assets for workload generation, fault injection, telemetry collection, and RCA benchmarking, without bundling the enhanced Sock Shop service source code or the full released telemetry dataset.

### Included

- workload generation scripts
- fault orchestration scripts
- Chaos Mesh YAML specifications
- telemetry collection and parsing scripts
- topology and metadata reference files
- baseline RCA runners and helper scripts
- LLM-based RCA implementations, configs, docs, and examples
- selected benchmark result summaries

### Repository Layout

- `sockshop-rca-benchmark/benchmark/`
  - workload generation, orchestration, chaos definitions, telemetry collection, topology, and metadata
- `sockshop-rca-benchmark/methods/`
  - baseline RCA methods and LLM-based RCA implementations
- `sockshop-rca-benchmark/evaluation/reference_results/`
  - compact reproduced result summaries
- `sockshop-rca-benchmark/docs/`
  - benchmark packaging notes
