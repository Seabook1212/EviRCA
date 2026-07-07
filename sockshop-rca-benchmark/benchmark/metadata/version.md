# Benchmark Metadata Version

## Version 1.0.0

**Status:** anonymized artifact metadata

This metadata describes the EviRCA enhanced Sock Shop RCA benchmark package used for anonymous review.

## Summary

- Application: enhanced Sock Shop
- Workload generator: Locust
- Fault injection: Chaos Mesh
- Fault cases: 320
- Normal baseline cases: 100
- Experiment days: 5
- Telemetry modalities: metrics, logs, traces
- Selected metric signals: 1,390
- Primary metric sampling interval: 5 seconds

## Fault Actions

- network delay
- network loss
- network partition
- CPU stress
- memory stress
- pod failure
- JVM exception injection
- I/O fault injection

## Included Metadata Modules

- `dataset_description.json`
- `services_topology.json`
- metric-name inventories under `benchmark/telemetry/data/metrics/types/`
- Chaos Mesh YAML specifications under `benchmark/chaos/chaosmesh/`

## Anonymous Review Notes

Full raw telemetry archives, full case-level ground-truth files, service repositories, and dataset hosting links are intentionally omitted from this repository. When the full dataset is available, case labels are derived from structured `fault_metadata.json` files and workload windows are derived from `workload_metadata.json`.
