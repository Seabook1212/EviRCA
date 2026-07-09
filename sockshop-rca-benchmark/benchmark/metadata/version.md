# Benchmark Metadata Version

## Version 1.0.0

**Status:** public artifact metadata

This metadata describes the EviRCA enhanced Sock Shop RCA benchmark package.

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

## Dataset Release Notes

Full raw telemetry archives and case-level metadata are distributed through the released dataset folder rather than duplicated directly in this Git repository:

https://drive.google.com/drive/folders/1dBNDc2YUYyjJch_BmhO-Hx-0YvdPHHl_?usp=drive_link

Case labels are derived from structured `fault_metadata.json` files and workload windows are derived from `workload_metadata.json`.
