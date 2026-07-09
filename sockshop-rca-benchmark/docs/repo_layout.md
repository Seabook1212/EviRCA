# Repo Layout Notes

This benchmark repository is organized around the experiment lifecycle:

1. `benchmark/workload/` for online workload generation
2. `benchmark/chaos/` for fault specifications and injection helpers
3. `benchmark/orchestration/` for batch execution and collection flow
4. `benchmark/telemetry/collectors/` for raw telemetry export and parsing
5. `benchmark/topology/` and `benchmark/metadata/` for shared benchmark context
6. `methods/` for reproduced baseline RCA method runners and compact outputs

The full enhanced Sock Shop application codebase and large telemetry payloads are not duplicated directly in Git; the released dataset folder provides the large benchmark payloads and case-level metadata.
