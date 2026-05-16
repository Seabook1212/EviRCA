# Repo Layout Notes

This benchmark repository is organized around the experiment lifecycle:

1. `benchmark/workload/` for online workload generation
2. `benchmark/chaos/` for fault specifications and injection helpers
3. `benchmark/orchestration/` for batch execution and collection flow
4. `benchmark/telemetry/collectors/` for raw telemetry export and parsing
5. `benchmark/topology/` and `benchmark/metadata/` for shared benchmark context
6. `methods/` for baseline and LLM-based RCA methods

The full Sock Shop application codebase is intentionally not duplicated here.
