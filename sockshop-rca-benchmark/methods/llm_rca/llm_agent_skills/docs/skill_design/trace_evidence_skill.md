# TraceEvidenceSkill

## Input

- abnormal traces
- baseline traces

## Logic

- normalize parsed traces
- reconstruct service-to-service edges from client spans
- compare baseline vs abnormal for:
  - edge latency spikes
  - edge failure spikes
  - service and pod path latency spikes
- emit propagation hints for downstream edges

