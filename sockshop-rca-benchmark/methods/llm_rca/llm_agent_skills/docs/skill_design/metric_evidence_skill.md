# MetricEvidenceSkill

## Input

- abnormal metrics window
- baseline metrics window
- KPI config
- thresholds

## Logic

- normalize metrics into long format
- infer service from pod if needed
- compare abnormal vs baseline per pod and per service
- compute:
  - z-score
  - robust z-score
  - delta ratio
  - persistence ratio
- emit anomaly records and aggregated evidence

## Output

- metric anomaly records
- service evidence
- pod evidence
- optional structured follow-up query intents

