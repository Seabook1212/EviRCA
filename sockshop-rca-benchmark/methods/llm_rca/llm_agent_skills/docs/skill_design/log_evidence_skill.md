# LogEvidenceSkill

## Input

- abnormal logs
- baseline logs

## Logic

- normalize message templates
- extract keywords
- compare baseline vs abnormal for:
  - template spikes
  - keyword spikes
  - error/fatal level shifts

## Output

- log anomaly records
- service evidence
- pod evidence
- optional structured follow-up intents

