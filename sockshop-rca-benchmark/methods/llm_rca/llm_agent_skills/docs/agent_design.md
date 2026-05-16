# RCAOrchestratorAgent Design

## Responsibilities

- validate and hold request context
- initialize state
- instantiate shared data access and LLM client
- run skills in fixed order
- persist warnings and errors
- return final RCA response

## State

The agent state contains:

- incident id
- abnormal and baseline windows
- backend mode
- topology
- metrics evidence
- logs evidence
- traces evidence
- final RCA result
- warnings
- errors
- query budgets

## Non-responsibilities

The agent does not perform:

- metric anomaly detection
- log anomaly detection
- trace anomaly detection
- final root-cause ranking

