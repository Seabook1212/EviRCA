# Architecture

## Flow

1. `RCAOrchestratorAgent` receives a unified `RCARequest`
2. It builds shared incident state and data access
3. It runs the fixed V1 SOP:
   1. metrics
   2. logs
   3. traces
   4. reasoning
4. It returns a structured `RCAResponse`

## Boundaries

- orchestrator: workflow, state, error accumulation
- evidence skills: telemetry-specific anomaly extraction
- reasoning skill: evidence fusion, fault matching, ranking
- data access: API vs CSV backend abstraction
- config: KPI list, fault types, query templates, feature flags

## Design Principles

- simple fixed orchestration
- deterministic default behavior
- extension hooks for richer follow-up queries and real LLM providers
- no hidden side effects in skills

