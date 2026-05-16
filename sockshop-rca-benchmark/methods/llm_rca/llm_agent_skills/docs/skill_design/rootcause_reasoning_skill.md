# RootCauseReasoningSkill

## Input

- service and pod evidence from metrics, logs, and traces
- topology
- fixed V1 fault taxonomy

## Internal Steps

1. aggregate evidence per service and per pod
2. generate coarse rule hints from `configs/rootcause_reasoning_rules.yaml`
3. coarse fault type matching with rule-based priors
4. light checks
5. LLM-backed ranking with heuristic fallback
6. final summary generation

## Output

- service top 5
- pod top 5
- final summary
- warnings and errors

## Notes

- Human-readable rule explanations live in `docs/skill_design/rootcause_reasoning_rules.md`.
- Machine-readable rule definitions live in `configs/rootcause_reasoning_rules.yaml`.
- Candidate metadata includes `rule_hints` and `active_rules` so reasoning remains debuggable.
