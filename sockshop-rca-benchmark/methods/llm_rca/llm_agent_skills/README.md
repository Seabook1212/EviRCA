# LLM Agent Skills RCA

V1 skill-based RCA framework for microservice systems.

It uses one orchestrator agent and four skills:

- `MetricEvidenceSkill`
- `LogEvidenceSkill`
- `TraceEvidenceSkill`
- `RootCauseReasoningSkill`

The orchestrator owns workflow and state. Skills own evidence extraction and RCA reasoning.

## Features

- unified RCA request schema for `api` and `csv` modes
- Sock Shop topology included
- baseline-vs-abnormal anomaly detection for metrics, logs, and traces
- fixed V1 fault taxonomy
- explicit coarse RCA rules config for reasoning transparency
- structured JSON-serializable RCA output
- configurable query templates and budgets
- deterministic heuristic LLM backend by default so the system stays runnable offline

## Quick Start

```bash
cd /Users/zhangfan/PyCharmMiscProject/chaos_experiment/RCA_method/LLM_agent_skills
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m rca_agent_skills.cli examples/sample_request.json
```

## CSV Example

```bash
python examples/run_csv_mode.py
```

## API Example

```bash
python examples/run_api_mode.py
```

## Request Shape

```json
{
  "incident_id": "incident_001",
  "backend_mode": "csv",
  "abnormal_window": {
    "start": "2026-04-25T11:00:00Z",
    "end": "2026-04-25T11:15:00Z"
  },
  "baseline_window": {
    "start": "2026-04-25T10:45:00Z",
    "end": "2026-04-25T11:00:00Z"
  },
  "csv_inputs": {
    "metrics_csv": "tests/fixtures/sample_metrics.csv",
    "logs_csv": "tests/fixtures/sample_logs.csv",
    "traces_csv": "tests/fixtures/sample_traces.csv",
    "topology_file": "tests/fixtures/sample_topology.json"
  }
}
```

## Project Notes

- API mode follows the URL and query style from your existing Prometheus, Loki, and Jaeger scripts.
- Follow-up query expansion is structured and budgeted. Raw unrestricted query generation is intentionally blocked.
- The default LLM client is heuristic. It is small on purpose so a real provider can be swapped in later.
- `RootCauseReasoningSkill` now loads coarse heuristic rules from `configs/rootcause_reasoning_rules.yaml` and exposes rule hints in reasoning metadata.

## Tests

```bash
pytest
```
