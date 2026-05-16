# Lightweight LLM + Agent RCA Baseline

This project implements a lightweight root-cause analysis (RCA) pipeline for microservice systems.

Pipeline:
1. Algorithmic anomaly scoring from metrics, logs, and traces.
2. Graph-based root ranking using reverse PageRank on service topology.
3. LLM verification only for Top-K algorithmic candidates (no free discovery).
4. Final Top-K service probabilities normalized to sum to 1.

## Files

- `data_loader.py`: Loads topology and observability data from folders.
- `metrics_agent.py`: Metrics anomaly detection.
- `logs_agent.py`: Log anomaly detection.
- `trace_agent.py`: Trace anomaly detection.
- `graph_ranker.py`: Score fusion + reverse propagation ranking.
- `llm_verifier.py`: ReAct-lite constrained LLM verification.
- `rca_pipeline.py`: End-to-end RCA pipeline.
- `main.py`: CLI entrypoint.
- `example_config.json`: Example run config.

## Requirements

Python 3.10+ and packages:
- `pandas`
- `numpy`
- `networkx`
- `scipy`
- `openai`

## Quick Start

```bash
cd chaos_experiment/RCA_method/LLM_agent
python3 main.py --config example_config.json
```

## Notes

- Pod-level files are combined into service-level analysis automatically.
- LLM is optional (`"use_llm": false` for purely algorithmic baseline).
- Metrics KPI filtering is configurable:
  - `metrics_kpis`: common KPI whitelist.
  - `latency_kpi`: choose one latency KPI (`pod_request_latency_p90/p95/p99`).
  - Runtime prints the exact KPI set used by `MetricsAgent`.
- Reverse PageRank tuning:
  - `pagerank_alpha`: propagation strength in PageRank.
  - `root_score_blend`: blend ratio between root personalization score and propagated score.
    `0.0` = pure Reverse PageRank, `1.0` = pure root score without propagation.
- LLM provider config supports OpenAI and Gemini (OpenAI-compatible endpoint):
  - OpenAI: set `llm_provider="openai"`, `llm_api_key_env="OPENAI_API_KEY"`.
  - Gemini: set `llm_provider="gemini"`, `llm_api_key_env="GEMINI_API_KEY"`,
    `llm_base_url="https://generativelanguage.googleapis.com/v1beta/openai/"`,
    `llm_model="gemini-2.0-flash"` (or another Gemini model).
- Endpoint extraction references:
  - `chaos_experiment/logs_script/loki_script.py`
  - `chaos_experiment/metrics_script/prometheus_node_script.py`
  - `chaos_experiment/metrics_script/prometheus_pod_specific_script.py`
  - `chaos_experiment/traces_script/jaeger_script.py`
