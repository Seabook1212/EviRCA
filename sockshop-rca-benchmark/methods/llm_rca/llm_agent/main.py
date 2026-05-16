"""
README
- Entry point for lightweight LLM + Agent RCA baseline.
- Usage:
    python3 main.py --config example_config.json
- Output:
  JSON with top-k root cause services and normalized probabilities.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from rca_pipeline import RCAPipeline


def load_config(config_path: str) -> Dict[str, Any]:
    path = Path(config_path).expanduser()
    if not path.exists():
        # Fallback: resolve relative to this script's directory.
        path = Path(__file__).resolve().parent / path
    if not path.exists():
        raise FileNotFoundError(
            f"Config not found: {config_path}. "
            f"Also checked: {Path(__file__).resolve().parent / config_path}"
        )

    path = path.resolve()
    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    # Resolve relative *_path entries against config file location.
    base_dir = path.resolve().parent
    for key, value in list(config.items()):
        if key.endswith("_path") and isinstance(value, str):
            p = Path(value)
            config[key] = str((base_dir / p).resolve()) if not p.is_absolute() else str(p)
    return config


def _sorted_score_items(score_map: Dict[str, float]) -> list[dict[str, float]]:
    return [
        {"service": service, "score": float(score)}
        for service, score in sorted(score_map.items(), key=lambda x: x[1], reverse=True)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Lightweight LLM + Agent RCA baseline")
    parser.add_argument(
        "--config",
        type=str,
        default="example_config.json",
        help="Path to JSON config file",
    )
    parser.add_argument(
        "--save-debug-json",
        type=str,
        default="",
        help="Optional path to save detailed pipeline outputs",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    pipeline = RCAPipeline(config)
    result = pipeline.run()

    print("=== MetricsAgent KPI Set ===")
    print(json.dumps(result.modality_scores.get("metric_kpis_used", []), indent=2, ensure_ascii=False))

    print("=== MetricsAgent Scores (normalized) ===")
    print(json.dumps(_sorted_score_items(result.modality_scores.get("metric_score", {})), indent=2, ensure_ascii=False))

    print("=== LogsAgent Scores (normalized) ===")
    print(json.dumps(_sorted_score_items(result.modality_scores.get("log_score", {})), indent=2, ensure_ascii=False))

    print("=== TraceAgent Scores (normalized) ===")
    print(json.dumps(_sorted_score_items(result.modality_scores.get("trace_score", {})), indent=2, ensure_ascii=False))

    print("=== Final Top-K Ranking ===")
    print(json.dumps(result.ranking, indent=2, ensure_ascii=False))

    if args.save_debug_json:
        debug_payload = {
            "ranking": result.ranking,
            "raw_graph_scores": result.raw_graph_scores,
            "raw_final_scores": result.raw_final_scores,
            "modality_scores": result.modality_scores,
            "llm_results": result.llm_results,
            "candidate_payloads": result.candidate_payloads,
            "endpoints": result.endpoints,
        }
        out_path = Path(args.save_debug_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(debug_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved debug details to: {out_path}")


if __name__ == "__main__":
    main()
