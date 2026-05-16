from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rca_agent_skills.common.io_utils import read_yaml
from rca_agent_skills.llm import LLMClient
from rca_agent_skills.main import run_rca
from rca_agent_skills.outputs.writer import write_outputs
from rca_agent_skills.skills.semantic_request_parsing_skill import (
    SemanticRequestParsingSkill,
)
from rca_agent_skills.utils.baseline_window_selector import select_baseline_window


DEFAULT_MESSAGE = (
    "Analyze sock-shop from 2026-05-02T22:13:41Z to 2026-05-02T22:28:41Z. "
    "Use metrics, logs, and traces. Return service, pod, service fault, and pod fault rankings."
)


def build_llm_client(settings: dict) -> LLMClient:
    llm_cfg = settings.get("llm", {})
    return LLMClient(
        provider=llm_cfg.get("provider", "heuristic"),
        model=llm_cfg.get("model", "heuristic-v1"),
        temperature=float(llm_cfg.get("temperature", 0.0)),
        debug=settings.get("debug", {}),
    )


def render_requested_outputs(result) -> str:
    requested = (result.metadata.get("execution_options", {}) or {}).get(
        "requested_outputs", {}
    )
    if not requested:
        requested = {
            "service_ranking": True,
            "pod_ranking": True,
            "service_fault_ranking": True,
            "pod_fault_ranking": True,
        }
    lines = [f"Incident: {result.incident_id}", f"Summary: {result.final_summary}", ""]
    if requested.get("service_fault_ranking", True):
        lines.append("Service hypotheses:")
        for item in result.service_top5:
            lines.append(f"- {item.service}: {item.fault_type} ({item.score:.2f})")
        lines.append("")
    if requested.get("pod_fault_ranking", True):
        lines.append("Pod hypotheses:")
        for item in result.pod_top5:
            lines.append(f"- {item.pod}: {item.fault_type} ({item.score:.2f})")
        lines.append("")
    if requested.get("service_ranking", True):
        lines.append("Top services:")
        seen = {}
        for item in result.service_top5:
            if item.service and item.service not in seen:
                seen[item.service] = item.score
        for index, (service, score) in enumerate(seen.items(), start=1):
            lines.append(f"{index}. {service} ({score:.2f})")
        lines.append("")
    if requested.get("pod_ranking", True):
        lines.append("Top pods:")
        seen = {}
        for item in result.pod_top5:
            if item.pod and item.pod not in seen:
                seen[item.pod] = item.score
        for index, (pod, score) in enumerate(seen.items(), start=1):
            lines.append(f"{index}. {pod} ({score:.2f})")
    return "\n".join(lines).strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run RCA from a natural-language request."
    )
    parser.add_argument("--incident-id", default="semantic_manual_001")
    parser.add_argument("--backend-mode", default="api", choices=["api", "csv"])
    parser.add_argument("--namespace", default="sock-shop")
    parser.add_argument("--message", default=DEFAULT_MESSAGE)
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "examples" / "output_semantic"
    )
    args = parser.parse_args()

    settings = read_yaml(PROJECT_ROOT / "configs" / "settings.yaml")
    baseline_config = read_yaml(PROJECT_ROOT / "configs" / "baseline_windows.yaml")
    semantic_skill = SemanticRequestParsingSkill(settings, build_llm_client(settings))
    parse_result = semantic_skill.run(args.message, namespace=args.namespace)
    if parse_result.needs_clarification or not parse_result.abnormal_window:
        print("Semantic request needs clarification:")
        for question in parse_result.clarification_questions:
            print(f"- {question}")
        return

    baseline = select_baseline_window(
        parse_result.abnormal_window,
        baseline_config,
        namespace=args.namespace,
    )
    payload = {
        "incident_id": args.incident_id,
        "backend_mode": args.backend_mode,
        "namespace": args.namespace,
        "abnormal_window": parse_result.abnormal_window,
        "baseline_window": {"start": baseline.start, "end": baseline.end},
        "execution_options": {
            "enabled_telemetry": parse_result.enabled_telemetry,
            "requested_outputs": parse_result.requested_outputs,
            "ranking_depth": parse_result.ranking_depth,
            "semantic_parse_notes": parse_result.notes,
            "baseline_selection": asdict(baseline),
        },
    }
    if args.backend_mode == "api":
        api_cfg = settings.get("api", {})
        payload["api_inputs"] = {
            "prometheus_url": api_cfg.get("prometheus", {}).get("base_url"),
            "loki_url": api_cfg.get("loki", {}).get("base_url"),
            "jaeger_url": api_cfg.get("jaeger", {}).get("base_url"),
            "namespace": args.namespace,
        }

    result = run_rca(payload, project_root=PROJECT_ROOT)
    write_outputs(result, args.output_dir)
    print(render_requested_outputs(result))
    print()
    print(
        f"Selected baseline: {baseline.baseline_id} {baseline.start} -> {baseline.end}"
    )
    print(f"Outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
