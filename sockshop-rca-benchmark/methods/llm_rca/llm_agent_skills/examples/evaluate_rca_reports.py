from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REPORT_DIR = (
    PROJECT_ROOT / "RCA_method" / "LLM_agent_skills" / "examples" / "output_api"
)
DEFAULT_GT_DIR = PROJECT_ROOT / "chaosmesh_yaml_v3"
DEFAULT_DETAIL_CSV = Path(__file__).resolve().parent / "rca_evaluation_details.csv"

FAULT_NAME_MAP = {
    "pod_cpu_hog": "cpu_stress",
    "pod_memory_hog": "memory_stress",
    "pod_do_fault": "pod_failure",
    "pod_java_exception": "exception_injection",
    "pod_io_fault": "io_fault",
    "pod_network_delay": "network_delay",
    "pod_network_loss": "network_loss",
    "pod_network_partition": "network_partition",
}

HYPOTHESIS_LINE_RE = re.compile(
    r"^-\s+(?P<entity>.+?):\s+(?P<fault>[a-zA-Z0-9_]+)\s+\((?P<score>[0-9.]+)\)"
)
TOP_ENTITY_LINE_RE = re.compile(r"^\d+\.\s+(?P<entity>.+?)\s+\((?P<score>[0-9.]+)\)")


@dataclass(frozen=True)
class GroundTruth:
    incident_id: str
    service: str
    pod: str | None
    fault_type: str


@dataclass
class ParsedReport:
    incident_id: str
    service_hypotheses: list[tuple[str, str]]
    pod_hypotheses: list[tuple[str, str]]
    top_services: list[str]
    top_pods: list[str]


def infer_fault_type(incident_id: str) -> str | None:
    for prefix, fault_type in FAULT_NAME_MAP.items():
        if incident_id.startswith(prefix + "_"):
            return fault_type
    return None


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def first_pod_from_yaml(data: dict) -> str | None:
    pods_by_namespace = data.get("spec", {}).get("selector", {}).get("pods", {}) or {}
    for pods in pods_by_namespace.values():
        if pods:
            return str(pods[0])
    return None


def service_from_yaml_or_incident(data: dict, incident_id: str) -> str:
    label_selectors = (
        data.get("spec", {}).get("selector", {}).get("labelSelectors", {}) or {}
    )
    service = label_selectors.get("name")
    if service:
        return str(service)

    fault_type = infer_fault_type(incident_id)
    if not fault_type:
        raise ValueError(f"Cannot infer service from incident id: {incident_id}")
    for prefix, mapped_fault in FAULT_NAME_MAP.items():
        if mapped_fault == fault_type and incident_id.startswith(prefix + "_"):
            return incident_id.removeprefix(prefix + "_").rsplit("_", 1)[0]
    raise ValueError(f"Cannot infer service from incident id: {incident_id}")


def build_ground_truth(gt_dir: Path) -> dict[str, GroundTruth]:
    ground_truth: dict[str, GroundTruth] = {}
    for yaml_path in sorted(gt_dir.glob("*.yaml")):
        incident_id = yaml_path.stem
        fault_type = infer_fault_type(incident_id)
        if not fault_type:
            continue
        data = load_yaml(yaml_path)
        ground_truth[incident_id] = GroundTruth(
            incident_id=incident_id,
            service=service_from_yaml_or_incident(data, incident_id),
            pod=first_pod_from_yaml(data),
            fault_type=fault_type,
        )
    return ground_truth


def parse_report(path: Path) -> ParsedReport:
    incident_id = path.stem.removeprefix("rca_report_")
    service_hypotheses: list[tuple[str, str]] = []
    pod_hypotheses: list[tuple[str, str]] = []
    top_services: list[str] = []
    top_pods: list[str] = []
    section: str | None = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("Incident:"):
            incident_id = line.split(":", 1)[1].strip()
            continue
        if line == "Service hypotheses:":
            section = "service_hypotheses"
            continue
        if line == "Pod hypotheses:":
            section = "pod_hypotheses"
            continue
        if line == "Top services:":
            section = "top_services"
            continue
        if line == "Top pods:":
            section = "top_pods"
            continue
        if not line:
            section = None
            continue

        if section in {"service_hypotheses", "pod_hypotheses"}:
            match = HYPOTHESIS_LINE_RE.match(line)
            if not match:
                continue
            item = (match.group("entity"), match.group("fault"))
            if section == "service_hypotheses":
                service_hypotheses.append(item)
            else:
                pod_hypotheses.append(item)
        elif section in {"top_services", "top_pods"}:
            match = TOP_ENTITY_LINE_RE.match(line)
            if not match:
                continue
            if section == "top_services":
                top_services.append(match.group("entity"))
            else:
                top_pods.append(match.group("entity"))

    return ParsedReport(
        incident_id=incident_id,
        service_hypotheses=service_hypotheses,
        pod_hypotheses=pod_hypotheses,
        top_services=top_services,
        top_pods=top_pods,
    )


def unique_ordered(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))


def top_entities(
    report_entities: list[str], hypothesis_pairs: list[tuple[str, str]]
) -> list[str]:
    return unique_ordered(report_entities + [entity for entity, _ in hypothesis_pairs])


def top_pairs(hypothesis_pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    return list(dict.fromkeys(hypothesis_pairs))


def hit_at_k(items: list, target, k: int) -> bool | None:
    if target is None:
        return None
    return target in items[:k]


def evaluate_metric_hit(
    metric: str, predictions: dict, targets: dict, k: int
) -> bool | None:
    if metric == "pod" and targets["pod"] is None:
        return hit_at_k(predictions["service"], targets["service"], k)
    if metric == "pod_fault" and targets["pod_fault"] is None:
        return hit_at_k(predictions["service_fault"], targets["service_fault"], k)
    return hit_at_k(predictions[metric], targets[metric], k)


def format_rate(correct: int, total: int) -> str:
    if total == 0:
        return "N/A"
    return f"{correct / total:.3f} ({correct}/{total})"


def evaluate(report_dir: Path, gt_dir: Path, detail_csv: Path) -> None:
    ground_truth = build_ground_truth(gt_dir)
    reports = [
        parse_report(path) for path in sorted(report_dir.glob("rca_report_*.txt"))
    ]

    metrics = [
        "service",
        "pod",
        "service_fault",
        "pod_fault",
    ]
    ks = [1, 3, 5]
    correct = {(metric, k): 0 for metric in metrics for k in ks}
    total = {(metric, k): 0 for metric in metrics for k in ks}
    rows: list[dict] = []
    missing_gt: list[str] = []

    for report in reports:
        gt = ground_truth.get(report.incident_id)
        if not gt:
            missing_gt.append(report.incident_id)
            continue

        service_predictions = top_entities(
            report.top_services, report.service_hypotheses
        )
        pod_predictions = top_entities(report.top_pods, report.pod_hypotheses)
        service_fault_predictions = top_pairs(report.service_hypotheses)
        pod_fault_predictions = top_pairs(report.pod_hypotheses)

        targets = {
            "service": gt.service,
            "pod": gt.pod,
            "service_fault": (gt.service, gt.fault_type),
            "pod_fault": (gt.pod, gt.fault_type) if gt.pod else None,
        }
        predictions = {
            "service": service_predictions,
            "pod": pod_predictions,
            "service_fault": service_fault_predictions,
            "pod_fault": pod_fault_predictions,
        }

        row = {
            "incident_id": report.incident_id,
            "gt_service": gt.service,
            "gt_pod": gt.pod or "",
            "gt_fault_type": gt.fault_type,
        }
        for metric in metrics:
            for k in ks:
                hit = evaluate_metric_hit(metric, predictions, targets, k)
                if hit is not None:
                    total[(metric, k)] += 1
                    correct[(metric, k)] += int(hit)
                row[f"{metric}_top{k}"] = "" if hit is None else int(hit)
        row["pred_services"] = " | ".join(service_predictions[:5])
        row["pred_pods"] = " | ".join(pod_predictions[:5])
        row["pred_service_faults"] = " | ".join(
            f"{service}:{fault}" for service, fault in service_fault_predictions[:5]
        )
        row["pred_pod_faults"] = " | ".join(
            f"{pod}:{fault}" for pod, fault in pod_fault_predictions[:5]
        )
        rows.append(row)

    detail_csv.parent.mkdir(parents=True, exist_ok=True)
    with detail_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "incident_id",
            "gt_service",
            "gt_pod",
            "gt_fault_type",
            *[f"{metric}_top{k}" for metric in metrics for k in ks],
            "pred_services",
            "pred_pods",
            "pred_service_faults",
            "pred_pod_faults",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Evaluated reports: {len(rows)}")
    if missing_gt:
        print(f"Skipped reports without ground truth: {len(missing_gt)}")
        print(
            "  " + ", ".join(missing_gt[:10]) + (" ..." if len(missing_gt) > 10 else "")
        )
    print()
    print("Accuracy:")
    for metric in metrics:
        print(f"- {metric}:")
        for k in ks:
            print(f"  top{k}: {format_rate(correct[(metric, k)], total[(metric, k)])}")
    print()
    print(f"Detail CSV: {detail_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate RCA report top-k accuracy against Chaos Mesh YAML ground truth."
    )
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--gt-dir", type=Path, default=DEFAULT_GT_DIR)
    parser.add_argument("--detail-csv", type=Path, default=DEFAULT_DETAIL_CSV)
    args = parser.parse_args()
    evaluate(args.report_dir, args.gt_dir, args.detail_csv)


if __name__ == "__main__":
    main()
