from __future__ import annotations

from collections import defaultdict


def aggregate_service_evidence(state) -> dict[str, list]:
    grouped = defaultdict(list)
    for skill_result in [state.metrics_evidence, state.logs_evidence, state.traces_evidence]:
        if not skill_result:
            continue
        for evidence in skill_result.service_evidence:
            grouped[evidence.service].append(evidence)
    return grouped


def aggregate_pod_evidence(state) -> dict[str, list]:
    grouped = defaultdict(list)
    for skill_result in [state.metrics_evidence, state.logs_evidence, state.traces_evidence]:
        if not skill_result:
            continue
        for evidence in skill_result.pod_evidence:
            grouped[evidence.pod].append(evidence)
    return grouped


def aggregate_pods_by_service(state) -> dict[str, list[tuple[str, list]]]:
    grouped = defaultdict(list)
    for pod, evidence_items in aggregate_pod_evidence(state).items():
        if not evidence_items:
            continue
        service = evidence_items[0].service
        grouped[service].append((pod, evidence_items))
    return grouped


def summarize_service_pod_scores(state) -> dict[str, list[dict]]:
    summary = defaultdict(list)
    for service, pod_items in aggregate_pods_by_service(state).items():
        for pod, evidence_items in pod_items:
            avg_score = sum(float(item.score) for item in evidence_items) / max(len(evidence_items), 1)
            summary[service].append(
                {
                    "pod": pod,
                    "score": round(avg_score, 4),
                    "evidence_count": sum(len(item.anomaly_records) for item in evidence_items),
                }
            )
        summary[service].sort(key=lambda item: item["score"], reverse=True)
    return summary
