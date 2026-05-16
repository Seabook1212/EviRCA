from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict

import pandas as pd

from .aggregator import aggregate_pod_evidence, aggregate_pods_by_service, aggregate_service_evidence, summarize_service_pod_scores
from .schemas import RuleEvaluation, RuleHints


LOCAL_RESOURCE_PATTERNS = {"cpu_usage_pct", "memory_usage_pct", "restart_count", "ready_ratio"}
LOCAL_FAILURE_PATTERNS = {"exception_injection", "pod_failure"}
LOCAL_FAILURE_KEYWORDS = {
    "oom",
    "outofmemory",
    "out of memory",
    "killed",
    "crash",
    "panic",
    "restart",
    "readiness",
    "unready",
}
DEFAULT_LOCAL_RESOURCE_MIN_DELTA = {
    "cpu_usage_pct": 0.20,
    "memory_usage_pct": 0.10,
    "restart_count": 0.10,
    "ready_ratio": 0.10,
}
SYMPTOM_PATTERNS = {
    "error_count",
    "latency_p50",
    "latency_p90",
    "latency_p95",
    "latency_p99",
    "edge_latency_spike",
    "edge_failure_spike",
    "path_latency_spike",
    "template_spike",
    "keyword_spike",
    "level_shift",
}


def _flatten_records(evidence_items: list) -> list:
    return [record for evidence in evidence_items for record in evidence.anomaly_records]


def _rule_map(rules_config: dict) -> dict[str, dict]:
    mapped: dict[str, dict] = {}
    for group in rules_config.get("rule_groups", {}).values():
        if not group.get("enabled", True):
            continue
        for rule in group.get("rules", []):
            mapped[rule["id"]] = rule
    return mapped


def _score_adjustments(rules_config: dict) -> dict[str, float]:
    return rules_config.get("settings", {}).get("score_adjustments", {})


def _local_resource_thresholds(rules_config: dict) -> dict[str, float]:
    configured = rules_config.get("settings", {}).get(
        "local_resource_min_delta_ratio", {}
    )
    return {**DEFAULT_LOCAL_RESOURCE_MIN_DELTA, **configured}


def _has_material_local_resource_signal(record, thresholds: dict[str, float]) -> bool:
    if record.metric_or_pattern not in LOCAL_RESOURCE_PATTERNS:
        return False
    if record.metric_or_pattern == "restart_count":
        abnormal = record.metadata.get("abnormal_max")
        baseline = record.metadata.get("baseline_max")
        abnormal = (
            float(record.abnormal_value or 0.0)
            if abnormal is None
            else float(abnormal)
        )
        baseline = (
            float(record.baseline_value or 0.0)
            if baseline is None
            else float(baseline)
        )
        return abnormal > baseline
    if record.metric_or_pattern == "ready_ratio":
        abnormal = record.metadata.get("abnormal_min")
        baseline = record.metadata.get("baseline_min")
        abnormal = (
            float(record.abnormal_value or 0.0)
            if abnormal is None
            else float(abnormal)
        )
        baseline = (
            float(record.baseline_value or 0.0)
            if baseline is None
            else float(baseline)
        )
        return abnormal < baseline
    try:
        ratio = abs(float(record.metadata.get("delta_ratio", 0.0)))
    except (TypeError, ValueError):
        ratio = 0.0
    return ratio >= float(thresholds.get(record.metric_or_pattern, 0.10))


def _has_local_failure_signal(record) -> bool:
    return record.metric_or_pattern in LOCAL_FAILURE_PATTERNS or any(
        keyword in record.summary.lower() for keyword in LOCAL_FAILURE_KEYWORDS
    )


def _has_direct_local_failure_evidence(
    records: list, thresholds: dict[str, float]
) -> bool:
    return any(
        _has_material_local_resource_signal(record, thresholds)
        or _has_local_failure_signal(record)
        for record in records
    )


def _timestamps_from_records(records: list) -> list[pd.Timestamp]:
    values: list[pd.Timestamp] = []
    for record in records:
        for field in ["first_seen_ts", "last_seen_ts"]:
            raw_value = record.metadata.get(field)
            if not raw_value:
                continue
            parsed = pd.to_datetime(raw_value, utc=True, errors="coerce")
            if pd.notna(parsed):
                values.append(parsed)
    return values


def _earliest_timestamp(records: list) -> pd.Timestamp | None:
    timestamps = _timestamps_from_records(records)
    return min(timestamps) if timestamps else None


def _topology_maps(topology: dict) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    downstream = defaultdict(set)
    upstream = defaultdict(set)
    for edge in topology.get("edges", []):
        source = edge.get("source")
        target = edge.get("target")
        if not source or not target:
            continue
        downstream[str(source)].add(str(target))
        upstream[str(target)].add(str(source))
    return dict(downstream), dict(upstream)


def build_rule_context(state, topology: dict, rules_config: dict) -> dict:
    service_evidence = aggregate_service_evidence(state)
    pod_evidence = aggregate_pod_evidence(state)
    pods_by_service = aggregate_pods_by_service(state)
    pod_score_summary = summarize_service_pod_scores(state)
    downstream_map, upstream_map = _topology_maps(topology)
    topology_services = set(topology.get("services", []))

    service_first_seen = {}
    pod_first_seen = {}
    trace_targets_by_service = defaultdict(set)
    path_services = set()

    for service, evidence_items in service_evidence.items():
        records = _flatten_records(evidence_items)
        service_first_seen[service] = _earliest_timestamp(records)
        for record in records:
            if record.source != "trace":
                continue
            edge_source = record.metadata.get("edge_source_service")
            edge_target = record.metadata.get("edge_target_service")
            if edge_source and edge_target:
                trace_targets_by_service[str(edge_source)].add(str(edge_target))
                path_services.add(str(edge_source))
                path_services.add(str(edge_target))
                continue
            peer_service = record.metadata.get("peer_service")
            if peer_service:
                trace_targets_by_service[service].add(str(peer_service))
                path_services.add(service)
                path_services.add(str(peer_service))

    for pod, evidence_items in pod_evidence.items():
        pod_first_seen[pod] = _earliest_timestamp(_flatten_records(evidence_items))

    shared_downstream_counter = Counter()
    anomalous_services = set(service_evidence.keys())
    anomalous_services.update(service for service, pods in pods_by_service.items() if pods)
    if topology_services:
        anomalous_services = {
            service for service in anomalous_services if service in topology_services
        }

    for service, targets in trace_targets_by_service.items():
        if service in anomalous_services:
            shared_downstream_counter.update(targets)
    min_shared_upstreams = int(rules_config.get("settings", {}).get("min_shared_upstreams", 2))
    shared_downstream_targets = {
        service for service, count in shared_downstream_counter.items() if count >= min_shared_upstreams
    }

    return {
        "service_evidence": service_evidence,
        "pod_evidence": pod_evidence,
        "pods_by_service": pods_by_service,
        "pod_score_summary": pod_score_summary,
        "service_first_seen": service_first_seen,
        "pod_first_seen": pod_first_seen,
        "trace_targets_by_service": {service: sorted(values) for service, values in trace_targets_by_service.items()},
        "shared_downstream_targets": sorted(shared_downstream_targets),
        "path_services": sorted(path_services),
        "downstream_map": downstream_map,
        "upstream_map": upstream_map,
        "anomalous_services": sorted(anomalous_services),
        "pods_by_service": pods_by_service,
    }


def evaluate_rule_hints(candidate: dict, evidence_items: list, rule_context: dict, topology: dict, rules_config: dict) -> RuleEvaluation:
    hints = RuleHints()
    active_rules: list[str] = []
    notes: list[str] = []
    records = _flatten_records(evidence_items)
    service = candidate.get("service")
    pod = candidate.get("pod")
    entity_type = candidate.get("entity_type", "service")
    rule_map = _rule_map(rules_config)
    adjustments = _score_adjustments(rules_config)
    resource_thresholds = _local_resource_thresholds(rules_config)
    downstream_map = rule_context.get("downstream_map", {})
    upstream_map = rule_context.get("upstream_map", {})
    path_services = set(rule_context.get("path_services", []))
    trace_targets = set(rule_context.get("trace_targets_by_service", {}).get(service, []))
    pod_summaries = list(rule_context.get("pod_score_summary", {}).get(service, []))
    pod_summaries.sort(key=lambda item: item["score"], reverse=True)

    local_resource_support = any(_has_material_local_resource_signal(record, resource_thresholds) for record in records)
    local_failure_support = any(_has_local_failure_signal(record) for record in records)
    symptom_only_signal = bool(records) and all(
        record.metric_or_pattern in SYMPTOM_PATTERNS or record.source in {"log", "trace"} for record in records
    )

    if local_resource_support or local_failure_support:
        hints.local_resource_support = True
        active_rules.append("local_resource_support")
        notes.append("Strong local resource or failure-like evidence supports a local fault hypothesis.")

    if symptom_only_signal and not hints.local_resource_support:
        hints.symptom_only_signal = True
        active_rules.append("symptom_only_signal")
        notes.append("Evidence is mostly symptom-like and lacks strong local support.")

    if pod_summaries:
        shared_rule = rule_map.get("shared_multi_pod_anomaly", {})
        local_rule = rule_map.get("single_pod_local_anomaly", {})
        strong_threshold = float(
            local_rule.get(
                "strong_pod_score_threshold",
                rules_config.get("settings", {}).get("min_strong_pod_score", 0.60),
            )
        )
        min_relative = float(shared_rule.get("min_sibling_relative_score", 0.70))
        max_relative = float(local_rule.get("max_sibling_relative_score", 0.45))
        min_affected = int(shared_rule.get("min_affected_pods", 2))
        strongest = pod_summaries[0]["score"]
        strong_peers = [
            item
            for item in pod_summaries
            if item["score"] >= strong_threshold
            and item["score"] >= strongest * min_relative
        ]
        sibling_scores = [item["score"] for item in pod_summaries[1:]]
        next_best = sibling_scores[0] if sibling_scores else 0.0

        if len(strong_peers) >= min_affected:
            hints.shared_issue_hint = True
            active_rules.append("shared_multi_pod_anomaly")
            notes.append(
                "Multiple sibling pods are similarly anomalous, which points to a shared cause."
            )

        if strongest >= strong_threshold and next_best <= strongest * max_relative:
            if entity_type == "service" or (
                entity_type == "pod" and pod and pod == pod_summaries[0]["pod"]
            ):
                hints.single_pod_local_hint = True
                active_rules.append("single_pod_local_anomaly")
                notes.append(
                    "One pod is much stronger than its siblings, which points to a local pod issue."
                )

    if trace_targets and not hints.local_resource_support:
        hints.dependency_symptom_hint = True
        active_rules.append("downstream_edge_symptom")
        notes.append(
            "Trace anomalies are concentrated on downstream edges, so this may be a symptom carrier."
        )

    downstream_candidates = trace_targets | set(downstream_map.get(service, set()))
    if hints.shared_issue_hint and hints.local_resource_support and downstream_candidates:
        service_evidence = rule_context.get("service_evidence", {})
        direct_failure_downstreams = []
        for downstream in sorted(downstream_candidates):
            downstream_records = _flatten_records(service_evidence.get(downstream, []))
            downstream_records.extend(
                record
                for _, pod_evidence_items in rule_context.get("pods_by_service", {}).get(
                    downstream, []
                )
                for record in _flatten_records(pod_evidence_items)
            )
            if _has_direct_local_failure_evidence(
                downstream_records, resource_thresholds
            ):
                direct_failure_downstreams.append(downstream)
        if direct_failure_downstreams:
            hints.dependency_symptom_hint = True
            hints.downstream_local_failure_hint = True
            active_rules.append("downstream_local_failure")
            notes.append(
                "Sibling pods show similar failure-like symptoms while a downstream dependency has direct local failure evidence, so this service may be a propagated failure carrier."
            )

    if service in set(rule_context.get("shared_downstream_targets", [])):
        hints.shared_downstream_dependency_hint = True
        active_rules.append("shared_downstream_dependency")
        notes.append("Several anomalous upstream services point to this same downstream dependency.")

    candidate_ts = None
    if entity_type == "pod" and pod:
        candidate_ts = rule_context.get("pod_first_seen", {}).get(pod)
    elif service:
        candidate_ts = rule_context.get("service_first_seen", {}).get(service)
    time_margin = pd.Timedelta(seconds=int(rules_config.get("settings", {}).get("temporal_margin_seconds", 45)))
    connected_services = set(downstream_map.get(service, set())) | set(upstream_map.get(service, set())) if service else set()
    connected_service_times = {
        related: rule_context.get("service_first_seen", {}).get(related)
        for related in connected_services
        if rule_context.get("service_first_seen", {}).get(related) is not None
    }
    if candidate_ts is not None and connected_service_times:
        if any(candidate_ts + time_margin <= related_ts for related_ts in connected_service_times.values()):
            hints.temporal_precedence_hint = True
            active_rules.append("earlier_anomaly_precedence")
            notes.append("This candidate becomes anomalous earlier than a nearby impacted dependency path.")
        if any(related_ts + time_margin <= candidate_ts for related_ts in connected_service_times.values()):
            hints.temporal_conflict = True
            active_rules.append("late_anomaly_penalty")
            notes.append("This candidate becomes anomalous later than nearby impacted components.")

    if service and (service in path_services or connected_services & path_services):
        hints.topology_support_hint = True
        active_rules.append("adjacent_path_support")
        notes.append("The candidate sits on or adjacent to the main anomalous topology path.")

    if service and service not in topology.get("services", []):
        hints.topology_conflict = True
        active_rules.append("topology_conflict")
        notes.append("The candidate service is not present in the configured topology.")
    elif service and not hints.local_resource_support and not hints.topology_support_hint and hints.symptom_only_signal:
        hints.topology_conflict = True
        active_rules.append("topology_conflict")
        notes.append("The candidate is weakly connected to the main anomalous path and mostly symptom-like.")

    score_adjustment = 0.0
    for hint_name, enabled in asdict(hints).items():
        if enabled:
            score_adjustment += float(adjustments.get(hint_name, 0.0))

    return RuleEvaluation(
        rule_hints=hints,
        active_rules=list(dict.fromkeys(active_rules)),
        notes=list(dict.fromkeys(notes)),
        score_adjustment=round(score_adjustment, 4),
    )


def run_light_checks(candidate: dict, topology: dict) -> list[str]:
    notes: list[str] = []
    service = candidate.get("service")
    rule_hints = candidate.get("rule_hints", {})
    if service and service not in topology.get("services", []):
        notes.append("service not present in topology")
    if candidate.get("dependency_boost", 0.0) > 0:
        notes.append("dependency-aware boost applied")
    if candidate.get("evidence_count", 0) < 2:
        notes.append("limited supporting evidence")
    if rule_hints.get("single_pod_local_hint"):
        notes.append("rule hint: single-pod-local pattern")
    if rule_hints.get("shared_issue_hint"):
        notes.append("rule hint: shared multi-pod pattern")
    if rule_hints.get("dependency_symptom_hint"):
        notes.append("rule hint: dependency-driven symptom pattern")
    if rule_hints.get("topology_conflict"):
        notes.append("rule hint: topology conflict")
    return notes
