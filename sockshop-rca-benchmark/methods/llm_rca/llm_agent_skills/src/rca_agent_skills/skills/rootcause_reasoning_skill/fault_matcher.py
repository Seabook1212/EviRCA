from __future__ import annotations

from rca_agent_skills.common.constants import EXCEPTION_KEYWORDS, NETWORK_KEYWORDS


DEFAULT_LOCAL_FAULT_MIN_DELTA = {
    "cpu_usage_pct": 0.20,
    "memory_usage_pct": 0.10,
    "restart_count": 0.10,
    "ready_ratio": 0.10,
}

FAULT_SUPPORT_PRIORITIES = {
    "pod_failure": [
        "restart_count",
        "ready_ratio",
        "error_count",
        "cpu_usage_pct",
        "memory_usage_pct",
    ],
    "cpu_stress": ["cpu_usage_pct", "restart_count", "ready_ratio", "error_count"],
    "memory_stress": [
        "memory_usage_pct",
        "restart_count",
        "ready_ratio",
        "error_count",
    ],
    "network_delay": [
        "latency_p99",
        "latency_p95",
        "latency_p90",
        "latency_p50",
        "edge_latency_spike",
        "path_latency_spike",
        "network_rx",
        "network_tx",
    ],
    "network_loss": [
        "edge_failure_spike",
        "error_count",
        "success_rate",
        "network_rx",
        "network_tx",
        "latency_p99",
        "latency_p95",
        "latency_p90",
        "latency_p50",
    ],
    "network_partition": [
        "missing_data_gap",
        "latency_p99",
        "latency_p95",
        "latency_p90",
        "latency_p50",
        "success_rate",
        "network_rx",
        "network_tx",
        "edge_failure_spike",
        "error_count",
    ],
    "io_fault": [
        "restart_count",
        "ready_ratio",
        "error_count",
        "edge_failure_spike",
        "success_rate",
        "latency_p99",
        "latency_p95",
        "latency_p90",
        "latency_p50",
        "template_spike",
    ],
    "exception_injection": ["keyword_spike", "template_spike", "level_shift"],
}

CONNECTIVITY_GAP_METRICS = {
    "request_rate",
    "success_rate",
    "error_count",
    "latency_p50",
    "latency_p90",
    "latency_p95",
    "latency_p99",
    "ready_ratio",
}
STATEFUL_SERVICE_TOKENS = {
    "-db",
    "db",
    "database",
    "mongo",
    "mysql",
    "postgres",
    "redis",
    "rabbitmq",
    "queue",
}
EXPLICIT_EXCEPTION_TOKENS = {
    "exception",
    "runtimeexception",
    "throwexception",
    "throwable",
    "stacktrace",
    "stack trace",
    "java.lang",
    "rule.execute",
    "byteman",
    "caught throw",
}


def _local_fault_thresholds(rules_config: dict | None) -> dict[str, float]:
    configured = (
        (rules_config or {})
        .get("settings", {})
        .get("local_resource_min_delta_ratio", {})
    )
    return {**DEFAULT_LOCAL_FAULT_MIN_DELTA, **configured}


def _metric_value(item, metadata_key: str, fallback_attr: str) -> float:
    value = item.metadata.get(metadata_key)
    if value is None:
        value = getattr(item, fallback_attr)
    return float(value or 0.0)


def _support_text(record) -> str:
    if record.source == "metric":
        pod = record.metadata.get("pod")
        service = record.metadata.get("service") or record.entity_name
        scope = (
            f"pod {pod}"
            if record.entity_type == "pod" and pod
            else f"service {service}"
        )
        return f"{scope} {record.metric_or_pattern}: {record.summary}"
    if record.source == "trace":
        source = record.metadata.get("edge_source_service")
        target = record.metadata.get("edge_target_service")
        role = record.metadata.get("edge_role")
        pod = record.metadata.get("pod")
        service = record.metadata.get("service") or record.entity_name
        if source and target:
            if role == "source_pod" and pod:
                return f"source pod {pod} {source}->{target} {record.summary}"
            if role == "target_pod" and pod:
                return f"target pod {pod} {source}->{target} {record.summary}"
            return f"service {service} {source}->{target} {record.summary}"
        scope = f"pod {pod}" if record.entity_type == "pod" and pod else f"service {service}"
        return f"{scope} {record.summary}"
    return record.summary


def _is_connectivity_gap(record) -> bool:
    if record.source != "metric":
        return False
    if record.metric_or_pattern not in CONNECTIVITY_GAP_METRICS:
        return False
    if record.metadata.get("in_window_pattern") != "missing_data_gap":
        return False
    try:
        return float(record.metadata.get("gap_seconds") or 0.0) > 0.0
    except (TypeError, ValueError):
        return True


def _is_stateful_entity(service: str | None, pod: str | None) -> bool:
    text = f"{service or ''} {pod or ''}".lower()
    return any(token in text for token in STATEFUL_SERVICE_TOKENS)


def _select_support(
    fault_type: str,
    record_support: list[tuple[str, str]],
    fallback: list[str],
    support_limit: int = 4,
) -> list[str]:
    priorities = FAULT_SUPPORT_PRIORITIES.get(fault_type, [])
    selected: list[str] = []
    for metric in priorities:
        for pattern, text in record_support:
            if pattern == metric or pattern.startswith(metric):
                selected.append(text)
    if not selected:
        for _, text in record_support:
            selected.append(text)
    if not selected:
        selected.extend(fallback)
    return list(dict.fromkeys(selected))[:support_limit]


def match_fault_types(
    evidence_items: list,
    topology: dict,
    rule_hints: dict[str, bool] | None = None,
    rules_config: dict | None = None,
    support_limit: int = 4,
) -> list[tuple[str, float, list[str], list[str], float]]:
    rule_hints = rule_hints or {}
    local_thresholds = _local_fault_thresholds(rules_config)
    text_evidence: list[str] = []
    record_support: list[tuple[str, str]] = []
    aggregate_score = 0.0
    dependency_boost = 0.0
    max_evidence_score = 0.0

    has_cpu = False
    has_memory = False
    has_restart = False
    has_latency = False
    has_network = False
    has_connectivity_gap = False
    has_io_fault = False
    has_stateful_entity = False
    has_exception = False
    has_explicit_exception = False
    explicit_exception_count = 0.0
    connectivity_gap_count = 0
    connectivity_gap_metrics: set[str] = set()
    local_resource_metrics_seen: set[str] = set()
    non_resource_signal_seen = False
    background_log_count = 0
    informative_log_count = 0

    for evidence in evidence_items:
        aggregate_score += evidence.score
        max_evidence_score = max(max_evidence_score, float(evidence.score))
        text_evidence.extend(evidence.supporting_evidence)
        for item in evidence.anomaly_records:
            support_text = _support_text(item)
            record_support.append((item.metric_or_pattern, support_text))
            summary = item.summary.lower()
            peer = item.metadata.get("peer_service")
            edge_target = item.metadata.get("edge_target_service")
            has_stateful_entity = has_stateful_entity or _is_stateful_entity(
                item.metadata.get("service"), item.metadata.get("pod")
            )
            if edge_target and item.metadata.get("service") == edge_target:
                dependency_boost += 0.05
            elif peer and any(edge.get("target") == peer for edge in topology.get("edges", [])):
                dependency_boost += 0.05

            if item.source == "metric":
                ratio = abs(float(item.metadata.get("delta_ratio", 0.0)))
                if item.metric_or_pattern in DEFAULT_LOCAL_FAULT_MIN_DELTA:
                    local_resource_metrics_seen.add(item.metric_or_pattern)
                else:
                    non_resource_signal_seen = True
                if item.metric_or_pattern == "cpu_usage_pct" and ratio >= float(
                    local_thresholds.get("cpu_usage_pct", 0.20)
                ):
                    has_cpu = True
                if item.metric_or_pattern == "memory_usage_pct" and ratio >= float(
                    local_thresholds.get("memory_usage_pct", 0.10)
                ):
                    has_memory = True
                if item.metric_or_pattern == "restart_count":
                    if _metric_value(
                        item, "abnormal_max", "abnormal_value"
                    ) > _metric_value(item, "baseline_max", "baseline_value"):
                        has_restart = True
                is_connectivity_gap = _is_connectivity_gap(item)
                if item.metric_or_pattern == "ready_ratio" and not is_connectivity_gap:
                    if _metric_value(
                        item, "abnormal_min", "abnormal_value"
                    ) < _metric_value(item, "baseline_min", "baseline_value"):
                        has_restart = True
                if item.metric_or_pattern.startswith("latency_p") and ratio >= 0.10:
                    has_latency = True
                if is_connectivity_gap:
                    has_connectivity_gap = True
                    connectivity_gap_count += 1
                    connectivity_gap_metrics.add(item.metric_or_pattern)
                    record_support.append(("missing_data_gap", support_text))
                if item.metric_or_pattern == "error_count" and (
                    item.abnormal_value or 0
                ) > (item.baseline_value or 0):
                    has_io_fault = True

            if item.source == "trace":
                non_resource_signal_seen = True
                if item.metric_or_pattern in {
                    "edge_latency_spike",
                    "path_latency_spike",
                }:
                    has_latency = True
                if item.metric_or_pattern == "edge_failure_spike":
                    has_io_fault = True

            if item.source == "log":
                non_resource_signal_seen = True
                lower_summary = summary.lower()
                is_background_noise = bool(item.metadata.get("background_noise"))
                if is_background_noise:
                    background_log_count += 1
                else:
                    informative_log_count += 1
                if (not is_background_noise) and any(
                    keyword in lower_summary for keyword in NETWORK_KEYWORDS
                ):
                    has_network = True
                if (not is_background_noise) and any(
                    keyword in lower_summary for keyword in EXCEPTION_KEYWORDS
                ):
                    has_exception = True
                if (not is_background_noise) and any(
                    token in lower_summary for token in EXPLICIT_EXCEPTION_TOKENS
                ):
                    has_exception = True
                    has_explicit_exception = True
                    explicit_exception_count += float(item.abnormal_value or 0.0)

    joined = " | ".join(text_evidence).lower()
    matches: list[tuple[str, float, list[str], list[str], float]] = []
    evidence_count = max(len(evidence_items), 1)
    base_score = min(
        0.82,
        0.5 * (aggregate_score / evidence_count)
        + 0.3 * max_evidence_score
        + 0.02 * min(len(text_evidence), 4),
    )

    def add(
        fault_type: str, weight: float, reason: str, max_score: float | None = None
    ) -> None:
        rule_notes: list[str] = []
        adjusted_weight = weight
        if rule_hints.get("single_pod_local_hint") and fault_type in {
            "cpu_stress",
            "memory_stress",
            "pod_failure",
            "exception_injection",
        }:
            adjusted_weight += 0.08
            rule_notes.append("single-pod-local hint strengthens this local fault type")
        if rule_hints.get("local_resource_support") and fault_type in {
            "cpu_stress",
            "memory_stress",
            "pod_failure",
        }:
            adjusted_weight += 0.08
            rule_notes.append("local-resource hint strengthens this local fault type")
        if rule_hints.get("shared_issue_hint") and fault_type in {
            "network_delay",
            "network_loss",
            "network_partition",
            "io_fault",
        }:
            adjusted_weight += 0.06
            rule_notes.append(
                "shared multi-pod hint strengthens shared-cause hypotheses"
            )
        if rule_hints.get("shared_downstream_dependency_hint") and fault_type in {
            "network_delay",
            "network_loss",
            "network_partition",
            "io_fault",
        }:
            adjusted_weight += 0.08
            rule_notes.append(
                "shared downstream hint strengthens dependency-related hypotheses"
            )
        if rule_hints.get("dependency_symptom_hint") and fault_type in {
            "network_delay",
            "network_loss",
            "network_partition",
            "io_fault",
        }:
            adjusted_weight += 0.04
            rule_notes.append(
                "dependency symptom hint supports downstream-propagation reasoning"
            )
        if rule_hints.get("symptom_only_signal") and fault_type in {
            "cpu_stress",
            "memory_stress",
            "pod_failure",
            "exception_injection",
        }:
            adjusted_weight -= 0.07
            rule_notes.append(
                "symptom-only hint weakens overconfident local-fault assumptions"
            )
        if rule_hints.get("temporal_precedence_hint"):
            adjusted_weight += 0.05
            rule_notes.append("temporal precedence hint slightly raises plausibility")
        if rule_hints.get("temporal_conflict"):
            adjusted_weight -= 0.07
            rule_notes.append("late anomaly hint lowers plausibility")
        if rule_hints.get("topology_support_hint"):
            adjusted_weight += 0.04
            rule_notes.append("topology support hint raises plausibility")
        if rule_hints.get("topology_conflict"):
            adjusted_weight -= 0.09
            rule_notes.append("topology conflict lowers plausibility")

        score = min(
            0.98, max(0.05, base_score + adjusted_weight + min(0.05, dependency_boost))
        )
        if max_score is not None:
            score = min(score, max_score)
        if background_log_count and not informative_log_count:
            score = max(0.12, score - 0.22)
            reason = f"{reason} Background-style database/listener logs dominate the signal, so confidence is reduced."
        matches.append(
            (
                fault_type,
                round(score, 4),
                _select_support(
                    fault_type, record_support, text_evidence, support_limit
                ),
                [reason] + rule_notes[:2],
                dependency_boost,
            )
        )

    if has_cpu:
        add("cpu_stress", 0.28, "CPU anomaly is materially elevated")
    if has_memory or "oom" in joined or "outofmemory" in joined:
        add("memory_stress", 0.30, "Memory pressure or OOM-like evidence is present")
    if has_restart or "killed" in joined or "crash" in joined:
        add("pod_failure", 0.34, "Restart or readiness symptoms suggest pod failure")
    if has_latency:
        add(
            "network_delay",
            0.18,
            "Latency-driven trace or metric anomalies are present",
        )
    if has_connectivity_gap:
        gap_metric_count = len(connectivity_gap_metrics)
        gap_weight = 0.42 + min(0.18, 0.04 * max(gap_metric_count - 1, 0))
        if has_restart:
            gap_weight -= 0.08
        gap_reason = (
            "Missing latency/success telemetry inside the abnormal window strongly "
            "suggests connectivity interruption or partition"
        )
        if gap_metric_count > 1:
            metrics = ", ".join(sorted(connectivity_gap_metrics))
            gap_reason = (
                f"Multiple abnormal-window telemetry gaps ({metrics}) strongly "
                "suggest a network partition or connectivity interruption rather "
                "than ordinary latency/loss"
            )
        if has_restart:
            gap_reason = (
                f"{gap_reason}; restart/readiness evidence is also present, so "
                "pod failure should remain a competing explanation"
            )
        add(
            "network_partition",
            gap_weight,
            gap_reason,
        )
        add(
            "network_loss",
            0.16,
            "Missing telemetry may accompany packet loss, but sustained data gaps are more partition-like than ordinary loss",
            max_score=0.78,
        )
    if has_network:
        add("network_loss", 0.24, "Network-related log evidence is present")
        add(
            "network_partition",
            0.20,
            "Connection failures may indicate partition-like behavior",
        )
    if has_io_fault:
        add("io_fault", 0.20, "Failing requests and downstream errors are visible")
    if has_stateful_entity and has_restart and (has_latency or has_network or has_exception):
        add(
            "io_fault",
            0.32,
            "Stateful dependency shows local restart/readiness symptoms together with latency, connection, or exception evidence; this is compatible with an IO fault on the dependency.",
        )
    if has_exception:
        if has_explicit_exception:
            exception_weight = 0.42
            if explicit_exception_count >= 100:
                exception_weight += 0.06
            add(
                "exception_injection",
                exception_weight,
                "High-volume explicit exception evidence is present",
            )
        else:
            add("exception_injection", 0.24, "Exception-like evidence is present")
    if not matches:
        if local_resource_metrics_seen and not non_resource_signal_seen:
            if "memory_usage_pct" in local_resource_metrics_seen:
                add(
                    "memory_stress",
                    0.02,
                    "Weak memory shift is present but below the material local-fault threshold",
                    max_score=0.35,
                )
            elif "cpu_usage_pct" in local_resource_metrics_seen:
                add(
                    "cpu_stress",
                    0.02,
                    "Weak CPU shift is present but below the material local-fault threshold",
                    max_score=0.35,
                )
            else:
                add(
                    "pod_failure",
                    0.02,
                    "Weak local resource signal is present but below the material local-fault threshold",
                    max_score=0.35,
                )
        else:
            add(
                "network_delay",
                0.06,
                "Fallback: strongest available evidence is nonspecific latency/symptom evidence",
                max_score=0.45,
            )

    deduped = {}
    for fault_type, score, support, notes, boost in matches:
        if fault_type not in deduped or score > deduped[fault_type][0]:
            deduped[fault_type] = (score, support, notes, boost)
    ordered = sorted(
        [
            (fault_type, score, support, notes, boost)
            for fault_type, (score, support, notes, boost) in deduped.items()
        ],
        key=lambda item: item[1],
        reverse=True,
    )
    selected = ordered[:2]
    if has_explicit_exception and "exception_injection" in deduped and not any(
        item[0] == "exception_injection" for item in selected
    ):
        exception_item = next(
            item for item in ordered if item[0] == "exception_injection"
        )
        if len(selected) >= 2:
            selected[-1] = exception_item
        else:
            selected.append(exception_item)
        selected = sorted(selected, key=lambda item: item[1], reverse=True)
    if has_connectivity_gap and "network_partition" in deduped and not any(
        item[0] == "network_partition" for item in selected
    ):
        partition = next(item for item in ordered if item[0] == "network_partition")
        if len(selected) >= 2:
            selected[-1] = partition
        else:
            selected.append(partition)
        selected = sorted(selected, key=lambda item: item[1], reverse=True)
    return selected[:2]
