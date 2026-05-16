from __future__ import annotations

from collections.abc import Iterable
from typing import Any


METRIC_TO_CANONICAL = {
    "cpu_usage_pct": "cpu_high",
    "memory_usage_pct": "memory_high",
    "restart_count": "restart_increase",
    "ready_ratio": "ready_drop",
    "error_count": "error_increase",
    "success_rate": "success_drop",
    "latency_p50": "latency_spike",
    "latency_p90": "latency_spike",
    "latency_p95": "latency_spike",
    "latency_p99": "latency_spike",
    "edge_latency_spike": "trace_edge_latency",
    "path_latency_spike": "trace_path_latency",
    "edge_failure_spike": "trace_edge_failure",
    "keyword_spike": "log_keyword_spike",
    "template_spike": "log_template_spike",
    "level_shift": "log_level_shift",
    "missing_data_gap": "missing_data_gap",
    "network_rx": "network_rx_tx",
    "network_tx": "network_rx_tx",
}


def _get_value(record: Any, field: str, default=None):
    if isinstance(record, dict):
        return record.get(field, default)
    return getattr(record, field, default)


def _metadata(record: Any) -> dict:
    value = _get_value(record, "metadata", {})
    return value if isinstance(value, dict) else {}


def _clamp_probability(value, default: float = 0.5) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = default
    return max(0.0, min(1.0, numeric))


def _relative_change_pct(value):
    try:
        return round(float(value) * 100.0, 2)
    except (TypeError, ValueError):
        return None


def canonicalize_evidence(record) -> str | None:
    metadata = _metadata(record)
    metric_or_pattern = str(_get_value(record, "metric_or_pattern", "") or "")
    source = str(_get_value(record, "source", "") or "")

    in_window_pattern = metadata.get("in_window_pattern")
    if in_window_pattern == "missing_data_gap":
        return "missing_data_gap"

    if metric_or_pattern in METRIC_TO_CANONICAL:
        return METRIC_TO_CANONICAL[metric_or_pattern]

    if source == "trace" and "latency" in metric_or_pattern:
        return "trace_edge_latency"
    if source == "trace" and "failure" in metric_or_pattern:
        return "trace_edge_failure"
    return None


def _iter_records(evidence_items: Iterable) -> Iterable:
    for item in evidence_items or []:
        anomaly_records = _get_value(item, "anomaly_records", None)
        if anomaly_records is None:
            yield item
            continue
        yield from anomaly_records


def extract_soft_evidence(evidence_items: list) -> list[dict]:
    merged: dict[str, dict] = {}
    for record in _iter_records(evidence_items):
        name = canonicalize_evidence(record)
        if not name:
            continue
        severity = _clamp_probability(_get_value(record, "severity", 0.5))
        metadata = _metadata(record)
        item = {
            "name": name,
            "severity": severity,
            "source": _get_value(record, "source", "unknown"),
            "raw_pattern": _get_value(record, "metric_or_pattern", ""),
            "summary": _get_value(record, "summary", ""),
            "baseline_value": _get_value(record, "baseline_value"),
            "abnormal_value": _get_value(record, "abnormal_value"),
            "delta": _get_value(record, "delta"),
            "delta_ratio": metadata.get("delta_ratio"),
            "ratio": metadata.get("ratio"),
            "relative_change_pct": _relative_change_pct(metadata.get("delta_ratio")),
            "service": metadata.get("service")
            or (
                _get_value(record, "entity_name")
                if _get_value(record, "entity_type") == "service"
                else None
            ),
            "pod": metadata.get("pod")
            or (
                _get_value(record, "entity_name")
                if _get_value(record, "entity_type") == "pod"
                else None
            ),
        }
        existing = merged.get(name)
        if not existing or severity > float(existing.get("severity", 0.0)):
            merged[name] = item
    return sorted(
        merged.values(),
        key=lambda item: (float(item.get("severity", 0.0)), str(item.get("name", ""))),
        reverse=True,
    )
