from __future__ import annotations

from collections import defaultdict

import pandas as pd

from rca_agent_skills.common.models import AnomalyRecord
from .schemas import TraceAnomalyFeature


def _clean_label(value) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _ratio(abnormal: float, baseline: float) -> float:
    if baseline <= 0:
        return 0.0
    return abnormal / baseline


def _volume_factor(count: int, min_count: int) -> float:
    return min(1.0, float(count) / max(float(min_count * 4), 1.0))


def _latency_severity(ratio: float, count: int, min_count: int, ratio_weight: float) -> float:
    return round(min(1.0, ratio_weight * min(ratio, 4.0) + 0.12 * _volume_factor(count, min_count)), 4)


def _failure_severity(
    failure_ratio: float,
    failure_rate: float,
    failure_count: int,
    sample_count: int,
    min_count: int,
    zero_baseline: bool,
) -> float:
    volume = _volume_factor(sample_count, min_count)
    failure_volume = min(1.0, float(failure_count) / max(float(min_count), 1.0))
    if zero_baseline:
        return round(min(1.0, 0.20 + 0.35 * failure_rate + 0.25 * volume + 0.20 * failure_volume), 4)
    return round(min(1.0, 0.28 * min(failure_ratio, 4.0) + 0.35 * failure_rate + 0.17 * volume + 0.20 * failure_volume), 4)


def _calibrate_trace_latency(
    severity_calibrator,
    *,
    baseline_values,
    abnormal_values,
    ratio: float,
    raw_severity: float,
    sample_count: int,
) -> tuple[float, float, str, dict, list[str]]:
    if severity_calibrator is None:
        return raw_severity, raw_severity, "heuristic", {}, []
    result = severity_calibrator.calibrate_trace_latency_severity(
        baseline_values=baseline_values,
        abnormal_values=abnormal_values,
        ratio=ratio,
        raw_severity=raw_severity,
        sample_count=sample_count,
    )
    return (
        float(result.get("severity", raw_severity)),
        float(result.get("raw_severity", raw_severity)),
        str(result.get("severity_method", "heuristic")),
        dict(result.get("calibration_metadata", {})),
        list(result.get("calibration_notes", [])),
    )


def _calibrate_trace_failure(
    severity_calibrator,
    *,
    baseline_failure_count: int,
    baseline_total_count: int,
    abnormal_failure_count: int,
    abnormal_total_count: int,
    failure_rate: float,
    failure_ratio: float,
    raw_severity: float,
) -> tuple[float, float, str, dict, list[str]]:
    if severity_calibrator is None:
        return raw_severity, raw_severity, "heuristic", {}, []
    result = severity_calibrator.calibrate_trace_failure_severity(
        baseline_failure_count=baseline_failure_count,
        baseline_total_count=baseline_total_count,
        abnormal_failure_count=abnormal_failure_count,
        abnormal_total_count=abnormal_total_count,
        failure_rate=failure_rate,
        failure_ratio=failure_ratio,
        raw_severity=raw_severity,
    )
    return (
        float(result.get("severity", raw_severity)),
        float(result.get("raw_severity", raw_severity)),
        str(result.get("severity_method", "heuristic")),
        dict(result.get("calibration_metadata", {})),
        list(result.get("calibration_notes", [])),
    )


def detect_trace_anomalies(
    baseline_df: pd.DataFrame,
    abnormal_df: pd.DataFrame,
    thresholds: dict,
    severity_calibrator=None,
) -> list[TraceAnomalyFeature]:
    features: list[TraceAnomalyFeature] = []
    min_count = int(thresholds.get("minimum_count", 3))
    ratio_threshold = float(thresholds.get("trace_spike_ratio_threshold", 1.8))
    min_failure_count = int(thresholds.get("trace_min_failure_count", 2))
    min_failure_rate = float(thresholds.get("trace_min_failure_rate", 0.2))

    def _new_edge_stats():
        return {"latencies": [], "failures": 0, "count": 0, "timestamps": []}

    def _add_edge_sample(stats: dict, key: tuple[str, ...], row) -> None:
        stats[key]["latencies"].append(float(row.get("duration", 0.0)))
        stats[key]["count"] += 1
        stats[key]["timestamps"].append(row.get("timestamp"))
        if row.get("status", "SUCCESS") != "SUCCESS" or str(row.get("status_code", "")).startswith("5"):
            stats[key]["failures"] += 1

    def _server_pod_lookup(frame: pd.DataFrame) -> dict[tuple[str, str, str], str]:
        lookup: dict[tuple[str, str, str], str] = {}
        server_spans = frame.loc[frame["span_kind"].astype(str).str.lower() == "server"]
        for _, row in server_spans.iterrows():
            trace_id = _clean_label(row.get("trace_id"))
            parent_span_id = _clean_label(row.get("parent_span_id"))
            service = _clean_label(row.get("service"))
            pod = _clean_label(row.get("pod"))
            if trace_id and parent_span_id and service and pod:
                lookup.setdefault((trace_id, parent_span_id, service), pod)
        return lookup

    def edge_stats(frame: pd.DataFrame) -> tuple[dict[tuple[str, str], dict], dict[tuple[str, str, str], dict], dict[tuple[str, str, str], dict]]:
        service_stats = defaultdict(_new_edge_stats)
        source_pod_stats = defaultdict(_new_edge_stats)
        target_pod_stats = defaultdict(_new_edge_stats)
        server_pods = _server_pod_lookup(frame)
        client_spans = frame.loc[frame["span_kind"].str.lower() == "client"]
        for _, row in client_spans.iterrows():
            src = _clean_label(row.get("service"))
            dst = _clean_label(row.get("peer_service"))
            if not src or not dst:
                continue
            _add_edge_sample(service_stats, (src, dst), row)
            source_pod = _clean_label(row.get("pod"))
            if source_pod:
                _add_edge_sample(source_pod_stats, (src, source_pod, dst), row)
            trace_id = _clean_label(row.get("trace_id"))
            span_id = _clean_label(row.get("span_id"))
            target_pod = server_pods.get((trace_id, span_id, dst))
            if target_pod:
                _add_edge_sample(target_pod_stats, (dst, target_pod, src), row)
        return service_stats, source_pod_stats, target_pod_stats

    def _clean_timestamps(stats: dict) -> pd.Series:
        return pd.to_datetime(pd.Series(stats.get("timestamps", [])), utc=True, errors="coerce").dropna()

    def _append_edge_features(
        *,
        base_edges: dict,
        abn_edges: dict,
        entity_type: str,
        edge_role: str,
    ) -> None:
        for edge in sorted(set(base_edges) | set(abn_edges)):
            if entity_type == "service":
                src, dst = edge
                service = dst
                pod = None
                peer_service = src
                edge_source_service = src
                edge_target_service = dst
                edge_source_pod = None
                edge_target_pod = None
                entity_name = service
            elif edge_role == "source_pod":
                src, pod, dst = edge
                service = src
                peer_service = dst
                edge_source_service = src
                edge_target_service = dst
                edge_source_pod = pod
                edge_target_pod = None
                entity_name = pod
            else:
                dst, pod, src = edge
                service = dst
                peer_service = src
                edge_source_service = src
                edge_target_service = dst
                edge_source_pod = None
                edge_target_pod = pod
                entity_name = pod
            base_count = base_edges[edge]["count"]
            abn_count = abn_edges[edge]["count"]
            clean_edge_ts = _clean_timestamps(abn_edges[edge])
            if abn_count < min_count:
                continue
            base_latency = sum(base_edges[edge]["latencies"]) / max(base_count, 1)
            abn_latency = sum(abn_edges[edge]["latencies"]) / max(abn_count, 1)
            latency_ratio = _ratio(abn_latency, base_latency)
            if latency_ratio >= ratio_threshold:
                raw_severity = _latency_severity(latency_ratio, abn_count, min_count, 0.26)
                severity, raw_severity, severity_method, calibration_metadata, calibration_notes = _calibrate_trace_latency(
                    severity_calibrator,
                    baseline_values=base_edges[edge]["latencies"],
                    abnormal_values=abn_edges[edge]["latencies"],
                    ratio=latency_ratio,
                    raw_severity=raw_severity,
                    sample_count=abn_count,
                )
                features.append(
                    TraceAnomalyFeature(
                        entity_type=entity_type,
                        entity_name=entity_name,
                        service=service,
                        pod=pod,
                        anomaly_type="edge_latency_spike",
                        abnormal_value=abn_latency,
                        baseline_value=base_latency,
                        ratio=latency_ratio,
                        severity=severity,
                        peer_service=peer_service,
                        edge_role=edge_role,
                        edge_source_service=edge_source_service,
                        edge_target_service=edge_target_service,
                        edge_source_pod=edge_source_pod,
                        edge_target_pod=edge_target_pod,
                        first_seen_ts=clean_edge_ts.min().isoformat() if not clean_edge_ts.empty else None,
                        last_seen_ts=clean_edge_ts.max().isoformat() if not clean_edge_ts.empty else None,
                        raw_severity=raw_severity,
                        severity_method=severity_method,
                        calibration_metadata=calibration_metadata,
                        calibration_notes=calibration_notes,
                    )
                )
            base_failure = float(base_edges[edge]["failures"]) / max(base_count, 1)
            abn_failure_count = int(abn_edges[edge]["failures"])
            abn_failure = float(abn_failure_count) / max(abn_count, 1)
            zero_baseline_failure = base_failure <= 0
            failure_ratio = _ratio(abn_failure, base_failure)
            failure_spike = (
                abn_failure_count >= min_failure_count
                and abn_failure >= min_failure_rate
                if zero_baseline_failure
                else abn_failure_count >= 1 and failure_ratio >= ratio_threshold
            )
            if failure_spike:
                raw_severity = _failure_severity(
                    failure_ratio,
                    abn_failure,
                    abn_failure_count,
                    abn_count,
                    min_count,
                    zero_baseline_failure,
                )
                severity, raw_severity, severity_method, calibration_metadata, calibration_notes = _calibrate_trace_failure(
                    severity_calibrator,
                    baseline_failure_count=int(base_edges[edge]["failures"]),
                    baseline_total_count=base_count,
                    abnormal_failure_count=abn_failure_count,
                    abnormal_total_count=abn_count,
                    failure_rate=abn_failure,
                    failure_ratio=failure_ratio,
                    raw_severity=raw_severity,
                )
                features.append(
                    TraceAnomalyFeature(
                        entity_type=entity_type,
                        entity_name=entity_name,
                        service=service,
                        pod=pod,
                        anomaly_type="edge_failure_spike",
                        abnormal_value=abn_failure,
                        baseline_value=base_failure,
                        ratio=failure_ratio,
                        severity=severity,
                        peer_service=peer_service,
                        edge_role=edge_role,
                        edge_source_service=edge_source_service,
                        edge_target_service=edge_target_service,
                        edge_source_pod=edge_source_pod,
                        edge_target_pod=edge_target_pod,
                        first_seen_ts=clean_edge_ts.min().isoformat() if not clean_edge_ts.empty else None,
                        last_seen_ts=clean_edge_ts.max().isoformat() if not clean_edge_ts.empty else None,
                        raw_severity=raw_severity,
                        severity_method=severity_method,
                        calibration_metadata=calibration_metadata,
                        calibration_notes=calibration_notes,
                    )
                )

    base_edges, base_source_pod_edges, base_target_pod_edges = edge_stats(baseline_df)
    abn_edges, abn_source_pod_edges, abn_target_pod_edges = edge_stats(abnormal_df)
    _append_edge_features(
        base_edges=base_edges,
        abn_edges=abn_edges,
        entity_type="service",
        edge_role="service_edge",
    )
    _append_edge_features(
        base_edges=base_target_pod_edges,
        abn_edges=abn_target_pod_edges,
        entity_type="pod",
        edge_role="target_pod",
    )

    for entity_key in ["service", "pod"]:
        base_group = baseline_df.groupby(entity_key)
        abn_group = abnormal_df.groupby(entity_key)
        for entity_name in sorted(set(base_group.groups.keys()) | set(abn_group.groups.keys())):
            if entity_name == "":
                continue
            base_slice = base_group.get_group(entity_name) if entity_name in base_group.groups else pd.DataFrame(columns=baseline_df.columns)
            abn_slice = abn_group.get_group(entity_name) if entity_name in abn_group.groups else pd.DataFrame(columns=abnormal_df.columns)
            if len(abn_slice) < min_count:
                continue
            base_latency = float(base_slice["duration"].mean()) if len(base_slice) else 0.0
            abn_latency = float(abn_slice["duration"].mean()) if len(abn_slice) else 0.0
            latency_ratio = _ratio(abn_latency, base_latency)
            if latency_ratio >= ratio_threshold:
                service = entity_name if entity_key == "service" else str(abn_slice.iloc[0].get("service", "unknown"))
                pod = entity_name if entity_key == "pod" else None
                clean_ts = pd.to_datetime(abn_slice["timestamp"], utc=True, errors="coerce").dropna()
                raw_severity = _latency_severity(latency_ratio, len(abn_slice), min_count, 0.22)
                severity, raw_severity, severity_method, calibration_metadata, calibration_notes = _calibrate_trace_latency(
                    severity_calibrator,
                    baseline_values=base_slice["duration"] if "duration" in base_slice.columns else [],
                    abnormal_values=abn_slice["duration"] if "duration" in abn_slice.columns else [],
                    ratio=latency_ratio,
                    raw_severity=raw_severity,
                    sample_count=len(abn_slice),
                )
                features.append(
                    TraceAnomalyFeature(
                        entity_type=entity_key,
                        entity_name=str(entity_name),
                        service=service,
                        pod=pod,
                        anomaly_type="path_latency_spike",
                        abnormal_value=abn_latency,
                        baseline_value=base_latency,
                        ratio=latency_ratio,
                        severity=severity,
                        first_seen_ts=clean_ts.min().isoformat() if not clean_ts.empty else None,
                        last_seen_ts=clean_ts.max().isoformat() if not clean_ts.empty else None,
                        raw_severity=raw_severity,
                        severity_method=severity_method,
                        calibration_metadata=calibration_metadata,
                        calibration_notes=calibration_notes,
                    )
                )
    return features


def to_anomaly_records(features: list[TraceAnomalyFeature]) -> list[AnomalyRecord]:
    def _trace_summary(item: TraceAnomalyFeature) -> str:
        if item.anomaly_type == "edge_failure_spike":
            calibration_metadata = item.calibration_metadata or {}
            baseline_failures = calibration_metadata.get("baseline_failure_count")
            baseline_total = calibration_metadata.get("baseline_total_count")
            abnormal_failures = calibration_metadata.get("abnormal_failure_count")
            abnormal_total = calibration_metadata.get("abnormal_total_count")
            if all(
                value is not None
                for value in [
                    baseline_failures,
                    baseline_total,
                    abnormal_failures,
                    abnormal_total,
                ]
            ):
                return (
                    "edge_failure_spike rate "
                    f"{float(item.baseline_value):.4f}->{float(item.abnormal_value):.4f} "
                    f"({int(baseline_failures)}/{int(baseline_total)}"
                    f"->{int(abnormal_failures)}/{int(abnormal_total)})"
                )
            return (
                "edge_failure_spike rate "
                f"{float(item.baseline_value):.4f}->{float(item.abnormal_value):.4f}"
            )
        return f"{item.anomaly_type} {item.baseline_value:.2f}->{item.abnormal_value:.2f}"

    records: list[AnomalyRecord] = []
    for item in features:
        records.append(
            AnomalyRecord(
                source="trace",
                entity_type=item.entity_type,
                entity_name=item.entity_name,
                metric_or_pattern=item.anomaly_type,
                abnormal_value=item.abnormal_value,
                baseline_value=item.baseline_value,
                delta=item.abnormal_value - item.baseline_value,
                zscore=None,
                severity=item.severity,
                summary=_trace_summary(item),
                metadata={
                    "ratio": item.ratio,
                    "service": item.service,
                    "pod": item.pod,
                    "peer_service": item.peer_service,
                    "edge_role": item.edge_role,
                    "edge_source_service": item.edge_source_service,
                    "edge_target_service": item.edge_target_service,
                    "edge_source_pod": item.edge_source_pod,
                    "edge_target_pod": item.edge_target_pod,
                    "first_seen_ts": item.first_seen_ts,
                    "last_seen_ts": item.last_seen_ts,
                    "raw_severity": item.raw_severity,
                    "severity_method": item.severity_method,
                    "calibration_metadata": item.calibration_metadata,
                    "calibration_notes": item.calibration_notes,
                },
            )
        )
    return records
