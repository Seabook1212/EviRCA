from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from rca_agent_skills.common.models import AnomalyRecord
from .schemas import MetricAnomalyFeatures


EVENT_INCREASE_METRICS = {"restart_count", "error_count"}
EVENT_DECREASE_METRICS = {"ready_ratio"}


def _robust_zscore(values: pd.Series, target: float) -> float:
    clean = values.dropna().astype(float)
    if clean.empty:
        return 0.0
    median = float(clean.median())
    mad = float(np.median(np.abs(clean - median)))
    if mad == 0:
        return 0.0
    return float((target - median) / (1.4826 * mad))


def _std(values: pd.Series) -> float:
    clean = values.dropna().astype(float)
    if len(clean) <= 1:
        return 0.0
    return float(clean.std(ddof=0))


def _relative_delta(delta: float, baseline: float) -> float:
    if baseline == 0:
        return 0.0
    return delta / max(abs(baseline), 1e-9)


def _clean_values(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").dropna()


def _group_key_sort_key(key) -> tuple:
    if not isinstance(key, tuple):
        key = (key,)
    return tuple("" if pd.isna(value) else str(value) for value in key)


def _metric_summary(item: MetricAnomalyFeatures) -> str:
    if item.in_window_pattern == "missing_data_gap":
        return (
            f"{item.metric} has an abnormal-window data gap of {(item.gap_seconds or 0.0):.0f}s"
        )
    if item.in_window_pattern in {"sudden_increase", "sudden_decrease", "short_burst"}:
        return (
            f"{item.metric} {item.in_window_pattern} inside abnormal window: "
            f"{(item.pre_segment_mean or 0.0):.3f}->{(item.segment_mean or 0.0):.3f}"
        )
    if item.metric in EVENT_INCREASE_METRICS:
        return (
            f"{item.metric} mean shifted from {item.baseline_mean:.3f} to {item.abnormal_mean:.3f}; "
            f"max shifted from {(item.baseline_max or 0.0):.3f} to {(item.abnormal_max or 0.0):.3f}"
        )
    if item.metric in EVENT_DECREASE_METRICS:
        return (
            f"{item.metric} mean shifted from {item.baseline_mean:.3f} to {item.abnormal_mean:.3f}; "
            f"min shifted from {(item.baseline_min or 0.0):.3f} to {(item.abnormal_min or 0.0):.3f}"
        )
    return f"{item.metric} shifted from {item.baseline_mean:.3f} to {item.abnormal_mean:.3f}"


def _feature_from_window_pattern(
    metric: str,
    baseline_mean: float,
    abnormal_mean: float,
    severity: float,
    entity_keys: Iterable[str],
    entity_dict: dict,
    pattern: str,
    segment_start_ts: str | None = None,
    segment_end_ts: str | None = None,
    segment_mean: float | None = None,
    pre_segment_mean: float | None = None,
    gap_seconds: float | None = None,
) -> MetricAnomalyFeatures:
    delta = abnormal_mean - baseline_mean
    return MetricAnomalyFeatures(
        metric=metric,
        abnormal_mean=abnormal_mean,
        baseline_mean=baseline_mean,
        delta=delta,
        delta_ratio=_relative_delta(delta, baseline_mean),
        zscore=0.0,
        robust_zscore=0.0,
        persistence_ratio=0.0,
        severity=round(float(severity), 4),
        entity_type="pod" if "pod" in entity_keys else "service",
        entity_name=str(entity_dict.get("pod") if "pod" in entity_keys else entity_dict.get("service")),
        service=entity_dict.get("service"),
        pod=entity_dict.get("pod"),
        first_seen_ts=segment_start_ts,
        last_seen_ts=segment_end_ts,
        in_window_pattern=pattern,
        segment_start_ts=segment_start_ts,
        segment_end_ts=segment_end_ts,
        segment_mean=segment_mean,
        pre_segment_mean=pre_segment_mean,
        gap_seconds=gap_seconds,
        raw_severity=round(float(severity), 4),
        severity_method="heuristic_in_window_pattern",
        calibration_notes=["in-window pattern severity uses existing heuristic formula"],
        notes=[f"in_window_pattern={pattern}"],
    )


def detect_metric_anomalies(
    baseline_df: pd.DataFrame,
    abnormal_df: pd.DataFrame,
    thresholds: dict,
    kpi_directions: dict[str, dict],
    entity_keys: Iterable[str],
    severity_calibrator=None,
) -> list[MetricAnomalyFeatures]:
    features: list[MetricAnomalyFeatures] = []
    grouped_baseline = baseline_df.groupby(list(entity_keys) + ["metric"])
    grouped_abnormal = abnormal_df.groupby(list(entity_keys) + ["metric"])

    all_keys = sorted(set(grouped_baseline.groups.keys()) | set(grouped_abnormal.groups.keys()), key=_group_key_sort_key)
    for key in all_keys:
        entity_values = key[:-1]
        metric = key[-1]
        entity_dict = dict(zip(entity_keys, entity_values))
        base_series = grouped_baseline.get_group(key)["value"] if key in grouped_baseline.groups else pd.Series(dtype=float)
        abn_series = grouped_abnormal.get_group(key)["value"] if key in grouped_abnormal.groups else pd.Series(dtype=float)
        abn_frame = grouped_abnormal.get_group(key) if key in grouped_abnormal.groups else pd.DataFrame(columns=abnormal_df.columns)
        if len(abn_series) < 1:
            continue

        abnormal_mean = float(pd.to_numeric(abn_series, errors="coerce").dropna().mean())
        baseline_mean = float(pd.to_numeric(base_series, errors="coerce").dropna().mean()) if len(base_series) else 0.0
        abn_values = pd.to_numeric(abn_series, errors="coerce").dropna()
        base_values = pd.to_numeric(base_series, errors="coerce").dropna()
        abnormal_max = float(abn_values.max()) if len(abn_values) else abnormal_mean
        abnormal_min = float(abn_values.min()) if len(abn_values) else abnormal_mean
        baseline_max = float(base_values.max()) if len(base_values) else baseline_mean
        baseline_min = float(base_values.min()) if len(base_values) else baseline_mean
        delta = abnormal_mean - baseline_mean
        delta_ratio = _relative_delta(delta, baseline_mean)
        baseline_std = _std(base_series)
        zscore = 0.0 if baseline_std == 0 else float(delta / baseline_std)
        robust_z = _robust_zscore(pd.to_numeric(base_series, errors="coerce"), abnormal_mean)
        direction = kpi_directions.get(metric, {}).get("direction", "both")
        min_relative_delta_ratio = float(thresholds.get("min_relative_delta_ratio", 0.05))

        if direction == "increase":
            direction_matches = delta > 0
        elif direction == "decrease":
            direction_matches = delta < 0
        else:
            direction_matches = abs(delta) > 0

        if len(base_series):
            threshold_high = baseline_mean + max(baseline_std, abs(baseline_mean) * 0.1, 1e-9)
            threshold_low = baseline_mean - max(baseline_std, abs(baseline_mean) * 0.1, 1e-9)
            if direction == "increase":
                persistence = float((abn_values > threshold_high).mean()) if len(abn_values) else 0.0
            elif direction == "decrease":
                persistence = float((abn_values < threshold_low).mean()) if len(abn_values) else 0.0
            else:
                persistence = float(((abn_values > threshold_high) | (abn_values < threshold_low)).mean()) if len(abn_values) else 0.0
        else:
            persistence = 1.0

        material_change = abs(delta_ratio) >= min_relative_delta_ratio or (baseline_mean == 0 and abs(abnormal_mean) > 0)
        event_delta_ratio = 0.0
        event_anomalous = False
        if metric in EVENT_INCREASE_METRICS:
            event_delta = max(delta, abnormal_max - baseline_max)
            event_delta_ratio = _relative_delta(event_delta, baseline_max)
            event_anomalous = event_delta > 0 and abnormal_max > baseline_max
        elif metric in EVENT_DECREASE_METRICS:
            event_delta = min(delta, abnormal_min - baseline_min)
            event_delta_ratio = _relative_delta(event_delta, baseline_min)
            event_anomalous = event_delta < 0 and abnormal_min < baseline_min

        is_anomalous = (
            event_anomalous
            or (
                direction_matches
                and material_change
                and (
                    abs(zscore) >= float(thresholds.get("zscore_threshold", 2.0))
                    or abs(robust_z) >= float(thresholds.get("robust_zscore_threshold", 3.0))
                    or (abs(delta_ratio) >= 0.3 and persistence >= float(thresholds.get("persistence_ratio_threshold", 0.5)))
                )
            )
        )
        if not is_anomalous:
            continue

        if event_anomalous:
            event_strength = abs(event_delta_ratio) if event_delta_ratio else min(abs(event_delta), 3.0)
            severity = float(min(1.0, 0.55 + 0.30 * min(event_strength, 1.5) + 0.15 * persistence))
        else:
            severity = float(
                min(
                    1.0,
                    0.35 * min(abs(zscore) / max(float(thresholds.get("zscore_threshold", 2.0)), 1e-6), 2.5)
                    + 0.35 * min(abs(robust_z) / max(float(thresholds.get("robust_zscore_threshold", 3.0)), 1e-6), 2.5)
                    + 0.30 * min(abs(delta_ratio), 2.0),
                )
            )
        raw_severity = severity
        severity_method = "heuristic"
        calibration_metadata = {}
        calibration_notes = []
        if severity_calibrator is not None:
            severity_result = severity_calibrator.calibrate_metric_severity(
                metric=metric,
                baseline_values=base_values,
                abnormal_values=abn_values,
                raw_severity=raw_severity,
                zscore=zscore,
                robust_zscore=robust_z,
                delta_ratio=delta_ratio,
                persistence=persistence,
                direction=direction,
                thresholds=thresholds,
            )
            severity = float(severity_result.get("severity", raw_severity))
            raw_severity = float(severity_result.get("raw_severity", raw_severity))
            severity_method = str(severity_result.get("severity_method", severity_method))
            calibration_metadata = dict(severity_result.get("calibration_metadata", {}))
            calibration_notes = list(severity_result.get("calibration_notes", []))
        first_seen_ts = None
        last_seen_ts = None
        if not abn_frame.empty and "timestamp" in abn_frame.columns:
            clean_ts = pd.to_datetime(abn_frame["timestamp"], utc=True, errors="coerce").dropna()
            if not clean_ts.empty:
                first_seen_ts = clean_ts.min().isoformat()
                last_seen_ts = clean_ts.max().isoformat()
        features.append(
            MetricAnomalyFeatures(
                metric=metric,
                abnormal_mean=abnormal_mean,
                baseline_mean=baseline_mean,
                delta=delta,
                delta_ratio=delta_ratio,
                zscore=zscore,
                robust_zscore=robust_z,
                persistence_ratio=persistence,
                severity=severity,
                entity_type="pod" if "pod" in entity_keys else "service",
                entity_name=str(entity_values[0]),
                service=entity_dict.get("service"),
                pod=entity_dict.get("pod"),
                first_seen_ts=first_seen_ts,
                last_seen_ts=last_seen_ts,
                abnormal_max=abnormal_max,
                abnormal_min=abnormal_min,
                baseline_max=baseline_max,
                baseline_min=baseline_min,
                raw_severity=raw_severity,
                severity_method=severity_method,
                calibration_metadata=calibration_metadata,
                calibration_notes=calibration_notes,
                notes=[f"delta_ratio={delta_ratio:.2f}", f"persistence={persistence:.2f}"],
            )
        )
    return features


def detect_in_window_metric_patterns(
    baseline_df: pd.DataFrame,
    abnormal_df: pd.DataFrame,
    thresholds: dict,
    kpi_directions: dict[str, dict],
    entity_keys: Iterable[str],
) -> list[MetricAnomalyFeatures]:
    cfg = thresholds.get("metric_in_window", {})
    if not cfg.get("enabled", True):
        return []

    entity_keys = list(entity_keys)
    min_points = int(cfg.get("min_points", 6))
    rolling_points = int(cfg.get("rolling_points", 6))
    min_segment_points = int(cfg.get("min_segment_points", 3))
    sudden_shift_ratio = float(cfg.get("sudden_shift_ratio", 0.3))
    burst_multiplier = float(cfg.get("burst_multiplier", 2.0))
    min_spike_points = int(cfg.get("min_spike_points", 2))
    gap_multiplier = float(cfg.get("gap_multiplier", 4.0))
    min_gap_seconds = float(cfg.get("min_gap_seconds", 60.0))
    max_patterns = int(cfg.get("max_patterns_per_entity_metric", 1))
    sudden_shift_metrics = set(cfg.get("sudden_shift_metrics", ["cpu_usage_pct", "memory_usage_pct", "ready_ratio", "request_rate"]))
    burst_metrics = set(cfg.get("burst_metrics", ["cpu_usage_pct", "memory_usage_pct", "network_rx", "network_tx"]))
    gap_metrics = set(cfg.get("gap_metrics", ["request_rate", "success_rate", "latency_p50", "latency_p90", "latency_p95", "latency_p99", "ready_ratio"]))

    grouped_baseline = baseline_df.groupby(entity_keys + ["metric"])
    grouped_abnormal = abnormal_df.groupby(entity_keys + ["metric"])
    features: list[MetricAnomalyFeatures] = []

    for key in sorted(grouped_abnormal.groups.keys(), key=_group_key_sort_key):
        entity_values = key[:-1]
        metric = key[-1]
        entity_dict = dict(zip(entity_keys, entity_values))
        abn_frame = grouped_abnormal.get_group(key).copy()
        if len(abn_frame) < min_points or "timestamp" not in abn_frame.columns:
            continue

        abn_frame["timestamp"] = pd.to_datetime(abn_frame["timestamp"], utc=True, errors="coerce")
        abn_frame["value"] = pd.to_numeric(abn_frame["value"], errors="coerce")
        abn_frame = abn_frame.dropna(subset=["timestamp", "value"]).sort_values("timestamp")
        if len(abn_frame) < min_points:
            continue

        base_series = grouped_baseline.get_group(key)["value"] if key in grouped_baseline.groups else pd.Series(dtype=float)
        base_values = _clean_values(base_series)
        abn_values = _clean_values(abn_frame["value"])
        baseline_mean = float(base_values.mean()) if len(base_values) else 0.0
        abnormal_mean = float(abn_values.mean()) if len(abn_values) else 0.0
        candidates: list[MetricAnomalyFeatures] = []

        if metric in gap_metrics:
            ts = abn_frame["timestamp"].dropna().sort_values()
            diffs = ts.diff().dropna().dt.total_seconds()
            if not diffs.empty:
                expected_step = float(diffs.median())
                max_gap = float(diffs.max())
                gap_threshold = max(min_gap_seconds, expected_step * gap_multiplier)
                if expected_step > 0 and max_gap >= gap_threshold:
                    gap_idx = diffs.idxmax()
                    end_ts = ts.loc[gap_idx]
                    start_pos = ts.index.get_loc(gap_idx) - 1
                    start_ts = ts.iloc[start_pos] if start_pos >= 0 else None
                    window_seconds = max(float((ts.max() - ts.min()).total_seconds()), 1.0)
                    severity = min(0.85, 0.35 + 0.50 * min(max_gap / window_seconds, 1.0))
                    candidates.append(
                        _feature_from_window_pattern(
                            metric,
                            baseline_mean,
                            abnormal_mean,
                            severity,
                            entity_keys,
                            entity_dict,
                            "missing_data_gap",
                            start_ts.isoformat() if start_ts is not None else None,
                            end_ts.isoformat(),
                            gap_seconds=max_gap,
                        )
                    )

        if metric in sudden_shift_metrics and len(abn_frame) >= rolling_points + min_segment_points:
            values = abn_frame["value"].astype(float).reset_index(drop=True)
            timestamps = abn_frame["timestamp"].reset_index(drop=True)
            direction = kpi_directions.get(metric, {}).get("direction", "both")
            for index in range(rolling_points, len(values) - min_segment_points + 1):
                previous = values.iloc[index - rolling_points : index]
                segment = values.iloc[index : index + min_segment_points]
                previous_mean = float(previous.median())
                segment_mean = float(segment.mean())
                delta = segment_mean - previous_mean
                ratio = _relative_delta(delta, previous_mean)
                absolute_change = abs(delta)

                if direction == "increase":
                    pattern_matches = ratio >= sudden_shift_ratio or (previous_mean == 0 and delta > 0)
                    pattern = "sudden_increase"
                elif direction == "decrease":
                    pattern_matches = ratio <= -sudden_shift_ratio or (previous_mean > 0 and delta < 0 and abs(ratio) >= sudden_shift_ratio)
                    pattern = "sudden_decrease"
                else:
                    pattern_matches = abs(ratio) >= sudden_shift_ratio or (previous_mean == 0 and absolute_change > 0)
                    pattern = "sudden_increase" if delta >= 0 else "sudden_decrease"

                if not pattern_matches:
                    continue
                magnitude_score = min(abs(ratio), 2.0) / 2.0 if ratio else min(absolute_change, 3.0) / 3.0
                duration_score = min_segment_points / max(len(values), 1)
                severity = min(0.95, 0.45 * magnitude_score + 0.30 * duration_score + 0.25 * 0.5)
                candidates.append(
                    _feature_from_window_pattern(
                        metric,
                        baseline_mean,
                        abnormal_mean,
                        severity,
                        entity_keys,
                        entity_dict,
                        pattern,
                        timestamps.iloc[index].isoformat(),
                        timestamps.iloc[index + min_segment_points - 1].isoformat(),
                        segment_mean=segment_mean,
                        pre_segment_mean=previous_mean,
                    )
                )
                break

        has_sudden_shift = any(item.in_window_pattern in {"sudden_increase", "sudden_decrease"} for item in candidates)
        if metric in burst_metrics and len(abn_values) >= min_points and not has_sudden_shift:
            median = float(abn_values.median())
            threshold = median * burst_multiplier if median > 0 else float(abn_values.quantile(0.95))
            spike_values = abn_frame.loc[abn_frame["value"] > threshold]
            if threshold > 0 and len(spike_values) >= min_spike_points:
                segment_mean = float(spike_values["value"].mean())
                ratio = _relative_delta(segment_mean - median, median)
                severity = min(0.9, 0.45 * min(abs(ratio), 2.0) / 2.0 + 0.30 * min(len(spike_values) / len(abn_frame), 1.0) + 0.20)
                candidates.append(
                    _feature_from_window_pattern(
                        metric,
                        baseline_mean,
                        abnormal_mean,
                        severity,
                        entity_keys,
                        entity_dict,
                        "short_burst",
                        spike_values["timestamp"].min().isoformat(),
                        spike_values["timestamp"].max().isoformat(),
                        segment_mean=segment_mean,
                        pre_segment_mean=median,
                    )
                )

        candidates.sort(key=lambda item: item.severity, reverse=True)
        features.extend(candidates[:max_patterns])

    return features


def to_anomaly_records(features: list[MetricAnomalyFeatures]) -> list[AnomalyRecord]:
    records: list[AnomalyRecord] = []
    for item in features:
        records.append(
            AnomalyRecord(
                source="metric",
                entity_type=item.entity_type,
                entity_name=item.entity_name,
                metric_or_pattern=item.metric,
                abnormal_value=item.abnormal_mean,
                baseline_value=item.baseline_mean,
                delta=item.delta,
                zscore=item.zscore,
                severity=item.severity,
                summary=_metric_summary(item),
                metadata={
                    "delta_ratio": item.delta_ratio,
                    "abnormal_max": getattr(item, "abnormal_max", None),
                    "abnormal_min": getattr(item, "abnormal_min", None),
                    "baseline_max": getattr(item, "baseline_max", None),
                    "baseline_min": getattr(item, "baseline_min", None),
                    "robust_zscore": item.robust_zscore,
                    "persistence_ratio": item.persistence_ratio,
                    "in_window_pattern": item.in_window_pattern,
                    "segment_start_ts": item.segment_start_ts,
                    "segment_end_ts": item.segment_end_ts,
                    "segment_mean": item.segment_mean,
                    "pre_segment_mean": item.pre_segment_mean,
                    "gap_seconds": item.gap_seconds,
                    "raw_severity": item.raw_severity,
                    "severity_method": item.severity_method,
                    "calibration_metadata": item.calibration_metadata,
                    "calibration_notes": item.calibration_notes,
                    "service": item.service,
                    "pod": item.pod,
                    "first_seen_ts": item.first_seen_ts,
                    "last_seen_ts": item.last_seen_ts,
                },
            )
        )
    return records
