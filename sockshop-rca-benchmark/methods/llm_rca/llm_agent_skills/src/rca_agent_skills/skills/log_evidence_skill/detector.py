from __future__ import annotations

from collections import Counter

import pandas as pd

from rca_agent_skills.common.models import AnomalyRecord
from .parser import extract_keywords, is_background_template
from .schemas import LogPatternFeature


def _ratio(abnormal: int, baseline: int) -> float:
    if baseline <= 0:
        return float(abnormal) if abnormal > 0 else 0.0
    return abnormal / max(baseline, 1)


def _template_severity(count: int, ratio: float, background_noise: bool) -> float:
    count_factor = min(float(count), 12.0) / 12.0
    ratio_factor = min(float(ratio), 6.0) / 6.0
    severity = 0.12 + 0.42 * count_factor + 0.30 * ratio_factor
    if background_noise:
        return round(min(0.25, severity * 0.25), 4)
    return round(min(0.92, severity), 4)


def _calibrate_log_severity(
    severity_calibrator,
    *,
    pattern_type: str,
    baseline_count: int,
    abnormal_count: int,
    ratio: float,
    raw_severity: float,
    background_noise: bool,
) -> tuple[float, float, str, dict, list[str]]:
    if severity_calibrator is None:
        return raw_severity, raw_severity, "heuristic", {}, []
    result = severity_calibrator.calibrate_log_severity(
        pattern_type=pattern_type,
        baseline_count=baseline_count,
        abnormal_count=abnormal_count,
        ratio=ratio,
        raw_severity=raw_severity,
        background_noise=background_noise,
    )
    return (
        float(result.get("severity", raw_severity)),
        float(result.get("raw_severity", raw_severity)),
        str(result.get("severity_method", "heuristic")),
        dict(result.get("calibration_metadata", {})),
        list(result.get("calibration_notes", [])),
    )


def detect_log_spikes(
    baseline_df: pd.DataFrame,
    abnormal_df: pd.DataFrame,
    entity_key: str,
    thresholds: dict,
    severity_calibrator=None,
) -> list[LogPatternFeature]:
    features: list[LogPatternFeature] = []
    log_spike_threshold = float(thresholds.get("log_spike_ratio_threshold", 2.0))
    min_count = int(thresholds.get("minimum_count", 3))
    keywords = thresholds.get("log_keywords")
    background_hints = thresholds.get("log_background_template_hints")

    for entity_name in sorted(set(baseline_df[entity_key].dropna()) | set(abnormal_df[entity_key].dropna())):
        base_slice = baseline_df.loc[baseline_df[entity_key] == entity_name]
        abn_slice = abnormal_df.loc[abnormal_df[entity_key] == entity_name]
        if abn_slice.empty:
            continue

        base_templates = Counter(base_slice["message_template"])
        abn_templates = Counter(abn_slice["message_template"])
        base_levels = Counter(base_slice["log_level"])
        abn_levels = Counter(abn_slice["log_level"])
        base_keywords = Counter(keyword for msg in base_slice["message"] for keyword in extract_keywords(msg, keywords))
        abn_keywords = Counter(keyword for msg in abn_slice["message"] for keyword in extract_keywords(msg, keywords))

        for template, count in abn_templates.items():
            baseline_count = base_templates.get(template, 0)
            ratio = _ratio(count, baseline_count)
            background_noise = is_background_template(str(template), background_hints)
            if count >= min_count and ratio >= log_spike_threshold:
                row = abn_slice.iloc[0]
                clean_ts = pd.to_datetime(abn_slice.loc[abn_slice["message_template"] == template, "timestamp"], utc=True, errors="coerce").dropna()
                raw_severity = _template_severity(count, ratio, background_noise)
                severity, raw_severity, severity_method, calibration_metadata, calibration_notes = _calibrate_log_severity(
                    severity_calibrator,
                    pattern_type="template_spike",
                    baseline_count=baseline_count,
                    abnormal_count=count,
                    ratio=ratio,
                    raw_severity=raw_severity,
                    background_noise=background_noise,
                )
                features.append(
                    LogPatternFeature(
                        entity_type="pod" if entity_key == "pod" else "service",
                        entity_name=str(entity_name),
                        service=str(row.get("service", "")),
                        pod=str(row.get("pod", "")) if entity_key == "pod" else None,
                        pattern_type="template_spike",
                        pattern_value=str(template),
                        baseline_count=baseline_count,
                        abnormal_count=count,
                        ratio=ratio,
                        severity=severity,
                        background_noise=background_noise,
                        first_seen_ts=clean_ts.min().isoformat() if not clean_ts.empty else None,
                        last_seen_ts=clean_ts.max().isoformat() if not clean_ts.empty else None,
                        raw_severity=raw_severity,
                        severity_method=severity_method,
                        calibration_metadata=calibration_metadata,
                        calibration_notes=calibration_notes,
                    )
                )
        for keyword, count in abn_keywords.items():
            baseline_count = base_keywords.get(keyword, 0)
            ratio = _ratio(count, baseline_count)
            if count >= min_count and ratio >= log_spike_threshold:
                row = abn_slice.iloc[0]
                keyword_rows = abn_slice.loc[abn_slice["message"].astype(str).str.contains(keyword, case=False, na=False)]
                clean_ts = pd.to_datetime(keyword_rows["timestamp"], utc=True, errors="coerce").dropna()
                raw_severity = min(1.0, 0.15 * count + 0.25 * min(ratio, 4.0))
                severity, raw_severity, severity_method, calibration_metadata, calibration_notes = _calibrate_log_severity(
                    severity_calibrator,
                    pattern_type="keyword_spike",
                    baseline_count=baseline_count,
                    abnormal_count=count,
                    ratio=ratio,
                    raw_severity=raw_severity,
                    background_noise=False,
                )
                features.append(
                    LogPatternFeature(
                        entity_type="pod" if entity_key == "pod" else "service",
                        entity_name=str(entity_name),
                        service=str(row.get("service", "")),
                        pod=str(row.get("pod", "")) if entity_key == "pod" else None,
                        pattern_type="keyword_spike",
                        pattern_value=keyword,
                        baseline_count=baseline_count,
                        abnormal_count=count,
                        ratio=ratio,
                        severity=severity,
                        background_noise=False,
                        first_seen_ts=clean_ts.min().isoformat() if not clean_ts.empty else None,
                        last_seen_ts=clean_ts.max().isoformat() if not clean_ts.empty else None,
                        raw_severity=raw_severity,
                        severity_method=severity_method,
                        calibration_metadata=calibration_metadata,
                        calibration_notes=calibration_notes,
                    )
                )
        error_like_count = abn_levels.get("ERROR", 0) + abn_levels.get("FATAL", 0)
        baseline_error_like_count = base_levels.get("ERROR", 0) + base_levels.get("FATAL", 0)
        error_like_ratio = _ratio(error_like_count, baseline_error_like_count)
        if error_like_count >= min_count and error_like_ratio >= log_spike_threshold:
            row = abn_slice.iloc[0]
            clean_ts = pd.to_datetime(
                abn_slice.loc[abn_slice["log_level"].isin(["ERROR", "FATAL"]), "timestamp"],
                utc=True,
                errors="coerce",
            ).dropna()
            raw_severity = min(1.0, 0.1 * error_like_count + 0.3 * min(error_like_ratio, 4.0))
            severity, raw_severity, severity_method, calibration_metadata, calibration_notes = _calibrate_log_severity(
                severity_calibrator,
                pattern_type="level_shift",
                baseline_count=baseline_error_like_count,
                abnormal_count=error_like_count,
                ratio=error_like_ratio,
                raw_severity=raw_severity,
                background_noise=False,
            )
            features.append(
                LogPatternFeature(
                    entity_type="pod" if entity_key == "pod" else "service",
                    entity_name=str(entity_name),
                    service=str(row.get("service", "")),
                    pod=str(row.get("pod", "")) if entity_key == "pod" else None,
                    pattern_type="level_shift",
                    pattern_value="ERROR/FATAL",
                    baseline_count=baseline_error_like_count,
                    abnormal_count=error_like_count,
                    ratio=error_like_ratio,
                    severity=severity,
                    background_noise=False,
                    first_seen_ts=clean_ts.min().isoformat() if not clean_ts.empty else None,
                    last_seen_ts=clean_ts.max().isoformat() if not clean_ts.empty else None,
                    raw_severity=raw_severity,
                    severity_method=severity_method,
                    calibration_metadata=calibration_metadata,
                    calibration_notes=calibration_notes,
                )
            )
    return features


def to_anomaly_records(features: list[LogPatternFeature]) -> list[AnomalyRecord]:
    records: list[AnomalyRecord] = []
    for item in features:
        records.append(
            AnomalyRecord(
                source="log",
                entity_type=item.entity_type,
                entity_name=item.entity_name,
                metric_or_pattern=item.pattern_type,
                abnormal_value=item.abnormal_count,
                baseline_value=item.baseline_count,
                delta=float(item.abnormal_count - item.baseline_count),
                zscore=None,
                severity=item.severity,
                summary=f"{item.pattern_type}='{item.pattern_value}' count {item.baseline_count}->{item.abnormal_count}",
                metadata={
                    "ratio": item.ratio,
                    "service": item.service,
                    "pod": item.pod,
                    "pattern_value": item.pattern_value,
                    "background_noise": item.background_noise,
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
