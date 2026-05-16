from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MetricAnomalyFeatures:
    metric: str
    abnormal_mean: float
    baseline_mean: float
    delta: float
    delta_ratio: float
    zscore: float
    robust_zscore: float
    persistence_ratio: float
    severity: float
    entity_type: str
    entity_name: str
    service: str | None = None
    pod: str | None = None
    first_seen_ts: str | None = None
    last_seen_ts: str | None = None
    abnormal_max: float | None = None
    abnormal_min: float | None = None
    baseline_max: float | None = None
    baseline_min: float | None = None
    in_window_pattern: str | None = None
    segment_start_ts: str | None = None
    segment_end_ts: str | None = None
    segment_mean: float | None = None
    pre_segment_mean: float | None = None
    gap_seconds: float | None = None
    raw_severity: float | None = None
    severity_method: str | None = None
    calibration_metadata: dict = field(default_factory=dict)
    calibration_notes: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
