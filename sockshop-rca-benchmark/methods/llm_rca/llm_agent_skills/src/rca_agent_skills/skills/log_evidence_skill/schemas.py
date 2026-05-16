from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LogPatternFeature:
    entity_type: str
    entity_name: str
    service: str
    pod: str | None
    pattern_type: str
    pattern_value: str
    baseline_count: int
    abnormal_count: int
    ratio: float
    severity: float
    background_noise: bool = False
    first_seen_ts: str | None = None
    last_seen_ts: str | None = None
    raw_severity: float | None = None
    severity_method: str | None = None
    calibration_metadata: dict = field(default_factory=dict)
    calibration_notes: list[str] = field(default_factory=list)
