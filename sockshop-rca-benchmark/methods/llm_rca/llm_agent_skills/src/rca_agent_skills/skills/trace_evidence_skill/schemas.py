from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TraceAnomalyFeature:
    entity_type: str
    entity_name: str
    service: str
    pod: str | None
    anomaly_type: str
    abnormal_value: float
    baseline_value: float
    ratio: float
    severity: float
    peer_service: str | None = None
    edge_role: str | None = None
    edge_source_service: str | None = None
    edge_target_service: str | None = None
    edge_source_pod: str | None = None
    edge_target_pod: str | None = None
    first_seen_ts: str | None = None
    last_seen_ts: str | None = None
    raw_severity: float | None = None
    severity_method: str | None = None
    calibration_metadata: dict = field(default_factory=dict)
    calibration_notes: list[str] = field(default_factory=list)
