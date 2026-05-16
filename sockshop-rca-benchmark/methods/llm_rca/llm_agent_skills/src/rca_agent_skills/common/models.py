from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TimeWindow:
    start: str
    end: str


@dataclass
class QueryBudgetStatus:
    limit: int
    used: int = 0


@dataclass
class AnomalyRecord:
    source: str
    entity_type: str
    entity_name: str
    metric_or_pattern: str
    abnormal_value: float | int | str | None
    baseline_value: float | int | str | None
    delta: float | None
    zscore: float | None
    severity: float
    summary: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ServiceEvidence:
    service: str
    score: float
    supporting_evidence: list[str]
    anomaly_records: list[AnomalyRecord] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PodEvidence:
    pod: str
    service: str
    score: float
    supporting_evidence: list[str]
    anomaly_records: list[AnomalyRecord] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RankedHypothesis:
    score: float
    fault_type: str
    supporting_evidence: list[str]
    notes: str
    service: str | None = None
    pod: str | None = None


@dataclass
class SkillResult:
    service_evidence: list[ServiceEvidence] = field(default_factory=list)
    pod_evidence: list[PodEvidence] = field(default_factory=list)
    anomaly_records: list[AnomalyRecord] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

