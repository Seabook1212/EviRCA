from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rca_agent_skills.common.models import RankedHypothesis, TimeWindow


@dataclass
class APIInputs:
    prometheus_url: str | None = None
    loki_url: str | None = None
    jaeger_url: str | None = None
    namespace: str | None = None


@dataclass
class CSVInputs:
    metrics_csv: str | None = None
    logs_csv: str | None = None
    traces_csv: str | None = None
    topology_file: str | None = None


@dataclass
class RCARequest:
    incident_id: str
    abnormal_window: TimeWindow
    baseline_window: TimeWindow
    backend_mode: str
    topology: dict[str, Any] | None = None
    api_inputs: APIInputs | None = None
    csv_inputs: CSVInputs | None = None
    namespace: str = "sock-shop"
    config_bundle: dict[str, Any] = field(default_factory=dict)
    execution_options: dict[str, Any] = field(default_factory=dict)


@dataclass
class RCAResponse:
    incident_id: str
    abnormal_window: TimeWindow
    baseline_window: TimeWindow
    service_top5: list[RankedHypothesis]
    pod_top5: list[RankedHypothesis]
    final_summary: str
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
