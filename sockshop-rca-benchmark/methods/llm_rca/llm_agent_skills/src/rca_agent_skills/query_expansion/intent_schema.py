from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MetricQueryIntent:
    skill: str
    reason: str
    service: str | None
    pod: str | None
    kpi: str
    granularity: str
    window: str


@dataclass
class LogQueryIntent:
    skill: str
    reason: str
    service: str | None
    pod: str | None
    keyword: str | None
    template: str | None
    window: str


@dataclass
class TraceQueryIntent:
    skill: str
    reason: str
    service: str | None
    peer_service: str | None
    operation: str | None
    status: str | None
    window: str

