from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SemanticParseResult:
    abnormal_window: dict[str, str] | None = None
    enabled_telemetry: dict[str, bool] = field(
        default_factory=lambda: {"metrics": True, "logs": True, "traces": True}
    )
    requested_outputs: dict[str, bool] = field(
        default_factory=lambda: {
            "service_ranking": True,
            "pod_ranking": True,
            "service_fault_ranking": True,
            "pod_fault_ranking": True,
        }
    )
    ranking_depth: int = 5
    needs_clarification: bool = False
    clarification_questions: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    raw_response: dict | None = None
