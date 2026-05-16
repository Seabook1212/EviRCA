from __future__ import annotations

from dataclasses import dataclass, field

from rca_agent_skills.common.models import QueryBudgetStatus, SkillResult
from rca_agent_skills.orchestrator_agent.schemas import RCAResponse


@dataclass
class RCAState:
    incident_id: str
    abnormal_window: dict
    baseline_window: dict
    backend_mode: str
    topology: dict
    metrics_evidence: SkillResult | None = None
    logs_evidence: SkillResult | None = None
    traces_evidence: SkillResult | None = None
    final_result: RCAResponse | None = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    query_budgets: dict[str, QueryBudgetStatus] = field(default_factory=dict)

