from __future__ import annotations

from dataclasses import asdict

from rca_agent_skills.common.logging_utils import get_logger, log_json
from rca_agent_skills.common.models import QueryBudgetStatus
from rca_agent_skills.data_access import build_data_access
from rca_agent_skills.llm import LLMClient
from rca_agent_skills.orchestrator_agent.planner import FixedSOPPlanner
from rca_agent_skills.orchestrator_agent.state import RCAState
from rca_agent_skills.skills.log_evidence_skill.skill import LogEvidenceSkill
from rca_agent_skills.skills.metric_evidence_skill.skill import MetricEvidenceSkill
from rca_agent_skills.skills.rootcause_reasoning_skill.skill import (
    RootCauseReasoningSkill,
)
from rca_agent_skills.skills.trace_evidence_skill.skill import TraceEvidenceSkill


class RCAOrchestratorAgent:
    def __init__(self, request, settings: dict):
        self.request = request
        self.settings = settings
        self.logger = get_logger(self.__class__.__name__)
        self.planner = FixedSOPPlanner()
        self.data_access = build_data_access(request, settings)
        self.debug = settings.get("debug", {})
        llm_cfg = settings.get("llm", {})
        self.llm_client = LLMClient(
            provider=llm_cfg.get("provider", "heuristic"),
            model=llm_cfg.get("model", "heuristic-v1"),
            temperature=float(llm_cfg.get("temperature", 0.0)),
            debug=self.debug,
        )
        self.skills = {
            "metric_evidence": MetricEvidenceSkill(
                settings, self.data_access, self.llm_client
            ),
            "log_evidence": LogEvidenceSkill(
                settings, self.data_access, self.llm_client
            ),
            "trace_evidence": TraceEvidenceSkill(
                settings, self.data_access, self.llm_client
            ),
            "rootcause_reasoning": RootCauseReasoningSkill(
                settings, self.data_access, self.llm_client
            ),
        }

    def _skill_enabled(self, skill_name: str) -> bool:
        if skill_name == "rootcause_reasoning":
            return True
        enabled_telemetry = (self.request.execution_options or {}).get(
            "enabled_telemetry", {}
        )
        telemetry_by_skill = {
            "metric_evidence": "metrics",
            "log_evidence": "logs",
            "trace_evidence": "traces",
        }
        telemetry_name = telemetry_by_skill.get(skill_name)
        if not telemetry_name:
            return True
        return bool(enabled_telemetry.get(telemetry_name, True))

    def _initialize_state(self) -> RCAState:
        budgets = self.settings.get("budgets", {})
        return RCAState(
            incident_id=self.request.incident_id,
            abnormal_window=asdict(self.request.abnormal_window),
            baseline_window=asdict(self.request.baseline_window),
            backend_mode=self.request.backend_mode,
            topology=self.data_access.get_topology(),
            query_budgets={
                "metric": QueryBudgetStatus(
                    limit=int(budgets.get("metric_followup", 2))
                ),
                "log": QueryBudgetStatus(limit=int(budgets.get("log_followup", 2))),
                "trace": QueryBudgetStatus(limit=int(budgets.get("trace_followup", 2))),
            },
        )

    def run(self):
        state = self._initialize_state()
        if self.debug.get("print_skill_inputs", True):
            log_json(
                self.logger,
                "[AGENT][REQUEST] ",
                {
                    "incident_id": self.request.incident_id,
                    "backend_mode": self.request.backend_mode,
                    "abnormal_window": asdict(self.request.abnormal_window),
                    "baseline_window": asdict(self.request.baseline_window),
                    "topology_services": len(state.topology.get("services", [])),
                },
            )
        for skill_name in self.planner.get_skill_order():
            if not self._skill_enabled(skill_name):
                self.logger.info("Skipping skill by execution_options: %s", skill_name)
                continue
            self.logger.info("Running skill: %s", skill_name)
            result = self.skills[skill_name].run(self.request, state)
            if skill_name == "metric_evidence":
                state.metrics_evidence = result
                state.warnings.extend(result.warnings)
                state.errors.extend(result.errors)
            elif skill_name == "log_evidence":
                state.logs_evidence = result
                state.warnings.extend(result.warnings)
                state.errors.extend(result.errors)
            elif skill_name == "trace_evidence":
                state.traces_evidence = result
                state.warnings.extend(result.warnings)
                state.errors.extend(result.errors)
            else:
                state.final_result = result
                state.warnings.extend(result.warnings)
                state.errors.extend(result.errors)
            if self.debug.get("print_skill_outputs", True):
                payload = {
                    "skill": skill_name,
                    "service_evidence_count": len(
                        getattr(result, "service_evidence", [])
                        or getattr(result, "service_top5", [])
                    ),
                    "pod_evidence_count": len(
                        getattr(result, "pod_evidence", [])
                        or getattr(result, "pod_top5", [])
                    ),
                    "anomaly_record_count": len(getattr(result, "anomaly_records", [])),
                    "warnings": getattr(result, "warnings", []),
                    "errors": getattr(result, "errors", []),
                    "metadata_keys": sorted(
                        list(getattr(result, "metadata", {}).keys())
                    ),
                }
                log_json(self.logger, "[AGENT][SKILL_OUTPUT] ", payload)
        if state.final_result:
            state.final_result.warnings = state.warnings
            state.final_result.errors = state.errors
        return state.final_result
