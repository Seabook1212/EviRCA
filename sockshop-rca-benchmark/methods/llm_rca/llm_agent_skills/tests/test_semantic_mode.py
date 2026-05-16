from __future__ import annotations

from types import SimpleNamespace

from rca_agent_skills.orchestrator_agent.agent import RCAOrchestratorAgent
from rca_agent_skills.skills.semantic_request_parsing_skill import (
    SemanticRequestParsingSkill,
)
from rca_agent_skills.utils.baseline_window_selector import select_baseline_window


def test_semantic_parser_extracts_window_and_metrics_only():
    skill = SemanticRequestParsingSkill({}, llm_client=None)

    result = skill.parse_heuristic(
        "Analyze from 2026-05-02T10:00:00Z to 2026-05-02T10:15:00Z using metrics only."
    )

    assert result.abnormal_window == {
        "start": "2026-05-02T10:00:00Z",
        "end": "2026-05-02T10:15:00Z",
    }
    assert result.enabled_telemetry == {
        "metrics": True,
        "logs": False,
        "traces": False,
    }
    assert not result.needs_clarification


def test_semantic_parser_defaults_to_all_outputs_and_telemetry():
    skill = SemanticRequestParsingSkill({}, llm_client=None)

    result = skill.parse_heuristic(
        "Analyze 2026-05-02T10:00:00Z to 2026-05-02T10:15:00Z."
    )

    assert all(result.enabled_telemetry.values())
    assert all(result.requested_outputs.values())


def test_baseline_selector_chooses_nearest_before_window():
    selected = select_baseline_window(
        {"start": "2026-05-02T10:00:00Z", "end": "2026-05-02T10:15:00Z"},
        {
            "baseline_windows": [
                {
                    "id": "older",
                    "namespace": "sock-shop",
                    "start": "2026-05-02T01:00:00Z",
                    "end": "2026-05-02T01:10:00Z",
                },
                {
                    "id": "nearer",
                    "namespace": "sock-shop",
                    "start": "2026-05-02T09:00:00Z",
                    "end": "2026-05-02T09:10:00Z",
                },
            ],
            "selection": {"strategy": "nearest_before", "allow_after": False},
        },
    )

    assert selected.baseline_id == "nearer"


def test_orchestrator_respects_disabled_telemetry_skill():
    request = SimpleNamespace(
        execution_options={
            "enabled_telemetry": {"metrics": True, "logs": False, "traces": True}
        }
    )
    agent = RCAOrchestratorAgent.__new__(RCAOrchestratorAgent)
    agent.request = request

    assert agent._skill_enabled("metric_evidence")
    assert not agent._skill_enabled("log_evidence")
    assert agent._skill_enabled("trace_evidence")
    assert agent._skill_enabled("rootcause_reasoning")
