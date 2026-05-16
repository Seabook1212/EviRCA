from pathlib import Path
from types import SimpleNamespace

from rca_agent_skills.main import build_request
from rca_agent_skills.common.io_utils import read_json
from rca_agent_skills.common.models import (
    AnomalyRecord,
    PodEvidence,
    RankedHypothesis,
    ServiceEvidence,
    SkillResult,
)
from rca_agent_skills.llm.schemas import LLMRankResponse
from rca_agent_skills.orchestrator_agent.state import RCAState
from rca_agent_skills.orchestrator_agent.agent import RCAOrchestratorAgent
from rca_agent_skills.skills.rootcause_reasoning_skill.skill import (
    RootCauseReasoningSkill,
)
from rca_agent_skills.skills.rootcause_reasoning_skill.checker import (
    build_rule_context,
    evaluate_rule_hints,
)
from rca_agent_skills.skills.rootcause_reasoning_skill.ranker import rank_with_llm


def test_reasoning_output_format():
    root = Path(__file__).resolve().parents[1]
    payload = read_json(root / "examples" / "sample_request.json")
    request = build_request(payload, root)
    agent = RCAOrchestratorAgent(request, request.config_bundle["settings"])
    result = agent.run()
    assert result.service_top5
    assert result.pod_top5
    assert result.final_summary
    assert (
        result.service_top5[0].fault_type
        in request.config_bundle["fault_types"]["fault_types"]
    )
    assert "rootcause_reasoning_rules" in request.config_bundle
    assert result.metadata["service_candidates"]
    assert "rule_hints" in result.metadata["service_candidates"][0]
    assert "reasoning_rules" in result.metadata


def test_ranker_preserves_candidate_shape_when_llm_returns_score_only():
    class ScoreOnlyLLM:
        def rank_candidates(self, request):
            return LLMRankResponse(
                rankings=[
                    {
                        "entity_type": "pod",
                        "service": "session-db",
                        "pod": "session-db-0",
                        "fault_type": "cpu_stress",
                        "score": 0.93,
                        "rationale": "Strong local CPU anomaly.",
                    }
                ],
                notes=["ranking_mode=openai"],
            )

    candidates = [
        {
            "entity_type": "pod",
            "service": "session-db",
            "pod": "session-db-0",
            "fault_type": "cpu_stress",
            "provisional_score": 0.88,
            "evidence_count": 2,
            "supporting_evidence": ["cpu_usage_pct shifted upward"],
            "notes": ["local resource support"],
            "rule_hints": {"local_resource_support": True},
        }
    ]

    response = rank_with_llm(ScoreOnlyLLM(), "pod_fault_ranking", {}, candidates)
    ranked = response.rankings[0]

    assert ranked["provisional_score"] == 0.93
    assert ranked["supporting_evidence"] == ["cpu_usage_pct shifted upward"]
    assert ranked["notes"] == ["local resource support", "Strong local CPU anomaly."]
    assert ranked["rule_hints"] == {"local_resource_support": True}


def test_candidate_supporting_evidence_limit_is_configurable():
    skill = RootCauseReasoningSkill(
        {
            "llm_context": {
                "max_candidate_supporting_evidence": 6,
                "max_candidate_brief_notes": 5,
            }
        },
        data_access=None,
        llm_client=None,
    )
    candidate = {
        "entity_type": "service",
        "service": "orders",
        "pod": None,
        "fault_type": "exception_injection",
        "provisional_score": 0.9,
        "evidence_count": 10,
        "supporting_evidence": [f"evidence-{index}" for index in range(8)],
        "notes": [f"note-{index}" for index in range(7)],
        "rule_hints": {"dependency_symptom_hint": True},
        "active_rules": ["downstream_edge_symptom", "adjacent_path_support"],
    }

    compact = skill._candidate_for_llm(candidate)

    assert compact["supporting_evidence"] == [f"evidence-{index}" for index in range(6)]
    assert compact["brief_notes"] == [f"note-{index}" for index in range(5)]
    assert compact["rule_hints"] == {"dependency_symptom_hint": True}
    assert compact["active_rules"] == [
        "downstream_edge_symptom",
        "adjacent_path_support",
    ]


def test_evidence_tree_can_be_omitted_from_initial_ranking_context():
    skill = RootCauseReasoningSkill(
        {
            "llm_context": {
                "include_evidence_tree_in_initial_ranking": False,
                "include_evidence_tree_in_reconciliation": True,
            }
        },
        data_access=None,
        llm_client=None,
    )
    full_context = {
        "evidence_tree": [{"service": "orders"}],
        "topology": {"services": ["orders"]},
    }

    initial_context = skill._llm_context_for_stage(
        full_context, include_evidence_tree=False
    )
    reconciliation_context = skill._llm_context_for_stage(
        full_context, include_evidence_tree=True
    )

    assert "evidence_tree" not in initial_context
    assert initial_context["evidence_tree_omitted"] is True
    assert reconciliation_context["evidence_tree"] == [{"service": "orders"}]


def test_finalize_sorts_by_final_confidence_score():
    class DataAccess:
        def get_topology(self):
            return {"services": ["low", "high"]}

    skill = RootCauseReasoningSkill(
        {"defaults": {"top_k": 2}}, DataAccess(), llm_client=None
    )
    finalized = skill._finalize(
        [
            {
                "service": "low",
                "fault_type": "network_delay",
                "provisional_score": 0.2,
                "notes": [],
            },
            {
                "service": "high",
                "fault_type": "memory_stress",
                "provisional_score": 0.9,
                "notes": [],
            },
        ],
        "service",
    )

    assert [item.service for item in finalized] == ["high", "low"]
    assert [item.score for item in finalized] == [0.9, 0.2]


def test_finalize_prefers_specific_pod_failure_evidence_on_score_tie():
    class DataAccess:
        def get_topology(self):
            return {"services": ["orders"]}

    skill = RootCauseReasoningSkill(
        {"defaults": {"top_k": 2}}, DataAccess(), llm_client=None
    )
    finalized = skill._finalize(
        [
            {
                "service": "orders",
                "fault_type": "cpu_stress",
                "provisional_score": 0.98,
                "supporting_evidence": [
                    "cpu_usage_pct: cpu_usage_pct shifted from 4.0 to 26.0"
                ],
                "notes": [],
            },
            {
                "service": "orders",
                "fault_type": "pod_failure",
                "provisional_score": 0.98,
                "supporting_evidence": [
                    "restart_count: restart_count max shifted from 0.000 to 4.000"
                ],
                "notes": [],
            },
        ],
        "service",
    )

    assert finalized[0].fault_type == "pod_failure"


def test_multi_pod_restart_on_upstream_with_failing_dependency_adds_propagation_hint():
    user_restart_1 = AnomalyRecord(
        source="metric",
        entity_type="pod",
        entity_name="user-1",
        metric_or_pattern="restart_count",
        abnormal_value=0.1,
        baseline_value=0.0,
        delta=0.1,
        zscore=None,
        severity=0.95,
        summary="restart_count mean shifted from 0.000 to 0.100; max shifted from 0.000 to 2.000",
        metadata={
            "service": "user",
            "pod": "user-1",
            "abnormal_max": 2,
            "baseline_max": 0,
        },
    )
    user_restart_2 = AnomalyRecord(
        source="metric",
        entity_type="pod",
        entity_name="user-2",
        metric_or_pattern="restart_count",
        abnormal_value=0.1,
        baseline_value=0.0,
        delta=0.1,
        zscore=None,
        severity=0.95,
        summary="restart_count mean shifted from 0.000 to 0.100; max shifted from 0.000 to 2.000",
        metadata={
            "service": "user",
            "pod": "user-2",
            "abnormal_max": 2,
            "baseline_max": 0,
        },
    )
    user_db_restart = AnomalyRecord(
        source="metric",
        entity_type="pod",
        entity_name="user-db-0",
        metric_or_pattern="restart_count",
        abnormal_value=0.4,
        baseline_value=0.0,
        delta=0.4,
        zscore=None,
        severity=0.99,
        summary="restart_count mean shifted from 0.000 to 0.400; max shifted from 0.000 to 6.000",
        metadata={
            "service": "user-db",
            "pod": "user-db-0",
            "abnormal_max": 6,
            "baseline_max": 0,
        },
    )
    state = RCAState(
        incident_id="pod_do_fault_user-db_001",
        abnormal_window={},
        baseline_window={},
        backend_mode="csv",
        topology={
            "services": ["user", "user-db"],
            "edges": [{"source": "user", "target": "user-db"}],
        },
        metrics_evidence=SkillResult(
            pod_evidence=[
                PodEvidence(
                    pod="user-1",
                    service="user",
                    score=0.95,
                    supporting_evidence=[],
                    anomaly_records=[user_restart_1],
                ),
                PodEvidence(
                    pod="user-2",
                    service="user",
                    score=0.95,
                    supporting_evidence=[],
                    anomaly_records=[user_restart_2],
                ),
                PodEvidence(
                    pod="user-db-0",
                    service="user-db",
                    score=0.99,
                    supporting_evidence=[],
                    anomaly_records=[user_db_restart],
                ),
            ],
            anomaly_records=[user_restart_1, user_restart_2, user_db_restart],
        ),
    )
    rules_config = {
        "settings": {
            "score_adjustments": {
                "shared_issue_hint": 0.06,
                "local_resource_support": 0.08,
                "dependency_symptom_hint": -0.08,
                "downstream_local_failure_hint": -0.08,
            }
        },
        "rule_groups": {
            "multi_pod_patterns": {
                "enabled": True,
                "rules": [
                    {
                        "id": "shared_multi_pod_anomaly",
                        "min_affected_pods": 2,
                        "min_sibling_relative_score": 0.7,
                    }
                ],
            }
        },
    }
    context = build_rule_context(state, state.topology, rules_config)
    evaluation = evaluate_rule_hints(
        {"entity_type": "pod", "service": "user", "pod": "user-1"},
        context["pod_evidence"]["user-1"],
        context,
        state.topology,
        rules_config,
    )

    assert evaluation.rule_hints.shared_issue_hint is True
    assert evaluation.rule_hints.local_resource_support is True
    assert evaluation.rule_hints.downstream_local_failure_hint is True
    assert evaluation.rule_hints.dependency_symptom_hint is True

    skill = RootCauseReasoningSkill({}, data_access=None, llm_client=None)
    hints = skill._build_propagation_hints(state, context, max_hints=10)
    hint = next(
        item
        for item in hints
        if item["type"] == "downstream_local_failure_vs_upstream_multi_pod_failures"
    )
    assert hint["upstream_service"] == "user"
    assert hint["downstream_service"] == "user-db"
    assert hint["upstream_affected_pods"] == ["user-1", "user-2"]
    assert hint["downstream_direct_failure_pods"] == ["user-db-0"]


def test_cross_level_reconciliation_uses_llm_combined_service_and_pod_judgment():
    class ReconciliationLLM:
        def __init__(self):
            self.prompt_names = []

        def rank_candidates(self, request):
            self.prompt_names.append(request.prompt_name)
            if request.prompt_name == "cross_level_ranking_reconciliation":
                return LLMRankResponse(
                    rankings=[
                        {
                            "entity_type": "service",
                            "service": "queue-master",
                            "pod": None,
                            "fault_type": "memory_stress",
                            "provisional_score": 0.95,
                            "rationale": "Aligned with strongest queue-master pod.",
                        },
                        {
                            "entity_type": "pod",
                            "service": "queue-master",
                            "pod": "queue-master-1",
                            "fault_type": "memory_stress",
                            "provisional_score": 0.94,
                            "rationale": "Strong local pod memory evidence.",
                        },
                        {
                            "entity_type": "service",
                            "service": "carts",
                            "pod": None,
                            "fault_type": "memory_stress",
                            "provisional_score": 0.70,
                        },
                        {
                            "entity_type": "pod",
                            "service": "carts",
                            "pod": "carts-1",
                            "fault_type": "memory_stress",
                            "provisional_score": 0.69,
                        },
                    ],
                    notes=["cross-level alignment applied"],
                )
            return LLMRankResponse(
                rankings=request.candidates, notes=[f"prompt={request.prompt_name}"]
            )

    skill = RootCauseReasoningSkill(
        {}, data_access=None, llm_client=ReconciliationLLM()
    )
    services = [
        {
            "entity_type": "service",
            "service": "carts",
            "pod": None,
            "fault_type": "memory_stress",
            "provisional_score": 0.90,
        },
        {
            "entity_type": "service",
            "service": "queue-master",
            "pod": None,
            "fault_type": "memory_stress",
            "provisional_score": 0.80,
        },
    ]
    pods = [
        {
            "entity_type": "pod",
            "service": "carts",
            "pod": "carts-1",
            "fault_type": "memory_stress",
            "provisional_score": 0.98,
        },
        {
            "entity_type": "pod",
            "service": "queue-master",
            "pod": "queue-master-1",
            "fault_type": "memory_stress",
            "provisional_score": 0.98,
        },
    ]

    reconciled_services, reconciled_pods, response = skill._reconcile_rankings_with_llm(
        {"evidence_tree": [], "topology": {}, "reasoning_rule_guide": []},
        services,
        pods,
    )

    assert skill.llm_client.prompt_names == ["cross_level_ranking_reconciliation"]
    assert response.notes == ["cross-level alignment applied"]
    assert reconciled_services[0]["service"] == "queue-master"
    assert reconciled_services[0]["notes"] == [
        "Aligned with strongest queue-master pod."
    ]
    assert reconciled_pods[0]["pod"] == "queue-master-1"


def test_cross_level_alignment_promotes_parent_service_for_top_local_pod():
    skill = RootCauseReasoningSkill({}, data_access=None, llm_client=None)
    services = [
        {
            "entity_type": "service",
            "service": "user",
            "pod": None,
            "fault_type": "network_delay",
            "provisional_score": 0.97,
            "supporting_evidence": ["pod user-1 latency_p99 shifted upward"],
            "notes": ["Evidence is mostly symptom-like."],
            "rule_hints": {"symptom_only_signal": True},
            "active_rules": ["symptom_only_signal"],
        },
        {
            "entity_type": "service",
            "service": "carts-db",
            "pod": None,
            "fault_type": "cpu_stress",
            "provisional_score": 0.55,
            "supporting_evidence": [],
            "notes": [],
            "rule_hints": {},
            "active_rules": [],
        },
    ]
    pods = [
        {
            "entity_type": "pod",
            "service": "carts-db",
            "pod": "carts-db-0",
            "fault_type": "cpu_stress",
            "provisional_score": 0.98,
            "supporting_evidence": [
                "pod carts-db-0 cpu_usage_pct: cpu_usage_pct shifted from 1.000 to 50.000"
            ],
            "notes": ["Strong local CPU anomaly."],
            "rule_hints": {"local_resource_support": True},
            "active_rules": ["local_resource_support"],
        }
    ]

    aligned = skill._align_parent_service_with_top_pod(services, pods)
    parent = next(item for item in aligned if item["service"] == "carts-db")

    assert parent["provisional_score"] == 0.98
    assert parent["supporting_evidence"][0].startswith("pod carts-db-0 cpu_usage_pct")
    assert "cross_level_parent_alignment" in parent["active_rules"]
    assert any("parent service promoted" in note for note in parent["notes"])


def test_cross_level_alignment_handles_moderate_top_local_pod_against_symptom_service():
    skill = RootCauseReasoningSkill({}, data_access=None, llm_client=None)
    services = [
        {
            "entity_type": "service",
            "service": "user",
            "pod": None,
            "fault_type": "network_delay",
            "provisional_score": 0.88,
            "supporting_evidence": ["pod user-1 latency_p99 shifted upward"],
            "notes": ["Multiple sibling pods show similar latency symptoms."],
            "rule_hints": {"symptom_only_signal": True, "shared_issue_hint": True},
            "active_rules": ["symptom_only_signal", "shared_multi_pod_anomaly"],
        },
        {
            "entity_type": "service",
            "service": "carts",
            "pod": None,
            "fault_type": "memory_stress",
            "provisional_score": 0.82,
            "supporting_evidence": [],
            "notes": [],
            "rule_hints": {},
            "active_rules": [],
        },
    ]
    pods = [
        {
            "entity_type": "pod",
            "service": "carts",
            "pod": "carts-1",
            "fault_type": "memory_stress",
            "provisional_score": 0.88,
            "supporting_evidence": [
                "pod carts-1 memory_usage_pct: memory_usage_pct shifted from 40.000 to 55.000"
            ],
            "notes": ["Concrete localized resource anomaly."],
            "rule_hints": {"local_resource_support": True},
            "active_rules": ["local_resource_support"],
        },
        {
            "entity_type": "pod",
            "service": "user",
            "pod": "user-1",
            "fault_type": "network_delay",
            "provisional_score": 0.82,
            "supporting_evidence": ["pod user-1 latency_p99 shifted upward"],
            "notes": [],
            "rule_hints": {"symptom_only_signal": True},
            "active_rules": ["symptom_only_signal"],
        },
    ]

    aligned = skill._align_parent_service_with_top_pod(services, pods)
    parent = next(item for item in aligned if item["service"] == "carts")

    assert parent["provisional_score"] == 0.88
    assert parent["supporting_evidence"][0].startswith("pod carts-1 memory_usage_pct")
    assert "cross_level_parent_alignment" in parent["active_rules"]


def test_cross_level_alignment_prefers_matching_service_fault_type():
    skill = RootCauseReasoningSkill({}, data_access=None, llm_client=None)
    services = [
        {
            "entity_type": "service",
            "service": "carts",
            "pod": None,
            "fault_type": "network_delay",
            "provisional_score": 0.98,
            "supporting_evidence": ["pod carts-1 latency_p99 shifted upward"],
            "notes": ["Mostly latency symptoms."],
            "rule_hints": {"symptom_only_signal": True},
            "active_rules": ["symptom_only_signal"],
        },
        {
            "entity_type": "service",
            "service": "orders-db",
            "pod": None,
            "fault_type": "network_delay",
            "provisional_score": 0.96,
            "supporting_evidence": ["pod orders-db-0 latency_p99 shifted upward"],
            "notes": [],
            "rule_hints": {"local_resource_support": True},
            "active_rules": ["local_resource_support"],
        },
        {
            "entity_type": "service",
            "service": "orders-db",
            "pod": None,
            "fault_type": "cpu_stress",
            "provisional_score": 0.94,
            "supporting_evidence": ["pod orders-db-0 cpu_usage_pct shifted upward"],
            "notes": [],
            "rule_hints": {"local_resource_support": True},
            "active_rules": ["local_resource_support"],
        },
    ]
    pods = [
        {
            "entity_type": "pod",
            "service": "orders-db",
            "pod": "orders-db-0",
            "fault_type": "cpu_stress",
            "provisional_score": 0.98,
            "supporting_evidence": [
                "pod orders-db-0 cpu_usage_pct: cpu_usage_pct shifted from 3.601 to 33.109"
            ],
            "notes": ["CPU anomaly is materially elevated."],
            "rule_hints": {"local_resource_support": True},
            "active_rules": ["local_resource_support"],
        }
    ]

    aligned = skill._align_parent_service_with_top_pod(services, pods)
    cpu_parent = next(
        item
        for item in aligned
        if item["service"] == "orders-db" and item["fault_type"] == "cpu_stress"
    )
    delay_parent = next(
        item
        for item in aligned
        if item["service"] == "orders-db" and item["fault_type"] == "network_delay"
    )

    assert cpu_parent["provisional_score"] == 0.98
    assert delay_parent["provisional_score"] == 0.96
    assert cpu_parent["supporting_evidence"][0].startswith(
        "pod orders-db-0 cpu_usage_pct"
    )


def test_cross_level_alignment_uses_final_pod_ordering_not_raw_llm_order():
    skill = RootCauseReasoningSkill({}, data_access=None, llm_client=None)
    services = [
        {
            "entity_type": "service",
            "service": "user",
            "pod": None,
            "fault_type": "exception_injection",
            "provisional_score": 0.98,
            "supporting_evidence": ["keyword_spike='error' count 1->4"],
            "notes": [],
            "rule_hints": {},
            "active_rules": [],
        },
        {
            "entity_type": "service",
            "service": "orders-db",
            "pod": None,
            "fault_type": "cpu_stress",
            "provisional_score": 0.95,
            "supporting_evidence": ["pod orders-db-0 cpu_usage_pct shifted upward"],
            "notes": [],
            "rule_hints": {"local_resource_support": True},
            "active_rules": ["local_resource_support"],
        },
    ]
    pods = [
        {
            "entity_type": "pod",
            "service": "user",
            "pod": "user-1",
            "fault_type": "exception_injection",
            "provisional_score": 0.98,
            "supporting_evidence": ["keyword_spike='error' count 1->4"],
            "notes": [],
            "rule_hints": {},
            "active_rules": [],
        },
        {
            "entity_type": "pod",
            "service": "orders-db",
            "pod": "orders-db-0",
            "fault_type": "cpu_stress",
            "provisional_score": 0.98,
            "supporting_evidence": [
                "pod orders-db-0 cpu_usage_pct: cpu_usage_pct shifted from 3.601 to 33.109"
            ],
            "notes": [],
            "rule_hints": {"local_resource_support": True},
            "active_rules": ["local_resource_support"],
        },
    ]

    aligned = skill._align_parent_service_with_top_pod(services, pods)
    orders_db = next(item for item in aligned if item["service"] == "orders-db")
    user = next(item for item in aligned if item["service"] == "user")

    assert orders_db["provisional_score"] == 0.98
    assert "cross_level_parent_alignment" in orders_db["active_rules"]
    assert user["active_rules"] == []


def test_final_service_alignment_promotes_top_pod_parent_when_scores_are_close():
    skill = RootCauseReasoningSkill({}, data_access=None, llm_client=None)
    service_top5 = [
        RankedHypothesis(
            service="queue-master",
            pod=None,
            fault_type="memory_stress",
            score=0.89,
            supporting_evidence=[],
            notes="",
        ),
        RankedHypothesis(
            service="user",
            pod=None,
            fault_type="network_delay",
            score=0.70,
            supporting_evidence=[],
            notes="",
        ),
    ]
    pod_top5 = [
        RankedHypothesis(
            service="user",
            pod="user-1",
            fault_type="network_delay",
            score=0.88,
            supporting_evidence=[],
            notes="",
        )
    ]

    aligned = skill._align_final_service_with_top_pod(service_top5, pod_top5)

    assert aligned[0].service == "user"
    assert aligned[0].fault_type == "network_delay"
    assert aligned[0].score == 0.89
    assert "Final cross-level alignment" in aligned[0].notes


def test_final_service_alignment_does_not_promote_top_pod_parent_when_score_gap_is_large():
    skill = RootCauseReasoningSkill({}, data_access=None, llm_client=None)
    service_top5 = [
        RankedHypothesis(
            service="queue-master",
            pod=None,
            fault_type="memory_stress",
            score=0.95,
            supporting_evidence=[],
            notes="",
        ),
        RankedHypothesis(
            service="user",
            pod=None,
            fault_type="network_delay",
            score=0.70,
            supporting_evidence=[],
            notes="",
        ),
    ]
    pod_top5 = [
        RankedHypothesis(
            service="user",
            pod="user-1",
            fault_type="network_delay",
            score=0.88,
            supporting_evidence=[],
            notes="",
        )
    ]

    aligned = skill._align_final_service_with_top_pod(service_top5, pod_top5)

    assert aligned[0].service == "queue-master"
    assert aligned[1].service == "user"
    assert aligned[1].score == 0.70


def test_pod_candidate_uses_only_own_pod_evidence_not_same_service_context():
    service_latency = AnomalyRecord(
        source="metric",
        entity_type="service",
        entity_name="catalogue",
        metric_or_pattern="latency_p99",
        abnormal_value=138.0,
        baseline_value=13.0,
        delta=125.0,
        zscore=5.0,
        severity=0.9,
        summary="latency_p99 shifted from 13.000 to 138.000",
        metadata={"service": "catalogue", "delta_ratio": 9.6},
    )
    pod_trace = AnomalyRecord(
        source="trace",
        entity_type="pod",
        entity_name="catalogue-58bdd4d4f9-48jpd",
        metric_or_pattern="path_latency_spike",
        abnormal_value=48972.0,
        baseline_value=3768.0,
        delta=45204.0,
        zscore=None,
        severity=0.95,
        summary="path_latency_spike 3768.00->48972.00",
        metadata={"service": "catalogue", "pod": "catalogue-58bdd4d4f9-48jpd"},
    )
    state = RCAState(
        incident_id="incident-1",
        abnormal_window={},
        baseline_window={},
        backend_mode="api",
        topology={
            "services": ["catalogue"],
            "edges": [{"source": "catalogue", "target": "catalogue-db"}],
        },
        metrics_evidence=SkillResult(
            service_evidence=[
                ServiceEvidence(
                    service="catalogue",
                    score=0.9,
                    supporting_evidence=[
                        "service catalogue latency_p99: latency_p99 shifted from 13.000 to 138.000"
                    ],
                    anomaly_records=[service_latency],
                )
            ],
            anomaly_records=[service_latency],
        ),
        traces_evidence=SkillResult(
            pod_evidence=[
                PodEvidence(
                    pod="catalogue-58bdd4d4f9-48jpd",
                    service="catalogue",
                    score=0.95,
                    supporting_evidence=["path_latency_spike 3768.00->48972.00"],
                    anomaly_records=[pod_trace],
                )
            ],
            anomaly_records=[pod_trace],
        ),
    )
    skill = RootCauseReasoningSkill({}, data_access=None, llm_client=None)

    candidates = skill._build_pod_candidates(
        state,
        {
            "anomalous_services": ["catalogue"],
            "shared_downstream_targets": [],
            "path_services": ["catalogue"],
            "trace_targets_by_service": {"catalogue": ["catalogue-db"]},
        },
        {},
    )

    catalogue_candidate = next(
        item for item in candidates if item["service"] == "catalogue"
    )
    assert catalogue_candidate["fault_type"] == "network_delay"
    assert (
        "path_latency_spike 3768.00->48972.00"
        in catalogue_candidate["supporting_evidence"]
    )
    assert (
        "service catalogue latency_p99: latency_p99 shifted from 13.000 to 138.000"
        not in catalogue_candidate["supporting_evidence"]
    )
    assert not any(
        "pod catalogue-58bdd4d4f9-48jpd latency_p99" in item
        for item in catalogue_candidate["supporting_evidence"]
    )


def test_llm_context_includes_topology_and_evidence_inventory():
    record = AnomalyRecord(
        source="metric",
        entity_type="pod",
        entity_name="orders-1",
        metric_or_pattern="restart_count",
        abnormal_value=0.2,
        baseline_value=0.0,
        delta=0.2,
        zscore=0.0,
        severity=1.0,
        summary="restart_count mean shifted from 0.000 to 0.200; max shifted from 0.000 to 1.000",
        metadata={
            "service": "orders",
            "pod": "orders-1",
            "abnormal_max": 1.0,
            "baseline_max": 0.0,
        },
    )
    evidence = SkillResult(
        service_evidence=[
            ServiceEvidence(
                service="orders",
                score=1.0,
                supporting_evidence=[record.summary],
                anomaly_records=[record],
            )
        ],
        pod_evidence=[
            PodEvidence(
                pod="orders-1",
                service="orders",
                score=1.0,
                supporting_evidence=[record.summary],
                anomaly_records=[record],
            )
        ],
    )
    state = RCAState(
        incident_id="incident-1",
        abnormal_window={
            "start": "2026-04-27T00:00:00Z",
            "end": "2026-04-27T00:15:00Z",
        },
        baseline_window={
            "start": "2026-04-27T00:30:00Z",
            "end": "2026-04-27T00:45:00Z",
        },
        backend_mode="csv",
        topology={
            "services": ["front-end", "orders"],
            "edges": [{"source": "front-end", "target": "orders"}],
        },
        metrics_evidence=evidence,
    )
    request = SimpleNamespace(
        incident_id="incident-1",
        abnormal_window=SimpleNamespace(
            start="2026-04-27T00:00:00Z", end="2026-04-27T00:15:00Z"
        ),
        baseline_window=SimpleNamespace(
            start="2026-04-27T00:30:00Z", end="2026-04-27T00:45:00Z"
        ),
        namespace="sock-shop",
        config_bundle={
            "rootcause_reasoning_rules": {
                "rule_groups": {
                    "local_vs_propagated": {
                        "enabled": True,
                        "rules": [
                            {
                                "id": "local_resource_support",
                                "description": "Strong local evidence supports local-fault hypotheses.",
                            }
                        ],
                    }
                }
            }
        },
    )
    skill = RootCauseReasoningSkill(
        {"llm_context": {"max_evidence_per_entity": 4}},
        data_access=None,
        llm_client=None,
    )

    context = skill._build_llm_context(
        request,
        state,
        {
            "anomalous_services": ["orders"],
            "pod_score_summary": {"orders": [{"pod": "orders-1", "score": 1.0}]},
        },
        service_candidates=[{"service": "orders", "fault_type": "pod_failure"}],
        pod_candidates=[
            {"service": "orders", "pod": "orders-1", "fault_type": "pod_failure"}
        ],
    )

    assert "incident_id" not in context
    assert context["candidate_counts"] == {"service": 1, "pod": 1}
    assert context["topology"]["edges"] == [{"source": "front-end", "target": "orders"}]
    assert "evidence_context" not in context
    assert "pod_score_summary" not in context["rule_context"]
    orders = next(
        item for item in context["evidence_tree"] if item["service"] == "orders"
    )
    assert orders["pods"][0]["pod"] == "orders-1"
    assert "severity" not in orders["pods"][0]["evidence"][0]
    assert "restart_count" in orders["pods"][0]["evidence"][0]["summary"]
    assert context["reasoning_rule_guide"][0]["category"] == "local_vs_propagated"
    assert (
        context["reasoning_rule_guide"][0]["rules"][0]["id"] == "local_resource_support"
    )


def test_llm_context_adds_single_exception_vs_adjacent_multi_pod_symptom_hint():
    shipping_exception = AnomalyRecord(
        source="log",
        entity_type="pod",
        entity_name="shipping-1",
        metric_or_pattern="template_spike",
        abnormal_value=12,
        baseline_value=0,
        delta=12,
        zscore=None,
        severity=0.95,
        summary="template_spike='java.lang.RuntimeException: failed shipping request' count 0->12",
        metadata={"service": "shipping", "pod": "shipping-1"},
    )
    orders_latency_1 = AnomalyRecord(
        source="metric",
        entity_type="pod",
        entity_name="orders-1",
        metric_or_pattern="latency_p99",
        abnormal_value=317.0,
        baseline_value=73.0,
        delta=244.0,
        zscore=4.0,
        severity=0.9,
        summary="latency_p99 shifted from 73.000 to 317.000",
        metadata={"service": "orders", "pod": "orders-1", "delta_ratio": 3.3},
    )
    orders_latency_2 = AnomalyRecord(
        source="metric",
        entity_type="pod",
        entity_name="orders-2",
        metric_or_pattern="latency_p99",
        abnormal_value=139.0,
        baseline_value=68.0,
        delta=71.0,
        zscore=3.0,
        severity=0.8,
        summary="latency_p99 shifted from 68.000 to 139.000",
        metadata={"service": "orders", "pod": "orders-2", "delta_ratio": 1.0},
    )
    state = RCAState(
        incident_id="pod_java_exception_shipping_001",
        abnormal_window={
            "start": "2026-04-27T00:00:00Z",
            "end": "2026-04-27T00:15:00Z",
        },
        baseline_window={
            "start": "2026-04-27T00:30:00Z",
            "end": "2026-04-27T00:45:00Z",
        },
        backend_mode="csv",
        topology={
            "services": ["orders", "shipping"],
            "edges": [{"source": "orders", "target": "shipping"}],
        },
        metrics_evidence=SkillResult(
            pod_evidence=[
                PodEvidence(
                    pod="orders-1",
                    service="orders",
                    score=0.9,
                    supporting_evidence=[],
                    anomaly_records=[orders_latency_1],
                ),
                PodEvidence(
                    pod="orders-2",
                    service="orders",
                    score=0.8,
                    supporting_evidence=[],
                    anomaly_records=[orders_latency_2],
                ),
            ],
            anomaly_records=[orders_latency_1, orders_latency_2],
        ),
        logs_evidence=SkillResult(
            pod_evidence=[
                PodEvidence(
                    pod="shipping-1",
                    service="shipping",
                    score=0.95,
                    supporting_evidence=[],
                    anomaly_records=[shipping_exception],
                )
            ],
            anomaly_records=[shipping_exception],
        ),
    )
    request = SimpleNamespace(
        incident_id="pod_java_exception_shipping_001",
        abnormal_window=SimpleNamespace(
            start="2026-04-27T00:00:00Z", end="2026-04-27T00:15:00Z"
        ),
        baseline_window=SimpleNamespace(
            start="2026-04-27T00:30:00Z", end="2026-04-27T00:45:00Z"
        ),
        namespace="sock-shop",
        config_bundle={"rootcause_reasoning_rules": {}},
    )
    skill = RootCauseReasoningSkill(
        {"llm_context": {"max_evidence_per_entity": 4}},
        data_access=None,
        llm_client=None,
    )

    context = skill._build_llm_context(
        request,
        state,
        {
            "anomalous_services": ["orders", "shipping"],
            "downstream_map": {"orders": {"shipping"}},
            "upstream_map": {"shipping": {"orders"}},
        },
        service_candidates=[],
        pod_candidates=[],
    )

    assert "incident_id" not in context
    hint = next(
        item
        for item in context["propagation_hints"]
        if item["type"] == "single_pod_exception_vs_adjacent_multi_pod_symptoms"
    )
    assert hint["candidate_service"] == "shipping"
    assert hint["candidate_pod"] == "shipping-1"
    assert hint["related_service"] == "orders"
    assert hint["related_affected_pods"] == ["orders-1", "orders-2"]
