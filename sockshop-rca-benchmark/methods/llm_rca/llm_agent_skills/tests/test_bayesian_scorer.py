from rca_agent_skills.common.models import AnomalyRecord, PodEvidence
from rca_agent_skills.skills.rootcause_reasoning_skill.bayesian_scorer import (
    BayesianScorer,
)
from rca_agent_skills.skills.rootcause_reasoning_skill.schemas import (
    RuleEvaluation,
    RuleHints,
)
from rca_agent_skills.skills.rootcause_reasoning_skill.skill import (
    RootCauseReasoningSkill,
)


def _pod_evidence(metric: str, severity: float = 0.9) -> PodEvidence:
    return PodEvidence(
        pod="orders-1",
        service="orders",
        score=severity,
        supporting_evidence=[metric],
        anomaly_records=[
            AnomalyRecord(
                source="metric",
                entity_type="pod",
                entity_name="orders-1",
                metric_or_pattern=metric,
                abnormal_value=1,
                baseline_value=0,
                delta=1,
                zscore=None,
                severity=severity,
                summary=f"{metric} shifted upward",
                metadata={"service": "orders", "pod": "orders-1"},
            )
        ],
    )


def test_cpu_evidence_favors_cpu_stress_over_network_delay():
    scorer = BayesianScorer({"enabled": True})
    evidence = [_pod_evidence("cpu_usage_pct", 0.95)]

    cpu_score = scorer.score_candidate(
        {"fault_type": "cpu_stress"}, evidence, RuleEvaluation()
    )
    delay_score = scorer.score_candidate(
        {"fault_type": "network_delay"}, evidence, RuleEvaluation()
    )

    assert cpu_score["bayes_log_score"] > delay_score["bayes_log_score"]


def test_latency_evidence_has_high_network_delay_likelihood():
    scorer = BayesianScorer({"enabled": True})
    result = scorer.score_candidate(
        {"fault_type": "network_delay"},
        [_pod_evidence("latency_p99", 0.9)],
        RuleEvaluation(),
    )

    latency_term = next(
        item for item in result["likelihood_terms"] if item["evidence"] == "latency_spike"
    )
    assert latency_term["p_e_given_fault"] == 0.85


def test_restart_and_ready_evidence_favor_pod_failure():
    scorer = BayesianScorer({"enabled": True})
    evidence = [_pod_evidence("restart_count", 0.9), _pod_evidence("ready_ratio", 0.9)]

    pod_failure_score = scorer.score_candidate(
        {"fault_type": "pod_failure"}, evidence, RuleEvaluation()
    )
    cpu_score = scorer.score_candidate(
        {"fault_type": "cpu_stress"}, evidence, RuleEvaluation()
    )

    assert pod_failure_score["bayes_log_score"] > cpu_score["bayes_log_score"]


def test_multiple_local_failure_evidence_is_not_penalized_below_weak_network_rx():
    scorer = BayesianScorer({"enabled": True})
    strong_local = [
        _pod_evidence("restart_count", 1.0),
        _pod_evidence("latency_p99", 1.0),
        _pod_evidence("memory_usage_pct", 0.9),
    ]
    weak_network = [_pod_evidence("network_rx", 0.9)]

    pod_failure_score = scorer.score_candidate(
        {"fault_type": "pod_failure"}, strong_local, RuleEvaluation()
    )
    network_delay_score = scorer.score_candidate(
        {"fault_type": "network_delay"}, weak_network, RuleEvaluation()
    )

    assert pod_failure_score["bayes_log_score"] > network_delay_score["bayes_log_score"]


def test_rule_hints_become_bayesian_prior_multipliers():
    scorer = BayesianScorer({"enabled": True})

    boosted = scorer.score_candidate(
        {"fault_type": "cpu_stress"},
        [_pod_evidence("cpu_usage_pct", 0.9)],
        RuleEvaluation(rule_hints=RuleHints(local_resource_support=True)),
    )
    penalized = scorer.score_candidate(
        {"fault_type": "network_delay"},
        [_pod_evidence("latency_p99", 0.9)],
        RuleEvaluation(rule_hints=RuleHints(symptom_only_signal=True)),
    )

    assert boosted["prior_multiplier"] > 1.0
    assert penalized["prior_multiplier"] < 1.0


def test_posterior_probability_normalization_sums_to_one():
    skill = RootCauseReasoningSkill(
        {"bayesian": {"enabled": True}}, data_access=None, llm_client=None
    )
    candidates = [
        {"fault_type": "cpu_stress", "bayes_log_score": -1.0, "provisional_score": 0.1},
        {
            "fault_type": "network_delay",
            "bayes_log_score": -2.0,
            "provisional_score": 0.1,
        },
    ]

    skill._normalize_posterior_probabilities(candidates)

    assert round(sum(item["posterior_probability"] for item in candidates), 6) == 1.0


def test_bayesian_disabled_keeps_heuristic_provisional_score():
    skill = RootCauseReasoningSkill({}, data_access=None, llm_client=None)

    candidate = skill._score_candidate(
        {"fault_type": "cpu_stress"},
        [_pod_evidence("cpu_usage_pct", 0.9)],
        RuleEvaluation(),
        topology=None,
        heuristic_score=0.77,
    )

    assert candidate["heuristic_score"] == 0.77
    assert candidate["posterior_score"] == 0.77
    assert candidate["provisional_score"] == 0.77
