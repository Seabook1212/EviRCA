from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RuleHints:
    shared_issue_hint: bool = False
    single_pod_local_hint: bool = False
    dependency_symptom_hint: bool = False
    downstream_local_failure_hint: bool = False
    shared_downstream_dependency_hint: bool = False
    temporal_precedence_hint: bool = False
    temporal_conflict: bool = False
    topology_conflict: bool = False
    topology_support_hint: bool = False
    local_resource_support: bool = False
    symptom_only_signal: bool = False


@dataclass
class RuleEvaluation:
    rule_hints: RuleHints = field(default_factory=RuleHints)
    active_rules: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    score_adjustment: float = 0.0


@dataclass
class CandidateHypothesis:
    entity_type: str
    entity_name: str
    service: str
    pod: str | None
    fault_type: str
    provisional_score: float
    evidence_count: int
    supporting_evidence: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    dependency_boost: float = 0.0
    fault_type_candidates: list[str] = field(default_factory=list)
    rule_hints: dict[str, bool] = field(default_factory=dict)
    active_rules: list[str] = field(default_factory=list)
    heuristic_score: float = 0.0
    bayes_log_score: float = 0.0
    posterior_score: float = 0.0
    posterior_probability: float = 0.0
    prior_probability: float = 0.0
    prior_multiplier: float = 1.0
    likelihood_terms: list[dict] = field(default_factory=list)
    prior_terms: list[dict] = field(default_factory=list)
