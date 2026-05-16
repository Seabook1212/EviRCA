from __future__ import annotations

import copy
import math
from dataclasses import asdict, is_dataclass
from typing import Any

from .evidence_mapper import extract_soft_evidence


DEFAULT_BAYESIAN_CONFIG = {
    "enabled": False,
    "temperature": 1.0,
    "default_fault_prior": 0.05,
    "default_evidence_likelihood": 0.10,
    "contradictory_evidence_weight": 0.25,
    "min_probability": 1.0e-6,
    "fault_priors": {
        "cpu_stress": 0.05,
        "memory_stress": 0.05,
        "pod_failure": 0.04,
        "network_delay": 0.06,
        "network_loss": 0.04,
        "network_partition": 0.03,
        "io_fault": 0.04,
        "exception_injection": 0.05,
    },
    "evidence_likelihoods": {
        "cpu_stress": {
            "cpu_high": 0.90,
            "memory_high": 0.10,
            "restart_increase": 0.20,
            "ready_drop": 0.15,
            "error_increase": 0.20,
            "latency_spike": 0.30,
            "trace_edge_latency": 0.25,
            "log_keyword_spike": 0.10,
            "log_template_spike": 0.10,
        },
        "memory_stress": {
            "memory_high": 0.90,
            "cpu_high": 0.15,
            "restart_increase": 0.35,
            "ready_drop": 0.25,
            "error_increase": 0.25,
            "latency_spike": 0.25,
            "log_keyword_spike": 0.20,
        },
        "pod_failure": {
            "restart_increase": 0.85,
            "ready_drop": 0.80,
            "error_increase": 0.45,
            "trace_edge_failure": 0.40,
            "latency_spike": 0.30,
            "log_keyword_spike": 0.35,
            "log_template_spike": 0.35,
        },
        "network_delay": {
            "latency_spike": 0.85,
            "trace_edge_latency": 0.85,
            "trace_path_latency": 0.75,
            "error_increase": 0.20,
            "success_drop": 0.20,
            "network_rx_tx": 0.30,
            "cpu_high": 0.10,
            "memory_high": 0.10,
        },
        "network_loss": {
            "trace_edge_failure": 0.80,
            "error_increase": 0.70,
            "success_drop": 0.75,
            "latency_spike": 0.55,
            "network_rx_tx": 0.35,
            "log_keyword_spike": 0.30,
        },
        "network_partition": {
            "missing_data_gap": 0.85,
            "success_drop": 0.75,
            "error_increase": 0.65,
            "trace_edge_failure": 0.75,
            "latency_spike": 0.60,
            "network_rx_tx": 0.40,
        },
        "io_fault": {
            "restart_increase": 0.70,
            "ready_drop": 0.65,
            "error_increase": 0.80,
            "trace_edge_failure": 0.60,
            "success_drop": 0.65,
            "latency_spike": 0.40,
            "log_keyword_spike": 0.45,
            "log_template_spike": 0.55,
            "memory_high": 0.20,
            "trace_edge_latency": 0.45,
        },
        "exception_injection": {
            "log_keyword_spike": 0.80,
            "log_template_spike": 0.75,
            "log_level_shift": 0.70,
            "error_increase": 0.55,
            "trace_edge_failure": 0.35,
        },
    },
    "rule_prior_multipliers": {
        "local_resource_support": 1.40,
        "single_pod_local_hint": 1.30,
        "shared_issue_hint": 1.15,
        "shared_downstream_dependency_hint": 1.20,
        "topology_support_hint": 1.20,
        "temporal_precedence_hint": 1.20,
        "symptom_only_signal": 0.60,
        "dependency_symptom_hint": 0.70,
        "downstream_local_failure_hint": 0.65,
        "topology_conflict": 0.40,
        "temporal_conflict": 0.50,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _clip_probability(value: float, min_probability: float) -> float:
    return max(min_probability, min(1.0 - min_probability, float(value)))


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _rule_hints_from_eval(rule_eval: Any) -> dict[str, bool]:
    rule_hints = getattr(rule_eval, "rule_hints", rule_eval)
    if is_dataclass(rule_hints):
        return asdict(rule_hints)
    if isinstance(rule_hints, dict):
        return rule_hints
    return {}


class BayesianScorer:
    def __init__(self, config: dict):
        self.config = _deep_merge(DEFAULT_BAYESIAN_CONFIG, config or {})

    @property
    def enabled(self) -> bool:
        return bool(self.config.get("enabled", False))

    @property
    def temperature(self) -> float:
        try:
            value = float(self.config.get("temperature", 1.0))
        except (TypeError, ValueError):
            value = 1.0
        return max(1.0e-6, value)

    @property
    def min_probability(self) -> float:
        try:
            value = float(self.config.get("min_probability", 1.0e-6))
        except (TypeError, ValueError):
            value = 1.0e-6
        return max(1.0e-12, min(0.01, value))

    def score_candidate(
        self,
        candidate: dict,
        evidence_items: list,
        rule_eval,
        topology: dict | None = None,
    ) -> dict:
        fault_type = candidate.get("fault_type")
        fault_priors = self.config.get("fault_priors", {})
        likelihoods_by_fault = self.config.get("evidence_likelihoods", {})
        default_prior = float(self.config.get("default_fault_prior", 0.05))
        default_likelihood = float(
            self.config.get("default_evidence_likelihood", 0.10)
        )
        contradictory_weight = float(
            self.config.get("contradictory_evidence_weight", 0.25)
        )
        min_probability = self.min_probability
        default_likelihood = _clip_probability(default_likelihood, min_probability)

        prior_probability = _clip_probability(
            float(fault_priors.get(fault_type, default_prior)), min_probability
        )
        rule_hints = _rule_hints_from_eval(rule_eval)
        rule_multipliers = self.config.get("rule_prior_multipliers", {})

        prior_multiplier = 1.0
        prior_terms: list[dict] = [
            {
                "name": "fault_prior",
                "multiplier": 1.0,
                "probability": round(prior_probability, 6),
                "reason": f"Base prior for fault_type={fault_type}.",
            }
        ]
        for hint_name, active in sorted(rule_hints.items()):
            if not active or hint_name not in rule_multipliers:
                continue
            multiplier = float(rule_multipliers[hint_name])
            prior_multiplier *= multiplier
            prior_terms.append(
                {
                    "name": hint_name,
                    "multiplier": round(multiplier, 4),
                    "reason": f"Rule/topology prior from active hint: {hint_name}.",
                }
            )

        fault_likelihoods = likelihoods_by_fault.get(fault_type, {})
        likelihood_terms = []
        likelihood_log_sum = 0.0
        for evidence in extract_soft_evidence(evidence_items):
            evidence_name = evidence["name"]
            severity = _clip_probability(
                float(evidence.get("severity", 0.5)), min_probability
            )
            p_present = _clip_probability(
                float(fault_likelihoods.get(evidence_name, default_likelihood)),
                min_probability,
            )
            p_term = severity * p_present + (1.0 - severity) * (1.0 - p_present)
            p_term = _clip_probability(p_term, min_probability)
            baseline_term = (
                severity * default_likelihood
                + (1.0 - severity) * (1.0 - default_likelihood)
            )
            baseline_term = _clip_probability(baseline_term, min_probability)
            log_contribution = math.log(p_term / baseline_term)
            if log_contribution < 0:
                log_contribution *= max(0.0, contradictory_weight)
            likelihood_log_sum += log_contribution
            likelihood_terms.append(
                {
                    "evidence": evidence_name,
                    "severity": round(severity, 4),
                    "p_e_given_fault": round(p_present, 4),
                    "p_term": round(p_term, 6),
                    "baseline_p_term": round(baseline_term, 6),
                    "log_contribution": round(log_contribution, 6),
                    "source": evidence.get("source"),
                    "raw_pattern": evidence.get("raw_pattern"),
                    "summary": evidence.get("summary"),
                    "baseline_value": evidence.get("baseline_value"),
                    "abnormal_value": evidence.get("abnormal_value"),
                    "delta": evidence.get("delta"),
                    "delta_ratio": evidence.get("delta_ratio"),
                    "ratio": evidence.get("ratio"),
                    "relative_change_pct": evidence.get("relative_change_pct"),
                }
            )

        log_score = (
            math.log(prior_probability)
            + math.log(max(min_probability, prior_multiplier))
            + likelihood_log_sum
        )
        raw_score = math.exp(max(-745.0, min(709.0, log_score)))
        posterior_score = _sigmoid(log_score / self.temperature)

        return {
            "bayes_log_score": round(log_score, 6),
            "bayes_raw_score": raw_score,
            "prior_probability": round(prior_probability, 6),
            "prior_multiplier": round(prior_multiplier, 6),
            "likelihood_score": math.exp(max(-745.0, min(709.0, likelihood_log_sum))),
            "posterior_score": round(posterior_score, 6),
            "likelihood_terms": likelihood_terms,
            "prior_terms": prior_terms,
        }
