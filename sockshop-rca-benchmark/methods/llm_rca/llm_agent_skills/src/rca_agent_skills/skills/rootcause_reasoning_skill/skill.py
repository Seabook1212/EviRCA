from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, replace

from rca_agent_skills.common.constants import DEFAULT_TOP_K
from rca_agent_skills.common.logging_utils import get_logger, log_json
from rca_agent_skills.common.models import RankedHypothesis
from rca_agent_skills.data_access.topology_loader import service_from_pod
from rca_agent_skills.orchestrator_agent.schemas import RCAResponse
from .aggregator import aggregate_pod_evidence, aggregate_service_evidence
from .bayesian_scorer import BayesianScorer
from .checker import build_rule_context, evaluate_rule_hints, run_light_checks
from .fault_matcher import match_fault_types
from .ranker import rank_with_llm


EVIDENCE_SOURCE_PRIORITY = {"metric": 3, "trace": 2, "log": 1}
EVIDENCE_PATTERN_PRIORITY = {
    "restart_count": 100,
    "ready_ratio": 95,
    "pod_failure": 92,
    "memory_usage_pct": 88,
    "cpu_usage_pct": 86,
    "error_count": 82,
    "success_rate": 80,
    "edge_failure_spike": 78,
    "keyword_spike": 72,
    "level_shift": 70,
    "latency_p99": 64,
    "latency_p95": 62,
    "latency_p90": 60,
    "latency_p50": 58,
    "edge_latency_spike": 56,
    "path_latency_spike": 54,
    "request_rate": 42,
    "network_rx": 36,
    "network_tx": 36,
    "template_spike": 30,
}
EXCEPTION_HINT_TOKENS = {
    "exception",
    "stacktrace",
    "stack trace",
    "throwable",
    "panic",
    "failed",
    "failure",
    "error",
}
EXCEPTION_HINT_PATTERNS = {
    "keyword_spike",
    "template_spike",
    "level_shift",
    "exception_injection",
}
PROPAGATED_SYMPTOM_PATTERNS = {
    "error_count",
    "success_rate",
    "latency_p50",
    "latency_p90",
    "latency_p95",
    "latency_p99",
    "edge_latency_spike",
    "edge_failure_spike",
    "path_latency_spike",
}
LOCAL_ROOT_FAULT_TYPES = {
    "cpu_stress",
    "memory_stress",
    "pod_failure",
    "exception_injection",
}
LOCAL_ROOT_EVIDENCE_TOKENS = {
    "cpu_usage_pct",
    "memory_usage_pct",
    "restart_count",
    "ready_ratio",
    "oom",
    "outofmemory",
    "crash",
    "killed",
    "exception",
    "stacktrace",
    "stack trace",
}


class RootCauseReasoningSkill:
    def __init__(self, settings: dict, data_access, llm_client):
        self.settings = settings
        self.data_access = data_access
        self.llm_client = llm_client
        self.logger = get_logger(self.__class__.__name__)
        self.debug = settings.get("debug", {})
        self.bayesian_scorer = BayesianScorer(settings.get("bayesian", {}))

    def _bayesian_enabled(self) -> bool:
        return self.bayesian_scorer.enabled

    def _is_business_service(self, service: str | None, topology: dict | None) -> bool:
        if not service:
            return False
        topology_services = set((topology or {}).get("services", []))
        return not topology_services or service in topology_services

    def _score_candidate(
        self,
        candidate: dict,
        evidence_items: list,
        rule_eval,
        topology: dict | None,
        heuristic_score: float,
    ) -> dict:
        scored = dict(candidate)
        heuristic_score = round(max(0.05, min(0.98, float(heuristic_score))), 4)
        scored["heuristic_score"] = heuristic_score
        if self._bayesian_enabled():
            bayes_result = self.bayesian_scorer.score_candidate(
                candidate=scored,
                evidence_items=evidence_items,
                rule_eval=rule_eval,
                topology=topology,
            )
            scored.update(bayes_result)
            scored["posterior_score"] = round(float(bayes_result["posterior_score"]), 6)
            scored["provisional_score"] = scored["posterior_score"]
        else:
            scored["posterior_score"] = heuristic_score
            scored["posterior_probability"] = 0.0
            scored["provisional_score"] = heuristic_score
            scored.setdefault("bayes_log_score", 0.0)
            scored.setdefault("prior_probability", 0.0)
            scored.setdefault("prior_multiplier", 1.0)
            scored.setdefault("likelihood_terms", [])
            scored.setdefault("prior_terms", [])
        return scored

    def _apply_bayesian_prior_multiplier(
        self, candidate: dict, name: str, multiplier: float, reason: str
    ) -> None:
        if not self._bayesian_enabled():
            return
        prior_terms = list(candidate.get("prior_terms", []) or [])
        prior_terms.append(
            {
                "name": name,
                "multiplier": round(float(multiplier), 4),
                "reason": reason,
            }
        )
        candidate["prior_terms"] = prior_terms
        candidate["prior_multiplier"] = round(
            float(candidate.get("prior_multiplier", 1.0)) * float(multiplier), 6
        )
        if "bayes_log_score" not in candidate:
            return
        candidate["bayes_log_score"] = round(
            float(candidate.get("bayes_log_score", 0.0))
            + math.log(max(1.0e-6, float(multiplier))),
            6,
        )
        temperature = self.bayesian_scorer.temperature
        log_score = float(candidate["bayes_log_score"]) / temperature
        if log_score >= 0:
            posterior_score = 1.0 / (1.0 + math.exp(-log_score))
        else:
            exp_value = math.exp(log_score)
            posterior_score = exp_value / (1.0 + exp_value)
        candidate["posterior_score"] = round(posterior_score, 6)
        candidate["provisional_score"] = candidate["posterior_score"]

    def _candidate_sort_score(self, candidate: dict) -> float:
        if self._bayesian_enabled() and candidate.get("posterior_probability"):
            return float(candidate.get("posterior_probability", 0.0))
        return float(candidate.get("provisional_score", 0.0))

    def _normalize_posterior_probabilities(self, candidates: list[dict]) -> list[dict]:
        if not candidates:
            return candidates
        temperature = self.bayesian_scorer.temperature
        log_scores = []
        for candidate in candidates:
            if self._bayesian_enabled() and "bayes_log_score" in candidate:
                log_scores.append(float(candidate.get("bayes_log_score", 0.0)) / temperature)
            else:
                log_scores.append(float(candidate.get("provisional_score", 0.0)) / temperature)
        max_log_score = max(log_scores)
        exp_scores = [math.exp(value - max_log_score) for value in log_scores]
        total = sum(exp_scores) or 1.0
        for candidate, exp_score in zip(candidates, exp_scores):
            candidate["posterior_probability"] = round(exp_score / total, 6)
        return candidates

    def _build_pod_candidates(
        self, state, rule_context: dict, rules_config: dict
    ) -> list[dict]:
        topology = state.topology
        support_limit = self._max_candidate_supporting_evidence()
        pod_candidates: list[dict] = []
        for pod, evidence_items in aggregate_pod_evidence(state).items():
            service = evidence_items[0].service if evidence_items else "unknown"
            if not self._is_business_service(service, topology):
                continue
            rule_eval = evaluate_rule_hints(
                {"entity_type": "pod", "service": service, "pod": pod},
                evidence_items,
                rule_context,
                topology,
                rules_config,
            )
            matches = match_fault_types(
                evidence_items,
                topology,
                rule_hints=asdict(rule_eval.rule_hints),
                rules_config=rules_config,
                support_limit=support_limit,
            )
            background_only = self._background_only(evidence_items)
            for fault_type, score, support, notes, dependency_boost in matches:
                heuristic_score = min(0.98, score + rule_eval.score_adjustment)
                candidate = {
                    "entity_type": "pod",
                    "service": service,
                    "pod": pod,
                    "fault_type": fault_type,
                    "fault_type_candidates": [fault_type],
                    "evidence_count": sum(
                        len(item.anomaly_records) for item in evidence_items
                    ),
                    "supporting_evidence": support[:support_limit],
                    "notes": list(dict.fromkeys(notes + rule_eval.notes)),
                    "dependency_boost": dependency_boost,
                    "background_only": background_only,
                    "rule_hints": asdict(rule_eval.rule_hints),
                    "active_rules": rule_eval.active_rules,
                    "rule_score_adjustment": rule_eval.score_adjustment,
                }
                pod_candidates.append(
                    self._score_candidate(
                        candidate,
                        evidence_items,
                        rule_eval,
                        topology,
                        heuristic_score,
                    )
                )
        return pod_candidates

    def _background_only(self, evidence_items: list) -> bool:
        records = [
            record for evidence in evidence_items for record in evidence.anomaly_records
        ]
        if not records:
            return False
        return all(
            record.source == "log" and bool(record.metadata.get("background_noise"))
            for record in records
        )

    def _max_candidate_supporting_evidence(self) -> int:
        return int(
            self.settings.get("llm_context", {}).get(
                "max_candidate_supporting_evidence", 4
            )
        )

    def _max_candidate_brief_notes(self) -> int:
        return int(
            self.settings.get("llm_context", {}).get("max_candidate_brief_notes", 3)
        )

    def _merge_rule_hints(self, *hint_maps: dict[str, bool]) -> dict[str, bool]:
        merged: dict[str, bool] = {}
        for hint_map in hint_maps:
            for key, value in (hint_map or {}).items():
                merged[key] = merged.get(key, False) or bool(value)
        return merged

    def _align_service_candidates(
        self,
        service: str,
        evidence_items: list,
        candidates: list[dict],
        pod_candidates: list[dict],
        service_rule_eval,
    ) -> list[dict]:
        aligned_candidates: list[dict] = []
        background_only = self._background_only(evidence_items)
        support_limit = self._max_candidate_supporting_evidence()
        suspicious_pods = list(
            dict.fromkeys(item["pod"] for item in pod_candidates if item.get("pod"))
        )[:3]
        strongest_pod = pod_candidates[0] if pod_candidates else None
        fault_types_present = set()

        for candidate in candidates:
            aligned = dict(candidate)
            notes = list(aligned.get("notes", []))
            support = list(aligned.get("supporting_evidence", []))
            score = float(aligned.get("provisional_score", 0.0))
            same_fault_pod = next(
                (
                    item
                    for item in pod_candidates
                    if item["fault_type"] == aligned["fault_type"]
                ),
                None,
            )

            if same_fault_pod:
                if self._bayesian_enabled():
                    self._apply_bayesian_prior_multiplier(
                        aligned,
                        "same_fault_pod_alignment",
                        1.20,
                        f"Service candidate has same-fault support from pod {same_fault_pod['pod']}.",
                    )
                    score = float(aligned.get("provisional_score", score))
                else:
                    score = min(
                        0.96,
                        score + 0.18 * float(same_fault_pod["provisional_score"]),
                    )
                notes.append(
                    f"Strongest pod-level signal matches this fault on {same_fault_pod['pod']}."
                )
                for evidence in same_fault_pod.get("supporting_evidence", [])[:2]:
                    if evidence not in support:
                        support.append(evidence)
                aligned["rule_hints"] = self._merge_rule_hints(
                    aligned.get("rule_hints", {}), same_fault_pod.get("rule_hints", {})
                )
                aligned["active_rules"] = list(
                    dict.fromkeys(
                        aligned.get("active_rules", [])
                        + same_fault_pod.get("active_rules", [])
                    )
                )
            elif (
                strongest_pod
                and float(strongest_pod["provisional_score"]) >= 0.8
                and aligned["fault_type"] != strongest_pod["fault_type"]
            ):
                if self._bayesian_enabled():
                    self._apply_bayesian_prior_multiplier(
                        aligned,
                        "conflicting_strong_pod",
                        0.75,
                        f"Stronger pod evidence points to {strongest_pod['fault_type']} on {strongest_pod['pod']}.",
                    )
                    score = float(aligned.get("provisional_score", score))
                else:
                    score = max(
                        0.08,
                        score - 0.16 * float(strongest_pod["provisional_score"]),
                    )
                notes.append(
                    f"Stronger localized pod evidence points to {strongest_pod['fault_type']} on {strongest_pod['pod']}."
                )

            if background_only:
                if self._bayesian_enabled():
                    self._apply_bayesian_prior_multiplier(
                        aligned,
                        "background_only_penalty",
                        0.70,
                        "Candidate evidence is mainly routine background-style logs.",
                    )
                    score = float(aligned.get("provisional_score", score))
                else:
                    score = max(0.08, score - 0.18)
                notes.append(
                    "This service is supported mainly by routine background-style logs."
                )

            if suspicious_pods:
                notes.append(f"suspect_pods={', '.join(suspicious_pods)}")

            aligned["provisional_score"] = round(min(0.98, score), 4)
            aligned["posterior_score"] = aligned["provisional_score"]
            aligned["supporting_evidence"] = support[:support_limit]
            aligned["notes"] = list(dict.fromkeys(notes + service_rule_eval.notes))
            aligned["background_only"] = background_only
            aligned_candidates.append(aligned)
            fault_types_present.add(aligned["fault_type"])

        if (
            strongest_pod
            and float(strongest_pod["provisional_score"]) >= 0.75
            and strongest_pod["fault_type"] not in fault_types_present
        ):
            combined_support = list(strongest_pod.get("supporting_evidence", [])[:3])
            for evidence in [
                record.summary
                for evidence in evidence_items
                for record in evidence.anomaly_records
            ][:2]:
                if evidence not in combined_support:
                    combined_support.append(evidence)
            synthetic_score = min(
                0.95,
                0.35 * max((float(item.score) for item in evidence_items), default=0.0)
                + 0.65 * float(strongest_pod["provisional_score"]),
            )
            synthetic_candidate = {
                "entity_type": "service",
                "service": service,
                "pod": None,
                "fault_type": strongest_pod["fault_type"],
                "fault_type_candidates": [strongest_pod["fault_type"]],
                "evidence_count": sum(
                    len(item.anomaly_records) for item in evidence_items
                ),
                "supporting_evidence": combined_support[:support_limit],
                "notes": list(
                    dict.fromkeys(
                        [
                            f"Derived from strongest pod-level anomaly on {strongest_pod['pod']}.",
                            (
                                f"suspect_pods={', '.join(suspicious_pods)}"
                                if suspicious_pods
                                else ""
                            ),
                        ]
                        + strongest_pod.get("notes", [])
                    )
                ),
                "dependency_boost": 0.0,
                "background_only": background_only,
                "rule_hints": self._merge_rule_hints(
                    asdict(service_rule_eval.rule_hints),
                    strongest_pod.get("rule_hints", {}),
                ),
                "active_rules": list(
                    dict.fromkeys(
                        service_rule_eval.active_rules
                        + strongest_pod.get("active_rules", [])
                    )
                ),
                "rule_score_adjustment": round(
                    service_rule_eval.score_adjustment
                    + float(strongest_pod.get("rule_score_adjustment", 0.0)),
                    4,
                ),
            }
            aligned_candidates.append(
                self._score_candidate(
                    synthetic_candidate,
                    evidence_items,
                    service_rule_eval,
                    None,
                    synthetic_score,
                )
            )

        return aligned_candidates

    def _build_candidates(self, state, rules_config: dict):
        topology = state.topology
        support_limit = self._max_candidate_supporting_evidence()
        rule_context = build_rule_context(state, topology, rules_config)
        pod_candidates = self._build_pod_candidates(state, rule_context, rules_config)
        self._normalize_posterior_probabilities(pod_candidates)
        pod_candidates_by_service = defaultdict(list)
        for item in sorted(
            pod_candidates,
            key=self._candidate_sort_score,
            reverse=True,
        ):
            pod_candidates_by_service[item["service"]].append(item)

        service_candidates: list[dict] = []
        seen_services = set()
        for service, evidence_items in aggregate_service_evidence(state).items():
            if not self._is_business_service(service, topology):
                continue
            seen_services.add(service)
            service_rule_eval = evaluate_rule_hints(
                {"entity_type": "service", "service": service, "pod": None},
                evidence_items,
                rule_context,
                topology,
                rules_config,
            )
            matches = match_fault_types(
                evidence_items,
                topology,
                rule_hints=asdict(service_rule_eval.rule_hints),
                rules_config=rules_config,
                support_limit=support_limit,
            )
            raw_candidates = []
            for fault_type, score, support, notes, dependency_boost in matches:
                heuristic_score = min(0.98, score + service_rule_eval.score_adjustment)
                candidate = {
                    "entity_type": "service",
                    "service": service,
                    "pod": None,
                    "fault_type": fault_type,
                    "fault_type_candidates": [fault_type],
                    "evidence_count": sum(
                        len(item.anomaly_records) for item in evidence_items
                    ),
                    "supporting_evidence": support[:support_limit],
                    "notes": list(
                        dict.fromkeys(list(notes) + service_rule_eval.notes)
                    ),
                    "dependency_boost": dependency_boost,
                    "rule_hints": asdict(service_rule_eval.rule_hints),
                    "active_rules": service_rule_eval.active_rules,
                    "rule_score_adjustment": service_rule_eval.score_adjustment,
                }
                raw_candidates.append(
                    self._score_candidate(
                        candidate,
                        evidence_items,
                        service_rule_eval,
                        topology,
                        heuristic_score,
                    )
                )
            service_candidates.extend(
                self._align_service_candidates(
                    service,
                    evidence_items,
                    raw_candidates,
                    pod_candidates_by_service.get(service, []),
                    service_rule_eval,
                )
            )

        for service, items in pod_candidates_by_service.items():
            if not self._is_business_service(service, topology):
                continue
            if service in seen_services or not items:
                continue
            strongest_pod = items[0]
            derived_score = min(0.93, 0.82 * float(strongest_pod["provisional_score"]))
            service_candidates.append(
                {
                    "entity_type": "service",
                    "service": service,
                    "pod": None,
                    "fault_type": strongest_pod["fault_type"],
                    "fault_type_candidates": [strongest_pod["fault_type"]],
                    "provisional_score": round(derived_score, 4),
                    "heuristic_score": round(derived_score, 4),
                    "posterior_score": round(derived_score, 4),
                    "bayes_log_score": strongest_pod.get("bayes_log_score", 0.0),
                    "prior_probability": strongest_pod.get("prior_probability", 0.0),
                    "prior_multiplier": strongest_pod.get("prior_multiplier", 1.0),
                    "likelihood_terms": strongest_pod.get("likelihood_terms", []),
                    "prior_terms": list(strongest_pod.get("prior_terms", []))
                    + [
                        {
                            "name": "derived_parent_from_pod",
                            "multiplier": 1.0,
                            "reason": f"Service candidate derived from strongest pod {strongest_pod['pod']}.",
                        }
                    ],
                    "evidence_count": int(strongest_pod.get("evidence_count", 0)),
                    "supporting_evidence": strongest_pod.get("supporting_evidence", [])[
                        :support_limit
                    ],
                    "notes": [
                        f"Service hypothesis derived from strongest pod-level anomaly on {strongest_pod['pod']}."
                    ],
                    "dependency_boost": 0.0,
                    "background_only": bool(strongest_pod.get("background_only")),
                    "rule_hints": strongest_pod.get("rule_hints", {}),
                    "active_rules": strongest_pod.get("active_rules", []),
                    "rule_score_adjustment": float(
                        strongest_pod.get("rule_score_adjustment", 0.0)
                    ),
                }
            )

        self._normalize_posterior_probabilities(service_candidates)
        self._normalize_posterior_probabilities(pod_candidates)
        return service_candidates, pod_candidates, rule_context

    def _evidence_text(self, record) -> str:
        if record.source == "metric":
            pod = record.metadata.get("pod")
            service = record.metadata.get("service") or record.entity_name
            scope = (
                f"pod {pod}"
                if record.entity_type == "pod" and pod
                else f"service {service}"
            )
            return f"{scope} {record.metric_or_pattern}: {record.summary}"
        return record.summary

    def _evidence_priority(self, record) -> tuple[int, int, int]:
        pattern = str(record.metric_or_pattern or "")
        text = f"{pattern} {record.summary}".lower()
        priority = EVIDENCE_PATTERN_PRIORITY.get(pattern, 0)
        if any(
            token in text
            for token in ["oom", "outofmemory", "crash", "killed", "readiness"]
        ):
            priority = max(priority, 94)
        if record.source == "log" and record.metadata.get("background_noise"):
            priority -= 40
        return (
            priority,
            EVIDENCE_SOURCE_PRIORITY.get(record.source, 0),
            len(str(record.summary)),
        )

    def _compact_evidence_record(self, record) -> dict:
        item = {
            "source": record.source,
            "metric_or_pattern": record.metric_or_pattern,
            "summary": self._evidence_text(record),
        }
        peer_service = record.metadata.get("peer_service")
        if peer_service:
            item["peer_service"] = peer_service
        return item

    def _dedupe_and_sort_records(self, records: list, max_items: int) -> list[dict]:
        unique = {}
        for record in records:
            key = (
                record.source,
                record.entity_type,
                record.entity_name,
                record.metric_or_pattern,
                self._evidence_text(record),
                record.metadata.get("peer_service"),
            )
            if key not in unique:
                unique[key] = record
        ordered = sorted(unique.values(), key=self._evidence_priority, reverse=True)
        return [self._compact_evidence_record(record) for record in ordered[:max_items]]

    def _all_anomaly_records(self, state) -> list:
        records = []
        for skill_result in [
            state.metrics_evidence,
            state.logs_evidence,
            state.traces_evidence,
        ]:
            if skill_result:
                records.extend(skill_result.anomaly_records)
                for evidence in skill_result.service_evidence:
                    records.extend(evidence.anomaly_records)
                for evidence in skill_result.pod_evidence:
                    records.extend(evidence.anomaly_records)
        return records

    def _build_evidence_tree(
        self, state, max_evidence_per_pod: int, max_service_level_evidence: int
    ) -> list[dict]:
        topology = state.topology or {}
        services = list(topology.get("services", []))
        tree: dict[str, dict] = {
            service: {"service": service, "service_level_evidence": [], "pods": {}}
            for service in services
        }

        for record in self._all_anomaly_records(state):
            service = record.metadata.get("service") or (
                record.entity_name if record.entity_type == "service" else None
            )
            pod = record.metadata.get("pod") or (
                record.entity_name if record.entity_type == "pod" else None
            )
            if not service and pod:
                service = service_from_pod(pod)
            if not service:
                continue
            if not self._is_business_service(service, topology):
                continue
            tree.setdefault(
                service, {"service": service, "service_level_evidence": [], "pods": {}}
            )
            if record.entity_type == "pod" or pod:
                pod_name = pod or record.entity_name
                tree[service]["pods"].setdefault(pod_name, [])
                tree[service]["pods"][pod_name].append(record)
            else:
                tree[service]["service_level_evidence"].append(record)

        result = []
        for service in sorted(tree):
            service_entry = tree[service]
            service_evidence = self._dedupe_and_sort_records(
                service_entry["service_level_evidence"], max_service_level_evidence
            )
            pod_entries = []
            for pod, records in sorted(service_entry["pods"].items()):
                evidence = self._dedupe_and_sort_records(records, max_evidence_per_pod)
                if evidence:
                    pod_entries.append({"pod": pod, "evidence": evidence})
            if service_evidence or pod_entries:
                result.append(
                    {
                        "service": service,
                        "service_level_evidence": service_evidence,
                        "pods": pod_entries,
                    }
                )
        return result

    def _build_reasoning_rule_guide(
        self, rules_config: dict, max_rules: int
    ) -> list[dict]:
        guide = []
        for group_name, group in rules_config.get("rule_groups", {}).items():
            if not group.get("enabled", True):
                continue
            rules = []
            for rule in group.get("rules", []):
                if rule.get("enabled", True) is False:
                    continue
                description = rule.get("description")
                if description:
                    rules.append({"id": rule.get("id"), "guidance": description})
                if len(rules) >= max_rules:
                    break
            if rules:
                guide.append({"category": group_name, "rules": rules})
        return guide

    def _record_service_and_pod(self, record) -> tuple[str | None, str | None]:
        service = record.metadata.get("service") or (
            record.entity_name if record.entity_type == "service" else None
        )
        pod = record.metadata.get("pod") or (
            record.entity_name if record.entity_type == "pod" else None
        )
        if not service and pod:
            service = service_from_pod(pod)
        return service, pod

    def _is_exception_like_record(self, record) -> bool:
        text = f"{record.metric_or_pattern} {record.summary}".lower()
        return record.metric_or_pattern in EXCEPTION_HINT_PATTERNS or any(
            token in text for token in EXCEPTION_HINT_TOKENS
        )

    def _is_propagated_symptom_record(self, record) -> bool:
        text = f"{record.metric_or_pattern} {record.summary}".lower()
        return (
            record.metric_or_pattern in PROPAGATED_SYMPTOM_PATTERNS
            or "latency" in text
            or "timeout" in text
            or "status code" in text
        )

    def _is_failure_like_record(self, record) -> bool:
        text = f"{record.metric_or_pattern} {record.summary}".lower()
        return record.metric_or_pattern in {
            "restart_count",
            "ready_ratio",
            "pod_failure",
            "exception_injection",
        } or any(
            token in text
            for token in {
                "restart",
                "readiness",
                "unready",
                "oom",
                "out of memory",
                "outofmemory",
                "crash",
                "panic",
                "killed",
                "exception",
            }
        )

    def _build_propagation_hints(
        self, state, rule_context: dict, max_hints: int = 20
    ) -> list[dict]:
        records_by_service_pod: dict[str, dict[str, list]] = defaultdict(
            lambda: defaultdict(list)
        )
        for record in self._all_anomaly_records(state):
            service, pod = self._record_service_and_pod(record)
            if service and pod:
                records_by_service_pod[service][pod].append(record)

        downstream_map = rule_context.get("downstream_map", {})
        upstream_map = rule_context.get("upstream_map", {})
        hints: list[dict] = []

        symptom_services = {}
        for service, pods in records_by_service_pod.items():
            symptom_pods = [
                pod
                for pod, records in pods.items()
                if any(self._is_propagated_symptom_record(record) for record in records)
            ]
            if len(symptom_pods) >= 2:
                symptom_services[service] = sorted(symptom_pods)
                hints.append(
                    {
                        "type": "multi_pod_similar_symptoms",
                        "service": service,
                        "affected_pods": sorted(symptom_pods)[:5],
                        "interpretation": (
                            "Similar latency/error symptoms across sibling pods often indicate a shared dependency, "
                            "propagated failure, traffic skew, or service-level impact rather than multiple independent pod roots."
                        ),
                    }
                )
                if len(hints) >= max_hints:
                    return hints[:max_hints]

        failure_services = {}
        for service, pods in records_by_service_pod.items():
            failure_pods = [
                pod
                for pod, records in pods.items()
                if any(self._is_failure_like_record(record) for record in records)
            ]
            if len(failure_pods) >= 2:
                failure_services[service] = sorted(failure_pods)

        for upstream_service, affected_pods in failure_services.items():
            for downstream_service in sorted(downstream_map.get(upstream_service, set())):
                downstream_pods = records_by_service_pod.get(downstream_service, {})
                direct_failure_pods = [
                    pod
                    for pod, records in downstream_pods.items()
                    if any(self._is_failure_like_record(record) for record in records)
                ]
                if not direct_failure_pods:
                    continue
                hints.append(
                    {
                        "type": "downstream_local_failure_vs_upstream_multi_pod_failures",
                        "upstream_service": upstream_service,
                        "upstream_affected_pods": affected_pods[:5],
                        "downstream_service": downstream_service,
                        "downstream_direct_failure_pods": sorted(direct_failure_pods)[:5],
                        "interpretation": (
                            "When multiple sibling pods in an upstream service show similar restart, readiness, crash, "
                            "or exception signals while a downstream dependency has direct local failure evidence, "
                            "the upstream service is often an impacted propagation carrier and the downstream dependency "
                            "is the stronger root-cause candidate."
                        ),
                    }
                )
                if len(hints) >= max_hints:
                    return hints[:max_hints]

        for service, pods in records_by_service_pod.items():
            exception_pods = [
                pod
                for pod, records in pods.items()
                if any(self._is_exception_like_record(record) for record in records)
            ]
            if len(exception_pods) != 1:
                continue
            candidate_pod = exception_pods[0]
            adjacent_services = sorted(
                set(downstream_map.get(service, set()))
                | set(upstream_map.get(service, set()))
            )
            for adjacent_service in adjacent_services:
                affected_pods = symptom_services.get(adjacent_service)
                if not affected_pods:
                    continue
                hints.append(
                    {
                        "type": "single_pod_exception_vs_adjacent_multi_pod_symptoms",
                        "candidate_service": service,
                        "candidate_pod": candidate_pod,
                        "related_service": adjacent_service,
                        "related_affected_pods": affected_pods[:5],
                        "interpretation": (
                            "A localized exception signal in one adjacent pod may be a root cause that explains "
                            "similar latency/error symptoms across multiple sibling pods in the related service. "
                            "Treat the multi-pod symptom service as potentially propagated unless it has stronger local root-cause evidence."
                        ),
                    }
                )
                if len(hints) >= max_hints:
                    return hints[:max_hints]

        return hints[:max_hints]

    def _candidate_for_llm(self, candidate: dict) -> dict:
        notes = candidate.get("notes", [])
        if isinstance(notes, str):
            notes = [notes]
        likelihood_terms = [
            {
                "evidence": item.get("evidence"),
                "severity": item.get("severity"),
                "likelihood": item.get("p_e_given_fault"),
                "source": item.get("source"),
                "raw_pattern": item.get("raw_pattern"),
                "baseline_value": item.get("baseline_value"),
                "abnormal_value": item.get("abnormal_value"),
                "delta_ratio": item.get("delta_ratio"),
                "ratio": item.get("ratio"),
                "relative_change_pct": item.get("relative_change_pct"),
            }
            for item in candidate.get("likelihood_terms", [])[:3]
        ]
        prior_terms = [
            {
                "name": item.get("name"),
                "multiplier": item.get("multiplier"),
            }
            for item in candidate.get("prior_terms", [])[:3]
        ]
        return {
            "entity_type": candidate.get("entity_type"),
            "service": candidate.get("service"),
            "pod": candidate.get("pod"),
            "fault_type": candidate.get("fault_type"),
            "provisional_score": candidate.get("provisional_score"),
            "posterior_probability": candidate.get("posterior_probability"),
            "evidence_count": candidate.get("evidence_count"),
            "rule_hints": candidate.get("rule_hints", {}),
            "active_rules": candidate.get("active_rules", [])[:5],
            "supporting_evidence": candidate.get("supporting_evidence", [])[
                : self._max_candidate_supporting_evidence()
            ],
            "bayesian_evidence_hints": likelihood_terms,
            "bayesian_prior_hints": prior_terms,
            "brief_notes": list(notes or [])[: self._max_candidate_brief_notes()],
        }

    def _candidate_for_reconciliation(
        self, candidate: dict, rank: int, ranking_scope: str
    ) -> dict:
        compact = self._candidate_for_llm(candidate)
        compact["current_rank"] = rank
        compact["ranking_scope"] = ranking_scope
        return compact

    def _llm_context_for_stage(
        self, llm_context: dict, include_evidence_tree: bool
    ) -> dict:
        context = dict(llm_context)
        if not include_evidence_tree:
            context.pop("evidence_tree", None)
            context["evidence_tree_omitted"] = True
        return context

    def _is_strong_local_root_candidate(self, candidate: dict) -> bool:
        if candidate.get("entity_type") != "pod" or not candidate.get("pod"):
            return False
        try:
            score = float(candidate.get("provisional_score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        min_score = float(
            self.settings.get("llm_context", {}).get(
                "min_parent_alignment_pod_score", 0.85
            )
        )
        if score < min_score:
            return False

        rule_hints = candidate.get("rule_hints", {}) or {}
        if rule_hints.get("local_resource_support"):
            return True
        if candidate.get("fault_type") in LOCAL_ROOT_FAULT_TYPES:
            support_text = " | ".join(
                candidate.get("supporting_evidence", []) + candidate.get("notes", [])
            ).lower()
            return any(token in support_text for token in LOCAL_ROOT_EVIDENCE_TOKENS)
        return False

    def _build_cross_level_alignment_hints(
        self, ranked_services: list[dict], ranked_pods: list[dict]
    ) -> list[dict]:
        hints = []
        for pod_candidate in ranked_pods[:10]:
            if not self._is_strong_local_root_candidate(pod_candidate):
                continue
            service_name = pod_candidate.get("service")
            service_candidate = next(
                (
                    item
                    for item in ranked_services
                    if item.get("service") == service_name
                    and item.get("fault_type") == pod_candidate.get("fault_type")
                ),
                None,
            )
            if not service_candidate:
                service_candidate = next(
                    (
                        item
                        for item in ranked_services
                        if item.get("service") == service_name
                    ),
                    None,
                )
            if not service_candidate:
                continue
            hints.append(
                {
                    "type": "top_local_pod_parent_alignment",
                    "service": service_name,
                    "pod": pod_candidate.get("pod"),
                    "pod_fault_type": pod_candidate.get("fault_type"),
                    "pod_score": pod_candidate.get("provisional_score"),
                    "service_fault_type": service_candidate.get("fault_type"),
                    "service_score": service_candidate.get("provisional_score"),
                    "interpretation": (
                        "If this pod remains the strongest localized root-cause pod, its parent service should also "
                        "rank as a plausible top service hypothesis. Multi-pod latency/error-only services should not "
                        "stay above that parent service unless they have stronger local root-cause evidence."
                    ),
                }
            )
            if len(hints) >= 5:
                break
        return hints

    def _align_parent_service_with_top_pod(
        self, ranked_services: list[dict], ranked_pods: list[dict]
    ) -> list[dict]:
        if not self.settings.get("llm_context", {}).get(
            "align_parent_service_with_top_pod", True
        ):
            return ranked_services
        local_entries = [
            (
                self._score_output_candidate(item),
                self._specificity_priority(item),
                -index,
                index,
                item,
            )
            for index, item in enumerate(ranked_pods)
            if self._is_strong_local_root_candidate(item)
        ]
        if not local_entries:
            return ranked_services
        _, _, _, _, top_local_pod = max(local_entries)

        service_name = top_local_pod.get("service")
        if not service_name:
            return ranked_services
        service_index = next(
            (
                index
                for index, item in enumerate(ranked_services)
                if item.get("service") == service_name
                and item.get("fault_type") == top_local_pod.get("fault_type")
            ),
            None,
        )
        if service_index is None:
            service_index = next(
                (
                    index
                    for index, item in enumerate(ranked_services)
                    if item.get("service") == service_name
                ),
                None,
            )
        if service_index is None:
            return ranked_services

        aligned_services = [dict(item) for item in ranked_services]
        service_candidate = dict(aligned_services[service_index])
        try:
            pod_score = float(top_local_pod.get("provisional_score", 0.0))
            service_score = float(service_candidate.get("provisional_score", 0.0))
        except (TypeError, ValueError):
            return ranked_services

        min_gap = float(
            self.settings.get("llm_context", {}).get(
                "parent_alignment_min_score_gap", 0.08
            )
        )
        top_service_score = 0.0
        if ranked_services:
            try:
                top_service_score = float(
                    ranked_services[0].get("provisional_score", 0.0)
                )
            except (TypeError, ValueError):
                top_service_score = 0.0
        max_symptom_lead = float(
            self.settings.get("llm_context", {}).get(
                "parent_alignment_max_symptom_lead", 0.08
            )
        )
        top_pod_parent_should_lead = (
            service_index != 0 and top_service_score - pod_score <= max_symptom_lead
        )
        if service_score >= pod_score - min_gap and not top_pod_parent_should_lead:
            return ranked_services

        promoted_score = min(0.98, pod_score)
        service_candidate["provisional_score"] = round(
            max(service_score, promoted_score), 4
        )
        notes = service_candidate.get("notes", [])
        if isinstance(notes, str):
            notes = [notes]
        notes = list(notes or [])
        notes.append(
            f"Cross-level alignment: parent service promoted because {top_local_pod.get('pod')} is the strongest localized root-cause pod."
        )
        service_candidate["notes"] = list(dict.fromkeys(note for note in notes if note))

        support_limit = self._max_candidate_supporting_evidence()
        support = list(service_candidate.get("supporting_evidence", []) or [])
        for evidence in top_local_pod.get("supporting_evidence", [])[:support_limit]:
            if evidence not in support:
                support.insert(0, evidence)
        service_candidate["supporting_evidence"] = support[:support_limit]
        service_candidate["rule_hints"] = self._merge_rule_hints(
            service_candidate.get("rule_hints", {}),
            top_local_pod.get("rule_hints", {}),
        )
        service_candidate["active_rules"] = list(
            dict.fromkeys(
                service_candidate.get("active_rules", [])
                + top_local_pod.get("active_rules", [])
                + ["cross_level_parent_alignment"]
            )
        )

        aligned_services[service_index] = service_candidate
        return aligned_services

    def _reconcile_rankings_with_llm(
        self,
        llm_context: dict,
        ranked_services: list[dict],
        ranked_pods: list[dict],
    ):
        combined_candidates = list(ranked_services) + list(ranked_pods)
        if not combined_candidates:
            return ranked_services, ranked_pods, None

        reconciliation_context = {
            **self._llm_context_for_stage(
                llm_context,
                bool(
                    self.settings.get("llm_context", {}).get(
                        "include_evidence_tree_in_reconciliation", True
                    )
                ),
            ),
            "ranking_scope": "cross_level_reconciliation",
            "service_ranking_preview": [
                self._candidate_for_reconciliation(item, index + 1, "service")
                for index, item in enumerate(ranked_services[:10])
            ],
            "pod_ranking_preview": [
                self._candidate_for_reconciliation(item, index + 1, "pod")
                for index, item in enumerate(ranked_pods[:10])
            ],
            "alignment_hints": self._build_cross_level_alignment_hints(
                ranked_services, ranked_pods
            ),
            "reconciliation_guidance": [
                "Reconcile service and pod rankings together using the evidence tree.",
                "High-ranking pod hypotheses should usually make their parent service plausible, unless evidence shows the pod is incidental.",
                "High-ranking service hypotheses should usually be explainable by one or more pods or explicit service-level evidence.",
                "If a strong localized pod root cause remains top-ranked, promote or keep its parent service near the top service results.",
                "Down-rank multi-pod latency/error-only services when they look like propagated symptoms and lack local resource, restart, readiness, or exception evidence.",
                "Use propagation_hints to distinguish localized root-cause evidence from adjacent multi-pod propagated symptoms.",
                "When one adjacent pod has explicit exception evidence and another service has similar symptoms across sibling pods, consider the localized exception source as a stronger root-cause story if topology supports it.",
                "Do not force service/pod agreement when the evidence clearly supports different scopes.",
                "Prefer explanations where the top service and top pod form a coherent root-cause story.",
            ],
        }
        llm_candidates = [
            self._candidate_for_reconciliation(item, index + 1, "service")
            for index, item in enumerate(ranked_services)
        ] + [
            self._candidate_for_reconciliation(item, index + 1, "pod")
            for index, item in enumerate(ranked_pods)
        ]
        response = rank_with_llm(
            self.llm_client,
            "cross_level_ranking_reconciliation",
            reconciliation_context,
            combined_candidates,
            llm_candidates=llm_candidates,
        )
        reconciled_services = [
            item for item in response.rankings if item.get("entity_type") == "service"
        ]
        reconciled_pods = [
            item for item in response.rankings if item.get("entity_type") == "pod"
        ]
        reconciled_services = self._align_parent_service_with_top_pod(
            reconciled_services, reconciled_pods
        )
        return reconciled_services, reconciled_pods, response

    def _build_llm_context(
        self,
        request,
        state,
        rule_context: dict,
        service_candidates: list[dict],
        pod_candidates: list[dict],
    ) -> dict:
        llm_settings = self.settings.get("llm_context", {})
        max_evidence_per_pod = int(
            llm_settings.get(
                "max_evidence_per_pod", llm_settings.get("max_evidence_per_entity", 6)
            )
        )
        max_service_level_evidence = int(
            llm_settings.get("max_service_level_evidence", 6)
        )
        max_rules_per_group = int(llm_settings.get("max_rules_per_group", 4))
        max_topology_edges = int(llm_settings.get("max_topology_edges", 200))
        topology = state.topology or {}
        config_bundle = getattr(request, "config_bundle", {}) or {}
        return {
            "abnormal_window": {
                "start": request.abnormal_window.start,
                "end": request.abnormal_window.end,
            },
            "baseline_window": {
                "start": request.baseline_window.start,
                "end": request.baseline_window.end,
            },
            "namespace": request.namespace,
            "candidate_counts": {
                "service": len(service_candidates),
                "pod": len(pod_candidates),
            },
            "scoring_model": {
                "name": "lightweight_bayesian_diagnostic_network",
                "enabled": self._bayesian_enabled(),
                "fallback_when_disabled": "heuristic_score",
            },
            "topology": {
                "services": topology.get("services", []),
                "edges": topology.get("edges", [])[:max_topology_edges],
            },
            "rule_context": {
                "anomalous_services": rule_context.get("anomalous_services", []),
                "shared_downstream_targets": rule_context.get(
                    "shared_downstream_targets", []
                ),
                "path_services": rule_context.get("path_services", []),
                "trace_targets_by_service": rule_context.get(
                    "trace_targets_by_service", {}
                ),
            },
            "evidence_tree": self._build_evidence_tree(
                state, max_evidence_per_pod, max_service_level_evidence
            ),
            "propagation_hints": self._build_propagation_hints(
                state,
                rule_context,
                int(llm_settings.get("max_propagation_hints", 20)),
            ),
            "reasoning_rule_guide": self._build_reasoning_rule_guide(
                config_bundle.get("rootcause_reasoning_rules", {}),
                max_rules_per_group,
            ),
            "ranking_guidance": [
                "Evaluate candidates independently against evidence, topology, and propagation patterns; do not rank by any score alone.",
                "posterior_probability and provisional_score are produced by a Lightweight Bayesian Diagnostic Network and should be treated as diagnostic hints, not calibrated truth.",
                "You may ignore Bayesian scores when they conflict with direct local evidence, dependency topology, temporal ordering, or propagated-symptom reasoning.",
                "If you reorder candidates because the supplied scores are misleading, also revise provisional_score so the final scores match your ranking confidence.",
                "Prefer direct, specific local root-cause evidence over generic weak metric movement such as network_rx/network_tx alone.",
                "Prefer concrete local evidence such as restart_count, ready_ratio, OOM/crash logs, CPU, or memory over generic propagated symptoms.",
                "Treat low absolute-count log spikes cautiously even when their ratio is high; count 1->5 or 2->6 is weaker than sustained high-volume error/exception evidence.",
                "When one modality shows low-volume generic symptoms but another modality shows strong, fault-specific anomalies, prefer the stronger multi-modal explanation unless topology or timing clearly contradicts it.",
                "Only override a strongly favored Bayesian candidate when the competing candidate has direct, specific, and sufficiently high-volume evidence rather than weak or incidental symptoms.",
                "When candidates have similar Bayesian posterior and equally local evidence, compare the structured metric effect size: larger fault-specific relative change is stronger evidence.",
                "Do not promote a database or dependency service solely because it is topologically plausible; require explicit topology/rule support or stronger direct evidence.",
                "Use topology to decide whether a service is likely the origin or a symptom carrier.",
                "When multiple sibling pods in an upstream service have similar restart/readiness symptoms and a downstream dependency has direct local restart/readiness/OOM/crash evidence, treat the upstream pods as likely propagated failure carriers unless stronger localized upstream evidence contradicts it.",
                "Use propagation_hints as investigation hints for cascade patterns; they are not deterministic rules.",
                "If propagation_hints show downstream_local_failure_vs_upstream_multi_pod_failures, prefer the downstream dependency over the upstream multi-pod failure carrier unless the upstream service has stronger direct local evidence.",
                "If an adjacent service has a single pod with explicit exception evidence while another service has repeated similar latency/error evidence across sibling pods, prefer the localized exception source unless stronger local evidence contradicts it.",
                "Use the reasoning_rule_guide as practical RCA guidance, not as deterministic truth.",
            ],
        }

    def _score_output_candidate(self, item: dict) -> float:
        score = float(item.get("provisional_score", 0.0))
        evidence_bonus = min(
            0.04, 0.01 * max(int(item.get("evidence_count", 0)) - 1, 0)
        )
        rule_hints = item.get("rule_hints", {}) or {}
        if item.get("background_only"):
            score -= 0.06
        if rule_hints.get("topology_conflict"):
            score -= 0.03
        if rule_hints.get("local_resource_support"):
            score += 0.02
        if rule_hints.get("shared_issue_hint") and rule_hints.get(
            "symptom_only_signal"
        ):
            score -= float(
                self.settings.get("llm_context", {}).get(
                    "shared_symptom_candidate_penalty", 0.06
                )
            )
        if rule_hints.get("single_pod_local_hint") and rule_hints.get(
            "shared_downstream_dependency_hint"
        ):
            score += float(
                self.settings.get("llm_context", {}).get(
                    "single_pod_downstream_candidate_bonus", 0.06
                )
            )
        if self._bayesian_enabled():
            posterior_score = self._safe_float(item.get("posterior_score"), 0.0)
            posterior_floor_weight = float(
                self.settings.get("llm_context", {}).get(
                    "final_posterior_score_floor_weight", 1.0
                )
            )
            score = max(score, posterior_score * posterior_floor_weight)
            posterior_probability = self._safe_float(
                item.get("posterior_probability"), 1.0
            )
            guard_threshold = float(
                self.settings.get("llm_context", {}).get(
                    "bayesian_override_guard_min_probability", 0.01
                )
            )
            if (
                posterior_probability < guard_threshold
                and score > 0.8
                and not self._has_direct_override_evidence(item)
            ):
                score = min(
                    score,
                    float(
                        self.settings.get("llm_context", {}).get(
                            "bayesian_override_guard_score_cap", 0.72
                        )
                    ),
                )
        score += evidence_bonus
        if self._is_shared_symptom_carrier(item):
            score = min(
                score,
                float(
                    self.settings.get("llm_context", {}).get(
                        "shared_symptom_carrier_score_cap", 0.82
                    )
                ),
            )
        return round(max(0.05, min(0.98, score)), 4)

    def _is_shared_symptom_carrier(self, item: dict) -> bool:
        rule_hints = item.get("rule_hints", {}) or {}
        if not (
            rule_hints.get("shared_issue_hint")
            and rule_hints.get("symptom_only_signal")
            and rule_hints.get("dependency_symptom_hint")
        ):
            return False
        if rule_hints.get("local_resource_support") or rule_hints.get(
            "shared_downstream_dependency_hint"
        ):
            return False
        return True

    def _safe_float(self, value, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _has_direct_override_evidence(self, item: dict) -> bool:
        support_text = " | ".join(
            item.get("supporting_evidence", []) + item.get("notes", [])
        ).lower()
        direct_tokens = {
            "restart_count",
            "ready_ratio",
            "oom",
            "outofmemory",
            "crash",
            "killed",
            "cpu_usage_pct",
            "memory_usage_pct",
            "edge_failure_spike",
            "missing_data_gap",
        }
        if any(token in support_text for token in direct_tokens):
            return True
        for hint in item.get("bayesian_evidence_hints", []) or []:
            source = hint.get("source")
            abnormal_value = self._safe_float(hint.get("abnormal_value"), 0.0)
            if source == "log" and abnormal_value >= float(
                self.settings.get("llm_context", {}).get(
                    "high_volume_log_override_count", 20
                )
            ):
                return True
            if source in {"metric", "trace"} and self._safe_float(
                hint.get("severity"), 0.0
            ) >= 0.95:
                raw_pattern = str(hint.get("raw_pattern") or "")
                if raw_pattern in {
                    "restart_count",
                    "ready_ratio",
                    "cpu_usage_pct",
                    "memory_usage_pct",
                    "edge_failure_spike",
                    "missing_data_gap",
                }:
                    return True
        return False

    def _specificity_priority(self, item: dict) -> int:
        support_text = " | ".join(
            item.get("supporting_evidence", []) + item.get("notes", [])
        ).lower()
        fault_type = item.get("fault_type")
        if fault_type == "pod_failure" and (
            "restart_count" in support_text or "ready_ratio" in support_text
        ):
            return 40
        if fault_type == "exception_injection" and any(
            token in support_text
            for token in {
                "runtimeexception",
                "throwexception",
                "rule.execute",
                "byteman",
                "java.lang",
                "caught throw",
            }
        ):
            return 38
        if fault_type in {"cpu_stress", "memory_stress"} and (
            "cpu_usage_pct" in support_text or "memory_usage_pct" in support_text
        ):
            return 30
        if fault_type == "network_partition" and "missing_data_gap" in support_text:
            return 35
        if fault_type in {"io_fault", "network_loss", "network_partition"} and (
            "error_count" in support_text or "edge_failure_spike" in support_text
        ):
            return 20
        if fault_type == "network_delay" and (
            "latency_" in support_text or "edge_latency_spike" in support_text
        ):
            return 10
        return 0

    def _finalize(self, ranked_candidates, entity_type: str):
        output = []
        scored_candidates = [
            (
                self._score_output_candidate(item),
                self._specificity_priority(item),
                original_rank,
                item,
            )
            for original_rank, item in enumerate(ranked_candidates)
        ]
        scored_candidates.sort(
            key=lambda entry: (entry[0], entry[1], -entry[2]), reverse=True
        )
        for score, _, _, item in scored_candidates[
            : int(self.settings.get("defaults", {}).get("top_k", DEFAULT_TOP_K))
        ]:
            notes = item.get("notes", []) + run_light_checks(
                item, self.data_access.get_topology()
            )
            active_rules = item.get("active_rules", [])
            if active_rules:
                notes.append(f"active_rules={', '.join(active_rules[:4])}")
            output.append(
                RankedHypothesis(
                    service=item.get("service"),
                    pod=item.get("pod"),
                    fault_type=item["fault_type"],
                    score=score,
                    supporting_evidence=item.get("supporting_evidence", [])[
                        : self._max_candidate_supporting_evidence()
                    ],
                    notes="; ".join([note for note in dict.fromkeys(notes) if note]),
                )
            )
        return output

    def _align_final_service_with_top_pod(
        self, service_top5: list[RankedHypothesis], pod_top5: list[RankedHypothesis]
    ) -> list[RankedHypothesis]:
        llm_settings = self.settings.get("llm_context", {})
        if not llm_settings.get("align_final_service_with_top_pod", True):
            return service_top5
        if not service_top5 or not pod_top5 or not pod_top5[0].service:
            return service_top5

        top_service = service_top5[0]
        top_pod = pod_top5[0]
        if (
            top_service.service == top_pod.service
            and top_service.fault_type == top_pod.fault_type
        ):
            return service_top5

        exact_match_index = next(
            (
                index
                for index, item in enumerate(service_top5)
                if item.service == top_pod.service
                and item.fault_type == top_pod.fault_type
            ),
            None,
        )
        if exact_match_index is None:
            return service_top5

        try:
            score_gap = float(top_service.score) - float(top_pod.score)
        except (TypeError, ValueError):
            return service_top5

        max_gap = float(llm_settings.get("final_service_pod_alignment_max_gap", 0.03))
        if top_service.service == top_pod.service:
            try:
                exact_match_score = float(service_top5[exact_match_index].score)
            except (TypeError, ValueError):
                return service_top5
            same_service_fault_gap = float(
                llm_settings.get("final_same_service_fault_alignment_max_gap", 0.08)
            )
            if float(top_service.score) - exact_match_score > same_service_fault_gap:
                return service_top5
        elif score_gap > max_gap:
            return service_top5

        aligned = list(service_top5)
        matched = aligned.pop(exact_match_index)
        note_suffix = (
            f"Final cross-level alignment: promoted {top_pod.service}/{top_pod.fault_type} because "
            f"top pod {top_pod.pod} has the same fault type."
        )
        notes = matched.notes
        if note_suffix not in notes:
            notes = f"{notes}; {note_suffix}" if notes else note_suffix
        promoted = replace(
            matched,
            score=round(max(float(matched.score), float(top_service.score)), 4),
            notes=notes,
        )
        return [promoted] + aligned

    def run(self, request, state):
        rules_config = request.config_bundle.get("rootcause_reasoning_rules", {})
        service_candidates, pod_candidates, rule_context = self._build_candidates(
            state, rules_config
        )
        if self.debug.get("print_skill_inputs", True):
            log_json(
                self.logger,
                "[REASONING][INPUT] ",
                {
                    "service_candidate_count": len(service_candidates),
                    "pod_candidate_count": len(pod_candidates),
                    "service_candidates_preview": service_candidates[:10],
                    "pod_candidates_preview": pod_candidates[:10],
                    "rule_context_preview": {
                        "shared_downstream_targets": rule_context.get(
                            "shared_downstream_targets", []
                        )[:10],
                        "path_services": rule_context.get("path_services", [])[:10],
                    },
                },
            )
        llm_context = self._build_llm_context(
            request, state, rule_context, service_candidates, pod_candidates
        )
        initial_llm_context = self._llm_context_for_stage(
            llm_context,
            bool(
                self.settings.get("llm_context", {}).get(
                    "include_evidence_tree_in_initial_ranking", False
                )
            ),
        )
        service_rank_response = rank_with_llm(
            self.llm_client,
            "service_fault_ranking",
            {**initial_llm_context, "ranking_scope": "service"},
            service_candidates,
            llm_candidates=[
                self._candidate_for_llm(item) for item in service_candidates
            ],
        )
        pod_rank_response = rank_with_llm(
            self.llm_client,
            "pod_fault_ranking",
            {**initial_llm_context, "ranking_scope": "pod"},
            pod_candidates,
            llm_candidates=[self._candidate_for_llm(item) for item in pod_candidates],
        )
        ranked_services = service_rank_response.rankings
        ranked_pods = pod_rank_response.rankings
        reconciliation_response = None
        if self.settings.get("llm_context", {}).get(
            "enable_cross_level_reconciliation", True
        ):
            ranked_services, ranked_pods, reconciliation_response = (
                self._reconcile_rankings_with_llm(
                    llm_context,
                    ranked_services,
                    ranked_pods,
                )
            )
        if self.debug.get("print_llm_io", True):
            log_json(
                self.logger, "[REASONING][LLM_RANKED_SERVICES] ", ranked_services[:10]
            )
            log_json(self.logger, "[REASONING][LLM_RANKED_PODS] ", ranked_pods[:10])
            if reconciliation_response:
                log_json(
                    self.logger,
                    "[REASONING][LLM_RECONCILIATION_NOTES] ",
                    reconciliation_response.notes,
                )
        service_top5 = self._finalize(ranked_services, "service")
        pod_top5 = self._finalize(ranked_pods, "pod")
        service_top5 = self._align_final_service_with_top_pod(service_top5, pod_top5)
        if service_top5:
            top_service = service_top5[0]
            matching_pod = next(
                (item for item in pod_top5 if item.service == top_service.service), None
            )
            if matching_pod:
                summary = (
                    f"Most likely root cause is service {top_service.service} with {top_service.fault_type}, "
                    f"most suspicious pod {matching_pod.pod}."
                )
            else:
                summary = f"Most likely root cause is {top_service.service} with {top_service.fault_type}."
        elif pod_top5:
            summary = f"Most likely root cause is pod {pod_top5[0].pod} with {pod_top5[0].fault_type}."
        else:
            summary = "No strong root-cause hypothesis was found."
        warnings = []
        llm_metadata = {
            "service_ranking_notes": service_rank_response.notes,
            "pod_ranking_notes": pod_rank_response.notes,
            "cross_level_reconciliation_notes": (
                reconciliation_response.notes if reconciliation_response else []
            ),
        }
        all_llm_notes = service_rank_response.notes + pod_rank_response.notes
        if reconciliation_response:
            all_llm_notes += reconciliation_response.notes
        if any("effective_provider=heuristic" in note for note in all_llm_notes):
            warnings.append(
                "Reasoning used heuristic fallback for at least one ranking pass; check LLM configuration and logs."
            )
        result = RCAResponse(
            incident_id=request.incident_id,
            abnormal_window=request.abnormal_window,
            baseline_window=request.baseline_window,
            service_top5=service_top5,
            pod_top5=pod_top5,
            final_summary=summary,
            warnings=warnings,
            errors=[],
            metadata={
                "execution_options": getattr(request, "execution_options", {}) or {},
                "scoring": {
                    "bayesian_enabled": self._bayesian_enabled(),
                    "candidate_score_field": (
                        "posterior_score"
                        if self._bayesian_enabled()
                        else "heuristic_score"
                    ),
                    "candidate_probability_field": "posterior_probability",
                },
                "service_candidates": ranked_services[:10],
                "pod_candidates": ranked_pods[:10],
                "cross_level_reconciliation": {
                    "enabled": bool(reconciliation_response),
                    "notes": (
                        reconciliation_response.notes if reconciliation_response else []
                    ),
                    "ranked_candidates": (
                        reconciliation_response.rankings[:10]
                        if reconciliation_response
                        else []
                    ),
                },
                "llm_context_summary": {
                    "candidate_counts": llm_context["candidate_counts"],
                    "topology_service_count": len(llm_context["topology"]["services"]),
                    "topology_edge_count": len(llm_context["topology"]["edges"]),
                    "evidence_tree_services": len(llm_context["evidence_tree"]),
                    "evidence_tree_pods": sum(
                        len(service_entry.get("pods", []))
                        for service_entry in llm_context["evidence_tree"]
                    ),
                    "max_evidence_per_pod": int(
                        self.settings.get("llm_context", {}).get(
                            "max_evidence_per_pod",
                            self.settings.get("llm_context", {}).get(
                                "max_evidence_per_entity", 6
                            ),
                        )
                    ),
                    "max_service_level_evidence": int(
                        self.settings.get("llm_context", {}).get(
                            "max_service_level_evidence", 6
                        )
                    ),
                    "reasoning_rule_categories": len(
                        llm_context["reasoning_rule_guide"]
                    ),
                },
                "llm": llm_metadata,
                "reasoning_rules": {
                    "version": rules_config.get("version", "v1"),
                    "enabled_rule_groups": [
                        group_name
                        for group_name, group_cfg in rules_config.get(
                            "rule_groups", {}
                        ).items()
                        if group_cfg.get("enabled", True)
                    ],
                    "rule_context_summary": {
                        "shared_downstream_targets": rule_context.get(
                            "shared_downstream_targets", []
                        )[:10],
                        "path_services": rule_context.get("path_services", [])[:10],
                    },
                },
            },
        )
        if self.debug.get("print_skill_outputs", True):
            log_json(
                self.logger,
                "[REASONING][OUTPUT] ",
                {
                    "service_top5": [
                        {
                            "service": item.service,
                            "fault_type": item.fault_type,
                            "score": item.score,
                            "notes": item.notes,
                        }
                        for item in service_top5
                    ],
                    "pod_top5": [
                        {
                            "pod": item.pod,
                            "service": item.service,
                            "fault_type": item.fault_type,
                            "score": item.score,
                            "notes": item.notes,
                        }
                        for item in pod_top5
                    ],
                    "final_summary": summary,
                },
            )
        return result
