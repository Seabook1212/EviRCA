from __future__ import annotations

import json
import os
from typing import Any

import requests

from rca_agent_skills.common.logging_utils import get_logger, log_json

from .schemas import LLMRankRequest, LLMRankResponse


class LLMClient:
    """
    V1 LLM client.

    - `heuristic` provider keeps the framework runnable offline and in tests.
    - `openai` provider calls the OpenAI Responses API when `OPENAI_API_KEY` is set.
    """

    def __init__(
        self,
        provider: str = "heuristic",
        model: str = "heuristic-v1",
        temperature: float = 0.0,
        debug: dict[str, Any] | None = None,
    ):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.debug = debug or {}
        self.logger = get_logger(self.__class__.__name__)
        self.openai_api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        self.openai_base_url = os.environ.get(
            "OPENAI_BASE_URL", "https://api.openai.com/v1"
        ).rstrip("/")
        if self.provider == "openai" and not self.openai_api_key:
            self.logger.warning(
                "LLM provider is set to openai, but OPENAI_API_KEY is not set. Falling back to heuristic ranking."
            )

    def rank_candidates(self, request: LLMRankRequest) -> LLMRankResponse:
        if self.provider == "openai" and self.openai_api_key:
            try:
                return self._rank_candidates_openai(request)
            except Exception as exc:
                self.logger.warning(
                    "OpenAI ranking failed, falling back to heuristic ranker: %s", exc
                )
                fallback = self._rank_candidates_heuristic(request)
                fallback.notes.extend(
                    [
                        "requested_provider=openai",
                        "effective_provider=heuristic",
                        f"fallback_reason=openai_error:{type(exc).__name__}",
                    ]
                )
                return fallback
        fallback = self._rank_candidates_heuristic(request)
        if self.provider == "openai" and not self.openai_api_key:
            fallback.notes.extend(
                [
                    "requested_provider=openai",
                    "effective_provider=heuristic",
                    "fallback_reason=missing_api_key",
                ]
            )
        return fallback

    def parse_semantic_request(self, message: str, namespace: str = "sock-shop"):
        from rca_agent_skills.skills.semantic_request_parsing_skill.skill import (
            SemanticRequestParsingSkill,
        )

        if self.provider == "openai" and self.openai_api_key:
            try:
                return self._parse_semantic_request_openai(message, namespace)
            except Exception as exc:
                self.logger.warning(
                    "OpenAI semantic parsing failed, falling back to heuristic parser: %s",
                    exc,
                )
        parser = SemanticRequestParsingSkill({}, llm_client=None)
        result = parser.parse_heuristic(message)
        result.notes.append(f"provider={self.provider}")
        result.notes.append("semantic_parse_mode=heuristic")
        return result

    def _parse_semantic_request_openai(self, message: str, namespace: str):
        from rca_agent_skills.skills.semantic_request_parsing_skill.schemas import (
            SemanticParseResult,
        )

        system_prompt = (
            "You parse a natural-language RCA request into structured execution options. "
            "Do not perform root-cause analysis. Do not infer ground truth. "
            "Do not generate or select a baseline window. Return JSON only."
        )
        instructions = [
            "Extract abnormal_window.start and abnormal_window.end from the message.",
            "Use ISO-8601 UTC timestamps.",
            "Do not output a baseline_window.",
            "If telemetry is unspecified, enable metrics, logs, and traces.",
            "If requested outputs are unspecified, enable service_ranking, pod_ranking, service_fault_ranking, and pod_fault_ranking.",
            "If the abnormal time window is missing or ambiguous, set needs_clarification=true and include clarification_questions.",
        ]
        user_payload = {
            "namespace": namespace,
            "message": message,
            "instructions": instructions,
            "json_shape": {
                "abnormal_window": {"start": "ISO-8601 UTC", "end": "ISO-8601 UTC"},
                "enabled_telemetry": {
                    "metrics": True,
                    "logs": True,
                    "traces": True,
                },
                "requested_outputs": {
                    "service_ranking": True,
                    "pod_ranking": True,
                    "service_fault_ranking": True,
                    "pod_fault_ranking": True,
                },
                "ranking_depth": 5,
                "needs_clarification": False,
                "clarification_questions": [],
                "notes": [],
            },
        }
        body = {
            "model": self.model,
            "input": [
                {
                    "role": "developer",
                    "content": [{"type": "input_text", "text": system_prompt}],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": json.dumps(user_payload, ensure_ascii=False),
                        }
                    ],
                },
            ],
            "temperature": self.temperature,
            "text": {"format": {"type": "json_object"}},
        }
        if self.debug.get("print_llm_io", True):
            log_json(self.logger, "[LLM][SEMANTIC_INPUT] ", body)

        response = requests.post(
            f"{self.openai_base_url}/responses",
            headers={
                "Authorization": f"Bearer {self.openai_api_key}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=120,
        )
        response.raise_for_status()
        raw_response = response.json()
        if self.debug.get("print_llm_io", True):
            log_json(self.logger, "[LLM][SEMANTIC_OUTPUT] ", raw_response)
        parsed = json.loads(self._extract_output_text(raw_response))
        parsed["raw_response"] = raw_response
        parsed.setdefault("notes", []).extend(
            [
                f"provider={self.provider}",
                f"model={self.model}",
                "semantic_parse_mode=openai",
            ]
        )
        defaults = SemanticParseResult()
        return SemanticParseResult(
            abnormal_window=parsed.get("abnormal_window"),
            enabled_telemetry={
                **defaults.enabled_telemetry,
                **(parsed.get("enabled_telemetry") or {}),
            },
            requested_outputs={
                **defaults.requested_outputs,
                **(parsed.get("requested_outputs") or {}),
            },
            ranking_depth=int(parsed.get("ranking_depth") or 5),
            needs_clarification=bool(parsed.get("needs_clarification", False)),
            clarification_questions=list(parsed.get("clarification_questions") or []),
            notes=list(parsed.get("notes") or []),
            raw_response=raw_response,
        )

    def _rank_candidates_heuristic(self, request: LLMRankRequest) -> LLMRankResponse:
        ranked = sorted(
            request.candidates,
            key=lambda item: (
                float(item.get("provisional_score", 0.0)),
                float(item.get("evidence_count", 0.0)),
                float(item.get("dependency_boost", 0.0)),
            ),
            reverse=True,
        )
        return LLMRankResponse(
            rankings=ranked,
            notes=[
                f"provider={self.provider}",
                f"model={self.model}",
                "ranking_mode=heuristic",
            ],
        )

    def _rank_candidates_openai(self, request: LLMRankRequest) -> LLMRankResponse:
        system_prompt = (
            "You are an RCA ranking model for microservice failures. "
            "Rank the candidate hypotheses using the provided candidates, evidence context, and topology. "
            "The candidates are generated by lightweight detectors and rules; treat them as hypotheses, not conclusions. "
            "Prefer pod-level specificity when a pod clearly explains a service-level anomaly. "
            "Return valid JSON only with the shape "
            '{"rankings": [...], "notes": [...]} .'
        )
        instructions = [
            "Re-rank candidates in descending order of root-cause likelihood.",
            "Think independently from the raw evidence, topology, propagation hints, and candidate explanations before relying on any score.",
            "posterior_probability is a lightweight Bayesian diagnostic hint, not calibrated truth and not a hard prior.",
            "provisional_score is a reference score that you may revise; do not rank by score alone.",
            "You may ignore or override Bayesian/heuristic scores whenever they are inconsistent with concrete local evidence, dependency topology, temporal order, or propagated-symptom reasoning.",
            "If a candidate has only generic weak evidence such as network_rx/network_tx movement, do not rank it above a candidate with direct restart, readiness, OOM/crash, exception, failure, or dependency-specific evidence unless topology strongly supports it.",
            "Use context.evidence_tree when present to check whether each candidate is supported by direct, specific evidence.",
            "Use context.propagation_hints to identify likely cascade patterns, especially localized exception evidence adjacent to multi-pod latency/error symptoms.",
            "If propagation_hints include downstream_local_failure_vs_upstream_multi_pod_failures, treat that as strong evidence that the downstream dependency may explain the upstream multi-pod restart/readiness failures.",
            "Use context.reasoning_rule_guide as practical RCA guidance, not deterministic truth.",
            "Use context.topology to distinguish plausible root causes from upstream or downstream symptoms.",
            "Prefer candidates with strong local anomalies over broad downstream symptoms.",
            "When one adjacent service has explicit exception evidence in a single pod and another service has similar symptoms across sibling pods, consider whether the former better explains the latter as propagated impact.",
            "When an upstream service and its sibling pods look very bad but a downstream dependency has more direct local failure evidence, you should explicitly lower the upstream candidate score or raise the downstream candidate score to reflect the more plausible root-cause story.",
            "Prefer explicit failure signals such as restart_count, ready_ratio drops, OOM, crash, and readiness failures over generic CPU, latency, or error symptoms when diagnosing pod_failure.",
            "Prefer pod-level candidates when they clearly localize the fault inside a service.",
            "Candidates may include rule_hints and active_rules. Treat them as coarse priors and plausibility signals, not hard truth.",
            "Do not simply preserve the incoming candidate order, posterior scores, or provisional scores.",
            "If you change the ranking order because the supplied scores look wrong, also assign new provisional_score values that reflect your revised confidence.",
            "Keep scores monotonic with your final ranking: higher-ranked candidates should generally have equal or higher provisional_score than lower-ranked candidates unless you explain a close tie.",
            "Use the full 0.05-0.98 score range when evidence quality differs materially; weak generic-symptom candidates should receive low or moderate scores.",
            "Keep service-level ranking aligned with the strongest pod-level evidence for the same service whenever the pod evidence is more direct.",
            "Down-rank routine database/listener logs such as connection accepted or checkpoint chatter unless they are corroborated by stronger failure signals.",
            "Avoid assigning a score near 1.0 unless the evidence is unusually direct, specific, and well corroborated.",
            "Keep the same fields, but you may adjust provisional_score and notes.",
            'Return JSON with top-level keys "rankings" and "notes" only.',
        ]
        if request.prompt_name == "cross_level_ranking_reconciliation":
            instructions.extend(
                [
                    "This is a cross-level reconciliation pass over already-ranked service and pod hypotheses.",
                    "Use context.service_ranking_preview and context.pod_ranking_preview as prior LLM judgments, not final truth.",
                    "Return one combined rankings array containing both service and pod candidates, keeping entity_type unchanged.",
                    "Improve service/pod consistency: a top pod should usually make its parent service plausible, and a top service should usually be explained by service-level or pod-level evidence.",
                    "Use context.alignment_hints to check whether a strong localized pod root cause should promote its parent service above symptom-only services.",
                    "If you rank a pod as the clearest root cause, adjust the matching parent service candidate score/rank so the final service and pod rankings tell the same root-cause story.",
                    "Do not leave multi-pod latency/error-only services above the parent service of a stronger CPU, memory, restart, readiness, or exception root-cause pod unless their local evidence is stronger.",
                    "Do not force alignment when evidence shows a pod is incidental or a service-level symptom has no local pod support.",
                    "If context.propagation_hints point to a single-pod exception source adjacent to a multi-pod symptom service, reconcile the rankings around the localized exception source when the evidence tree supports it.",
                    "If context.propagation_hints point to downstream_local_failure_vs_upstream_multi_pod_failures, reconcile the rankings around the downstream dependency unless the upstream service has stronger direct local evidence.",
                    "Use notes to explain any important service/pod inconsistency or alignment decision.",
                ]
            )
        user_payload = {
            "prompt_name": request.prompt_name,
            "context": request.context,
            "candidates": request.candidates,
            "instructions": instructions,
        }
        body = {
            "model": self.model,
            "input": [
                {
                    "role": "developer",
                    "content": [{"type": "input_text", "text": system_prompt}],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": json.dumps(user_payload, ensure_ascii=False),
                        }
                    ],
                },
            ],
            "temperature": self.temperature,
            "text": {
                "format": {
                    "type": "json_object",
                }
            },
        }
        if self.debug.get("print_llm_io", True):
            log_json(self.logger, "[LLM][INPUT] ", body)

        response = requests.post(
            f"{self.openai_base_url}/responses",
            headers={
                "Authorization": f"Bearer {self.openai_api_key}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=120,
        )
        response.raise_for_status()
        raw_response = response.json()
        if self.debug.get("print_llm_io", True):
            log_json(self.logger, "[LLM][OUTPUT] ", raw_response)

        output_text = self._extract_output_text(raw_response)
        parsed = json.loads(output_text)
        rankings = parsed.get("rankings", request.candidates)
        if not isinstance(rankings, list):
            rankings = request.candidates
        return LLMRankResponse(
            rankings=rankings,
            notes=parsed.get("notes", [])
            + [
                f"provider={self.provider}",
                f"model={self.model}",
                "ranking_mode=openai",
            ],
            raw_request=body,
            raw_response=raw_response,
        )

    def _extract_output_text(self, payload: dict[str, Any]) -> str:
        output_text = payload.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text

        parts: list[str] = []
        for item in payload.get("output", []):
            for content in item.get("content", []):
                text = content.get("text")
                if isinstance(text, str) and text:
                    parts.append(text)
        if parts:
            return "".join(parts)
        raise ValueError("No textual JSON output found in OpenAI response")

    def propose_query_intents(
        self, skill: str, context: dict[str, Any], max_items: int
    ) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        focus_entities = context.get("focus_entities", [])
        for entity in focus_entities[:max_items]:
            candidates.append(
                {
                    "skill": skill,
                    "reason": f"Need follow-up evidence for {entity.get('name')}",
                    "service": entity.get("service"),
                    "pod": entity.get("pod"),
                    "window": "abnormal",
                }
            )
        return candidates
