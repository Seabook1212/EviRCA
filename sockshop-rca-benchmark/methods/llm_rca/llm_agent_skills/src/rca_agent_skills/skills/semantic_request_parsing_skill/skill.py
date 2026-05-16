from __future__ import annotations

import json
import re
from typing import Any

from rca_agent_skills.common.logging_utils import get_logger

from .schemas import SemanticParseResult


TIMESTAMP_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}(?:Z|[+-]\d{2}:?\d{2})?"
)


class SemanticRequestParsingSkill:
    def __init__(self, settings: dict, llm_client):
        self.settings = settings
        self.llm_client = llm_client
        self.logger = get_logger(self.__class__.__name__)

    def run(self, message: str, namespace: str = "sock-shop") -> SemanticParseResult:
        if hasattr(self.llm_client, "parse_semantic_request"):
            result = self.llm_client.parse_semantic_request(
                message, namespace=namespace
            )
            if isinstance(result, SemanticParseResult):
                return result
            if isinstance(result, dict):
                return self._coerce_result(result)
        return self.parse_heuristic(message)

    def parse_heuristic(self, message: str) -> SemanticParseResult:
        lower = message.lower()
        timestamps = [
            self._normalize_timestamp(item) for item in TIMESTAMP_RE.findall(message)
        ]
        abnormal_window = None
        questions = []
        if len(timestamps) >= 2:
            abnormal_window = {"start": timestamps[0], "end": timestamps[1]}
        else:
            questions.append(
                "Please provide the abnormal start and end time in ISO-8601 format."
            )

        enabled_telemetry = {"metrics": True, "logs": True, "traces": True}
        telemetry_terms = {
            "metrics": "metrics",
            "metric": "metrics",
            "logs": "logs",
            "log": "logs",
            "traces": "traces",
            "trace": "traces",
        }
        mentioned = {
            target for term, target in telemetry_terms.items() if term in lower
        }
        if mentioned and ("only" in lower or "只" in message):
            enabled_telemetry = {key: key in mentioned for key in enabled_telemetry}
        for key, words in {
            "metrics": [
                "no metrics",
                "without metrics",
                "disable metrics",
                "不要用 metrics",
                "不用 metrics",
            ],
            "logs": [
                "no logs",
                "without logs",
                "disable logs",
                "不要用 logs",
                "不用 logs",
                "不要用 log",
                "不用 log",
            ],
            "traces": [
                "no traces",
                "without traces",
                "disable traces",
                "不要用 traces",
                "不用 traces",
                "不要用 trace",
                "不用 trace",
            ],
        }.items():
            if any(word in lower or word in message for word in words):
                enabled_telemetry[key] = False

        requested_outputs = {
            "service_ranking": True,
            "pod_ranking": True,
            "service_fault_ranking": True,
            "pod_fault_ranking": True,
        }
        output_mentioned = any(
            word in lower for word in ["service", "pod", "fault", "type", "ranking"]
        ) or any(word in message for word in ["服务", "异常类型", "排名"])
        if output_mentioned and ("only" in lower or "只" in message):
            wants_service = "service" in lower or "服务" in message
            wants_pod = "pod" in lower
            wants_fault = "fault" in lower or "type" in lower or "异常类型" in message
            requested_outputs = {
                "service_ranking": wants_service and not wants_fault,
                "pod_ranking": wants_pod and not wants_fault,
                "service_fault_ranking": wants_service and wants_fault,
                "pod_fault_ranking": wants_pod and wants_fault,
            }
            if wants_service and not wants_pod and not wants_fault:
                requested_outputs["service_ranking"] = True
            if wants_pod and not wants_service and not wants_fault:
                requested_outputs["pod_ranking"] = True
            if not any(requested_outputs.values()):
                requested_outputs = {
                    "service_ranking": True,
                    "pod_ranking": True,
                    "service_fault_ranking": True,
                    "pod_fault_ranking": True,
                }

        return SemanticParseResult(
            abnormal_window=abnormal_window,
            enabled_telemetry=enabled_telemetry,
            requested_outputs=requested_outputs,
            needs_clarification=bool(questions),
            clarification_questions=questions,
            notes=["semantic_parse_mode=heuristic"],
        )

    def _coerce_result(self, payload: dict[str, Any]) -> SemanticParseResult:
        defaults = SemanticParseResult()
        return SemanticParseResult(
            abnormal_window=payload.get("abnormal_window"),
            enabled_telemetry={
                **defaults.enabled_telemetry,
                **(payload.get("enabled_telemetry") or {}),
            },
            requested_outputs={
                **defaults.requested_outputs,
                **(payload.get("requested_outputs") or {}),
            },
            ranking_depth=int(payload.get("ranking_depth") or 5),
            needs_clarification=bool(payload.get("needs_clarification", False)),
            clarification_questions=list(payload.get("clarification_questions") or []),
            notes=list(payload.get("notes") or []),
            raw_response=payload.get("raw_response"),
        )

    def _normalize_timestamp(self, value: str) -> str:
        normalized = value.strip().replace(" ", "T")
        if normalized.endswith("Z") or re.search(r"[+-]\d{2}:?\d{2}$", normalized):
            return normalized
        return f"{normalized}Z"


def parse_semantic_json(text: str) -> dict[str, Any]:
    return json.loads(text)
