from __future__ import annotations

from collections import defaultdict

import pandas as pd

from rca_agent_skills.common.logging_utils import get_logger, log_json
from rca_agent_skills.common.models import PodEvidence, ServiceEvidence, SkillResult
from rca_agent_skills.common.severity_calibrator import SeverityCalibrator
from rca_agent_skills.data_access.topology_loader import service_from_pod
from .detector import detect_log_spikes, to_anomaly_records
from .parser import normalize_message, parse_raw_log
from .query_builder import build_log_followup_intents


class LogEvidenceSkill:
    def __init__(self, settings: dict, data_access, llm_client):
        self.settings = settings
        self.data_access = data_access
        self.llm_client = llm_client
        self.logger = get_logger(self.__class__.__name__)
        self.debug = settings.get("debug", {})
        self.severity_calibrator = SeverityCalibrator(settings.get("severity_calibration", {}))

    def _prepare_logs(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame(columns=["timestamp", "service", "pod", "log_level", "message", "raw_log"])
        prepared = frame.copy()
        prepared["timestamp"] = pd.to_datetime(prepared["timestamp"], utc=True, errors="coerce")
        if "service" not in prepared.columns:
            prepared["service"] = prepared.get("container", prepared.get("pod", "")).fillna("").map(service_from_pod)
        prepared["pod"] = prepared.get("pod", "").fillna("")
        prepared["raw_log"] = prepared.get("raw_log", prepared.get("message", "")).fillna("").astype(str)
        levels = prepared.get("log_level", pd.Series(["UNKNOWN"] * len(prepared))).fillna("UNKNOWN").astype(str).str.upper()
        needs_raw_parse = "message" not in prepared.columns or "log_level" not in prepared.columns
        if not needs_raw_parse and "raw_log" in prepared.columns:
            needs_raw_parse = bool((levels == "UNKNOWN").all())
        if needs_raw_parse:
            parsed = prepared.apply(
                lambda row: parse_raw_log(str(row.get("raw_log", "")), row.get("container")),
                axis=1,
                result_type="expand",
            )
            for column in ["trace_id", "span_id", "log_source", "log_type"]:
                if column not in prepared.columns and column in parsed.columns:
                    prepared[column] = parsed[column]
            if ("message" not in prepared.columns or prepared["message"].fillna("").astype(str).eq("").all()) and "message" in parsed.columns:
                prepared["message"] = parsed["message"]
            if "log_level" in parsed.columns and ("log_level" not in prepared.columns or levels.eq("UNKNOWN").all()):
                prepared["log_level"] = parsed["log_level"]
        prepared["message"] = prepared.get("message", prepared["raw_log"]).fillna("").astype(str)
        template_source = prepared.get("message_template", prepared["message"]).fillna("").astype(str)
        prepared["message_template"] = template_source.map(normalize_message)
        prepared["log_level"] = prepared.get("log_level", "UNKNOWN").fillna("UNKNOWN").astype(str).str.upper()
        return prepared

    def _group_service_evidence(self, records):
        grouped = defaultdict(list)
        for record in records:
            grouped[record.metadata.get("service") or record.entity_name].append(record)
        evidence = []
        for service, items in grouped.items():
            score = round(sum(item.severity for item in items) / max(len(items), 1), 4)
            support = [item.summary for item in sorted(items, key=lambda x: x.severity, reverse=True)[:3]]
            evidence.append(ServiceEvidence(service=service, score=score, supporting_evidence=support, anomaly_records=items))
        return sorted(evidence, key=lambda item: item.score, reverse=True)

    def _group_pod_evidence(self, records):
        grouped = defaultdict(list)
        for record in records:
            if record.entity_type == "pod":
                grouped[record.entity_name].append(record)
        evidence = []
        for pod, items in grouped.items():
            service = items[0].metadata.get("service") or service_from_pod(pod)
            score = round(sum(item.severity for item in items) / max(len(items), 1), 4)
            support = [item.summary for item in sorted(items, key=lambda x: x.severity, reverse=True)[:3]]
            evidence.append(PodEvidence(pod=pod, service=service, score=score, supporting_evidence=support, anomaly_records=items))
        return sorted(evidence, key=lambda item: item.score, reverse=True)

    def _log_pattern_key(self, record) -> tuple[str, str, str]:
        service = str(record.metadata.get("service") or record.entity_name)
        pattern_value = str(record.metadata.get("pattern_value") or record.summary)
        return service, str(record.metric_or_pattern), pattern_value

    def _remove_service_duplicates_when_pod_data_exists(self, records):
        pod_keys = {self._log_pattern_key(record) for record in records if record.entity_type == "pod"}
        if not pod_keys:
            return records
        return [
            record
            for record in records
            if not (record.entity_type == "service" and self._log_pattern_key(record) in pod_keys)
        ]

    def run(self, request, state) -> SkillResult:
        thresholds = self.settings.get("detection", {})
        baseline = self._prepare_logs(self.data_access.get_logs("baseline"))
        abnormal = self._prepare_logs(self.data_access.get_logs("abnormal"))
        if self.debug.get("print_skill_inputs", True):
            log_json(self.logger, "[LOG][INPUT] ", {"baseline_rows": len(baseline), "abnormal_rows": len(abnormal)})
        features = detect_log_spikes(
            baseline,
            abnormal,
            "service",
            thresholds,
            severity_calibrator=self.severity_calibrator,
        ) + detect_log_spikes(
            baseline,
            abnormal,
            "pod",
            thresholds,
            severity_calibrator=self.severity_calibrator,
        )
        records = self._remove_service_duplicates_when_pod_data_exists(to_anomaly_records(features))
        service_evidence = self._group_service_evidence(records)
        pod_evidence = self._group_pod_evidence(records)
        metadata = {}
        if self.settings.get("features", {}).get("enable_followup_query_expansion", False):
            intents = build_log_followup_intents(service_evidence, int(self.settings.get("budgets", {}).get("log_followup", 2)))
            metadata["followup_intents"] = [intent.__dict__ for intent in intents]
        result = SkillResult(service_evidence=service_evidence, pod_evidence=pod_evidence, anomaly_records=records, metadata=metadata)
        if self.debug.get("print_anomaly_records", True):
            max_items = int(self.debug.get("max_logged_records_per_skill", 10))
            log_json(
                self.logger,
                "[LOG][ANOMALIES] ",
                [
                    {
                        "entity_type": item.entity_type,
                        "entity_name": item.entity_name,
                        "pattern": item.metric_or_pattern,
                        "summary": item.summary,
                        "severity": item.severity,
                        "raw_severity": item.metadata.get("raw_severity"),
                        "severity_method": item.metadata.get("severity_method"),
                    }
                    for item in sorted(records, key=lambda x: x.severity, reverse=True)[:max_items]
                ],
            )
        if self.debug.get("print_skill_outputs", True):
            log_json(
                self.logger,
                "[LOG][OUTPUT] ",
                {
                    "service_evidence": [{"service": item.service, "score": item.score, "supporting_evidence": item.supporting_evidence} for item in service_evidence[:5]],
                    "pod_evidence": [{"pod": item.pod, "service": item.service, "score": item.score, "supporting_evidence": item.supporting_evidence} for item in pod_evidence[:5]],
                },
            )
        return result
