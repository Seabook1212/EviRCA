from __future__ import annotations

from collections import defaultdict

from rca_agent_skills.common.logging_utils import get_logger, log_json
from rca_agent_skills.common.models import PodEvidence, ServiceEvidence, SkillResult
from rca_agent_skills.common.severity_calibrator import SeverityCalibrator
from rca_agent_skills.data_access.topology_loader import service_from_pod
from .detector import detect_trace_anomalies, to_anomaly_records
from .parser import build_trace_paths, prepare_traces
from .query_builder import build_trace_followup_intents


class TraceEvidenceSkill:
    def __init__(self, settings: dict, data_access, llm_client):
        self.settings = settings
        self.data_access = data_access
        self.llm_client = llm_client
        self.logger = get_logger(self.__class__.__name__)
        self.debug = settings.get("debug", {})
        self.severity_calibrator = SeverityCalibrator(settings.get("severity_calibration", {}))

    def _format_record_summary(self, record) -> str:
        source = record.metadata.get("edge_source_service")
        target = record.metadata.get("edge_target_service")
        role = record.metadata.get("edge_role")
        pod = record.metadata.get("pod")
        if source and target:
            if role == "source_pod" and pod:
                return f"source pod {pod} {source}->{target}: {record.summary}"
            if role == "target_pod" and pod:
                return f"target pod {pod} {source}->{target}: {record.summary}"
            return f"{source}->{target}: {record.summary}"
        return record.summary

    def _group_service_evidence(self, records):
        grouped = defaultdict(list)
        for record in records:
            grouped[record.metadata.get("service") or record.entity_name].append(record)
        evidence = []
        for service, items in grouped.items():
            score = round(sum(item.severity for item in items) / max(len(items), 1), 4)
            support = [self._format_record_summary(item) for item in sorted(items, key=lambda x: x.severity, reverse=True)[:3]]
            notes = [f"peer_service={item.metadata.get('peer_service')}" for item in items if item.metadata.get("peer_service")]
            evidence.append(ServiceEvidence(service=service, score=score, supporting_evidence=support, anomaly_records=items, notes=notes))
        return sorted(evidence, key=lambda item: item.score, reverse=True)

    def _group_pod_evidence(self, records):
        grouped = defaultdict(list)
        for record in records:
            pod = record.metadata.get("pod")
            if pod:
                grouped[pod].append(record)
        evidence = []
        for pod, items in grouped.items():
            service = items[0].metadata.get("service") or service_from_pod(pod)
            score = round(sum(item.severity for item in items) / max(len(items), 1), 4)
            support = [self._format_record_summary(item) for item in sorted(items, key=lambda x: x.severity, reverse=True)[:3]]
            evidence.append(PodEvidence(pod=pod, service=service, score=score, supporting_evidence=support, anomaly_records=items))
        return sorted(evidence, key=lambda item: item.score, reverse=True)

    def run(self, request, state) -> SkillResult:
        thresholds = self.settings.get("detection", {})
        baseline = prepare_traces(self.data_access.get_traces("baseline"))
        abnormal = prepare_traces(self.data_access.get_traces("abnormal"))
        if self.debug.get("print_skill_inputs", True):
            log_json(self.logger, "[TRACE][INPUT] ", {"baseline_rows": len(baseline), "abnormal_rows": len(abnormal)})
        features = detect_trace_anomalies(
            baseline,
            abnormal,
            thresholds,
            severity_calibrator=self.severity_calibrator,
        )
        records = to_anomaly_records(features)
        service_evidence = self._group_service_evidence(records)
        pod_evidence = self._group_pod_evidence(records)
        metadata = {
            "abnormal_paths": build_trace_paths(abnormal).head(10).to_dict(orient="records"),
            "propagation_hints": list(
                dict.fromkeys(
                    f"{record.metadata.get('edge_source_service')} -> {record.metadata.get('edge_target_service')}"
                    if record.metadata.get("edge_source_service") and record.metadata.get("edge_target_service")
                    else f"{record.entity_name} -> {record.metadata.get('peer_service')}"
                    for record in records
                    if record.metadata.get("peer_service")
                )
            ),
        }
        if self.settings.get("features", {}).get("enable_followup_query_expansion", False):
            intents = build_trace_followup_intents(service_evidence, int(self.settings.get("budgets", {}).get("trace_followup", 2)))
            metadata["followup_intents"] = [intent.__dict__ for intent in intents]
        result = SkillResult(service_evidence=service_evidence, pod_evidence=pod_evidence, anomaly_records=records, metadata=metadata)
        if self.debug.get("print_anomaly_records", True):
            max_items = int(self.debug.get("max_logged_records_per_skill", 10))
            log_json(
                self.logger,
                "[TRACE][ANOMALIES] ",
                [
                    {
                        "entity_type": item.entity_type,
                        "entity_name": item.entity_name,
                        "pattern": item.metric_or_pattern,
                        "summary": item.summary,
                        "severity": item.severity,
                        "raw_severity": item.metadata.get("raw_severity"),
                        "severity_method": item.metadata.get("severity_method"),
                        "service": item.metadata.get("service"),
                        "pod": item.metadata.get("pod"),
                        "peer_service": item.metadata.get("peer_service"),
                        "edge_role": item.metadata.get("edge_role"),
                        "edge_source_service": item.metadata.get("edge_source_service"),
                        "edge_target_service": item.metadata.get("edge_target_service"),
                    }
                    for item in sorted(records, key=lambda x: x.severity, reverse=True)[:max_items]
                ],
            )
        if self.debug.get("print_skill_outputs", True):
            log_json(
                self.logger,
                "[TRACE][OUTPUT] ",
                {
                    "service_evidence": [{"service": item.service, "score": item.score, "notes": item.notes, "supporting_evidence": item.supporting_evidence} for item in service_evidence[:5]],
                    "pod_evidence": [{"pod": item.pod, "service": item.service, "score": item.score, "supporting_evidence": item.supporting_evidence} for item in pod_evidence[:5]],
                    "propagation_hints": metadata.get("propagation_hints", [])[:10],
                },
            )
        return result
