from __future__ import annotations

from collections import defaultdict

import pandas as pd

from rca_agent_skills.common.logging_utils import get_logger, log_json
from rca_agent_skills.common.models import PodEvidence, ServiceEvidence, SkillResult
from rca_agent_skills.common.severity_calibrator import SeverityCalibrator
from rca_agent_skills.data_access.topology_loader import service_from_pod
from .detector import detect_in_window_metric_patterns, detect_metric_anomalies, to_anomaly_records
from .query_builder import build_metric_followup_intents


POD_PREFERRED_SERVICE_METRICS = {
    "cpu_usage_pct",
    "memory_usage_pct",
    "restart_count",
    "ready_ratio",
    "request_rate",
    "success_rate",
    "error_count",
    "latency_p50",
    "latency_p90",
    "latency_p95",
    "latency_p99",
    "network_rx",
    "network_tx",
}


class MetricEvidenceSkill:
    def __init__(self, settings: dict, data_access, llm_client):
        self.settings = settings
        self.data_access = data_access
        self.llm_client = llm_client
        self.logger = get_logger(self.__class__.__name__)
        self.debug = settings.get("debug", {})
        self.severity_calibrator = SeverityCalibrator(settings.get("severity_calibration", {}))

    def _prepare_metrics(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame(columns=["timestamp", "pod", "service", "metric", "value"])
        prepared = frame.copy()
        prepared["timestamp"] = pd.to_datetime(prepared["timestamp"], utc=True, errors="coerce")
        if "service" not in prepared.columns:
            prepared["service"] = prepared["pod"].fillna("").map(service_from_pod)
        prepared["value"] = pd.to_numeric(prepared["value"], errors="coerce")
        prepared = prepared.dropna(subset=["metric", "value"])
        return prepared

    def _metric_support_text(self, record) -> str:
        pod = record.metadata.get("pod")
        service = record.metadata.get("service") or record.entity_name
        scope = f"pod {pod}" if record.entity_type == "pod" and pod else f"service {service}"
        return f"{scope} {record.metric_or_pattern}: {record.summary}"

    def _build_service_evidence(self, records):
        grouped = defaultdict(list)
        for record in records:
            service = record.metadata.get("service") or record.entity_name
            grouped[service].append(record)
        evidence_list: list[ServiceEvidence] = []
        for service, items in grouped.items():
            score = round(sum(item.severity for item in items) / max(len(items), 1), 4)
            support = [self._metric_support_text(item) for item in sorted(items, key=lambda x: x.severity, reverse=True)[:3]]
            suspicious_pods = []
            for item in sorted(items, key=lambda x: x.severity, reverse=True):
                pod = item.metadata.get("pod")
                if pod and pod not in suspicious_pods:
                    suspicious_pods.append(pod)
            notes = [f"suspect_pods={', '.join(suspicious_pods[:3])}"] if suspicious_pods else []
            evidence_list.append(ServiceEvidence(service=service, score=score, supporting_evidence=support, anomaly_records=items, notes=notes))
        return sorted(evidence_list, key=lambda item: item.score, reverse=True)

    def _build_pod_evidence(self, records):
        grouped = defaultdict(list)
        for record in records:
            if record.entity_type != "pod":
                continue
            grouped[record.entity_name].append(record)
        evidence_list: list[PodEvidence] = []
        for pod, items in grouped.items():
            service = items[0].metadata.get("service") or service_from_pod(pod)
            score = round(sum(item.severity for item in items) / max(len(items), 1), 4)
            support = [self._metric_support_text(item) for item in sorted(items, key=lambda x: x.severity, reverse=True)[:3]]
            evidence_list.append(PodEvidence(pod=pod, service=service, score=score, supporting_evidence=support, anomaly_records=items))
        return sorted(evidence_list, key=lambda item: item.score, reverse=True)

    def _pod_scoped_metrics(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty or "pod" not in frame.columns:
            return pd.DataFrame(columns=frame.columns)
        return frame[frame["pod"].notna() & (frame["pod"].astype(str).str.strip() != "")].copy()

    def _pod_metric_keys(self, *frames: pd.DataFrame) -> set[tuple[str, str]]:
        keys: set[tuple[str, str]] = set()
        for frame in frames:
            if frame.empty:
                continue
            for row in frame[["service", "metric"]].dropna().itertuples(index=False):
                service, metric = row
                if metric in POD_PREFERRED_SERVICE_METRICS:
                    keys.add((str(service), str(metric)))
        return keys

    def _remove_service_duplicates_when_pod_data_exists(self, pod_metric_keys: set[tuple[str, str]], service_features: list) -> list:
        if not pod_metric_keys:
            return service_features
        return [
            feature
            for feature in service_features
            if not (
                feature.entity_type == "service"
                and (feature.service or feature.entity_name, feature.metric) in pod_metric_keys
            )
        ]

    def run(self, request, state) -> SkillResult:
        thresholds = self.settings.get("detection", {})
        kpi_directions = request.config_bundle["metric_kpis"].get("kpis", {})
        baseline = self._prepare_metrics(self.data_access.get_metrics("baseline"))
        abnormal = self._prepare_metrics(self.data_access.get_metrics("abnormal"))
        if self.debug.get("print_skill_inputs", True):
            log_json(
                self.logger,
                "[METRIC][INPUT] ",
                {
                    "baseline_rows": len(baseline),
                    "abnormal_rows": len(abnormal),
                    "baseline_services": sorted(baseline["service"].dropna().unique().tolist())[:20] if not baseline.empty else [],
                    "abnormal_services": sorted(abnormal["service"].dropna().unique().tolist())[:20] if not abnormal.empty else [],
                },
            )

        pod_baseline = self._pod_scoped_metrics(baseline)
        pod_abnormal = self._pod_scoped_metrics(abnormal)
        pod_metric_keys = self._pod_metric_keys(pod_baseline, pod_abnormal)
        pod_features = detect_metric_anomalies(
            pod_baseline,
            pod_abnormal,
            thresholds,
            kpi_directions,
            ["pod", "service"],
            severity_calibrator=self.severity_calibrator,
        )
        pod_shape_features = detect_in_window_metric_patterns(pod_baseline, pod_abnormal, thresholds, kpi_directions, ["pod", "service"])
        service_base = baseline.groupby(["service", "metric"], as_index=False)["value"].mean() if not baseline.empty else pd.DataFrame(columns=["service", "metric", "value"])
        service_abn = abnormal.groupby(["service", "metric"], as_index=False)["value"].mean() if not abnormal.empty else pd.DataFrame(columns=["service", "metric", "value"])
        service_features = detect_metric_anomalies(
            service_base,
            service_abn,
            thresholds,
            kpi_directions,
            ["service"],
            severity_calibrator=self.severity_calibrator,
        )
        if not baseline.empty and "timestamp" in baseline.columns:
            service_shape_base = baseline.groupby(["timestamp", "service", "metric"], as_index=False)["value"].mean()
        else:
            service_shape_base = pd.DataFrame(columns=["timestamp", "service", "metric", "value"])
        if not abnormal.empty and "timestamp" in abnormal.columns:
            service_shape_abn = abnormal.groupby(["timestamp", "service", "metric"], as_index=False)["value"].mean()
        else:
            service_shape_abn = pd.DataFrame(columns=["timestamp", "service", "metric", "value"])
        service_shape_features = detect_in_window_metric_patterns(service_shape_base, service_shape_abn, thresholds, kpi_directions, ["service"])
        service_features = self._remove_service_duplicates_when_pod_data_exists(pod_metric_keys, service_features)
        service_shape_features = self._remove_service_duplicates_when_pod_data_exists(pod_metric_keys, service_shape_features)
        records = to_anomaly_records(pod_features + pod_shape_features + service_features + service_shape_features)
        service_evidence = self._build_service_evidence(records)
        pod_evidence = self._build_pod_evidence(records)

        metadata = {}
        if self.settings.get("features", {}).get("enable_followup_query_expansion", False):
            intents = build_metric_followup_intents(service_evidence, int(self.settings.get("budgets", {}).get("metric_followup", 2)))
            metadata["followup_intents"] = [intent.__dict__ for intent in intents]
        result = SkillResult(service_evidence=service_evidence, pod_evidence=pod_evidence, anomaly_records=records, metadata=metadata)
        if self.debug.get("print_anomaly_records", True):
            max_items = int(self.debug.get("max_logged_records_per_skill", 10))
            log_json(
                self.logger,
                "[METRIC][ANOMALIES] ",
                [
                    {
                        "entity_type": item.entity_type,
                        "entity_name": item.entity_name,
                        "metric": item.metric_or_pattern,
                        "summary": item.summary,
                        "severity": item.severity,
                        "raw_severity": item.metadata.get("raw_severity"),
                        "severity_method": item.metadata.get("severity_method"),
                        "service": item.metadata.get("service"),
                        "pod": item.metadata.get("pod"),
                    }
                    for item in sorted(records, key=lambda x: x.severity, reverse=True)[:max_items]
                ],
            )
        if self.debug.get("print_skill_outputs", True):
            log_json(
                self.logger,
                "[METRIC][OUTPUT] ",
                {
                    "service_evidence": [
                        {
                            "service": item.service,
                            "score": item.score,
                            "notes": item.notes,
                            "supporting_evidence": item.supporting_evidence,
                        }
                        for item in service_evidence[: int(self.settings.get("defaults", {}).get("top_k", 5))]
                    ],
                    "pod_evidence": [
                        {
                            "pod": item.pod,
                            "service": item.service,
                            "score": item.score,
                            "supporting_evidence": item.supporting_evidence,
                        }
                        for item in pod_evidence[: int(self.settings.get("defaults", {}).get("top_k", 5))]
                    ],
                },
            )
        return result
