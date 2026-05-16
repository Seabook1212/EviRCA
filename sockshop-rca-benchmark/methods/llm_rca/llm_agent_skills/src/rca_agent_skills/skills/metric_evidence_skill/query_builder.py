from __future__ import annotations

from rca_agent_skills.query_expansion.intent_schema import MetricQueryIntent


def build_metric_followup_intents(service_evidence: list, max_items: int) -> list[MetricQueryIntent]:
    intents: list[MetricQueryIntent] = []
    for evidence in service_evidence[:max_items]:
        kpi = None
        if evidence.anomaly_records:
            kpi = evidence.anomaly_records[0].metric_or_pattern
        if not kpi:
            continue
        intents.append(
            MetricQueryIntent(
                skill="metric",
                reason=f"Inspect top anomalous KPI for {evidence.service}",
                service=evidence.service,
                pod=None,
                kpi=kpi,
                granularity="service",
                window="abnormal",
            )
        )
    return intents

