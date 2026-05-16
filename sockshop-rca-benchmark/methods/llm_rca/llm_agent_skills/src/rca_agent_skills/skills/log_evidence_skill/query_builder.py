from __future__ import annotations

from rca_agent_skills.query_expansion.intent_schema import LogQueryIntent


def build_log_followup_intents(service_evidence: list, max_items: int) -> list[LogQueryIntent]:
    intents: list[LogQueryIntent] = []
    for evidence in service_evidence[:max_items]:
        keyword = None
        if evidence.anomaly_records:
            keyword = evidence.anomaly_records[0].metadata.get("pattern_value")
        intents.append(
            LogQueryIntent(
                skill="log",
                reason=f"Inspect dominant abnormal log pattern for {evidence.service}",
                service=evidence.service,
                pod=None,
                keyword=keyword,
                template=keyword,
                window="abnormal",
            )
        )
    return intents

