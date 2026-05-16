from __future__ import annotations

from rca_agent_skills.query_expansion.intent_schema import TraceQueryIntent


def build_trace_followup_intents(service_evidence: list, max_items: int) -> list[TraceQueryIntent]:
    intents: list[TraceQueryIntent] = []
    for evidence in service_evidence[:max_items]:
        peer_service = None
        if evidence.anomaly_records:
            peer_service = evidence.anomaly_records[0].metadata.get("peer_service")
        intents.append(
            TraceQueryIntent(
                skill="trace",
                reason=f"Inspect high-latency or failing downstream edges for {evidence.service}",
                service=evidence.service,
                peer_service=peer_service,
                operation=None,
                status=None,
                window="abnormal",
            )
        )
    return intents

