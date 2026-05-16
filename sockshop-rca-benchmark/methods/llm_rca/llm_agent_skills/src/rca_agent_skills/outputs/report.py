from __future__ import annotations


def _top_entities(items, entity_field: str) -> list[tuple[str, float]]:
    best_scores: dict[str, float] = {}
    for item in items:
        entity_name = getattr(item, entity_field, None)
        if not entity_name:
            continue
        score = float(getattr(item, "score", 0.0))
        if entity_name not in best_scores or score > best_scores[entity_name]:
            best_scores[entity_name] = score
    return sorted(best_scores.items(), key=lambda pair: pair[1], reverse=True)[:5]


def render_text_report(result) -> str:
    lines = [f"Incident: {result.incident_id}", f"Summary: {result.final_summary}", ""]
    llm_metadata = (getattr(result, "metadata", {}) or {}).get("llm", {})
    llm_notes = []
    llm_notes.extend(llm_metadata.get("service_ranking_notes", []))
    llm_notes.extend(llm_metadata.get("pod_ranking_notes", []))
    if llm_notes:
        lines.append(f"Ranking engine: {' | '.join(dict.fromkeys(llm_notes))}")
        lines.append("")
    lines.append("Service hypotheses:")
    for item in result.service_top5:
        lines.append(f"- {item.service}: {item.fault_type} ({item.score:.2f})")
    lines.append("")
    lines.append("Pod hypotheses:")
    for item in result.pod_top5:
        lines.append(f"- {item.pod}: {item.fault_type} ({item.score:.2f})")
    lines.append("")
    lines.append("Top services:")
    top_services = _top_entities(result.service_top5, "service")
    for idx, (service, score) in enumerate(top_services, start=1):
        lines.append(f"{idx}. {service} ({score:.2f})")
    if not top_services:
        lines.append("No service anomaly evidence.")
    lines.append("")
    lines.append("Top pods:")
    top_pods = _top_entities(result.pod_top5, "pod")
    for idx, (pod, score) in enumerate(top_pods, start=1):
        lines.append(f"{idx}. {pod} ({score:.2f})")
    if not top_pods:
        lines.append("No pod anomaly evidence.")
    return "\n".join(lines)
