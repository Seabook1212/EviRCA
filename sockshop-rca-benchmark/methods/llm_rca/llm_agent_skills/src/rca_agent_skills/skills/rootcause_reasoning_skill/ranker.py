from __future__ import annotations

from rca_agent_skills.llm.schemas import LLMRankRequest


def _candidate_key(item: dict) -> tuple:
    return (
        item.get("entity_type"),
        item.get("service"),
        item.get("pod"),
        item.get("fault_type"),
    )


def _as_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_rankings(candidates: list[dict], rankings: list[dict]) -> list[dict]:
    candidates_by_key = {_candidate_key(candidate): candidate for candidate in candidates}
    used_keys = set()
    normalized: list[dict] = []

    for ranked in rankings:
        if not isinstance(ranked, dict):
            continue
        key = _candidate_key(ranked)
        original = candidates_by_key.get(key)
        merged = dict(original or {})
        merged.update(ranked)

        adjusted_score = _as_float(ranked.get("provisional_score"))
        if adjusted_score is None:
            adjusted_score = _as_float(ranked.get("score"))
        if adjusted_score is not None:
            merged["provisional_score"] = adjusted_score

        if not merged.get("supporting_evidence") and original:
            merged["supporting_evidence"] = original.get("supporting_evidence", [])

        notes = merged.get("notes", [])
        if isinstance(notes, str):
            notes = [notes]
        else:
            notes = list(notes or [])
        rationale = ranked.get("rationale")
        if rationale:
            notes.append(str(rationale))
        if not notes and original:
            original_notes = original.get("notes", [])
            notes = [original_notes] if isinstance(original_notes, str) else list(original_notes or [])
        merged["notes"] = list(dict.fromkeys(note for note in notes if note))

        normalized.append(merged)
        if original:
            used_keys.add(key)

    for candidate in candidates:
        key = _candidate_key(candidate)
        if key not in used_keys:
            normalized.append(candidate)

    return normalized


def rank_with_llm(llm_client, prompt_name: str, context: dict, candidates: list[dict], llm_candidates: list[dict] | None = None):
    request_candidates = llm_candidates if llm_candidates is not None else candidates
    response = llm_client.rank_candidates(LLMRankRequest(prompt_name=prompt_name, context=context, candidates=request_candidates))
    response.rankings = _normalize_rankings(candidates, response.rankings)
    return response
