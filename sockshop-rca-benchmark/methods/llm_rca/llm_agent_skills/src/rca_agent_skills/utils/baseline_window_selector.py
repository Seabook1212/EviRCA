from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class SelectedBaselineWindow:
    start: str
    end: str
    baseline_id: str | None
    strategy: str
    distance_seconds: float
    warnings: list[str]


def parse_utc_timestamp(value: str) -> datetime:
    normalized = value.strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _overlaps(
    start_a: datetime, end_a: datetime, start_b: datetime, end_b: datetime
) -> bool:
    return start_a < end_b and start_b < end_a


def select_baseline_window(
    abnormal_window: dict[str, str],
    baseline_config: dict[str, Any],
    namespace: str = "sock-shop",
) -> SelectedBaselineWindow:
    windows = baseline_config.get("baseline_windows", [])
    selection = baseline_config.get("selection", {})
    strategy = selection.get("strategy", "nearest_before")
    allow_after = bool(selection.get("allow_after", False))
    max_distance_hours = float(selection.get("max_distance_hours", 72))
    prefer_same_duration = bool(selection.get("prefer_same_duration", True))

    abnormal_start = parse_utc_timestamp(abnormal_window["start"])
    abnormal_end = parse_utc_timestamp(abnormal_window["end"])
    abnormal_duration = abs((abnormal_end - abnormal_start).total_seconds())

    candidates = []
    for item in windows:
        if item.get("namespace", namespace) != namespace:
            continue
        start = parse_utc_timestamp(item["start"])
        end = parse_utc_timestamp(item["end"])
        if _overlaps(start, end, abnormal_start, abnormal_end):
            continue
        if strategy == "nearest_before" and not allow_after and end > abnormal_start:
            continue

        if end <= abnormal_start:
            distance = abs((abnormal_start - end).total_seconds())
        else:
            distance = abs((start - abnormal_end).total_seconds())
        duration_gap = abs(abs((end - start).total_seconds()) - abnormal_duration)
        tie_breaker = duration_gap if prefer_same_duration else 0.0
        candidates.append((distance, tie_breaker, item, start, end))

    if not candidates:
        raise ValueError(
            f"No baseline window matched namespace={namespace!r} and abnormal_window={abnormal_window}"
        )

    distance, _, item, _, _ = min(candidates, key=lambda entry: (entry[0], entry[1]))
    warnings = []
    if distance > max_distance_hours * 3600:
        warnings.append(
            f"Selected baseline {item.get('id')} is {distance / 3600:.2f}h away from the abnormal window, exceeding max_distance_hours={max_distance_hours}."
        )

    return SelectedBaselineWindow(
        start=item["start"],
        end=item["end"],
        baseline_id=item.get("id"),
        strategy=strategy,
        distance_seconds=distance,
        warnings=warnings,
    )
