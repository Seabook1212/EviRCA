from __future__ import annotations

from datetime import datetime, timezone


def parse_time(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def to_iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def to_prometheus_time(value: datetime | str) -> str:
    if isinstance(value, str):
        return to_iso_z(parse_time(value))
    return to_iso_z(value)

