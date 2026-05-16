from __future__ import annotations

import pandas as pd

from rca_agent_skills.data_access.topology_loader import service_from_pod


def _normalize_trace_timestamps(raw_series: pd.Series) -> pd.Series:
    numeric_ts = pd.to_numeric(raw_series, errors="coerce")
    text_ts = pd.to_datetime(raw_series.where(numeric_ts.isna()), utc=True, errors="coerce")

    if not numeric_ts.notna().any():
        return text_ts

    non_null_numeric = numeric_ts.dropna()
    max_value = float(non_null_numeric.abs().max())
    if max_value > 1e14:
        unit = "us"
    elif max_value > 1e11:
        unit = "ms"
    else:
        unit = "s"

    numeric_as_dt = pd.Series(pd.NaT, index=raw_series.index, dtype="datetime64[ns, UTC]")
    numeric_as_dt.loc[non_null_numeric.index] = pd.to_datetime(non_null_numeric, unit=unit, utc=True, errors="coerce")
    return numeric_as_dt.fillna(text_ts)


def prepare_traces(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "trace_id",
                "span_id",
                "parent_span_id",
                "service",
                "operation",
                "duration",
                "span_kind",
                "status_code",
                "status",
                "peer_service",
                "pod",
            ]
        )
    prepared = frame.copy()
    if "timestamp" not in prepared.columns and "start_time" in prepared.columns:
        prepared["timestamp"] = prepared["start_time"]
    prepared["timestamp"] = _normalize_trace_timestamps(prepared["timestamp"])
    prepared["duration"] = pd.to_numeric(prepared["duration"], errors="coerce").fillna(0.0)
    prepared["service"] = prepared.get("service", prepared.get("pod", "")).fillna("")
    prepared["pod"] = prepared.get("pod", "").fillna("")
    prepared.loc[prepared["service"] == "", "service"] = prepared.loc[prepared["service"] == "", "pod"].map(service_from_pod)
    prepared["peer_service"] = prepared.get("peer_service", "").fillna("")
    prepared["status"] = prepared.get("status", "SUCCESS").fillna("SUCCESS").astype(str)
    prepared["status_code"] = prepared.get("status_code", "").fillna("").astype(str)
    prepared["span_kind"] = prepared.get("span_kind", "").fillna("").astype(str)
    prepared["operation"] = prepared.get("operation", "").fillna("").astype(str)
    return prepared


def build_trace_paths(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["trace_id", "path", "duration"])
    grouped = frame.sort_values(["trace_id", "timestamp"]).groupby("trace_id")
    rows = []
    for trace_id, group in grouped:
        services = [service for service in group["service"].tolist() if service]
        path = " > ".join(dict.fromkeys(services))
        rows.append({"trace_id": trace_id, "path": path, "duration": float(group["duration"].sum())})
    return pd.DataFrame(rows)
