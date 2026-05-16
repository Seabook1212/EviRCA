import csv
import json
import pickle
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import Birch
from sklearn import preprocessing
from scipy.stats import pearsonr


# ===============================
# CONFIG
# ===============================

TELEMETRY_DAYS = ["2026_03_12"]

# Optional: provide a subset per day. If omitted, all fault experiments under
# the day's fault_run directory will be processed.
EXP_IDS_BY_DAY: dict[str, list[str]] = {}

TELEMETRY_METRICS_DIR_TEMPLATE = "../../../dataset_v2/telemetry/{telemetry_day}/metrics"
TELEMETRY_TRACES_DIR_TEMPLATE = "../../../dataset_v2/telemetry/{telemetry_day}/traces"
FAULT_RUN_ROOT_TEMPLATE = "../../../dataset_v2/fault_run_{day_suffix}"
NORMAL_RUN_ROOT_TEMPLATE = "../../../dataset_v2/normal_run_{day_suffix}"
OUTPUT_ROOT = "../../MicroRCA/data_v2/fault_metrics"
INDEX_CACHE_ROOT = "../../MicroRCA/data_v2/file_index_cache"

CONTAINER_OUTPUT = "container_metrics.csv"
HOST_OUTPUT = "host_metrics.csv"
SERVICE_OUTPUT = "service_level_metrics.csv"
GRAPH_OUTPUT = "attributed_graph.gpickle"
OUTPUT_SUBGRAPH_NAME = "anomalous_subgraph.gpickle"
OUTPUT_RANKING_NAME = "root_cause_ranking.csv"
DETAILS_OUTPUT_NAME = "microrca_accuracy_details.csv"
SUMMARY_OUTPUT_NAME = "microrca_accuracy_summary.csv"
ALL_DAYS_SUMMARY_OUTPUT_NAME = "microrca_accuracy_summary_all_days.csv"
ERROR_OUTPUT_NAME = "microrca_error.txt"
EXPECTED_EXCEPTIONS_PER_DAY = 64

# The raw KPI CSVs already provide container-style metrics, so the script uses
# them directly in memory and only writes filtered copies when requested.
SAVE_FILTERED_KPI_OUTPUTS = False
TRACE_CHUNKSIZE = 100_000
KPI_CHUNKSIZE = 100_000

DAY_CONFIG_OVERRIDES: dict[str, dict[str, str]] = {}
NORMAL_BASELINE_STRATEGY = "latest_preceding"
NORMAL_WINDOW_OFFSET_MINUTES = 5
NORMAL_WINDOW_DURATION_MINUTES = 5
FAULT_WINDOW_MINUTES = 5

ANOMALY_THRESHOLD = 1.5
ALPHA = 0.55
PAGERANK_C = 0.15
BIRCH_AD_THRESHOLD = 0.045
BIRCH_AD_THRESHOLD_PAYMENT_SHIPPING_NON_LATENCY = 0.02
SERVICE_SPECIFIC_BIRCH_THRESHOLDS = {
    "payment": {
        "non_latency": BIRCH_AD_THRESHOLD_PAYMENT_SHIPPING_NON_LATENCY,
    },
    "shipping": {
        "non_latency": BIRCH_AD_THRESHOLD_PAYMENT_SHIPPING_NON_LATENCY,
    },
}
SMOOTHING_WINDOW = 12
MICRORCA_MIN_CORRELATION = 0.01
# Keep DB/MQ toggles configurable because this Sock Shop benchmark labels them
# explicitly, even though the original MicroRCA service-level code excludes them.
PAPER_STYLE_INCLUDE_DB_SERVICES = True
PAPER_STYLE_INCLUDE_RABBITMQ = True
FALLBACK_TOP_ANOMALOUS_EDGES = 5

SERVICE_ALLOWLIST = {
    "catalogue",
    "carts",
    "orders",
    "user",
    "session-db",
    "front-end",
    "rabbitmq",
    "orders-db",
    "payment",
    "shipping",
    "queue-master",
    "user-db",
    "carts-db",
    "catalogue-db",
}

CONTAINER_SUM_METRICS = {"request_rate", "error_count", "network_rx", "network_tx", "restart_count"}
CONTAINER_MEAN_METRICS = {"cpu_usage_pct", "memory_usage_pct", "success_rate", "ready_ratio"}
CONTAINER_MAX_METRICS = {"latency_p99"}


@dataclass(frozen=True)
class NormalBaseline:
    run_id: str
    workload_start: pd.Timestamp
    workload_end: pd.Timestamp
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    midpoint: pd.Timestamp


@dataclass(frozen=True)
class DayContext:
    telemetry_day: str
    metrics_dir: Path
    traces_dir: Path
    fault_run_root: Path
    normal_run_root: Path
    output_day_dir: Path
    cache_dir: Path
    normal_baselines: list[NormalBaseline]


@dataclass(frozen=True)
class RunConfig:
    telemetry_day: str
    exp_id: str
    output_dir: Path
    analysis_start: pd.Timestamp
    analysis_end: pd.Timestamp
    workload_start: pd.Timestamp
    workload_end: pd.Timestamp
    fault_start: pd.Timestamp
    fault_end: pd.Timestamp
    normal_run_id: str
    normal_start: pd.Timestamp
    normal_end: pd.Timestamp
    ground_truth_service: str
    kpi_windows: list[tuple[pd.Timestamp, pd.Timestamp]]
    trace_windows: list[tuple[pd.Timestamp, pd.Timestamp]]
    normal_trace_windows: list[tuple[pd.Timestamp, pd.Timestamp]]
    metric_files: list[Path]
    trace_files: list[Path]
    normal_trace_files: list[Path]


def _read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_error_file(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(message, encoding="utf-8")


def _parse_utc_timestamp(value: str) -> pd.Timestamp:
    ts = pd.to_datetime(value, utc=True, errors="raise")
    return ts.tz_convert(None)


def _telemetry_day_to_suffix(telemetry_day: str) -> str:
    return datetime.strptime(telemetry_day, "%Y_%m_%d").strftime("%m%d")


def _resolve_day_suffix(telemetry_day: str, key: str) -> str:
    default_suffix = _telemetry_day_to_suffix(telemetry_day)
    return DAY_CONFIG_OVERRIDES.get(telemetry_day, {}).get(key, default_suffix)


def _load_normal_baselines(normal_run_root: Path) -> list[NormalBaseline]:
    metadata_paths = sorted(normal_run_root.glob("*/workload/workload_metadata.json"))
    if not metadata_paths:
        raise FileNotFoundError(f"No workload metadata found in {normal_run_root}")

    baselines: list[NormalBaseline] = []
    for path in metadata_paths:
        data = _read_json(path)
        start_raw = data.get("workload_start_time")
        end_raw = data.get("workload_end_time")
        if not start_raw or not end_raw:
            continue

        workload_start = _parse_utc_timestamp(start_raw)
        workload_end = _parse_utc_timestamp(end_raw)
        window_start = workload_start + pd.Timedelta(minutes=NORMAL_WINDOW_OFFSET_MINUTES)
        window_end = window_start + pd.Timedelta(minutes=NORMAL_WINDOW_DURATION_MINUTES)
        if window_end > workload_end:
            continue

        baselines.append(
            NormalBaseline(
                run_id=path.parents[1].name,
                workload_start=workload_start,
                workload_end=workload_end,
                window_start=window_start,
                window_end=window_end,
                midpoint=window_start + (window_end - window_start) / 2,
            )
        )

    if not baselines:
        raise ValueError(f"No valid normal baselines found in {normal_run_root}")
    return sorted(baselines, key=lambda item: item.window_start)


def _select_normal_baseline(normal_baselines: list[NormalBaseline], fault_start: pd.Timestamp) -> NormalBaseline:
    if NORMAL_BASELINE_STRATEGY == "earliest":
        return min(normal_baselines, key=lambda item: item.window_start)

    if NORMAL_BASELINE_STRATEGY == "latest_preceding":
        preceding = [item for item in normal_baselines if item.window_end <= fault_start]
        if preceding:
            return max(preceding, key=lambda item: item.window_end)
        return min(normal_baselines, key=lambda item: abs((fault_start - item.midpoint).total_seconds()))

    if NORMAL_BASELINE_STRATEGY == "nearest":
        return min(normal_baselines, key=lambda item: abs((fault_start - item.midpoint).total_seconds()))

    raise ValueError(f"Unsupported NORMAL_BASELINE_STRATEGY={NORMAL_BASELINE_STRATEGY!r}")


def _resolve_day_context(script_dir: Path, telemetry_day: str) -> DayContext:
    fault_day_suffix = _resolve_day_suffix(telemetry_day, "fault_run_day_suffix")
    normal_day_suffix = _resolve_day_suffix(telemetry_day, "normal_run_day_suffix")

    metrics_dir = (script_dir / TELEMETRY_METRICS_DIR_TEMPLATE.format(telemetry_day=telemetry_day)).resolve()
    traces_dir = (script_dir / TELEMETRY_TRACES_DIR_TEMPLATE.format(telemetry_day=telemetry_day)).resolve()
    fault_run_root = (script_dir / FAULT_RUN_ROOT_TEMPLATE.format(day_suffix=fault_day_suffix)).resolve()
    normal_run_root = (script_dir / NORMAL_RUN_ROOT_TEMPLATE.format(day_suffix=normal_day_suffix)).resolve()
    output_day_dir = (script_dir / OUTPUT_ROOT / telemetry_day).resolve()
    cache_dir = (script_dir / INDEX_CACHE_ROOT).resolve()

    if not metrics_dir.exists():
        raise FileNotFoundError(f"Metrics dir not found: {metrics_dir}")
    if not traces_dir.exists():
        raise FileNotFoundError(f"Traces dir not found: {traces_dir}")
    if not fault_run_root.exists():
        raise FileNotFoundError(f"Fault run root not found: {fault_run_root}")
    if not normal_run_root.exists():
        raise FileNotFoundError(f"Normal run root not found: {normal_run_root}")

    return DayContext(
        telemetry_day=telemetry_day,
        metrics_dir=metrics_dir,
        traces_dir=traces_dir,
        fault_run_root=fault_run_root,
        normal_run_root=normal_run_root,
        output_day_dir=output_day_dir,
        cache_dir=cache_dir,
        normal_baselines=_load_normal_baselines(normal_run_root),
    )


def _discover_experiment_ids(day_context: DayContext) -> list[str]:
    configured = EXP_IDS_BY_DAY.get(day_context.telemetry_day)
    if configured:
        return configured

    exp_ids = []
    for path in sorted(day_context.fault_run_root.iterdir()):
        metadata_path = path / "fault_info" / "fault_metadata.json"
        if path.is_dir() and metadata_path.exists():
            exp_ids.append(path.name)
    return exp_ids


def _load_fault_workload_window(day_context: DayContext, exp_id: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    metadata_path = day_context.fault_run_root / exp_id / "workload" / "workload_metadata.json"
    data = _read_json(metadata_path)
    start_raw = data.get("workload_start_time")
    end_raw = data.get("workload_end_time")
    if not start_raw or not end_raw:
        raise ValueError(f"workload_metadata.json missing workload_start_time/workload_end_time: {metadata_path}")
    return _parse_utc_timestamp(start_raw), _parse_utc_timestamp(end_raw)


def _load_fault_metadata(day_context: DayContext, exp_id: str) -> dict:
    metadata_path = day_context.fault_run_root / exp_id / "fault_info" / "fault_metadata.json"
    return _read_json(metadata_path)


def _fault_window_from_metadata(metadata: dict) -> tuple[pd.Timestamp, pd.Timestamp]:
    injection_info = metadata.get("injection_info")
    if not isinstance(injection_info, dict):
        raise ValueError("fault_metadata.json missing injection_info object")

    inject_start_raw = injection_info.get("inject_start")
    if not inject_start_raw:
        raise ValueError("fault_metadata.json missing injection_info.inject_start")

    start = _parse_utc_timestamp(inject_start_raw)
    end = start + pd.Timedelta(minutes=FAULT_WINDOW_MINUTES)
    return start, end


def _ground_truth_service_from_metadata(metadata: dict) -> str:
    injection_info = metadata.get("injection_info")
    if not isinstance(injection_info, dict):
        raise ValueError("fault_metadata.json missing injection_info object")
    service = injection_info.get("service")
    if not isinstance(service, str) or not service.strip():
        raise ValueError("fault_metadata.json missing injection_info.service")
    return service.strip()


def _pod_to_service_name(pod_name: str) -> str:
    if not isinstance(pod_name, str):
        return str(pod_name)

    parts = pod_name.split("-")
    if len(parts) >= 3 and re.fullmatch(r"[a-z0-9]{5,}", parts[-2]) and re.fullmatch(r"[a-z0-9]{5}", parts[-1]):
        return "-".join(parts[:-2])
    if len(parts) >= 2 and parts[-1].isdigit():
        return "-".join(parts[:-1])
    return pod_name


def _service_allowed_for_paper(service_name: str) -> bool:
    if service_name not in SERVICE_ALLOWLIST:
        return False
    if not PAPER_STYLE_INCLUDE_RABBITMQ and "rabbitmq" in service_name:
        return False
    if not PAPER_STYLE_INCLUDE_DB_SERVICES and "db" in service_name:
        return False
    return True


def _is_latency_like_fault(exp_id: str) -> bool:
    exp_id = str(exp_id or "")
    return any(token in exp_id for token in ("network_delay", "network_loss", "network_partition", "svc_latency"))


def _effective_birch_threshold(exp_id: str, ground_truth_service: str) -> float:
    service = str(ground_truth_service or "")
    service_thresholds = SERVICE_SPECIFIC_BIRCH_THRESHOLDS.get(service)
    if service_thresholds and not _is_latency_like_fault(exp_id):
        return float(service_thresholds.get("non_latency", BIRCH_AD_THRESHOLD))
    return BIRCH_AD_THRESHOLD


def _node_type(node_attrs: dict) -> str | None:
    return node_attrs.get("node_type") or node_attrs.get("type")


def _read_first_data_row(path: Path) -> list[str] | None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        return next(reader, None)


def _read_last_nonempty_line(path: Path) -> str:
    with path.open("rb") as handle:
        handle.seek(0, 2)
        position = handle.tell()
        if position == 0:
            return ""

        buffer = bytearray()
        while position > 0:
            position -= 1
            handle.seek(position)
            char = handle.read(1)
            if char == b"\n" and buffer:
                break
            if char != b"\n":
                buffer.extend(char)
        return buffer[::-1].decode("utf-8")


def _current_file_state(files: list[Path]) -> pd.DataFrame:
    rows = []
    for path in files:
        stat = path.stat()
        rows.append(
            {
                "file_name": path.name,
                "file_path": str(path),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return pd.DataFrame(rows).sort_values("file_name").reset_index(drop=True)


def _cache_is_current(cache_df: pd.DataFrame, files: list[Path]) -> bool:
    expected = _current_file_state(files)
    required_columns = ["file_name", "file_path", "size", "mtime_ns"]
    if cache_df.empty or not set(required_columns).issubset(cache_df.columns):
        return False

    cached = cache_df[required_columns].sort_values("file_name").reset_index(drop=True)
    expected = expected[required_columns].sort_values("file_name").reset_index(drop=True)
    if len(cached) != len(expected):
        return False
    return cached.equals(expected)


def _build_metric_file_index(files: list[Path]) -> pd.DataFrame:
    rows = []
    for path in files:
        first_row = _read_first_data_row(path)
        last_line = _read_last_nonempty_line(path)
        if not first_row or not last_line:
            continue
        last_row = next(csv.reader([last_line]), None)
        if not last_row:
            continue
        stat = path.stat()
        rows.append(
            {
                "file_name": path.name,
                "file_path": str(path),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "start": pd.to_datetime(first_row[0], errors="coerce"),
                "end": pd.to_datetime(last_row[0], errors="coerce"),
            }
        )
    return pd.DataFrame(rows)


def _build_trace_file_index(files: list[Path]) -> pd.DataFrame:
    rows = []
    for path in files:
        first_row = _read_first_data_row(path)
        last_line = _read_last_nonempty_line(path)
        if not first_row or not last_line:
            continue
        last_row = next(csv.reader([last_line]), None)
        if not last_row:
            continue
        stat = path.stat()
        rows.append(
            {
                "file_name": path.name,
                "file_path": str(path),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "start_us": pd.to_numeric(first_row[0], errors="coerce"),
                "end_us": pd.to_numeric(last_row[0], errors="coerce"),
            }
        )
    return pd.DataFrame(rows)


def _load_or_build_metric_file_index(cache_dir: Path, telemetry_day: str, metrics_dir: Path) -> pd.DataFrame:
    files = sorted(metrics_dir.glob("prometheus_metrics_KPI_*.csv"))
    if not files:
        raise FileNotFoundError(f"No KPI CSV files found in: {metrics_dir}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"metric_file_index_{telemetry_day}.csv"
    if cache_path.exists():
        cache_df = pd.read_csv(cache_path, parse_dates=["start", "end"])
        if _cache_is_current(cache_df, files):
            return cache_df

    index_df = _build_metric_file_index(files)
    index_df.to_csv(cache_path, index=False)
    return index_df


def _load_or_build_trace_file_index(cache_dir: Path, telemetry_day: str, traces_dir: Path) -> pd.DataFrame:
    files = sorted(traces_dir.glob("jaeger_traces_parsed_*.csv"))
    if not files:
        raise FileNotFoundError(f"No trace CSV files found in: {traces_dir}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"trace_file_index_{telemetry_day}.csv"
    if cache_path.exists():
        cache_df = pd.read_csv(cache_path)
        if _cache_is_current(cache_df, files):
            return cache_df

    index_df = _build_trace_file_index(files)
    index_df.to_csv(cache_path, index=False)
    return index_df


def select_overlapping_metric_files(
    metrics_dir: Path,
    cache_dir: Path,
    telemetry_day: str,
    windows: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> list[Path]:
    index_df = _load_or_build_metric_file_index(cache_dir, telemetry_day, metrics_dir)
    mask = pd.Series(False, index=index_df.index)
    for start, end in windows:
        mask |= (index_df["end"] >= start) & (index_df["start"] <= end)
    overlap = index_df[mask].copy()
    if overlap.empty:
        raise ValueError(f"No metric files overlap the requested KPI windows: {windows}")
    return [Path(path) for path in overlap.sort_values("start")["file_path"].tolist()]


def select_overlapping_trace_files(
    traces_dir: Path,
    cache_dir: Path,
    telemetry_day: str,
    windows: list[tuple[int, int]],
) -> list[Path]:
    index_df = _load_or_build_trace_file_index(cache_dir, telemetry_day, traces_dir)
    mask = pd.Series(False, index=index_df.index)
    for start, end in windows:
        mask |= (index_df["end_us"] >= start) & (index_df["start_us"] <= end)
    overlap = index_df[mask].copy()
    if overlap.empty:
        raise ValueError(f"No trace files overlap the requested trace windows: {windows}")
    return [Path(path) for path in overlap.sort_values("start_us")["file_path"].tolist()]


def _build_timestamp_window_mask(series: pd.Series, windows: list[tuple[pd.Timestamp, pd.Timestamp]]) -> pd.Series:
    mask = pd.Series(False, index=series.index)
    for start, end in windows:
        mask |= (series >= start) & (series <= end)
    return mask


def _build_numeric_window_mask(series: pd.Series, windows: list[tuple[int, int]]) -> pd.Series:
    mask = pd.Series(False, index=series.index)
    for start, end in windows:
        mask |= (series >= start) & (series <= end)
    return mask


def load_filtered_kpi_metrics(metric_files: list[Path], windows: list[tuple[pd.Timestamp, pd.Timestamp]]) -> pd.DataFrame:
    filtered_chunks: list[pd.DataFrame] = []
    for path in metric_files:
        for chunk in pd.read_csv(path, usecols=["timestamp", "pod", "metric", "value"], chunksize=KPI_CHUNKSIZE):
            chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], errors="coerce")
            chunk = chunk.dropna(subset=["timestamp"])
            chunk = chunk[_build_timestamp_window_mask(chunk["timestamp"], windows)]
            if chunk.empty:
                continue
            chunk["value"] = pd.to_numeric(chunk["value"], errors="coerce")
            chunk = chunk.dropna(subset=["value"])
            if chunk.empty:
                continue
            chunk["service"] = chunk["pod"].astype(str).map(_pod_to_service_name)
            chunk = chunk[chunk["service"].isin(SERVICE_ALLOWLIST)].copy()
            if not chunk.empty:
                filtered_chunks.append(chunk)

    if not filtered_chunks:
        raise ValueError(f"No KPI rows found in requested windows: {windows}")

    return pd.concat(filtered_chunks, ignore_index=True).sort_values(["timestamp", "service", "metric", "pod"])


def build_container_metric_view(kpi_df: pd.DataFrame) -> pd.DataFrame:
    aggregated_frames: list[pd.DataFrame] = []

    if CONTAINER_SUM_METRICS:
        sum_df = (
            kpi_df[kpi_df["metric"].isin(CONTAINER_SUM_METRICS)]
            .groupby(["timestamp", "service", "metric"], as_index=False)["value"]
            .sum()
        )
        aggregated_frames.append(sum_df)

    if CONTAINER_MEAN_METRICS:
        mean_df = (
            kpi_df[kpi_df["metric"].isin(CONTAINER_MEAN_METRICS)]
            .groupby(["timestamp", "service", "metric"], as_index=False)["value"]
            .mean()
        )
        aggregated_frames.append(mean_df)

    if CONTAINER_MAX_METRICS:
        max_df = (
            kpi_df[kpi_df["metric"].isin(CONTAINER_MAX_METRICS)]
            .groupby(["timestamp", "service", "metric"], as_index=False)["value"]
            .max()
        )
        aggregated_frames.append(max_df)

    if not aggregated_frames:
        return pd.DataFrame(columns=["timestamp", "service", "metric", "value"])

    return pd.concat(aggregated_frames, ignore_index=True).sort_values(["timestamp", "service", "metric"])


def build_host_metric_view() -> pd.DataFrame:
    return pd.DataFrame(columns=["timestamp", "host", "metric", "value"])


def maybe_write_filtered_kpi_outputs(output_dir: Path, container_df: pd.DataFrame, host_df: pd.DataFrame) -> None:
    if not SAVE_FILTERED_KPI_OUTPUTS:
        return

    container_path = output_dir / CONTAINER_OUTPUT
    host_path = output_dir / HOST_OUTPUT
    output_dir.mkdir(parents=True, exist_ok=True)
    container_df.to_csv(container_path, index=False)
    host_df.to_csv(host_path, index=False)


def parse_tags(tags_json_str):
    if pd.isna(tags_json_str):
        return {}
    try:
        parsed = json.loads(tags_json_str)
    except Exception:
        return {}

    if isinstance(parsed, dict):
        return parsed
    if isinstance(parsed, list):
        result = {}
        for item in parsed:
            if isinstance(item, dict) and "key" in item and "value" in item:
                result[str(item["key"])] = item["value"]
        return result
    return {}


def normalize_service_name(raw_value):
    if raw_value is None:
        return None
    value = str(raw_value).strip()
    if not value:
        return None
    if "://" in value:
        value = value.split("://", 1)[1]
    value = value.split("/", 1)[0]
    value = value.split(":", 1)[0]
    if ".svc." in value:
        value = value.split(".svc.", 1)[0]
    return value


def get_destination_from_row(row, tags):
    direct_candidates = [
        row.get("peer_service"),
        tags.get("peer.service"),
        tags.get("peer.address"),
        tags.get("net.peer.name"),
        tags.get("client.name"),
        tags.get("service"),
    ]
    for value in direct_candidates:
        normalized = normalize_service_name(value)
        if normalized:
            return normalized
    return None


def build_service_level_metrics(trace_files: list[Path], windows: list[tuple[pd.Timestamp, pd.Timestamp]]) -> pd.DataFrame:
    windows_us = [(int(start.value // 1_000), int(end.value // 1_000)) for start, end in windows]
    span_dict: dict[str, dict] = {}
    records = []
    seen = set()
    use_cols = [
        "timestamp",
        "trace_id",
        "span_id",
        "parent_span_id",
        "service",
        "duration",
        "span_kind",
        "peer_service",
        "tags_json",
    ]

    for file in trace_files:
        for chunk in pd.read_csv(file, usecols=["span_id", "service", "span_kind", "tags_json"], chunksize=TRACE_CHUNKSIZE):
            if chunk.empty:
                continue
            chunk = chunk.drop_duplicates(subset=["span_id"]).copy()
            for row in chunk.itertuples(index=False):
                if row.span_id in span_dict:
                    continue
                tags = parse_tags(row.tags_json)
                span_dict[row.span_id] = {
                    "service": row.service,
                    "span_kind": row.span_kind or tags.get("span.kind") or "",
                }

    for file in trace_files:
        for chunk in pd.read_csv(file, usecols=use_cols, chunksize=TRACE_CHUNKSIZE):
            chunk["timestamp"] = pd.to_numeric(chunk["timestamp"], errors="coerce")
            chunk = chunk.dropna(subset=["timestamp"])
            chunk = chunk[_build_numeric_window_mask(chunk["timestamp"], windows_us)]
            if chunk.empty:
                continue

            chunk = chunk.drop_duplicates(subset=["span_id"]).copy()
            chunk["duration"] = pd.to_numeric(chunk["duration"], errors="coerce")
            chunk = chunk.dropna(subset=["duration", "service"])
            if chunk.empty:
                continue
            chunk["span_kind"] = chunk["span_kind"].fillna("")

            for row in chunk.itertuples(index=False):
                tags = parse_tags(row.tags_json)
                span_kind = row.span_kind or tags.get("span.kind") or ""
                source_service = normalize_service_name(row.service)
                parent_id = row.parent_span_id if pd.notna(row.parent_span_id) and row.parent_span_id != "" else None

                destination_service = None
                if span_kind == "client":
                    destination_service = get_destination_from_row(row._asdict(), tags)
                elif parent_id and parent_id in span_dict:
                    parent = span_dict[parent_id]
                    parent_service = normalize_service_name(parent.get("service"))
                    parent_kind = parent.get("span_kind")
                    if parent_service != source_service and (
                        span_kind == "server" or parent_kind == "client" or not span_kind
                    ):
                        source_service = parent_service
                        destination_service = normalize_service_name(row.service)

                if not source_service or not destination_service:
                    continue
                if source_service not in SERVICE_ALLOWLIST or destination_service not in SERVICE_ALLOWLIST:
                    continue
                if source_service == destination_service:
                    continue

                timestamp = pd.to_datetime(row.timestamp, unit="us", errors="coerce")
                if pd.isna(timestamp):
                    continue
                dedupe_key = (row.trace_id, source_service, destination_service, timestamp)
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)

                records.append(
                    {
                        "timestamp": timestamp,
                        "source": source_service,
                        "destination": destination_service,
                        "response_time": row.duration / 1000.0,
                    }
                )

    service_df = pd.DataFrame(records)
    if service_df.empty:
        raise ValueError(f"No service edges were derived from traces in requested windows: {windows}")
    return service_df.sort_values(["timestamp", "source", "destination"])


def build_invocation_latency_matrix(
    service_df: pd.DataFrame,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    bucket_frequency: str = "5s",
) -> pd.DataFrame:
    window_df = service_df[(service_df["timestamp"] >= window_start) & (service_df["timestamp"] <= window_end)].copy()
    if window_df.empty:
        raise ValueError(f"No service invocations found in workload window {window_start} -> {window_end}")

    window_df["timestamp"] = window_df["timestamp"].dt.floor(bucket_frequency)
    window_df["edge"] = window_df["source"] + "_" + window_df["destination"]
    latency_df = (
        window_df.groupby(["timestamp", "edge"])["response_time"]
        .median()
        .unstack(fill_value=0.0)
        .sort_index()
    )

    full_index = pd.date_range(
        start=window_start.floor(bucket_frequency),
        end=window_end.ceil(bucket_frequency),
        freq=bucket_frequency,
    )
    latency_df = latency_df.reindex(full_index, fill_value=0.0)
    latency_df = latency_df.reset_index().rename(columns={"index": "timestamp"})
    return latency_df


def birch_ad_with_smoothing(latency_df: pd.DataFrame, threshold: float) -> list[str]:
    anomalies = []
    for column in latency_df.columns:
        if column == "timestamp" or "Unnamed" in column:
            continue
        if not PAPER_STYLE_INCLUDE_RABBITMQ and "rabbitmq" in column:
            continue
        if not PAPER_STYLE_INCLUDE_DB_SERVICES and "db" in column:
            continue

        latency = latency_df[column].rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()
        x = np.nan_to_num(latency.to_numpy(dtype=float), nan=0.0)
        if len(x) < 2 or np.allclose(x, x[0]):
            continue

        normalized_x = preprocessing.normalize([x])
        X = normalized_x.reshape(-1, 1)
        birch = Birch(branching_factor=50, n_clusters=None, threshold=threshold, compute_labels=True)
        labels = birch.fit_predict(X)
        if np.unique(labels).size > 1:
            anomalies.append(column)
    return anomalies


def _safe_positive_corr(series_a: pd.Series | None, series_b: pd.Series | None) -> float:
    if series_a is None or series_b is None:
        return 0.0
    aligned = pd.concat([series_a.rename("a"), series_b.rename("b")], axis=1).dropna()
    if aligned.empty or len(aligned) < 2:
        return 0.0
    corr = aligned["a"].corr(aligned["b"])
    if pd.isna(corr):
        return 0.0
    return max(float(corr), 0.0)


def build_paper_attributed_graph(service_df: pd.DataFrame) -> nx.DiGraph:
    graph = nx.DiGraph()
    call_pairs = service_df[["source", "destination"]].drop_duplicates()
    for row in call_pairs.itertuples(index=False):
        if not _service_allowed_for_paper(row.source) or not _service_allowed_for_paper(row.destination):
            continue
        graph.add_edge(row.source, row.destination)

    for node in graph.nodes:
        graph.nodes[node]["type"] = "service"
        graph.nodes[node]["node_type"] = "service"
    return graph


def _paper_container_metric_candidates(container_df: pd.DataFrame, service_name: str) -> pd.DataFrame:
    service_metrics = container_df[container_df["service"] == service_name].copy()
    if service_metrics.empty:
        return pd.DataFrame(columns=["ctn_cpu", "ctn_network", "ctn_memory"])

    pivot = service_metrics.pivot(index="timestamp", columns="metric", values="value").sort_index()
    paper_df = pd.DataFrame(index=pivot.index)
    if "cpu_usage_pct" in pivot.columns:
        paper_df["ctn_cpu"] = pivot["cpu_usage_pct"]
    if "memory_usage_pct" in pivot.columns:
        paper_df["ctn_memory"] = pivot["memory_usage_pct"]
    if "network_rx" in pivot.columns or "network_tx" in pivot.columns:
        paper_df["ctn_network"] = pivot.get("network_rx", 0.0) + pivot.get("network_tx", 0.0)
    return paper_df


def _paper_service_personalization(
    service_name: str,
    anomaly_graph: nx.DiGraph,
    baseline_series: pd.Series,
    container_df: pd.DataFrame,
) -> float:
    metric_df = _paper_container_metric_candidates(container_df, service_name)
    max_corr = MICRORCA_MIN_CORRELATION
    for column in metric_df.columns:
        max_corr = max(max_corr, abs(_safe_positive_corr(baseline_series, metric_df[column])))

    degree = anomaly_graph.degree(service_name)
    if degree == 0:
        return max_corr

    edge_weight_sum = 0.0
    edge_weight_count = 0
    for _, _, edge_data in anomaly_graph.in_edges(service_name, data=True):
        edge_weight_sum += float(edge_data.get("weight", 0.0))
        edge_weight_count += 1
    for _, neighbor, edge_data in anomaly_graph.out_edges(service_name, data=True):
        if _node_type(anomaly_graph.nodes[neighbor]) == "service":
            edge_weight_sum += float(edge_data.get("weight", 0.0))
            edge_weight_count += 1

    if edge_weight_count == 0:
        return max_corr / degree

    average_edge_weight = edge_weight_sum / edge_weight_count
    return (average_edge_weight * max_corr) / degree


def microrca_rank_services(
    service_df: pd.DataFrame,
    container_df: pd.DataFrame,
    workload_start: pd.Timestamp,
    workload_end: pd.Timestamp,
    alpha: float,
    birch_threshold: float,
) -> tuple[list[tuple[str, float]], pd.DataFrame, nx.DiGraph, nx.DiGraph]:
    latency_df = build_invocation_latency_matrix(service_df, workload_start, workload_end)
    anomalies = birch_ad_with_smoothing(latency_df, birch_threshold)
    if not anomalies:
        raise ValueError("No anomalous invocations were detected by the Birch-based anomaly detector.")

    graph = build_paper_attributed_graph(service_df)
    baseline_df = pd.DataFrame(index=latency_df["timestamp"])
    anomalous_edges = set()
    anomalous_services = []

    for anomaly in anomalies:
        if anomaly not in latency_df.columns:
            continue
        source, destination = anomaly.split("_", 1)
        if not _service_allowed_for_paper(source) or not _service_allowed_for_paper(destination):
            continue
        anomalous_edges.add((source, destination))
        anomalous_services.append(destination)
        baseline_df[destination] = latency_df[anomaly]

    anomalous_services = sorted(set(anomalous_services))
    if not anomalous_services:
        raise ValueError("Birch reported anomalous invocations, but none mapped to supported services.")

    anomaly_graph = nx.DiGraph()
    personalization = {service: 0.0 for service in anomalous_services if service in graph.nodes}

    for service in anomalous_services:
        if service not in graph:
            continue

        for source, destination in graph.in_edges(service):
            if (source, destination) in anomalous_edges:
                weight = alpha
            else:
                weight = _safe_positive_corr(
                    baseline_df.get(destination),
                    latency_df.get(f"{source}_{destination}"),
                )
            anomaly_graph.add_edge(source, destination, weight=round(weight, 3))
            anomaly_graph.nodes[source]["type"] = graph.nodes[source]["type"]
            anomaly_graph.nodes[source]["node_type"] = _node_type(graph.nodes[source])
            anomaly_graph.nodes[destination]["type"] = graph.nodes[destination]["type"]
            anomaly_graph.nodes[destination]["node_type"] = _node_type(graph.nodes[destination])

        for source, destination in graph.out_edges(service):
            if (source, destination) in anomalous_edges:
                weight = alpha
            else:
                weight = _safe_positive_corr(
                    baseline_df.get(source),
                    latency_df.get(f"{source}_{destination}"),
                )
            anomaly_graph.add_edge(source, destination, weight=round(weight, 3))
            anomaly_graph.nodes[source]["type"] = graph.nodes[source]["type"]
            anomaly_graph.nodes[source]["node_type"] = _node_type(graph.nodes[source])
            anomaly_graph.nodes[destination]["type"] = graph.nodes[destination]["type"]
            anomaly_graph.nodes[destination]["node_type"] = _node_type(graph.nodes[destination])

    for service in anomalous_services:
        if service not in anomaly_graph or service not in baseline_df:
            continue
        personalization[service] = _paper_service_personalization(
            service_name=service,
            anomaly_graph=anomaly_graph,
            baseline_series=baseline_df[service],
            container_df=container_df,
        )

    reversed_graph = anomaly_graph.reverse(copy=True)
    anomaly_score = nx.pagerank(
        reversed_graph,
        alpha=1 - PAGERANK_C,
        personalization=personalization,
        max_iter=10000,
        weight="weight",
    )
    ranked = sorted(anomaly_score.items(), key=lambda item: item[1], reverse=True)
    return ranked, latency_df, graph, anomaly_graph


def build_attributed_graph(service_df: pd.DataFrame, host_df: pd.DataFrame) -> nx.DiGraph:
    graph = nx.DiGraph()

    services = set(service_df["source"]).union(set(service_df["destination"])) if not service_df.empty else set()
    for service in services:
        graph.add_node(service, node_type="service")

    if not host_df.empty and "host" in host_df.columns:
        for host in host_df["host"].dropna().unique():
            graph.add_node(host, node_type="host")

    for _, row in service_df[["source", "destination"]].drop_duplicates().iterrows():
        graph.add_edge(row["source"], row["destination"], edge_type="call")

    return graph


def compute_window_mean(service_df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    window_df = service_df[(service_df["timestamp"] >= start) & (service_df["timestamp"] <= end)]
    return (
        window_df.groupby(["source", "destination"])["response_time"]
        .mean()
        .reset_index()
        .rename(columns={"response_time": "mean_rt"})
    )


def build_anomalous_subgraph(
    service_df: pd.DataFrame,
    graph: nx.DiGraph,
    normal_start: pd.Timestamp,
    normal_end: pd.Timestamp,
    fault_start: pd.Timestamp,
    fault_end: pd.Timestamp,
    anomaly_threshold: float,
) -> nx.DiGraph:
    anomalous_edges = compute_anomalous_edge_summary(
        service_df=service_df,
        normal_start=normal_start,
        normal_end=normal_end,
        fault_start=fault_start,
        fault_end=fault_end,
    )
    anomalous_edges = anomalous_edges[anomalous_edges["ratio"] > anomaly_threshold].copy()
    if anomalous_edges.empty:
        raise ValueError(
            f"No service edges exceeded ANOMALY_THRESHOLD={anomaly_threshold}. "
            "Try lowering the threshold or using a more comparable normal window."
        )
    return build_subgraph_from_edge_summary(graph, anomalous_edges)


def compute_anomalous_edge_summary(
    service_df: pd.DataFrame,
    normal_start: pd.Timestamp,
    normal_end: pd.Timestamp,
    fault_start: pd.Timestamp,
    fault_end: pd.Timestamp,
) -> pd.DataFrame:
    normal_mean = compute_window_mean(service_df, normal_start, normal_end)
    fault_mean = compute_window_mean(service_df, fault_start, fault_end)

    if normal_mean.empty:
        raise ValueError("No service-level rows were found in the selected normal window.")
    if fault_mean.empty:
        raise ValueError("No service-level rows were found in the selected fault window.")

    merged = normal_mean.merge(
        fault_mean,
        on=["source", "destination"],
        suffixes=("_normal", "_fault"),
    )
    if merged.empty:
        raise ValueError("Normal and fault windows do not share any service call edges.")

    merged["delta"] = merged["mean_rt_fault"] - merged["mean_rt_normal"]
    merged["ratio"] = np.where(
        merged["mean_rt_normal"] > 0,
        merged["mean_rt_fault"] / merged["mean_rt_normal"],
        np.where(merged["mean_rt_fault"] > 0, np.inf, 1.0),
    )
    return merged.sort_values(
        ["ratio", "delta", "mean_rt_fault"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def build_subgraph_from_edge_summary(graph: nx.DiGraph, anomalous_edges: pd.DataFrame) -> nx.DiGraph:
    anomalous_services = set(anomalous_edges["destination"])
    rt_a_dict = {}
    for service in anomalous_services:
        service_edges = anomalous_edges[anomalous_edges["destination"] == service]
        rt_a_dict[service] = service_edges["mean_rt_fault"].mean()

    subgraph = nx.DiGraph()
    for service in anomalous_services:
        subgraph.add_node(service, node_type="service", type="service", rt_a=rt_a_dict[service])
        if service not in graph:
            continue
        for succ in graph.successors(service):
            attrs = dict(graph.nodes[succ])
            attrs.setdefault("node_type", _node_type(attrs) or "service")
            attrs.setdefault("type", attrs["node_type"])
            subgraph.add_node(succ, **attrs)
            subgraph.add_edge(service, succ)
        for pred in graph.predecessors(service):
            attrs = dict(graph.nodes[pred])
            attrs.setdefault("node_type", _node_type(attrs) or "service")
            attrs.setdefault("type", attrs["node_type"])
            subgraph.add_node(pred, **attrs)
            subgraph.add_edge(pred, service)

    return subgraph


def compute_fault_only_edge_summary(
    service_df: pd.DataFrame,
    fault_start: pd.Timestamp,
    fault_end: pd.Timestamp,
) -> pd.DataFrame:
    fault_mean = compute_window_mean(service_df, fault_start, fault_end)
    if fault_mean.empty:
        raise ValueError("No service-level rows were found in the selected fault window.")
    fault_mean["ratio"] = np.inf
    fault_mean["delta"] = fault_mean["mean_rt"]
    return fault_mean.rename(columns={"mean_rt": "mean_rt_fault"}).sort_values(
        ["mean_rt_fault", "delta"],
        ascending=[False, False],
    ).reset_index(drop=True)


def build_anomalous_subgraph_with_fallback(
    service_df: pd.DataFrame,
    graph: nx.DiGraph,
    normal_start: pd.Timestamp,
    normal_end: pd.Timestamp,
    fault_start: pd.Timestamp,
    fault_end: pd.Timestamp,
    anomaly_threshold: float,
    fallback_top_edges: int,
) -> nx.DiGraph:
    try:
        anomalous_edges = compute_anomalous_edge_summary(
            service_df=service_df,
            normal_start=normal_start,
            normal_end=normal_end,
            fault_start=fault_start,
            fault_end=fault_end,
        )
        threshold_edges = anomalous_edges[anomalous_edges["ratio"] > anomaly_threshold].copy()
        chosen_edges = threshold_edges if not threshold_edges.empty else anomalous_edges.head(fallback_top_edges).copy()
    except ValueError:
        anomalous_edges = compute_fault_only_edge_summary(
            service_df=service_df,
            fault_start=fault_start,
            fault_end=fault_end,
        )
        chosen_edges = anomalous_edges.head(fallback_top_edges).copy()

    if chosen_edges.empty:
        raise ValueError("Unable to derive any anomalous service edges for the fallback subgraph.")
    return build_subgraph_from_edge_summary(graph, chosen_edges)


def safe_corr(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if len(x) < 2 or len(y) < 2:
        return 0.0
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    try:
        corr = pearsonr(x, y)[0]
    except Exception:
        return 0.0
    if not np.isfinite(corr):
        return 0.0
    return abs(corr)


def build_service_rt_series(service_df: pd.DataFrame, fault_start: pd.Timestamp, fault_end: pd.Timestamp) -> dict[str, pd.Series]:
    fault_df = service_df[(service_df["timestamp"] >= fault_start) & (service_df["timestamp"] <= fault_end)].copy()
    grouped = (
        fault_df.groupby(["timestamp", "destination"])["response_time"]
        .mean()
        .reset_index()
        .rename(columns={"destination": "service"})
    )

    series_map = {}
    for service, group in grouped.groupby("service"):
        series_map[service] = group.sort_values("timestamp").set_index("timestamp")["response_time"]
    return series_map


def localize_root_cause(
    subgraph: nx.DiGraph,
    service_df: pd.DataFrame,
    container_df: pd.DataFrame,
    host_df: pd.DataFrame,
    fault_start: pd.Timestamp,
    fault_end: pd.Timestamp,
    alpha: float,
    pagerank_c: float,
) -> list[tuple[str, float]]:
    service_rt_series = build_service_rt_series(service_df, fault_start, fault_end)

    for u, v in subgraph.edges():
        if "rt_a" in subgraph.nodes[u]:
            if _node_type(subgraph.nodes[u]) == "service" and _node_type(subgraph.nodes[v]) == "service":
                subgraph[u][v]["weight"] = alpha
                continue

            if _node_type(subgraph.nodes[v]) == "host" and not host_df.empty:
                rt_series = service_rt_series.get(u)
                if rt_series is None or rt_series.empty:
                    subgraph[u][v]["weight"] = 0.0
                    continue

                host_series = host_df[(host_df["host"] == v)].pivot(index="timestamp", columns="metric", values="value")
                max_corr = 0.0
                for col in host_series.columns:
                    aligned = pd.concat([rt_series.rename("rt"), host_series[col].rename("metric")], axis=1).dropna()
                    if aligned.empty:
                        continue
                    max_corr = max(max_corr, safe_corr(aligned["metric"].values, aligned["rt"].values))
                subgraph[u][v]["weight"] = max_corr
        else:
            subgraph[u][v]["weight"] = 0.1

    anomaly_scores = {}
    for node in subgraph.nodes():
        if _node_type(subgraph.nodes[node]) != "service" or "rt_a" not in subgraph.nodes[node]:
            anomaly_scores[node] = 0.0
            continue

        rt_series = service_rt_series.get(node)
        if rt_series is None or rt_series.empty:
            anomaly_scores[node] = 0.0
            continue

        service_metrics = container_df[container_df["service"] == node]
        if service_metrics.empty:
            anomaly_scores[node] = 0.0
            continue

        container_series = service_metrics.pivot(index="timestamp", columns="metric", values="value")
        max_corr = 0.0
        for col in container_series.columns:
            aligned = pd.concat([rt_series.rename("rt"), container_series[col].rename("metric")], axis=1).dropna()
            if aligned.empty:
                continue
            max_corr = max(max_corr, safe_corr(aligned["metric"].values, aligned["rt"].values))

        preds = list(subgraph.predecessors(node))
        avg_weight = (
            np.mean([subgraph[p][node]["weight"] for p in preds if "weight" in subgraph[p][node]])
            if preds
            else 0.0
        )
        anomaly_scores[node] = avg_weight * max_corr

    graph_pr = subgraph.reverse(copy=True)
    personalization = {node: anomaly_scores.get(node, 0.0) for node in graph_pr.nodes()}
    personalization = {
        node: (value if np.isfinite(value) and value > 0 else 0.0)
        for node, value in personalization.items()
    }
    if sum(personalization.values()) == 0:
        personalization = {node: 1.0 for node in graph_pr.nodes()}

    pr = nx.pagerank(
        graph_pr,
        alpha=1 - pagerank_c,
        personalization=personalization,
        weight="weight",
    )
    service_pr = {
        node: score
        for node, score in pr.items()
        if _node_type(subgraph.nodes[node]) == "service"
    }
    return sorted(service_pr.items(), key=lambda item: item[1], reverse=True)


def _resolve_run_config(day_context: DayContext, exp_id: str) -> RunConfig:
    metadata = _load_fault_metadata(day_context, exp_id)
    workload_start, workload_end = _load_fault_workload_window(day_context, exp_id)
    fault_start, fault_end = _fault_window_from_metadata(metadata)
    normal_baseline = _select_normal_baseline(day_context.normal_baselines, fault_start)
    ground_truth_service = _ground_truth_service_from_metadata(metadata)

    # MicroRCA detects anomalies directly from the workload-period latency series
    # rather than comparing separate normal and fault windows.
    kpi_windows = [(workload_start, workload_end)]
    trace_windows = [(workload_start, workload_end)]
    analysis_start = workload_start
    analysis_end = workload_end

    metric_files = select_overlapping_metric_files(
        metrics_dir=day_context.metrics_dir,
        cache_dir=day_context.cache_dir,
        telemetry_day=day_context.telemetry_day,
        windows=kpi_windows,
    )
    trace_files = select_overlapping_trace_files(
        traces_dir=day_context.traces_dir,
        cache_dir=day_context.cache_dir,
        telemetry_day=day_context.telemetry_day,
        windows=[(int(start.value // 1_000), int(end.value // 1_000)) for start, end in trace_windows],
    )
    normal_trace_windows = [(normal_baseline.window_start, normal_baseline.window_end)]
    normal_trace_files = select_overlapping_trace_files(
        traces_dir=day_context.traces_dir,
        cache_dir=day_context.cache_dir,
        telemetry_day=day_context.telemetry_day,
        windows=[(int(start.value // 1_000), int(end.value // 1_000)) for start, end in normal_trace_windows],
    )

    return RunConfig(
        telemetry_day=day_context.telemetry_day,
        exp_id=exp_id,
        output_dir=day_context.output_day_dir / exp_id,
        analysis_start=analysis_start,
        analysis_end=analysis_end,
        workload_start=workload_start,
        workload_end=workload_end,
        fault_start=fault_start,
        fault_end=fault_end,
        normal_run_id=normal_baseline.run_id,
        normal_start=normal_baseline.window_start,
        normal_end=normal_baseline.window_end,
        ground_truth_service=ground_truth_service,
        kpi_windows=kpi_windows,
        trace_windows=trace_windows,
        normal_trace_windows=normal_trace_windows,
        metric_files=metric_files,
        trace_files=trace_files,
        normal_trace_files=normal_trace_files,
    )


def _evaluate_topk(ranked_services: list[str], ground_truth_service: str, k: int) -> bool:
    return ground_truth_service in ranked_services[:k]


def _topk_services(ranked_services: list[str], k: int) -> str | None:
    topk = ranked_services[:k]
    return "|".join(topk) if topk else None


def _should_use_fallback(exc: Exception) -> bool:
    if not isinstance(exc, ValueError):
        return False
    message = str(exc)
    return (
        "No anomalous invocations were detected by the Birch-based anomaly detector." in message
        or "Birch reported anomalous invocations, but none mapped to supported services." in message
    )


def _load_fallback_service_df(run_config: RunConfig) -> pd.DataFrame:
    pre_fault_end = min(run_config.fault_start, run_config.workload_end)
    pre_fault_windows = []
    if pre_fault_end > run_config.workload_start:
        pre_fault_windows.append((run_config.workload_start, pre_fault_end))

    windows = pre_fault_windows + [(run_config.fault_start, run_config.fault_end)]
    trace_files = run_config.trace_files

    if not pre_fault_windows:
        windows = run_config.normal_trace_windows + [(run_config.fault_start, run_config.fault_end)]
        trace_files = sorted(set(run_config.trace_files + run_config.normal_trace_files))

    return build_service_level_metrics(trace_files, windows)


def run_single_experiment(run_config: RunConfig) -> dict:
    start_time = time.perf_counter()
    run_config.output_dir.mkdir(parents=True, exist_ok=True)

    container_output_path = run_config.output_dir / CONTAINER_OUTPUT
    host_output_path = run_config.output_dir / HOST_OUTPUT
    service_output_path = run_config.output_dir / SERVICE_OUTPUT
    graph_output_path = run_config.output_dir / GRAPH_OUTPUT
    output_subgraph_path = run_config.output_dir / OUTPUT_SUBGRAPH_NAME
    output_ranking_path = run_config.output_dir / OUTPUT_RANKING_NAME
    error_output_path = run_config.output_dir / ERROR_OUTPUT_NAME

    try:
        kpi_df = load_filtered_kpi_metrics(run_config.metric_files, run_config.kpi_windows)
        container_df = build_container_metric_view(kpi_df)
        host_df = build_host_metric_view()
        maybe_write_filtered_kpi_outputs(run_config.output_dir, container_df, host_df)

        service_df = build_service_level_metrics(run_config.trace_files, run_config.trace_windows)
        service_df.to_csv(service_output_path, index=False)

        ranking_method = "paper_birch"
        try:
            ranked, _, graph, subgraph = microrca_rank_services(
                service_df=service_df,
                container_df=container_df,
                workload_start=run_config.workload_start,
                workload_end=run_config.workload_end,
                alpha=ALPHA,
                birch_threshold=_effective_birch_threshold(
                    exp_id=run_config.exp_id,
                    ground_truth_service=run_config.ground_truth_service,
                ),
            )
        except Exception as exc:
            if not _should_use_fallback(exc):
                raise

            ranking_method = "window_ratio_fallback"
            fallback_service_df = _load_fallback_service_df(run_config)
            graph = build_paper_attributed_graph(fallback_service_df)

            pre_fault_start = run_config.workload_start
            pre_fault_end = min(run_config.fault_start, run_config.workload_end)
            if pre_fault_end <= pre_fault_start:
                pre_fault_start = run_config.normal_start
                pre_fault_end = run_config.normal_end

            subgraph = build_anomalous_subgraph_with_fallback(
                service_df=fallback_service_df,
                graph=graph,
                normal_start=pre_fault_start,
                normal_end=pre_fault_end,
                fault_start=run_config.fault_start,
                fault_end=run_config.fault_end,
                anomaly_threshold=ANOMALY_THRESHOLD,
                fallback_top_edges=FALLBACK_TOP_ANOMALOUS_EDGES,
            )
            ranked = localize_root_cause(
                subgraph=subgraph,
                service_df=fallback_service_df,
                container_df=container_df,
                host_df=host_df,
                fault_start=run_config.fault_start,
                fault_end=run_config.fault_end,
                alpha=ALPHA,
                pagerank_c=PAGERANK_C,
            )

        with open(graph_output_path, "wb") as handle:
            pickle.dump(graph, handle)
        with open(output_subgraph_path, "wb") as handle:
            pickle.dump(subgraph, handle)

        ranking_df = pd.DataFrame(ranked, columns=["service", "score"])
        ranking_df.to_csv(output_ranking_path, index=False)

        ranked_services = [service for service, _ in ranked]
        runtime_seconds = time.perf_counter() - start_time

        if error_output_path.exists():
            error_output_path.unlink()

        return {
            "telemetry_day": run_config.telemetry_day,
            "exp_id": run_config.exp_id,
            "ground_truth_service": run_config.ground_truth_service,
            "predicted_top1_service": ranked_services[0] if ranked_services else None,
            "predicted_top3_service": _topk_services(ranked_services, 3),
            "predicted_top5_service": _topk_services(ranked_services, 5),
            "n_ranked_services": len(ranked_services),
            "top1_hit": _evaluate_topk(ranked_services, run_config.ground_truth_service, 1),
            "top3_hit": _evaluate_topk(ranked_services, run_config.ground_truth_service, 3),
            "top5_hit": _evaluate_topk(ranked_services, run_config.ground_truth_service, 5),
            "status": "ok",
            "error_message": None,
            "ranking_method": ranking_method,
            "normal_run_id": run_config.normal_run_id,
            "normal_start": run_config.normal_start,
            "normal_end": run_config.normal_end,
            "fault_start": run_config.fault_start,
            "fault_end": run_config.fault_end,
            "runtime_seconds": runtime_seconds,
            "selected_kpi_files": len(run_config.metric_files),
            "selected_trace_files": len(run_config.trace_files),
        }
    except Exception as exc:
        runtime_seconds = time.perf_counter() - start_time
        _write_error_file(error_output_path, f"{type(exc).__name__}: {exc}\n")
        return {
            "telemetry_day": run_config.telemetry_day,
            "exp_id": run_config.exp_id,
            "ground_truth_service": run_config.ground_truth_service,
            "predicted_top1_service": None,
            "predicted_top3_service": None,
            "predicted_top5_service": None,
            "n_ranked_services": 0,
            "top1_hit": False,
            "top3_hit": False,
            "top5_hit": False,
            "status": "error",
            "error_message": f"{type(exc).__name__}: {exc}",
            "ranking_method": None,
            "normal_run_id": run_config.normal_run_id,
            "normal_start": run_config.normal_start,
            "normal_end": run_config.normal_end,
            "fault_start": run_config.fault_start,
            "fault_end": run_config.fault_end,
            "runtime_seconds": runtime_seconds,
            "selected_kpi_files": len(run_config.metric_files),
            "selected_trace_files": len(run_config.trace_files),
        }


def build_day_summary(details_df: pd.DataFrame, telemetry_day: str, total_script_runtime_seconds: float) -> pd.DataFrame:
    n_total = len(details_df)
    n_ok = int((details_df["status"] == "ok").sum())
    n_error = int((details_df["status"] == "error").sum())

    summary = pd.DataFrame(
        [
            {
                "telemetry_day": telemetry_day,
                "n_total": n_total,
                "n_ok": n_ok,
                "n_error": n_error,
                "service_top1_accuracy": float(details_df["top1_hit"].fillna(False).mean()) if n_total else 0.0,
                "service_top3_accuracy": float(details_df["top3_hit"].fillna(False).mean()) if n_total else 0.0,
                "service_top5_accuracy": float(details_df["top5_hit"].fillna(False).mean()) if n_total else 0.0,
                "total_runtime_seconds": total_script_runtime_seconds,
                "avg_runtime_per_exception_seconds": (
                    total_script_runtime_seconds / EXPECTED_EXCEPTIONS_PER_DAY
                    if EXPECTED_EXCEPTIONS_PER_DAY
                    else 0.0
                ),
                "sum_of_individual_runtime_seconds": float(details_df["runtime_seconds"].sum()) if n_total else 0.0,
                "avg_runtime_per_processed_experiment_seconds": float(details_df["runtime_seconds"].mean()) if n_total else 0.0,
            }
        ]
    )
    return summary


def process_day(day_context: DayContext) -> tuple[pd.DataFrame, pd.DataFrame]:
    day_start_time = time.perf_counter()
    exp_ids = _discover_experiment_ids(day_context)
    if not exp_ids:
        raise ValueError(f"No fault experiments found in {day_context.fault_run_root}")

    details = []
    total = len(exp_ids)
    print(f"[DAY] {day_context.telemetry_day}: processing {total} fault experiments")

    for index, exp_id in enumerate(exp_ids, start=1):
        print(f"[RUN] {day_context.telemetry_day} {index}/{total} {exp_id}")
        run_config = _resolve_run_config(day_context, exp_id)
        details.append(run_single_experiment(run_config))

    details_df = pd.DataFrame(details).sort_values("exp_id").reset_index(drop=True)
    total_script_runtime_seconds = time.perf_counter() - day_start_time
    summary_df = build_day_summary(details_df, day_context.telemetry_day, total_script_runtime_seconds)

    day_context.output_day_dir.mkdir(parents=True, exist_ok=True)
    details_path = day_context.output_day_dir / DETAILS_OUTPUT_NAME
    summary_path = day_context.output_day_dir / SUMMARY_OUTPUT_NAME
    details_df.to_csv(details_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"[OK] Saved day details to {details_path}")
    print(f"[OK] Saved day summary to {summary_path}")
    print(summary_df.to_string(index=False))
    print("\n[Runtime] Per-experiment diagnosis time (seconds)")
    print(
        details_df[["exp_id", "runtime_seconds", "status"]]
        .to_string(index=False)
    )

    return details_df, summary_df


def main():
    script_dir = Path(__file__).resolve().parent
    all_summaries = []

    for telemetry_day in TELEMETRY_DAYS:
        day_context = _resolve_day_context(script_dir, telemetry_day)
        _, summary_df = process_day(day_context)
        all_summaries.append(summary_df)

    if all_summaries:
        combined_summary_df = pd.concat(all_summaries, ignore_index=True)
        combined_summary_path = (script_dir / OUTPUT_ROOT / ALL_DAYS_SUMMARY_OUTPUT_NAME).resolve()
        combined_summary_path.parent.mkdir(parents=True, exist_ok=True)
        combined_summary_df.to_csv(combined_summary_path, index=False)
        print(f"[OK] Saved all-day summary to {combined_summary_path}")


if __name__ == "__main__":
    main()
