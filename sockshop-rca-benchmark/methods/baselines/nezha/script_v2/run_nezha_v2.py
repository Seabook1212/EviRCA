import argparse
import json
import os
import re
from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter

import networkx as nx
import numpy as np
import pandas as pd
from drain3.template_miner import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent

TELEMETRY_DAYS = [
    "2026_03_12",
    "2026_03_13",
    "2026_03_14",
    "2026_03_17",
    "2026_03_18",
]
# EXP_ID = "pod_do_fault_orders_001"
EXP_ID = None

FAULT_WINDOW_MINUTES = 8
NORMAL_WINDOW_OFFSET_MINUTES = 0
NORMAL_WINDOW_DURATION_MINUTES = FAULT_WINDOW_MINUTES

IMPORTANT_METRICS = [
    "cpu_usage_pct",
    "memory_usage_pct",
    "network_rx",
    "network_tx",
    "error_count",
    "restart_count",
    "ready_ratio",
    "latency_p99",
    "request_rate",
    "success_rate",
]
RCA_RESOURCE_METRICS = (
    "cpu_usage_pct",
    "memory_usage_pct",
    "network_rx",
    "network_tx",
    "error_count",
    "restart_count",
    "ready_ratio",
)
RCA_SYMPTOM_METRICS = (
    "latency_p99",
    "request_rate",
    "success_rate",
)
INCLUDE_SYMPTOM_METRIC_EVENTS = False

Z_THRESHOLD = 3.0
PATTERN_MIN_COUNT = 1
PATTERN_MIN_SCORE = 0.10
EXPECTED_RANK_WEIGHT = 1.0
ABNORMAL_RANK_WEIGHT = 1.25
PAIR_RANK_WEIGHT = 0.75
RRF_K = 60
LOG_TEMPLATE_TOKEN_LIMIT = 12
PATTERN_NODE_LENGTHS = (2, 3)
BACKGROUND_LOG_TEMPLATE_WEIGHT = 0.25
ALIGNMENT_MODE = "repo_aligned"
REPO_ALIGNED_PAIR_EXPECTED_PENALTY = 0.75
BASELINE_OUTPUT_ROOT = "../data_v2"
REPO_ALIGNED_OUTPUT_ROOT = "../data_v2_repo_aligned"
OUTPUT_ROOT = REPO_ALIGNED_OUTPUT_ROOT if ALIGNMENT_MODE == "repo_aligned" else BASELINE_OUTPUT_ROOT
NORMAL_BASELINE_DIR = "normal_baseline"
METRIC_BASELINE_FILE = "metric_baseline.json"
NORMAL_PATTERN_FILE = "normal_patterns.json"
NORMAL_SUMMARY_FILE = "normal_baseline_summary.json"
PATTERN_SCORES_FILE = "pattern_scores.json"
SERVICE_RANKING_FILE = "service_ranking.csv"
RUN_SUMMARY_FILE = "nezha_run_summary.json"
EXPECTED_PATTERNS_FILE = "expected_patterns.json"
ACTUAL_PATTERNS_FILE = "actual_patterns.json"
PAIR_CANDIDATES_FILE = "pair_candidates.json"
DAY_DETAILS_FILE = "nezha_accuracy_details.csv"
DAY_SUMMARY_FILE = "nezha_accuracy_summary.csv"
ALL_DAYS_DETAILS_FILE = "nezha_accuracy_details_all_days.csv"
ALL_DAYS_SUMMARY_FILE = "nezha_accuracy_summary_all_days.csv"
OVERALL_SUMMARY_FILE = "nezha_accuracy_summary_overall.csv"
COMPARISON_FILE = "repo_alignment_comparison.json"

FAULT_RUN_ROOT_TEMPLATE = "../../../dataset_v2/fault_run_{day_suffix}"
NORMAL_RUN_ROOT_TEMPLATE = "../../../dataset_v2/normal_run_{day_suffix}"
TELEMETRY_METRICS_DIR_TEMPLATE = "../../../dataset_v2/telemetry/{telemetry_day}/metrics"
TELEMETRY_LOGS_DIR_TEMPLATE = "../../../dataset_v2/telemetry/{telemetry_day}/logs"
TELEMETRY_TRACES_DIR_TEMPLATE = "../../../dataset_v2/telemetry/{telemetry_day}/traces"

DEPLOYMENT_HASH_RE = re.compile(r"^[a-f0-9]{8,}$")
POD_SUFFIX_RE = re.compile(r"^[a-z0-9]{4,6}$")
UUID_RE = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b", re.IGNORECASE)
HEX_TOKEN_RE = re.compile(r"\b(?:0x)?[0-9a-f]{8,}\b", re.IGNORECASE)
IP_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
NUMBER_RE = re.compile(r"\b\d+\b")
NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
ANGLE_TOKEN_RE = re.compile(r"<\*>")
LONG_NUMBER_RE = re.compile(r"\b\d{4,}\b")
HEX_SUFFIX_RE = re.compile(r"(?<=[a-z_])[0-9a-f]{8,}(?=[a-z_]|$)", re.IGNORECASE)
METRIC_EVENT_LABELS = {
    "CpuUsageRate",
    "MemoryUsage",
    "NetworkReceive",
    "NetworkTransmit",
    "LatencyP99",
    "RequestRate",
    "SuccessRate",
    "ErrorCount",
    "RestartCount",
    "ReadyRatio",
}
DB_MQ_SERVICE_HINTS = ("-db", "rabbitmq", "mongodb", "mysql", "postgres", "redis", "mq")
BACKGROUND_TEMPLATE_HINTS = (
    "connection accepted",
    "connection ended",
    "client metadata",
    "client disconnected",
    "heartbeat",
    "metadata",
    "accepted",
    "ended",
)


@dataclass(frozen=True)
class RunWindow:
    run_id: str
    label: str
    start: pd.Timestamp
    end: pd.Timestamp
    metadata_path: Path
    inject_start: pd.Timestamp | None = None
    ground_truth_service: str | None = None


class DrainLogTemplateExtractor:
    def __init__(self):
        config = TemplateMinerConfig()
        config.parametrize_numeric_tokens = True
        self._miner = TemplateMiner(config=config)
        self._cache: dict[str, str] = {}

    def extract(self, raw_log: str | None, message: str | None) -> str:
        log_text = canonicalize_raw_log(raw_log, message)
        if not log_text:
            return "unknown"
        cached = self._cache.get(log_text)
        if cached is not None:
            return cached
        mined = self._miner.add_log_message(log_text)
        template = mined.get("template_mined") if isinstance(mined, dict) else None
        normalized = normalize_mined_template(template or log_text)
        self._cache[log_text] = normalized
        return normalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Nezha RCA on one day or a single fault experiment.")
    parser.add_argument(
        "--telemetry-day",
        default=None,
        help="Optional telemetry day in YYYY_MM_DD format. If omitted, all TELEMETRY_DAYS are processed.",
    )
    parser.add_argument(
        "--exp-id",
        default=EXP_ID,
        help="Optional fault experiment id, e.g. pod_do_fault_orders_001. If omitted, all fault runs for the day are processed.",
    )
    return parser.parse_args()


class HourlyMetricsCache:
    def __init__(self):
        self._cache: dict[Path, pd.DataFrame] = {}

    def load_file(self, file_path: Path) -> pd.DataFrame:
        cached = self._cache.get(file_path)
        if cached is not None:
            return cached

        chunks: list[pd.DataFrame] = []
        for chunk in pd.read_csv(file_path, usecols=["timestamp", "pod", "metric", "value"], chunksize=100_000):
            chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], errors="coerce")
            chunk["value"] = pd.to_numeric(chunk["value"], errors="coerce")
            chunk = chunk.dropna(subset=["timestamp", "pod", "metric", "value"])
            if chunk.empty:
                continue
            chunk["service"] = chunk["pod"].astype(str).map(derive_service_from_pod)
            chunk = chunk.dropna(subset=["service"])
            chunks.append(chunk)

        if chunks:
            loaded = pd.concat(chunks, ignore_index=True)
        else:
            loaded = pd.DataFrame(columns=["timestamp", "pod", "metric", "value", "service"])
        self._cache[file_path] = loaded
        return loaded


class HourlyLogsCache:
    def __init__(self):
        self._cache: dict[Path, pd.DataFrame] = {}
        self._template_extractor = DrainLogTemplateExtractor()

    def load_file(self, file_path: Path) -> pd.DataFrame:
        cached = self._cache.get(file_path)
        if cached is not None:
            return cached

        usecols = ["timestamp", "trace_id", "span_id", "service", "pod", "log_level", "log_type", "message", "raw_log"]
        chunks: list[pd.DataFrame] = []
        for chunk in pd.read_csv(file_path, usecols=usecols, chunksize=100_000):
            chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], utc=True, errors="coerce").dt.tz_convert(None)
            chunk = chunk.dropna(subset=["timestamp", "service"])
            if chunk.empty:
                continue
            chunk["message"] = chunk["message"].fillna("").astype(str)
            chunk["raw_log"] = chunk["raw_log"].fillna("").astype(str)
            chunk["log_level"] = chunk["log_level"].fillna("").astype(str)
            chunk["log_type"] = chunk["log_type"].fillna("").astype(str)
            chunk["log_template"] = [
                self._template_extractor.extract(raw_log, message)
                for raw_log, message in zip(chunk["raw_log"], chunk["message"])
            ]
            chunk["log_event_label"] = [
                build_log_event_label(log_type, log_level, template)
                for log_type, log_level, template in zip(chunk["log_type"], chunk["log_level"], chunk["log_template"])
            ]
            chunk["event_weight"] = [
                log_template_weight(service, template)
                for service, template in zip(chunk["service"].astype(str), chunk["log_template"])
            ]
            chunks.append(chunk)

        if chunks:
            loaded = pd.concat(chunks, ignore_index=True)
        else:
            loaded = pd.DataFrame(columns=usecols + ["log_template", "log_event_label", "event_weight"])
        self._cache[file_path] = loaded
        return loaded


class HourlyTracesCache:
    def __init__(self):
        self._cache: dict[Path, pd.DataFrame] = {}

    def load_file(self, file_path: Path) -> pd.DataFrame:
        cached = self._cache.get(file_path)
        if cached is not None:
            return cached

        usecols = ["timestamp", "trace_id", "span_id", "parent_span_id", "service", "operation", "duration", "span_kind", "pod"]
        chunks: list[pd.DataFrame] = []
        for chunk in pd.read_csv(file_path, usecols=usecols, chunksize=100_000):
            chunk["timestamp"] = pd.to_numeric(chunk["timestamp"], errors="coerce")
            chunk["duration"] = pd.to_numeric(chunk["duration"], errors="coerce")
            chunk = chunk.dropna(subset=["timestamp", "trace_id", "span_id", "service", "duration"])
            if chunk.empty:
                continue
            chunk["timestamp"] = chunk["timestamp"].astype(np.int64)
            chunk["service"] = chunk["service"].astype(str)
            chunk["operation"] = chunk["operation"].fillna("").astype(str)
            chunk["span_kind"] = chunk["span_kind"].fillna("").astype(str)
            chunk["pod"] = chunk["pod"].fillna(chunk["service"]).astype(str)
            chunks.append(chunk)

        if chunks:
            loaded = pd.concat(chunks, ignore_index=True)
        else:
            loaded = pd.DataFrame(columns=usecols)
        self._cache[file_path] = loaded
        return loaded


def _read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def parse_utc_timestamp(value: str) -> pd.Timestamp:
    ts = pd.to_datetime(value, utc=True, errors="raise")
    return ts.tz_convert(None)


def timestamp_to_us(value: pd.Timestamp) -> int:
    return int(value.value // 1_000)


def telemetry_day_to_suffix(telemetry_day: str) -> str:
    return datetime.strptime(telemetry_day, "%Y_%m_%d").strftime("%m%d")


def derive_service_from_pod(pod_name: str) -> str:
    parts = pod_name.split("-")
    if len(parts) >= 2 and parts[-1].isdigit():
        return "-".join(parts[:-1])
    if len(parts) >= 3 and DEPLOYMENT_HASH_RE.match(parts[-2]) and POD_SUFFIX_RE.match(parts[-1]):
        return "-".join(parts[:-2])
    return pod_name


def get_day_paths(script_dir: Path, telemetry_day: str) -> tuple[Path, Path, Path, Path, Path]:
    day_suffix = telemetry_day_to_suffix(telemetry_day)
    fault_root = (script_dir / FAULT_RUN_ROOT_TEMPLATE.format(day_suffix=day_suffix)).resolve()
    normal_root = (script_dir / NORMAL_RUN_ROOT_TEMPLATE.format(day_suffix=day_suffix)).resolve()
    metrics_dir = (script_dir / TELEMETRY_METRICS_DIR_TEMPLATE.format(telemetry_day=telemetry_day)).resolve()
    logs_dir = (script_dir / TELEMETRY_LOGS_DIR_TEMPLATE.format(telemetry_day=telemetry_day)).resolve()
    traces_dir = (script_dir / TELEMETRY_TRACES_DIR_TEMPLATE.format(telemetry_day=telemetry_day)).resolve()
    return fault_root, normal_root, metrics_dir, logs_dir, traces_dir


def discover_fault_runs(script_dir: Path, telemetry_day: str) -> list[RunWindow]:
    fault_root, _, _, _, _ = get_day_paths(script_dir, telemetry_day)
    runs: list[RunWindow] = []
    for exp_dir in sorted(fault_root.iterdir()):
        if not exp_dir.is_dir():
            continue

        metadata_path = exp_dir / "fault_info" / "fault_metadata.json"
        workload_path = exp_dir / "workload" / "workload_metadata.json"
        if not metadata_path.exists():
            continue

        metadata = _read_json(metadata_path)
        injection_info = metadata.get("injection_info", {})
        inject_start_raw = injection_info.get("inject_start")
        ground_truth_service = injection_info.get("service") or metadata.get("service")
        if not inject_start_raw or not ground_truth_service:
            continue

        inject_start = parse_utc_timestamp(inject_start_raw)
        start = inject_start
        end = inject_start + pd.Timedelta(minutes=FAULT_WINDOW_MINUTES)
        if workload_path.exists():
            workload = _read_json(workload_path)
            workload_end_raw = workload.get("workload_end_time")
            if workload_end_raw:
                end = min(end, parse_utc_timestamp(workload_end_raw))

        if end <= start:
            continue

        runs.append(
            RunWindow(
                run_id=exp_dir.name,
                label="fail",
                start=start,
                end=end,
                metadata_path=metadata_path.resolve(),
                inject_start=inject_start,
                ground_truth_service=str(ground_truth_service),
            )
        )

    if not runs:
        raise FileNotFoundError(f"No fault runs found for {telemetry_day}")
    return runs


def discover_normal_runs(script_dir: Path, telemetry_day: str) -> list[RunWindow]:
    _, normal_root, _, _, _ = get_day_paths(script_dir, telemetry_day)
    runs: list[RunWindow] = []
    for normal_dir in sorted(normal_root.iterdir()):
        if not normal_dir.is_dir():
            continue
        metadata_path = normal_dir / "workload" / "workload_metadata.json"
        if not metadata_path.exists():
            continue

        metadata = _read_json(metadata_path)
        workload_start_raw = metadata.get("workload_start_time")
        workload_end_raw = metadata.get("workload_end_time")
        if not workload_start_raw or not workload_end_raw:
            continue

        workload_start = parse_utc_timestamp(workload_start_raw)
        workload_end = parse_utc_timestamp(workload_end_raw)
        normal_start = workload_start + pd.Timedelta(minutes=NORMAL_WINDOW_OFFSET_MINUTES)
        normal_end = workload_end if NORMAL_WINDOW_DURATION_MINUTES is None else min(
            workload_end, normal_start + pd.Timedelta(minutes=NORMAL_WINDOW_DURATION_MINUTES)
        )
        if normal_end <= normal_start:
            continue

        runs.append(
            RunWindow(
                run_id=normal_dir.name,
                label="normal",
                start=normal_start,
                end=normal_end,
                metadata_path=metadata_path.resolve(),
            )
        )

    if not runs:
        raise FileNotFoundError(f"No normal runs found for {telemetry_day}")
    return runs


def _iter_hour_starts(windows: list[tuple[pd.Timestamp, pd.Timestamp]]) -> list[pd.Timestamp]:
    hours = set()
    for start, end in windows:
        current = start.floor("h")
        final = end.floor("h")
        while current <= final:
            hours.add(current)
            current += pd.Timedelta(hours=1)
    return sorted(hours)


def select_hourly_files(folder: Path, prefix: str, windows: list[tuple[pd.Timestamp, pd.Timestamp]]) -> list[Path]:
    selected = []
    for hour_start in _iter_hour_starts(windows):
        file_path = folder / f"{prefix}_{hour_start.strftime('%H')}.csv"
        if file_path.exists():
            selected.append(file_path)
    if not selected:
        raise FileNotFoundError(f"No hourly files found in {folder} for prefix {prefix}")
    return selected


def build_dt_mask(series: pd.Series, windows: list[tuple[pd.Timestamp, pd.Timestamp]]) -> pd.Series:
    mask = pd.Series(False, index=series.index)
    for start, end in windows:
        mask |= (series >= start) & (series <= end)
    return mask


def build_us_mask(series: pd.Series, windows_us: list[tuple[int, int]]) -> pd.Series:
    mask = pd.Series(False, index=series.index)
    for start_us, end_us in windows_us:
        mask |= (series >= start_us) & (series <= end_us)
    return mask


def load_metrics_for_run(cache: HourlyMetricsCache, metrics_dir: Path, run: RunWindow) -> pd.DataFrame:
    windows = [(run.start, run.end)]
    selected_files = select_hourly_files(metrics_dir, "prometheus_metrics_KPI", windows)
    frames = []
    for file_path in selected_files:
        df = cache.load_file(file_path)
        if df.empty:
            continue
        frames.append(df[build_dt_mask(df["timestamp"], windows)])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["timestamp", "pod", "metric", "value", "service"])


def load_logs_for_run(cache: HourlyLogsCache, logs_dir: Path, run: RunWindow) -> pd.DataFrame:
    windows = [(run.start, run.end)]
    selected_files = select_hourly_files(logs_dir, "loki_logs_parsed", windows)
    frames = []
    for file_path in selected_files:
        df = cache.load_file(file_path)
        if df.empty:
            continue
        frames.append(df[build_dt_mask(df["timestamp"], windows)])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["timestamp", "trace_id", "span_id", "service", "pod", "log_level", "log_type", "message", "raw_log", "log_template", "log_event_label", "event_weight"]
    )


def load_traces_for_run(cache: HourlyTracesCache, traces_dir: Path, run: RunWindow) -> pd.DataFrame:
    windows = [(run.start, run.end)]
    windows_us = [(timestamp_to_us(start), timestamp_to_us(end)) for start, end in windows]
    selected_files = select_hourly_files(traces_dir, "jaeger_traces_parsed", windows)
    frames = []
    for file_path in selected_files:
        df = cache.load_file(file_path)
        if df.empty:
            continue
        frames.append(df[build_us_mask(df["timestamp"], windows_us)])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["timestamp", "trace_id", "span_id", "parent_span_id", "service", "operation", "duration", "span_kind"])


def build_metric_baseline(normal_runs: list[RunWindow], metrics_dir: Path, cache: HourlyMetricsCache) -> dict[tuple[str, str], tuple[float, float]]:
    baseline_values: defaultdict[tuple[str, str], list[float]] = defaultdict(list)
    global_values: defaultdict[str, list[float]] = defaultdict(list)

    for run in normal_runs:
        metrics_df = load_metrics_for_run(cache, metrics_dir, run)
        if metrics_df.empty:
            continue
        metrics_df = metrics_df[metrics_df["metric"].isin(IMPORTANT_METRICS)]
        for row in metrics_df.itertuples(index=False):
            key = (str(row.service), str(row.metric))
            value = float(row.value)
            baseline_values[key].append(value)
            global_values[str(row.metric)].append(value)

    baseline = {}
    for key, values in baseline_values.items():
        series = pd.Series(values)
        baseline[key] = (float(series.mean()), float(series.std()))

    for metric, values in global_values.items():
        series = pd.Series(values)
        baseline[("*", metric)] = (float(series.mean()), float(series.std()))

    return baseline


def normalize_operation_name(value: str) -> str:
    text = str(value).strip().lower()
    if not text:
        return "unknown"
    text = UUID_RE.sub(" uuid ", text)
    text = HEX_TOKEN_RE.sub(" hex ", text)
    text = HEX_SUFFIX_RE.sub(" hex ", text)
    text = LONG_NUMBER_RE.sub(" num ", text)
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", text)
    cleaned = re.sub(r"_(hex|uuid|num)(?:_(hex|uuid|num))+", r"_\1", cleaned)
    return cleaned.strip("_") or "unknown"


def normalize_mined_template(template: str) -> str:
    text = str(template or "").strip().lower()
    text = UUID_RE.sub(" uuid ", text)
    text = IP_RE.sub(" ip ", text)
    text = HEX_TOKEN_RE.sub(" hex ", text)
    text = NUMBER_RE.sub(" num ", text)
    text = ANGLE_TOKEN_RE.sub(" param ", text)
    tokens = [token for token in NON_ALNUM_RE.sub(" ", text).split() if token]
    return "_".join(tokens[:LOG_TEMPLATE_TOKEN_LIMIT]) if tokens else "unknown"


def canonicalize_raw_log(raw_log: str | None, message: str | None) -> str:
    raw_text = str(raw_log or "").strip()
    if raw_text:
        try:
            parsed = json.loads(raw_text)
        except (TypeError, ValueError, json.JSONDecodeError):
            return raw_text
        if isinstance(parsed, dict):
            parts = []
            for key in ("msg", "message", "level", "caller", "service", "operation", "ctx", "c"):
                value = parsed.get(key)
                if value not in (None, ""):
                    parts.append(f"{key}={value}")
            attr = parsed.get("attr")
            if isinstance(attr, dict):
                for key in ("remote", "collection", "command", "commandName", "error", "connectionId", "connectionCount"):
                    value = attr.get(key)
                    if value not in (None, ""):
                        parts.append(f"{key}={value}")
            if parts:
                return " ".join(str(part) for part in parts)
    return str(message or "").strip()


def is_db_or_mq_service(service: str) -> bool:
    normalized = str(service or "").strip().lower()
    return bool(normalized) and any(hint in normalized for hint in DB_MQ_SERVICE_HINTS)


def log_template_weight(service: str, template: str) -> float:
    template_text = str(template or "").replace("_", " ").lower()
    if is_db_or_mq_service(service) and any(hint in template_text for hint in BACKGROUND_TEMPLATE_HINTS):
        return BACKGROUND_LOG_TEMPLATE_WEIGHT
    return 1.0


def build_log_event_label(log_type: str, log_level: str, log_template: str) -> str:
    parts = [
        normalize_operation_name(log_type),
        normalize_operation_name(log_level),
        normalize_mined_template(log_template),
    ]
    return "log_" + "_".join(part for part in parts if part and part != "unknown")


def metric_to_event_label(metric: str) -> str:
    mapping = {
        "cpu_usage_pct": "CpuUsageRate",
        "memory_usage_pct": "MemoryUsage",
        "network_rx": "NetworkReceive",
        "network_tx": "NetworkTransmit",
        "latency_p99": "LatencyP99",
        "request_rate": "RequestRate",
        "success_rate": "SuccessRate",
        "error_count": "ErrorCount",
        "restart_count": "RestartCount",
        "ready_ratio": "ReadyRatio",
    }
    return mapping.get(metric, metric)


def is_resource_event_label(event_label: str) -> bool:
    return event_label in METRIC_EVENT_LABELS


def is_resource_metric(metric: str) -> bool:
    return metric in RCA_RESOURCE_METRICS


def should_include_metric_event_in_graph(metric: str) -> bool:
    return is_resource_metric(metric) or (INCLUDE_SYMPTOM_METRIC_EVENTS and metric in RCA_SYMPTOM_METRICS)


def metric_event_weight(metric: str) -> float:
    return 1.0 if is_resource_metric(metric) else BACKGROUND_LOG_TEMPLATE_WEIGHT


def active_pattern_node_lengths() -> tuple[int, ...]:
    return (2,) if ALIGNMENT_MODE == "repo_aligned" else PATTERN_NODE_LENGTHS


def active_pattern_method_name() -> str:
    if ALIGNMENT_MODE == "repo_aligned":
        return "direct_edge_support_repo_aligned"
    return f"local_path_support_{'-'.join(str(length) for length in PATTERN_NODE_LENGTHS)}_nodes"


def active_service_ranking_fusion_name() -> str:
    return "pair_candidates_only" if ALIGNMENT_MODE == "repo_aligned" else "reciprocal_rank_fusion"


def active_pair_ranking_strategy_name() -> str:
    if ALIGNMENT_MODE == "repo_aligned":
        return "actual_score_minus_expected_penalty_with_expected_only_fallback"
    return "combined_score"


def is_trivial_expected_pattern(source_label: str, target_label: str) -> bool:
    if source_label == target_label and is_resource_event_label(source_label):
        return True
    if source_label.endswith(" start") and target_label.endswith(" end"):
        source_base = source_label[: -len(" start")]
        target_base = target_label[: -len(" end")]
        if source_base == target_base:
            return True
    return False


def collect_known_services(*frames: pd.DataFrame) -> set[str]:
    services: set[str] = set()
    for frame in frames:
        if frame.empty or "service" not in frame.columns:
            continue
        for value in frame["service"].dropna().astype(str):
            service = value.strip()
            if not service:
                continue
            services.add(service)
    return services


def filter_service_ranking(service_ranking: list[tuple[str, float]], valid_services: set[str]) -> list[tuple[str, float]]:
    filtered: list[tuple[str, float]] = []
    seen: set[str] = set()
    for service, score in service_ranking:
        service_name = str(service).strip()
        if not service_name or service_name in seen:
            continue
        if valid_services and service_name not in valid_services:
            continue
        filtered.append((service_name, float(score)))
        seen.add(service_name)
    return filtered


def select_service_from_pattern(source_label: str, target_label: str, pod: str) -> str:
    for label in (source_label, target_label):
        service = event_label_to_service(label)
        if service:
            return service
    return derive_service_from_pod(pod) if pod else ""


def pattern_key_from_labels(labels: list[str]) -> str:
    return "|||".join(labels)


def pattern_labels_from_key(pattern_key: str) -> list[str]:
    return [label for label in pattern_key.split("|||") if label]


def service_candidates_from_labels(labels: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for label in labels:
        service = event_label_to_service(label)
        if not service or service in seen:
            continue
        ordered.append(service)
        seen.add(service)
    return ordered


def get_pattern_depth_and_pod(labels: list[str], normal_event_depth_index: dict[str, tuple[int, str]]) -> tuple[int, str]:
    best_depth = 1
    best_pod = ""
    for label in labels:
        depth, pod = normal_event_depth_index.get(label, (0, ""))
        if depth > best_depth:
            best_depth = depth
            best_pod = pod
    return best_depth, best_pod


def select_service_from_labels(labels: list[str], normal_event_depth_index: dict[str, tuple[int, str]], pod: str) -> str:
    candidates = []
    for label in labels:
        service = event_label_to_service(label)
        if not service:
            continue
        depth, _ = normal_event_depth_index.get(label, (1, ""))
        candidates.append((depth, service))
    if candidates:
        if ALIGNMENT_MODE == "repo_aligned":
            return candidates[0][1]
        return max(candidates, key=lambda item: item[0])[1]
    return derive_service_from_pod(pod) if pod else ""


def collect_local_pattern_support(
    node_info: dict[str, dict],
    predecessors: dict[str, list[str]],
) -> dict[str, float]:
    support: defaultdict[str, float] = defaultdict(float)
    successors: defaultdict[str, list[str]] = defaultdict(list)
    pattern_lengths = active_pattern_node_lengths()
    for child_id, parent_ids in predecessors.items():
        for parent_id in parent_ids:
            successors[parent_id].append(child_id)
            if 2 in pattern_lengths:
                labels = [node_info[parent_id]["event_label"], node_info[child_id]["event_label"]]
                weight = min(float(node_info[parent_id].get("weight", 1.0)), float(node_info[child_id].get("weight", 1.0)))
                support[pattern_key_from_labels(labels)] += weight

    if 3 in pattern_lengths:
        for middle_id in node_info:
            for parent_id in predecessors.get(middle_id, []):
                for child_id in successors.get(middle_id, []):
                    if parent_id == child_id:
                        continue
                    labels = [
                        node_info[parent_id]["event_label"],
                        node_info[middle_id]["event_label"],
                        node_info[child_id]["event_label"],
                    ]
                    weight = min(
                        float(node_info[parent_id].get("weight", 1.0)),
                        float(node_info[middle_id].get("weight", 1.0)),
                        float(node_info[child_id].get("weight", 1.0)),
                    )
                    support[pattern_key_from_labels(labels)] += weight

    return dict(sorted(support.items(), key=lambda item: (-item[1], item[0])))


def extract_metric_anomalies(metrics_df: pd.DataFrame, baseline: dict[tuple[str, str], tuple[float, float]]) -> list[dict]:
    events = []
    if metrics_df.empty:
        return events

    metrics_df = metrics_df[metrics_df["metric"].isin(IMPORTANT_METRICS)]
    for row in metrics_df.itertuples(index=False):
        service = str(row.service)
        metric = str(row.metric)
        mean_std = baseline.get((service, metric)) or baseline.get(("*", metric))
        if mean_std is None:
            continue
        mean, std = mean_std
        if std <= 1e-8:
            continue
        z_score = abs((float(row.value) - mean) / std)
        if z_score <= Z_THRESHOLD:
            continue
        events.append(
            {
                "trace_id": None,
                "span_id": None,
                "service": service,
                "pod": str(row.pod),
                "metric": metric,
                "event_type": metric_to_event_label(metric),
                "timestamp_us": timestamp_to_us(pd.Timestamp(row.timestamp)),
                "source": "metric",
                "z_score": z_score,
                "value": float(row.value),
                "weight": metric_event_weight(metric),
            }
        )
    return events


def build_time_event_index(events: list[dict], key_name: str) -> dict[str, tuple[list[int], list[dict]]]:
    grouped: defaultdict[str, list[dict]] = defaultdict(list)
    for event in events:
        key = str(event.get(key_name) or "")
        if key:
            grouped[key].append(event)

    indexed = {}
    for key, key_events in grouped.items():
        ordered = sorted(key_events, key=lambda item: int(item["timestamp_us"]))
        indexed[key] = ([int(item["timestamp_us"]) for item in ordered], ordered)
    return indexed


def slice_time_event_index(
    event_index: dict[str, tuple[list[int], list[dict]]],
    key: str,
    start_us: int,
    end_us: int,
) -> list[dict]:
    timestamps, events = event_index.get(key, ([], []))
    if not timestamps:
        return []
    left = bisect_left(timestamps, start_us)
    right = bisect_right(timestamps, end_us)
    return events[left:right]


def collect_pattern_support_and_depth_index(
    traces_df: pd.DataFrame,
    logs_df: pd.DataFrame,
    metric_events: list[dict],
    include_depth_index: bool,
) -> tuple[dict[str, int], int, dict[str, tuple[int, str]]]:
    if traces_df.empty:
        return {}, 0, {}

    trace_groups = traces_df.copy()
    trace_groups["trace_id"] = trace_groups["trace_id"].astype(str)
    trace_groups["span_id"] = trace_groups["span_id"].astype(str)
    trace_groups["parent_span_id"] = trace_groups["parent_span_id"].where(
        trace_groups["parent_span_id"].notna(), None
    )
    if "pod" not in trace_groups.columns:
        trace_groups["pod"] = trace_groups["service"]
    trace_groups["pod"] = trace_groups["pod"].fillna(trace_groups["service"]).astype(str)

    log_rows = logs_df.copy()
    logs_by_trace_span: dict[tuple[str, str], pd.DataFrame] = {}
    logs_by_trace_service: dict[tuple[str, str], pd.DataFrame] = {}
    if not log_rows.empty:
        log_rows["trace_id"] = log_rows["trace_id"].fillna("").astype(str)
        log_rows["span_id"] = log_rows["span_id"].fillna("").astype(str)
        log_rows["timestamp_us"] = log_rows["timestamp"].map(timestamp_to_us)
        for (trace_id, span_id), span_logs in log_rows.groupby(["trace_id", "span_id"], sort=False):
            if span_id:
                logs_by_trace_span[(str(trace_id), str(span_id))] = span_logs.sort_values("timestamp_us")
        for (trace_id, service), service_logs in log_rows.groupby(["trace_id", "service"], sort=False):
            logs_by_trace_service[(str(trace_id), str(service))] = service_logs.sort_values("timestamp_us")

    metric_by_service = build_time_event_index(metric_events, "service")
    metric_by_pod = build_time_event_index(metric_events, "pod")

    total_support: defaultdict[str, float] = defaultdict(float)
    depth_index: dict[str, tuple[int, str]] = {}
    graph_count = 0

    for trace_id, trace_df in tqdm(trace_groups.groupby("trace_id"), desc="building-graphs", leave=False):
        span_state: dict[str, dict] = {}
        node_info: dict[str, dict] = {}
        predecessors: defaultdict[str, list[str]] = defaultdict(list)

        for row in trace_df.sort_values(["timestamp", "span_id"]).itertuples(index=False):
            span_id = str(row.span_id)
            parent_span_id = None if row.parent_span_id is None else str(row.parent_span_id)
            service = str(row.service)
            pod = str(row.pod)
            start_us = int(row.timestamp)
            end_us = int(row.timestamp + row.duration)
            operation = normalize_operation_name(row.operation)
            span_events = [
                {
                    "node_id": f"{span_id}:start",
                    "event_label": f"{service} {operation} start",
                    "timestamp_us": start_us,
                    "source": "trace",
                    "service": service,
                    "pod": pod,
                    "is_span_start": True,
                    "weight": 1.0,
                }
            ]

            span_metric_events = []
            span_metric_events.extend(slice_time_event_index(metric_by_pod, pod, start_us, end_us))
            span_metric_events.extend(slice_time_event_index(metric_by_service, service, start_us, end_us))
            unique_metric_events = {
                (event["event_type"], int(event["timestamp_us"]), str(event.get("pod") or ""), str(event["service"])): event
                for event in span_metric_events
                if should_include_metric_event_in_graph(str(event["metric"]))
            }
            for index, event in enumerate(sorted(unique_metric_events.values(), key=lambda item: item["timestamp_us"]), start=1):
                span_events.append(
                    {
                        "node_id": f"{span_id}:metric:{index}",
                        "event_label": str(event["event_type"]),
                        "timestamp_us": int(event["timestamp_us"]),
                        "source": "metric",
                        "service": service,
                        "pod": pod,
                        "metric": str(event["metric"]),
                        "is_span_start": False,
                        "weight": float(event.get("weight", 1.0)),
                    }
                )

            span_log_df = logs_by_trace_span.get((trace_id, span_id))
            if span_log_df is None:
                service_log_df = logs_by_trace_service.get((trace_id, service))
                if service_log_df is not None:
                    span_log_df = service_log_df[
                        (service_log_df["timestamp_us"] >= start_us)
                        & (service_log_df["timestamp_us"] <= end_us)
                    ]
            if span_log_df is not None and not span_log_df.empty:
                for index, log_row in enumerate(span_log_df.itertuples(index=False), start=1):
                    span_events.append(
                        {
                            "node_id": f"{span_id}:log:{index}",
                            "event_label": str(getattr(log_row, "log_event_label", "")),
                            "timestamp_us": int(log_row.timestamp_us),
                            "source": "log",
                            "service": service,
                            "pod": pod,
                            "is_span_start": False,
                            "weight": float(getattr(log_row, "event_weight", 1.0)),
                        }
                    )

            span_events.append(
                {
                    "node_id": f"{span_id}:end",
                    "event_label": f"{service} {operation} end",
                    "timestamp_us": end_us,
                    "source": "trace",
                    "service": service,
                    "pod": pod,
                    "is_span_start": False,
                    "weight": 1.0,
                }
            )
            span_events.sort(key=lambda item: (item["timestamp_us"], 0 if item["node_id"].endswith(":start") else 1))

            for event in span_events:
                node_info[event["node_id"]] = {
                    "event_label": event["event_label"],
                    "pod": event["pod"],
                    "is_span_start": event["is_span_start"],
                    "weight": float(event.get("weight", 1.0)),
                }

            for prev_event, next_event in zip(span_events, span_events[1:]):
                predecessors[next_event["node_id"]].append(prev_event["node_id"])

            span_state[span_id] = {
                "parent_span_id": parent_span_id,
                "pod": pod,
                "start_us": start_us,
                "events": span_events,
            }

        for span_id, span in span_state.items():
            parent_span_id = span["parent_span_id"]
            if not parent_span_id or parent_span_id not in span_state:
                continue
            parent_span = span_state[parent_span_id]
            parent_node_id = parent_span["events"][0]["node_id"]
            if parent_span["pod"] == span["pod"]:
                for prev_event, next_event in zip(parent_span["events"], parent_span["events"][1:]):
                    if next_event["timestamp_us"] > span["start_us"]:
                        parent_node_id = prev_event["node_id"]
                        break
            child_node_id = span["events"][0]["node_id"]
            predecessors[child_node_id].append(parent_node_id)

        pattern_support = collect_local_pattern_support(node_info, predecessors)
        if not pattern_support:
            continue

        graph_count += 1
        for key, value in pattern_support.items():
            total_support[key] += value

        if include_depth_index:
            depth_cache: dict[str, int] = {}

            def compute_depth(node_id: str) -> int:
                cached_depth = depth_cache.get(node_id)
                if cached_depth is not None:
                    return cached_depth
                base = 1 if node_info[node_id]["is_span_start"] else 0
                parent_ids = predecessors.get(node_id, [])
                if not parent_ids:
                    depth = base
                else:
                    depth = base + max(compute_depth(parent_id) for parent_id in parent_ids)
                depth_cache[node_id] = depth
                return depth

            for node_id, info in node_info.items():
                label = info["event_label"]
                pod = info["pod"]
                depth = compute_depth(node_id)
                previous = depth_index.get(label)
                if previous is None or depth > previous[0]:
                    depth_index[label] = (depth, pod)

    return dict(sorted(total_support.items(), key=lambda item: (-item[1], item[0]))), graph_count, depth_index


def build_event_graphs(
    traces_df: pd.DataFrame,
    logs_df: pd.DataFrame,
    metric_events: list[dict],
) -> dict[str, nx.DiGraph]:
    if traces_df.empty:
        return {}

    trace_groups = traces_df.copy()
    trace_groups["trace_id"] = trace_groups["trace_id"].astype(str)
    trace_groups["span_id"] = trace_groups["span_id"].astype(str)
    trace_groups["parent_span_id"] = trace_groups["parent_span_id"].where(
        trace_groups["parent_span_id"].notna(), None
    )
    if "pod" not in trace_groups.columns:
        trace_groups["pod"] = trace_groups["service"]
    trace_groups["pod"] = trace_groups["pod"].fillna(trace_groups["service"]).astype(str)

    log_rows = logs_df.copy()
    logs_by_trace_span: dict[tuple[str, str], pd.DataFrame] = {}
    logs_by_trace_service: dict[tuple[str, str], pd.DataFrame] = {}
    if not log_rows.empty:
        log_rows["trace_id"] = log_rows["trace_id"].fillna("").astype(str)
        log_rows["span_id"] = log_rows["span_id"].fillna("").astype(str)
        log_rows["timestamp_us"] = log_rows["timestamp"].map(timestamp_to_us)
        for (trace_id, span_id), span_logs in log_rows.groupby(["trace_id", "span_id"], sort=False):
            if span_id:
                logs_by_trace_span[(str(trace_id), str(span_id))] = span_logs.sort_values("timestamp_us")
        for (trace_id, service), service_logs in log_rows.groupby(["trace_id", "service"], sort=False):
            logs_by_trace_service[(str(trace_id), str(service))] = service_logs.sort_values("timestamp_us")

    metric_by_service = build_time_event_index(metric_events, "service")
    metric_by_pod = build_time_event_index(metric_events, "pod")

    graphs: dict[str, nx.DiGraph] = {}
    for trace_id, trace_df in tqdm(trace_groups.groupby("trace_id"), desc="building-graphs", leave=False):
        graph = nx.DiGraph()
        span_state: dict[str, dict] = {}

        for row in trace_df.sort_values(["timestamp", "span_id"]).itertuples(index=False):
            span_id = str(row.span_id)
            parent_span_id = None if row.parent_span_id is None else str(row.parent_span_id)
            service = str(row.service)
            pod = str(row.pod)
            start_us = int(row.timestamp)
            end_us = int(row.timestamp + row.duration)
            operation = normalize_operation_name(row.operation)
            span_events = [
                {
                    "node_id": f"{span_id}:start",
                    "event_label": f"{service} {operation} start",
                    "timestamp_us": start_us,
                    "source": "trace",
                    "service": service,
                    "pod": pod,
                    "is_span_start": True,
                }
            ]

            span_metric_events = []
            span_metric_events.extend(slice_time_event_index(metric_by_pod, pod, start_us, end_us))
            span_metric_events.extend(slice_time_event_index(metric_by_service, service, start_us, end_us))
            unique_metric_events = {
                (event["event_type"], int(event["timestamp_us"]), str(event.get("pod") or ""), str(event["service"])): event
                for event in span_metric_events
                if should_include_metric_event_in_graph(str(event["metric"]))
            }
            for index, event in enumerate(sorted(unique_metric_events.values(), key=lambda item: item["timestamp_us"]), start=1):
                span_events.append(
                    {
                        "node_id": f"{span_id}:metric:{index}",
                        "event_label": str(event["event_type"]),
                        "timestamp_us": int(event["timestamp_us"]),
                        "source": "metric",
                        "service": service,
                        "pod": pod,
                        "metric": str(event["metric"]),
                        "is_span_start": False,
                    }
                )

            span_log_df = logs_by_trace_span.get((trace_id, span_id))
            if span_log_df is None:
                service_log_df = logs_by_trace_service.get((trace_id, service))
                if service_log_df is not None:
                    span_log_df = service_log_df[
                        (service_log_df["timestamp_us"] >= start_us)
                        & (service_log_df["timestamp_us"] <= end_us)
                    ]
            if span_log_df is not None and not span_log_df.empty:
                for index, log_row in enumerate(span_log_df.sort_values("timestamp_us").itertuples(index=False), start=1):
                    span_events.append(
                        {
                            "node_id": f"{span_id}:log:{index}",
                            "event_label": str(getattr(log_row, "log_event_label", "")),
                            "timestamp_us": int(log_row.timestamp_us),
                            "source": "log",
                            "service": service,
                            "pod": pod,
                            "is_span_start": False,
                        }
                    )

            span_events.append(
                {
                    "node_id": f"{span_id}:end",
                    "event_label": f"{service} {operation} end",
                    "timestamp_us": end_us,
                    "source": "trace",
                    "service": service,
                    "pod": pod,
                    "is_span_start": False,
                }
            )
            span_events = sorted(
                span_events,
                key=lambda item: (item["timestamp_us"], 0 if item["node_id"].endswith(":start") else 1),
            )

            for event in span_events:
                graph.add_node(
                    event["node_id"],
                    event_label=event["event_label"],
                    service=event["service"],
                    pod=event["pod"],
                    timestamp_us=event["timestamp_us"],
                    source=event["source"],
                    is_span_start=event["is_span_start"],
                )
            for prev_event, next_event in zip(span_events, span_events[1:]):
                graph.add_edge(prev_event["node_id"], next_event["node_id"], edge_type="sequence")

            span_state[span_id] = {
                "span_id": span_id,
                "parent_span_id": parent_span_id,
                "pod": pod,
                "service": service,
                "start_us": start_us,
                "events": span_events,
            }

        for span in span_state.values():
            parent_span_id = span["parent_span_id"]
            if not parent_span_id or parent_span_id not in span_state:
                continue
            parent_span = span_state[parent_span_id]
            if parent_span["pod"] == span["pod"]:
                insert_from = parent_span["events"][0]["node_id"]
                for prev_event, next_event in zip(parent_span["events"], parent_span["events"][1:]):
                    if next_event["timestamp_us"] > span["start_us"]:
                        insert_from = prev_event["node_id"]
                        break
                graph.add_edge(insert_from, span["events"][0]["node_id"], edge_type="parent")
            else:
                graph.add_edge(parent_span["events"][0]["node_id"], span["events"][0]["node_id"], edge_type="parent")

        if graph.number_of_edges() > 0:
            graphs[trace_id] = graph

    return graphs


def pattern_key_from_edge(graph: nx.DiGraph, source: str, target: str) -> str:
    source_label = graph.nodes[source].get("event_label", "unknown")
    target_label = graph.nodes[target].get("event_label", "unknown")
    return pattern_key_from_labels([source_label, target_label])


def split_pattern_key(pattern_key: str) -> tuple[str, str]:
    labels = pattern_labels_from_key(pattern_key)
    if not labels:
        return "", ""
    return labels[0], labels[-1]


def get_pattern_support(graphs: dict[str, nx.DiGraph]) -> dict[str, int]:
    result_support_dict: defaultdict[str, int] = defaultdict(int)
    for _, graph in tqdm(graphs.items(), desc="mining-patterns", leave=False):
        graph_support: defaultdict[str, int] = defaultdict(int)
        for source, target in graph.edges():
            graph_support[pattern_key_from_edge(graph, source, target)] += 1
        for key, value in graph_support.items():
            result_support_dict[key] += value
    return dict(sorted(result_support_dict.items(), key=lambda item: (-item[1], item[0])))


def support_to_ratio(pattern_support: dict[str, int], total_graphs: int, min_support: float) -> dict[str, float]:
    if total_graphs <= 0:
        return {}
    converted = {}
    for key, count in pattern_support.items():
        support = count / total_graphs
        if support >= min_support:
            converted[key] = support
    return converted


def compute_pattern_scores(normal_patterns: dict[str, int], fault_patterns: dict[str, int]) -> dict[str, dict[str, float]]:
    all_patterns = set(normal_patterns.keys()) | set(fault_patterns.keys())
    scores = {}
    for pattern in all_patterns:
        support_c = float(normal_patterns.get(pattern, 0))
        support_p = float(fault_patterns.get(pattern, 0))
        if support_c + support_p == 0:
            continue
        scores[pattern] = {
            "support_C": support_c,
            "support_P": support_p,
            "Score_E": support_c / (support_c + support_p),
            "Score_A": support_p / (support_c + support_p),
        }
    return scores


def abnormal_pattern_ranker(
    normal_pattern_dict: dict[str, int],
    abnormal_pattern_dict: dict[str, int],
    min_score: float,
    min_count: int,
) -> dict[str, float]:
    score_dict = {}
    for key, abnormal_count in abnormal_pattern_dict.items():
        if abnormal_count <= min_count:
            continue
        normal_count = normal_pattern_dict.get(key, 0)
        score = abnormal_count / (abnormal_count + normal_count) if (abnormal_count + normal_count) > 0 else 0.0
        if score >= min_score:
            score_dict[key] = float(score)
    return dict(sorted(score_dict.items(), key=lambda item: (-item[1], -abnormal_pattern_dict.get(item[0], 0), item[0])))


def is_root_of_child_pattern(candidate_key: str, score_dict: dict[str, float]) -> bool:
    source_label, target_label = split_pattern_key(candidate_key)
    if is_resource_event_label(target_label):
        return False
    candidate_score = score_dict.get(candidate_key, 0.0)
    for other_key, other_score in score_dict.items():
        if other_key == candidate_key:
            continue
        other_source, other_target = split_pattern_key(other_key)
        if source_label == other_target and candidate_score <= other_score:
            return True
    return False


def compute_node_depth(graph: nx.DiGraph, node_id: str) -> int:
    max_depth = 0
    stack = [(node_id, 0)]
    seen: dict[str, int] = {}
    while stack:
        current, current_depth = stack.pop()
        next_depth = current_depth + (1 if graph.nodes[current].get("is_span_start") else 0)
        if next_depth <= seen.get(current, -1):
            continue
        seen[current] = next_depth
        predecessors = list(graph.predecessors(current))
        if not predecessors:
            max_depth = max(max_depth, next_depth)
            continue
        for predecessor in predecessors:
            stack.append((predecessor, next_depth))
    return max_depth


def get_event_depth_pod(normal_graphs: dict[str, nx.DiGraph], source_label: str) -> tuple[int, str]:
    max_depth = 0
    event_pod = ""
    for graph in normal_graphs.values():
        for node_id, data in graph.nodes(data=True):
            if data.get("event_label") != source_label:
                continue
            depth = compute_node_depth(graph, node_id)
            if depth > max_depth:
                max_depth = depth
                event_pod = str(data.get("pod") or "")
    return max_depth, event_pod


def build_normal_event_depth_index(normal_graphs: dict[str, nx.DiGraph]) -> dict[str, tuple[int, str]]:
    index: dict[str, tuple[int, str]] = {}
    for graph in normal_graphs.values():
        for node_id, data in graph.nodes(data=True):
            source_label = str(data.get("event_label") or "")
            if not source_label:
                continue
            depth = compute_node_depth(graph, node_id)
            pod = str(data.get("pod") or "")
            previous = index.get(source_label)
            if previous is None or depth > previous[0]:
                index[source_label] = (depth, pod)
    return index


def build_alarm_lookup(metric_events: list[dict]) -> dict[str, list[str]]:
    lookup: defaultdict[str, list[str]] = defaultdict(list)
    for event in metric_events:
        metric = str(event.get("metric") or "")
        if not is_resource_metric(metric):
            continue
        pod = str(event.get("pod") or "")
        service = str(event.get("service") or "")
        label = str(event.get("event_type") or "")
        if pod and label not in lookup[pod]:
            lookup[pod].append(label)
        if service and label not in lookup[service]:
            lookup[service].append(label)
    return lookup


def select_result_resource(
    labels: list[str],
    pod: str,
    service: str,
    alarm_lookup: dict[str, list[str]],
) -> str | None:
    for label in labels:
        if is_resource_event_label(label):
            return label
    for alarm_label in alarm_lookup.get(pod, []) or alarm_lookup.get(service, []):
        return alarm_label
    return None


def expected_pattern_ranker(
    normal_pattern_dict: dict[str, int],
    abnormal_pattern_dict: dict[str, int],
    normal_event_depth_index: dict[str, tuple[int, str]],
    metric_events: list[dict],
    min_score: float,
    min_count: int,
) -> list[dict]:
    score_dict = {}
    for key, normal_count in normal_pattern_dict.items():
        if normal_count <= min_count:
            continue
        abnormal_count = abnormal_pattern_dict.get(key, 0)
        score = normal_count / (normal_count + abnormal_count) if (normal_count + abnormal_count) > 0 else 0.0
        if score >= min_score:
            score_dict[key] = float(score)

    pruned_score_dict = {
        key: value
        for key, value in score_dict.items()
        if not is_root_of_child_pattern(key, score_dict)
    }

    alarm_lookup = build_alarm_lookup(metric_events)
    result_list = []
    for key, value in pruned_score_dict.items():
        labels = pattern_labels_from_key(key)
        source_label, target_label = split_pattern_key(key)
        if is_trivial_expected_pattern(source_label, target_label):
            continue
        depth, pod = get_pattern_depth_and_pod(labels, normal_event_depth_index)
        service = select_service_from_labels(labels, normal_event_depth_index, pod)
        if not service:
            service = "unknown"
        result = {
            "events": key,
            "labels": labels,
            "score": float(value),
            "depth": int(depth) if depth > 0 else 1,
            "pod": pod,
            "service": service,
            "service_candidates": service_candidates_from_labels(labels),
        }
        resource_label = select_result_resource(labels, pod, service, alarm_lookup)
        if resource_label:
            result["resource"] = resource_label
        result_list.append(result)

    deduped: dict[tuple[str, str], dict] = {}
    for item in result_list:
        dedupe_key = (item.get("pod") or item["service"], item.get("resource") or "")
        previous = deduped.get(dedupe_key)
        if previous is None or (item["depth"], item["score"]) > (previous["depth"], previous["score"]):
            deduped[dedupe_key] = item

    return sorted(deduped.values(), key=lambda item: (item["score"], item["depth"]), reverse=True)


def actual_pattern_ranker(
    abnormal_patterns: dict[str, float],
    normal_event_depth_index: dict[str, tuple[int, str]],
    metric_events: list[dict],
) -> list[dict]:
    alarm_lookup = build_alarm_lookup(metric_events)
    results = []
    for key, value in abnormal_patterns.items():
        labels = pattern_labels_from_key(key)
        source_label, target_label = split_pattern_key(key)
        if is_trivial_expected_pattern(source_label, target_label):
            continue
        depth, pod = get_pattern_depth_and_pod(labels, normal_event_depth_index)
        service = select_service_from_labels(labels, normal_event_depth_index, pod)
        if not service:
            service = "unknown"
        item = {
            "events": key,
            "labels": labels,
            "score": float(value),
            "depth": int(depth) if depth > 0 else 1,
            "pod": pod,
            "service": service,
            "service_candidates": service_candidates_from_labels(labels),
        }
        resource_label = select_result_resource(labels, pod, service, alarm_lookup)
        if resource_label:
            item["resource"] = resource_label
        results.append(item)
    return sorted(results, key=lambda item: (-item["score"], -item["depth"], item["events"]))


def build_pair_candidates(expected_patterns: list[dict], actual_patterns: list[dict]) -> list[dict]:
    grouped: defaultdict[str, dict[str, list[dict]]] = defaultdict(lambda: {"expected": [], "actual": []})
    for item in expected_patterns:
        services = item.get("service_candidates") or [item.get("service")]
        valid_services = [str(service) for service in services if str(service or "").strip() and str(service) != "unknown"]
        if not valid_services:
            continue
        shared_score = float(item.get("score", 0.0)) / len(valid_services) if ALIGNMENT_MODE == "repo_aligned" else float(item.get("score", 0.0))
        for service in valid_services:
            grouped[service]["expected"].append({**item, "_score_for_group": shared_score})
    for item in actual_patterns:
        services = item.get("service_candidates") or [item.get("service")]
        valid_services = [str(service) for service in services if str(service or "").strip() and str(service) != "unknown"]
        if not valid_services:
            continue
        shared_score = float(item.get("score", 0.0)) / len(valid_services) if ALIGNMENT_MODE == "repo_aligned" else float(item.get("score", 0.0))
        for service in valid_services:
            grouped[service]["actual"].append({**item, "_score_for_group": shared_score})

    candidates = []
    for service, bucket in grouped.items():
        expected_items = sorted(
            bucket["expected"],
            key=lambda item: (-float(item.get("_score_for_group", item["score"])), -item.get("depth", 0), item["events"]),
        )
        actual_items = sorted(
            bucket["actual"],
            key=lambda item: (-float(item.get("_score_for_group", item["score"])), -item.get("depth", 0), item["events"]),
        )
        expected_score = float(sum(float(item.get("_score_for_group", item["score"])) for item in expected_items[:3]))
        actual_score = float(sum(float(item.get("_score_for_group", item["score"])) for item in actual_items[:3]))
        overlap_bonus = 1.0 if expected_items and actual_items else 0.0
        resources = sorted(
            {
                str(item.get("resource"))
                for item in expected_items[:3] + actual_items[:3]
                if item.get("resource")
            }
        )
        candidates.append(
            {
                "service": service,
                "expected_score": expected_score,
                "actual_score": actual_score,
                "combined_score": expected_score + actual_score + overlap_bonus,
                "expected_pattern_count": len(expected_items),
                "actual_pattern_count": len(actual_items),
                "resource_hints": resources,
                "expected_patterns": [item["events"] for item in expected_items[:3]],
                "actual_patterns": [item["events"] for item in actual_items[:3]],
            }
        )
    return sorted(candidates, key=lambda item: (-item["combined_score"], -item["actual_score"], -item["expected_score"], item["service"]))


def pair_candidate_service_ranking(pair_candidates: list[dict]) -> list[tuple[str, float]]:
    if ALIGNMENT_MODE == "repo_aligned":
        valid_candidates = [
            item
            for item in pair_candidates
            if str(item.get("service") or "").strip()
        ]
        if not valid_candidates:
            return []

        has_actual_signal = any(
            float(item.get("actual_score", 0.0)) > 0.0 or int(item.get("actual_pattern_count", 0)) > 0
            for item in valid_candidates
        )
        if has_actual_signal:
            ranked_candidates = sorted(
                valid_candidates,
                key=lambda item: (
                    -(
                        float(item.get("actual_score", 0.0))
                        - REPO_ALIGNED_PAIR_EXPECTED_PENALTY * float(item.get("expected_score", 0.0))
                    ),
                    -float(item.get("actual_score", 0.0)),
                    -int(item.get("actual_pattern_count", 0)),
                    -float(item.get("combined_score", 0.0)),
                    str(item.get("service", "")),
                ),
            )
            return [
                (
                    str(item["service"]),
                    float(item.get("actual_score", 0.0))
                    - REPO_ALIGNED_PAIR_EXPECTED_PENALTY * float(item.get("expected_score", 0.0)),
                )
                for item in ranked_candidates
            ]

        ranked_candidates = sorted(
            valid_candidates,
            key=lambda item: (
                -float(item.get("combined_score", 0.0)),
                -float(item.get("actual_score", 0.0)),
                -float(item.get("expected_score", 0.0)),
                str(item.get("service", "")),
            ),
        )
        return [
            (str(item["service"]), float(item.get("combined_score", 0.0)))
            for item in ranked_candidates
        ]

    ranking = [
        (str(item["service"]), float(item["combined_score"]))
        for item in pair_candidates
        if str(item.get("service") or "").strip()
    ]
    return ranking


def event_label_to_service(event_label: str) -> str:
    if event_label.startswith("log_") or is_resource_event_label(event_label):
        return ""
    return event_label.split(" ", 1)[0].strip()


def abnormal_service_ranking(
    abnormal_patterns: dict[str, float],
    normal_event_depth_index: dict[str, tuple[int, str]],
) -> list[tuple[str, float]]:
    service_scores: defaultdict[str, dict[str, float]] = defaultdict(lambda: {"score": 0.0, "depth": 0.0, "count": 0.0})
    for key, score in abnormal_patterns.items():
        labels = pattern_labels_from_key(key)
        source_label, target_label = split_pattern_key(key)
        if is_trivial_expected_pattern(source_label, target_label):
            continue

        candidates = [(label, event_label_to_service(label)) for label in labels]
        candidates = [(label, service) for label, service in candidates if service]
        if not candidates:
            continue

        service_depths: dict[str, int] = {}
        for label, service in candidates:
            service_depths[service] = max(
                service_depths.get(service, 0),
                int(normal_event_depth_index.get(label, (1, ""))[0]),
            )

        shared_score = float(score) / max(len(service_depths), 1)
        for service, depth in service_depths.items():
            service_scores[service]["score"] += shared_score
            service_scores[service]["depth"] = max(service_scores[service]["depth"], float(depth))
            service_scores[service]["count"] += 1.0
    ranked = sorted(
        service_scores.items(),
        key=lambda item: (-item[1]["score"], -item[1]["depth"], -item[1]["count"], item[0]),
    )
    return [(service, values["score"]) for service, values in ranked]


def reciprocal_rank_fusion(rankings: list[list[tuple[str, float]]], weights: list[float]) -> list[tuple[str, float]]:
    fused_scores: defaultdict[str, float] = defaultdict(float)
    for ranking, weight in zip(rankings, weights):
        for rank, (service, _) in enumerate(ranking, start=1):
            fused_scores[service] += weight / (RRF_K + rank)
    return sorted(fused_scores.items(), key=lambda item: (-item[1], item[0]))


def aggregate_service_scores(result_list: list[dict]) -> list[tuple[str, float]]:
    service_scores: defaultdict[str, dict[str, float]] = defaultdict(lambda: {"score": 0.0, "depth": 0.0, "count": 0.0})
    for item in result_list:
        service = str(item.get("service") or "unknown")
        if service == "unknown":
            continue
        service_scores[service]["score"] += float(item.get("score", 0.0))
        service_scores[service]["depth"] = max(service_scores[service]["depth"], float(item.get("depth", 0.0)))
        service_scores[service]["count"] += 1.0
    ranked = sorted(
        service_scores.items(),
        key=lambda item: (-item[1]["score"], -item[1]["depth"], -item[1]["count"], item[0]),
    )
    return [(service, values["score"]) for service, values in ranked]


def evaluate_topk(service_ranking: list[tuple[str, float]], ground_truth_service: str) -> dict:
    ranked_services = [service for service, _ in service_ranking]
    return {
        "predicted_service_top1": ranked_services[0] if ranked_services else None,
        "predicted_service_top3": ranked_services[:3],
        "predicted_service_top5": ranked_services[:5],
        "service_top1_hit": ground_truth_service in ranked_services[:1],
        "service_top3_hit": ground_truth_service in ranked_services[:3],
        "service_top5_hit": ground_truth_service in ranked_services[:5],
    }


def build_repo_alignment_comparison(
    script_dir: Path,
    telemetry_day: str,
    details_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> dict | None:
    if ALIGNMENT_MODE != "repo_aligned":
        return None

    baseline_dir = (script_dir / BASELINE_OUTPUT_ROOT / telemetry_day).resolve()
    baseline_details_path = baseline_dir / DAY_DETAILS_FILE
    baseline_summary_path = baseline_dir / DAY_SUMMARY_FILE
    if not baseline_details_path.exists() or not baseline_summary_path.exists():
        return None

    baseline_details_df = pd.read_csv(baseline_details_path)
    baseline_summary_df = pd.read_csv(baseline_summary_path)
    if baseline_summary_df.empty:
        return None

    repo_summary = summary_df.iloc[0].to_dict() if not summary_df.empty else {}
    baseline_summary = baseline_summary_df.iloc[0].to_dict()
    baseline_top1 = baseline_details_df["predicted_service_top1"].fillna("None").astype(str).value_counts().to_dict()
    repo_top1 = details_df["predicted_service_top1"].fillna("None").astype(str).value_counts().to_dict() if not details_df.empty else {}
    comparison = {
        "telemetry_day": telemetry_day,
        "baseline_output_root": str(baseline_dir),
        "repo_aligned_output_root": str((script_dir / OUTPUT_ROOT / telemetry_day).resolve()),
        "baseline_service_top1_accuracy": float(baseline_summary.get("service_top1_accuracy", np.nan)),
        "repo_aligned_service_top1_accuracy": float(repo_summary.get("service_top1_accuracy", np.nan)),
        "baseline_service_top3_accuracy": float(baseline_summary.get("service_top3_accuracy", np.nan)),
        "repo_aligned_service_top3_accuracy": float(repo_summary.get("service_top3_accuracy", np.nan)),
        "baseline_service_top5_accuracy": float(baseline_summary.get("service_top5_accuracy", np.nan)),
        "repo_aligned_service_top5_accuracy": float(repo_summary.get("service_top5_accuracy", np.nan)),
        "baseline_top1_counts": baseline_top1,
        "repo_aligned_top1_counts": repo_top1,
    }
    return comparison


def save_normal_baseline(
    output_dir: Path,
    telemetry_day: str,
    requested_exp_id: str | None,
    baseline: dict,
    normal_patterns: dict[str, int],
    normal_run_count: int,
    normal_graph_count: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    serializable_baseline = {
        f"{service}|{metric}": {"mean": mean, "std": std}
        for (service, metric), (mean, std) in baseline.items()
    }
    (output_dir / METRIC_BASELINE_FILE).write_text(json.dumps(serializable_baseline, indent=2), encoding="utf-8")
    (output_dir / NORMAL_PATTERN_FILE).write_text(json.dumps(normal_patterns, indent=2), encoding="utf-8")
    (output_dir / NORMAL_SUMMARY_FILE).write_text(
        json.dumps(
            {
                "telemetry_day": telemetry_day,
                "requested_exp_id": requested_exp_id,
                "normal_run_count": normal_run_count,
                "normal_graph_count": normal_graph_count,
                "important_metrics": IMPORTANT_METRICS,
                "rca_resource_metrics": list(RCA_RESOURCE_METRICS),
                "rca_symptom_metrics": list(RCA_SYMPTOM_METRICS),
                "include_symptom_metric_events": INCLUDE_SYMPTOM_METRIC_EVENTS,
                "z_threshold": Z_THRESHOLD,
                "pattern_min_count": PATTERN_MIN_COUNT,
                "pattern_min_score": PATTERN_MIN_SCORE,
                "pattern_method": active_pattern_method_name(),
                "alignment_mode": ALIGNMENT_MODE,
                "expected_rank_weight": EXPECTED_RANK_WEIGHT,
                "abnormal_rank_weight": ABNORMAL_RANK_WEIGHT,
                "pair_rank_weight": PAIR_RANK_WEIGHT,
                "service_ranking_fusion": active_service_ranking_fusion_name(),
                "pair_ranking_strategy": active_pair_ranking_strategy_name(),
                "log_template_method": "drain3_raw_log_fallback_message",
                "log_background_template_weight": BACKGROUND_LOG_TEMPLATE_WEIGHT,
                "normal_window_duration_minutes": NORMAL_WINDOW_DURATION_MINUTES,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def save_fault_outputs(
    output_dir: Path,
    pattern_scores: dict,
    service_ranking: list[tuple[str, float]],
    expected_patterns: list[dict],
    actual_patterns: list[dict],
    pair_candidates: list[dict],
    run_summary: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / PATTERN_SCORES_FILE).write_text(json.dumps(pattern_scores, indent=2), encoding="utf-8")
    (output_dir / EXPECTED_PATTERNS_FILE).write_text(json.dumps(expected_patterns, indent=2), encoding="utf-8")
    (output_dir / ACTUAL_PATTERNS_FILE).write_text(json.dumps(actual_patterns, indent=2), encoding="utf-8")
    (output_dir / PAIR_CANDIDATES_FILE).write_text(json.dumps(pair_candidates, indent=2), encoding="utf-8")
    pd.DataFrame(service_ranking, columns=["service", "score"]).to_csv(output_dir / SERVICE_RANKING_FILE, index=False)
    (output_dir / RUN_SUMMARY_FILE).write_text(json.dumps(run_summary, indent=2, ensure_ascii=False), encoding="utf-8")


def build_day_outputs(
    output_dir: Path,
    telemetry_day: str,
    requested_exp_id: str | None,
    detail_rows: list[dict],
    total_runtime_seconds: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    details_df = pd.DataFrame(detail_rows).sort_values("exp_id").reset_index(drop=True)
    details_df.to_csv(output_dir / DAY_DETAILS_FILE, index=False)

    n_total = len(details_df)
    summary_df = pd.DataFrame(
        [
            {
                "telemetry_day": telemetry_day,
                "telemetry_day_suffix": telemetry_day_to_suffix(telemetry_day),
                "requested_exp_id": requested_exp_id,
                "n_total": n_total,
                "n_ok": int(details_df["predicted_service_top1"].notna().sum()) if n_total else 0,
                "n_error": int(details_df["predicted_service_top1"].isna().sum()) if n_total else 0,
                "service_top1_accuracy": float(details_df["service_top1_hit"].fillna(False).mean()) if n_total else np.nan,
                "service_top3_accuracy": float(details_df["service_top3_hit"].fillna(False).mean()) if n_total else np.nan,
                "service_top5_accuracy": float(details_df["service_top5_hit"].fillna(False).mean()) if n_total else np.nan,
                "total_runtime_seconds": total_runtime_seconds,
                "avg_runtime_per_exception_seconds": total_runtime_seconds / n_total if n_total else np.nan,
                "sum_of_individual_runtime_seconds": float(details_df["runtime_seconds"].fillna(0).sum()) if n_total else np.nan,
                "avg_runtime_per_processed_exception_seconds": float(details_df["runtime_seconds"].mean()) if n_total else np.nan,
                "important_metrics": ",".join(IMPORTANT_METRICS),
                "rca_resource_metrics": ",".join(RCA_RESOURCE_METRICS),
                "rca_symptom_metrics": ",".join(RCA_SYMPTOM_METRICS),
                "include_symptom_metric_events": INCLUDE_SYMPTOM_METRIC_EVENTS,
                "z_threshold": Z_THRESHOLD,
                "pattern_min_count": PATTERN_MIN_COUNT,
                "pattern_min_score": PATTERN_MIN_SCORE,
                "pattern_method": active_pattern_method_name(),
                "alignment_mode": ALIGNMENT_MODE,
                "expected_rank_weight": EXPECTED_RANK_WEIGHT,
                "abnormal_rank_weight": ABNORMAL_RANK_WEIGHT,
                "pair_rank_weight": PAIR_RANK_WEIGHT,
                "service_ranking_fusion": active_service_ranking_fusion_name(),
                "pair_ranking_strategy": active_pair_ranking_strategy_name(),
                "log_template_method": "drain3_raw_log_fallback_message",
                "normal_window_duration_minutes": NORMAL_WINDOW_DURATION_MINUTES,
                "days_count": 1,
            }
        ]
    )
    summary_df.to_csv(output_dir / DAY_SUMMARY_FILE, index=False)
    return details_df, summary_df


def build_all_days_outputs(
    output_root: Path,
    all_details_dfs: list[pd.DataFrame],
    all_summary_dfs: list[pd.DataFrame],
    total_script_runtime_seconds: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    combined_details_df = pd.concat(all_details_dfs, ignore_index=True) if all_details_dfs else pd.DataFrame()
    combined_summary_df = pd.concat(all_summary_dfs, ignore_index=True) if all_summary_dfs else pd.DataFrame()

    if combined_details_df.empty:
        overall_summary_df = pd.DataFrame(
            columns=[
                "telemetry_day",
                "telemetry_day_suffix",
                "requested_exp_id",
                "n_total",
                "n_ok",
                "n_error",
                "service_top1_accuracy",
                "service_top3_accuracy",
                "service_top5_accuracy",
                "total_runtime_seconds",
                "avg_runtime_per_exception_seconds",
                "sum_of_individual_runtime_seconds",
                "avg_runtime_per_processed_exception_seconds",
                "important_metrics",
                "rca_resource_metrics",
                "rca_symptom_metrics",
                "include_symptom_metric_events",
                "z_threshold",
                "pattern_min_count",
                "pattern_min_score",
                "pattern_method",
                "alignment_mode",
                "expected_rank_weight",
                "abnormal_rank_weight",
                "pair_rank_weight",
                "service_ranking_fusion",
                "pair_ranking_strategy",
                "log_template_method",
                "normal_window_duration_minutes",
                "days_count",
            ]
        )
    else:
        n_total = len(combined_details_df)
        processed_days = int(combined_details_df["telemetry_day"].nunique()) if "telemetry_day" in combined_details_df.columns else len(all_summary_dfs)
        overall_summary_df = pd.DataFrame(
            [
                {
                    "telemetry_day": "ALL_DAYS",
                    "telemetry_day_suffix": "",
                    "requested_exp_id": combined_details_df["requested_exp_id"].dropna().iloc[0]
                    if "requested_exp_id" in combined_details_df.columns and combined_details_df["requested_exp_id"].dropna().any()
                    else None,
                    "n_total": n_total,
                    "n_ok": int(combined_details_df["predicted_service_top1"].notna().sum()),
                    "n_error": int(combined_details_df["predicted_service_top1"].isna().sum()),
                    "service_top1_accuracy": float(combined_details_df["service_top1_hit"].fillna(False).mean()),
                    "service_top3_accuracy": float(combined_details_df["service_top3_hit"].fillna(False).mean()),
                    "service_top5_accuracy": float(combined_details_df["service_top5_hit"].fillna(False).mean()),
                    "total_runtime_seconds": total_script_runtime_seconds,
                    "avg_runtime_per_exception_seconds": total_script_runtime_seconds / n_total if n_total else np.nan,
                    "sum_of_individual_runtime_seconds": float(combined_details_df["runtime_seconds"].fillna(0).sum()),
                    "avg_runtime_per_processed_exception_seconds": float(combined_details_df["runtime_seconds"].mean()),
                    "important_metrics": ",".join(IMPORTANT_METRICS),
                    "rca_resource_metrics": ",".join(RCA_RESOURCE_METRICS),
                    "rca_symptom_metrics": ",".join(RCA_SYMPTOM_METRICS),
                    "include_symptom_metric_events": INCLUDE_SYMPTOM_METRIC_EVENTS,
                    "z_threshold": Z_THRESHOLD,
                    "pattern_min_count": PATTERN_MIN_COUNT,
                    "pattern_min_score": PATTERN_MIN_SCORE,
                    "pattern_method": active_pattern_method_name(),
                    "alignment_mode": ALIGNMENT_MODE,
                    "expected_rank_weight": EXPECTED_RANK_WEIGHT,
                    "abnormal_rank_weight": ABNORMAL_RANK_WEIGHT,
                    "pair_rank_weight": PAIR_RANK_WEIGHT,
                    "service_ranking_fusion": active_service_ranking_fusion_name(),
                    "pair_ranking_strategy": active_pair_ranking_strategy_name(),
                    "log_template_method": "drain3_raw_log_fallback_message",
                    "normal_window_duration_minutes": NORMAL_WINDOW_DURATION_MINUTES,
                    "days_count": processed_days,
                }
            ]
        )

    output_root.mkdir(parents=True, exist_ok=True)
    combined_details_df.to_csv(output_root / ALL_DAYS_DETAILS_FILE, index=False)
    combined_summary_df.to_csv(output_root / ALL_DAYS_SUMMARY_FILE, index=False)
    overall_summary_df.to_csv(output_root / OVERALL_SUMMARY_FILE, index=False)
    return combined_details_df, combined_summary_df, overall_summary_df


def run_single_day(
    script_dir: Path,
    telemetry_day: str,
    exp_id: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    total_start = perf_counter()
    fault_root, normal_root, metrics_dir, logs_dir, traces_dir = get_day_paths(script_dir, telemetry_day)
    _ = fault_root, normal_root
    fault_runs = discover_fault_runs(script_dir, telemetry_day)
    if exp_id:
        fault_runs = [run for run in fault_runs if run.run_id == exp_id]
        if not fault_runs:
            print(f"[WARN] EXP_ID not found for {telemetry_day}: {exp_id}; skipping day")
            return pd.DataFrame(), pd.DataFrame()
    normal_runs = discover_normal_runs(script_dir, telemetry_day)
    day_output_dir = (script_dir / OUTPUT_ROOT / telemetry_day).resolve()

    print(
        f"[INFO] telemetry_day={telemetry_day} "
        f"fault_runs={len(fault_runs)} normal_runs={len(normal_runs)}"
    )

    metrics_cache = HourlyMetricsCache()
    logs_cache = HourlyLogsCache()
    traces_cache = HourlyTracesCache()

    metric_baseline = build_metric_baseline(normal_runs, metrics_dir, metrics_cache)
    print(f"[INFO] Metric baseline built: keys={len(metric_baseline)}")

    normal_patterns_accumulator: defaultdict[str, int] = defaultdict(int)
    normal_graph_count = 0
    normal_event_depth_index: dict[str, tuple[int, str]] = {}
    known_services: set[str] = set()
    for index, normal_run in enumerate(normal_runs, start=1):
        traces_df = load_traces_for_run(traces_cache, traces_dir, normal_run)
        logs_df = load_logs_for_run(logs_cache, logs_dir, normal_run)
        metrics_df = load_metrics_for_run(metrics_cache, metrics_dir, normal_run)
        known_services.update(collect_known_services(traces_df, logs_df, metrics_df))
        metric_events = extract_metric_anomalies(metrics_df, metric_baseline)
        run_patterns, run_graph_count, run_depth_index = collect_pattern_support_and_depth_index(
            traces_df,
            logs_df,
            metric_events,
            include_depth_index=True,
        )
        for key, value in run_patterns.items():
            normal_patterns_accumulator[key] += value
        normal_graph_count += run_graph_count
        for label, (depth, pod) in run_depth_index.items():
            previous = normal_event_depth_index.get(label)
            if previous is None or depth > previous[0]:
                normal_event_depth_index[label] = (depth, pod)
        print(f"[INFO] Loaded normal run {index}/{len(normal_runs)}: {normal_run.run_id} graphs={run_graph_count}")

    if normal_graph_count == 0:
        raise ValueError("No normal graphs were generated.")

    normal_patterns = dict(sorted(normal_patterns_accumulator.items(), key=lambda item: (-item[1], item[0])))
    save_normal_baseline(
        day_output_dir / NORMAL_BASELINE_DIR,
        telemetry_day,
        exp_id,
        metric_baseline,
        normal_patterns,
        len(normal_runs),
        normal_graph_count,
    )

    detail_rows = []
    for index, fault_run in enumerate(fault_runs, start=1):
        run_start = perf_counter()
        print(f"[INFO] [{index}/{len(fault_runs)}] Running {fault_run.run_id}")
        try:
            traces_df = load_traces_for_run(traces_cache, traces_dir, fault_run)
            logs_df = load_logs_for_run(logs_cache, logs_dir, fault_run)
            metrics_df = load_metrics_for_run(metrics_cache, metrics_dir, fault_run)
            run_valid_services = set(known_services)
            run_valid_services.update(collect_known_services(traces_df, logs_df, metrics_df))

            metric_events = extract_metric_anomalies(metrics_df, metric_baseline)
            fault_patterns, fault_graph_count, _ = collect_pattern_support_and_depth_index(
                traces_df,
                logs_df,
                metric_events,
                include_depth_index=False,
            )
            pattern_scores = compute_pattern_scores(normal_patterns, fault_patterns)
            abnormal_patterns = abnormal_pattern_ranker(
                normal_patterns,
                fault_patterns,
                min_score=PATTERN_MIN_SCORE,
                min_count=PATTERN_MIN_COUNT,
            )
            expected_patterns = expected_pattern_ranker(
                normal_patterns,
                fault_patterns,
                normal_event_depth_index,
                metric_events,
                min_score=PATTERN_MIN_SCORE,
                min_count=PATTERN_MIN_COUNT,
            )
            actual_patterns = actual_pattern_ranker(
                abnormal_patterns,
                normal_event_depth_index,
                metric_events,
            )
            pair_candidates = build_pair_candidates(expected_patterns, actual_patterns)
            pair_service_ranking = pair_candidate_service_ranking(pair_candidates)
            expected_service_ranking = aggregate_service_scores(expected_patterns)
            abnormal_service_scores = abnormal_service_ranking(abnormal_patterns, normal_event_depth_index)
            if ALIGNMENT_MODE == "repo_aligned":
                if pair_service_ranking:
                    service_ranking = pair_service_ranking
                elif expected_service_ranking:
                    service_ranking = expected_service_ranking
                else:
                    service_ranking = abnormal_service_scores
            else:
                ranking_inputs: list[list[tuple[str, float]]] = []
                ranking_weights: list[float] = []
                if expected_service_ranking:
                    ranking_inputs.append(expected_service_ranking)
                    ranking_weights.append(EXPECTED_RANK_WEIGHT)
                if abnormal_service_scores:
                    ranking_inputs.append(abnormal_service_scores)
                    ranking_weights.append(ABNORMAL_RANK_WEIGHT)
                if pair_service_ranking:
                    ranking_inputs.append(pair_service_ranking)
                    ranking_weights.append(PAIR_RANK_WEIGHT)

                if len(ranking_inputs) >= 2:
                    service_ranking = reciprocal_rank_fusion(
                        ranking_inputs,
                        ranking_weights,
                    )
                elif expected_service_ranking:
                    service_ranking = expected_service_ranking
                elif pair_service_ranking:
                    service_ranking = pair_service_ranking
                else:
                    service_ranking = abnormal_service_scores
            service_ranking = filter_service_ranking(service_ranking, run_valid_services)

            runtime_seconds = perf_counter() - run_start
            eval_result = evaluate_topk(service_ranking, fault_run.ground_truth_service or "")
            run_summary = {
                "telemetry_day": telemetry_day,
                "requested_exp_id": exp_id,
                "exp_id": fault_run.run_id,
                "ground_truth_service": fault_run.ground_truth_service,
                "fault_start": str(fault_run.start),
                "fault_end": str(fault_run.end),
                "inject_start": str(fault_run.inject_start) if fault_run.inject_start is not None else None,
                "fault_metadata_path": str(fault_run.metadata_path),
                "trace_row_count": len(traces_df),
                "log_row_count": len(logs_df),
                "metric_anomaly_event_count": len(metric_events),
                "fault_graph_count": fault_graph_count,
                "fault_pattern_count": len(fault_patterns),
                "abnormal_pattern_count": len(abnormal_patterns),
                "expected_pattern_count": len(expected_patterns),
                "actual_pattern_count": len(actual_patterns),
                "pair_candidate_count": len(pair_candidates),
                "expected_service_ranking": expected_service_ranking,
                "abnormal_service_ranking": abnormal_service_scores,
                "abnormal_patterns": abnormal_patterns,
                "expected_patterns": expected_patterns,
                "actual_patterns": actual_patterns,
                "pair_candidates": pair_candidates,
                "pair_ranking_strategy": active_pair_ranking_strategy_name(),
                "runtime_seconds": runtime_seconds,
                **eval_result,
            }
            save_fault_outputs(
                day_output_dir / fault_run.run_id,
                pattern_scores,
                service_ranking,
                expected_patterns,
                actual_patterns,
                pair_candidates,
                run_summary,
            )
            detail_rows.append(
                {
                    "telemetry_day": telemetry_day,
                    "requested_exp_id": exp_id,
                    "exp_id": fault_run.run_id,
                    "ground_truth_service": fault_run.ground_truth_service,
                    "fault_graph_count": fault_graph_count,
                    "fault_pattern_count": len(fault_patterns),
                    "expected_pattern_count": len(expected_patterns),
                    "actual_pattern_count": len(actual_patterns),
                    "pair_candidate_count": len(pair_candidates),
                    "runtime_seconds": runtime_seconds,
                    **eval_result,
                }
            )
            print(
                f"[OK] {fault_run.run_id} top1={eval_result['service_top1_hit']} "
                f"top3={eval_result['service_top3_hit']} top5={eval_result['service_top5_hit']} "
                f"runtime={runtime_seconds:.2f}s graphs={fault_graph_count}"
            )
        except Exception as exc:
            runtime_seconds = perf_counter() - run_start
            detail_rows.append(
                {
                    "telemetry_day": telemetry_day,
                    "requested_exp_id": exp_id,
                    "exp_id": fault_run.run_id,
                    "ground_truth_service": fault_run.ground_truth_service,
                    "fault_graph_count": 0,
                    "fault_pattern_count": 0,
                    "expected_pattern_count": 0,
                    "actual_pattern_count": 0,
                    "pair_candidate_count": 0,
                    "predicted_service_top1": None,
                    "predicted_service_top3": [],
                    "predicted_service_top5": [],
                    "service_top1_hit": False,
                    "service_top3_hit": False,
                    "service_top5_hit": False,
                    "runtime_seconds": runtime_seconds,
                    "error": str(exc),
                }
            )
            print(f"[WARN] {fault_run.run_id} failed: {exc}")

    total_runtime_seconds = perf_counter() - total_start
    details_df, summary_df = build_day_outputs(day_output_dir, telemetry_day, exp_id, detail_rows, total_runtime_seconds)
    comparison = build_repo_alignment_comparison(script_dir, telemetry_day, details_df, summary_df)
    if comparison is not None:
        (day_output_dir / COMPARISON_FILE).write_text(json.dumps(comparison, indent=2), encoding="utf-8")
        print(f"[INFO] Repo-aligned comparison saved: {(day_output_dir / COMPARISON_FILE).resolve()}")
        print(
            f"[INFO] Compare top1 old={comparison['baseline_service_top1_accuracy']:.4f} "
            f"repo_aligned={comparison['repo_aligned_service_top1_accuracy']:.4f}"
        )

    if detail_rows:
        print("\n[DONE] Nezha day summary")
        print(
            details_df[["service_top1_hit", "service_top3_hit", "service_top5_hit"]]
            .fillna(False)
            .mean()
            .rename(
                {
                    "service_top1_hit": "service_top1_accuracy",
                    "service_top3_hit": "service_top3_accuracy",
                    "service_top5_hit": "service_top5_accuracy",
                }
            )
            .to_frame()
            .T
            .to_string(index=False)
        )
        print(f"[INFO] Total runtime: {total_runtime_seconds:.2f}s")
        print(f"[INFO] Avg runtime per anomaly: {total_runtime_seconds / len(detail_rows):.2f}s")
    return details_df, summary_df


def main() -> None:
    args = parse_args()
    telemetry_days = [args.telemetry_day] if args.telemetry_day else TELEMETRY_DAYS
    exp_id = args.exp_id
    script_start = perf_counter()
    output_root = (SCRIPT_DIR / OUTPUT_ROOT).resolve()

    all_details_dfs: list[pd.DataFrame] = []
    all_summary_dfs: list[pd.DataFrame] = []

    for telemetry_day in telemetry_days:
        details_df, summary_df = run_single_day(SCRIPT_DIR, telemetry_day, exp_id)
        if not details_df.empty:
            all_details_dfs.append(details_df)
        if not summary_df.empty:
            all_summary_dfs.append(summary_df)

    total_script_runtime_seconds = perf_counter() - script_start
    _, _, overall_summary_df = build_all_days_outputs(
        output_root=output_root,
        all_details_dfs=all_details_dfs,
        all_summary_dfs=all_summary_dfs,
        total_script_runtime_seconds=total_script_runtime_seconds,
    )

    print(f"[DONE] Multi-day details saved: {(output_root / ALL_DAYS_DETAILS_FILE).resolve()}")
    print(f"[DONE] Multi-day summary saved: {(output_root / ALL_DAYS_SUMMARY_FILE).resolve()}")
    print(f"[DONE] Overall summary saved: {(output_root / OVERALL_SUMMARY_FILE).resolve()}")
    if not overall_summary_df.empty:
        print("\n[DONE] Nezha overall summary")
        print(overall_summary_df.to_string(index=False))
    print(f"[DONE] End-to-end script runtime across {len(telemetry_days)} day(s): {total_script_runtime_seconds:.2f}s")


if __name__ == "__main__":
    main()
