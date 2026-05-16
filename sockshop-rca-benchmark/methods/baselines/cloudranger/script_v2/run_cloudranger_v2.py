import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter

SCRIPT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str((SCRIPT_DIR / ".mplconfig").resolve()))

import numpy as np
import pandas as pd
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz


# ===============================
# CONFIG
# ===============================

TELEMETRY_DAYS = ["2026_03_12", "2026_03_13", "2026_03_14", "2026_03_17", "2026_03_18"]
EXP_ID = None  # Set to an exp_id string to run a single anomaly case.

SUPPORTED_METRICS = (
    "request_rate",
    "latency_p50",
    "latency_p90",
    "latency_p95",
    "latency_p99",
)

MULTI_METRIC_FUSION_MODES = (
    "avg_rank",
    "avg_score",
)

SEARCH_PROFILE = "paper"  # "paper" | "paper_compare" | "best" | "fast" | "focused" | "balanced" | "exhaustive"
OPTIMIZATION_TARGET = "top1"  # "top1" | "balanced" | "top5"

AGGREGATION_SECONDS = 5
FAULT_WINDOW_MINUTES = 10
NORMAL_WINDOW_OFFSET_MINUTES = 2
NORMAL_WINDOW_DURATION_MINUTES = 10
FAULT_WINDOW_SOURCE = "inject_window"  # "inject_window" | "workload_if_available"

PC_ALPHA = 0.1
PC_ALPHA_VALUES = (0.1,)  # Set to (0.1, 0.3, 0.5) to sweep the paper-style PC threshold.
VARIANCE_EPS = 1e-10
MAX_NODES = None

ANOMALY_Z_THRESHOLD = 1.0
ANOMALY_SCORE_PERCENTILE = 95
ANOMALY_PERCENTILE_THRESHOLD = 90.0
ANOMALY_TOP_K = 8
PAPER_CANDIDATE_PERCENTILE = 80.0
MIN_CANDIDATES = 5
FRONTEND_SERVICE = "front-end"
PAPER_COMPARE_METRICS = (("latency_p95",), ("latency_p99",))
PAPER_COMPARE_CANDIDATE_PERCENTILES = (70.0, 80.0)
PAPER_COMPARE_MIN_CANDIDATES = (5, 8)

BETA = 0.5
RHO = 0.5
WALK_STEPS = 20_000
RANDOM_SEED = 20260320

OUTPUT_ROOT = "../data_v2"
DAY_DETAILS_FILE = "cloudranger_accuracy_details.csv"
DAY_SUMMARY_FILE = "cloudranger_accuracy_summary.csv"
ABLATION_SUMMARY_FILE = "cloudranger_ablation_summary.csv"
BEST_CONFIG_FILE = "cloudranger_best_config.json"
ALL_DAYS_DETAILS_FILE = "cloudranger_accuracy_details_all_days.csv"
ALL_DAYS_SUMMARY_FILE = "cloudranger_accuracy_summary_all_days.csv"
OVERALL_SUMMARY_FILE = "cloudranger_accuracy_summary_overall.csv"
ALL_DAYS_ABLATION_FILE = "cloudranger_ablation_summary_all_days.csv"
ALL_DAYS_BEST_CONFIG_FILE = "cloudranger_best_config_all_days.json"
RUN_SUMMARY_FILE = "cloudranger_run_summary.json"
SERVICE_RANKING_FILE = "service_ranking.csv"
VARIABLE_RANKING_FILE = "variable_ranking.csv"
IMPACT_GRAPH_FILE = "impact_graph.csv"
CANDIDATE_FILE = "candidate_services.csv"
NORMAL_BASELINE_DIR = "normal_baseline"
NORMAL_MEAN_FILE = "normal_mean.csv"
NORMAL_STD_FILE = "normal_std.csv"
NORMAL_SCORE_FILE = "normal_baseline_summary.json"

FAULT_RUN_ROOT_TEMPLATE = "../../../dataset_v2/fault_run_{day_suffix}"
NORMAL_RUN_ROOT_TEMPLATE = "../../../dataset_v2/normal_run_{day_suffix}"
TELEMETRY_METRICS_DIR_TEMPLATE = "../../../dataset_v2/telemetry/{telemetry_day}/metrics"

DEPLOYMENT_HASH_RE = re.compile(r"^[a-f0-9]{8,}$")
POD_SUFFIX_RE = re.compile(r"^[a-z0-9]{4,6}$")
SLUG_RE = re.compile(r"[^a-z0-9._-]+")


@dataclass(frozen=True)
class RunWindow:
    run_id: str
    label: str
    start: pd.Timestamp
    end: pd.Timestamp
    metadata_path: Path
    inject_start: pd.Timestamp | None = None
    ground_truth_service: str | None = None


@dataclass(frozen=True)
class CandidateConfig:
    mode: str
    top_k: int | None = None
    percentile: float | None = None
    zscore_threshold: float | None = None
    min_candidates: int | None = None

    def slug(self) -> str:
        if self.mode == "top_k":
            base = f"topk-{self.top_k}"
        elif self.mode == "percentile":
            base = f"pct-{self.percentile:g}"
        elif self.mode == "zscore":
            base = f"z-{self.zscore_threshold:g}"
        else:
            raise ValueError(f"Unsupported candidate mode: {self.mode}")
        if self.min_candidates is not None:
            base += f"-min-{self.min_candidates}"
        return base


@dataclass(frozen=True)
class ExperimentConfig:
    metrics: tuple[str, ...]
    fusion_mode: str
    service_agg: str
    candidate: CandidateConfig
    pc_alpha: float

    @property
    def metrics_label(self) -> str:
        return ",".join(self.metrics)

    @property
    def slug(self) -> str:
        metric_part = "+".join(self.metrics)
        raw = (
            f"metrics-{metric_part}"
            f"__fusion-{self.fusion_mode}"
            f"__cand-{self.candidate.slug()}"
            f"__agg-{self.service_agg}"
            f"__pc-{self.pc_alpha:g}"
        )
        return slugify(raw)


@dataclass
class NormalBaseline:
    metric: str
    service_agg: str
    normal_mean: pd.Series
    normal_std: pd.Series
    normal_run_count: int
    normal_row_count: int


@dataclass
class MetricRunResult:
    metric: str
    service_ranking: list[tuple[str, float]]
    node_ranking: list[tuple[str, float]]
    alarm_scores: pd.Series
    selected_candidates: list[str]
    impact_matrix: np.ndarray
    impact_nodes: list[str]
    frontend_service: str
    correlations: dict[str, float]


class HourlyMetricsCache:
    def __init__(self, metric_name: str):
        self.metric_name = metric_name
        self._cache: dict[Path, pd.DataFrame] = {}

    def load_file(self, file_path: Path) -> pd.DataFrame:
        cached = self._cache.get(file_path)
        if cached is not None:
            return cached

        chunks: list[pd.DataFrame] = []
        usecols = ["timestamp", "pod", "metric", "value"]
        for chunk in pd.read_csv(file_path, usecols=usecols, chunksize=100_000):
            chunk = chunk[chunk["metric"] == self.metric_name].copy()
            if chunk.empty:
                continue

            chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], errors="coerce")
            chunk = chunk.dropna(subset=["timestamp", "pod", "value"])
            if chunk.empty:
                continue

            chunk["service"] = chunk["pod"].astype(str).map(derive_service_from_pod)
            chunk["value"] = pd.to_numeric(chunk["value"], errors="coerce")
            chunk = chunk.dropna(subset=["value", "service"])
            if chunk.empty:
                continue

            chunks.append(chunk[["timestamp", "service", "value"]])

        if chunks:
            loaded = pd.concat(chunks, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
        else:
            loaded = pd.DataFrame(columns=["timestamp", "service", "value"])

        self._cache[file_path] = loaded
        return loaded


class MetricsRepository:
    def __init__(self, metrics_dir: Path):
        self.metrics_dir = metrics_dir
        self._hourly_caches: dict[str, HourlyMetricsCache] = {}
        self._matrix_cache: dict[tuple[str, str, str, str], pd.DataFrame] = {}

    def get_run_matrix(self, metric: str, service_agg: str, run: RunWindow) -> pd.DataFrame:
        cache_key = (metric, service_agg, run.label, run.run_id)
        cached = self._matrix_cache.get(cache_key)
        if cached is not None:
            return cached

        hourly_cache = self._hourly_caches.setdefault(metric, HourlyMetricsCache(metric))
        window_df = load_metrics_for_window(hourly_cache, self.metrics_dir, run)
        matrix_df = build_service_matrix(window_df, run.start, run.end, service_agg)
        self._matrix_cache[cache_key] = matrix_df
        return matrix_df


def slugify(value: str) -> str:
    return SLUG_RE.sub("-", value.lower()).strip("-")


def _read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def parse_utc_timestamp(value: str) -> pd.Timestamp:
    ts = pd.to_datetime(value, utc=True, errors="raise")
    return ts.tz_convert(None)


def telemetry_day_to_suffix(telemetry_day: str) -> str:
    return datetime.strptime(telemetry_day, "%Y_%m_%d").strftime("%m%d")


def get_day_paths(script_dir: Path, telemetry_day: str) -> tuple[Path, Path, Path]:
    day_suffix = telemetry_day_to_suffix(telemetry_day)
    fault_root = (script_dir / FAULT_RUN_ROOT_TEMPLATE.format(day_suffix=day_suffix)).resolve()
    normal_root = (script_dir / NORMAL_RUN_ROOT_TEMPLATE.format(day_suffix=day_suffix)).resolve()
    metrics_dir = (script_dir / TELEMETRY_METRICS_DIR_TEMPLATE.format(telemetry_day=telemetry_day)).resolve()
    return fault_root, normal_root, metrics_dir


def derive_service_from_pod(pod_name: str) -> str:
    parts = pod_name.split("-")
    if len(parts) >= 2 and parts[-1].isdigit():
        return "-".join(parts[:-1])
    if len(parts) >= 3 and DEPLOYMENT_HASH_RE.match(parts[-2]) and POD_SUFFIX_RE.match(parts[-1]):
        return "-".join(parts[:-2])
    return pod_name


def build_experiment_configs() -> list[ExperimentConfig]:
    if SEARCH_PROFILE not in {"paper", "paper_compare", "best", "fast", "focused", "balanced", "exhaustive"}:
        raise ValueError(f"Unsupported SEARCH_PROFILE: {SEARCH_PROFILE}")

    if SEARCH_PROFILE == "paper":
        config_specs = [
            (
                ("latency_p95",),
                "single_metric",
                "mean",
                CandidateConfig(
                    mode="percentile",
                    percentile=PAPER_CANDIDATE_PERCENTILE,
                    min_candidates=MIN_CANDIDATES,
                ),
            ),
        ]
    elif SEARCH_PROFILE == "paper_compare":
        config_specs = [
            (
                metrics,
                "single_metric",
                "mean",
                CandidateConfig(
                    mode="percentile",
                    percentile=percentile,
                    min_candidates=min_candidates,
                ),
            )
            for metrics in PAPER_COMPARE_METRICS
            for percentile in PAPER_COMPARE_CANDIDATE_PERCENTILES
            for min_candidates in PAPER_COMPARE_MIN_CANDIDATES
        ]
    elif SEARCH_PROFILE == "best":
        if OPTIMIZATION_TARGET == "top1":
            config_specs = [
                (("latency_p95",), "single_metric", "mean", CandidateConfig(mode="percentile", percentile=ANOMALY_PERCENTILE_THRESHOLD)),
            ]
        elif OPTIMIZATION_TARGET in {"balanced", "top5"}:
            config_specs = [
                (("latency_p50", "latency_p90", "latency_p95", "latency_p99"), "avg_score", "mean", CandidateConfig(mode="top_k", top_k=ANOMALY_TOP_K)),
            ]
        else:
            raise ValueError(f"Unsupported OPTIMIZATION_TARGET: {OPTIMIZATION_TARGET}")
    elif SEARCH_PROFILE == "fast":
        config_specs = [
            (("latency_p95",), "single_metric", "mean", CandidateConfig(mode="zscore", zscore_threshold=ANOMALY_Z_THRESHOLD)),
            (("latency_p99",), "single_metric", "mean", CandidateConfig(mode="zscore", zscore_threshold=ANOMALY_Z_THRESHOLD)),
            (("latency_p95", "latency_p99"), "avg_score", "mean", CandidateConfig(mode="top_k", top_k=ANOMALY_TOP_K)),
        ]
    elif SEARCH_PROFILE == "focused":
        config_specs = [
            (("latency_p95",), "single_metric", "mean", CandidateConfig(mode="percentile", percentile=ANOMALY_PERCENTILE_THRESHOLD)),
            (("latency_p50", "latency_p90", "latency_p95", "latency_p99"), "avg_score", "mean", CandidateConfig(mode="top_k", top_k=ANOMALY_TOP_K)),
            (("latency_p95", "latency_p99"), "avg_rank", "mean", CandidateConfig(mode="top_k", top_k=ANOMALY_TOP_K)),
            (("latency_p95", "latency_p99"), "avg_score", "mean", CandidateConfig(mode="top_k", top_k=ANOMALY_TOP_K)),
            (("latency_p99",), "single_metric", "max", CandidateConfig(mode="top_k", top_k=ANOMALY_TOP_K)),
        ]
    elif SEARCH_PROFILE == "balanced":
        config_specs = [
            (("request_rate",), "single_metric", "mean", CandidateConfig(mode="zscore", zscore_threshold=ANOMALY_Z_THRESHOLD)),
            (("latency_p50",), "single_metric", "mean", CandidateConfig(mode="zscore", zscore_threshold=ANOMALY_Z_THRESHOLD)),
            (("latency_p90",), "single_metric", "mean", CandidateConfig(mode="zscore", zscore_threshold=ANOMALY_Z_THRESHOLD)),
            (("latency_p95",), "single_metric", "mean", CandidateConfig(mode="zscore", zscore_threshold=ANOMALY_Z_THRESHOLD)),
            (("latency_p99",), "single_metric", "mean", CandidateConfig(mode="zscore", zscore_threshold=ANOMALY_Z_THRESHOLD)),
            (("latency_p95",), "single_metric", "max", CandidateConfig(mode="top_k", top_k=ANOMALY_TOP_K)),
            (("latency_p95",), "single_metric", "mean", CandidateConfig(mode="percentile", percentile=ANOMALY_PERCENTILE_THRESHOLD)),
            (("latency_p99",), "single_metric", "max", CandidateConfig(mode="top_k", top_k=ANOMALY_TOP_K)),
            (("latency_p99",), "single_metric", "mean", CandidateConfig(mode="percentile", percentile=ANOMALY_PERCENTILE_THRESHOLD)),
            (("latency_p95", "latency_p99"), "avg_rank", "mean", CandidateConfig(mode="top_k", top_k=ANOMALY_TOP_K)),
            (("latency_p95", "latency_p99"), "avg_score", "mean", CandidateConfig(mode="top_k", top_k=ANOMALY_TOP_K)),
            (("latency_p95", "latency_p99"), "avg_score", "max", CandidateConfig(mode="percentile", percentile=ANOMALY_PERCENTILE_THRESHOLD)),
            (("latency_p50", "latency_p90", "latency_p95", "latency_p99"), "avg_rank", "mean", CandidateConfig(mode="zscore", zscore_threshold=ANOMALY_Z_THRESHOLD)),
            (("latency_p50", "latency_p90", "latency_p95", "latency_p99"), "avg_score", "mean", CandidateConfig(mode="top_k", top_k=ANOMALY_TOP_K)),
            (("latency_p50", "latency_p90", "latency_p95", "latency_p99"), "avg_score", "max", CandidateConfig(mode="percentile", percentile=ANOMALY_PERCENTILE_THRESHOLD)),
        ]
    else:
        metric_groups = (
            ("request_rate",),
            ("latency_p50",),
            ("latency_p90",),
            ("latency_p95",),
            ("latency_p99",),
            ("latency_p95", "latency_p99"),
            ("latency_p50", "latency_p90", "latency_p95", "latency_p99"),
            ("request_rate", "latency_p50", "latency_p90", "latency_p95", "latency_p99"),
        )
        candidate_configs = (
            CandidateConfig(mode="zscore", zscore_threshold=ANOMALY_Z_THRESHOLD),
            CandidateConfig(mode="top_k", top_k=ANOMALY_TOP_K),
            CandidateConfig(mode="percentile", percentile=ANOMALY_PERCENTILE_THRESHOLD),
        )
        config_specs = []
        for metrics in metric_groups:
            fusion_modes = ("single_metric",) if len(metrics) == 1 else MULTI_METRIC_FUSION_MODES
            for fusion_mode in fusion_modes:
                for service_agg in ("mean", "max"):
                    for candidate_config in candidate_configs:
                        config_specs.append((metrics, fusion_mode, service_agg, candidate_config))

    configs: list[ExperimentConfig] = []
    for pc_alpha in PC_ALPHA_VALUES:
        for metrics, fusion_mode, service_agg, candidate_config in config_specs:
            unknown_metrics = [metric for metric in metrics if metric not in SUPPORTED_METRICS]
            if unknown_metrics:
                raise ValueError(f"Unsupported metrics configured: {unknown_metrics}")
            configs.append(
                ExperimentConfig(
                    metrics=tuple(metrics),
                    fusion_mode=fusion_mode,
                    service_agg=service_agg,
                    candidate=candidate_config,
                    pc_alpha=float(pc_alpha),
                )
            )
    return configs


def discover_fault_runs(script_dir: Path, telemetry_day: str) -> list[RunWindow]:
    fault_root, _, _ = get_day_paths(script_dir, telemetry_day)
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
        analysis_start = inject_start
        analysis_end = inject_start + pd.Timedelta(minutes=FAULT_WINDOW_MINUTES)

        if workload_path.exists():
            workload = _read_json(workload_path)
            workload_start_raw = workload.get("workload_start_time")
            workload_end_raw = workload.get("workload_end_time")
            if workload_start_raw and workload_end_raw:
                workload_start = parse_utc_timestamp(workload_start_raw)
                workload_end = parse_utc_timestamp(workload_end_raw)
                if FAULT_WINDOW_SOURCE == "workload_if_available":
                    analysis_start = workload_start
                    analysis_end = workload_end
                elif FAULT_WINDOW_SOURCE == "inject_window":
                    analysis_start = max(analysis_start, workload_start)
                    analysis_end = min(analysis_end, workload_end)
                else:
                    raise ValueError(f"Unsupported FAULT_WINDOW_SOURCE: {FAULT_WINDOW_SOURCE}")

        if analysis_end <= analysis_start:
            continue

        runs.append(
            RunWindow(
                run_id=exp_dir.name,
                label="fail",
                start=analysis_start,
                end=analysis_end,
                metadata_path=metadata_path.resolve(),
                inject_start=inject_start,
                ground_truth_service=str(ground_truth_service),
            )
        )

    if not runs:
        raise FileNotFoundError(f"No fault runs found for {telemetry_day}")
    return runs


def discover_normal_runs(script_dir: Path, telemetry_day: str) -> list[RunWindow]:
    _, normal_root, _ = get_day_paths(script_dir, telemetry_day)
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
        normal_end = normal_start + pd.Timedelta(minutes=NORMAL_WINDOW_DURATION_MINUTES)
        if normal_end > workload_end or normal_end <= normal_start:
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


def select_hourly_metric_files(metrics_dir: Path, windows: list[tuple[pd.Timestamp, pd.Timestamp]]) -> list[Path]:
    selected_files = []
    for hour_start in _iter_hour_starts(windows):
        file_path = metrics_dir / f"prometheus_metrics_KPI_{hour_start.strftime('%H')}.csv"
        if file_path.exists():
            selected_files.append(file_path)
    if not selected_files:
        raise FileNotFoundError(f"No hourly metric files found in {metrics_dir}")
    return selected_files


def build_window_mask(series: pd.Series, windows: list[tuple[pd.Timestamp, pd.Timestamp]]) -> pd.Series:
    mask = pd.Series(False, index=series.index)
    for start, end in windows:
        mask |= (series >= start) & (series <= end)
    return mask


def load_metrics_for_window(cache: HourlyMetricsCache, metrics_dir: Path, run: RunWindow) -> pd.DataFrame:
    windows = [(run.start, run.end)]
    selected_files = select_hourly_metric_files(metrics_dir, windows)
    selected_chunks: list[pd.DataFrame] = []
    for metric_file in selected_files:
        file_df = cache.load_file(metric_file)
        if file_df.empty:
            continue
        windowed = file_df[build_window_mask(file_df["timestamp"], windows)]
        if not windowed.empty:
            selected_chunks.append(windowed)

    if not selected_chunks:
        return pd.DataFrame(columns=["timestamp", "service", "value"])

    return pd.concat(selected_chunks, ignore_index=True).sort_values("timestamp").reset_index(drop=True)


def build_service_matrix(
    window_df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    service_agg: str,
) -> pd.DataFrame:
    if window_df.empty:
        return pd.DataFrame()

    grouped = (
        window_df.groupby(["timestamp", "service"], as_index=False)["value"]
        .agg(service_agg)
    )
    matrix_df = grouped.pivot(index="timestamp", columns="service", values="value")
    matrix_df = matrix_df.sort_index()
    resample_rule = f"{AGGREGATION_SECONDS}s"
    matrix_df = matrix_df.resample(resample_rule).agg(service_agg)
    full_index = pd.date_range(
        start=start.floor(resample_rule),
        end=end.ceil(resample_rule),
        freq=resample_rule,
    )
    matrix_df = matrix_df.reindex(full_index)
    matrix_df = matrix_df.ffill().bfill()
    matrix_df = matrix_df.dropna(axis=1, how="all").fillna(0.0)
    return matrix_df


def concat_run_matrices(matrices: list[pd.DataFrame]) -> pd.DataFrame:
    non_empty = [matrix for matrix in matrices if not matrix.empty]
    if not non_empty:
        return pd.DataFrame()
    combined = pd.concat(non_empty, axis=0, sort=False)
    combined = combined.sort_index().ffill().bfill()
    combined = combined.dropna(axis=1, how="all").fillna(0.0)
    return combined


def compute_normal_statistics(normal_matrix_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    normal_mean = normal_matrix_df.mean(axis=0)
    normal_std = normal_matrix_df.std(axis=0)
    sigma_floor = float(np.percentile(normal_std[normal_std > 0], 20)) if (normal_std > 0).any() else 1e-6
    normal_std = normal_std.clip(lower=sigma_floor if sigma_floor > 0 else 1e-6)
    return normal_mean, normal_std


def align_to_reference_columns(matrix_df: pd.DataFrame, reference_columns: list[str]) -> pd.DataFrame:
    aligned = matrix_df.reindex(columns=reference_columns)
    aligned = aligned.ffill().bfill().fillna(0.0)
    return aligned


def compute_alarm_scores(
    fault_matrix_df: pd.DataFrame,
    normal_mean: pd.Series,
    normal_std: pd.Series,
) -> pd.Series:
    reference_columns = sorted(set(fault_matrix_df.columns) & set(normal_mean.index))
    if not reference_columns:
        raise ValueError("Fault matrix and normal baseline do not share any services.")

    aligned_fault = align_to_reference_columns(fault_matrix_df, reference_columns)
    mu = normal_mean.reindex(reference_columns)
    sigma = normal_std.reindex(reference_columns)
    z_fault = ((aligned_fault - mu) / sigma).abs()
    scores = z_fault.quantile(ANOMALY_SCORE_PERCENTILE / 100.0, axis=0)
    return scores.sort_values(ascending=False)


def select_candidate_services(
    alarm_scores: pd.Series,
    fault_matrix_df: pd.DataFrame,
    candidate_config: CandidateConfig,
) -> list[str]:
    if candidate_config.mode == "top_k":
        if not candidate_config.top_k or candidate_config.top_k <= 0:
            raise ValueError("top_k candidate selection requires top_k > 0")
        candidates = alarm_scores.head(candidate_config.top_k).index.tolist()
    elif candidate_config.mode == "percentile":
        if candidate_config.percentile is None:
            raise ValueError("percentile candidate selection requires percentile")
        threshold = float(alarm_scores.quantile(candidate_config.percentile / 100.0))
        candidates = alarm_scores[alarm_scores >= threshold].index.tolist()
    elif candidate_config.mode == "zscore":
        if candidate_config.zscore_threshold is None:
            raise ValueError("zscore candidate selection requires zscore_threshold")
        candidates = alarm_scores[alarm_scores >= candidate_config.zscore_threshold].index.tolist()
    else:
        raise ValueError(f"Unsupported candidate mode: {candidate_config.mode}")

    if candidate_config.min_candidates is not None and len(candidates) < candidate_config.min_candidates:
        extra = [service for service in alarm_scores.index.tolist() if service not in candidates]
        candidates.extend(extra[: max(0, candidate_config.min_candidates - len(candidates))])

    if FRONTEND_SERVICE in fault_matrix_df.columns and FRONTEND_SERVICE not in candidates:
        candidates.append(FRONTEND_SERVICE)

    if not candidates and not alarm_scores.empty:
        candidates = alarm_scores.index.tolist()

    return [service for service in candidates if service in fault_matrix_df.columns]


def choose_frontend_service(candidate_df: pd.DataFrame, alarm_scores: pd.Series) -> str:
    _ = alarm_scores
    if FRONTEND_SERVICE not in candidate_df.columns:
        raise ValueError(f"Fixed frontend service not present in candidate set: {FRONTEND_SERVICE}")
    return FRONTEND_SERVICE


def _drop_duplicate_service_series(
    matrix_df: pd.DataFrame,
    required_columns: list[str],
) -> pd.DataFrame:
    required_set = set(required_columns)
    kept_columns: list[str] = []
    kept_arrays: list[np.ndarray] = []

    for column in matrix_df.columns:
        values = matrix_df[column].to_numpy(dtype=float)
        duplicate_index = next(
            (idx for idx, kept_values in enumerate(kept_arrays) if np.array_equal(values, kept_values)),
            None,
        )
        if duplicate_index is None:
            kept_columns.append(column)
            kept_arrays.append(values)
            continue

        kept_column = kept_columns[duplicate_index]
        if kept_column in required_set:
            continue
        if column in required_set:
            kept_columns[duplicate_index] = column
            kept_arrays[duplicate_index] = values

    return matrix_df[kept_columns].copy()


def _drop_redundant_pc_columns(
    matrix_df: pd.DataFrame,
    required_columns: list[str],
) -> pd.DataFrame:
    reduced_df = _drop_duplicate_service_series(matrix_df, required_columns)
    required_set = set(required_columns)

    while len(reduced_df.columns) > 1:
        corr_df = reduced_df.corr().fillna(0.0)
        corr = corr_df.to_numpy(dtype=float)
        if np.linalg.matrix_rank(corr) == corr.shape[0]:
            break

        removable_columns = [column for column in reduced_df.columns if column not in required_set]
        if not removable_columns:
            break

        variances = reduced_df.var(axis=0)
        abs_corr = corr_df.abs()
        np.fill_diagonal(abs_corr.values, 0.0)
        drop_column = max(
            removable_columns,
            key=lambda column: (
                float(abs_corr[column].max()),
                -float(variances.get(column, 0.0)),
                column,
            ),
        )
        reduced_df = reduced_df.drop(columns=[drop_column])

    return reduced_df


def learn_pc_graph(
    matrix_df: pd.DataFrame,
    alpha: float,
    variance_eps: float,
    max_nodes: int | None,
    required_columns: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    X = matrix_df.to_numpy(dtype=float)
    columns = matrix_df.columns.tolist()
    required_columns = required_columns or []

    if not np.isfinite(X).all():
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    variances = np.var(X, axis=0)
    keep_mask = variances > variance_eps
    if required_columns:
        keep_mask = np.array([keep or (column in required_columns) for column, keep in zip(columns, keep_mask)], dtype=bool)
    if int((~keep_mask).sum()) > 0:
        X = X[:, keep_mask]
        columns = [column for column, keep in zip(columns, keep_mask) if keep]

    if max_nodes is not None and X.shape[1] > max_nodes:
        var_order = np.argsort(np.var(X, axis=0))[::-1]
        keep_idx = np.sort(var_order[:max_nodes])
        required_idx = [idx for idx, column in enumerate(columns) if column in required_columns]
        if required_idx:
            keep_idx = np.unique(np.concatenate([keep_idx, np.asarray(required_idx, dtype=int)]))
        X = X[:, keep_idx]
        columns = [columns[i] for i in keep_idx]

    if X.shape[1] == 0:
        raise ValueError("PC graph input has no remaining columns after filtering.")

    matrix_df = pd.DataFrame(X, columns=columns, index=matrix_df.index)
    matrix_df = _drop_redundant_pc_columns(matrix_df, required_columns)
    X = matrix_df.to_numpy(dtype=float)
    columns = matrix_df.columns.tolist()

    if X.shape[1] == 0:
        raise ValueError("PC graph input has no remaining columns after redundancy pruning.")

    cg = pc(
        X,
        alpha=alpha,
        indep_test=fisherz,
        stable=True,
        verbose=False,
        show_progress=False,
    )
    impact_adj = convert_pc_graph_to_impact_adjacency(cg.G.graph)
    return impact_adj, columns


def convert_pc_graph_to_impact_adjacency(raw_graph: np.ndarray) -> np.ndarray:
    n = raw_graph.shape[0]
    impact = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            left = raw_graph[i, j]
            right = raw_graph[j, i]
            if left == 0 and right == 0:
                continue

            if left == -1 and right == 1:
                impact[j, i] = 1
            elif left == 1 and right == -1:
                impact[i, j] = 1
            else:
                impact[i, j] = 1
                impact[j, i] = 1
    return impact


def prune_isolated_nodes(
    adjacency_matrix: np.ndarray,
    columns: list[str],
    matrix_df: pd.DataFrame,
    required_nodes: list[str] | None = None,
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    if adjacency_matrix.size == 0:
        raise ValueError("Impact graph is empty.")
    keep_mask = (adjacency_matrix.sum(axis=0) + adjacency_matrix.sum(axis=1)) > 0
    required_nodes = required_nodes or []
    if required_nodes:
        keep_mask = np.array([keep or (column in required_nodes) for column, keep in zip(columns, keep_mask)], dtype=bool)
    if not keep_mask.any():
        raise ValueError("All candidate services are isolated in the impact graph.")
    keep_idx = np.where(keep_mask)[0]
    pruned_adj = adjacency_matrix[np.ix_(keep_idx, keep_idx)]
    pruned_columns = [columns[i] for i in keep_idx]
    pruned_df = matrix_df[pruned_columns].copy()
    return pruned_adj, pruned_columns, pruned_df


def compute_frontend_correlations(matrix_df: pd.DataFrame, frontend_service: str) -> dict[str, float]:
    frontend = matrix_df[frontend_service]
    frontend_std = float(frontend.std())
    correlations: dict[str, float] = {}
    for service in matrix_df.columns:
        series = matrix_df[service]
        if frontend_std <= 0 or float(series.std()) <= 0:
            correlations[service] = 1.0 if service == frontend_service else 0.0
            continue
        aligned = pd.concat([frontend.rename("frontend"), series.rename("service")], axis=1).dropna()
        if aligned.empty or len(aligned) < 2:
            correlations[service] = 1.0 if service == frontend_service else 0.0
            continue
        corr = aligned["service"].corr(aligned["frontend"])
        correlations[service] = float(abs(corr)) if pd.notna(corr) else 0.0
    correlations[frontend_service] = max(correlations.get(frontend_service, 0.0), 1.0)
    return correlations


def _normalize_weights(weights: dict[int, float]) -> dict[int, float]:
    positive = {node: float(weight) for node, weight in weights.items() if weight > 0}
    total = sum(positive.values())
    if total <= 0:
        return {}
    return {node: weight / total for node, weight in positive.items()}


def _first_order_probabilities(
    adjacency_matrix: np.ndarray,
    correlations: dict[str, float],
    nodes: list[str],
) -> dict[int, dict[int, float]]:
    base: dict[int, dict[int, float]] = {}
    for i in range(len(nodes)):
        out_neighbors = np.where(adjacency_matrix[i] > 0)[0].tolist()
        weights = {j: correlations.get(nodes[j], 0.0) for j in out_neighbors}
        base[i] = _normalize_weights(weights)
    return base


def second_order_random_walk(
    adjacency_matrix: np.ndarray,
    nodes: list[str],
    correlations: dict[str, float],
    frontend_service: str,
    beta: float,
    rho: float,
    steps: int,
    seed: int,
) -> tuple[list[tuple[str, float]], dict[str, float]]:
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    if frontend_service not in node_to_idx:
        raise ValueError(f"Frontend service not in impact graph: {frontend_service}")

    base_probs = _first_order_probabilities(adjacency_matrix, correlations, nodes)
    in_neighbors = {i: np.where(adjacency_matrix[:, i] > 0)[0].tolist() for i in range(len(nodes))}
    out_neighbors = {i: np.where(adjacency_matrix[i] > 0)[0].tolist() for i in range(len(nodes))}

    rng = np.random.default_rng(seed)
    prev = node_to_idx[frontend_service]
    curr = node_to_idx[frontend_service]
    counts = np.zeros(len(nodes), dtype=float)
    counts[curr] += 1.0

    for _ in range(max(steps - 1, 0)):
        prev_to_curr = base_probs.get(prev, {}).get(curr, correlations.get(nodes[curr], 0.0))

        forward_raw = {
            nxt: (1.0 - beta) * prev_to_curr + beta * base_probs.get(curr, {}).get(nxt, 0.0)
            for nxt in out_neighbors[curr]
        }
        backward_raw = {
            nxt: rho * ((1.0 - beta) * prev_to_curr + beta * correlations.get(nodes[nxt], 0.0))
            for nxt in in_neighbors[curr]
        }

        self_base = (1.0 - beta) * prev_to_curr + beta * correlations.get(nodes[curr], 0.0)
        neighbor_max = 0.0
        if forward_raw:
            neighbor_max = max(neighbor_max, max(forward_raw.values()))
        if backward_raw:
            neighbor_max = max(neighbor_max, max(backward_raw.values()))
        self_weight = max(0.0, self_base - neighbor_max)

        transition = {}
        transition.update(forward_raw)
        for nxt, weight in backward_raw.items():
            transition[nxt] = transition.get(nxt, 0.0) + weight
        transition[curr] = transition.get(curr, 0.0) + self_weight
        normalized = _normalize_weights(transition)

        if not normalized:
            counts[curr] += 1.0
            continue

        next_nodes = np.array(list(normalized.keys()), dtype=int)
        probs = np.array([normalized[idx] for idx in next_nodes], dtype=float)
        nxt = int(rng.choice(next_nodes, p=probs))
        prev, curr = curr, nxt
        counts[curr] += 1.0

    if counts.sum() <= 0:
        raise ValueError("Random walk produced no node visits.")

    scores = counts / counts.sum()
    ranking = sorted(
        [(nodes[i], float(scores[i])) for i in range(len(nodes))],
        key=lambda item: -item[1],
    )
    return ranking, {nodes[i]: float(scores[i]) for i in range(len(nodes))}


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


def get_normal_baseline(
    metric: str,
    service_agg: str,
    normal_runs: list[RunWindow],
    repository: MetricsRepository,
    baseline_cache: dict[tuple[str, str], NormalBaseline],
) -> NormalBaseline:
    cache_key = (metric, service_agg)
    cached = baseline_cache.get(cache_key)
    if cached is not None:
        return cached

    normal_matrices = []
    for normal_run in normal_runs:
        matrix_df = repository.get_run_matrix(metric, service_agg, normal_run)
        if not matrix_df.empty:
            normal_matrices.append(matrix_df)

    combined_normal_df = concat_run_matrices(normal_matrices)
    if combined_normal_df.empty:
        raise ValueError(f"Combined normal metrics matrix is empty for metric={metric} agg={service_agg}")

    normal_mean, normal_std = compute_normal_statistics(combined_normal_df)
    baseline = NormalBaseline(
        metric=metric,
        service_agg=service_agg,
        normal_mean=normal_mean,
        normal_std=normal_std,
        normal_run_count=len(normal_matrices),
        normal_row_count=len(combined_normal_df),
    )
    baseline_cache[cache_key] = baseline
    return baseline


def analyze_metric_run(
    metric: str,
    fault_run: RunWindow,
    config: ExperimentConfig,
    baseline: NormalBaseline,
    repository: MetricsRepository,
    run_index: int,
) -> MetricRunResult:
    fault_matrix_df = repository.get_run_matrix(metric, config.service_agg, fault_run)
    if fault_matrix_df.empty:
        raise ValueError(f"Fault metrics matrix is empty for metric={metric}.")

    alarm_scores = compute_alarm_scores(fault_matrix_df, baseline.normal_mean, baseline.normal_std)
    selected_candidates = select_candidate_services(alarm_scores, fault_matrix_df, config.candidate)
    if not selected_candidates:
        raise ValueError(f"No candidate services selected for metric={metric}.")

    candidate_df = fault_matrix_df[selected_candidates].copy()
    candidate_df = candidate_df.ffill().bfill().fillna(0.0)

    frontend_service = choose_frontend_service(candidate_df, alarm_scores)
    impact_adj, impact_nodes = learn_pc_graph(
        matrix_df=candidate_df,
        alpha=config.pc_alpha,
        variance_eps=VARIANCE_EPS,
        max_nodes=MAX_NODES,
        required_columns=[frontend_service],
    )
    impact_adj, impact_nodes, candidate_df = prune_isolated_nodes(
        impact_adj,
        impact_nodes,
        candidate_df,
        required_nodes=[frontend_service],
    )
    if frontend_service not in impact_nodes:
        raise ValueError(f"Fixed frontend service was removed from the impact graph: {frontend_service}")

    correlations = compute_frontend_correlations(candidate_df, frontend_service)
    node_ranking, _ = second_order_random_walk(
        adjacency_matrix=impact_adj,
        nodes=impact_nodes,
        correlations=correlations,
        frontend_service=frontend_service,
        beta=BETA,
        rho=RHO,
        steps=WALK_STEPS,
        seed=RANDOM_SEED + run_index,
    )
    return MetricRunResult(
        metric=metric,
        service_ranking=node_ranking,
        node_ranking=node_ranking,
        alarm_scores=alarm_scores,
        selected_candidates=selected_candidates,
        impact_matrix=impact_adj,
        impact_nodes=impact_nodes,
        frontend_service=frontend_service,
        correlations=correlations,
    )


def fuse_service_rankings(metric_results: list[MetricRunResult], fusion_mode: str) -> list[tuple[str, float]]:
    if not metric_results:
        return []
    if fusion_mode == "single_metric":
        return metric_results[0].service_ranking

    all_services = sorted(
        {
            service
            for result in metric_results
            for service, _ in result.service_ranking
        }
    )
    if not all_services:
        return []

    if fusion_mode == "avg_rank":
        avg_ranks = {}
        for service in all_services:
            ranks = []
            for result in metric_results:
                rank_map = {name: rank for rank, (name, _) in enumerate(result.service_ranking, start=1)}
                ranks.append(rank_map.get(service, len(result.service_ranking) + 1))
            avg_rank = float(np.mean(ranks))
            avg_ranks[service] = avg_rank
        return sorted(
            [(service, 1.0 / avg_rank) for service, avg_rank in avg_ranks.items()],
            key=lambda item: (-item[1], item[0]),
        )

    if fusion_mode == "avg_score":
        avg_scores = {}
        for service in all_services:
            scores = []
            for result in metric_results:
                score_map = dict(result.service_ranking)
                max_score = max(score_map.values()) if score_map else 0.0
                if max_score <= 0:
                    scores.append(0.0)
                else:
                    scores.append(score_map.get(service, 0.0) / max_score)
            avg_scores[service] = float(np.mean(scores))
        return sorted(avg_scores.items(), key=lambda item: (-item[1], item[0]))

    raise ValueError(f"Unsupported fusion mode: {fusion_mode}")


def save_baseline_outputs(
    output_dir: Path,
    baselines: dict[str, NormalBaseline],
    config: ExperimentConfig,
    telemetry_day: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    mean_rows = []
    std_rows = []
    for metric, baseline in baselines.items():
        for service, value in baseline.normal_mean.items():
            mean_rows.append({"metric": metric, "service": service, "mean": float(value)})
        for service, value in baseline.normal_std.items():
            std_rows.append({"metric": metric, "service": service, "std": float(value)})

    pd.DataFrame(mean_rows).to_csv(output_dir / NORMAL_MEAN_FILE, index=False)
    pd.DataFrame(std_rows).to_csv(output_dir / NORMAL_STD_FILE, index=False)

    summary = {
        "telemetry_day": telemetry_day,
        "metrics": list(config.metrics),
        "fusion_mode": config.fusion_mode,
        "service_agg": config.service_agg,
        "candidate_mode": config.candidate.mode,
        "candidate_top_k": config.candidate.top_k,
        "candidate_percentile": config.candidate.percentile,
        "candidate_zscore_threshold": config.candidate.zscore_threshold,
        "candidate_min": config.candidate.min_candidates,
        "aggregation_seconds": AGGREGATION_SECONDS,
        "fault_window_minutes": FAULT_WINDOW_MINUTES,
        "normal_window_offset_minutes": NORMAL_WINDOW_OFFSET_MINUTES,
        "normal_window_duration_minutes": NORMAL_WINDOW_DURATION_MINUTES,
        "fault_window_source": FAULT_WINDOW_SOURCE,
        "pc_alpha": config.pc_alpha,
        "beta": BETA,
        "rho": RHO,
        "walk_steps": WALK_STEPS,
        "normal_run_count": {metric: baseline.normal_run_count for metric, baseline in baselines.items()},
        "normal_row_count": {metric: baseline.normal_row_count for metric, baseline in baselines.items()},
    }
    (output_dir / NORMAL_SCORE_FILE).write_text(json.dumps(summary, indent=2), encoding="utf-8")


def build_variable_ranking_rows(metric_results: list[MetricRunResult]) -> pd.DataFrame:
    rows = []
    for result in metric_results:
        for rank, (service, score) in enumerate(result.node_ranking, start=1):
            rows.append(
                {
                    "metric": result.metric,
                    "service": service,
                    "score": score,
                    "rank": rank,
                    "frontend_service": result.frontend_service,
                }
            )
    return pd.DataFrame(rows)


def build_candidate_rows(metric_results: list[MetricRunResult]) -> pd.DataFrame:
    rows = []
    for result in metric_results:
        selected = set(result.selected_candidates)
        kept = set(result.impact_nodes)
        for service, score in result.alarm_scores.items():
            rows.append(
                {
                    "metric": result.metric,
                    "service": service,
                    "alarm_score": float(score),
                    "selected_candidate": service in selected,
                    "kept_in_impact_graph": service in kept,
                    "frontend_service": result.frontend_service == service,
                }
            )
    return pd.DataFrame(rows)


def build_impact_graph_output(metric_results: list[MetricRunResult]) -> pd.DataFrame:
    if len(metric_results) == 1:
        result = metric_results[0]
        return pd.DataFrame(result.impact_matrix, index=result.impact_nodes, columns=result.impact_nodes)

    rows = []
    for result in metric_results:
        for src_index, source in enumerate(result.impact_nodes):
            for dst_index, target in enumerate(result.impact_nodes):
                if int(result.impact_matrix[src_index, dst_index]) > 0:
                    rows.append({"metric": result.metric, "source": source, "target": target})
    return pd.DataFrame(rows, columns=["metric", "source", "target"])


def save_run_outputs(
    output_dir: Path,
    final_service_ranking: list[tuple[str, float]],
    metric_results: list[MetricRunResult],
    run_summary: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    service_df = pd.DataFrame(final_service_ranking, columns=["service", "score"])
    if not service_df.empty:
        service_df.insert(0, "rank", np.arange(1, len(service_df) + 1))
    service_df.to_csv(output_dir / SERVICE_RANKING_FILE, index=False)

    build_variable_ranking_rows(metric_results).to_csv(output_dir / VARIABLE_RANKING_FILE, index=False)
    build_candidate_rows(metric_results).to_csv(output_dir / CANDIDATE_FILE, index=False)
    build_impact_graph_output(metric_results).to_csv(output_dir / IMPACT_GRAPH_FILE, index=True)

    (output_dir / RUN_SUMMARY_FILE).write_text(
        json.dumps(run_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def build_day_outputs(
    output_dir: Path,
    detail_rows: list[dict],
    total_runtime_seconds: float,
    config: ExperimentConfig,
    telemetry_day: str,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    details_df = pd.DataFrame(detail_rows).sort_values("exp_id").reset_index(drop=True)
    details_df.to_csv(output_dir / DAY_DETAILS_FILE, index=False)

    n_total = len(details_df)
    summary_df = pd.DataFrame(
        [
            {
                "telemetry_day": telemetry_day,
                "config_slug": config.slug,
                "metrics": config.metrics_label,
                "fusion_mode": config.fusion_mode,
                "service_agg": config.service_agg,
                "candidate_mode": config.candidate.mode,
                "candidate_top_k": config.candidate.top_k,
                "candidate_percentile": config.candidate.percentile,
                "candidate_zscore_threshold": config.candidate.zscore_threshold,
                "candidate_min": config.candidate.min_candidates,
                "aggregation_seconds": AGGREGATION_SECONDS,
                "fault_window_minutes": FAULT_WINDOW_MINUTES,
                "normal_window_offset_minutes": NORMAL_WINDOW_OFFSET_MINUTES,
                "normal_window_duration_minutes": NORMAL_WINDOW_DURATION_MINUTES,
                "fault_window_source": FAULT_WINDOW_SOURCE,
                "search_profile": SEARCH_PROFILE,
                "optimization_target": OPTIMIZATION_TARGET,
                "pc_alpha": config.pc_alpha,
                "n_total": n_total,
                "n_ok": int(details_df["predicted_service_top1"].notna().sum()) if n_total else 0,
                "n_error": int(details_df["predicted_service_top1"].isna().sum()) if n_total else 0,
                "service_top1_accuracy": float(details_df["service_top1_hit"].fillna(False).mean()) if n_total else np.nan,
                "service_top3_accuracy": float(details_df["service_top3_hit"].fillna(False).mean()) if n_total else np.nan,
                "service_top5_accuracy": float(details_df["service_top5_hit"].fillna(False).mean()) if n_total else np.nan,
                "beta": BETA,
                "rho": RHO,
                "walk_steps": WALK_STEPS,
                "total_runtime_seconds": total_runtime_seconds,
                "avg_runtime_per_exception_seconds": total_runtime_seconds / n_total if n_total else np.nan,
                "sum_of_individual_runtime_seconds": float(details_df["runtime_seconds"].fillna(0).sum()) if n_total else np.nan,
                "avg_runtime_per_processed_exception_seconds": float(details_df["runtime_seconds"].mean()) if n_total else np.nan,
            }
        ]
    )
    summary_df.to_csv(output_dir / DAY_SUMMARY_FILE, index=False)
    return summary_df


def compute_selection_score(summary_df: pd.DataFrame) -> pd.Series:
    if OPTIMIZATION_TARGET == "top1":
        return (
            summary_df["service_top1_accuracy"].fillna(0.0) * 1000.0
            + summary_df["service_top3_accuracy"].fillna(0.0) * 100.0
            + summary_df["service_top5_accuracy"].fillna(0.0) * 10.0
            - summary_df["total_runtime_seconds"].fillna(0.0) / 1000.0
        )
    if OPTIMIZATION_TARGET == "balanced":
        return (
            summary_df["service_top1_accuracy"].fillna(0.0) * 500.0
            + summary_df["service_top3_accuracy"].fillna(0.0) * 300.0
            + summary_df["service_top5_accuracy"].fillna(0.0) * 200.0
            - summary_df["total_runtime_seconds"].fillna(0.0) / 1000.0
        )
    if OPTIMIZATION_TARGET == "top5":
        return (
            summary_df["service_top5_accuracy"].fillna(0.0) * 1000.0
            + summary_df["service_top3_accuracy"].fillna(0.0) * 100.0
            + summary_df["service_top1_accuracy"].fillna(0.0) * 10.0
            - summary_df["total_runtime_seconds"].fillna(0.0) / 1000.0
        )
    raise ValueError(f"Unsupported OPTIMIZATION_TARGET: {OPTIMIZATION_TARGET}")


def summarize_metric_results(metric_results: list[MetricRunResult]) -> list[dict]:
    rows = []
    for result in metric_results:
        rows.append(
            {
                "metric": result.metric,
                "frontend_service": result.frontend_service,
                "selected_candidates": result.selected_candidates,
                "impact_nodes": result.impact_nodes,
                "top_service": result.service_ranking[0][0] if result.service_ranking else None,
            }
        )
    return rows


def get_config_output_dir(day_output_dir: Path, config: ExperimentConfig, config_count: int) -> Path:
    if config_count == 1:
        return day_output_dir
    return day_output_dir / config.slug


def build_all_days_outputs(
    output_root: Path,
    all_details_dfs: list[pd.DataFrame],
    all_summary_dfs: list[pd.DataFrame],
    total_script_runtime_seconds: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    combined_details_df = pd.concat(all_details_dfs, ignore_index=True) if all_details_dfs else pd.DataFrame()
    combined_summary_df = pd.concat(all_summary_dfs, ignore_index=True) if all_summary_dfs else pd.DataFrame()
    overall_summary_df = pd.DataFrame()

    if combined_details_df.empty:
        overall_summary_df = pd.DataFrame(
            columns=[
                "telemetry_day",
                "config_slug",
                "metrics",
                "fusion_mode",
                "service_agg",
                "candidate_mode",
                "candidate_top_k",
                "candidate_percentile",
                "candidate_zscore_threshold",
                "candidate_min",
                "aggregation_seconds",
                "fault_window_minutes",
                "normal_window_offset_minutes",
                "normal_window_duration_minutes",
                "fault_window_source",
                "search_profile",
                "optimization_target",
                "pc_alpha",
                "n_total",
                "n_ok",
                "n_error",
                "service_top1_accuracy",
                "service_top3_accuracy",
                "service_top5_accuracy",
                "beta",
                "rho",
                "walk_steps",
                "total_runtime_seconds",
                "avg_runtime_per_exception_seconds",
                "sum_of_individual_runtime_seconds",
                "avg_runtime_per_processed_exception_seconds",
            ]
        )
        all_days_ablation_df = pd.DataFrame(
            columns=[
                "config_slug",
                "metrics",
                "fusion_mode",
                "service_agg",
                "candidate_mode",
                "candidate_top_k",
                "candidate_percentile",
                "candidate_zscore_threshold",
                "candidate_min",
                "search_profile",
                "optimization_target",
                "pc_alpha",
                "n_total",
                "n_ok",
                "n_error",
                "service_top1_accuracy",
                "service_top3_accuracy",
                "service_top5_accuracy",
                "beta",
                "rho",
                "walk_steps",
                "total_runtime_seconds",
                "avg_runtime_per_exception_seconds",
                "sum_of_individual_runtime_seconds",
                "avg_runtime_per_processed_exception_seconds",
                "selection_score",
                "ablation_total_runtime_seconds",
            ]
        )
    else:
        config_columns = [
            "config_slug",
            "metrics",
            "fusion_mode",
            "service_agg",
            "candidate_mode",
            "candidate_top_k",
            "candidate_percentile",
            "candidate_zscore_threshold",
            "candidate_min",
            "search_profile",
            "optimization_target",
            "pc_alpha",
        ]
        summary_constant_columns = [
            "aggregation_seconds",
            "fault_window_minutes",
            "normal_window_offset_minutes",
            "normal_window_duration_minutes",
            "fault_window_source",
            "beta",
            "rho",
            "walk_steps",
        ]
        for column, default_value in (
            ("search_profile", SEARCH_PROFILE),
            ("optimization_target", OPTIMIZATION_TARGET),
            ("pc_alpha", np.nan),
            ("candidate_top_k", np.nan),
            ("candidate_percentile", np.nan),
            ("candidate_zscore_threshold", np.nan),
            ("candidate_min", np.nan),
            ("aggregation_seconds", AGGREGATION_SECONDS),
            ("fault_window_minutes", FAULT_WINDOW_MINUTES),
            ("normal_window_offset_minutes", NORMAL_WINDOW_OFFSET_MINUTES),
            ("normal_window_duration_minutes", NORMAL_WINDOW_DURATION_MINUTES),
            ("fault_window_source", FAULT_WINDOW_SOURCE),
            ("beta", BETA),
            ("rho", RHO),
            ("walk_steps", WALK_STEPS),
        ):
            if column not in combined_details_df.columns:
                combined_details_df[column] = default_value
            if column not in combined_summary_df.columns:
                combined_summary_df[column] = default_value
        detail_group = combined_details_df.groupby(config_columns, dropna=False)
        all_days_ablation_df = detail_group.agg(
            n_total=("exp_id", "size"),
            n_ok=("predicted_service_top1", lambda s: int(s.notna().sum())),
            n_error=("predicted_service_top1", lambda s: int(s.isna().sum())),
            service_top1_accuracy=("service_top1_hit", lambda s: float(pd.Series(s).fillna(False).mean())),
            service_top3_accuracy=("service_top3_hit", lambda s: float(pd.Series(s).fillna(False).mean())),
            service_top5_accuracy=("service_top5_hit", lambda s: float(pd.Series(s).fillna(False).mean())),
            sum_of_individual_runtime_seconds=("runtime_seconds", lambda s: float(pd.Series(s).fillna(0).sum())),
            avg_runtime_per_processed_exception_seconds=("runtime_seconds", "mean"),
        ).reset_index()

        summary_group = combined_summary_df.groupby(config_columns, dropna=False)
        runtime_df = summary_group.agg(
            total_runtime_seconds=("total_runtime_seconds", "sum"),
        ).reset_index()
        summary_constants_df = summary_group.agg(
            aggregation_seconds=("aggregation_seconds", "first"),
            fault_window_minutes=("fault_window_minutes", "first"),
            normal_window_offset_minutes=("normal_window_offset_minutes", "first"),
            normal_window_duration_minutes=("normal_window_duration_minutes", "first"),
            fault_window_source=("fault_window_source", "first"),
            beta=("beta", "first"),
            rho=("rho", "first"),
            walk_steps=("walk_steps", "first"),
        ).reset_index()

        overall_summary_df = (
            all_days_ablation_df
            .merge(runtime_df, on=config_columns, how="left")
            .merge(summary_constants_df, on=config_columns, how="left")
        )
        overall_summary_df.insert(0, "telemetry_day", "ALL_DAYS")
        overall_summary_df["avg_runtime_per_exception_seconds"] = np.where(
            overall_summary_df["n_total"] > 0,
            overall_summary_df["total_runtime_seconds"] / overall_summary_df["n_total"],
            np.nan,
        )
        overall_summary_df = overall_summary_df[
            [
                "telemetry_day",
                *config_columns,
                *summary_constant_columns,
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
            ]
        ].sort_values(
            ["service_top1_accuracy", "service_top3_accuracy", "service_top5_accuracy", "config_slug"],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)

        all_days_ablation_df = all_days_ablation_df.merge(runtime_df, on=config_columns, how="left")
        all_days_ablation_df["beta"] = BETA
        all_days_ablation_df["rho"] = RHO
        all_days_ablation_df["walk_steps"] = WALK_STEPS
        all_days_ablation_df["avg_runtime_per_exception_seconds"] = np.where(
            all_days_ablation_df["n_total"] > 0,
            all_days_ablation_df["total_runtime_seconds"] / all_days_ablation_df["n_total"],
            np.nan,
        )
        all_days_ablation_df["selection_score"] = compute_selection_score(all_days_ablation_df)
        all_days_ablation_df = all_days_ablation_df.sort_values(
            ["selection_score", "service_top1_accuracy", "service_top3_accuracy", "service_top5_accuracy", "config_slug"],
            ascending=[False, False, False, False, True],
        ).reset_index(drop=True)
        all_days_ablation_df["ablation_total_runtime_seconds"] = total_script_runtime_seconds

    output_root.mkdir(parents=True, exist_ok=True)
    combined_details_df.to_csv(output_root / ALL_DAYS_DETAILS_FILE, index=False)
    combined_summary_df.to_csv(output_root / ALL_DAYS_SUMMARY_FILE, index=False)
    overall_summary_df.to_csv(output_root / OVERALL_SUMMARY_FILE, index=False)
    all_days_ablation_df.to_csv(output_root / ALL_DAYS_ABLATION_FILE, index=False)

    if not all_days_ablation_df.empty:
        best_row = all_days_ablation_df.iloc[0].to_dict()
        (output_root / ALL_DAYS_BEST_CONFIG_FILE).write_text(
            json.dumps(best_row, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return combined_details_df, combined_summary_df, overall_summary_df, all_days_ablation_df


def run_single_day(script_dir: Path, telemetry_day: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total_start = perf_counter()
    _, _, metrics_dir = get_day_paths(script_dir, telemetry_day)

    fault_runs = discover_fault_runs(script_dir, telemetry_day)
    if EXP_ID:
        fault_runs = [run for run in fault_runs if run.run_id == EXP_ID]
        if not fault_runs:
            raise ValueError(f"EXP_ID not found for {telemetry_day}: {EXP_ID}")
    normal_runs = discover_normal_runs(script_dir, telemetry_day)
    experiment_configs = build_experiment_configs()

    print(
        f"[INFO] telemetry_day={telemetry_day} "
        f"fault_runs={len(fault_runs)} normal_runs={len(normal_runs)} "
        f"configs={len(experiment_configs)}"
    )

    repository = MetricsRepository(metrics_dir)
    baseline_cache: dict[tuple[str, str], NormalBaseline] = {}
    day_output_dir = (script_dir / OUTPUT_ROOT / telemetry_day).resolve()
    ablation_rows = []

    for config_index, config in enumerate(experiment_configs, start=1):
        config_start = perf_counter()
        config_output_dir = get_config_output_dir(day_output_dir, config, len(experiment_configs))
        print(
            f"[INFO] [config {config_index}/{len(experiment_configs)}] "
            f"metrics={config.metrics_label} fusion={config.fusion_mode} "
            f"candidate={config.candidate.slug()} agg={config.service_agg} pc_alpha={config.pc_alpha:g}"
        )

        baselines = {
            metric: get_normal_baseline(metric, config.service_agg, normal_runs, repository, baseline_cache)
            for metric in config.metrics
        }
        save_baseline_outputs(config_output_dir / NORMAL_BASELINE_DIR, baselines, config, telemetry_day)

        detail_rows = []
        for run_index, fault_run in enumerate(fault_runs, start=1):
            run_start = perf_counter()
            print(
                f"[INFO] [config {config_index}/{len(experiment_configs)}] "
                f"[{run_index}/{len(fault_runs)}] Running {fault_run.run_id}"
            )
            try:
                metric_results = [
                    analyze_metric_run(
                        metric=metric,
                        fault_run=fault_run,
                        config=config,
                        baseline=baselines[metric],
                        repository=repository,
                        run_index=run_index,
                    )
                    for metric in config.metrics
                ]
                final_service_ranking = fuse_service_rankings(metric_results, config.fusion_mode)
                runtime_seconds = perf_counter() - run_start
                eval_result = evaluate_topk(final_service_ranking, fault_run.ground_truth_service or "")

                run_summary = {
                    "telemetry_day": telemetry_day,
                    "config_slug": config.slug,
                    "exp_id": fault_run.run_id,
                    "ground_truth_service": fault_run.ground_truth_service,
                    "metrics": list(config.metrics),
                    "fusion_mode": config.fusion_mode,
                    "service_agg": config.service_agg,
                    "candidate_mode": config.candidate.mode,
                    "candidate_top_k": config.candidate.top_k,
                    "candidate_percentile": config.candidate.percentile,
                    "candidate_zscore_threshold": config.candidate.zscore_threshold,
                    "candidate_min": config.candidate.min_candidates,
                    "aggregation_seconds": AGGREGATION_SECONDS,
                    "fault_window_minutes": FAULT_WINDOW_MINUTES,
                    "normal_window_offset_minutes": NORMAL_WINDOW_OFFSET_MINUTES,
                    "normal_window_duration_minutes": NORMAL_WINDOW_DURATION_MINUTES,
                    "fault_window_source": FAULT_WINDOW_SOURCE,
                    "fault_start": str(fault_run.start),
                    "fault_end": str(fault_run.end),
                    "inject_start": str(fault_run.inject_start) if fault_run.inject_start is not None else None,
                    "pc_alpha": config.pc_alpha,
                    "beta": BETA,
                    "rho": RHO,
                    "walk_steps": WALK_STEPS,
                    "runtime_seconds": runtime_seconds,
                    "fault_metadata_path": str(fault_run.metadata_path),
                    "per_metric": summarize_metric_results(metric_results),
                    **eval_result,
                }

                save_run_outputs(
                    output_dir=config_output_dir / fault_run.run_id,
                    final_service_ranking=final_service_ranking,
                    metric_results=metric_results,
                    run_summary=run_summary,
                )
                detail_rows.append(
                    {
                        "telemetry_day": telemetry_day,
                        "config_slug": config.slug,
                        "metrics": config.metrics_label,
                        "fusion_mode": config.fusion_mode,
                        "service_agg": config.service_agg,
                        "candidate_mode": config.candidate.mode,
                        "candidate_top_k": config.candidate.top_k,
                        "candidate_percentile": config.candidate.percentile,
                        "candidate_zscore_threshold": config.candidate.zscore_threshold,
                        "candidate_min": config.candidate.min_candidates,
                        "search_profile": SEARCH_PROFILE,
                        "optimization_target": OPTIMIZATION_TARGET,
                        "aggregation_seconds": AGGREGATION_SECONDS,
                        "fault_window_minutes": FAULT_WINDOW_MINUTES,
                        "normal_window_offset_minutes": NORMAL_WINDOW_OFFSET_MINUTES,
                        "normal_window_duration_minutes": NORMAL_WINDOW_DURATION_MINUTES,
                        "fault_window_source": FAULT_WINDOW_SOURCE,
                        "pc_alpha": config.pc_alpha,
                        "exp_id": fault_run.run_id,
                        "ground_truth_service": fault_run.ground_truth_service,
                        "frontend_service": metric_results[0].frontend_service if metric_results else None,
                        "candidate_count": int(np.mean([len(result.impact_nodes) for result in metric_results])) if metric_results else 0,
                        "runtime_seconds": runtime_seconds,
                        **eval_result,
                    }
                )
                print(
                    f"[OK] {fault_run.run_id} top1={eval_result['service_top1_hit']} "
                    f"top3={eval_result['service_top3_hit']} top5={eval_result['service_top5_hit']} "
                    f"runtime={runtime_seconds:.2f}s"
                )
            except Exception as exc:
                runtime_seconds = perf_counter() - run_start
                detail_rows.append(
                    {
                        "telemetry_day": telemetry_day,
                        "config_slug": config.slug,
                        "metrics": config.metrics_label,
                        "fusion_mode": config.fusion_mode,
                        "service_agg": config.service_agg,
                        "candidate_mode": config.candidate.mode,
                        "candidate_top_k": config.candidate.top_k,
                        "candidate_percentile": config.candidate.percentile,
                        "candidate_zscore_threshold": config.candidate.zscore_threshold,
                        "candidate_min": config.candidate.min_candidates,
                        "search_profile": SEARCH_PROFILE,
                        "optimization_target": OPTIMIZATION_TARGET,
                        "aggregation_seconds": AGGREGATION_SECONDS,
                        "fault_window_minutes": FAULT_WINDOW_MINUTES,
                        "normal_window_offset_minutes": NORMAL_WINDOW_OFFSET_MINUTES,
                        "normal_window_duration_minutes": NORMAL_WINDOW_DURATION_MINUTES,
                        "fault_window_source": FAULT_WINDOW_SOURCE,
                        "pc_alpha": config.pc_alpha,
                        "exp_id": fault_run.run_id,
                        "ground_truth_service": fault_run.ground_truth_service,
                        "frontend_service": None,
                        "candidate_count": 0,
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

        config_runtime_seconds = perf_counter() - config_start
        summary_df = build_day_outputs(config_output_dir, detail_rows, config_runtime_seconds, config, telemetry_day)
        ablation_rows.extend(summary_df.to_dict("records"))

    total_runtime_seconds = perf_counter() - total_start
    ablation_df = pd.DataFrame(ablation_rows).sort_values(
        ["service_top1_accuracy", "service_top3_accuracy", "service_top5_accuracy", "config_slug"],
        ascending=[False, False, False, True],
    )
    if not ablation_df.empty:
        ablation_df["selection_score"] = compute_selection_score(ablation_df)
        ablation_df = ablation_df.sort_values(
            ["selection_score", "service_top1_accuracy", "service_top3_accuracy", "service_top5_accuracy", "config_slug"],
            ascending=[False, False, False, False, True],
        ).reset_index(drop=True)
    ablation_df["ablation_total_runtime_seconds"] = total_runtime_seconds
    day_output_dir.mkdir(parents=True, exist_ok=True)
    ablation_df.to_csv(day_output_dir / ABLATION_SUMMARY_FILE, index=False)

    if not ablation_df.empty:
        best_row = ablation_df.iloc[0].to_dict()
        (day_output_dir / BEST_CONFIG_FILE).write_text(
            json.dumps(best_row, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    if not ablation_df.empty:
        print(f"\n[DONE] CloudRanger ablation summary: {telemetry_day}")
        print(
            ablation_df[
                [
                    "config_slug",
                    "metrics",
                    "fusion_mode",
                    "candidate_mode",
                    "candidate_percentile",
                    "candidate_min",
                    "service_agg",
                    "pc_alpha",
                    "service_top1_accuracy",
                    "service_top3_accuracy",
                    "service_top5_accuracy",
                    "selection_score",
                    "total_runtime_seconds",
                ]
            ].to_string(index=False)
        )
        print(f"[INFO] Total script runtime across all configs: {total_runtime_seconds:.2f}s")
    return pd.concat(
        [
            pd.read_csv(get_config_output_dir(day_output_dir, config, len(experiment_configs)) / DAY_DETAILS_FILE)
            for config in experiment_configs
        ],
        ignore_index=True,
    ), pd.DataFrame(ablation_rows), ablation_df


def main() -> None:
    script_start = perf_counter()
    script_dir = SCRIPT_DIR
    all_details_dfs: list[pd.DataFrame] = []
    all_summary_dfs: list[pd.DataFrame] = []

    for telemetry_day in TELEMETRY_DAYS:
        details_df, summary_df, ablation_df = run_single_day(script_dir, telemetry_day)
        all_details_dfs.append(details_df)
        all_summary_dfs.append(summary_df)
        _ = ablation_df

    total_script_runtime_seconds = perf_counter() - script_start
    output_root = (script_dir / OUTPUT_ROOT).resolve()
    _, _, overall_summary_df, all_days_ablation_df = build_all_days_outputs(
        output_root=output_root,
        all_details_dfs=all_details_dfs,
        all_summary_dfs=all_summary_dfs,
        total_script_runtime_seconds=total_script_runtime_seconds,
    )

    print(f"[DONE] Multi-day details saved: {(output_root / ALL_DAYS_DETAILS_FILE).resolve()}")
    print(f"[DONE] Multi-day summary saved: {(output_root / ALL_DAYS_SUMMARY_FILE).resolve()}")
    print(f"[DONE] Overall summary saved: {(output_root / OVERALL_SUMMARY_FILE).resolve()}")
    print(f"[DONE] Multi-day ablation saved: {(output_root / ALL_DAYS_ABLATION_FILE).resolve()}")
    print(f"[DONE] Multi-day best config saved: {(output_root / ALL_DAYS_BEST_CONFIG_FILE).resolve()}")
    if not overall_summary_df.empty:
        print("\n[DONE] CloudRanger overall summary")
        print(
            overall_summary_df[
                [
                    "config_slug",
                    "metrics",
                    "fusion_mode",
                    "candidate_mode",
                    "candidate_percentile",
                    "candidate_min",
                    "service_agg",
                    "pc_alpha",
                    "service_top1_accuracy",
                    "service_top3_accuracy",
                    "service_top5_accuracy",
                    "total_runtime_seconds",
                ]
            ].to_string(index=False)
        )
    if not all_days_ablation_df.empty:
        print("\n[DONE] CloudRanger all-days ablation summary")
        print(
            all_days_ablation_df[
                [
                    "config_slug",
                    "metrics",
                    "fusion_mode",
                    "candidate_mode",
                    "candidate_percentile",
                    "candidate_min",
                    "service_agg",
                    "pc_alpha",
                    "service_top1_accuracy",
                    "service_top3_accuracy",
                    "service_top5_accuracy",
                    "selection_score",
                    "total_runtime_seconds",
                ]
            ].to_string(index=False)
        )
    print(f"[DONE] End-to-end script runtime across {len(TELEMETRY_DAYS)} day(s): {total_script_runtime_seconds:.2f}s")


if __name__ == "__main__":
    main()
