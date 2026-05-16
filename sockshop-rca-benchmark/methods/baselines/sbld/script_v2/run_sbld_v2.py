import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from scipy.cluster.hierarchy import fcluster, linkage


# ===============================
# CONFIG
# ===============================

TELEMETRY_DAYS = ["2026_03_12", "2026_03_13", "2026_03_14", "2026_03_17", "2026_03_18"]
EXP_ID = None  # Set to an exp_id string to run a single target experiment.

LOG_VARIANT = "parsed"  # "parsed" is recommended for this dataset.
EVIDENCE_MODE = "maximal"  # Paper-style maximal evidence: all failing + passing runs.

FAULT_WINDOW_MINUTES = 5
NORMAL_WINDOW_OFFSET_MINUTES = 5
NORMAL_WINDOW_DURATION_MINUTES = 5

DRAIN_DEPTH = 4
DRAIN_SIM_TH = 0.5
DRAIN_MAX_CHILDREN = 100
NORMALIZE_DYNAMIC_IDS = True

SCORING_MODE = "tarantula"  # Alternatives: "median_rank" or any INTERESTINGNESS_MEASURES entry.
CLUSTERING_MODE = "paper_hac"  # Alternatives: "current_gap" "paper_hac"

OUTPUT_ROOT = "../data_v2"
SERVICE_SCORE_FILE = "service_scores.csv"
TEMPLATE_SCORE_FILE = "template_scores.csv"
EVENT_CLUSTER_FILE = "event_clusters.csv"
RUN_SUMMARY_FILE = "sbld_run_summary.json"
DAY_DETAILS_FILE = "sbld_accuracy_details.csv"
DAY_SUMMARY_FILE = "sbld_accuracy_summary.csv"
ALL_DAYS_DETAILS_FILE = "sbld_accuracy_details_all_days.csv"
ALL_DAYS_SUMMARY_FILE = "sbld_accuracy_summary_all_days.csv"
OVERALL_SUMMARY_FILE = "sbld_accuracy_summary_overall.csv"
EXCLUDED_RANKED_SERVICES = {"istio-proxy"}

SAVE_INTERMEDIATE_OUTPUTS = False
PARSED_OUTPUT_FILE = "parsed_logs.csv"
COVERAGE_MATRIX_FILE = "coverage_matrix.csv"

FAULT_RUN_ROOT_TEMPLATE = "../../../dataset_v2/fault_run_{day_suffix}"
NORMAL_RUN_ROOT_TEMPLATE = "../../../dataset_v2/normal_run_{day_suffix}"
TELEMETRY_LOG_DIR_TEMPLATE = "../../../dataset_v2/telemetry/{telemetry_day}/logs"

INTERESTINGNESS_MEASURES = [
    "tarantula",
    "jaccard",
    "ochiai",
    "ochiai2",
    "zoltar",
    "dstar2",
    "op2",
    "wong3",
    "kulczynski2",
    "failed_only",
]

TRACE_RE = re.compile(r"(traceId:)[0-9a-fA-F]+")
SPAN_RE = re.compile(r"(spanId:)[0-9a-fA-F]+")
UUID_RE = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")
HEX_ID_RE = re.compile(r"\b[0-9a-fA-F]{16,32}\b")
ISO_TS_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z\b")
NUMBER_RE = re.compile(r"(?<![A-Za-z])\d+(?![A-Za-z])")


@dataclass(frozen=True)
class RunWindow:
    run_id: str
    label: str
    start: pd.Timestamp
    end: pd.Timestamp
    metadata_path: Path
    telemetry_day: str
    exp_id: str | None = None
    ground_truth_service: str | None = None


class HourlyLogCache:
    def __init__(self, log_variant: str):
        self.log_variant = log_variant
        self._cache: dict[Path, pd.DataFrame] = {}

    def load_file(self, file_path: Path) -> pd.DataFrame:
        cached = self._cache.get(file_path)
        if cached is not None:
            return cached

        if self.log_variant == "parsed":
            usecols = ["timestamp", "service", "pod", "message", "raw_log"]
        else:
            usecols = ["timestamp", "container", "pod", "log"]

        chunks: list[pd.DataFrame] = []
        for chunk in pd.read_csv(file_path, usecols=usecols, chunksize=100_000):
            standardized = _standardize_log_chunk(chunk, self.log_variant)
            standardized["timestamp"] = pd.to_datetime(standardized["timestamp"], errors="coerce", utc=True)
            standardized = standardized.dropna(subset=["timestamp", "service", "log"])
            if standardized.empty:
                continue

            standardized["timestamp"] = standardized["timestamp"].dt.tz_convert(None)
            standardized["service"] = standardized["service"].astype(str)
            standardized["pod"] = standardized["pod"].astype(str)
            standardized["log"] = standardized["log"].astype(str)
            chunks.append(standardized)

        loaded = (
            pd.concat(chunks, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
            if chunks else
            pd.DataFrame(columns=["timestamp", "service", "pod", "log"])
        )
        self._cache[file_path] = loaded
        return loaded


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
    log_dir = (script_dir / TELEMETRY_LOG_DIR_TEMPLATE.format(telemetry_day=telemetry_day)).resolve()
    return fault_root, normal_root, log_dir


def discover_fault_runs(script_dir: Path, telemetry_day: str) -> list[RunWindow]:
    fault_root, _, _ = get_day_paths(script_dir, telemetry_day)
    runs: list[RunWindow] = []
    for exp_dir in sorted(fault_root.iterdir()):
        if not exp_dir.is_dir():
            continue
        metadata_path = exp_dir / "fault_info" / "fault_metadata.json"
        if not metadata_path.exists():
            continue
        metadata = _read_json(metadata_path)
        injection_info = metadata.get("injection_info", {})
        inject_start_raw = injection_info.get("inject_start")
        ground_truth_service = injection_info.get("service") or metadata.get("service")
        if not inject_start_raw or not ground_truth_service:
            continue
        fault_start = parse_utc_timestamp(inject_start_raw)
        runs.append(
            RunWindow(
                run_id=exp_dir.name,
                label="fail",
                start=fault_start,
                end=fault_start + pd.Timedelta(minutes=FAULT_WINDOW_MINUTES),
                metadata_path=metadata_path.resolve(),
                telemetry_day=telemetry_day,
                exp_id=exp_dir.name,
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
        if normal_end > workload_end:
            continue
        runs.append(
            RunWindow(
                run_id=normal_dir.name,
                label="pass",
                start=normal_start,
                end=normal_end,
                metadata_path=metadata_path.resolve(),
                telemetry_day=telemetry_day,
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


def select_hourly_log_files(log_dir: Path, windows: list[tuple[pd.Timestamp, pd.Timestamp]], variant: str) -> list[Path]:
    if variant not in {"raw", "parsed"}:
        raise ValueError(f"Unsupported log variant: {variant}")

    selected_files = []
    for hour_start in _iter_hour_starts(windows):
        file_path = log_dir / f"loki_logs_{variant}_{hour_start.strftime('%H')}.csv"
        if file_path.exists():
            selected_files.append(file_path)
    if not selected_files:
        raise FileNotFoundError(f"No hourly log files found for {variant=} in {log_dir}")
    return selected_files


def build_window_mask(series: pd.Series, windows: list[tuple[pd.Timestamp, pd.Timestamp]]) -> pd.Series:
    mask = pd.Series(False, index=series.index)
    for start, end in windows:
        mask |= (series >= start) & (series <= end)
    return mask


def _standardize_log_chunk(chunk: pd.DataFrame, log_variant: str) -> pd.DataFrame:
    if log_variant == "parsed":
        if "service" not in chunk.columns:
            chunk["service"] = chunk.get("container")
        text_col = "message" if "message" in chunk.columns else "raw_log"
        standardized = chunk[["timestamp", "service", "pod", text_col]].copy()
        standardized.rename(columns={text_col: "log"}, inplace=True)
        return standardized

    if "container" not in chunk.columns:
        raise ValueError("Raw log schema missing container column")
    standardized = chunk[["timestamp", "container", "pod", "log"]].copy()
    standardized.rename(columns={"container": "service"}, inplace=True)
    return standardized


def load_logs_for_run(cache: HourlyLogCache, log_dir: Path, run: RunWindow) -> pd.DataFrame:
    window = [(run.start, run.end)]
    selected_files = select_hourly_log_files(log_dir, window, LOG_VARIANT)
    selected_chunks: list[pd.DataFrame] = []
    for log_file in selected_files:
        standardized = cache.load_file(log_file)
        if standardized.empty:
            continue
        windowed = standardized[build_window_mask(standardized["timestamp"], window)]
        if not windowed.empty:
            selected_chunks.append(windowed)
    if not selected_chunks:
        return pd.DataFrame(columns=["timestamp", "service", "pod", "log"])
    return pd.concat(selected_chunks, ignore_index=True).sort_values("timestamp").reset_index(drop=True)


def init_drain() -> TemplateMiner:
    config = TemplateMinerConfig()
    config.drain_depth = DRAIN_DEPTH
    config.drain_sim_th = DRAIN_SIM_TH
    config.drain_max_children = DRAIN_MAX_CHILDREN
    config.profiling_enabled = False
    return TemplateMiner(config=config)


def normalize_log(text: str) -> str:
    text = TRACE_RE.sub(r"\1<TRACE_ID>", text)
    text = SPAN_RE.sub(r"\1<SPAN_ID>", text)
    text = UUID_RE.sub("<UUID>", text)
    text = HEX_ID_RE.sub("<HEX_ID>", text)
    text = ISO_TS_RE.sub("<TS>", text)
    text = NUMBER_RE.sub("<NUM>", text)
    return text


def prepare_run_logs_for_parsing(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "service", "combined_log"])
    df = df.dropna(subset=["timestamp", "service", "log"]).copy()
    df["service"] = df["service"].astype(str)
    df["log"] = df["log"].astype(str)
    df["combined_log"] = "[" + df["service"] + "] " + df["log"]
    if NORMALIZE_DYNAMIC_IDS:
        df["combined_log"] = df["combined_log"].map(normalize_log)
    return df.sort_values("timestamp").reset_index(drop=True)


def abstract_all_runs(
    runs: list[RunWindow],
    log_dir: Path,
    cache: HourlyLogCache,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    template_miner = init_drain()
    parsed_outputs: dict[str, pd.DataFrame] = {}
    parsed_rows: list[pd.DataFrame] = []

    for index, run in enumerate(sorted(runs, key=lambda item: (item.start, item.run_id)), start=1):
        raw_df = load_logs_for_run(cache, log_dir, run)
        prepared_df = prepare_run_logs_for_parsing(raw_df)
        run_rows = []
        for row in prepared_df.itertuples(index=False):
            result = template_miner.add_log_message(row.combined_log)
            run_rows.append(
                {
                    "run_id": run.run_id,
                    "label": run.label,
                    "timestamp": row.timestamp,
                    "service": row.service,
                    "template_id": int(result["cluster_id"]),
                    "template_text": result["template_mined"],
                }
            )

        run_parsed = pd.DataFrame(run_rows)
        if run_parsed.empty:
            counts_df = pd.DataFrame(columns=["service", "template_id", "template_text", "count"])
        else:
            counts_df = (
                run_parsed.groupby(["service", "template_id", "template_text"])
                .size()
                .reset_index(name="count")
                .sort_values(["template_id", "service"])
                .reset_index(drop=True)
            )
        parsed_outputs[run.run_id] = counts_df
        if not run_parsed.empty:
            parsed_rows.append(run_parsed)
        print(f"[INFO] Abstracted {index}/{len(runs)} runs: {run.run_id} rows={len(run_parsed)}")

    parsed_logs_df = (
        pd.concat(parsed_rows, ignore_index=True).sort_values(["timestamp", "run_id"]).reset_index(drop=True)
        if parsed_rows else
        pd.DataFrame(columns=["run_id", "label", "timestamp", "service", "template_id", "template_text"])
    )
    return parsed_outputs, parsed_logs_df


def build_coverage_matrix(run_template_counts: dict[str, pd.DataFrame], runs: list[RunWindow]) -> pd.DataFrame:
    template_ids = sorted(
        {
            int(template_id)
            for df in run_template_counts.values()
            for template_id in df.get("template_id", pd.Series(dtype=int)).tolist()
        }
    )
    rows = []
    for run in runs:
        df = run_template_counts.get(run.run_id)
        present = set(df["template_id"].astype(int).tolist()) if df is not None and not df.empty else set()
        rows.append(
            {
                "run_id": run.run_id,
                **{template_id: int(template_id in present) for template_id in template_ids},
            }
        )
    coverage_df = pd.DataFrame(rows).set_index("run_id")
    if coverage_df.empty:
        raise ValueError("Coverage matrix is empty")
    return coverage_df


def _safe_divide(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.zeros_like(num, dtype=float)
    np.divide(num, den, out=out, where=den != 0)
    return out


def _wong3_penalty(aep: np.ndarray) -> np.ndarray:
    penalty = np.zeros_like(aep, dtype=float)
    mask1 = aep <= 2
    mask2 = (aep > 2) & (aep <= 10)
    mask3 = aep > 10
    penalty[mask1] = aep[mask1]
    penalty[mask2] = 2 + 0.1 * (aep[mask2] - 2)
    penalty[mask3] = 2.8 + 0.01 * (aep[mask3] - 10)
    return penalty


def compute_interestingness_scores(
    coverage_df: pd.DataFrame,
    failing_run_ids: list[str],
    passing_run_ids: list[str],
) -> pd.DataFrame:
    fail_df = coverage_df.loc[failing_run_ids]
    pass_df = coverage_df.loc[passing_run_ids]

    aef = fail_df.sum(axis=0).to_numpy(dtype=float)
    aep = pass_df.sum(axis=0).to_numpy(dtype=float)
    anf = len(failing_run_ids) - aef
    anp = len(passing_run_ids) - aep

    fail_rate = _safe_divide(aef, aef + anf)
    pass_rate = _safe_divide(aep, aep + anp)

    tarantula = _safe_divide(fail_rate, fail_rate + pass_rate)
    jaccard = _safe_divide(aef, aef + aep + anf)
    ochiai = _safe_divide(aef, np.sqrt((aef + aep) * (aef + anf)))
    ochiai2 = _safe_divide(
        aef * anp,
        np.sqrt((aef + aep) * (anf + anp) * (aef + anp) * (anf + aep)),
    )
    zoltar = np.zeros_like(aef, dtype=float)
    nz = aef != 0
    zoltar[nz] = aef[nz] / (
        aef[nz] + aep[nz] + anf[nz] + (10000.0 * anf[nz] * aep[nz] / aef[nz])
    )
    dstar2 = _safe_divide(aef ** 2, aep + anf)
    op2 = aef - _safe_divide(aep, anp + 1)
    wong3 = aef - _wong3_penalty(aep)
    kulczynski2 = 0.5 * (
        _safe_divide(aef, aef + anf) +
        _safe_divide(aef, aef + aep)
    )
    failed_only = np.where(aep == 0, aef, 0.0)

    scores_df = pd.DataFrame(
        {
            "template_id": coverage_df.columns.astype(int),
            "tarantula": tarantula,
            "jaccard": jaccard,
            "ochiai": ochiai,
            "ochiai2": ochiai2,
            "zoltar": zoltar,
            "dstar2": dstar2,
            "op2": op2,
            "wong3": wong3,
            "kulczynski2": kulczynski2,
            "failed_only": failed_only,
            "aef": aef,
            "aep": aep,
            "anf": anf,
            "anp": anp,
        }
    )
    return scores_df


def build_consensus_scores(scores_df: pd.DataFrame) -> pd.DataFrame:
    working = scores_df.copy()
    rank_columns = []
    norm_rank_columns = []
    n_templates = len(working)

    for measure in INTERESTINGNESS_MEASURES:
        rank_col = f"{measure}_rank"
        norm_col = f"{measure}_norm_rank"
        working[rank_col] = working[measure].rank(method="average", ascending=False)
        if n_templates <= 1:
            working[norm_col] = 1.0
        else:
            working[norm_col] = 1.0 - ((working[rank_col] - 1.0) / (n_templates - 1.0))
        rank_columns.append(rank_col)
        norm_rank_columns.append(norm_col)

    if SCORING_MODE == "median_rank":
        working["consensus_rank"] = working[rank_columns].median(axis=1)
        if n_templates <= 1:
            working["consensus_score"] = 1.0
        else:
            working["consensus_score"] = 1.0 - ((working["consensus_rank"] - 1.0) / (n_templates - 1.0))
    else:
        working["consensus_score"] = working[norm_rank_columns].median(axis=1)

    return working.sort_values("consensus_score", ascending=False).reset_index(drop=True)


def select_template_scores(scores_df: pd.DataFrame) -> pd.DataFrame:
    if SCORING_MODE == "median_rank":
        return build_consensus_scores(scores_df)

    if SCORING_MODE not in INTERESTINGNESS_MEASURES:
        raise ValueError(f"Unsupported SCORING_MODE: {SCORING_MODE}")

    working = scores_df.copy()
    n_templates = len(working)
    working["consensus_rank"] = working[SCORING_MODE].rank(method="average", ascending=False)
    if n_templates <= 1:
        working["consensus_score"] = 1.0
    else:
        working["consensus_score"] = 1.0 - ((working["consensus_rank"] - 1.0) / (n_templates - 1.0))
    return working.sort_values("consensus_score", ascending=False).reset_index(drop=True)


def _weighted_std(values: np.ndarray, weights: np.ndarray) -> float:
    if values.size == 0 or weights.sum() == 0:
        return 0.0
    mean = np.average(values, weights=weights)
    variance = np.average((values - mean) ** 2, weights=weights)
    return float(np.sqrt(variance))


def _empty_clustered_events() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "cluster_id",
            "cluster_rank",
            "cluster_score",
            "service",
            "template_id",
            "template_text",
            "count",
            "consensus_score",
        ]
    )


def _finalize_clustered_events(clustered: pd.DataFrame) -> pd.DataFrame:
    if clustered.empty:
        return _empty_clustered_events()

    working = clustered.copy()
    working["weighted_score"] = working["consensus_score"] * working["count"]
    cluster_scores = (
        working.groupby("cluster_id")
        .agg(
            cluster_score=("consensus_score", "mean"),
            cluster_event_count=("count", "sum"),
            cluster_max_score=("consensus_score", "max"),
        )
        .reset_index()
        .sort_values(
            ["cluster_score", "cluster_max_score", "cluster_event_count", "cluster_id"],
            ascending=[False, False, False, True],
        )
        .reset_index(drop=True)
    )
    cluster_scores["cluster_rank"] = np.arange(1, len(cluster_scores) + 1)

    working = working.merge(
        cluster_scores[["cluster_id", "cluster_score", "cluster_rank"]],
        on="cluster_id",
        how="left",
    )
    working = working.sort_values(
        ["cluster_rank", "weighted_score", "service", "template_id"],
        ascending=[True, False, True, True],
    ).reset_index(drop=True)
    return working.drop(columns=["weighted_score"])


def _cluster_target_events_current_gap(target_df: pd.DataFrame) -> pd.DataFrame:
    ordered = target_df.copy().sort_values(["consensus_score", "template_id"], ascending=[False, True]).reset_index(drop=True)
    threshold = _weighted_std(
        ordered["consensus_score"].to_numpy(dtype=float),
        ordered["count"].to_numpy(dtype=float),
    )

    cluster_ids = []
    current_cluster = 0
    current_max = None
    for score in ordered["consensus_score"].to_numpy(dtype=float):
        if current_max is None:
            current_cluster = 0
            current_max = score
        elif (current_max - score) > threshold:
            current_cluster += 1
            current_max = score
        cluster_ids.append(current_cluster)

    ordered["cluster_id"] = cluster_ids
    return _finalize_clustered_events(ordered)


def _cluster_target_events_paper_hac(target_df: pd.DataFrame) -> pd.DataFrame:
    ordered = target_df.copy().sort_values(["consensus_score", "template_id"], ascending=[False, True]).reset_index(drop=True)
    scores = ordered["consensus_score"].to_numpy(dtype=float)
    threshold = float(np.std(scores))

    # The paper clusters target-log events by their interestingness scores.
    # This implementation follows that more closely by applying HAC with
    # complete linkage directly on the 1D score values and using std(score)
    # as the distance threshold.
    if len(ordered) == 1:
        ordered["cluster_id"] = 0
        return _finalize_clustered_events(ordered)

    if threshold <= 0.0:
        unique_scores = pd.Series(scores).rank(method="dense", ascending=False).astype(int) - 1
        ordered["cluster_id"] = unique_scores.to_numpy(dtype=int)
        return _finalize_clustered_events(ordered)

    linkage_matrix = linkage(scores.reshape(-1, 1), method="complete", metric="euclidean")
    raw_cluster_ids = fcluster(linkage_matrix, t=threshold, criterion="distance")
    ordered["cluster_id"] = pd.Series(raw_cluster_ids).rank(method="dense").astype(int) - 1
    return _finalize_clustered_events(ordered)


def cluster_target_events(target_df: pd.DataFrame) -> pd.DataFrame:
    if target_df.empty:
        return _empty_clustered_events()

    if CLUSTERING_MODE == "paper_hac":
        return _cluster_target_events_paper_hac(target_df)
    if CLUSTERING_MODE == "current_gap":
        return _cluster_target_events_current_gap(target_df)
    raise ValueError(f"Unsupported CLUSTERING_MODE: {CLUSTERING_MODE}")


def rank_services_from_clusters(clustered_events: pd.DataFrame) -> pd.DataFrame:
    if clustered_events.empty:
        return pd.DataFrame(columns=["service", "rank", "first_cluster_rank", "cluster_score", "service_cluster_score"])

    service_rows = []
    seen = set()
    for cluster_rank in sorted(clustered_events["cluster_rank"].unique()):
        cluster_df = clustered_events[clustered_events["cluster_rank"] == cluster_rank].copy()
        service_scores = (
            cluster_df.assign(service_cluster_score=cluster_df["consensus_score"] * cluster_df["count"])
            .groupby("service")
            .agg(
                service_cluster_score=("service_cluster_score", "sum"),
                cluster_score=("cluster_score", "max"),
            )
            .reset_index()
            .loc[lambda df: ~df["service"].astype(str).isin(EXCLUDED_RANKED_SERVICES)]
            .sort_values(["service_cluster_score", "service"], ascending=[False, True])
        )
        for row in service_scores.itertuples(index=False):
            if row.service in seen:
                continue
            seen.add(row.service)
            service_rows.append(
                {
                    "service": row.service,
                    "rank": len(service_rows) + 1,
                    "first_cluster_rank": int(cluster_rank),
                    "cluster_score": float(row.cluster_score),
                    "service_cluster_score": float(row.service_cluster_score),
                }
            )
    return pd.DataFrame(service_rows)


def evaluate_topk(service_scores: pd.DataFrame, ground_truth_service: str) -> dict:
    ranked_services = service_scores["service"].astype(str).tolist()
    return {
        "predicted_service_top1": ranked_services[0] if ranked_services else None,
        "predicted_service_top3": ranked_services[:3],
        "predicted_service_top5": ranked_services[:5],
        "service_top1_hit": ground_truth_service in ranked_services[:1],
        "service_top3_hit": ground_truth_service in ranked_services[:3],
        "service_top5_hit": ground_truth_service in ranked_services[:5],
    }


def save_run_outputs(
    output_dir: Path,
    service_scores: pd.DataFrame,
    template_scores: pd.DataFrame,
    event_clusters: pd.DataFrame,
    run_summary: dict,
    parsed_target_events: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    service_scores.to_csv(output_dir / SERVICE_SCORE_FILE, index=False)
    template_scores.to_csv(output_dir / TEMPLATE_SCORE_FILE, index=False)
    event_clusters.to_csv(output_dir / EVENT_CLUSTER_FILE, index=False)
    (output_dir / RUN_SUMMARY_FILE).write_text(
        json.dumps(run_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if SAVE_INTERMEDIATE_OUTPUTS:
        parsed_target_events.to_csv(output_dir / PARSED_OUTPUT_FILE, index=False)


def build_day_outputs(
    script_dir: Path,
    telemetry_day: str,
    rows: list[dict],
    end_to_end_runtime_seconds: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    day_output_dir = (script_dir / OUTPUT_ROOT / telemetry_day).resolve()
    day_output_dir.mkdir(parents=True, exist_ok=True)

    details_df = pd.DataFrame(rows).sort_values("exp_id").reset_index(drop=True)
    details_df.to_csv(day_output_dir / DAY_DETAILS_FILE, index=False)

    n_total = len(details_df)
    summary_df = pd.DataFrame(
        [
            {
                "telemetry_day": telemetry_day,
                "n_total": n_total,
                "n_ok": int(details_df["predicted_service_top1"].notna().sum()) if n_total else 0,
                "n_error": int(details_df["predicted_service_top1"].isna().sum()) if n_total else 0,
                "service_top1_accuracy": float(details_df["service_top1_hit"].fillna(False).mean()) if n_total else np.nan,
                "service_top3_accuracy": float(details_df["service_top3_hit"].fillna(False).mean()) if n_total else np.nan,
                "service_top5_accuracy": float(details_df["service_top5_hit"].fillna(False).mean()) if n_total else np.nan,
                "evidence_mode": EVIDENCE_MODE,
                "scoring_mode": SCORING_MODE,
                "clustering_mode": CLUSTERING_MODE,
                "log_variant": LOG_VARIANT,
                "fault_window_minutes": FAULT_WINDOW_MINUTES,
                "normal_window_offset_minutes": NORMAL_WINDOW_OFFSET_MINUTES,
                "normal_window_duration_minutes": NORMAL_WINDOW_DURATION_MINUTES,
                "target_scoring_runtime_seconds_total": float(details_df["runtime_seconds"].fillna(0).sum()) if n_total else np.nan,
                "target_scoring_runtime_seconds_avg_per_target": float(details_df["runtime_seconds"].mean()) if n_total else np.nan,
                "end_to_end_runtime_seconds": end_to_end_runtime_seconds,
                "end_to_end_runtime_seconds_avg_per_target": (
                    end_to_end_runtime_seconds / n_total if n_total else np.nan
                ),
            }
        ]
    )
    summary_df.to_csv(day_output_dir / DAY_SUMMARY_FILE, index=False)
    return details_df, summary_df


def _rate(details_df: pd.DataFrame, column: str) -> float:
    if details_df.empty:
        return 0.0
    return float(details_df[column].fillna(False).mean())


def build_overall_outputs(
    script_dir: Path,
    all_details_dfs: list[pd.DataFrame],
    all_summary_dfs: list[pd.DataFrame],
    total_runtime_seconds: float,
) -> None:
    output_root = (script_dir / OUTPUT_ROOT).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    combined_details_df = pd.concat(all_details_dfs, ignore_index=True) if all_details_dfs else pd.DataFrame()
    combined_summary_df = pd.concat(all_summary_dfs, ignore_index=True) if all_summary_dfs else pd.DataFrame()

    if combined_details_df.empty:
        overall_summary_df = pd.DataFrame(
            [
                {
                    "telemetry_day": "ALL_DAYS",
                    "n_total": 0,
                    "n_ok": 0,
                    "n_error": 0,
                    "service_top1_accuracy": np.nan,
                    "service_top3_accuracy": np.nan,
                    "service_top5_accuracy": np.nan,
                    "evidence_mode": EVIDENCE_MODE,
                    "scoring_mode": SCORING_MODE,
                    "clustering_mode": CLUSTERING_MODE,
                    "log_variant": LOG_VARIANT,
                    "fault_window_minutes": FAULT_WINDOW_MINUTES,
                    "normal_window_offset_minutes": NORMAL_WINDOW_OFFSET_MINUTES,
                    "normal_window_duration_minutes": NORMAL_WINDOW_DURATION_MINUTES,
                    "target_scoring_runtime_seconds_total": 0.0,
                    "target_scoring_runtime_seconds_avg_per_target": np.nan,
                    "end_to_end_runtime_seconds": total_runtime_seconds,
                    "end_to_end_runtime_seconds_avg_per_target": np.nan,
                }
            ]
        )
    else:
        n_total = len(combined_details_df)
        overall_summary_df = pd.DataFrame(
            [
                {
                    "telemetry_day": "ALL_DAYS",
                    "n_total": n_total,
                    "n_ok": int(combined_details_df["predicted_service_top1"].notna().sum()),
                    "n_error": int(combined_details_df["predicted_service_top1"].isna().sum()),
                    "service_top1_accuracy": _rate(combined_details_df, "service_top1_hit"),
                    "service_top3_accuracy": _rate(combined_details_df, "service_top3_hit"),
                    "service_top5_accuracy": _rate(combined_details_df, "service_top5_hit"),
                    "evidence_mode": EVIDENCE_MODE,
                    "scoring_mode": SCORING_MODE,
                    "clustering_mode": CLUSTERING_MODE,
                    "log_variant": LOG_VARIANT,
                    "fault_window_minutes": FAULT_WINDOW_MINUTES,
                    "normal_window_offset_minutes": NORMAL_WINDOW_OFFSET_MINUTES,
                    "normal_window_duration_minutes": NORMAL_WINDOW_DURATION_MINUTES,
                    "target_scoring_runtime_seconds_total": float(combined_details_df["runtime_seconds"].fillna(0).sum()),
                    "target_scoring_runtime_seconds_avg_per_target": float(combined_details_df["runtime_seconds"].mean()),
                    "end_to_end_runtime_seconds": total_runtime_seconds,
                    "end_to_end_runtime_seconds_avg_per_target": total_runtime_seconds / n_total,
                }
            ]
        )

    combined_details_df.to_csv(output_root / ALL_DAYS_DETAILS_FILE, index=False)
    pd.concat([combined_summary_df, overall_summary_df], ignore_index=True).to_csv(
        output_root / ALL_DAYS_SUMMARY_FILE,
        index=False,
    )
    overall_summary_df.to_csv(output_root / OVERALL_SUMMARY_FILE, index=False)


def run_single_day(script_dir: Path, telemetry_day: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    total_start = perf_counter()
    full_fault_runs = discover_fault_runs(script_dir, telemetry_day)
    target_fault_runs = full_fault_runs
    normal_runs = discover_normal_runs(script_dir, telemetry_day)
    if EXP_ID:
        target_fault_runs = [run for run in full_fault_runs if run.run_id == EXP_ID]
        if not target_fault_runs:
            raise ValueError(f"EXP_ID not found for {telemetry_day}: {EXP_ID}")

    _, _, log_dir = get_day_paths(script_dir, telemetry_day)
    cache = HourlyLogCache(LOG_VARIANT)

    all_runs = normal_runs + full_fault_runs
    all_windows = [(run.start, run.end) for run in all_runs]
    selected_files = select_hourly_log_files(log_dir, all_windows, LOG_VARIANT)

    print(
        f"[INFO] telemetry_day={telemetry_day} targets={len(target_fault_runs)} "
        f"failing_runs={len(full_fault_runs)} passing_runs={len(normal_runs)} "
        f"hourly_files={len(selected_files)}"
    )

    run_template_counts, parsed_logs_df = abstract_all_runs(all_runs, log_dir, cache)
    coverage_df = build_coverage_matrix(run_template_counts, all_runs)
    if SAVE_INTERMEDIATE_OUTPUTS:
        day_output_dir = (script_dir / OUTPUT_ROOT / telemetry_day).resolve()
        day_output_dir.mkdir(parents=True, exist_ok=True)
        parsed_logs_df.to_csv(day_output_dir / PARSED_OUTPUT_FILE, index=False)
        coverage_df.to_csv(day_output_dir / COVERAGE_MATRIX_FILE)

    failing_run_ids = [run.run_id for run in full_fault_runs]
    passing_run_ids = [run.run_id for run in normal_runs]
    template_scores = select_template_scores(
        compute_interestingness_scores(coverage_df, failing_run_ids, passing_run_ids)
    )
    template_score_map = template_scores.set_index("template_id")

    detail_rows = []
    for index, target_run in enumerate(target_fault_runs, start=1):
        run_start = perf_counter()
        target_counts = run_template_counts.get(target_run.run_id)
        if target_counts is None or target_counts.empty:
            row = {
                "telemetry_day": telemetry_day,
                "exp_id": target_run.run_id,
                "ground_truth_service": target_run.ground_truth_service,
                "predicted_service_top1": None,
                "predicted_service_top3": [],
                "predicted_service_top5": [],
                "service_top1_hit": False,
                "service_top3_hit": False,
                "service_top5_hit": False,
                "runtime_seconds": np.nan,
                "error": "Target run has no parsed log events",
            }
            detail_rows.append(row)
            print(f"[WARN] [{index}/{len(target_fault_runs)}] {target_run.run_id} has no parsed events")
            continue

        scored_target = target_counts.merge(
            template_score_map,
            on="template_id",
            how="left",
        )
        scored_target["consensus_score"] = scored_target["consensus_score"].fillna(0.0)
        clustered_events = cluster_target_events(
            scored_target[["service", "template_id", "template_text", "count", "consensus_score"]]
        )
        service_scores = rank_services_from_clusters(clustered_events)
        eval_result = evaluate_topk(service_scores, target_run.ground_truth_service or "")
        runtime_seconds = perf_counter() - run_start

        output_dir = (script_dir / OUTPUT_ROOT / telemetry_day / target_run.run_id).resolve()
        run_summary = {
            "telemetry_day": telemetry_day,
            "exp_id": target_run.run_id,
            "ground_truth_service": target_run.ground_truth_service,
            "fault_start": str(target_run.start),
            "fault_end": str(target_run.end),
            "evidence_mode": EVIDENCE_MODE,
            "scoring_mode": SCORING_MODE,
            "clustering_mode": CLUSTERING_MODE,
            "log_variant": LOG_VARIANT,
            "template_count_in_target": int(scored_target["template_id"].nunique()),
            "event_count_in_target": int(scored_target["count"].sum()),
            "cluster_count": int(clustered_events["cluster_id"].nunique()) if not clustered_events.empty else 0,
            "service_count": int(service_scores["service"].nunique()),
            "fault_metadata_path": str(target_run.metadata_path),
            "runtime_seconds": runtime_seconds,
            **eval_result,
        }
        save_run_outputs(
            output_dir=output_dir,
            service_scores=service_scores,
            template_scores=template_scores,
            event_clusters=clustered_events,
            run_summary=run_summary,
            parsed_target_events=scored_target,
        )

        detail_rows.append(
            {
                "telemetry_day": telemetry_day,
                "exp_id": target_run.run_id,
                "ground_truth_service": target_run.ground_truth_service,
                "fault_start": target_run.start,
                "fault_end": target_run.end,
                "runtime_seconds": runtime_seconds,
                **eval_result,
            }
        )
        print(
            f"[OK] [{index}/{len(target_fault_runs)}] {target_run.run_id} "
            f"top1={eval_result['service_top1_hit']} top3={eval_result['service_top3_hit']} "
            f"top5={eval_result['service_top5_hit']} runtime={runtime_seconds:.2f}s"
        )

    total_runtime = perf_counter() - total_start
    details_df, summary_df = build_day_outputs(script_dir, telemetry_day, detail_rows, total_runtime)
    print(f"\n[DONE] SBLD day summary: {telemetry_day}")
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
    print(f"[INFO] Total runtime: {total_runtime:.2f}s")
    return details_df, summary_df


def main() -> None:
    script_start = perf_counter()
    script_dir = Path(__file__).resolve().parent
    all_details_dfs: list[pd.DataFrame] = []
    all_summary_dfs: list[pd.DataFrame] = []

    for telemetry_day in TELEMETRY_DAYS:
        details_df, summary_df = run_single_day(script_dir, telemetry_day)
        all_details_dfs.append(details_df)
        all_summary_dfs.append(summary_df)

    total_runtime_seconds = perf_counter() - script_start
    build_overall_outputs(script_dir, all_details_dfs, all_summary_dfs, total_runtime_seconds)
    output_root = (script_dir / OUTPUT_ROOT).resolve()
    print(f"[DONE] Multi-day details saved: {(output_root / ALL_DAYS_DETAILS_FILE).resolve()}")
    print(f"[DONE] Multi-day summary saved: {(output_root / ALL_DAYS_SUMMARY_FILE).resolve()}")
    print(f"[DONE] Overall summary saved: {(output_root / OVERALL_SUMMARY_FILE).resolve()}")
    print(f"[DONE] End-to-end script runtime across {len(TELEMETRY_DAYS)} day(s): {total_runtime_seconds:.2f}s")


if __name__ == "__main__":
    main()
