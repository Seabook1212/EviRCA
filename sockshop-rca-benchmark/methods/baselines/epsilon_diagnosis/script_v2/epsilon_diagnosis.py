import json
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import energy_distance
from tqdm import tqdm

# ===============================
# Configuration
# ===============================
TELEMETRY_DAYS = [
    "2026_03_12","2026_03_13","2026_03_14","2026_03_17","2026_03_18",
]

TELEMETRY_METRICS_DIR_TEMPLATE = "../../../dataset_v2/telemetry/{telemetry_day}/metrics"
TELEMETRY_FILE_GLOB = "prometheus_metrics_KPI_*.csv"

BATCH_PLAN_JSON_TEMPLATE = "../../../automatic_task_script/batch_task_data_v2_vm_{batch_plan_day_suffix}.json"
FAULT_METADATA_ROOT_TEMPLATE = "../../../dataset_v2/fault_run_{fault_metadata_day_suffix}"
NORMAL_RUN_ROOT_TEMPLATE = "../../../dataset_v2/normal_run_{normal_run_day_suffix}"
OUTPUT_ROOT = "../data_v2/fault_metrics"

# Override only when a telemetry day should use a different dataset day.
# By default, all day suffixes are derived directly from the telemetry day.
DAY_CONFIG_OVERRIDES: dict[str, dict[str, str]] = {
    # Example:
    # "2026_03_13": {"fault_metadata_day_suffix": "0312"},
}

# Choose which normal run should be used as the reference for each fault.
# For a baseline closer to the paper, prefer a historical normal period from the snapshot.
NORMAL_BASELINE_STRATEGY = "latest_preceding"

# Normal baseline window is inferred from each selected normal workload.
NORMAL_WINDOW_OFFSET_MINUTES = 5
NORMAL_WINDOW_DURATION_MINUTES = 5

# Fault window is derived from each exp's fault_metadata.inject_start.
FAULT_WINDOW_MINUTES = 5

N_PERMUTATIONS = 1000
MIN_SAMPLES = 30
RANDOM_SEED = 42
P_VALUE_THRESHOLD = 0.05
TOP_K = 10
# Rank pod/service candidates from detected metrics.
# Options:
# - count_then_sum: candidate_metric_count desc, then epsilon_sum desc
# - epsilon_sum: epsilon_sum desc
# - mean_epsilon: mean_epsilon desc
# - max_epsilon: max_epsilon desc
RANKING_MODE = "epsilon_sum"
# The paper uses all candidate metrics after the two-sample test, so no row cap is applied by default.
AGGREGATION_TOP_METRIC_ROWS: int | None = None
MAX_EXPERIMENTS: int | None = None
REUSE_EXISTING_EPSILON_RESULTS = True
PRINT_REMOVED_INDICATORS = False

# Filter threshold: treat very small variance as constant.
EPSILON = 1e-6

# Each indicator is defined by (pod, metric).
ENTITY_COL = "pod"
GROUP_KEYS = [ENTITY_COL, "metric"]

# Remove these KPIs (use p99 only for latency).
EXCLUDED_KPI_METRICS = {"latency_p50", "latency_p90", "latency_p95"}

# KPI file should contain these 10 metric indicators after exclusion.
EXPECTED_KPI_METRICS = {
    "request_rate",
    "success_rate",
    "error_count",
    "latency_p99",
    "cpu_usage_pct",
    "memory_usage_pct",
    "network_rx",
    "network_tx",
    "restart_count",
    "ready_ratio",
}

# For DB/pod-less middleware faults, service-level correctness is treated as pod-level correctness.
SERVICE_MATCH_IMPLIES_POD_MATCH = {
    "carts-db",
    "catalogue-db",
    "orders-db",
    "user-db",
    "session-db",
    "rabbitmq",
}

POD_RANKING_FILE = "epsilon_pod_ranking.csv"
SERVICE_RANKING_FILE = "epsilon_service_ranking.csv"

np.random.seed(RANDOM_SEED)


@dataclass(frozen=True)
class DayConfig:
    telemetry_day: str
    telemetry_day_suffix: str
    batch_plan_day_suffix: str
    fault_metadata_day_suffix: str
    normal_run_day_suffix: str
    metrics_dir: Path
    batch_plan_path: Path
    fault_metadata_root: Path
    normal_run_root: Path
    output_dir: Path


@dataclass(frozen=True)
class NormalBaseline:
    run_id: str
    workload_start: pd.Timestamp
    workload_end: pd.Timestamp
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    midpoint: pd.Timestamp


def _read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _telemetry_day_to_suffix(telemetry_day: str) -> str:
    return datetime.strptime(telemetry_day, "%Y_%m_%d").strftime("%m%d")


def _resolve_day_config(base_dir: Path, telemetry_day: str) -> DayConfig:
    telemetry_day_suffix = _telemetry_day_to_suffix(telemetry_day)
    overrides = DAY_CONFIG_OVERRIDES.get(telemetry_day, {})

    batch_plan_day_suffix = overrides.get("batch_plan_day_suffix", telemetry_day_suffix)
    fault_metadata_day_suffix = overrides.get("fault_metadata_day_suffix", telemetry_day_suffix)
    normal_run_day_suffix = overrides.get("normal_run_day_suffix", telemetry_day_suffix)

    metrics_dir = (base_dir / TELEMETRY_METRICS_DIR_TEMPLATE.format(telemetry_day=telemetry_day)).resolve()
    batch_plan_path = (base_dir / BATCH_PLAN_JSON_TEMPLATE.format(batch_plan_day_suffix=batch_plan_day_suffix)).resolve()
    fault_metadata_root = (
        base_dir / FAULT_METADATA_ROOT_TEMPLATE.format(fault_metadata_day_suffix=fault_metadata_day_suffix)
    ).resolve()
    normal_run_root = (base_dir / NORMAL_RUN_ROOT_TEMPLATE.format(normal_run_day_suffix=normal_run_day_suffix)).resolve()
    output_dir = (base_dir / OUTPUT_ROOT / telemetry_day).resolve()

    if not metrics_dir.exists():
        raise FileNotFoundError(f"Metrics dir not found for {telemetry_day}: {metrics_dir}")
    if not batch_plan_path.exists():
        raise FileNotFoundError(f"Batch plan not found for {telemetry_day}: {batch_plan_path}")
    if not fault_metadata_root.exists():
        raise FileNotFoundError(f"Fault metadata root not found for {telemetry_day}: {fault_metadata_root}")
    if not normal_run_root.exists():
        raise FileNotFoundError(f"Normal run root not found for {telemetry_day}: {normal_run_root}")

    return DayConfig(
        telemetry_day=telemetry_day,
        telemetry_day_suffix=telemetry_day_suffix,
        batch_plan_day_suffix=batch_plan_day_suffix,
        fault_metadata_day_suffix=fault_metadata_day_suffix,
        normal_run_day_suffix=normal_run_day_suffix,
        metrics_dir=metrics_dir,
        batch_plan_path=batch_plan_path,
        fault_metadata_root=fault_metadata_root,
        normal_run_root=normal_run_root,
        output_dir=output_dir,
    )


def _load_fault_exp_ids(plan_path: Path) -> list[str]:
    plan = _read_json(plan_path)
    if not isinstance(plan, list):
        raise ValueError(f"Batch plan must be a JSON list: {plan_path}")

    exp_ids: list[str] = []
    seen: set[str] = set()
    for step in plan:
        if not isinstance(step, dict):
            continue
        if str(step.get("task", "")).strip().lower() != "fault":
            continue
        exp_id = str(step.get("exp_id", "")).strip()
        if exp_id and exp_id not in seen:
            exp_ids.append(exp_id)
            seen.add(exp_id)

    if not exp_ids:
        raise ValueError(f"No fault exp_id found in {plan_path}")
    if MAX_EXPERIMENTS is not None:
        exp_ids = exp_ids[:MAX_EXPERIMENTS]
    return exp_ids


def _load_day_metrics(metrics_dir: Path, telemetry_day: str) -> pd.DataFrame:
    csv_files = sorted(metrics_dir.glob(TELEMETRY_FILE_GLOB))
    if not csv_files:
        raise FileNotFoundError(f"No telemetry files found in {metrics_dir} ({TELEMETRY_FILE_GLOB})")

    dfs: list[pd.DataFrame] = []
    use_cols = ["timestamp", "pod", "metric", "value"]
    for csv_path in csv_files:
        df = pd.read_csv(csv_path, usecols=use_cols)
        df["__source_file"] = csv_path.name
        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)
    return _validate_and_prepare_df(all_df, label=f"{telemetry_day} metrics")


def _validate_and_prepare_df(df: pd.DataFrame, label: str) -> pd.DataFrame:
    required_columns = {"timestamp", "pod", "metric", "value"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"{label} missing required columns: {sorted(missing)}")

    prepared = df.copy()
    prepared["timestamp"] = pd.to_datetime(prepared["timestamp"], errors="coerce")
    bad_ts_count = int(prepared["timestamp"].isna().sum())
    if bad_ts_count > 0:
        prepared = prepared[prepared["timestamp"].notna()].copy()
        print(f"[WARN] {label}: dropped {bad_ts_count} rows with invalid timestamp")

    prepared["value"] = pd.to_numeric(prepared["value"], errors="coerce")
    bad_val_count = int(prepared["value"].isna().sum())
    if bad_val_count > 0:
        prepared = prepared[prepared["value"].notna()].copy()
        print(f"[WARN] {label}: dropped {bad_val_count} rows with invalid value")

    if prepared.empty:
        raise ValueError(f"{label} has no valid rows after cleanup")

    return prepared


def _parse_utc_timestamp(value: str) -> pd.Timestamp:
    ts = pd.to_datetime(value, utc=True, errors="raise")
    return ts.tz_convert(None)


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
        raise ValueError(f"No workload metadata with start/end timestamps found in {normal_run_root}")

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

    raise ValueError(
        f"Unsupported NORMAL_BASELINE_STRATEGY={NORMAL_BASELINE_STRATEGY!r}. "
        "Use one of: earliest, latest_preceding, nearest."
    )


def _build_epsilon_result_context(
    day_config: DayConfig,
    exp_id: str,
    normal_baseline: NormalBaseline,
    fault_start: pd.Timestamp,
    fault_end: pd.Timestamp,
) -> dict[str, Any]:
    return {
        "telemetry_day": day_config.telemetry_day,
        "exp_id": exp_id,
        "normal_baseline_strategy": NORMAL_BASELINE_STRATEGY,
        "normal_run_id": normal_baseline.run_id,
        "normal_window_start": str(normal_baseline.window_start),
        "normal_window_end": str(normal_baseline.window_end),
        "fault_window_start": str(fault_start),
        "fault_window_end": str(fault_end),
        "normal_window_offset_minutes": NORMAL_WINDOW_OFFSET_MINUTES,
        "normal_window_duration_minutes": NORMAL_WINDOW_DURATION_MINUTES,
        "fault_window_minutes": FAULT_WINDOW_MINUTES,
        "min_samples": MIN_SAMPLES,
        "n_permutations": N_PERMUTATIONS,
        "epsilon_threshold": EPSILON,
        "excluded_metrics": sorted(EXCLUDED_KPI_METRICS),
    }


def _can_reuse_epsilon_results(context_path: Path, expected_context: dict[str, Any]) -> bool:
    if not context_path.exists():
        return False
    try:
        saved_context = _read_json(context_path)
    except Exception:
        return False
    return saved_context == expected_context


def _apply_time_window(
    df: pd.DataFrame,
    start_time: pd.Timestamp | None,
    end_time: pd.Timestamp | None,
    label: str,
    verbose: bool = True,
) -> pd.DataFrame:
    filtered = df
    if start_time is not None:
        filtered = filtered[filtered["timestamp"] >= start_time]
    if end_time is not None:
        filtered = filtered[filtered["timestamp"] <= end_time]

    filtered = filtered.copy()
    if filtered.empty:
        raise ValueError(
            f"{label} data is empty after time filtering: start={start_time}, end={end_time}"
        )

    if verbose:
        print(
            f"[INFO] {label} time range used: "
            f"{filtered['timestamp'].min()} -> {filtered['timestamp'].max()} (rows={len(filtered)})"
        )
    return filtered


def _exclude_metrics(df: pd.DataFrame, label: str, verbose: bool = True) -> pd.DataFrame:
    if not EXCLUDED_KPI_METRICS:
        return df
    filtered = df[~df["metric"].isin(EXCLUDED_KPI_METRICS)].copy()
    removed_rows = len(df) - len(filtered)
    if verbose:
        print(f"[INFO] {label} excluded metrics {sorted(EXCLUDED_KPI_METRICS)}: removed_rows={removed_rows}")
    if filtered.empty:
        raise ValueError(f"{label} data is empty after excluding metrics: {sorted(EXCLUDED_KPI_METRICS)}")
    return filtered


def _validate_kpi_metrics(df: pd.DataFrame, label: str) -> None:
    metrics = set(df["metric"].dropna().astype(str).unique())
    missing = sorted(EXPECTED_KPI_METRICS - metrics)
    extra = sorted(metrics - EXPECTED_KPI_METRICS)
    if missing or extra:
        raise ValueError(
            f"{label} KPI metrics mismatch. Expected {len(EXPECTED_KPI_METRICS)} metrics; "
            f"actual={len(metrics)}. Missing={missing}, Extra={extra}"
        )


def _calc_indicator_stats(df: pd.DataFrame) -> pd.DataFrame:
    stats = (
        df.groupby(GROUP_KEYS)["value"]
        .agg(std=lambda s: s.std(ddof=0), nunique="nunique")
        .reset_index()
    )
    stats["std"] = stats["std"].fillna(0.0)
    stats["is_constant"] = stats["nunique"] <= 1
    return stats


def _load_fault_metadata(fault_metadata_root: Path, exp_id: str) -> dict[str, Any]:
    metadata_path = (fault_metadata_root / exp_id / "fault_info" / "fault_metadata.json").resolve()
    data = _read_json(metadata_path)
    if not isinstance(data, dict):
        raise ValueError(f"fault_metadata.json is not an object: {metadata_path}")
    return data


def _fault_window_from_metadata(metadata: dict[str, Any]) -> tuple[pd.Timestamp, pd.Timestamp]:
    injection_info = metadata.get("injection_info")
    if not isinstance(injection_info, dict):
        raise ValueError("fault_metadata.json missing injection_info object")

    inject_start_raw = injection_info.get("inject_start")
    if not inject_start_raw:
        raise ValueError("fault_metadata.json missing injection_info.inject_start")

    start = _parse_utc_timestamp(inject_start_raw)
    end = start + pd.Timedelta(minutes=FAULT_WINDOW_MINUTES)
    return start, end


def _build_remove_key_set(
    normal_stats: pd.DataFrame,
    fault_stats: pd.DataFrame,
) -> tuple[set[tuple[str, str]], pd.DataFrame]:
    merged_stats = normal_stats.merge(fault_stats, on=GROUP_KEYS, how="inner")

    # Remove indicator if BOTH files are near-constant OR BOTH are exactly constant.
    std_based_mask = (merged_stats["normal_std"] < EPSILON) & (merged_stats["fault_std"] < EPSILON)
    constant_based_mask = merged_stats["normal_constant"] & merged_stats["fault_constant"]
    remove_mask = std_based_mask | constant_based_mask

    removed_stats = merged_stats.loc[remove_mask].copy()
    removed_stats["reason"] = ""
    removed_stats.loc[std_based_mask[remove_mask], "reason"] = "std < epsilon in both"
    removed_stats.loc[constant_based_mask[remove_mask], "reason"] = (
        removed_stats.loc[constant_based_mask[remove_mask], "reason"].replace("", "constant in both")
    )
    both_reason_mask = std_based_mask[remove_mask] & constant_based_mask[remove_mask]
    removed_stats.loc[both_reason_mask, "reason"] = "std < epsilon in both; constant in both"

    remove_key_set = set(map(tuple, removed_stats[GROUP_KEYS].to_numpy()))
    return remove_key_set, removed_stats


def _apply_constant_filter(df: pd.DataFrame, remove_key_set: set[tuple[str, str]]) -> pd.DataFrame:
    if not remove_key_set:
        return df.copy()
    key_index = pd.MultiIndex.from_frame(df[GROUP_KEYS])
    keep_mask = ~key_index.isin(remove_key_set)
    return df.loc[keep_mask].copy()


def _prepare_filtered_pair(
    normal_df: pd.DataFrame,
    fault_df: pd.DataFrame,
    remove_key_set: set[tuple[str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    normal_filtered = _apply_constant_filter(normal_df, remove_key_set)
    fault_filtered = _apply_constant_filter(fault_df, remove_key_set)
    normal_filtered = normal_filtered.sort_values(["timestamp", "pod", "metric"]).reset_index(drop=True)
    fault_filtered = fault_filtered.sort_values(["timestamp", "pod", "metric"]).reset_index(drop=True)
    return normal_filtered, fault_filtered


def _pod_to_service_name(pod_name: str) -> str:
    if not isinstance(pod_name, str):
        return str(pod_name)

    parts = pod_name.split("-")
    if len(parts) >= 3 and re.fullmatch(r"[a-z0-9]{5,}", parts[-2]) and re.fullmatch(
        r"[a-z0-9]{5}", parts[-1]
    ):
        return "-".join(parts[:-2])

    if len(parts) >= 2 and parts[-1].isdigit():
        return "-".join(parts[:-1])

    return pod_name


def permutation_test_energy(x: np.ndarray, y: np.ndarray, n_perm: int = 1000) -> tuple[float, float]:
    observed = float(energy_distance(x, y))

    combined = np.concatenate([x, y])
    n_x = len(x)

    count = 0
    for _ in range(n_perm):
        perm = np.random.permutation(combined)
        x_perm = perm[:n_x]
        y_perm = perm[n_x:]
        perm_stat = energy_distance(x_perm, y_perm)
        if perm_stat >= observed:
            count += 1

    p_value = (count + 1) / (n_perm + 1)
    return observed, float(p_value)


def _compute_epsilon_result(normal_df: pd.DataFrame, fault_df: pd.DataFrame) -> pd.DataFrame:
    results: list[dict[str, Any]] = []

    normal_groups = normal_df.groupby(GROUP_KEYS)
    fault_groups = fault_df.groupby(GROUP_KEYS)
    common_keys = sorted(set(normal_groups.groups.keys()) & set(fault_groups.groups.keys()))

    for entity_id, metric in common_keys:
        normal_values = normal_groups.get_group((entity_id, metric))["value"].to_numpy()
        fault_values = fault_groups.get_group((entity_id, metric))["value"].to_numpy()

        if len(normal_values) < MIN_SAMPLES or len(fault_values) < MIN_SAMPLES:
            continue

        epsilon, p_value = permutation_test_energy(normal_values, fault_values, n_perm=N_PERMUTATIONS)
        results.append(
            {
                ENTITY_COL: entity_id,
                "metric": metric,
                "epsilon": epsilon,
                "p_value": p_value,
                "n_normal": len(normal_values),
                "n_fault": len(fault_values),
            }
        )

    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df = result_df.sort_values(by="epsilon", ascending=False).reset_index(drop=True)
    return result_df


def _sort_ranking_table(ranking_df: pd.DataFrame) -> pd.DataFrame:
    normalized_mode = str(RANKING_MODE).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized_mode.endswith("_desc"):
        normalized_mode = normalized_mode[: -len("_desc")]

    aliases = {
        "count_then_sum": "count_then_sum",
        "count_sum": "count_then_sum",
        "epsilon_sum": "epsilon_sum",
        "sum_epsilon": "epsilon_sum",
        "mean_epsilon": "mean_epsilon",
        "avg_epsilon": "mean_epsilon",
        "average_epsilon": "mean_epsilon",
        "max_epsilon": "max_epsilon",
    }
    normalized_mode = aliases.get(normalized_mode, normalized_mode)

    if normalized_mode == "count_then_sum":
        order_cols = ["candidate_metric_count", "epsilon_sum"]
    elif normalized_mode == "epsilon_sum":
        order_cols = ["epsilon_sum"]
    elif normalized_mode == "mean_epsilon":
        order_cols = ["mean_epsilon"]
    elif normalized_mode == "max_epsilon":
        order_cols = ["max_epsilon"]
    else:
        raise ValueError(
            f"Unsupported RANKING_MODE={RANKING_MODE!r}. "
            "Use one of: count_then_sum, epsilon_sum, mean_epsilon, max_epsilon."
        )

    ascending = [False] * len(order_cols) + [True]
    return ranking_df.sort_values(by=order_cols + [ranking_df.columns[0]], ascending=ascending).reset_index(drop=True)


def _build_ranking_table(result_df: pd.DataFrame, level: str) -> pd.DataFrame:
    if result_df.empty:
        key_col = "pod" if level == "pod" else "service"
        return pd.DataFrame(columns=[key_col, "candidate_metric_count", "epsilon_sum", "mean_epsilon", "max_epsilon"])

    # Follow the paper's core decision rule: candidate metrics are those with P < alpha.
    ranked_df = result_df[result_df["p_value"] < P_VALUE_THRESHOLD].copy()
    if ranked_df.empty:
        # Fallback only for evaluation completeness when no candidate metrics are detected.
        ranked_df = result_df.sort_values(by=["p_value", "epsilon"], ascending=[True, False]).copy()

    if AGGREGATION_TOP_METRIC_ROWS is not None:
        ranked_df = ranked_df.head(AGGREGATION_TOP_METRIC_ROWS)

    if level == "pod":
        working_df = ranked_df.copy()
        key_col = "pod"
    elif level == "service":
        working_df = ranked_df.assign(service=lambda d: d["pod"].astype(str).map(_pod_to_service_name))
        key_col = "service"
    else:
        raise ValueError(f"Unsupported ranking level: {level!r}")

    ranking_df = (
        working_df.groupby(key_col)
        .agg(
            candidate_metric_count=("metric", "count"),
            epsilon_sum=("epsilon", "sum"),
            mean_epsilon=("epsilon", "mean"),
            max_epsilon=("epsilon", "max"),
        )
        .reset_index()
    )
    return _sort_ranking_table(ranking_df)


def _ground_truth_from_metadata(metadata: dict[str, Any]) -> tuple[str, str, bool]:
    injection_info = metadata.get("injection_info")
    if not isinstance(injection_info, dict):
        raise ValueError("fault_metadata.json missing injection_info object")

    gt_service = str(injection_info.get("service", "")).strip()
    raw_pod = injection_info.get("pod")
    gt_pod = str(raw_pod).strip() if raw_pod else (f"{gt_service}-0" if gt_service else "")

    service_implies_pod = (gt_service in SERVICE_MATCH_IMPLIES_POD_MATCH) or (not raw_pod)
    return gt_service, gt_pod, service_implies_pod


def _slice_or_all(values: list[str], size: int) -> list[str]:
    return values[:size]


def _evaluate_accuracy(
    telemetry_day: str,
    exp_id: str,
    normal_baseline: NormalBaseline,
    result_df: pd.DataFrame,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    gt_service, gt_pod, service_implies_pod = _ground_truth_from_metadata(metadata)
    candidate_df = result_df[result_df["p_value"] < P_VALUE_THRESHOLD].copy()
    candidate_pods = sorted(candidate_df["pod"].astype(str).unique()) if not candidate_df.empty else []
    candidate_services = (
        sorted(candidate_df["pod"].astype(str).map(_pod_to_service_name).unique()) if not candidate_df.empty else []
    )
    pod_ranking_df = _build_ranking_table(result_df, level="pod")
    service_ranking_df = _build_ranking_table(result_df, level="service")
    top_pods = list(pod_ranking_df["pod"].head(TOP_K)) if not pod_ranking_df.empty else []
    top_services = list(service_ranking_df["service"].head(TOP_K)) if not service_ranking_df.empty else []

    pred_service_top1 = top_services[0] if top_services else ""
    pred_pod_top1 = top_pods[0] if top_pods else ""
    pred_service_top3 = _slice_or_all(top_services, 3)
    pred_pod_top3 = _slice_or_all(top_pods, 3)
    pred_service_top5 = _slice_or_all(top_services, 5)
    pred_pod_top5 = _slice_or_all(top_pods, 5)

    service_top1_hit = bool(gt_service) and gt_service == pred_service_top1
    service_top3_hit = bool(gt_service) and gt_service in pred_service_top3
    service_top5_hit = bool(gt_service) and gt_service in pred_service_top5

    if service_implies_pod:
        pod_top1_hit = service_top1_hit
        pod_top3_hit = service_top3_hit
        pod_top5_hit = service_top5_hit
    else:
        pod_top1_hit = service_top1_hit and bool(gt_pod) and gt_pod == pred_pod_top1
        pod_top3_hit = service_top3_hit and bool(gt_pod) and gt_pod in pred_pod_top3
        pod_top5_hit = service_top5_hit and bool(gt_pod) and gt_pod in pred_pod_top5

    return {
        "telemetry_day": telemetry_day,
        "exp_id": exp_id,
        "status": "ok",
        "normal_run_id": normal_baseline.run_id,
        "normal_window_start": normal_baseline.window_start,
        "normal_window_end": normal_baseline.window_end,
        "pairs_evaluated": len(result_df),
        "candidate_metric_count": len(candidate_df),
        "candidate_pod_count": len(candidate_pods),
        "candidate_service_count": len(candidate_services),
        "ranking_mode": RANKING_MODE,
        "gt_service": gt_service,
        "gt_pod": gt_pod,
        "service_implies_pod_rule": service_implies_pod,
        "pred_service_top1": pred_service_top1,
        "pred_service_top3": "|".join(pred_service_top3),
        "pred_service_top5": "|".join(pred_service_top5),
        "pred_pod_top1": pred_pod_top1,
        "pred_pod_top3": "|".join(pred_pod_top3),
        "pred_pod_top5": "|".join(pred_pod_top5),
        "service_top1_hit": bool(service_top1_hit),
        "service_top3_hit": bool(service_top3_hit),
        "service_top5_hit": bool(service_top5_hit),
        "pod_top1_hit": bool(pod_top1_hit),
        "pod_top3_hit": bool(pod_top3_hit),
        "pod_top5_hit": bool(pod_top5_hit),
    }


def _build_filter_summary_row(
    telemetry_day: str,
    exp_id: str,
    normal_baseline: NormalBaseline,
    fault_start: pd.Timestamp,
    fault_end: pd.Timestamp,
    normal_filtered: pd.DataFrame,
    fault_filtered: pd.DataFrame,
    remove_key_set: set[tuple[str, str]],
) -> dict[str, Any]:
    return {
        "telemetry_day": telemetry_day,
        "exp_id": exp_id,
        "status": "ok",
        "normal_run_id": normal_baseline.run_id,
        "normal_window_start": normal_baseline.window_start,
        "normal_window_end": normal_baseline.window_end,
        "fault_start": fault_start,
        "fault_end": fault_end,
        "normal_rows": len(normal_filtered),
        "fault_rows": len(fault_filtered),
        "removed_indicators": len(remove_key_set),
        "error": "",
    }


def _build_error_detail_row(
    telemetry_day: str,
    exp_id: str,
    exc: Exception,
    normal_baseline: NormalBaseline | None = None,
) -> dict[str, Any]:
    return {
        "telemetry_day": telemetry_day,
        "exp_id": exp_id,
        "status": "error",
        "normal_run_id": normal_baseline.run_id if normal_baseline else "",
        "normal_window_start": normal_baseline.window_start if normal_baseline else "",
        "normal_window_end": normal_baseline.window_end if normal_baseline else "",
        "pairs_evaluated": 0,
        "candidate_metric_count": 0,
        "candidate_pod_count": 0,
        "candidate_service_count": 0,
        "gt_service": "",
        "gt_pod": "",
        "service_implies_pod_rule": False,
        "pred_service_top1": "",
        "pred_service_top3": "",
        "pred_service_top5": "",
        "pred_pod_top1": "",
        "pred_pod_top3": "",
        "pred_pod_top5": "",
        "service_top1_hit": False,
        "service_top3_hit": False,
        "service_top5_hit": False,
        "pod_top1_hit": False,
        "pod_top3_hit": False,
        "pod_top5_hit": False,
        "error": str(exc),
    }


def _build_error_filter_row(
    telemetry_day: str,
    exp_id: str,
    exc: Exception,
    normal_baseline: NormalBaseline | None = None,
) -> dict[str, Any]:
    return {
        "telemetry_day": telemetry_day,
        "exp_id": exp_id,
        "status": "error",
        "normal_run_id": normal_baseline.run_id if normal_baseline else "",
        "normal_window_start": normal_baseline.window_start if normal_baseline else "",
        "normal_window_end": normal_baseline.window_end if normal_baseline else "",
        "fault_start": "",
        "fault_end": "",
        "normal_rows": 0,
        "fault_rows": 0,
        "removed_indicators": 0,
        "error": str(exc),
    }


def _rate(df: pd.DataFrame, col: str) -> float:
    if df.empty:
        return 0.0
    return float(df[col].astype(float).mean())


def _build_overall_summary(
    details_df: pd.DataFrame,
    day_summary_df: pd.DataFrame,
    days_count: int,
) -> pd.DataFrame:
    ok_df = details_df[details_df["status"] == "ok"].copy()
    n_total = len(details_df)
    n_ok = len(ok_df)
    n_error = n_total - n_ok

    total_runtime_seconds = float(day_summary_df["total_runtime_seconds"].astype(float).sum()) if not day_summary_df.empty else 0.0
    avg_runtime_per_exception_seconds = total_runtime_seconds / n_total if n_total else 0.0

    summary_row = {
        "telemetry_day": "ALL_DAYS",
        "telemetry_day_suffix": "",
        "batch_plan_day_suffix": "",
        "fault_metadata_day_suffix": "",
        "normal_run_day_suffix": "",
        "normal_baseline_strategy": NORMAL_BASELINE_STRATEGY,
        "normal_runs_available": "",
        "normal_runs_used": "",
        "ranking_mode": RANKING_MODE,
        "days_count": days_count,
        "n_total": n_total,
        "n_ok": n_ok,
        "n_error": n_error,
        "service_top1_accuracy": _rate(ok_df, "service_top1_hit"),
        "service_top3_accuracy": _rate(ok_df, "service_top3_hit"),
        "service_top5_accuracy": _rate(ok_df, "service_top5_hit"),
        "pod_top1_accuracy": _rate(ok_df, "pod_top1_hit"),
        "pod_top3_accuracy": _rate(ok_df, "pod_top3_hit"),
        "pod_top5_accuracy": _rate(ok_df, "pod_top5_hit"),
        "total_runtime_seconds": total_runtime_seconds,
        "avg_runtime_per_exception_seconds": avg_runtime_per_exception_seconds,
    }
    return pd.DataFrame([summary_row])


def _run_single_day(day_config: DayConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    day_start = time.perf_counter()

    day_config.output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics_df = _load_day_metrics(day_config.metrics_dir, day_config.telemetry_day)
    exp_ids = _load_fault_exp_ids(day_config.batch_plan_path)
    normal_baselines = _load_normal_baselines(day_config.normal_run_root)
    print(
        f"[INFO] {day_config.telemetry_day} normal baseline strategy={NORMAL_BASELINE_STRATEGY} "
        f"(available_runs={len(normal_baselines)})"
    )

    filter_rows: list[dict[str, Any]] = []
    details_rows: list[dict[str, Any]] = []
    normal_context_cache: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    normal_runs_used: set[str] = set()

    print(f"[INFO] Evaluating {len(exp_ids)} fault experiment(s) for {day_config.telemetry_day}")

    for exp_id in tqdm(exp_ids, desc=f"epsilon-eval {day_config.telemetry_day}"):
        exp_dir = day_config.output_dir / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        normal_output_path = exp_dir / "normal_metrics_filtered.csv"
        fault_output_path = exp_dir / "fault_metrics_filtered.csv"
        result_output_path = exp_dir / "epsilon_results.csv"
        candidate_output_path = exp_dir / "epsilon_candidate_metrics.csv"
        pod_ranking_output_path = exp_dir / POD_RANKING_FILE
        service_ranking_output_path = exp_dir / SERVICE_RANKING_FILE
        context_output_path = exp_dir / "epsilon_result_context.json"
        normal_baseline: NormalBaseline | None = None

        try:
            metadata = _load_fault_metadata(day_config.fault_metadata_root, exp_id)
            fault_start, fault_end = _fault_window_from_metadata(metadata)
            normal_baseline = _select_normal_baseline(normal_baselines, fault_start)
            normal_runs_used.add(normal_baseline.run_id)

            if normal_baseline.run_id not in normal_context_cache:
                normal_df = _apply_time_window(
                    all_metrics_df,
                    normal_baseline.window_start,
                    normal_baseline.window_end,
                    label=f"{day_config.telemetry_day} normal {normal_baseline.run_id}",
                    verbose=False,
                )
                normal_df = _exclude_metrics(
                    normal_df,
                    f"{day_config.telemetry_day} normal {normal_baseline.run_id}",
                    verbose=False,
                )
                _validate_kpi_metrics(normal_df, f"{day_config.telemetry_day} normal {normal_baseline.run_id}")
                normal_stats = _calc_indicator_stats(normal_df).rename(
                    columns={
                        "std": "normal_std",
                        "nunique": "normal_nunique",
                        "is_constant": "normal_constant",
                    }
                )
                normal_context_cache[normal_baseline.run_id] = (normal_df, normal_stats)

            normal_df, normal_stats = normal_context_cache[normal_baseline.run_id]

            fault_df = _apply_time_window(
                all_metrics_df,
                fault_start,
                fault_end,
                label=f"{exp_id} fault",
                verbose=False,
            )
            fault_df = _exclude_metrics(fault_df, f"{exp_id} fault", verbose=False)
            _validate_kpi_metrics(fault_df, f"{exp_id} fault")

            fault_stats = _calc_indicator_stats(fault_df).rename(
                columns={
                    "std": "fault_std",
                    "nunique": "fault_nunique",
                    "is_constant": "fault_constant",
                }
            )

            remove_key_set, removed_stats = _build_remove_key_set(normal_stats, fault_stats)
            normal_filtered, fault_filtered = _prepare_filtered_pair(normal_df, fault_df, remove_key_set)
            normal_filtered.to_csv(normal_output_path, index=False)
            fault_filtered.to_csv(fault_output_path, index=False)

            epsilon_context = _build_epsilon_result_context(
                day_config,
                exp_id,
                normal_baseline,
                fault_start,
                fault_end,
            )

            if (
                REUSE_EXISTING_EPSILON_RESULTS
                and result_output_path.exists()
                and _can_reuse_epsilon_results(context_output_path, epsilon_context)
            ):
                result_df = pd.read_csv(result_output_path)
            else:
                result_df = _compute_epsilon_result(normal_filtered, fault_filtered)
                result_df.to_csv(result_output_path, index=False)
                _write_json(context_output_path, epsilon_context)

            candidate_df = result_df[result_df["p_value"] < P_VALUE_THRESHOLD].copy()
            candidate_df.to_csv(candidate_output_path, index=False)
            _build_ranking_table(result_df, level="pod").to_csv(pod_ranking_output_path, index=False)
            _build_ranking_table(result_df, level="service").to_csv(service_ranking_output_path, index=False)

            filter_rows.append(
                _build_filter_summary_row(
                    day_config.telemetry_day,
                    exp_id,
                    normal_baseline,
                    fault_start,
                    fault_end,
                    normal_filtered,
                    fault_filtered,
                    remove_key_set,
                )
            )
            details_rows.append(
                _evaluate_accuracy(
                    day_config.telemetry_day,
                    exp_id,
                    normal_baseline,
                    result_df,
                    metadata,
                )
            )

            if PRINT_REMOVED_INDICATORS and not removed_stats.empty:
                for row in removed_stats.sort_values(GROUP_KEYS).itertuples(index=False):
                    print(
                        f"  - pod={row.pod}, metric={row.metric}, reason={row.reason}, "
                        f"normal_std={row.normal_std:.6g}, fault_std={row.fault_std:.6g}"
                    )

        except Exception as exc:
            filter_rows.append(_build_error_filter_row(day_config.telemetry_day, exp_id, exc, normal_baseline))
            details_rows.append(_build_error_detail_row(day_config.telemetry_day, exp_id, exc, normal_baseline))

    filter_df = pd.DataFrame(filter_rows)
    details_df = pd.DataFrame(details_rows)

    ok_df = details_df[details_df["status"] == "ok"].copy()
    n_total = len(details_df)
    n_ok = len(ok_df)
    n_error = n_total - n_ok

    total_runtime_seconds = time.perf_counter() - day_start
    avg_runtime_per_exception_seconds = total_runtime_seconds / n_total if n_total else 0.0

    summary_row = {
        "telemetry_day": day_config.telemetry_day,
        "telemetry_day_suffix": day_config.telemetry_day_suffix,
        "batch_plan_day_suffix": day_config.batch_plan_day_suffix,
        "fault_metadata_day_suffix": day_config.fault_metadata_day_suffix,
        "normal_run_day_suffix": day_config.normal_run_day_suffix,
        "normal_baseline_strategy": NORMAL_BASELINE_STRATEGY,
        "normal_runs_available": len(normal_baselines),
        "normal_runs_used": len(normal_runs_used),
        "ranking_mode": RANKING_MODE,
        "n_total": n_total,
        "n_ok": n_ok,
        "n_error": n_error,
        "service_top1_accuracy": _rate(ok_df, "service_top1_hit"),
        "service_top3_accuracy": _rate(ok_df, "service_top3_hit"),
        "service_top5_accuracy": _rate(ok_df, "service_top5_hit"),
        "pod_top1_accuracy": _rate(ok_df, "pod_top1_hit"),
        "pod_top3_accuracy": _rate(ok_df, "pod_top3_hit"),
        "pod_top5_accuracy": _rate(ok_df, "pod_top5_hit"),
        "total_runtime_seconds": total_runtime_seconds,
        "avg_runtime_per_exception_seconds": avg_runtime_per_exception_seconds,
    }
    summary_df = pd.DataFrame([summary_row])

    filter_path = day_config.output_dir / f"filter_summary_{day_config.telemetry_day}.csv"
    details_path = day_config.output_dir / f"epsilon_accuracy_details_{day_config.telemetry_day}.csv"
    summary_path = day_config.output_dir / f"epsilon_accuracy_summary_{day_config.telemetry_day}.csv"
    resolved_config_path = day_config.output_dir / f"resolved_day_config_{day_config.telemetry_day}.json"

    filter_df.to_csv(filter_path, index=False)
    details_df.to_csv(details_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    _write_json(
        resolved_config_path,
        {
            **asdict(day_config),
            "metrics_dir": str(day_config.metrics_dir),
            "batch_plan_path": str(day_config.batch_plan_path),
            "fault_metadata_root": str(day_config.fault_metadata_root),
            "normal_run_root": str(day_config.normal_run_root),
            "output_dir": str(day_config.output_dir),
            "normal_baseline_strategy": NORMAL_BASELINE_STRATEGY,
            "normal_window_offset_minutes": NORMAL_WINDOW_OFFSET_MINUTES,
            "normal_window_duration_minutes": NORMAL_WINDOW_DURATION_MINUTES,
            "normal_runs_available": len(normal_baselines),
            "normal_runs_used": sorted(normal_runs_used),
            "aggregation_top_metric_rows": AGGREGATION_TOP_METRIC_ROWS,
            "ranking_mode": RANKING_MODE,
        },
    )

    print("[DONE] Accuracy summary")
    print(summary_df.to_string(index=False))
    print(f"[DONE] Filter summary saved: {filter_path}")
    print(f"[DONE] Details saved: {details_path}")
    print(f"[DONE] Summary saved: {summary_path}")
    print(f"[DONE] Resolved config saved: {resolved_config_path}")
    print(f"[DONE] Total script execution time: {total_runtime_seconds:.6f}s")
    print(f"[DONE] Average diagnosis time per exception: {avg_runtime_per_exception_seconds:.6f}s")

    return filter_df, details_df, summary_df


def main() -> None:
    script_start = time.perf_counter()
    script_dir = Path(__file__).resolve().parent
    aggregate_output_root = (script_dir / OUTPUT_ROOT).resolve()
    aggregate_output_root.mkdir(parents=True, exist_ok=True)

    all_filter_dfs: list[pd.DataFrame] = []
    all_details_dfs: list[pd.DataFrame] = []
    all_summary_dfs: list[pd.DataFrame] = []

    for telemetry_day in TELEMETRY_DAYS:
        day_config = _resolve_day_config(script_dir, telemetry_day)
        filter_df, details_df, summary_df = _run_single_day(day_config)
        all_filter_dfs.append(filter_df)
        all_details_dfs.append(details_df)
        all_summary_dfs.append(summary_df)

    if len(TELEMETRY_DAYS) > 1:
        combined_filter_df = pd.concat(all_filter_dfs, ignore_index=True)
        combined_details_df = pd.concat(all_details_dfs, ignore_index=True)
        combined_summary_df = pd.concat(all_summary_dfs, ignore_index=True)
        overall_summary_df = _build_overall_summary(
            combined_details_df,
            combined_summary_df,
            days_count=len(TELEMETRY_DAYS),
        )
        combined_summary_with_overall_df = pd.concat([combined_summary_df, overall_summary_df], ignore_index=True)

        combined_filter_path = aggregate_output_root / "filter_summary_all_days.csv"
        combined_details_path = aggregate_output_root / "epsilon_accuracy_details_all_days.csv"
        combined_summary_path = aggregate_output_root / "epsilon_accuracy_summary_all_days.csv"
        overall_summary_path = aggregate_output_root / "epsilon_accuracy_summary_overall.csv"

        combined_filter_df.to_csv(combined_filter_path, index=False)
        combined_details_df.to_csv(combined_details_path, index=False)
        combined_summary_with_overall_df.to_csv(combined_summary_path, index=False)
        overall_summary_df.to_csv(overall_summary_path, index=False)

        print(f"[DONE] Multi-day filter summary saved: {combined_filter_path}")
        print(f"[DONE] Multi-day details saved: {combined_details_path}")
        print(f"[DONE] Multi-day summary saved: {combined_summary_path}")
        print(f"[DONE] Overall summary saved: {overall_summary_path}")
        print("[DONE] Overall accuracy summary")
        print(overall_summary_df.to_string(index=False))

    total_script_runtime = time.perf_counter() - script_start
    print(f"[DONE] End-to-end script runtime across {len(TELEMETRY_DAYS)} day(s): {total_script_runtime:.6f}s")


if __name__ == "__main__":
    main()
