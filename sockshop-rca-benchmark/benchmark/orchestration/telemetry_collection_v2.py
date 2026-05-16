from __future__ import annotations

import csv
import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATASET_BASE = PROJECT_ROOT / "chaos_experiment" / "dataset_v2" / "telemetry"

# =========================
# Config
# =========================
DATE_TEXT = "2026_02_28"
HOURS: str | list[int] = [15]  # "all" or e.g. [1, 2, 13]
TIMEZONE = timezone.utc
PYTHON_BIN = sys.executable

OUTPUT_ROOT = DATASET_BASE
OUTPUT_DIR_NAMES = {
    "logs": "logs",
    "traces": "traces",
    "metrics": "metrics",
}

PROM_URL = os.environ.get("PROM_URL", "http://34.28.33.102:30990")
PROM_NAMESPACE = os.environ.get("PROM_NAMESPACE", "sock-shop")
LOKI_URL = os.environ.get("LOKI_URL", "http://34.28.33.102:31300")
LOKI_QUERY = os.environ.get("LOKI_QUERY", '{namespace="sock-shop"}')
JAEGER_URL = os.environ.get("JAEGER_URL", "http://34.28.33.102:32614")

PROM_STEP = os.environ.get("PROM_STEP", "5s")
KUBE_POD_STEP = os.environ.get("KUBE_POD_STEP", "5m")
KPI_WINDOW = os.environ.get("KPI_WINDOW", "30s")
ISTIO_WINDOW = os.environ.get("ISTIO_WINDOW", "30s")

RUN_METRICS = True
RUN_LOGS = True
RUN_TRACES = True

METRIC_COLLECTORS = [
    {
        "name": "application",
        "script": PROJECT_ROOT / "chaos_experiment" / "metrics_script" / "prometheus_application_script_v2.py",
        "output_env": "APPLICATION_METRIC_OUTPUT_FILE",
        "output_name": "prometheus_metrics_application_raw",
    },
    {
        "name": "container",
        "script": PROJECT_ROOT / "chaos_experiment" / "metrics_script" / "prometheus_container_script_v2.py",
        "output_env": "CONTAINER_METRIC_OUTPUT_FILE",
        "output_name": "prometheus_metrics_container_raw",
    },
    {
        "name": "KPI",
        "script": PROJECT_ROOT / "chaos_experiment" / "metrics_script" / "prometheus_KPI_script_v2.py",
        "output_env": "KPI_OUTPUT_FILE",
        "output_name": "prometheus_metrics_KPI",
    },
    {
        "name": "middleware",
        "script": PROJECT_ROOT / "chaos_experiment" / "metrics_script" / "prometheus_middleware_script_v2.py",
        "output_env": "MIDDLEWARE_METRIC_OUTPUT_FILE",
        "output_name": "prometheus_metrics_middleware_raw",
    },
    {
        "name": "network",
        "script": PROJECT_ROOT / "chaos_experiment" / "metrics_script" / "prometheus_network_script_v2.py",
        "output_env": "NGINX_METRIC_OUTPUT_FILE",
        "output_name": "prometheus_metrics_network_raw",
        "step": "30s",
    },
    {
        "name": "node",
        "script": PROJECT_ROOT / "chaos_experiment" / "metrics_script" / "prometheus_node_script_v2.py",
        "output_env": "NODE_METRIC_OUTPUT_FILE",
        "output_name": "prometheus_metrics_node_raw",
    },
    {
        "name": "service_proxy",
        "script": PROJECT_ROOT / "chaos_experiment" / "metrics_script" / "prometheus_service_proxy_script_v2.py",
        "output_env": "ISTIO_METRIC_OUTPUT_FILE",
        "output_name": "prometheus_metrics_service_proxy_raw",
        "step": "30s",
    },
]

LOKI_SCRIPT = PROJECT_ROOT / "chaos_experiment" / "logs_script" / "loki_script_v2.py"
LOKI_PARSE_SCRIPT = PROJECT_ROOT / "chaos_experiment" / "logs_script" / "loki_logs_parse_script_v2.py"
JAEGER_SCRIPT = PROJECT_ROOT / "chaos_experiment" / "traces_script" / "jaeger_script_v2.py"
JAEGER_PARSE_SCRIPT = PROJECT_ROOT / "chaos_experiment" / "traces_script" / "jaeger_parse_script_v2.py"

SUMMARY_FILE_NAME = "collection_summary.json"

LOG_RAW_COLUMNS = ["timestamp", "node", "pod", "container", "log"]
TRACE_RAW_COLUMNS = [
    "start_time",
    "trace_id",
    "span_id",
    "service",
    "operation",
    "duration",
    "references",
    "tags",
    "logs",
]


def parse_date(date_text: str) -> datetime:
    return datetime.strptime(date_text, "%Y_%m_%d").replace(tzinfo=TIMEZONE)


def normalize_hours(hours: str | list[int]) -> list[int]:
    if isinstance(hours, str):
        if hours.lower() == "all":
            return list(range(24))
        raise ValueError("HOURS must be 'all' or a list of integers between 0 and 23.")

    normalized = sorted(set(int(hour) for hour in hours))
    invalid = [hour for hour in normalized if hour < 0 or hour > 23]
    if invalid:
        raise ValueError(f"Invalid HOURS entries: {invalid}")
    return normalized


def to_env_time(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def hour_suffix(hour: int) -> str:
    return f"{hour:02d}"


def build_hour_windows(date_text: str, hours: str | list[int]) -> list[dict[str, Any]]:
    base = parse_date(date_text)
    windows = []
    for hour in normalize_hours(hours):
        start_dt = base + timedelta(hours=hour)
        end_dt = start_dt + timedelta(hours=1)
        windows.append(
            {
                "hour": hour,
                "suffix": hour_suffix(hour),
                "start_dt": start_dt,
                "end_dt": end_dt,
                "start_iso": to_env_time(start_dt),
                "end_iso": to_env_time(end_dt),
            }
        )
    return windows


def ensure_output_dirs(date_text: str) -> dict[str, Path]:
    date_dir = OUTPUT_ROOT / date_text
    dirs = {"date": date_dir}
    for key, dirname in OUTPUT_DIR_NAMES.items():
        path = date_dir / dirname
        path.mkdir(parents=True, exist_ok=True)
        dirs[key] = path
    return dirs


def write_empty_csv(path: Path, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)


def run_subprocess(script_path: Path, extra_env: dict[str, str]) -> None:
    env = os.environ.copy()
    env.update(extra_env)
    subprocess.run(
        [PYTHON_BIN, str(script_path)],
        cwd=str(PROJECT_ROOT),
        env=env,
        check=True,
    )


def load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def collect_logs(window: dict[str, Any], logs_dir: Path) -> list[dict[str, str]]:
    loki_module = load_module("telemetry_loki_script_v2", LOKI_SCRIPT)

    raw_name = f"loki_logs_raw_{window['suffix']}.csv"
    parsed_name = f"loki_logs_parsed_{window['suffix']}.csv"
    raw_path = logs_dir / raw_name
    parsed_path = logs_dir / parsed_name

    loki_module.LOKI_URL = LOKI_URL
    loki_module.LOG_QUERY = LOKI_QUERY
    loki_module.USE_RELATIVE_WINDOW = False
    loki_module.START_ISO = window["start_dt"].astimezone(timezone.utc).isoformat()
    loki_module.END_ISO = window["end_dt"].astimezone(timezone.utc).isoformat()

    rows = loki_module.fetch_logs_to_rows()
    if rows:
        loki_module.save_rows(rows, output_dir=logs_dir, output_file=raw_name)
    else:
        write_empty_csv(raw_path, LOG_RAW_COLUMNS)

    run_subprocess(
        LOKI_PARSE_SCRIPT,
        {
            "LOKI_PARSE_INPUT_FILE": str(raw_path),
            "LOKI_PARSE_OUTPUT_FILE": str(parsed_path),
        },
    )

    return [
        {"type": "logs_raw", "path": str(raw_path)},
        {"type": "logs_parsed", "path": str(parsed_path)},
    ]


def collect_traces(window: dict[str, Any], traces_dir: Path) -> list[dict[str, str]]:
    raw_name = f"jaeger_traces_raw_{window['suffix']}.csv"
    parsed_name = f"jaeger_traces_parsed_{window['suffix']}.csv"
    raw_path = traces_dir / raw_name
    parsed_path = traces_dir / parsed_name

    run_subprocess(
        JAEGER_SCRIPT,
        {
            "JAEGER_URL": JAEGER_URL,
            "JAEGER_START": window["start_iso"],
            "JAEGER_END": window["end_iso"],
            "JAEGER_OUTPUT_DIR": str(traces_dir),
            "JAEGER_OUTPUT_FILE": raw_name,
            "PROM_START": window["start_iso"],
            "PROM_END": window["end_iso"],
        },
    )
    if not raw_path.exists():
        write_empty_csv(raw_path, TRACE_RAW_COLUMNS)
    run_subprocess(
        JAEGER_PARSE_SCRIPT,
        {
            "JAEGER_PARSE_INPUT_FILE": str(raw_path),
            "JAEGER_PARSE_OUTPUT_FILE": str(parsed_path),
        },
    )

    return [
        {"type": "traces_raw", "path": str(raw_path)},
        {"type": "traces_parsed", "path": str(parsed_path)},
    ]


def collect_metrics(window: dict[str, Any], metrics_dir: Path) -> list[dict[str, str]]:
    outputs: list[dict[str, str]] = []

    base_env = {
        "PROM_URL": PROM_URL,
        "PROM_NAMESPACE": PROM_NAMESPACE,
        "PROM_START": window["start_iso"],
        "PROM_END": window["end_iso"],
        "START_TIME": window["start_iso"],
        "END_TIME": window["end_iso"],
        "PROM_STEP": PROM_STEP,
        "KUBE_POD_STEP": KUBE_POD_STEP,
        "KPI_WINDOW": KPI_WINDOW,
        "ISTIO_WINDOW": ISTIO_WINDOW,
    }

    for collector in METRIC_COLLECTORS:
        output_name = f"{collector['output_name']}_{window['suffix']}.csv"
        output_path = metrics_dir / output_name
        env = dict(base_env)
        env["PROM_STEP"] = collector.get("step", PROM_STEP)
        env[collector["output_env"]] = str(output_path)
        run_subprocess(collector["script"], env)
        outputs.append({"type": collector["name"], "path": str(output_path)})

    return outputs


def write_summary(summary_path: Path, summary: dict[str, Any]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    windows = build_hour_windows(DATE_TEXT, HOURS)
    output_dirs = ensure_output_dirs(DATE_TEXT)

    summary: dict[str, Any] = {
        "date": DATE_TEXT,
        "timezone": str(TIMEZONE),
        "python_bin": PYTHON_BIN,
        "output_root": str(output_dirs["date"]),
        "config": {
            "hours": normalize_hours(HOURS),
            "run_metrics": RUN_METRICS,
            "run_logs": RUN_LOGS,
            "run_traces": RUN_TRACES,
            "prom_url": PROM_URL,
            "prom_namespace": PROM_NAMESPACE,
            "loki_url": LOKI_URL,
            "loki_query": LOKI_QUERY,
            "jaeger_url": JAEGER_URL,
            "prom_step": PROM_STEP,
            "kube_pod_step": KUBE_POD_STEP,
            "kpi_window": KPI_WINDOW,
            "istio_window": ISTIO_WINDOW,
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "windows": [],
    }

    print(f"Collecting telemetry for {DATE_TEXT}")
    print(f"Hours: {summary['config']['hours']}")
    print(f"Output root: {output_dirs['date']}")

    for window in windows:
        print("\n" + "=" * 72)
        print(
            f"Hour {window['suffix']}: {window['start_iso']} -> {window['end_iso']}"
        )

        window_result: dict[str, Any] = {
            "hour": window["hour"],
            "suffix": window["suffix"],
            "start_iso": window["start_iso"],
            "end_iso": window["end_iso"],
            "status": "success",
            "outputs": [],
            "errors": [],
        }

        if RUN_METRICS:
            try:
                print("[metrics] collecting...")
                window_result["outputs"].extend(collect_metrics(window, output_dirs["metrics"]))
            except Exception as exc:
                window_result["status"] = "partial_failed"
                window_result["errors"].append({"stage": "metrics", "message": str(exc)})
                print(f"[metrics] failed: {exc}")

        if RUN_LOGS:
            try:
                print("[logs] collecting and parsing...")
                window_result["outputs"].extend(collect_logs(window, output_dirs["logs"]))
            except Exception as exc:
                window_result["status"] = "partial_failed"
                window_result["errors"].append({"stage": "logs", "message": str(exc)})
                print(f"[logs] failed: {exc}")

        if RUN_TRACES:
            try:
                print("[traces] collecting and parsing...")
                window_result["outputs"].extend(collect_traces(window, output_dirs["traces"]))
            except Exception as exc:
                window_result["status"] = "partial_failed"
                window_result["errors"].append({"stage": "traces", "message": str(exc)})
                print(f"[traces] failed: {exc}")

        summary["windows"].append(window_result)
        write_summary(output_dirs["date"] / SUMMARY_FILE_NAME, summary)

    success_count = sum(1 for window in summary["windows"] if window["status"] == "success")
    partial_count = len(summary["windows"]) - success_count
    print("\nCollection finished")
    print(f"Successful windows: {success_count}")
    print(f"Windows with errors: {partial_count}")
    print(f"Summary: {output_dirs['date'] / SUMMARY_FILE_NAME}")


if __name__ == "__main__":
    main()
