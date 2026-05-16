#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Common utilities and functions shared between normal_task.py and fault_task.py

This module contains all the shared functionality for data collection orchestration.
"""

import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests


def _create_locust_session() -> requests.Session:
    """Create a requests session that bypasses proxy env vars."""
    session = requests.Session()
    session.trust_env = False  # Do not read HTTP_PROXY / HTTPS_PROXY.
    session.headers.update({"User-Agent": "fault-task/1.0"})
    return session


# --- HTTP session for local services (Locust/UI etc.) ---
LOCUST_SESSION = _create_locust_session()

LOCUST_STATS_COLUMNS = [
    "avg_content_length",
    "avg_response_time",
    "current_fail_per_sec",
    "current_rps",
    "max_response_time",
    "median_response_time",
    "method",
    "min_response_time",
    "name",
    "num_failures",
    "num_requests",
    "response_time_percentile_0.95",
    "response_time_percentile_0.99",
    "total_fail_per_sec",
    "total_rps",
]
LOCUST_STATS_RETRIES = max(1, int(os.environ.get("LOCUST_STATS_RETRIES", "4")))
LOCUST_STATS_RETRY_BACKOFF_SECONDS = max(
    0.0, float(os.environ.get("LOCUST_STATS_RETRY_BACKOFF_SECONDS", "2.0"))
)


def log(message: str, exp_id: str, task_type: str = "TASK"):
    """Print timestamped log message"""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{timestamp}] [{task_type}][{exp_id}] {message}")


def create_output_directories(output_dirs: dict, exp_id: str, task_type: str):
    """Create all required output directories"""
    log("Creating output directory structure", exp_id, task_type)
    for name, path in output_dirs.items():
        path.mkdir(parents=True, exist_ok=True)
        log(f"  Created: {path}", exp_id, task_type)


def reset_locust_stats(locust_url: str, exp_id: str, task_type: str):
    """Reset Locust statistics before starting new test"""
    log("Resetting Locust statistics", exp_id, task_type)
    try:
        log(f"{locust_url}/stats/reset", exp_id, task_type)
        response = LOCUST_SESSION.get(f"{locust_url}/stats/reset", timeout=10)
        if response.status_code == 200:
            log("  Locust stats reset successful", exp_id, task_type)
        else:
            log(f"  WARN: Reset returned status {response.status_code}", exp_id, task_type)
    except Exception as e:
        log(f"  WARN: Reset failed (maybe first run): {e}", exp_id, task_type)


def start_locust_swarm(
    locust_url: str,
    users: int,
    spawn_rate: float,
    target_host: str,
    run_time: str,
    exp_id: str,
    task_type: str
) -> bool:
    """Start Locust load testing swarm"""
    log(f"Starting Locust swarm: users={users}, spawn_rate={spawn_rate}, "
        f"host={target_host}, run_time={run_time}", exp_id, task_type)

    data = {
        "user_count": users,
        "spawn_rate": spawn_rate,
        "host": target_host,
        "run_time": run_time,
    }

    try:
        response = LOCUST_SESSION.post(
            f"{locust_url}/swarm",
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=10
        )
        response.raise_for_status()
        log(f"  Swarm started successfully. Locust will run for ~{run_time}", exp_id, task_type)
        return True
    except Exception as e:
        log(f"  ERROR: Failed to start swarm: {e}", exp_id, task_type)
        return False


def collect_locust_stats(locust_url: str, output_dirs: dict, exp_id: str, task_type: str) -> bool:
    """Collect Locust statistics and save to CSV"""
    log("Collecting Locust statistics", exp_id, task_type)

    try:
        stats_data = _fetch_locust_stats_with_retries(locust_url, exp_id, task_type)
        _write_locust_stats_files(stats_data, output_dirs, exp_id, task_type)
        return True
    except Exception as e:
        log(f"  ERROR: Failed to collect Locust stats: {e}", exp_id, task_type)
        _write_locust_stats_failure_artifacts(output_dirs, locust_url, str(e), exp_id, task_type)
        return False


def _fetch_locust_stats_with_retries(locust_url: str, exp_id: str, task_type: str) -> dict:
    """Fetch Locust stats with retries using a fresh session to avoid stale keep-alive sockets."""
    last_error = None
    stats_url = f"{locust_url}/stats/requests"

    for attempt in range(1, LOCUST_STATS_RETRIES + 1):
        try:
            with _create_locust_session() as session:
                response = session.get(
                    stats_url,
                    timeout=(5, 30),
                    headers={"Connection": "close"},
                )
            response.raise_for_status()
            stats_data = response.json()
            if not isinstance(stats_data, dict):
                raise ValueError(f"Unexpected Locust stats payload type: {type(stats_data).__name__}")
            return stats_data
        except Exception as exc:
            last_error = exc
            if attempt >= LOCUST_STATS_RETRIES:
                break
            sleep_seconds = LOCUST_STATS_RETRY_BACKOFF_SECONDS * attempt
            log(
                f"  WARN: Locust stats attempt {attempt}/{LOCUST_STATS_RETRIES} failed: {exc}. "
                f"Retrying in {sleep_seconds:.1f}s",
                exp_id,
                task_type,
            )
            time.sleep(sleep_seconds)

    raise RuntimeError(
        f"Unable to fetch {stats_url} after {LOCUST_STATS_RETRIES} attempts"
    ) from last_error


def _write_locust_stats_files(stats_data: dict, output_dirs: dict, exp_id: str, task_type: str) -> None:
    """Write Locust stats JSON and CSV outputs."""
    json_path = output_dirs["workload"] / "locust_stats.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stats_data, f, indent=2)
    log(f"  Saved raw stats to: {json_path}", exp_id, task_type)

    csv_path = output_dirs["workload"] / "locust_stats.csv"
    stats_rows = stats_data.get("stats", [])
    if not isinstance(stats_rows, list):
        log("  WARN: Locust response field 'stats' is not a list; writing empty CSV", exp_id, task_type)
        stats_rows = []

    if stats_rows:
        df = pd.DataFrame(stats_rows)
        df.to_csv(csv_path, index=False)
        log(f"  Saved {len(df)} rows to: {csv_path}", exp_id, task_type)
    else:
        pd.DataFrame(columns=LOCUST_STATS_COLUMNS).to_csv(csv_path, index=False)
        log(f"  WARN: Locust response contains no stats rows. Wrote empty CSV to: {csv_path}", exp_id, task_type)


def _write_locust_stats_failure_artifacts(
    output_dirs: dict,
    locust_url: str,
    error_message: str,
    exp_id: str,
    task_type: str,
) -> None:
    """Write placeholder Locust stats artifacts so downstream steps still have files to read."""
    json_path = output_dirs["workload"] / "locust_stats.json"
    csv_path = output_dirs["workload"] / "locust_stats.csv"
    payload = {
        "current_response_time_percentiles": {
            "response_time_percentile_0.5": None,
            "response_time_percentile_0.95": None,
        },
        "errors": [],
        "fail_ratio": None,
        "state": "unavailable",
        "stats": [],
        "collection_error": error_message,
        "collection_endpoint": f"{locust_url}/stats/requests",
        "collected_at": datetime.now(timezone.utc).isoformat(),
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    pd.DataFrame(columns=LOCUST_STATS_COLUMNS).to_csv(csv_path, index=False)
    log(f"  Wrote diagnostic Locust JSON to: {json_path}", exp_id, task_type)
    log(f"  Wrote empty Locust CSV to: {csv_path}", exp_id, task_type)


def run_script(script_path: Path, script_name: str, exp_id: str, task_type: str) -> bool:
    """Run a Python script and wait for completion"""
    log(f"Running {script_name}: {script_path}", exp_id, task_type)

    if not script_path.exists():
        log(f"  ERROR: Script not found: {script_path}", exp_id, task_type)
        return False

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=script_path.parent,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )

        if result.returncode == 0:
            log(f"  {script_name} completed successfully", exp_id, task_type)
            if result.stdout:
                for line in result.stdout.strip().split('\n'):
                    log(f"    {line}", exp_id, task_type)
            return True
        else:
            log(f"  ERROR: {script_name} failed with return code {result.returncode}", exp_id, task_type)
            if result.stderr:
                log(f"  STDERR: {result.stderr}", exp_id, task_type)
            return False

    except subprocess.TimeoutExpired:
        log(f"  ERROR: {script_name} timed out after 10 minutes", exp_id, task_type)
        return False
    except Exception as e:
        log(f"  ERROR: Failed to run {script_name}: {e}", exp_id, task_type)
        return False


def move_generated_files(
    source_dir: Path,
    dest_dir: Path,
    pattern: str,
    exp_id: str,
    task_type: str
) -> int:
    """Move files matching pattern from source to destination directory"""
    moved_count = 0

    for file_path in source_dir.glob(pattern):
        if file_path.is_file():
            dest_path = dest_dir / file_path.name
            shutil.move(str(file_path), str(dest_path))
            log(f"  Moved: {file_path.name} -> {dest_path}", exp_id, task_type)
            moved_count += 1

    return moved_count


def collect_kubernetes_events(scripts: dict, output_dirs: dict, exp_id: str, task_type: str) -> bool:
    """Collect Kubernetes events"""
    log("=== Step 1: Collecting Kubernetes Events ===", exp_id, task_type)

    if not run_script(scripts["events"], "export_k8s_events.py", exp_id, task_type):
        return False

    # Move generated CSV
    source_dir = scripts["events"].parent
    moved = move_generated_files(source_dir, output_dirs["events"], "kubernetes_events.csv", exp_id, task_type)

    if moved > 0:
        log(f"  Successfully collected {moved} events file(s)", exp_id, task_type)
        return True
    else:
        log("  WARN: No kubernetes_events.csv file found", exp_id, task_type)
        return False


def collect_container_restarts(scripts: dict, output_dirs: dict, exp_id: str, task_type: str) -> bool:
    """Collect container restart information"""
    log("=== Collecting Container Restarts ===", exp_id, task_type)

    if not run_script(scripts["container_restarts"], "export_container_restarts.py", exp_id, task_type):
        return False

    # Move generated CSV
    source_dir = scripts["container_restarts"].parent
    moved = move_generated_files(source_dir, output_dirs["events"], "container_restarts.csv", exp_id, task_type)

    if moved > 0:
        log(f"  Successfully collected {moved} container restarts file(s)", exp_id, task_type)
        return True
    else:
        log("  WARN: No container_restarts.csv file found", exp_id, task_type)
        return False


def collect_logs(scripts: dict, output_dirs: dict, exp_id: str, task_type: str) -> bool:
    """Collect logs from Loki"""
    log("=== Step 2: Collecting Logs from Loki ===", exp_id, task_type)

    if not run_script(scripts["logs"], "loki_script.py", exp_id, task_type):
        return False

    # Move generated CSV files
    source_dir = scripts["logs"].parent
    moved = move_generated_files(source_dir, output_dirs["logs"], "loki_logs_*.csv", exp_id, task_type)

    if moved > 0:
        log(f"  Successfully collected {moved} log file(s)", exp_id, task_type)
        return True
    else:
        log("  WARN: No loki_logs_*.csv files found", exp_id, task_type)
        return False


def collect_node_metrics(scripts: dict, output_dirs: dict, exp_id: str, task_type: str) -> bool:
    """Collect node metrics from Prometheus"""
    log("=== Step 3: Collecting Node Metrics ===", exp_id, task_type)

    if not run_script(scripts["node_metrics"], "prometheus_node_script.py", exp_id, task_type):
        return False

    # Move generated CSV
    source_dir = scripts["node_metrics"].parent
    moved = move_generated_files(source_dir, output_dirs["metrics"], "prometheus_node_metrics.csv", exp_id, task_type)

    if moved > 0:
        log(f"  Successfully collected {moved} node metrics file(s)", exp_id, task_type)
        return True
    else:
        log("  WARN: No prometheus_node_metrics.csv file found", exp_id, task_type)
        return False


def collect_pod_metrics(scripts: dict, output_dirs: dict, exp_id: str, task_type: str) -> bool:
    """Collect pod-specific metrics from Prometheus"""
    log("=== Step 4: Collecting Pod-Specific Metrics ===", exp_id, task_type)

    if not run_script(scripts["pod_metrics"], "prometheus_pod_specific_script.py", exp_id, task_type):
        return False

    # Move generated CSV files
    source_dir = scripts["pod_metrics"].parent
    moved = move_generated_files(source_dir, output_dirs["metrics"], "prometheus_pod_metrics_*.csv", exp_id, task_type)

    if moved > 0:
        log(f"  Successfully collected {moved} pod metrics file(s)", exp_id, task_type)
        return True
    else:
        log("  WARN: No prometheus_pod_metrics_*.csv files found", exp_id, task_type)
        return False


def collect_traces(scripts: dict, output_dirs: dict, exp_id: str, task_type: str) -> bool:
    """Collect distributed traces from Jaeger"""
    log("=== Step 5: Collecting Distributed Traces ===", exp_id, task_type)

    trace_script = scripts["traces"]
    if not run_script(trace_script, trace_script.name, exp_id, task_type):
        return False

    # Move generated CSV files
    source_dir = trace_script.parent
    moved = move_generated_files(source_dir, output_dirs["traces"], "jaeger_traces_*.csv", exp_id, task_type)

    if moved > 0:
        log(f"  Successfully collected {moved} trace file(s)", exp_id, task_type)
        return True
    else:
        log("  WARN: No jaeger_traces_*.csv files found", exp_id, task_type)
        return False


def collect_all_data_parallel(scripts: dict, output_dirs: dict, exp_id: str, task_type: str) -> dict:
    """
    Collect all data from various sources simultaneously to minimize time skew.

    This function runs all data collection tasks in parallel using ThreadPoolExecutor,
    ensuring that data from different sources is collected at approximately the same time.

    Returns:
        dict: A dictionary with task names as keys and boolean success status as values
    """
    log("=== Collecting All Data Sources in Parallel ===", exp_id, task_type)

    # Define all collection tasks
    tasks = {
        "events": (collect_kubernetes_events, scripts, output_dirs, exp_id, task_type),
        "container_restarts": (collect_container_restarts, scripts, output_dirs, exp_id, task_type),
        "logs": (collect_logs, scripts, output_dirs, exp_id, task_type),
        "node_metrics": (collect_node_metrics, scripts, output_dirs, exp_id, task_type),
        "pod_metrics": (collect_pod_metrics, scripts, output_dirs, exp_id, task_type),
        "traces": (collect_traces, scripts, output_dirs, exp_id, task_type),
    }

    results = {}

    # Execute all tasks in parallel
    with ThreadPoolExecutor(max_workers=6) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(func, *args): task_name
            for task_name, (func, *args) in tasks.items()
        }

        # Wait for all tasks to complete
        for future in as_completed(future_to_task):
            task_name = future_to_task[future]
            try:
                result = future.result()
                results[task_name] = result
                status = "✅ SUCCESS" if result else "❌ FAILED"
                log(f"  {task_name}: {status}", exp_id, task_type)
            except Exception as e:
                results[task_name] = False
                log(f"  {task_name}: ❌ EXCEPTION: {e}", exp_id, task_type)

    # Summary
    successful = sum(1 for success in results.values() if success)
    total = len(results)
    log(f"Parallel collection complete: {successful}/{total} tasks succeeded", exp_id, task_type)

    return results
