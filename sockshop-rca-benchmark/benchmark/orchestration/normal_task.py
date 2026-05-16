#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Normal Task Orchestrator for Chaos Experiment Data Collection

This script orchestrates the complete data collection workflow for a normal (baseline) run:
1. Trigger Locust load testing
2. Wait for load test duration
3. Collect Locust statistics
4. Collect Kubernetes events
5. Collect logs from Loki
6. Collect node metrics from Prometheus
7. Collect pod-specific metrics from Prometheus
8. Collect distributed traces from Jaeger

All collected data is organized into the dataset directory structure.
"""

import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Import common utilities
from task_common import (
    log,
    create_output_directories,
    reset_locust_stats,
    start_locust_swarm,
    collect_locust_stats,
    collect_all_data_parallel,
)

# ===== Configuration =====
LOCUST_URL = os.environ.get("LOCUST_URL", "http://127.0.0.1:8089")
USERS = int(os.environ.get("USERS", "100"))
SPAWN_RATE = float(os.environ.get("SPAWN_RATE", "0.5"))
TARGET_HOST = os.environ.get("TARGET_HOST", "http://sockshop.local:31728")
RUN_TIME = os.environ.get("RUN_TIME", "12m")

# Experiment configuration
EXP_ID = os.environ.get("EXP_ID", f"normal011")
WAIT_TIME_MINUTES = 13  # Wait time after starting Locust before collecting data

# Base directories
SCRIPT_DIR = Path(__file__).parent.parent
DATASET_BASE = SCRIPT_DIR / "dataset" / "normal_run" / EXP_ID

# Output directories
OUTPUT_DIRS = {
    "workload": DATASET_BASE / "workload",
    "events": DATASET_BASE / "events",
    "logs": DATASET_BASE / "logs",
    "metrics": DATASET_BASE / "metrics",
    "traces": DATASET_BASE / "traces",
}

# Script paths
SCRIPTS = {
    "events": SCRIPT_DIR / "event_script" / "export_k8s_events.py",
    "container_restarts": SCRIPT_DIR / "event_script" / "export_container_restarts.py",
    "logs": SCRIPT_DIR / "logs_script" / "loki_script.py",
    "node_metrics": SCRIPT_DIR / "metrics_script" / "prometheus_node_script.py",
    "pod_metrics": SCRIPT_DIR / "metrics_script" / "prometheus_pod_specific_script.py",
    "traces": SCRIPT_DIR / "traces_script" / "jaeger_script_v2.py",
}

# Task type for logging
TASK_TYPE = "NORMAL_TASK"


def wait_for_load_test():
    """Wait for load test to run"""
    log(f"Waiting {WAIT_TIME_MINUTES} minutes for load test to run...", EXP_ID, TASK_TYPE)

    # Show progress every minute
    for minute in range(1, WAIT_TIME_MINUTES + 1):
        time.sleep(60)
        log(f"  Waited {minute}/{WAIT_TIME_MINUTES} minutes", EXP_ID, TASK_TYPE)

    log(f"Wait complete. Starting data collection.", EXP_ID, TASK_TYPE)


def generate_summary():
    """Generate a summary report of collected data"""
    log("=== Generating Summary Report ===", EXP_ID, TASK_TYPE)

    summary_path = DATASET_BASE / "collection_summary.txt"

    with open(summary_path, "w") as f:
        f.write(f"Data Collection Summary\n")
        f.write(f"=" * 80 + "\n\n")
        f.write(f"Experiment ID: {EXP_ID}\n")
        f.write(f"Collection Time: {datetime.now(timezone.utc).isoformat()}\n")
        f.write(f"Locust Configuration:\n")
        f.write(f"  - Users: {USERS}\n")
        f.write(f"  - Spawn Rate: {SPAWN_RATE}\n")
        f.write(f"  - Target Host: {TARGET_HOST}\n")
        f.write(f"  - Run Time: {RUN_TIME}\n")
        f.write(f"\n")
        f.write(f"Collected Files:\n")
        f.write(f"-" * 80 + "\n")

        for dir_name, dir_path in OUTPUT_DIRS.items():
            files = list(dir_path.glob("*.*"))
            f.write(f"\n{dir_name.upper()} ({len(files)} files):\n")
            for file_path in sorted(files):
                size_mb = file_path.stat().st_size / (1024 * 1024)
                f.write(f"  - {file_path.name} ({size_mb:.2f} MB)\n")

    log(f"Summary report saved to: {summary_path}", EXP_ID, TASK_TYPE)


def main():
    """Main orchestration function"""
    start_time = datetime.now(timezone.utc)
    log("=" * 80, EXP_ID, TASK_TYPE)
    log("Starting Normal Task Data Collection", EXP_ID, TASK_TYPE)
    log("=" * 80, EXP_ID, TASK_TYPE)

    try:
        # Step 0: Setup
        create_output_directories(OUTPUT_DIRS, EXP_ID, TASK_TYPE)

        # Step 1: Start Locust
        reset_locust_stats(LOCUST_URL, EXP_ID, TASK_TYPE)
        if not start_locust_swarm(LOCUST_URL, USERS, SPAWN_RATE, TARGET_HOST, RUN_TIME, EXP_ID, TASK_TYPE):
            log("ERROR: Failed to start Locust swarm. Aborting.", EXP_ID, TASK_TYPE)
            return 1

        # Step 2: Wait for load test
        wait_for_load_test()

        # Step 3: Collect Locust statistics
        locust_stats_ok = collect_locust_stats(LOCUST_URL, OUTPUT_DIRS, EXP_ID, TASK_TYPE)

        # Step 4: Collect data from various sources in parallel
        collection_results = collect_all_data_parallel(SCRIPTS, OUTPUT_DIRS, EXP_ID, TASK_TYPE)

        # Log collection results summary
        failed_tasks = [task for task, success in collection_results.items() if not success]
        if not locust_stats_ok:
            failed_tasks.append("locust_stats")
        if failed_tasks:
            log(f"WARNING: {len(failed_tasks)} task(s) failed: {', '.join(failed_tasks)}", EXP_ID, TASK_TYPE)
        else:
            log("All data collection tasks completed successfully!", EXP_ID, TASK_TYPE)

        # Step 5: Generate summary
        # generate_summary()

        # Calculate duration
        end_time = datetime.now(timezone.utc)
        duration = (end_time - start_time).total_seconds() / 60

        log("=" * 80, EXP_ID, TASK_TYPE)
        log(f"Data Collection Complete! Duration: {duration:.2f} minutes", EXP_ID, TASK_TYPE)
        log(f"Data saved to: {DATASET_BASE}", EXP_ID, TASK_TYPE)
        log("=" * 80, EXP_ID, TASK_TYPE)

        return 0

    except KeyboardInterrupt:
        log("Interrupted by user", EXP_ID, TASK_TYPE)
        return 130
    except Exception as e:
        log(f"FATAL ERROR: {e}", EXP_ID, TASK_TYPE)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
