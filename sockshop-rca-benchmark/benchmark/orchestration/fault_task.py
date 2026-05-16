#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fault Task Orchestrator for Chaos Experiment Data Collection

This script orchestrates the complete data collection workflow for a fault injection run:
1. Trigger Locust load testing
2. Wait for a random time (0-6 minutes), then inject chaos experiment
3. Wait for remaining time until 13 minutes total
4. Collect Locust statistics
5. Collect Kubernetes events
6. Collect logs from Loki
7. Collect node metrics from Prometheus
8. Collect pod-specific metrics from Prometheus
9. Collect distributed traces from Jaeger

All collected data is organized into the dataset directory structure.
"""

import os
import sys
import time
import json
import random
import re
import subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path


import yaml  # PyYAML


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
EXP_ID = os.environ.get("EXP_ID", "pod_do_fault_orders_004")

# Chaos tool selection: "litmus" or "chaosmesh"
CHAOS_TOOL = os.environ.get("CHAOS_TOOL", "chaosmesh").lower()

# Litmus configuration
LITMUS_EXPERIMENT_ID = os.environ.get("LITMUS_EXPERIMENT_ID", "93393cab-03de-4bf6-97ca-b9ed3f58036f")

# ChaosMesh configuration - YAML file is EXP_ID + ".yaml"
SCRIPT_DIR = Path(__file__).parent.parent
CHAOSMESH_YAML_DIR = SCRIPT_DIR / "chaosmesh_yaml"
CHAOSMESH_YAML_FILE = os.environ.get("CHAOSMESH_YAML_FILE", str(CHAOSMESH_YAML_DIR / f"{EXP_ID}.yaml"))

WAIT_TIME_MINUTES = 13  # Total wait time after starting Locust before collecting data
CHAOS_INJECTION_WINDOW_MINUTES = 7  # Inject chaos randomly within first 7 minutes
CHAOS_INJECTION_STARTING_MINUTES = 2  # Start chaos injection at least 1 minute after Locust starts
FAULT_WINDOW_MINUTES = 15  # Fault observation window for metadata (fault_start to fault_end)

# Base directories
DATASET_BASE = SCRIPT_DIR / "dataset" / "fault_run" / EXP_ID

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
    "litmus_chaos": SCRIPT_DIR / "litmus_script" / "litmus_experiment.py",
    "chaosmesh_chaos": SCRIPT_DIR / "chaosmesh_script" / "chaosmesh_experiment.py",
    "events": SCRIPT_DIR / "event_script" / "export_k8s_events.py",
    "container_restarts": SCRIPT_DIR / "event_script" / "export_container_restarts.py",
    "logs": SCRIPT_DIR / "logs_script" / "loki_script.py",
    "node_metrics": SCRIPT_DIR / "metrics_script" / "prometheus_node_script.py",
    "pod_metrics": SCRIPT_DIR / "metrics_script" / "prometheus_pod_specific_script.py",
    "traces": SCRIPT_DIR / "traces_script" / "jaeger_script.py",
}

# Task type for logging
TASK_TYPE = "FAULT_TASK"


# Global variable to store chaos injection time (for ChaosMesh metadata)
CHAOS_INJECTION_TIME = None


def inject_chaos_experiment():
    """Inject chaos experiment using either Litmus or ChaosMesh based on CHAOS_TOOL setting."""
    global CHAOS_INJECTION_TIME

    if CHAOS_TOOL == "litmus":
        return _inject_litmus_experiment()
    elif CHAOS_TOOL == "chaosmesh":
        return _inject_chaosmesh_experiment()
    else:
        log(f"ERROR: Unknown CHAOS_TOOL: {CHAOS_TOOL}. Use 'litmus' or 'chaosmesh'.", EXP_ID, TASK_TYPE)
        return False


def _inject_litmus_experiment():
    """Inject chaos experiment using Litmus."""
    global CHAOS_INJECTION_TIME

    if not LITMUS_EXPERIMENT_ID:
        log("WARN: LITMUS_EXPERIMENT_ID not set. Skipping chaos injection.", EXP_ID, TASK_TYPE)
        return False

    log(f"=== Injecting Litmus Chaos Experiment: {LITMUS_EXPERIMENT_ID} ===", EXP_ID, TASK_TYPE)

    chaos_script = SCRIPTS["litmus_chaos"]
    if not chaos_script.exists():
        log(f"  ERROR: Litmus chaos script not found: {chaos_script}", EXP_ID, TASK_TYPE)
        return False

    try:
        # Set environment variable for the chaos script
        env = os.environ.copy()
        env["LITMUS_EXPERIMENT_ID"] = LITMUS_EXPERIMENT_ID

        # Record injection time
        CHAOS_INJECTION_TIME = datetime.now(timezone.utc)

        result = subprocess.run(
            [sys.executable, str(chaos_script)],
            cwd=chaos_script.parent,
            env=env,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        if result.returncode == 0:
            log(f"  Litmus chaos experiment injected successfully", EXP_ID, TASK_TYPE)
            if result.stdout:
                for line in result.stdout.strip().split('\n'):
                    log(f"    {line}", EXP_ID, TASK_TYPE)
            return True
        else:
            log(f"  ERROR: Litmus chaos injection failed with return code {result.returncode}", EXP_ID, TASK_TYPE)
            if result.stderr:
                log(f"  STDERR: {result.stderr}", EXP_ID, TASK_TYPE)
            return False

    except subprocess.TimeoutExpired:
        log(f"  ERROR: Litmus chaos injection timed out after 5 minutes", EXP_ID, TASK_TYPE)
        return False
    except Exception as e:
        log(f"  ERROR: Failed to inject Litmus chaos: {e}", EXP_ID, TASK_TYPE)
        return False


def _inject_chaosmesh_experiment():
    """Inject chaos experiment using ChaosMesh."""
    global CHAOS_INJECTION_TIME

    yaml_file = Path(CHAOSMESH_YAML_FILE)
    if not yaml_file.exists():
        log(f"WARN: ChaosMesh YAML file not found: {yaml_file}. Skipping chaos injection.", EXP_ID, TASK_TYPE)
        return False

    log(f"=== Injecting ChaosMesh Experiment: {yaml_file.name} ===", EXP_ID, TASK_TYPE)
    log(f"  ChaosMesh YAML path: {yaml_file}", EXP_ID, TASK_TYPE)

    chaos_script = SCRIPTS["chaosmesh_chaos"]
    if not chaos_script.exists():
        log(f"  ERROR: ChaosMesh chaos script not found: {chaos_script}", EXP_ID, TASK_TYPE)
        return False

    try:
        # Set environment variable for the chaos script
        env = os.environ.copy()
        env["CHAOSMESH_YAML_FILE"] = str(yaml_file)

        # Record injection time
        CHAOS_INJECTION_TIME = datetime.now(timezone.utc)

        result = subprocess.run(
            [sys.executable, str(chaos_script)],
            cwd=chaos_script.parent,
            env=env,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        if result.returncode == 0:
            log(f"  ChaosMesh experiment injected successfully", EXP_ID, TASK_TYPE)
            if result.stdout:
                for line in result.stdout.strip().split('\n'):
                    log(f"    {line}", EXP_ID, TASK_TYPE)
            return True
        else:
            log(f"  ERROR: ChaosMesh injection failed with return code {result.returncode}", EXP_ID, TASK_TYPE)
            if result.stdout:
                log("  STDOUT:", EXP_ID, TASK_TYPE)
                for line in result.stdout.strip().split('\n'):
                    log(f"    {line}", EXP_ID, TASK_TYPE)
            if result.stderr:
                log("  STDERR:", EXP_ID, TASK_TYPE)
                for line in result.stderr.strip().split('\n'):
                    log(f"    {line}", EXP_ID, TASK_TYPE)
            return False

    except subprocess.TimeoutExpired as e:
        log(f"  ERROR: ChaosMesh injection timed out after 5 minutes", EXP_ID, TASK_TYPE)
        if e.stdout:
            log("  STDOUT(before timeout):", EXP_ID, TASK_TYPE)
            out = e.stdout.decode() if isinstance(e.stdout, bytes) else e.stdout
            for line in str(out).strip().split('\n'):
                if line:
                    log(f"    {line}", EXP_ID, TASK_TYPE)
        if e.stderr:
            log("  STDERR(before timeout):", EXP_ID, TASK_TYPE)
            err = e.stderr.decode() if isinstance(e.stderr, bytes) else e.stderr
            for line in str(err).strip().split('\n'):
                if line:
                    log(f"    {line}", EXP_ID, TASK_TYPE)
        return False
    except Exception as e:
        log(f"  ERROR: Failed to inject ChaosMesh chaos: {e}", EXP_ID, TASK_TYPE)
        return False


def wait_with_chaos_injection():
    """Wait for load test with chaos injection at random time within first 6 minutes"""
    log(f"Starting {WAIT_TIME_MINUTES}-minute wait period with chaos injection", EXP_ID, TASK_TYPE)

    # Randomly select injection time within first 6 minutes
    chaos_injection_seconds = random.randint(CHAOS_INJECTION_STARTING_MINUTES * 60, CHAOS_INJECTION_WINDOW_MINUTES * 60)
    chaos_injection_minute = chaos_injection_seconds / 60

    log(f"  Chaos will be injected at {chaos_injection_minute:.2f} minutes", EXP_ID, TASK_TYPE)

    # Wait until chaos injection time
    if chaos_injection_seconds > 0:
        log(f"  Waiting {chaos_injection_minute:.2f} minutes before chaos injection...", EXP_ID, TASK_TYPE)
        for second in range(chaos_injection_seconds):
            if second > 0 and second % 60 == 0:
                elapsed_minutes = second // 60
                log(f"    Elapsed: {elapsed_minutes}/{WAIT_TIME_MINUTES} minutes (pre-chaos)", EXP_ID, TASK_TYPE)
            time.sleep(1)

    # Inject chaos
    inject_chaos_experiment()

    # Wait for remaining time
    remaining_seconds = (WAIT_TIME_MINUTES * 60) - chaos_injection_seconds
    remaining_minutes = remaining_seconds / 60
    log(f"  Waiting additional {remaining_minutes:.2f} minutes after chaos injection...", EXP_ID, TASK_TYPE)

    for second in range(remaining_seconds):
        if second > 0 and second % 60 == 0:
            total_elapsed_minutes = (chaos_injection_seconds + second) // 60
            log(f"    Elapsed: {total_elapsed_minutes}/{WAIT_TIME_MINUTES} minutes (post-chaos)", EXP_ID, TASK_TYPE)
        time.sleep(1)

    log(f"Wait complete. Total time: {WAIT_TIME_MINUTES} minutes", EXP_ID, TASK_TYPE)


def generate_fault_metadata(end_time: datetime):
    """
    Generate fault_metadata.json with fault information.

    For Litmus: fetches experiment details from Litmus API.
    For ChaosMesh: reads experiment details from YAML file.

    Args:
        end_time: The end time of the script execution
    """
    log("=== Generating Fault Metadata ===", EXP_ID, TASK_TYPE)

    metadata_path = DATASET_BASE / "fault_metadata.json"
    metadata = None

    if CHAOS_TOOL == "litmus":
        metadata = _generate_litmus_metadata()
    elif CHAOS_TOOL == "chaosmesh":
        metadata = _generate_chaosmesh_metadata()

    # Fallback to basic metadata generation if specific tool metadata failed
    if metadata is None:
        log("  Using fallback metadata generation", EXP_ID, TASK_TYPE)
        metadata = _generate_fallback_metadata(end_time)

    # Write metadata to file
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    log(f"Fault metadata saved to: {metadata_path}", EXP_ID, TASK_TYPE)
    log(f"  Target service: {metadata.get('target_service', 'unknown')}", EXP_ID, TASK_TYPE)
    if "injection_info" in metadata:
        log(f"  Fault type: {metadata['injection_info'].get('fault_type', 'unknown')}", EXP_ID, TASK_TYPE)


def _generate_litmus_metadata() -> dict:
    """Generate metadata by fetching from Litmus API."""
    if not LITMUS_EXPERIMENT_ID:
        return None

    try:
        # Import Litmus API client
        sys.path.insert(0, str(SCRIPT_DIR / "litmus_script"))
        from litmus_api import get_latest_experiment_info, format_fault_metadata

        log(f"Fetching experiment info from Litmus API for: {LITMUS_EXPERIMENT_ID}", EXP_ID, TASK_TYPE)

        # Get experiment info from Litmus
        experiment_info = get_latest_experiment_info(LITMUS_EXPERIMENT_ID)

        if "error" not in experiment_info:
            # Format the metadata
            metadata = format_fault_metadata(experiment_info, EXP_ID)
            log("  Successfully retrieved experiment info from Litmus", EXP_ID, TASK_TYPE)
            return metadata
        else:
            log(f"  WARN: {experiment_info.get('error', 'Unknown error')}", EXP_ID, TASK_TYPE)
            return None

    except ImportError as e:
        log(f"  WARN: Could not import litmus_api module: {e}", EXP_ID, TASK_TYPE)
        return None
    except Exception as e:
        log(f"  WARN: Failed to fetch from Litmus API: {e}", EXP_ID, TASK_TYPE)
        return None


def _generate_chaosmesh_metadata() -> dict:
    """Generate metadata by reading from ChaosMesh YAML file."""
    yaml_file = Path(CHAOSMESH_YAML_FILE)
    if not yaml_file.exists():
        log(f"  WARN: ChaosMesh YAML file not found: {yaml_file}", EXP_ID, TASK_TYPE)
        return None

    if yaml is None:
        log(
            "  WARN: PyYAML is not installed in current environment. "
            "Skipping ChaosMesh YAML parsing and using fallback metadata.",
            EXP_ID,
            TASK_TYPE,
        )
        return None

    try:
        log(f"Reading ChaosMesh experiment info from: {yaml_file.name}", EXP_ID, TASK_TYPE)

        with open(yaml_file, 'r') as f:
            yaml_content = yaml.safe_load(f)

        # Extract basic info
        kind = yaml_content.get("kind", "")  # e.g., "HTTPChaos", "PodChaos", "StressChaos"
        spec = yaml_content.get("spec", {})

        # Extract target service from selector
        selector = spec.get("selector", {})
        label_selectors = selector.get("labelSelectors", {})
        target_service = label_selectors.get("name", "") or label_selectors.get("app", "")

        # If not found in labelSelectors, try to extract from EXP_ID
        if not target_service:
            target_service = _extract_target_service_from_exp_id()

        # Build injection_info based on chaos type
        injection_info = {
            "fault_type": kind.lower(),  # e.g., "httpchaos", "podchaos"
        }

        # Add inject_start time (when chaos was injected)
        if CHAOS_INJECTION_TIME:
            injection_info["inject_start"] = CHAOS_INJECTION_TIME.strftime("%Y-%m-%dT%H:%M:%SZ")

        # Extract parameters based on chaos kind
        # Duration
        if "duration" in spec:
            injection_info["duration"] = spec["duration"]

        # HTTPChaos specific
        if kind == "HTTPChaos":
            if "delay" in spec:
                injection_info["delay"] = spec["delay"]
            if "port" in spec:
                injection_info["port"] = spec["port"]
            if "path" in spec:
                injection_info["path"] = spec["path"]
            if "method" in spec:
                injection_info["method"] = spec["method"]
            if "target" in spec:
                injection_info["target"] = spec["target"]

        # StressChaos specific (CPU/Memory stress)
        if kind == "StressChaos":
            stressors = spec.get("stressors", {})
            cpu_stressor = stressors.get("cpu", {})
            memory_stressor = stressors.get("memory", {})

            if cpu_stressor:
                if "workers" in cpu_stressor:
                    injection_info["cpu_workers"] = cpu_stressor["workers"]
                if "load" in cpu_stressor:
                    injection_info["cpu_load"] = cpu_stressor["load"]

            if memory_stressor:
                if "workers" in memory_stressor:
                    injection_info["memory_workers"] = memory_stressor["workers"]
                if "size" in memory_stressor:
                    injection_info["memory_size"] = memory_stressor["size"]

        # NetworkChaos specific
        if kind == "NetworkChaos":
            if "action" in spec:
                injection_info["action"] = spec["action"]
            if "delay" in spec:
                delay_spec = spec.get("delay", {})
                if isinstance(delay_spec, dict):
                    injection_info["latency"] = delay_spec.get("latency", "")
                    injection_info["jitter"] = delay_spec.get("jitter", "")
                else:
                    injection_info["delay"] = delay_spec
            if "loss" in spec:
                loss_spec = spec.get("loss", {})
                if isinstance(loss_spec, dict):
                    injection_info["loss"] = loss_spec.get("loss", "")
                else:
                    injection_info["loss"] = loss_spec

        # PodChaos specific
        if kind == "PodChaos":
            if "action" in spec:
                injection_info["action"] = spec["action"]

        # JVMChaos specific
        if kind == "JVMChaos":
            if "action" in spec:
                injection_info["action"] = spec["action"]
            if "value" in spec:
                injection_info["value"] = spec["value"]

        # IOChaos specific
        if kind == "IOChaos":
            if "action" in spec:
                injection_info["action"] = spec["action"]
            if "delay" in spec:
                injection_info["delay"] = spec["delay"]
            if "errno" in spec:
                injection_info["errno"] = spec["errno"]

        # Mode (all, one, fixed, etc.)
        if "mode" in spec:
            injection_info["mode"] = spec["mode"]

        # direction (from, to, both)
        if "direction" in spec:
            injection_info["direction"] = spec["direction"]

        # percent (for loss, corruption, etc.)
        if "percent" in spec:
            injection_info["percent"] = spec["percent"]

        # errno (for IOChaos)
        if "errno" in spec:
            injection_info["errno"] = spec["errno"]

        log("  Successfully read experiment info from ChaosMesh YAML", EXP_ID, TASK_TYPE)

        return {
            "fault_id": EXP_ID,
            "target_service": target_service,
            "injection_tool": "chaosmesh",
            "injection_info": injection_info,
        }

    except Exception as e:
        log(f"  WARN: Failed to read ChaosMesh YAML: {e}", EXP_ID, TASK_TYPE)
        return None


def _extract_target_service_from_exp_id() -> str:
    """Extract target service name from EXP_ID."""
    # Known services
    known_services = ["carts", "catalogue", "user", "orders", "payment", "shipping",
                     "front-end", "queue-master", "rabbitmq", "session-db",
                     "carts-db", "user-db", "orders-db", "catalogue-db"]

    # Check if EXP_ID contains a known service
    for svc in sorted(known_services, key=len, reverse=True):
        if svc in EXP_ID:
            return svc

    # Fallback: try to parse from EXP_ID pattern
    parts = EXP_ID.replace("-", "_").split("_")
    if len(parts) >= 3:
        # Skip fault type prefix and numeric suffix
        target_service = '_'.join(parts[2:])
        target_service = re.sub(r'_\d+$', '', target_service)
        return target_service

    return "unknown"


def _generate_fallback_metadata(end_time: datetime) -> dict:
    """
    Generate basic fault metadata when Litmus API is not available.

    Args:
        end_time: The end time of the script execution

    Returns:
        dict: Basic fault metadata
    """
    # Calculate fault start time (FAULT_WINDOW_MINUTES before end)
    fault_start = end_time - timedelta(minutes=FAULT_WINDOW_MINUTES)

    # Known fault type prefixes (ordered by specificity - longer patterns first)
    KNOWN_FAULT_TYPES = [
        "pod_cpu_hog",
        "pod_memory_hog",
        "pod_network_loss",
        "pod_network_latency",
        "pod_network_corruption",
        "pod_io_stress",
        "pod_delete",
        "pod_container_kill",
        "container_kill",
        "node_cpu_hog",
        "node_memory_hog",
        "node_io_stress",
    ]

    fault_type = None
    target_service = None

    # Try to match known fault types
    for known_type in KNOWN_FAULT_TYPES:
        if EXP_ID.startswith(known_type + "_"):
            fault_type = known_type
            # Everything after the fault type prefix is target_service (with potential numeric suffix)
            remainder = EXP_ID[len(known_type) + 1:]  # +1 for the underscore
            # Strip numeric suffix (e.g., "user_001" -> "user", "front-end_001" -> "front-end")
            target_service = re.sub(r'_\d+$', '', remainder)
            break

    # Fallback parsing if no known fault type matched
    if fault_type is None:
        parts = EXP_ID.split('_')
        if len(parts) >= 3:
            fault_type = '_'.join(parts[:2])
            target_service = '_'.join(parts[2:])
            target_service = re.sub(r'_\d+$', '', target_service)
        else:
            fault_type = EXP_ID
            target_service = "unknown"

    return {
        "fault_id": EXP_ID,
        "target_service": target_service,
        "fault_start": fault_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "fault_end": end_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "injection_tool": CHAOS_TOOL,
        "injection_info": {
            "fault_type": fault_type,
            "inject_start": fault_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "inject_end": end_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "note": f"Generated from fallback - {CHAOS_TOOL} API/YAML not available"
        }
    }


def generate_summary():
    """Generate a summary report of collected data"""
    log("=== Generating Summary Report ===", EXP_ID, TASK_TYPE)

    summary_path = DATASET_BASE / "collection_summary.txt"

    with open(summary_path, "w") as f:
        f.write(f"Data Collection Summary (Fault Injection Run)\n")
        f.write(f"=" * 80 + "\n\n")
        f.write(f"Experiment ID: {EXP_ID}\n")
        f.write(f"Litmus Experiment ID: {LITMUS_EXPERIMENT_ID}\n")
        f.write(f"Collection Time: {datetime.now(timezone.utc).isoformat()}\n")
        f.write(f"Locust Configuration:\n")
        f.write(f"  - Users: {USERS}\n")
        f.write(f"  - Spawn Rate: {SPAWN_RATE}\n")
        f.write(f"  - Target Host: {TARGET_HOST}\n")
        f.write(f"  - Run Time: {RUN_TIME}\n")
        f.write(f"Chaos Configuration:\n")
        f.write(f"  - Total Wait Time: {WAIT_TIME_MINUTES} minutes\n")
        f.write(f"  - Chaos Injection Start Window: After {CHAOS_INJECTION_STARTING_MINUTES} minutes\n")
        f.write(f"  - Chaos Injection Window: First {CHAOS_INJECTION_WINDOW_MINUTES} minutes\n")
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
    log("Starting Fault Task Data Collection", EXP_ID, TASK_TYPE)
    log("=" * 80, EXP_ID, TASK_TYPE)
    log(f"Chaos Tool: {CHAOS_TOOL.upper()}", EXP_ID, TASK_TYPE)

    try:
        # Validate chaos tool configuration
        if CHAOS_TOOL == "litmus":
            if not LITMUS_EXPERIMENT_ID:
                log("WARNING: LITMUS_EXPERIMENT_ID is not set!", EXP_ID, TASK_TYPE)
                log("  Please set the LITMUS_EXPERIMENT_ID environment variable", EXP_ID, TASK_TYPE)
                log("  Example: export LITMUS_EXPERIMENT_ID='pod-cpu-hog-1234567890'", EXP_ID, TASK_TYPE)
                response = input("Continue without chaos injection? (y/N): ")
                if response.lower() != 'y':
                    log("Aborting.", EXP_ID, TASK_TYPE)
                    return 1
            else:
                log(f"Litmus Experiment ID: {LITMUS_EXPERIMENT_ID}", EXP_ID, TASK_TYPE)
        elif CHAOS_TOOL == "chaosmesh":
            yaml_file = Path(CHAOSMESH_YAML_FILE)
            if not yaml_file.exists():
                log(f"WARNING: ChaosMesh YAML file not found: {yaml_file}", EXP_ID, TASK_TYPE)
                log("  Please ensure the YAML file exists or set CHAOSMESH_YAML_FILE environment variable", EXP_ID, TASK_TYPE)
                response = input("Continue without chaos injection? (y/N): ")
                if response.lower() != 'y':
                    log("Aborting.", EXP_ID, TASK_TYPE)
                    return 1
            else:
                log(f"ChaosMesh YAML: {yaml_file}", EXP_ID, TASK_TYPE)
        else:
            log(f"ERROR: Unknown CHAOS_TOOL: {CHAOS_TOOL}. Use 'litmus' or 'chaosmesh'.", EXP_ID, TASK_TYPE)
            return 1

        # Step 0: Setup
        create_output_directories(OUTPUT_DIRS, EXP_ID, TASK_TYPE)

        # Step 1: Start Locust
        reset_locust_stats(LOCUST_URL, EXP_ID, TASK_TYPE)
        if not start_locust_swarm(LOCUST_URL, USERS, SPAWN_RATE, TARGET_HOST, RUN_TIME, EXP_ID, TASK_TYPE):
            log("ERROR: Failed to start Locust swarm. Aborting.", EXP_ID, TASK_TYPE)
            return 1

        # Step 2: Wait with chaos injection
        wait_with_chaos_injection()

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

        # Calculate duration
        end_time = datetime.now(timezone.utc)
        duration = (end_time - start_time).total_seconds() / 60

        # Step 5: Generate fault metadata
        generate_fault_metadata(end_time)

        # Step 6: Generate summary (optional)
        # generate_summary()

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
