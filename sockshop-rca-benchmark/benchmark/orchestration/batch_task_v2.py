#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch orchestrator for running fault_task_v2.py and normal_task_v2.py in sequence.

Configuration priority:
1) BATCH_PLAN_FILE: path to JSON file containing a list of steps
2) BATCH_PLAN_JSON: JSON string containing a list of steps
3) DEFAULT_PLAN (defined below)

Step format:
[
  {"task": "fault", "exp_id": "pod_cpu_hog_orders_001"},
  {"task": "fault", "exp_id": "pod_do_fault_orders_001"},
  {"task": "normal"},
  {"task": "fault", "exp_id": "pod_io_latency_carts-db_001"}
]

Optional fields per step:
- interval_seconds: override wait time after this step
- env: extra env vars for the called script, e.g. {"RUN_TIME": "12m"}

Environment variables:
- EXP_ID: default fault EXP_ID when a fault step omits exp_id
- BATCH_INTERVAL_SECONDS: default wait between steps (default 600)
- BATCH_STOP_ON_ERROR: stop batch when one step fails (default true)
- BATCH_PLAN_FILE / BATCH_PLAN_JSON: run plan source
- BATCH_STEP_TIMEOUT_SECONDS: timeout for each subprocess (0 = no timeout)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
FAULT_SCRIPT = SCRIPT_DIR / "fault_task_v2.py"
NORMAL_SCRIPT = SCRIPT_DIR / "normal_task_v2.py"

DEFAULT_FAULT_EXP_ID = os.environ.get("EXP_ID", "pod_cpu_hog_orders_001")
DEFAULT_INTERVAL_SECONDS = int(os.environ.get("BATCH_INTERVAL_SECONDS", "180"))
STOP_ON_ERROR = os.environ.get("BATCH_STOP_ON_ERROR", "true").strip().lower() not in {"0", "false", "no"}
STEP_TIMEOUT_SECONDS = int(os.environ.get("BATCH_STEP_TIMEOUT_SECONDS", "0"))
DEFAULT_PLAN_FILE = SCRIPT_DIR / "batch_task_data_v2.json"

def _load_default_plan() -> list[dict[str, Any]]:
    if not DEFAULT_PLAN_FILE.exists():
        raise FileNotFoundError(f"Default plan file not found: {DEFAULT_PLAN_FILE}")
    raw = json.loads(DEFAULT_PLAN_FILE.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Default plan must be a JSON list in {DEFAULT_PLAN_FILE}")
    return raw


DEFAULT_PLAN: list[dict[str, Any]] = _load_default_plan()


def log(message: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{ts}] [BATCH_TASK] {message}")


def _load_plan_from_env() -> list[dict[str, Any]]:
    plan_file = os.environ.get("BATCH_PLAN_FILE", "").strip()
    plan_json = os.environ.get("BATCH_PLAN_JSON", "").strip()

    if plan_file:
        path = Path(plan_file).expanduser()
        if not path.is_absolute():
            path = (SCRIPT_DIR / path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"BATCH_PLAN_FILE not found: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    if plan_json:
        return json.loads(plan_json)

    return DEFAULT_PLAN


def _validate_plan(plan: Any) -> list[dict[str, Any]]:
    if not isinstance(plan, list) or not plan:
        raise ValueError("Batch plan must be a non-empty JSON list.")

    normalized: list[dict[str, Any]] = []
    for i, raw in enumerate(plan, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"Plan step #{i} must be an object.")

        task = str(raw.get("task", "")).strip().lower()
        if task not in {"fault", "normal"}:
            raise ValueError(f"Plan step #{i} has invalid task: {task!r}. Use 'fault' or 'normal'.")

        step: dict[str, Any] = {"task": task}
        if task == "fault":
            step["exp_id"] = str(raw.get("exp_id") or DEFAULT_FAULT_EXP_ID).strip()
            if not step["exp_id"]:
                raise ValueError(f"Plan step #{i} fault task requires a non-empty exp_id.")
        elif "exp_id" in raw and raw.get("exp_id") is not None:
            step["exp_id"] = str(raw["exp_id"]).strip()

        if "interval_seconds" in raw and raw["interval_seconds"] is not None:
            interval_seconds = int(raw["interval_seconds"])
            if interval_seconds < 0:
                raise ValueError(f"Plan step #{i} interval_seconds must be >= 0.")
            step["interval_seconds"] = interval_seconds

        extra_env = raw.get("env")
        if extra_env is not None:
            if not isinstance(extra_env, dict):
                raise ValueError(f"Plan step #{i} env must be an object.")
            step["env"] = {str(k): str(v) for k, v in extra_env.items()}

        normalized.append(step)

    return normalized


def _run_step(step: dict[str, Any], index: int, total: int) -> int:
    task = step["task"]
    script_path = FAULT_SCRIPT if task == "fault" else NORMAL_SCRIPT
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    env = os.environ.copy()
    if task == "fault":
        env["EXP_ID"] = step["exp_id"]
        label = step["exp_id"]
    else:
        label = step.get("exp_id", "normal")
        if step.get("exp_id"):
            env["EXP_ID"] = step["exp_id"]

    for k, v in step.get("env", {}).items():
        env[k] = v

    log(f"Step {index}/{total}: run {task} ({label})")
    log(f"  command: {sys.executable} {script_path.name}")

    run_kwargs: dict[str, Any] = {
        "args": [sys.executable, str(script_path)],
        "cwd": str(SCRIPT_DIR),
        "env": env,
        "capture_output": True,
        "text": True,
    }
    if STEP_TIMEOUT_SECONDS > 0:
        run_kwargs["timeout"] = STEP_TIMEOUT_SECONDS

    try:
        result = subprocess.run(**run_kwargs)
    except subprocess.TimeoutExpired:
        log(f"  ERROR: step timed out after {STEP_TIMEOUT_SECONDS}s")
        return 124

    if result.stdout:
        for line in result.stdout.strip().splitlines():
            log(f"  stdout | {line}")
    if result.stderr:
        for line in result.stderr.strip().splitlines():
            log(f"  stderr | {line}")

    if result.returncode == 0:
        log("  step completed successfully")
    else:
        log(f"  ERROR: step failed with return code {result.returncode}")
    return result.returncode


def _sleep_between(step: dict[str, Any], index: int, total: int) -> None:
    if index >= total:
        return

    wait_seconds = int(step.get("interval_seconds", DEFAULT_INTERVAL_SECONDS))
    if wait_seconds <= 0:
        return

    log(f"Waiting {wait_seconds}s before next step...")
    time.sleep(wait_seconds)


def main() -> int:
    log("Starting batch task orchestrator")
    log(f"Default fault EXP_ID: {DEFAULT_FAULT_EXP_ID}")
    log(f"Default interval: {DEFAULT_INTERVAL_SECONDS}s")
    log(f"Stop on error: {STOP_ON_ERROR}")

    try:
        plan = _validate_plan(_load_plan_from_env())
    except Exception as exc:
        log(f"ERROR: invalid batch plan: {exc}")
        return 1

    log(f"Loaded {len(plan)} plan step(s)")

    failures: list[tuple[int, dict[str, Any], int]] = []
    total = len(plan)
    for idx, step in enumerate(plan, start=1):
        rc = _run_step(step, idx, total)
        if rc != 0:
            failures.append((idx, step, rc))
            if STOP_ON_ERROR:
                log("Stopping batch due to failure")
                break
        _sleep_between(step, idx, total)

    if failures:
        log("Batch finished with failures:")
        for idx, step, rc in failures:
            ident = step.get("exp_id", "normal")
            log(f"  step {idx}: task={step['task']} exp_id={ident} rc={rc}")
        return 1

    log("Batch finished successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
