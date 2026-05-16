#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fetch logs from Loki within a time range and save as CSV / Parquet for AIOps / RCA experiments.

Please modify before use:
  - LOKI_URL
  - LOG_QUERY
  - start / end
  - experiment_meta
"""

import requests
from requests.exceptions import ChunkedEncodingError, ConnectionError, Timeout
import pandas as pd
from datetime import datetime, timezone, timedelta
from pathlib import Path
import json
import time

# ========== 0. Loki & Query Configuration ==========
# If you're using NodePort, e.g., the newly created loki-stack-nodeport:
#   kubectl get svc -n logging | grep loki-stack-nodeport
# If you see 3100:31300/TCP, then:
#   LOKI_URL = "http://<any node external IP>:31300"
LOKI_URL = "http://34.28.33.102:31300"  # TODO: Modify according to your environment

# LogQL debugged in Grafana Explore
# Example: collect logs only from sock-shop namespace
LOG_QUERY = '{namespace="sock-shop"}'
# If you want only carts service, you can do:
# LOG_QUERY = '{namespace="sock-shop", app="carts"}'

# Maximum log entries per request (increase or use time slicing if logs are numerous)
LOKI_LIMIT = 5000

# Request direction: forward = time from start → end
LOKI_DIRECTION = "forward"

# ========== 1. Time Window Configuration ==========
# Recommended: align with your Prometheus / Jaeger experiment time

# Option A: Last N hours (relative to now)
USE_RELATIVE_WINDOW = True
RELATIVE_HOURS = 0.25  # e.g., last 15 minutes (0.25 hours)

# Option B: Fixed time period (UTC ISO8601), uncomment and set these when USE_RELATIVE_WINDOW = False
START_ISO = "2025-11-23T10:00:00+00:00"
END_ISO   = "2025-11-23T11:00:00+00:00"

def get_time_range():
    """Returns (start_dt, end_dt) based on configuration, both are timezone-aware UTC times"""
    if USE_RELATIVE_WINDOW:
        end = datetime.now(timezone.utc)
        start = end - timedelta(hours=RELATIVE_HOURS)
    else:
        start = datetime.fromisoformat(START_ISO)
        end = datetime.fromisoformat(END_ISO)
        # Ensure tz info exists (default to UTC if not)
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
    return start, end

def to_unix_ns(dt: datetime) -> int:
    """Loki requires Unix nanosecond timestamp"""
    return int(dt.timestamp() * 1_000_000_000)

# ========== 2. Experiment Metadata (aligned with metrics / traces) ==========
experiment_meta = {
    "experiment_id": "exp_001",
    "fault_type": "pod_kill",
    "target_service": "sock-shop-carts",
    "note": "kill carts pod once at t+10min",
}

# ========== 3. Loki HTTP Call Wrapper ==========

def loki_query_range(
    base_url: str,
    query: str,
    start_ns: int,
    end_ns: int,
    limit: int = 5000,
    direction: str = "forward",
    max_retries: int = 3,
):
    url = f"{base_url.rstrip('/')}/loki/api/v1/query_range"
    params = {
        "query": query,
        "start": str(start_ns),
        "end": str(end_ns),
        "limit": str(limit),
        "direction": direction,
    }
    print(f"[Loki] GET {url}")
    print(f"[Loki] params = {params}")

    last_error = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=120)

            # Debug output added here
            if resp.status_code != 200:
                print("[Loki] status_code:", resp.status_code)
                print("[Loki] response body:", resp.text)
                resp.raise_for_status()

            data = resp.json()
            if data.get("status") != "success":
                raise RuntimeError(f"Loki error: {data}")

            result = data.get("data", {}).get("result", [])
            return result

        except (ChunkedEncodingError, ConnectionError, Timeout) as e:
            last_error = e
            wait_time = 2 ** attempt  # Exponential backoff: 1, 2, 4 seconds
            print(f"[Loki] Attempt {attempt + 1}/{max_retries} failed: {type(e).__name__}: {e}")
            if attempt < max_retries - 1:
                print(f"[Loki] Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"[Loki] All {max_retries} attempts failed")

    # If all retries failed, raise the last error
    raise last_error

# ========== 4. Log Fetching + Flattening ==========

def fetch_logs_to_rows():
    """
    Fetch logs from Loki and flatten into rows (list[dict]).
    One row corresponds to one log entry in Loki.

    Uses "time slicing" approach to avoid triggering max_entries_limit in a single query.
    Returns a dictionary grouped by pod: {pod_name: [rows]}
    """
    start_dt, end_dt = get_time_range()

    # Length of each time slice (adjust as needed)
    slice_minutes = 5
    slice_delta = timedelta(minutes=slice_minutes)

    print(f"[INFO] Fetching logs from Loki: {LOKI_URL}")
    print(f"[INFO]   query       = {LOG_QUERY}")
    print(f"[INFO]   time range  = {start_dt.isoformat()}  ~  {end_dt.isoformat()}")
    print(f"[INFO]   slice       = {slice_minutes} minutes per query, limit={LOKI_LIMIT}")

    # Store logs grouped by pod
    pod_rows = {}

    cur_start = start_dt
    slice_idx = 0

    while cur_start < end_dt:
        cur_end = min(cur_start + slice_delta, end_dt)
        slice_idx += 1

        s_ns = to_unix_ns(cur_start)
        e_ns = to_unix_ns(cur_end)

        print(f"\n[INFO] Slice #{slice_idx}: {cur_start.isoformat()}  ~  {cur_end.isoformat()}")

        streams = loki_query_range(
            base_url=LOKI_URL,
            query=LOG_QUERY,
            start_ns=s_ns,
            end_ns=e_ns,
            limit=LOKI_LIMIT,         # Single request <= 5000
            direction=LOKI_DIRECTION,
        )

        print(f"[INFO]   Got {len(streams)} streams in this slice")

        for idx, stream in enumerate(streams):
            labels = stream.get("stream", {})   # {namespace, pod, container, ...}
            values = stream.get("values", [])   # [[ts_ns, line], ...]

            # If log volume is very large, this debug can be commented out
            # print(f"[DEBUG]   stream #{idx}, labels={labels}, num_values={len(values)}")

            for ts_ns_str, line in values:
                try:
                    ts_ns = int(ts_ns_str)
                except (TypeError, ValueError):
                    continue

                ts_sec = ts_ns / 1_000_000_000
                dt = datetime.fromtimestamp(ts_sec, tz=timezone.utc).isoformat()

                # Try to parse JSON logs (optional)
                json_fields = None
                if line.strip().startswith("{") and line.strip().endswith("}"):
                    try:
                        json_fields = json.loads(line)
                    except json.JSONDecodeError:
                        json_fields = None

                row = {
                    "timestamp": dt,
                    "log": line,
                    "namespace": labels.get("namespace"),
                    "pod": labels.get("pod"),
                    "container": labels.get("container"),
                    "app": labels.get("app") or labels.get("service"),
                    "host": labels.get("host") or labels.get("instance") or labels.get("node"),
                }

                if isinstance(json_fields, dict):
                    row["log_level"] = json_fields.get("level")
                    row["log_message"] = json_fields.get("msg") or json_fields.get("message")
                    row["trace_id"] = json_fields.get("trace_id") or json_fields.get("traceId")
                    row["span_id"] = json_fields.get("span_id") or json_fields.get("spanId")
                    row["json_fields"] = json.dumps(json_fields, ensure_ascii=False)
                else:
                    row["log_level"] = None
                    row["log_message"] = None
                    row["trace_id"] = None
                    row["span_id"] = None
                    row["json_fields"] = None

                # No longer adding experiment_meta to each row
                # row.update(experiment_meta)

                # Get pod name directly from labels
                pod_name = labels.get("pod", "unknown")

                # Group by pod
                if pod_name not in pod_rows:
                    pod_rows[pod_name] = []
                pod_rows[pod_name].append(row)

        # Next time slice
        cur_start = cur_end

    total_logs = sum(len(rows) for rows in pod_rows.values())
    print(f"\n[INFO] Total log lines collected: {total_logs}")
    print(f"[INFO] Number of unique pods: {len(pod_rows)}")
    for pod_name, rows in pod_rows.items():
        print(f"[INFO]   - {pod_name}: {len(rows)} logs")

    return pod_rows


# ========== 5. Save as CSV / Parquet ==========

def save_rows_by_pod(pod_rows: dict, output_dir=None):
    """
    Save CSV files separately by pod
    pod_rows: {pod_name: [rows]}
    output_dir: Path object or None (defaults to script directory)
    """
    if not pod_rows:
        print("[WARN] No logs fetched. Please check LOKI_URL / LOG_QUERY / time range / LIMIT.")
        return

    # Use absolute path based on script location if not provided
    if output_dir is None:
        output_dir = Path(__file__).parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exp_id = experiment_meta.get("experiment_id", "exp")

    for pod_name, rows in pod_rows.items():
        if not rows:
            continue

        df = pd.DataFrame(rows)

        # Remove columns where all values are None
        df = df.dropna(axis=1, how='all')

        # Sort: timestamp + container
        sort_cols = [col for col in ["timestamp", "container"] if col in df.columns]
        if sort_cols:
            df.sort_values(sort_cols, inplace=True)

        # Clean pod name, remove invalid filename characters
        safe_pod_name = pod_name.replace("/", "_").replace(":", "_")
        csv_file = output_dir / f"loki_logs_{safe_pod_name}.csv"

        # Save CSV
        df.to_csv(csv_file, index=False)
        print(f"[INFO] Saved {len(df)} logs to {csv_file}")

    print(f"\n[INFO] Total files saved: {len(pod_rows)}")


# ========== 6. Main Entry Point ==========

if __name__ == "__main__":
    pod_rows = fetch_logs_to_rows()

    # Save to different CSV files by pod (uses script directory by default)
    save_rows_by_pod(pod_rows)
