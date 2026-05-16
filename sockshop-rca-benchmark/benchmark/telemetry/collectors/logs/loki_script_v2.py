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

# Maximum log entries per request.
# The script will page through the time window until no more entries remain.
LOKI_LIMIT = 5000

# Request direction: forward = time from start → end
LOKI_DIRECTION = "forward"

# ========== 1. Time Window Configuration ==========
# Recommended: align with your Prometheus / Jaeger experiment time

# Option A: Last N hours (relative to now)
USE_RELATIVE_WINDOW = False
RELATIVE_HOURS = 0.25  # e.g., last 15 minutes (0.25 hours)

# Option B: Fixed time period (UTC ISO8601), uncomment and set these when USE_RELATIVE_WINDOW = False
START_ISO = "2026-03-03T09:01:36+00:00"
END_ISO   = "2026-03-03T09:16:36+00:00"

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

# ========== 3. Loki HTTP Call Wrapper ==========

def loki_query_range(
    base_url: str,
    query: str,
    start_ns: int,
    end_ns: int,
    limit: int | None = None,
    direction: str = "forward",
    max_retries: int = 3,
):
    url = f"{base_url.rstrip('/')}/loki/api/v1/query_range"
    params = {
        "query": query,
        "start": str(start_ns),
        "end": str(end_ns),
        "direction": direction,
    }
    if limit is not None and limit > 0:
        params["limit"] = str(limit)
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


def fetch_slice_rows(start_ns: int, end_ns: int, seen_pods: set[str]):
    """
    Fetch one time slice completely by paging forward until Loki returns no more logs.
    This avoids losing data when the server applies a per-request entry cap.
    """
    slice_rows = []
    seen_row_keys = set()
    cursor_ns = start_ns
    request_idx = 0

    while cursor_ns < end_ns:
        request_idx += 1
        streams = loki_query_range(
            base_url=LOKI_URL,
            query=LOG_QUERY,
            start_ns=cursor_ns,
            end_ns=end_ns,
            limit=LOKI_LIMIT,
            direction=LOKI_DIRECTION,
        )

        batch_count = 0
        max_ts_ns = None

        for stream in streams:
            labels = stream.get("stream", {})
            values = stream.get("values", [])

            for ts_ns_str, line in values:
                try:
                    ts_ns = int(ts_ns_str)
                except (TypeError, ValueError):
                    continue

                if max_ts_ns is None or ts_ns > max_ts_ns:
                    max_ts_ns = ts_ns

                ts_sec = ts_ns / 1_000_000_000
                dt = datetime.fromtimestamp(ts_sec, tz=timezone.utc).isoformat()

                row = {
                    "timestamp": dt,
                    "node": labels.get("node_name"),
                    "pod": labels.get("pod"),
                    "container": labels.get("container"),
                    "log": line,
                }
                row_key = (
                    ts_ns,
                    row["node"],
                    row["pod"],
                    row["container"],
                    row["log"],
                )
                if row_key in seen_row_keys:
                    continue

                seen_row_keys.add(row_key)
                slice_rows.append(row)
                seen_pods.add(labels.get("pod", "unknown"))
                batch_count += 1

        print(f"[INFO]   Batch #{request_idx}: collected {batch_count} log lines")

        if max_ts_ns is None:
            break

        next_cursor_ns = max_ts_ns + 1
        if next_cursor_ns <= cursor_ns:
            print("[WARN] Loki pagination cursor did not advance. Stopping to avoid infinite loop.")
            break
        cursor_ns = next_cursor_ns

    return slice_rows

# ========== 4. Log Fetching + Flattening ==========

def fetch_logs_to_rows():
    """
    Fetch logs from Loki and flatten into rows (list[dict]).
    One row corresponds to one log entry in Loki.

    Uses "time slicing" approach to avoid triggering max_entries_limit in a single query.
    Returns a flat list of rows.
    """
    start_dt, end_dt = get_time_range()

    # Length of each time slice (adjust as needed)
    slice_minutes = 5
    slice_delta = timedelta(minutes=slice_minutes)

    print(f"[INFO] Fetching logs from Loki: {LOKI_URL}")
    print(f"[INFO]   query       = {LOG_QUERY}")
    print(f"[INFO]   time range  = {start_dt.isoformat()}  ~  {end_dt.isoformat()}")
    limit_text = LOKI_LIMIT if LOKI_LIMIT is not None else "no explicit limit"
    print(f"[INFO]   slice       = {slice_minutes} minutes per query, limit={limit_text}")

    all_rows = []
    seen_pods = set()

    cur_start = start_dt
    slice_idx = 0

    while cur_start < end_dt:
        cur_end = min(cur_start + slice_delta, end_dt)
        slice_idx += 1

        s_ns = to_unix_ns(cur_start)
        e_ns = to_unix_ns(cur_end)

        print(f"\n[INFO] Slice #{slice_idx}: {cur_start.isoformat()}  ~  {cur_end.isoformat()}")

        slice_rows = fetch_slice_rows(s_ns, e_ns, seen_pods)
        print(f"[INFO]   Total collected in this slice: {len(slice_rows)}")
        all_rows.extend(slice_rows)

        # Next time slice
        cur_start = cur_end

    total_logs = len(all_rows)
    print(f"\n[INFO] Total log lines collected: {total_logs}")
    print(f"[INFO] Number of unique pods: {len(seen_pods)}")

    return all_rows


# ========== 5. Save as CSV / Parquet ==========

def save_rows(all_rows: list, output_dir=None, output_file: str = "loki_logs_raw.csv"):
    """
    Save a single merged CSV for all pods.
    all_rows: list[dict]
    output_dir: Path object or None (defaults to script's data directory)
    """
    if not all_rows:
        print("[WARN] No logs fetched. Please check LOKI_URL / LOG_QUERY / time range / LIMIT.")
        return

    # Use script-local data directory by default.
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "data"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(all_rows, columns=["timestamp", "node", "pod", "container", "log"])

    # Sort: timestamp + node + pod + container
    sort_cols = [col for col in ["timestamp", "node", "pod", "container"] if col in df.columns]
    if sort_cols:
        df.sort_values(sort_cols, inplace=True)

    csv_file = output_dir / output_file
    df.to_csv(csv_file, index=False)
    print(f"[INFO] Saved {len(df)} logs to {csv_file}")


# ========== 6. Main Entry Point ==========

if __name__ == "__main__":
    all_rows = fetch_logs_to_rows()

    # Save merged CSV for all pods (uses script-local data directory by default)
    save_rows(all_rows)
