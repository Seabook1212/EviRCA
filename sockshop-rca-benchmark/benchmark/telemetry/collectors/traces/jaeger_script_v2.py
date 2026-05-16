import argparse
import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
from requests.exceptions import ChunkedEncodingError, ConnectionError, HTTPError, RequestException, Timeout

# ===== 0. Basic Configuration =====
# Same address as Jaeger UI. If using NodePort, use NodePort address.
DEFAULT_JAEGER_URL = "http://34.28.33.102:32614"
DEFAULT_EXCLUDED_SERVICES = ["jaeger-all-in-one", "jaeger-query", "jaeger-collector"]
DEFAULT_OUTPUT_DIR = str((Path(__file__).resolve().parent / "data"))


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


# Environment-based defaults (CLI can still override these).
ENV_JAEGER_URL = os.environ.get("JAEGER_URL", DEFAULT_JAEGER_URL)
ENV_START_TIME = os.environ.get("JAEGER_START", os.environ.get("PROM_START", "2026-03-03T13:25:56Z")).strip()
ENV_END_TIME = os.environ.get("JAEGER_END", os.environ.get("PROM_END", "2026-03-03T13:40:56Z")).strip()
ENV_MINUTES = _env_int("JAEGER_MINUTES", 15)
# 0 means "no explicit client-side limit parameter".
ENV_LIMIT = _env_int("JAEGER_LIMIT", 10000)
ENV_OUTPUT_DIR = os.environ.get("JAEGER_OUTPUT_DIR", DEFAULT_OUTPUT_DIR).strip() or DEFAULT_OUTPUT_DIR
ENV_OUTPUT_FILE = os.environ.get("JAEGER_OUTPUT_FILE", "jaeger_traces_raw.csv").strip() or "jaeger_traces_raw.csv"
ENV_HTTP_TIMEOUT_SECONDS = _env_int("JAEGER_HTTP_TIMEOUT_SECONDS", 120)
ENV_HTTP_RETRIES = _env_int("JAEGER_HTTP_RETRIES", 4)
ENV_HTTP_BACKOFF_SECONDS = float(os.environ.get("JAEGER_HTTP_BACKOFF_SECONDS", "2"))
ENV_FETCH_MODE = os.environ.get("JAEGER_FETCH_MODE", "per_service").strip() or "per_service"
ENV_SPLIT_MIN_SECONDS = _env_int("JAEGER_SPLIT_MIN_SECONDS", 1)


def to_unix_us(dt: datetime) -> int:
    # Jaeger uses Unix microseconds.
    return int(dt.timestamp() * 1_000_000)


def parse_utc_time(text: str) -> datetime:
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Jaeger traces into one raw CSV.")
    parser.add_argument(
        "--jaeger-url",
        default=ENV_JAEGER_URL,
        help="Jaeger base URL, e.g. http://34.28.33.102:32614",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=ENV_START_TIME,
        help="UTC start time in ISO8601, e.g. 2026-02-22T00:00:00Z",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=ENV_END_TIME,
        help="UTC end time in ISO8601, e.g. 2026-02-22T00:15:00Z",
    )
    parser.add_argument(
        "--minutes",
        type=int,
        default=ENV_MINUTES,
        help="If --start/--end not provided, collect the last N minutes (default: 15).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=ENV_LIMIT,
        help=(
            "Max traces per query before the script time-splits the window. "
            "Use 0 for no explicit limit parameter."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=ENV_OUTPUT_DIR,
        help="Output directory (default: traces_script/data).",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=ENV_OUTPUT_FILE,
        help="Raw output CSV filename.",
    )
    parser.add_argument(
        "--fetch-mode",
        type=str,
        default=ENV_FETCH_MODE,
        choices=["per_service"],
        help="Trace fetch strategy. per_service=query each service separately.",
    )
    parser.add_argument(
        "--split-min-seconds",
        type=int,
        default=ENV_SPLIT_MIN_SECONDS,
        help="Smallest time window (seconds) allowed when recursively splitting saturated queries.",
    )
    return parser.parse_args()


def jaeger_get(jaeger_url: str, path: str, params=None):
    url = f"{jaeger_url.rstrip('/')}{path}"
    last_error = None
    retries = max(1, ENV_HTTP_RETRIES)

    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(
                url,
                params=params,
                timeout=ENV_HTTP_TIMEOUT_SECONDS,
                headers={"Connection": "close"},
            )
            resp.raise_for_status()
            return resp.json()
        except HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            body = ""
            if e.response is not None and e.response.text:
                body = e.response.text.strip().replace("\n", " ")[:300]

            # 4xx (except 429) is usually a bad query and retrying is pointless.
            if status is not None and 400 <= status < 500 and status != 429:
                raise RuntimeError(
                    f"Jaeger request rejected with HTTP {status} for {path}. "
                    f"Check query parameters. Response: {body}"
                ) from e

            last_error = e
            if attempt >= retries:
                break
            sleep_s = max(0.0, ENV_HTTP_BACKOFF_SECONDS) * attempt
            print(
                f"[WARN] Jaeger request failed (attempt {attempt}/{retries}) for {path}: {e}. "
                f"Retrying in {sleep_s:.1f}s..."
            )
            time.sleep(sleep_s)
        except (ChunkedEncodingError, ConnectionError, Timeout, RequestException) as e:
            last_error = e
            if attempt >= retries:
                break
            sleep_s = max(0.0, ENV_HTTP_BACKOFF_SECONDS) * attempt
            print(
                f"[WARN] Jaeger request failed (attempt {attempt}/{retries}) for {path}: {e}. "
                f"Retrying in {sleep_s:.1f}s..."
            )
            time.sleep(sleep_s)

    raise RuntimeError(f"Jaeger request failed after {retries} attempts for {path}: {last_error}")


def list_services(jaeger_url: str):
    data = jaeger_get(jaeger_url, "/api/services")
    return data.get("data", [])


def query_traces(jaeger_url: str, service: str | None, start_us: int, end_us: int, limit: int):
    params = {
        "start": start_us,
        "end": end_us,
    }
    if service:
        params["service"] = service
    if limit and int(limit) > 0:
        params["limit"] = int(limit)
    data = jaeger_get(jaeger_url, "/api/traces", params=params)
    return data.get("data", [])


def fetch_traces_with_time_splitting(
    jaeger_url: str,
    service: str | None,
    start_us: int,
    end_us: int,
    limit: int,
    min_window_us: int,
    label: str,
    depth: int = 0,
):
    traces = query_traces(jaeger_url, service, start_us, end_us, limit=limit)
    window_us = max(0, end_us - start_us)
    indent = "  " * depth
    print(f"{indent}Window {label}: got {len(traces)} traces")

    # If limit is disabled, we cannot infer truncation from result size.
    if not limit or int(limit) <= 0:
        return traces

    # Jaeger may truncate when result count reaches the requested limit.
    if len(traces) < int(limit):
        return traces

    if window_us <= max(1, min_window_us):
        print(
            f"{indent}[WARN] Window {label} still reached limit={limit} at the minimum split size. "
            "Results may still be truncated."
        )
        return traces

    mid_us = start_us + (window_us // 2)
    if mid_us <= start_us or mid_us >= end_us:
        print(
            f"{indent}[WARN] Cannot split window {label} further even though it reached limit={limit}. "
            "Results may still be truncated."
        )
        return traces

    print(f"{indent}[INFO] Window {label} reached limit={limit}; splitting time range")
    left_traces = fetch_traces_with_time_splitting(
        jaeger_url=jaeger_url,
        service=service,
        start_us=start_us,
        end_us=mid_us,
        limit=limit,
        min_window_us=min_window_us,
        label=f"{label}.L",
        depth=depth + 1,
    )
    right_traces = fetch_traces_with_time_splitting(
        jaeger_url=jaeger_url,
        service=service,
        start_us=mid_us + 1,
        end_us=end_us,
        limit=limit,
        min_window_us=min_window_us,
        label=f"{label}.R",
        depth=depth + 1,
    )
    return left_traces + right_traces


def flatten_traces_raw(traces, fallback_service: str):
    rows = []
    for trace in traces:
        trace_id = trace.get("traceID")
        processes = trace.get("processes", {})
        spans = trace.get("spans", [])

        proc_service = {
            pid: pinfo.get("serviceName")
            for pid, pinfo in processes.items()
        }

        for span in spans:
            svc = proc_service.get(span.get("processID"), fallback_service)

            rows.append(
                {
                    "start_time": span.get("startTime"),
                    "trace_id": trace_id,
                    "span_id": span.get("spanID"),
                    "service": svc,
                    "operation": span.get("operationName"),
                    "duration": span.get("duration"),
                    "references": json.dumps(span.get("references", []), ensure_ascii=False),
                    "tags": json.dumps(span.get("tags", []), ensure_ascii=False),
                    "logs": json.dumps(span.get("logs", []), ensure_ascii=False),
                }
            )
    return rows


def main() -> None:
    args = parse_args()

    # Time window: either explicit --start/--end, or "last N minutes".
    if args.start and args.end:
        start_dt = parse_utc_time(args.start)
        end_dt = parse_utc_time(args.end)
    else:
        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(minutes=max(1, int(args.minutes)))

    if end_dt <= start_dt:
        raise ValueError("Invalid time window: end must be later than start.")

    start_us = to_unix_us(start_dt)
    end_us = to_unix_us(end_dt)
    min_window_us = max(1, int(args.split_min_seconds)) * 1_000_000

    all_services = list_services(args.jaeger_url)
    services_of_interest = [svc for svc in all_services if svc not in DEFAULT_EXCLUDED_SERVICES]
    print(f"Available services in Jaeger: {all_services}")
    print(f"Filtered out: {[svc for svc in all_services if svc in DEFAULT_EXCLUDED_SERVICES]}")
    print(f"Collecting traces from {len(services_of_interest)} services")
    print(f"Time window UTC: {start_dt.isoformat()} -> {end_dt.isoformat()}")
    print(f"Fetch mode: {args.fetch_mode}")
    print(f"Minimum split window: {int(args.split_min_seconds)}s")
    print(
        "Requested limit per query: "
        + (str(int(args.limit)) if int(args.limit) > 0 else "server default (no limit param)")
    )

    if not services_of_interest:
        print("[WARN] No services found in Jaeger. Exiting.")
        return

    all_rows = []
    seen_span_keys = set()

    def _append_rows(rows):
        for row in rows:
            key = (row.get("trace_id"), row.get("span_id"))
            if key in seen_span_keys:
                continue
            seen_span_keys.add(key)
            all_rows.append(row)

    for service in services_of_interest:
        print(f"\n=== Fetching traces for service: {service} ===")
        traces = fetch_traces_with_time_splitting(
            jaeger_url=args.jaeger_url,
            service=service,
            start_us=start_us,
            end_us=end_us,
            limit=int(args.limit),
            min_window_us=min_window_us,
            label=service,
        )
        print(f"  Got {len(traces)} traces for {service} after splitting")
        _append_rows(flatten_traces_raw(traces, fallback_service=service))

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / args.output_file

    if not all_rows:
        print("No spans fetched. Check service names / time range / Jaeger URL.")
        print(
            "Hint: this usually means the selected UTC window has no traffic. "
            "Try explicit --start/--end from your fault/normal run window."
        )
        return

    ordered_columns = [
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
    df = pd.DataFrame(all_rows)
    df = df.reindex(columns=ordered_columns)
    df.sort_values(["start_time", "trace_id", "span_id"], inplace=True)
    df.to_csv(output_file, index=False)

    print(f"\n[INFO] Saved raw spans: {len(df)}")
    print(f"[INFO] Output file: {output_file}")


if __name__ == "__main__":
    main()
