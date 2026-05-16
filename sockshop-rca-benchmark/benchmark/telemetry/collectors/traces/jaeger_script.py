import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
from pathlib import Path
import json

# ===== 0. Basic Configuration =====
# Same address as Jaeger UI. If using NodePort, use NodePort address.
# Example: JAEGER_URL = "http://35.224.111.144:30686"
JAEGER_URL = "http://34.28.33.102:32614"

# Collect traces from all services (no filtering)
# Services of interest will be auto-discovered from Jaeger
services_of_interest = []  # Will be populated automatically

# Experiment time window (UTC)
# Can align with your Prometheus exp_001 time range
end = datetime.now(timezone.utc)
start = end - timedelta(minutes=15)   # Last 15 minutes

# Jaeger uses "Unix microseconds"
def to_unix_us(dt: datetime) -> int:
    return int(dt.timestamp() * 1_000_000)

start_us = to_unix_us(start)
end_us = to_unix_us(end)

# Experiment metadata (aligned with your metrics dataset)
experiment_meta = {
    "experiment_id": "exp_001",
    "fault_type": "pod_kill",
    "target_service": "sock-shop-carts",
    "note": "kill carts pod once at t+10min"
}

# ===== 1. Utility Functions =====

def jaeger_get(path: str, params=None):
    url = f"{JAEGER_URL}{path}"
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()

def list_services():
    data = jaeger_get("/api/services")
    return data.get("data", [])

def list_operations(service: str):
    data = jaeger_get("/api/operations", params={"service": service})
    return data.get("data", [])

def query_traces(service: str, start_us: int, end_us: int, limit: int = 200):
    """
    Fetch a batch of traces from Jaeger.
    Note: No strict pagination mechanism, limit is just an upper bound.
    If you're worried about missing traces, shorten the time window and call multiple times.
    """
    params = {
        "service": service,
        "start": start_us,
        "end": end_us,
        "limit": limit,
        # Can also add filters like operation/tags/minDuration/maxDuration
    }
    data = jaeger_get("/api/traces", params=params)
    return data.get("data", [])


# ===== 2. Fetch Traces and Flatten into Table =====

# Store rows grouped by service
service_rows = {}

# Auto-discover all services from Jaeger
all_services = list_services()
print(f"Available services in Jaeger: {all_services}")

# Filter out Jaeger internal services
excluded_services = ["jaeger-all-in-one", "jaeger-query", "jaeger-collector"]
services_of_interest = [svc for svc in all_services if svc not in excluded_services]
print(f"\nFiltered out: {[svc for svc in all_services if svc in excluded_services]}")
print(f"Collecting traces from {len(services_of_interest)} services")

if not services_of_interest:
    print(f"[WARN] No services found in Jaeger. Exiting.")
    exit(0)

for service in services_of_interest:
    print(f"\n=== Fetching traces for service: {service} ===")
    traces = query_traces(service, start_us, end_us, limit=500)  # Adjust limit as needed

    print(f"  Got {len(traces)} traces for {service}")

    # Initialize rows list for this service if not exists
    if service not in service_rows:
        service_rows[service] = []

    for trace in traces:
        trace_id = trace.get("traceID")
        processes = trace.get("processes", {})
        spans = trace.get("spans", [])

        # Build a processID -> serviceName index for easy span mapping
        proc_service = {
            pid: pinfo.get("serviceName")
            for pid, pinfo in processes.items()
        }

        for span in spans:
            span_id = span.get("spanID")
            operation = span.get("operationName")
            start_time_us = span.get("startTime")   # microseconds
            duration_us = span.get("duration")      # microseconds

            # Convert to ISO8601 time string
            dt = datetime.fromtimestamp(start_time_us / 1_000_000, tz=timezone.utc)
            start_ts = dt.isoformat()

            # Service that span belongs to
            svc = proc_service.get(span.get("processID"), service)

            # Flatten tags/logs a bit (simple approach: store as JSON string)
            tags = {t.get("key"): t.get("value") for t in span.get("tags", [])}
            logs = []
            for lg in span.get("logs", []):
                lg_ts = datetime.fromtimestamp(lg.get("timestamp") / 1_000_000, tz=timezone.utc).isoformat()
                lg_fields = {f.get("key"): f.get("value") for f in lg.get("fields", [])}
                logs.append({"timestamp": lg_ts, "fields": lg_fields})

            row = {
                "trace_id": trace_id,
                "span_id": span_id,
                "parent_span_id": span.get("references", [{}])[0].get("spanID") if span.get("references") else None,
                "service": svc,
                "operation": operation,
                "start_time": start_ts,
                "duration_us": duration_us,
                "tags_json": json.dumps(tags, ensure_ascii=False),
                "logs_json": json.dumps(logs, ensure_ascii=False),
            }

            # No longer adding experiment metadata to each row
            # row.update(experiment_meta)

            # Add row to the service's row list
            service_rows[service].append(row)

# ===== 3. Save as CSV / Parquet =====

# Use absolute path based on script location
script_dir = Path(__file__).parent
output_dir = script_dir
output_dir.mkdir(parents=True, exist_ok=True)

exp_id = experiment_meta.get("experiment_id", "exp")

if not service_rows:
    print("No spans fetched. Check service names / time range / Jaeger URL.")
else:
    total_spans = 0
    for service, rows in service_rows.items():
        if not rows:
            continue

        df = pd.DataFrame(rows)

        # Remove columns where all values are None
        df = df.dropna(axis=1, how='all')

        # Sort by start_time, trace_id, span_id
        df.sort_values(["start_time", "trace_id", "span_id"], inplace=True)

        # Clean service name, remove invalid filename characters
        safe_service_name = service.replace("/", "_").replace(":", "_").replace(".", "_")
        csv_file = output_dir / f"jaeger_traces_{safe_service_name}.csv"

        # Save CSV
        df.to_csv(csv_file, index=False)
        print(f"[INFO] Saved {len(df)} spans to {csv_file}")
        total_spans += len(df)

    print(f"\n[INFO] Total files saved: {len(service_rows)}")
    print(f"[INFO] Total spans saved: {total_spans}")
