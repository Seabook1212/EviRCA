import os
import json
import requests
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

# ---------------- CONFIG ----------------

PROM_URL = os.environ.get("PROM_URL", "http://34.28.33.102:30990")

START_TIME = os.environ.get("PROM_START", "2026-02-22T00:00:00Z")
END_TIME   = os.environ.get("PROM_END",   "2026-02-22T00:15:00Z")

STEP = os.environ.get("PROM_STEP", "30s")

SCRIPT_DIR = Path(__file__).resolve().parent

METRIC_LIST_FILE = os.environ.get(
    "NGINX_METRIC_LIST_FILE",
    str(SCRIPT_DIR / "data" / "applied" / "nginx_ingress_level_applied_metrics_name.csv"),
)

OUTPUT_FILE = os.environ.get(
    "NGINX_METRIC_OUTPUT_FILE",
    str(SCRIPT_DIR / "data" / "demo_data" / "prometheus_metrics_network_raw.csv"),
)

TARGET_NAMESPACE = os.environ.get("NGINX_NAMESPACE", "sock-shop")

# ---------------- TIME ----------------

start_time = datetime.fromisoformat(START_TIME.replace("Z", "+00:00"))
end_time   = datetime.fromisoformat(END_TIME.replace("Z", "+00:00"))

# ---------------- LOAD METRICS ----------------

metric_list_path = Path(METRIC_LIST_FILE).expanduser()
if not metric_list_path.is_absolute():
    metric_list_path = (SCRIPT_DIR / metric_list_path).resolve()

if not metric_list_path.exists():
    raise FileNotFoundError(f"Metric list CSV not found: {metric_list_path}")

metrics_df = pd.read_csv(metric_list_path)

if "metric_name" not in metrics_df.columns:
    raise ValueError("CSV must contain metric_name column")

metric_list = metrics_df["metric_name"].tolist()

print(f"Total NGINX Ingress metrics: {len(metric_list)}")

# ---------------- LABEL EXTRACTION ----------------

IMPORTANT_LABELS = [
    "status",
    "method",
    "ingress",
    "path",
    "host",
]


def extract_labels(metric_labels):
    filtered = {}
    for key in IMPORTANT_LABELS:
        if key in metric_labels:
            filtered[key] = metric_labels[key]
    return json.dumps(filtered, sort_keys=True)

# ---------------- QUERY FUNCTION ----------------

def query_metric(metric_name):

    url = f"{PROM_URL}/api/v1/query_range"

    promql = f'{metric_name}{{exported_namespace="{TARGET_NAMESPACE}"}}'

    params = {
        "query": promql,
        "start": start_time.isoformat(),
        "end": end_time.isoformat(),
        "step": STEP,
    }

    response = requests.get(url, params=params)
    response.raise_for_status()

    result = response.json()["data"]["result"]

    rows = []

    for series in result:

        labels = series.get("metric", {})

        exported_service = labels.get("exported_service")
        labels_json = extract_labels(labels)

        if not exported_service:
            continue

        for ts, val in series.get("values", []):

            timestamp = datetime.fromtimestamp(float(ts), tz=timezone.utc)\
                                 .strftime("%Y-%m-%d %H:%M:%S")

            try:
                fval = float(val)
            except:
                continue

            rows.append({
                "timestamp": timestamp,
                "service": exported_service,
                "metric": metric_name,
                "value": fval,
                "labels": labels_json,
            })

    return rows

# ---------------- EXECUTE ----------------

all_rows = []

for metric in metric_list:

    print(f"Querying {metric} ...")

    try:
        rows = query_metric(metric)
        all_rows.extend(rows)

    except Exception as e:
        print(f"⚠ Failed {metric} → {e}")

# ---------------- SAVE ----------------

df = pd.DataFrame(all_rows)

output_path = Path(OUTPUT_FILE)
if not output_path.is_absolute():
    output_path = (SCRIPT_DIR / output_path).resolve()

output_path.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(output_path, index=False)

print("✅ Done")
print(f"Saved → {output_path}")
print(f"Total rows → {len(df)}")
