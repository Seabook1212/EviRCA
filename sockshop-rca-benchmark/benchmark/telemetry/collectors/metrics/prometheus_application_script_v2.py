import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests


PROM_URL = os.environ.get("PROM_URL", "http://34.28.33.102:30990")
START_TIME = os.environ.get("START_TIME", os.environ.get("PROM_START", "2026-02-22T00:00:00Z"))
END_TIME = os.environ.get("END_TIME", os.environ.get("PROM_END", "2026-02-22T00:15:00Z"))
STEP = os.environ.get("PROM_STEP", "5s")
NAMESPACE = os.environ.get("PROM_NAMESPACE", "sock-shop")

SCRIPT_DIR = Path(__file__).resolve().parent

GO_METRIC_FILE = os.environ.get(
    "GO_METRIC_LIST_FILE",
    str(SCRIPT_DIR / "data" / "applied" / "go_level_applied_metrics_name.csv"),
)
JAVA_METRIC_FILE = os.environ.get(
    "JAVA_METRIC_LIST_FILE",
    str(SCRIPT_DIR / "data" / "applied" / "java_level_applied_metrics_name.csv"),
)
NODEJS_METRIC_FILE = os.environ.get(
    "NODEJS_METRIC_LIST_FILE",
    str(SCRIPT_DIR / "data" / "applied" / "nodejs_level_applied_metrics_name.csv"),
)

OUTPUT_FILE = os.environ.get(
    "APPLICATION_METRIC_OUTPUT_FILE",
    str(SCRIPT_DIR / "data" / "demo_data" / "prometheus_metrics_application_raw.csv"),
)

OUTPUT_COLUMNS = ["timestamp", "pod", "metric", "value", "labels"]

SOURCE_CONFIGS = [
    {
        "name": "Go",
        "metric_file": GO_METRIC_FILE,
        "selector": f'namespace="{NAMESPACE}",container=~"user|payment|catalogue"',
        "important_labels": ["le", "method", "path", "status_code", "route"],
    },
    {
        "name": "Java",
        "metric_file": JAVA_METRIC_FILE,
        "selector": f'namespace="{NAMESPACE}",container=~"carts|orders|queue-master|shipping"',
        "important_labels": ["name", "method", "status", "net_peer_name", "uri", "id", "area", "state", "queue"],
    },
    {
        "name": "NodeJS",
        "metric_file": NODEJS_METRIC_FILE,
        "selector": f'namespace="{NAMESPACE}",container="front-end"',
        "important_labels": ["space", "kind", "le", "method", "path", "status_code"],
    },
]


def resolve_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = (SCRIPT_DIR / path).resolve()
    return path


def load_metric_list(path_text: str) -> list[str]:
    metric_list_path = resolve_path(path_text)
    if not metric_list_path.exists():
        raise FileNotFoundError(f"Metric list CSV not found: {metric_list_path}")

    metrics_df = pd.read_csv(metric_list_path)
    if "metric_name" not in metrics_df.columns:
        raise ValueError(f"`metric_name` column not found in {metric_list_path}")

    metrics = []
    for metric in metrics_df["metric_name"].dropna().astype(str).tolist():
        metric = metric.strip()
        if metric:
            metrics.append(metric)

    return list(dict.fromkeys(metrics))


def extract_labels(metric_labels: dict, important_labels: list[str]) -> str:
    filtered = {}
    for key in important_labels:
        if key in metric_labels:
            filtered[key] = metric_labels[key]
    return json.dumps(filtered, sort_keys=True, ensure_ascii=False)


def query_promql(promql: str, step: str):
    url = f"{PROM_URL}/api/v1/query_range"
    params = {
        "query": promql,
        "start": START_TIME,
        "end": END_TIME,
        "step": step,
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json().get("data", {}).get("result", [])


def query_range(metric_name: str, selector: str, important_labels: list[str]) -> list[dict]:
    promql = f"{metric_name}{{{selector}}}"
    data = query_promql(promql, STEP)

    rows = []
    for series in data:
        metric_labels = series.get("metric", {})
        pod = metric_labels.get("pod", "unknown")
        labels_json = extract_labels(metric_labels, important_labels)

        for ts, val in series.get("values", []):
            timestamp = datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            try:
                value = float(val)
            except (TypeError, ValueError):
                continue

            rows.append(
                {
                    "timestamp": timestamp,
                    "pod": pod,
                    "metric": metric_name,
                    "value": value,
                    "labels": labels_json,
                }
            )

    return rows


def main() -> None:
    all_rows = []

    for source in SOURCE_CONFIGS:
        metrics = load_metric_list(source["metric_file"])
        print(f"Loaded {len(metrics)} {source['name']} metrics")

        for metric in metrics:
            print(f"Querying [{source['name']}] {metric} ...")
            try:
                rows = query_range(metric, source["selector"], source["important_labels"])
                all_rows.extend(rows)
            except Exception as e:
                print(f"Failed [{source['name']}] {metric} -> {e}")

    result_df = pd.DataFrame(all_rows, columns=OUTPUT_COLUMNS)

    output_path = resolve_path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)

    print("\nExport completed")
    print(f"Output file -> {output_path}")
    print(f"Total rows -> {len(result_df)}")


if __name__ == "__main__":
    main()
