import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

# ---------------- CONFIG ----------------

PROM_URL = os.environ.get("PROM_URL", "http://34.28.33.102:30990")
START_TIME = os.environ.get("PROM_START", "2026-02-22T00:00:00Z")
END_TIME = os.environ.get("PROM_END", "2026-02-22T00:15:00Z")
CONTAINER_STEP = os.environ.get("PROM_STEP", "5s")
KUBE_POD_STEP = os.environ.get("KUBE_POD_STEP", "5m")

SCRIPT_DIR = Path(__file__).resolve().parent

CONTAINER_METRIC_LIST_FILE = os.environ.get(
    "CONTAINER_METRIC_LIST_FILE",
    str(SCRIPT_DIR / "data" / "applied" / "container_level_applied_metrics_name.csv"),
)
KUBE_POD_METRIC_LIST_FILE = os.environ.get(
    "KUBE_POD_METRIC_LIST_FILE",
    str(SCRIPT_DIR / "data" / "applied" / "kube_pod_level_applied_metrics_name.csv"),
)

OUTPUT_FILE = os.environ.get(
    "CONTAINER_METRIC_OUTPUT_FILE",
    str(SCRIPT_DIR / "data" / "demo_data" / "prometheus_metrics_container_raw.csv"),
)

KUBE_POD_NAMESPACE = os.environ.get("KUBE_POD_NAMESPACE", "sock-shop").strip()

TARGET_CONTAINERS = {
    "front-end",
    "catalogue",
    "carts",
    "orders",
    "catalogue-db",
    "carts-db",
    "orders-db",
    "payment",
    "shipping",
    "queue-master",
    "user",
    "user-db",
    "rabbitmq",
    "session-db",
}

OUTPUT_COLUMNS = ["timestamp", "pod", "metric", "value", "labels"]

# ---------------- TIME ----------------

start_time = datetime.fromisoformat(START_TIME.replace("Z", "+00:00"))
end_time = datetime.fromisoformat(END_TIME.replace("Z", "+00:00"))


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


def query_range(promql: str, step: str):
    url = f"{PROM_URL}/api/v1/query_range"
    params = {
        "query": promql,
        "start": start_time.isoformat(),
        "end": end_time.isoformat(),
        "step": step,
    }
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    return response.json().get("data", {}).get("result", [])


def query_container_metric(metric_name: str):
    rows = []
    for series in query_range(metric_name, CONTAINER_STEP):
        metric_labels = series.get("metric", {})
        pod = metric_labels.get("pod")
        container = metric_labels.get("container")
        if not pod or not container:
            continue
        if container not in TARGET_CONTAINERS:
            continue

        for ts, val in series.get("values", []):
            try:
                value = float(val)
            except (TypeError, ValueError):
                continue

            timestamp = datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            rows.append(
                {
                    "timestamp": timestamp,
                    "pod": pod,
                    "metric": metric_name,
                    "value": value,
                    "labels": "",
                }
            )
    return rows


def query_kube_pod_metric(metric_name: str):
    rows = []
    if KUBE_POD_NAMESPACE:
        promql = f'{metric_name}{{namespace="{KUBE_POD_NAMESPACE}"}}'
    else:
        promql = metric_name

    for series in query_range(promql, KUBE_POD_STEP):
        metric_labels = series.get("metric", {})
        pod = metric_labels.get("pod")
        if not pod:
            continue

        labels_for_row = {k: v for k, v in metric_labels.items() if k not in {"__name__", "pod"}}
        labels_json = json.dumps(labels_for_row, sort_keys=True, ensure_ascii=False)

        for ts, val in series.get("values", []):
            try:
                value = float(val)
            except (TypeError, ValueError):
                continue

            timestamp = datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
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
    container_metrics = load_metric_list(CONTAINER_METRIC_LIST_FILE)
    kube_pod_metrics = load_metric_list(KUBE_POD_METRIC_LIST_FILE)

    print(f"Total container metrics: {len(container_metrics)}")
    print(f"Total kube_pod metrics: {len(kube_pod_metrics)}")

    all_rows = []

    for metric in container_metrics:
        print(f"Querying [container] {metric}...")
        try:
            all_rows.extend(query_container_metric(metric))
        except Exception as e:
            print(f"Failed [container] {metric} -> {e}")

    for metric in kube_pod_metrics:
        print(f"Querying [kube_pod] {metric}...")
        try:
            all_rows.extend(query_kube_pod_metric(metric))
        except Exception as e:
            print(f"Failed [kube_pod] {metric} -> {e}")

    result_df = pd.DataFrame(all_rows, columns=OUTPUT_COLUMNS)

    output_path = resolve_path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)

    print("Done")
    print(f"Saved -> {output_path}")
    print(f"Total rows -> {len(result_df)}")


if __name__ == "__main__":
    main()
