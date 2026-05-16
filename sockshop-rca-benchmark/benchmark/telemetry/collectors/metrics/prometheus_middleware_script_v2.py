import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests


PROM_URL = os.environ.get("PROM_URL", "http://34.28.33.102:30990")
START_TIME = os.environ.get("START_TIME", os.environ.get("PROM_START", "2026-02-22T00:00:00Z"))
END_TIME = os.environ.get("END_TIME", os.environ.get("PROM_END", "2026-02-22T00:15:00Z"))
STEP = os.environ.get("PROM_STEP", "5s")

DEFAULT_NAMESPACE = os.environ.get("MIDDLEWARE_NAMESPACE", "sock-shop").strip()
MONGODB_NAMESPACE = os.environ.get("MONGODB_NAMESPACE", DEFAULT_NAMESPACE).strip()
MYSQL_NAMESPACE = os.environ.get("MYSQL_NAMESPACE", DEFAULT_NAMESPACE).strip()
RABBITMQ_NAMESPACE = os.environ.get("RABBITMQ_NAMESPACE", DEFAULT_NAMESPACE).strip()
REDIS_NAMESPACE = os.environ.get("REDIS_NAMESPACE", DEFAULT_NAMESPACE).strip()

SCRIPT_DIR = Path(__file__).resolve().parent

MONGODB_METRIC_FILE = os.environ.get(
    "MONGODB_METRIC_LIST_FILE",
    str(SCRIPT_DIR / "data" / "applied" / "mongodb_level_applied_metrics_name.csv"),
)
MYSQL_METRIC_FILE = os.environ.get(
    "MYSQL_METRIC_LIST_FILE",
    str(SCRIPT_DIR / "data" / "applied" / "mysql_level_applied_metrics_name.csv"),
)
RABBITMQ_METRIC_FILE = os.environ.get(
    "RABBITMQ_METRIC_LIST_FILE",
    str(SCRIPT_DIR / "data" / "applied" / "rabbitmq_level_applied_metrics_name.csv"),
)
REDIS_METRIC_FILE = os.environ.get(
    "REDIS_METRIC_LIST_FILE",
    str(SCRIPT_DIR / "data" / "applied" / "redis_level_applied_metrics_name.csv"),
)

OUTPUT_FILE = os.environ.get(
    "MIDDLEWARE_METRIC_OUTPUT_FILE",
    str(SCRIPT_DIR / "data" / "demo_data" / "prometheus_metrics_middleware_raw.csv"),
)

MONGODB_TARGET_PODS = [
    pod.strip()
    for pod in os.environ.get("MONGODB_TARGET_PODS", "carts-db-0,user-db-0,orders-db-0").split(",")
    if pod.strip()
]
MYSQL_TARGET_PODS = [
    pod.strip()
    for pod in os.environ.get("MYSQL_TARGET_PODS", "catalogue-db-0").split(",")
    if pod.strip()
]
RABBITMQ_TARGET_PODS = [
    pod.strip()
    for pod in os.environ.get("RABBITMQ_TARGET_PODS", "rabbitmq-0").split(",")
    if pod.strip()
]
REDIS_TARGET_PODS = [
    pod.strip()
    for pod in os.environ.get("REDIS_TARGET_PODS", "session-db-0").split(",")
    if pod.strip()
]

OUTPUT_COLUMNS = ["timestamp", "pod", "metric", "value"]

SOURCE_CONFIGS = [
    {
        "name": "MongoDB",
        "metric_file": MONGODB_METRIC_FILE,
        "namespace": MONGODB_NAMESPACE,
        "pods": MONGODB_TARGET_PODS,
    },
    {
        "name": "MySQL",
        "metric_file": MYSQL_METRIC_FILE,
        "namespace": MYSQL_NAMESPACE,
        "pods": MYSQL_TARGET_PODS,
    },
    {
        "name": "RabbitMQ",
        "metric_file": RABBITMQ_METRIC_FILE,
        "namespace": RABBITMQ_NAMESPACE,
        "pods": RABBITMQ_TARGET_PODS,
    },
    {
        "name": "Redis",
        "metric_file": REDIS_METRIC_FILE,
        "namespace": REDIS_NAMESPACE,
        "pods": REDIS_TARGET_PODS,
    },
]


def parse_iso_to_utc(text: str) -> datetime:
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


start_time = parse_iso_to_utc(START_TIME)
end_time = parse_iso_to_utc(END_TIME)


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


def build_promql(metric_name: str, namespace: str, pod: str) -> str:
    filters = []
    if namespace:
        filters.append(f'namespace="{namespace}"')
    if pod:
        filters.append(f'pod="{pod}"')
    if filters:
        return f'{metric_name}{{{",".join(filters)}}}'
    return metric_name


def query_metric_for_pod(metric_name: str, namespace: str, pod: str) -> list[dict]:
    promql = build_promql(metric_name, namespace, pod)
    rows = []

    for series in query_range(promql, STEP):
        labels = series.get("metric", {})
        row_pod = labels.get("pod", pod or "unknown")
        if pod and row_pod and row_pod != pod:
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
                    "pod": row_pod,
                    "metric": metric_name,
                    "value": value,
                }
            )

    return rows


def main() -> None:
    all_rows = []

    for source in SOURCE_CONFIGS:
        source_name = source["name"]
        source_pods = source["pods"]
        source_namespace = source["namespace"]
        if not source_pods:
            print(f"Skip {source_name}: no target pods configured")
            continue

        metrics = load_metric_list(source["metric_file"])
        print(f"Loaded {len(metrics)} {source_name} metrics")

        for metric in metrics:
            print(f"Querying [{source_name}] {metric} ...")
            for pod in source_pods:
                try:
                    all_rows.extend(query_metric_for_pod(metric, source_namespace, pod))
                except Exception as e:
                    print(f"Failed [{source_name}] {metric} (pod={pod}) -> {e}")

    result_df = pd.DataFrame(all_rows, columns=OUTPUT_COLUMNS)

    output_path = resolve_path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)

    print("\nDone")
    print(f"Saved -> {output_path}")
    print(f"Total rows -> {len(result_df)}")


if __name__ == "__main__":
    main()
