import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

# =========================
# Config (env overridable)
# =========================
PROM_URL = os.environ.get("PROM_URL", "http://34.28.33.102:30990")
NAMESPACE = os.environ.get("PROM_NAMESPACE", "sock-shop")

# Input time range (RFC3339 or Prometheus-acceptable time string)
START_TIME = os.environ.get("START_TIME", os.environ.get("PROM_START", "2026-03-07T14:10:26Z"))
END_TIME = os.environ.get("END_TIME", os.environ.get("PROM_END", "2026-03-07T14:25:26Z"))

# Step for query_range (e.g., 5s / 30s / 1m)
STEP = os.environ.get("PROM_STEP", "5s")

# KPI window (used by rate/increase/avg_over_time/histogram_quantile inputs)
KPI_WINDOW = os.environ.get("KPI_WINDOW", "30s")
NETWORK_KPI_WINDOW = os.environ.get("NETWORK_KPI_WINDOW", "1m")  # separate window for network metrics if desired
RESTART_COUNT_WINDOW = os.environ.get("RESTART_COUNT_WINDOW", "1m")  # separate window for sparse events
ISTIO_WINDOW = os.environ.get("ISTIO_WINDOW", "30s")

OUTPUT_FILE = os.environ.get("KPI_OUTPUT_FILE", "./data/demo_data/prometheus_metrics_KPI.csv")

TIMEOUT_SEC = int(os.environ.get("PROM_TIMEOUT_SEC", "120"))

# =========================
# Prometheus HTTP helpers
# =========================
def prom_query(expr: str, ts: Optional[str] = None) -> List[Dict[str, Any]]:
    """Instant query. If ts is None, Prom uses 'now'."""
    url = f"{PROM_URL}/api/v1/query"
    params = {"query": expr}
    if ts is not None:
        params["time"] = ts
    r = requests.get(url, params=params, timeout=TIMEOUT_SEC)
    r.raise_for_status()
    data = r.json()
    if data.get("status") != "success":
        return []
    return data.get("data", {}).get("result", [])


def prom_query_range(expr: str) -> List[Dict[str, Any]]:
    url = f"{PROM_URL}/api/v1/query_range"
    params = {
        "query": expr,
        "start": START_TIME,
        "end": END_TIME,
        "step": STEP,
    }
    r = requests.get(url, params=params, timeout=TIMEOUT_SEC)
    r.raise_for_status()
    data = r.json()
    if data.get("status") != "success":
        return []
    return data.get("data", {}).get("result", [])


def ts_to_utc_str(ts_float: float) -> str:
    return datetime.fromtimestamp(ts_float, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


# =========================
# Pod discovery
# =========================
def list_pods(namespace: str) -> List[str]:
    """
    Discover pods via kube_pod_info at END_TIME.
    Filters out common noise pods if desired; customize as needed.
    """
    expr = f'kube_pod_info{{namespace="{namespace}"}}'
    series = prom_query(expr, ts=END_TIME)

    pods: List[str] = []
    for s in series:
        labels = s.get("metric", {})
        pod = labels.get("pod")
        if not pod:
            continue
        # optional filters (keep them minimal)
        if pod.startswith("prometheus") or pod.startswith("grafana"):
            continue
        pods.append(pod)

    # de-dup preserving order
    seen = set()
    out = []
    for p in pods:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


# =========================
# KPI PromQL builders (with fallbacks)
# =========================
def kpi_promql_candidates(kpi: str, pod: str, namespace: str) -> List[str]:
    """
    Return candidate PromQL expressions for given KPI and pod.
    We try multiple label conventions (destination_pod / pod / destination_workload) for Istio.
    """
    # Common selectors
    cpu_sel = (
        f'namespace="{namespace}",pod="{pod}",container!="POD",container!="istio-proxy"'
    )

    # kube-state-metrics resource limits (cpu cores / memory bytes)
    cpu_limit_sel = (
        f'namespace="{namespace}",pod="{pod}",resource="cpu",unit="core",container!="POD",container!="istio-proxy"'
    )
    mem_limit_sel = (
        f'namespace="{namespace}",pod="{pod}",resource="memory",unit="byte",container!="POD",container!="istio-proxy"'
    )
    # Some kube-state-metrics setups may not expose `unit`, so keep no-unit fallbacks.
    cpu_limit_sel_no_unit = (
        f'namespace="{namespace}",pod="{pod}",resource="cpu",container!="POD",container!="istio-proxy"'
    )
    mem_limit_sel_no_unit = (
        f'namespace="{namespace}",pod="{pod}",resource="memory",container!="POD",container!="istio-proxy"'
    )
    cpu_req_sel = (
        f'namespace="{namespace}",pod="{pod}",resource="cpu",unit="core",container!="POD",container!="istio-proxy"'
    )
    mem_req_sel = (
        f'namespace="{namespace}",pod="{pod}",resource="memory",unit="byte",container!="POD",container!="istio-proxy"'
    )
    cpu_req_sel_no_unit = (
        f'namespace="{namespace}",pod="{pod}",resource="cpu",container!="POD",container!="istio-proxy"'
    )
    mem_req_sel_no_unit = (
        f'namespace="{namespace}",pod="{pod}",resource="memory",container!="POD",container!="istio-proxy"'
    )
    pod_info_sel = f'namespace="{namespace}",pod="{pod}"'

    # Istio selectors (try different label variants)
    istio_base = f'reporter="destination",destination_workload_namespace="{namespace}"'
    # some setups expose destination_pod; some expose pod; some only destination_workload
    istio_by_dest_pod = f'{istio_base},destination_pod="{pod}"'
    istio_by_pod = f'{istio_base},pod="{pod}"'
    # derive workload name from pod prefix (best-effort): "orders-xxxxx" -> "orders"
    workload_guess = pod.split("-")[0] if "-" in pod else pod
    istio_by_workload = f'{istio_base},destination_workload="{workload_guess}"'

    if kpi == "request_rate":
        return [
            f'sum by (pod) (rate(istio_requests_total{{{istio_by_dest_pod}}}[{ISTIO_WINDOW}]))',
            f'sum by (pod) (rate(istio_requests_total{{{istio_by_pod}}}[{ISTIO_WINDOW}]))',
            # if only destination_workload exists, we still emit pod column as the real pod name (handled in parsing)
            f'sum(rate(istio_requests_total{{{istio_by_workload}}}[{ISTIO_WINDOW}]))',
        ]

    if kpi == "success_rate":
        # Define "success" as 2xx/3xx/304/202 etc. You can tighten this if needed.
        success_re = 'response_code=~"200|201|202|204|300|301|302|304"'
        return [
            f'(sum(rate(istio_requests_total{{{istio_by_dest_pod},{success_re}}}[{ISTIO_WINDOW}]))'
            f' / sum(rate(istio_requests_total{{{istio_by_dest_pod}}}[{ISTIO_WINDOW}]))) * 100',
            f'(sum(rate(istio_requests_total{{{istio_by_pod},{success_re}}}[{ISTIO_WINDOW}]))'
            f' / sum(rate(istio_requests_total{{{istio_by_pod}}}[{ISTIO_WINDOW}]))) * 100',
            f'(sum(rate(istio_requests_total{{{istio_by_workload},{success_re}}}[{ISTIO_WINDOW}]))'
            f' / sum(rate(istio_requests_total{{{istio_by_workload}}}[{ISTIO_WINDOW}]))) * 100',
        ]

    if kpi == "error_count":
        # "error" as 5xx; you may only want 5xx depending on your fault definition
        err_re = 'response_code=~"5.."'
        err_by_dest_pod = f'sum(increase(istio_requests_total{{{istio_by_dest_pod},{err_re}}}[{KPI_WINDOW}]))'
        all_by_dest_pod = f'sum(increase(istio_requests_total{{{istio_by_dest_pod}}}[{KPI_WINDOW}]))'
        err_by_pod = f'sum(increase(istio_requests_total{{{istio_by_pod},{err_re}}}[{KPI_WINDOW}]))'
        all_by_pod = f'sum(increase(istio_requests_total{{{istio_by_pod}}}[{KPI_WINDOW}]))'
        err_by_workload = f'sum(increase(istio_requests_total{{{istio_by_workload},{err_re}}}[{KPI_WINDOW}]))'
        all_by_workload = f'sum(increase(istio_requests_total{{{istio_by_workload}}}[{KPI_WINDOW}]))'
        return [
            f'({err_by_dest_pod}) or (0 * ({all_by_dest_pod}))',
            f'({err_by_pod}) or (0 * ({all_by_pod}))',
            f'({err_by_workload}) or (0 * ({all_by_workload}))',
        ]

    if kpi.startswith("latency_p"):
        # Istio histogram is in milliseconds
        q = kpi.replace("latency_p", "")
        quantile = float(q) / 100.0

        def hist(expr_sel: str) -> str:
            return (
                f'histogram_quantile({quantile}, '
                f'sum by (le) (rate(istio_request_duration_milliseconds_bucket{{{expr_sel}}}[{ISTIO_WINDOW}])))'
            )

        return [
            hist(istio_by_dest_pod),
            hist(istio_by_pod),
            hist(istio_by_workload),
        ]

    if kpi == "cpu_usage_pct":
        # cpu usage cores / cpu limits(or requests fallback) * 100
        usage = f'sum by (pod) (rate(container_cpu_usage_seconds_total{{{cpu_sel}}}[{KPI_WINDOW}]))'
        limit = f'sum by (pod) (kube_pod_container_resource_limits{{{cpu_limit_sel}}})'
        limit_no_unit = f'sum by (pod) (kube_pod_container_resource_limits{{{cpu_limit_sel_no_unit}}})'
        req = f'sum by (pod) (kube_pod_container_resource_requests{{{cpu_req_sel}}})'
        req_no_unit = f'sum by (pod) (kube_pod_container_resource_requests{{{cpu_req_sel_no_unit}}})'
        pod_node = f'max by (pod, node) (kube_pod_info{{{pod_info_sel}}})'
        node_alloc = f'max by (node) (kube_node_status_allocatable{{resource="cpu",unit="core"}})'
        node_alloc_no_unit = f'max by (node) (kube_node_status_allocatable{{resource="cpu"}})'
        node_based = f'sum by (pod) (({pod_node}) * on (node) group_left {node_alloc})'
        node_based_no_unit = f'sum by (pod) (({pod_node}) * on (node) group_left {node_alloc_no_unit})'
        return [
            f'({usage} / clamp_min({limit}, 0.001)) * 100',
            f'({usage} / clamp_min({limit_no_unit}, 0.001)) * 100',
            f'({usage} / clamp_min({req}, 0.001)) * 100',
            f'({usage} / clamp_min({req_no_unit}, 0.001)) * 100',
            f'({usage} / clamp_min({node_based}, 0.001)) * 100',
            f'({usage} / clamp_min({node_based_no_unit}, 0.001)) * 100',
        ]

    if kpi == "memory_usage_pct":
        # memory working set / memory limits(or requests fallback) * 100
        usage = f'sum by (pod) (container_memory_working_set_bytes{{{cpu_sel}}})'
        limit = f'sum by (pod) (kube_pod_container_resource_limits{{{mem_limit_sel}}})'
        limit_no_unit = f'sum by (pod) (kube_pod_container_resource_limits{{{mem_limit_sel_no_unit}}})'
        req = f'sum by (pod) (kube_pod_container_resource_requests{{{mem_req_sel}}})'
        req_no_unit = f'sum by (pod) (kube_pod_container_resource_requests{{{mem_req_sel_no_unit}}})'
        pod_node = f'max by (pod, node) (kube_pod_info{{{pod_info_sel}}})'
        node_alloc = f'max by (node) (kube_node_status_allocatable{{resource="memory",unit="byte"}})'
        node_alloc_no_unit = f'max by (node) (kube_node_status_allocatable{{resource="memory"}})'
        node_based = f'sum by (pod) (({pod_node}) * on (node) group_left {node_alloc})'
        node_based_no_unit = f'sum by (pod) (({pod_node}) * on (node) group_left {node_alloc_no_unit})'
        return [
            f'({usage} / clamp_min({limit}, 1)) * 100',
            f'({usage} / clamp_min({limit_no_unit}, 1)) * 100',
            f'({usage} / clamp_min({req}, 1)) * 100',
            f'({usage} / clamp_min({req_no_unit}, 1)) * 100',
            f'({usage} / clamp_min({node_based}, 1)) * 100',
            f'({usage} / clamp_min({node_based_no_unit}, 1)) * 100',
        ]

    if kpi == "restart_count":
        # number of restarts in the window (per pod) – more stable than rate() for sparse events
        return [
            f'sum(increase(kube_pod_container_status_restarts_total{{namespace="{namespace}",pod="{pod}"}}[{RESTART_COUNT_WINDOW}]))'
        ]

    if kpi == "ready_ratio":
        # Average readiness in the window (0..1)
        return [
            f'avg_over_time(kube_pod_status_ready{{namespace="{namespace}",pod="{pod}",condition="true"}}[{KPI_WINDOW}])'
        ]

    if kpi == "network_rx":
        return [
            f'sum by (pod) (rate(container_network_receive_bytes_total{{namespace="{namespace}",pod="{pod}"}}[{NETWORK_KPI_WINDOW}]))'
        ]

    if kpi == "network_tx":
        return [
            f'sum by (pod) (rate(container_network_transmit_bytes_total{{namespace="{namespace}",pod="{pod}"}}[{NETWORK_KPI_WINDOW}]))'
        ]

    raise ValueError(f"Unknown KPI: {kpi}")


# =========================
# Extraction / Export
# =========================
KPIS = [
    "request_rate",
    "success_rate",
    "error_count",
    "latency_p50",
    "latency_p90",
    "latency_p95",
    "latency_p99",
    "cpu_usage_pct",
    "memory_usage_pct",
    "restart_count",
    "ready_ratio",
    "network_rx",
    "network_tx",
]


def run_one_kpi(pod: str, kpi: str) -> List[Dict[str, Any]]:
    """
    Try candidate queries in order; first non-empty result wins.
    Output rows: timestamp, pod, metric, value
    """
    candidates = kpi_promql_candidates(kpi, pod, NAMESPACE)

    last_err: Optional[str] = None
    for i, expr in enumerate(candidates, start=1):
        try:
            results = prom_query_range(expr)
        except Exception as e:
            last_err = str(e)
            continue

        if not results:
            continue

        rows: List[Dict[str, Any]] = []
        for series in results:
            labels = series.get("metric", {})
            values = series.get("values", [])

            # Determine row pod name:
            # - prefer explicit pod label
            # - else fallback to input `pod` (for workload-level queries)
            row_pod = labels.get("pod") or labels.get("destination_pod") or pod

            for ts, val in values:
                try:
                    v = float(val)
                except (TypeError, ValueError):
                    continue
                rows.append(
                    {
                        "timestamp": ts_to_utc_str(float(ts)),
                        "pod": row_pod,
                        "metric": kpi,
                        "value": v,
                    }
                )

        if rows:
            print(f"[OK] {kpi} pod={pod} (candidate {i}/{len(candidates)})")
            return rows

    if last_err:
        print(f"[WARN] {kpi} pod={pod} failed: {last_err}")
    else:
        print(f"[WARN] {kpi} pod={pod} returned empty (all candidates)")
    return []


def main() -> None:
    print("=== KPI Collection (pod-level, long format) ===")
    print(f"PROM_URL   : {PROM_URL}")
    print(f"NAMESPACE  : {NAMESPACE}")
    print(f"START_TIME : {START_TIME}")
    print(f"END_TIME   : {END_TIME}")
    print(f"STEP       : {STEP}")
    print(f"KPI_WINDOW : {KPI_WINDOW}")
    print(f"ISTIO_WINDOW: {ISTIO_WINDOW}")
    print()

    pods = list_pods(NAMESPACE)
    print(f"Discovered pods: {len(pods)}")
    for p in pods[:20]:
        print(f"  - {p}")
    if len(pods) > 20:
        print("  ...")
    print()

    all_rows: List[Dict[str, Any]] = []
    for pod in pods:
        for kpi in KPIS:
            all_rows.extend(run_one_kpi(pod, kpi))

    df = pd.DataFrame(all_rows, columns=["timestamp", "pod", "metric", "value"])
    df.sort_values(["timestamp", "pod", "metric"], inplace=True)

    out_path = Path(OUTPUT_FILE).expanduser()
    if not out_path.is_absolute():
        out_path = (Path(__file__).resolve().parent / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_path, index=False)
    print("\n=== Export Completed ===")
    print(f"Output file: {out_path}")
    print(f"Total rows : {len(df)}")


if __name__ == "__main__":
    main()
