from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import time
from typing import Any

import pandas as pd
import requests
from requests import RequestException

from rca_agent_skills.common.logging_utils import get_logger
from rca_agent_skills.common.constants import DEFAULT_METRIC_KPIS
from rca_agent_skills.common.time_utils import to_prometheus_time
from rca_agent_skills.data_access.topology_loader import service_from_pod
from rca_agent_skills.query_expansion.renderer import render_template


@dataclass
class PrometheusMetricClient:
    request: object
    settings: dict

    def __post_init__(self) -> None:
        self.logger = get_logger(self.__class__.__name__)
        self.debug = self.settings.get("debug", {})

    def _config(self) -> dict[str, Any]:
        return self.settings.get("api", {}).get("prometheus", {})

    def _get(self, url: str, params: dict[str, Any]) -> requests.Response:
        cfg = self._config()
        retry_statuses = set(cfg.get("retry_statuses", [429, 500, 502, 503, 504]))
        max_retries = int(cfg.get("max_retries", 3))
        backoff_sec = float(cfg.get("retry_backoff_sec", 2.0))
        timeout_sec = int(cfg.get("timeout_sec", 120))
        last_error: Exception | None = None
        response = None
        for attempt in range(1, max(1, max_retries) + 1):
            try:
                response = requests.get(url, params=params, timeout=timeout_sec)
                if response.status_code not in retry_statuses:
                    response.raise_for_status()
                    return response
                last_error = requests.HTTPError(
                    f"{response.status_code} Server Error from Prometheus",
                    response=response,
                )
                if attempt >= max(1, max_retries):
                    response.raise_for_status()
            except RequestException as exc:
                last_error = exc
                if attempt >= max(1, max_retries):
                    raise
            sleep_sec = backoff_sec * attempt
            self.logger.warning(
                "Prometheus request failed on attempt %s/%s: %s; retrying in %.1fs",
                attempt,
                max(1, max_retries),
                last_error,
                sleep_sec,
            )
            time.sleep(max(0.0, sleep_sec))
        if response is None:
            raise RuntimeError(f"Prometheus request failed without response: {last_error}")
        return response

    def _query_range(self, expr: str, start: str, end: str, step: str) -> list[dict[str, Any]]:
        cfg = self._config()
        if self.debug.get("print_queries", True):
            self.logger.info("[PROMQL][RANGE] %s", expr)
        url = f"{cfg['base_url'].rstrip('/')}/api/v1/query_range"
        response = self._get(url, {"query": expr, "start": start, "end": end, "step": step})
        payload = response.json()
        if payload.get("status") != "success":
            return []
        return payload.get("data", {}).get("result", [])

    def _query_instant(self, expr: str, ts: str) -> list[dict[str, Any]]:
        cfg = self._config()
        if self.debug.get("print_queries", True):
            self.logger.info("[PROMQL][INSTANT] %s", expr)
        url = f"{cfg['base_url'].rstrip('/')}/api/v1/query"
        response = self._get(url, {"query": expr, "time": ts})
        payload = response.json()
        if payload.get("status") != "success":
            return []
        return payload.get("data", {}).get("result", [])

    def _list_pods(self, end_ts: str) -> list[str]:
        templates = self.request.config_bundle["prometheus_queries"]
        expr = render_template(
            templates["pod_discovery"],
            {"namespace": self.request.namespace},
        )
        result = self._query_instant(expr, end_ts)
        pods: list[str] = []
        for item in result:
            pod = item.get("metric", {}).get("pod")
            if pod:
                pods.append(pod)
        return list(dict.fromkeys(pods))

    def _is_service_scoped_template(self, template: str) -> bool:
        return "{service}" in template and "{pod}" not in template

    def fetch_window(self, window_name: str) -> pd.DataFrame:
        window = getattr(self.request, f"{window_name}_window")
        start = to_prometheus_time(window.start)
        end = to_prometheus_time(window.end)
        cfg = self._config()
        templates = self.request.config_bundle["prometheus_queries"]["kpi_candidates"]
        rows: list[dict[str, Any]] = []

        pods = self._list_pods(end)
        pods_by_service: dict[str, list[str]] = defaultdict(list)
        for pod in pods:
            pods_by_service[service_from_pod(pod)].append(pod)

        for service, service_pods in pods_by_service.items():
            for kpi in DEFAULT_METRIC_KPIS:
                candidates = templates.get(kpi, [])
                pod_templates = [template for template in candidates if not self._is_service_scoped_template(template)]
                service_templates = [template for template in candidates if self._is_service_scoped_template(template)]

                for pod in service_pods:
                    for template in pod_templates:
                        expr = render_template(
                            template,
                            {
                                "namespace": self.request.namespace,
                                "pod": pod,
                                "service": service,
                                "kpi_window": cfg.get("kpi_window", "30s"),
                                "istio_window": cfg.get("istio_window", "30s"),
                                "network_window": cfg.get("network_window", "1m"),
                                "restart_count_window": cfg.get("restart_count_window", "1m"),
                            },
                        )
                        try:
                            result = self._query_range(expr, start, end, cfg.get("step", "5s"))
                        except Exception:
                            result = []
                        if not result:
                            continue
                        for series in result:
                            labels = series.get("metric", {})
                            series_pod = labels.get("pod") or labels.get("destination_pod") or pod
                            row_service = service_from_pod(series_pod)
                            for ts, value in series.get("values", []):
                                try:
                                    numeric_value = float(value)
                                except (TypeError, ValueError):
                                    continue
                                rows.append(
                                    {
                                        "timestamp": pd.to_datetime(float(ts), unit="s", utc=True).isoformat(),
                                        "pod": series_pod,
                                        "service": row_service,
                                        "metric": kpi,
                                        "value": numeric_value,
                                    }
                                )
                        break

                for template in service_templates:
                    expr = render_template(
                        template,
                        {
                            "namespace": self.request.namespace,
                            "service": service,
                            "kpi_window": cfg.get("kpi_window", "30s"),
                            "istio_window": cfg.get("istio_window", "30s"),
                            "network_window": cfg.get("network_window", "1m"),
                            "restart_count_window": cfg.get("restart_count_window", "1m"),
                        },
                    )
                    try:
                        result = self._query_range(expr, start, end, cfg.get("step", "5s"))
                    except Exception:
                        result = []
                    if not result:
                        continue
                    for series in result:
                        labels = series.get("metric", {})
                        row_service = labels.get("destination_workload") or labels.get("workload") or service
                        for ts, value in series.get("values", []):
                            try:
                                numeric_value = float(value)
                            except (TypeError, ValueError):
                                continue
                            rows.append(
                                {
                                    "timestamp": pd.to_datetime(float(ts), unit="s", utc=True).isoformat(),
                                    "pod": None,
                                    "service": row_service,
                                    "metric": kpi,
                                    "value": numeric_value,
                                }
                            )
                    break
        return pd.DataFrame(rows, columns=["timestamp", "pod", "service", "metric", "value"])
