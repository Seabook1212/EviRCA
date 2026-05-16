from __future__ import annotations

from types import SimpleNamespace

from rca_agent_skills.data_access.prometheus_client import PrometheusMetricClient


class FakeMixedPrometheusMetricClient(PrometheusMetricClient):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.queries: list[str] = []

    def _list_pods(self, end_ts: str) -> list[str]:
        return ["catalogue-58bdd4d4f9-48jpd", "catalogue-58bdd4d4f9-sld8g"]

    def _query_range(self, expr: str, start: str, end: str, step: str):
        self.queries.append(expr)
        if 'destination_pod="catalogue-58bdd4d4f9-48jpd"' in expr:
            return [
                {
                    "metric": {},
                    "values": [[1_000.0, "120.0"]],
                }
            ]
        if 'destination_pod="catalogue-58bdd4d4f9-sld8g"' in expr:
            return []
        if 'destination_workload="catalogue"' in expr:
            return [
                {
                    "metric": {},
                    "values": [[1_000.0, "80.0"]],
                }
            ]
        return []


def test_service_scoped_prometheus_query_runs_alongside_pod_queries_without_pod_stamping():
    request = SimpleNamespace(
        namespace="sock-shop",
        baseline_window=SimpleNamespace(start="1970-01-01T00:00:00Z", end="1970-01-01T00:01:00Z"),
        config_bundle={
            "prometheus_queries": {
                "kpi_candidates": {
                    "latency_p99": [
                        'histogram_quantile(0.99, sum by (le) (rate(metric{destination_pod="{pod}"}[30s])))',
                        'histogram_quantile(0.99, sum by (le) (rate(metric{destination_workload="{service}"}[30s])))',
                    ]
                }
            }
        },
    )
    settings = {
        "debug": {"print_queries": False},
        "api": {"prometheus": {"base_url": "http://prometheus.example", "step": "5s"}},
    }

    frame = FakeMixedPrometheusMetricClient(request, settings).fetch_window("baseline")

    pod_rows = frame[frame["pod"].notna()].to_dict("records")
    service_rows = frame[frame["pod"].isna()].to_dict("records")

    assert len(pod_rows) == 1
    assert pod_rows[0]["pod"] == "catalogue-58bdd4d4f9-48jpd"
    assert pod_rows[0]["value"] == 120.0
    assert len(service_rows) == 1
    assert service_rows[0]["service"] == "catalogue"
    assert service_rows[0]["value"] == 80.0
    assert not any(row.get("pod") == "catalogue-58bdd4d4f9-sld8g" for row in frame.to_dict("records"))


class FakeServiceFallbackPrometheusMetricClient(FakeMixedPrometheusMetricClient):
    def _query_range(self, expr: str, start: str, end: str, step: str):
        self.queries.append(expr)
        if "destination_pod=" in expr:
            return []
        if 'destination_workload="catalogue"' in expr:
            return [
                {
                    "metric": {},
                    "values": [[1_000.0, "80.0"]],
                }
            ]
        return []


class FakePodLabelFallbackPrometheusMetricClient(FakeMixedPrometheusMetricClient):
    def _query_range(self, expr: str, start: str, end: str, step: str):
        self.queries.append(expr)
        if "destination_pod=" in expr:
            return []
        if 'pod="catalogue-58bdd4d4f9-48jpd"' in expr:
            return [
                {
                    "metric": {"pod": "catalogue-58bdd4d4f9-48jpd"},
                    "values": [[1_000.0, "120.0"]],
                }
            ]
        return []


def test_prometheus_pod_templates_try_plain_pod_label_after_destination_pod_label():
    request = SimpleNamespace(
        namespace="sock-shop",
        baseline_window=SimpleNamespace(start="1970-01-01T00:00:00Z", end="1970-01-01T00:01:00Z"),
        config_bundle={
            "prometheus_queries": {
                "kpi_candidates": {
                    "latency_p99": [
                        'histogram_quantile(0.99, sum by (destination_pod, le) (rate(metric{destination_pod="{pod}"}[30s])))',
                        'histogram_quantile(0.99, sum by (pod, le) (rate(metric{pod="{pod}"}[30s])))',
                    ]
                }
            }
        },
    )
    settings = {
        "debug": {"print_queries": False},
        "api": {"prometheus": {"base_url": "http://prometheus.example", "step": "5s"}},
    }

    client = FakePodLabelFallbackPrometheusMetricClient(request, settings)
    frame = client.fetch_window("baseline")

    assert any('destination_pod="catalogue-58bdd4d4f9-48jpd"' in query for query in client.queries)
    assert any('pod="catalogue-58bdd4d4f9-48jpd"' in query for query in client.queries)
    assert frame.to_dict("records") == [
        {
            "timestamp": "1970-01-01T00:16:40+00:00",
            "pod": "catalogue-58bdd4d4f9-48jpd",
            "service": "catalogue",
            "metric": "latency_p99",
            "value": 120.0,
        }
    ]


def test_service_scoped_prometheus_query_is_used_when_no_pod_query_has_data():
    request = SimpleNamespace(
        namespace="sock-shop",
        baseline_window=SimpleNamespace(start="1970-01-01T00:00:00Z", end="1970-01-01T00:01:00Z"),
        config_bundle={
            "prometheus_queries": {
                "kpi_candidates": {
                    "latency_p99": [
                        'histogram_quantile(0.99, sum by (le) (rate(metric{destination_pod="{pod}"}[30s])))',
                        'histogram_quantile(0.99, sum by (le) (rate(metric{destination_workload="{service}"}[30s])))',
                    ]
                }
            }
        },
    )
    settings = {
        "debug": {"print_queries": False},
        "api": {"prometheus": {"base_url": "http://prometheus.example", "step": "5s"}},
    }

    frame = FakeServiceFallbackPrometheusMetricClient(request, settings).fetch_window("baseline")

    assert frame.to_dict("records") == [
        {
            "timestamp": "1970-01-01T00:16:40+00:00",
            "pod": None,
            "service": "catalogue",
            "metric": "latency_p99",
            "value": 80.0,
        }
    ]
