from __future__ import annotations

from types import SimpleNamespace

from rca_agent_skills.data_access.loki_client import LokiLogClient


class FakeLimitedLokiClient(LokiLogClient):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.lines = [
            (1_000_000_000, "line-1"),
            (2_000_000_000, "line-2"),
            (3_000_000_000, "line-3"),
            (4_000_000_000, "line-4"),
            (5_000_000_000, "line-5"),
        ]

    def _query_range(self, query: str, start_ns: int, end_ns: int):
        limit = int(self._config().get("limit", 2))
        values = [[str(ts_ns), line] for ts_ns, line in self.lines if start_ns <= ts_ns <= end_ns]
        return [
            {
                "stream": {"pod": "orders-abc-1", "container": "orders", "node_name": "node-a"},
                "values": values[:limit],
            }
        ]


class FakeParsedLokiClient(LokiLogClient):
    def _query_range(self, query: str, start_ns: int, end_ns: int):
        return [
            {
                "stream": {"pod": "payment-abc-1", "container": "payment", "node_name": "node-a"},
                "values": [
                    [
                        "1000000000",
                        '{"level":"error","trace_id":"abc123","span_id":"def456","msg":"payment failed after retry"}',
                    ]
                ],
            }
        ]


def test_loki_client_splits_limit_saturated_ranges_until_all_logs_are_returned():
    request = SimpleNamespace(
        namespace="sock-shop",
        baseline_window=SimpleNamespace(start="1970-01-01T00:00:00Z", end="1970-01-01T00:00:06Z"),
        config_bundle={"loki_queries": {"base_query": '{namespace="{namespace}"}'}},
    )
    settings = {
        "debug": {"print_queries": False},
        "api": {
            "loki": {
                "base_url": "http://loki.example",
                "limit": 2,
                "slice_minutes": 5,
                "min_split_interval_ms": 1000,
            }
        },
    }

    frame = FakeLimitedLokiClient(request, settings).fetch_window("baseline")

    assert list(frame["message"]) == ["line-1", "line-2", "line-3", "line-4", "line-5"]


def test_loki_client_parses_raw_log_fields_from_api_lines():
    request = SimpleNamespace(
        namespace="sock-shop",
        baseline_window=SimpleNamespace(start="1970-01-01T00:00:00Z", end="1970-01-01T00:00:01Z"),
        config_bundle={"loki_queries": {"base_query": '{namespace="{namespace}"}'}},
    )
    settings = {
        "debug": {"print_queries": False},
        "api": {
            "loki": {
                "base_url": "http://loki.example",
                "limit": 5000,
                "slice_minutes": 5,
            }
        },
    }

    frame = FakeParsedLokiClient(request, settings).fetch_window("baseline")

    assert frame.iloc[0]["log_level"] == "ERROR"
    assert frame.iloc[0]["trace_id"] == "abc123"
    assert frame.iloc[0]["span_id"] == "def456"
    assert frame.iloc[0]["message"] == "payment failed after retry"
    assert frame.iloc[0]["log_type"] == "exception_log"
