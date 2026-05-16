from __future__ import annotations

from .enums import FaultType


DEFAULT_TOP_K = 5
DEFAULT_NAMESPACE = "sock-shop"
DEFAULT_MIN_SERIES_POINTS = 3

FIXED_FAULT_TYPES = [fault.value for fault in FaultType]
DEFAULT_METRIC_KPIS = [
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

NETWORK_KEYWORDS = [
    "timeout",
    "connection reset",
    "broken pipe",
    "connection refused",
    "unreachable",
    "disconnect",
    "partition",
    "reset by peer",
]
EXCEPTION_KEYWORDS = ["exception", "error", "panic", "oom", "outofmemory", "segfault", "failed"]
