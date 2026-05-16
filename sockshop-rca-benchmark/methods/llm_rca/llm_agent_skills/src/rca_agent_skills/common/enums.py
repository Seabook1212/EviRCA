from __future__ import annotations

from enum import Enum


class BackendMode(str, Enum):
    API = "api"
    CSV = "csv"


class Granularity(str, Enum):
    SERVICE = "service"
    POD = "pod"


class EvidenceSource(str, Enum):
    METRIC = "metric"
    LOG = "log"
    TRACE = "trace"


class FaultType(str, Enum):
    IO_FAULT = "io_fault"
    NETWORK_LOSS = "network_loss"
    NETWORK_DELAY = "network_delay"
    NETWORK_PARTITION = "network_partition"
    POD_FAILURE = "pod_failure"
    CPU_STRESS = "cpu_stress"
    MEMORY_STRESS = "memory_stress"
    EXCEPTION_INJECTION = "exception_injection"
