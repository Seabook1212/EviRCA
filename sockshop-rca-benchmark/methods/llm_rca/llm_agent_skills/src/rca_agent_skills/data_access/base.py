from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pandas import DataFrame

from rca_agent_skills.common.enums import BackendMode
from rca_agent_skills.data_access.csv_loader import CSVTelemetryLoader
from rca_agent_skills.data_access.jaeger_client import JaegerTraceClient
from rca_agent_skills.data_access.loki_client import LokiLogClient
from rca_agent_skills.data_access.prometheus_client import PrometheusMetricClient
from rca_agent_skills.data_access.topology_loader import load_topology


class DataAccess(Protocol):
    def get_metrics(self, window_name: str) -> DataFrame: ...
    def get_logs(self, window_name: str) -> DataFrame: ...
    def get_traces(self, window_name: str) -> DataFrame: ...
    def get_topology(self) -> dict: ...


@dataclass
class APIDataAccess:
    request: object
    settings: dict

    def get_metrics(self, window_name: str) -> DataFrame:
        return PrometheusMetricClient(self.request, self.settings).fetch_window(window_name)

    def get_logs(self, window_name: str) -> DataFrame:
        return LokiLogClient(self.request, self.settings).fetch_window(window_name)

    def get_traces(self, window_name: str) -> DataFrame:
        return JaegerTraceClient(self.request, self.settings).fetch_window(window_name)

    def get_topology(self) -> dict:
        return load_topology(self.request.topology, self.request.csv_inputs.topology_file if self.request.csv_inputs else None)


def build_data_access(request: object, settings: dict) -> DataAccess:
    if request.backend_mode == BackendMode.CSV.value:
        return CSVTelemetryLoader(request, settings)
    return APIDataAccess(request=request, settings=settings)

