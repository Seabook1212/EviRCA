from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from rca_agent_skills.common.time_utils import parse_time
from rca_agent_skills.data_access.topology_loader import load_topology


@dataclass
class CSVTelemetryLoader:
    request: object
    settings: dict

    def _filter_window(self, frame: pd.DataFrame, ts_column: str, window_name: str) -> pd.DataFrame:
        window = getattr(self.request, f"{window_name}_window")
        if frame.empty or ts_column not in frame.columns:
            return frame.copy()
        start = parse_time(window.start)
        end = parse_time(window.end)
        raw = frame[ts_column]
        if raw.dtype.kind in {"i", "u", "f"}:
            numeric = pd.to_numeric(raw, errors="coerce")
            if numeric.dropna().empty:
                series = pd.to_datetime(raw, utc=True, errors="coerce")
            else:
                max_value = float(numeric.dropna().abs().max())
                if max_value > 1e14:
                    series = pd.to_datetime(numeric, unit="us", utc=True, errors="coerce")
                elif max_value > 1e11:
                    series = pd.to_datetime(numeric, unit="ms", utc=True, errors="coerce")
                else:
                    series = pd.to_datetime(numeric, unit="s", utc=True, errors="coerce")
        else:
            series = pd.to_datetime(raw, utc=True, errors="coerce")
        mask = (series >= start) & (series <= end)
        return frame.loc[mask].copy()

    def _read_csv(self, path: str | None) -> pd.DataFrame:
        if not path:
            return pd.DataFrame()
        csv_path = Path(path).expanduser()
        if not csv_path.exists():
            return pd.DataFrame()
        return pd.read_csv(csv_path)

    def get_metrics(self, window_name: str) -> pd.DataFrame:
        df = self._read_csv(self.request.csv_inputs.metrics_csv if self.request.csv_inputs else None)
        return self._filter_window(df, "timestamp", window_name)

    def get_logs(self, window_name: str) -> pd.DataFrame:
        df = self._read_csv(self.request.csv_inputs.logs_csv if self.request.csv_inputs else None)
        return self._filter_window(df, "timestamp", window_name)

    def get_traces(self, window_name: str) -> pd.DataFrame:
        df = self._read_csv(self.request.csv_inputs.traces_csv if self.request.csv_inputs else None)
        if "timestamp" not in df.columns and "start_time" in df.columns:
            df = df.rename(columns={"start_time": "timestamp"})
        return self._filter_window(df, "timestamp", window_name)

    def get_topology(self) -> dict:
        return load_topology(self.request.topology, self.request.csv_inputs.topology_file if self.request.csv_inputs else None)
