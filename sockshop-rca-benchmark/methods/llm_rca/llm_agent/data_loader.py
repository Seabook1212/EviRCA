"""
README
- Purpose: Load topology + metrics/logs/traces from folder inputs for RCA.
- Data assumptions:
  - Metrics CSV schema like prometheus_pod_metrics_*.csv (has `service`, `metric`, `value`, `timestamp`).
  - Logs CSV schema like loki_logs_*.csv (has `timestamp`, `log`, optional `app`/`service`).
  - Traces CSV schema like jaeger_traces_*.csv (has `service`, `start_time`, `duration_us`, `tags_json`).
- Pod-level files are merged by service name automatically.
- Includes helper to read observability endpoint URLs from existing scripts.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import networkx as nx
import pandas as pd


@dataclass
class WindowSpec:
    start: Optional[str] = None
    end: Optional[str] = None


@dataclass
class RCADataBundle:
    normal_metrics: pd.DataFrame
    abnormal_metrics: pd.DataFrame
    normal_logs: pd.DataFrame
    abnormal_logs: pd.DataFrame
    normal_traces: pd.DataFrame
    abnormal_traces: pd.DataFrame
    topology_graph: nx.DiGraph
    endpoints: Dict[str, str]


class DataLoader:
    """Load and normalize inputs for the RCA pipeline."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

    def load_all(self) -> RCADataBundle:
        topology_graph = self.load_topology_graph(self.config["topology_path"])

        normal_metrics = self.load_metrics_folder(
            self.config["normal_metrics_path"],
            WindowSpec(
                self.config.get("normal_start_timestamp"),
                self.config.get("normal_end_timestamp"),
            ),
        )
        abnormal_metrics = self.load_metrics_folder(
            self.config["abnormal_metrics_path"],
            WindowSpec(
                self.config.get("abnormal_start_timestamp"),
                self.config.get("abnormal_end_timestamp"),
            ),
        )

        normal_logs = self.load_logs_folder(
            self.config["normal_logs_path"],
            WindowSpec(
                self.config.get("normal_start_timestamp"),
                self.config.get("normal_end_timestamp"),
            ),
        )
        abnormal_logs = self.load_logs_folder(
            self.config["abnormal_logs_path"],
            WindowSpec(
                self.config.get("abnormal_start_timestamp"),
                self.config.get("abnormal_end_timestamp"),
            ),
        )

        normal_traces = self.load_traces_folder(
            self.config["normal_traces_path"],
            WindowSpec(
                self.config.get("normal_start_timestamp"),
                self.config.get("normal_end_timestamp"),
            ),
        )
        abnormal_traces = self.load_traces_folder(
            self.config["abnormal_traces_path"],
            WindowSpec(
                self.config.get("abnormal_start_timestamp"),
                self.config.get("abnormal_end_timestamp"),
            ),
        )

        # Keep only known services from topology to avoid invalid labels like "nan"
        # and non-service data (e.g., node-level metrics files) in service RCA.
        normal_metrics = self._filter_to_topology_services(normal_metrics, topology_graph)
        abnormal_metrics = self._filter_to_topology_services(abnormal_metrics, topology_graph)
        normal_logs = self._filter_to_topology_services(normal_logs, topology_graph)
        abnormal_logs = self._filter_to_topology_services(abnormal_logs, topology_graph)
        normal_traces = self._filter_to_topology_services(normal_traces, topology_graph)
        abnormal_traces = self._filter_to_topology_services(abnormal_traces, topology_graph)

        endpoints = self.load_observability_endpoints()

        return RCADataBundle(
            normal_metrics=normal_metrics,
            abnormal_metrics=abnormal_metrics,
            normal_logs=normal_logs,
            abnormal_logs=abnormal_logs,
            normal_traces=normal_traces,
            abnormal_traces=abnormal_traces,
            topology_graph=topology_graph,
            endpoints=endpoints,
        )

    @staticmethod
    def load_topology_graph(topology_path: str) -> nx.DiGraph:
        path = Path(topology_path)
        if not path.exists():
            raise FileNotFoundError(f"Topology file not found: {topology_path}")
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        graph = nx.DiGraph()
        services = data.get("services", [])
        if not isinstance(services, list):
            raise ValueError("Invalid topology JSON format: expected `services` list")

        for entry in services:
            name = entry.get("name")
            depends_on = entry.get("depends_on", [])
            if not name:
                continue
            graph.add_node(name)
            for dep in depends_on:
                graph.add_node(dep)
                graph.add_edge(name, dep)  # service -> dependency
        return graph

    @staticmethod
    def _read_csv_files(folder: str) -> pd.DataFrame:
        folder_path = Path(folder)
        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder}")

        csv_files = sorted(folder_path.glob("*.csv"))
        if not csv_files:
            return pd.DataFrame()

        frames = []
        for file in csv_files:
            try:
                df = pd.read_csv(file)
                df["__source_file__"] = file.name
                frames.append(df)
            except Exception as exc:
                raise RuntimeError(f"Failed to read CSV `{file}`: {exc}") from exc

        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    @staticmethod
    def _infer_service_from_filename(filename: str) -> str:
        stem = Path(filename).stem
        stem = re.sub(r"^(loki_logs_|prometheus_pod_metrics_|jaeger_traces_)", "", stem)
        pod_like = stem.strip()
        if not pod_like:
            return "unknown"

        parts = pod_like.split("-")
        if len(parts) >= 3 and re.fullmatch(r"[a-f0-9]{8,}", parts[-2]) and re.fullmatch(r"[a-z0-9]{4,}", parts[-1]):
            return "-".join(parts[:-2])
        return pod_like

    @staticmethod
    def _ensure_service_column(df: pd.DataFrame, preferred_cols: Iterable[str]) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()

        for col in preferred_cols:
            if col in df.columns:
                # Keep raw values first, so NaN stays NaN (not "nan" string).
                df["service"] = df[col]
                break

        if "service" not in df.columns:
            if "__source_file__" in df.columns:
                df["service"] = df["__source_file__"].astype(str).apply(DataLoader._infer_service_from_filename)
            else:
                df["service"] = "unknown"

        invalid_mask = (
            df["service"].isna()
            | df["service"].astype(str).str.strip().eq("")
            | df["service"].astype(str).str.lower().isin({"nan", "none", "null"})
        )
        if invalid_mask.any() and "__source_file__" in df.columns:
            df.loc[invalid_mask, "service"] = df.loc[invalid_mask, "__source_file__"].astype(str).apply(
                DataLoader._infer_service_from_filename
            )

        # Final cleanup pass.
        invalid_mask = (
            df["service"].isna()
            | df["service"].astype(str).str.strip().eq("")
            | df["service"].astype(str).str.lower().isin({"nan", "none", "null"})
        )
        df.loc[invalid_mask, "service"] = "unknown"
        df["service"] = df["service"].astype(str).str.strip()
        return df

    @staticmethod
    def _filter_to_topology_services(df: pd.DataFrame, graph: nx.DiGraph) -> pd.DataFrame:
        if df.empty or "service" not in df.columns:
            return df
        allowed = set(graph.nodes())
        return df[df["service"].isin(allowed)].copy()

    @staticmethod
    def _parse_and_filter_time(
        df: pd.DataFrame,
        time_col: str,
        window: WindowSpec,
    ) -> pd.DataFrame:
        if df.empty or time_col not in df.columns:
            return df
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
        df = df.dropna(subset=[time_col])

        if window.start:
            start = pd.to_datetime(window.start, utc=True, errors="coerce")
            if pd.notna(start):
                df = df[df[time_col] >= start]
        if window.end:
            end = pd.to_datetime(window.end, utc=True, errors="coerce")
            if pd.notna(end):
                df = df[df[time_col] <= end]
        return df

    def load_metrics_folder(self, folder: str, window: WindowSpec) -> pd.DataFrame:
        df = self._read_csv_files(folder)
        if df.empty:
            return df

        df = self._ensure_service_column(df, preferred_cols=("service", "app"))
        if "timestamp" in df.columns:
            df = self._parse_and_filter_time(df, "timestamp", window)

        if "metric" not in df.columns:
            raise ValueError(f"Metrics data in `{folder}` must contain `metric` column")
        if "value" not in df.columns:
            raise ValueError(f"Metrics data in `{folder}` must contain `value` column")

        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"])
        return df

    def load_logs_folder(self, folder: str, window: WindowSpec) -> pd.DataFrame:
        df = self._read_csv_files(folder)
        if df.empty:
            return df

        df = self._ensure_service_column(df, preferred_cols=("service", "app"))
        if "timestamp" in df.columns:
            df = self._parse_and_filter_time(df, "timestamp", window)

        if "log" not in df.columns:
            df["log"] = ""
        df["log"] = df["log"].fillna("").astype(str)
        return df

    def load_traces_folder(self, folder: str, window: WindowSpec) -> pd.DataFrame:
        df = self._read_csv_files(folder)
        if df.empty:
            return df

        df = self._ensure_service_column(df, preferred_cols=("service", "app"))
        time_col = "start_time" if "start_time" in df.columns else "timestamp" if "timestamp" in df.columns else None
        if time_col:
            df = self._parse_and_filter_time(df, time_col, window)
            if time_col != "start_time":
                df["start_time"] = df[time_col]

        if "duration_us" in df.columns:
            df["duration_us"] = pd.to_numeric(df["duration_us"], errors="coerce")
            df = df.dropna(subset=["duration_us"])
        else:
            df["duration_us"] = 0.0

        if "tags_json" not in df.columns:
            df["tags_json"] = "{}"
        return df

    @staticmethod
    def _extract_url_from_file(file_path: str, key: str) -> Optional[str]:
        path = Path(file_path)
        if not path.exists():
            return None
        text = path.read_text(encoding="utf-8", errors="ignore")

        pattern = rf"{re.escape(key)}\s*=\s*(?:os\.environ\.get\([^,]+,\s*)?[\"']([^\"']+)[\"']"
        match = re.search(pattern, text)
        return match.group(1) if match else None

    def load_observability_endpoints(self) -> Dict[str, str]:
        """
        Resolve Loki/Prometheus/Jaeger endpoints by reading existing scripts.
        """
        chaos_root = Path(__file__).resolve().parents[2]
        loki_script = chaos_root / "logs_script" / "loki_script.py"
        prom_node_script = chaos_root / "metrics_script" / "prometheus_node_script.py"
        prom_pod_script = chaos_root / "metrics_script" / "prometheus_pod_specific_script.py"
        jaeger_script = chaos_root / "traces_script" / "jaeger_script.py"

        return {
            "loki_url": self._extract_url_from_file(str(loki_script), "LOKI_URL") or "",
            "prom_url_node": self._extract_url_from_file(str(prom_node_script), "PROM_URL") or "",
            "prom_url_pod": self._extract_url_from_file(str(prom_pod_script), "PROM_URL") or "",
            "jaeger_url": self._extract_url_from_file(str(jaeger_script), "JAEGER_URL") or "",
            "openai_model": self.config.get("openai_model", "gpt-4o-mini"),
            "openai_api_key_env": self.config.get("openai_api_key_env", "OPENAI_API_KEY"),
            "workspace_root": str(chaos_root.parent),
            "chaos_root": str(chaos_root),
            "cwd": os.getcwd(),
        }
