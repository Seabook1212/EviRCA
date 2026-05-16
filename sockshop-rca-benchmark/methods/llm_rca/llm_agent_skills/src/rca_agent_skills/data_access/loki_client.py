from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import time
from typing import Any

import pandas as pd
import requests
from requests import RequestException

from rca_agent_skills.common.logging_utils import get_logger
from rca_agent_skills.common.time_utils import parse_time
from rca_agent_skills.data_access.topology_loader import service_from_pod
from rca_agent_skills.query_expansion.renderer import render_template
from rca_agent_skills.skills.log_evidence_skill.parser import parse_raw_log


@dataclass
class LokiLogClient:
    request: object
    settings: dict

    def __post_init__(self) -> None:
        self.logger = get_logger(self.__class__.__name__)
        self.debug = self.settings.get("debug", {})

    def _config(self) -> dict[str, Any]:
        return self.settings.get("api", {}).get("loki", {})

    def _query_range(self, query: str, start_ns: int, end_ns: int) -> list[dict[str, Any]]:
        cfg = self._config()
        if self.debug.get("print_queries", True):
            self.logger.info("[LOGQL] %s", query)
        url = f"{cfg['base_url'].rstrip('/')}/loki/api/v1/query_range"
        params = {
            "query": query,
            "start": str(start_ns),
            "end": str(end_ns),
            "limit": str(cfg.get("limit", 5000)),
            "direction": cfg.get("direction", "forward"),
        }
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
                    break
                last_error = requests.HTTPError(
                    f"{response.status_code} Server Error from Loki query_range",
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
                "Loki query_range failed on attempt %s/%s: %s; retrying in %.1fs",
                attempt,
                max(1, max_retries),
                last_error,
                sleep_sec,
            )
            time.sleep(max(0.0, sleep_sec))

        if response is None:
            raise RuntimeError(f"Loki query_range failed without response: {last_error}")
        payload = response.json()
        if payload.get("status") != "success":
            return []
        return payload.get("data", {}).get("result", [])

    def _count_entries(self, streams: list[dict[str, Any]]) -> int:
        return sum(len(stream.get("values", [])) for stream in streams)

    def _query_range_complete(self, query: str, start_ns: int, end_ns: int) -> list[dict[str, Any]]:
        cfg = self._config()
        limit = int(cfg.get("limit", 5000))
        min_interval_ns = int(cfg.get("min_split_interval_ms", 1)) * 1_000_000

        streams = self._query_range(query, start_ns, end_ns)
        entry_count = self._count_entries(streams)
        if entry_count < limit:
            return streams

        if end_ns - start_ns <= min_interval_ns:
            self.logger.warning(
                "Loki query hit limit=%s inside an unsplittable interval; logs may still be truncated. start_ns=%s end_ns=%s",
                limit,
                start_ns,
                end_ns,
            )
            return streams

        midpoint_ns = start_ns + (end_ns - start_ns) // 2
        left = self._query_range_complete(query, start_ns, midpoint_ns)
        right = self._query_range_complete(query, midpoint_ns, end_ns)
        return left + right

    def fetch_window(self, window_name: str) -> pd.DataFrame:
        cfg = self._config()
        templates = self.request.config_bundle["loki_queries"]
        base_query = render_template(
            templates.get("base_query", '{namespace="{namespace}"}'),
            {"namespace": self.request.namespace},
        )
        window = getattr(self.request, f"{window_name}_window")
        start = parse_time(window.start)
        end = parse_time(window.end)
        slice_minutes = int(cfg.get("slice_minutes", 5))
        rows: list[dict[str, Any]] = []
        current = start
        seen: set[tuple[Any, ...]] = set()
        while current < end:
            slice_end = min(current + timedelta(minutes=slice_minutes), end)
            streams = self._query_range_complete(
                base_query,
                int(current.timestamp() * 1_000_000_000),
                int(slice_end.timestamp() * 1_000_000_000),
            )
            for stream in streams:
                labels = stream.get("stream", {})
                for ts_ns, line in stream.get("values", []):
                    key = (ts_ns, labels.get("pod"), labels.get("container"), line)
                    if key in seen:
                        continue
                    seen.add(key)
                    pod = labels.get("pod") or ""
                    container = labels.get("container")
                    service = container or service_from_pod(pod)
                    parsed_log = parse_raw_log(str(line), container)
                    rows.append(
                        {
                            "timestamp": pd.to_datetime(int(ts_ns), unit="ns", utc=True).isoformat(),
                            "trace_id": parsed_log["trace_id"],
                            "span_id": parsed_log["span_id"],
                            "service": service,
                            "node": labels.get("node_name"),
                            "pod": pod,
                            "container": container,
                            "log_level": parsed_log["log_level"],
                            "log_source": parsed_log["log_source"],
                            "log_type": parsed_log["log_type"],
                            "message": parsed_log["message"],
                            "message_template": parsed_log["message_template"],
                            "raw_log": parsed_log["raw_log"],
                        }
                    )
            current = slice_end
        return pd.DataFrame(rows)
