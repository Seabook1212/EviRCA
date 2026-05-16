from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests

from rca_agent_skills.common.logging_utils import get_logger
from rca_agent_skills.common.time_utils import parse_time


@dataclass
class JaegerTraceClient:
    request: object
    settings: dict

    def __post_init__(self) -> None:
        self.logger = get_logger(self.__class__.__name__)
        self.debug = self.settings.get("debug", {})

    def _config(self) -> dict[str, Any]:
        return self.settings.get("api", {}).get("jaeger", {})

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        cfg = self._config()
        url = f"{cfg['base_url'].rstrip('/')}{path}"
        if self.debug.get("print_queries", True):
            self.logger.info("[JAEGER] path=%s params=%s", path, params or {})
        response = requests.get(url, params=params or {}, timeout=int(cfg.get("timeout_sec", 120)), headers={"Connection": "close"})
        response.raise_for_status()
        return response.json()

    def _list_services(self) -> list[str]:
        cfg = self._config()
        payload = self._get("/api/services")
        excluded = set(cfg.get("excluded_services", []))
        return [svc for svc in payload.get("data", []) if svc not in excluded]

    def fetch_window(self, window_name: str) -> pd.DataFrame:
        cfg = self._config()
        window = getattr(self.request, f"{window_name}_window")
        start_us = int(parse_time(window.start).timestamp() * 1_000_000)
        end_us = int(parse_time(window.end).timestamp() * 1_000_000)
        rows: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for service in self._list_services():
            payload = self._get("/api/traces", {"service": service, "start": start_us, "end": end_us, "limit": int(cfg.get("limit", 10000))})
            for trace in payload.get("data", []):
                trace_id = trace.get("traceID")
                processes = trace.get("processes", {})
                proc_service = {pid: item.get("serviceName", service) for pid, item in processes.items()}
                for span in trace.get("spans", []):
                    key = (trace_id, span.get("spanID"))
                    if key in seen:
                        continue
                    seen.add(key)
                    tags = span.get("tags", [])
                    tag_lookup = {tag.get("key"): tag.get("value") for tag in tags if tag.get("key")}
                    parent_span_id = None
                    for ref in span.get("references", []):
                        if ref.get("refType") == "CHILD_OF":
                            parent_span_id = ref.get("spanID")
                            break
                    rows.append(
                        {
                            "timestamp": span.get("startTime"),
                            "trace_id": trace_id,
                            "span_id": span.get("spanID"),
                            "parent_span_id": parent_span_id,
                            "service": proc_service.get(span.get("processID"), service),
                            "operation": span.get("operationName"),
                            "duration": span.get("duration"),
                            "span_kind": tag_lookup.get("span.kind", ""),
                            "status_code": str(tag_lookup.get("http.status_code", "")),
                            "status": str(tag_lookup.get("otel.status_code", "SUCCESS")),
                            "peer_service": str(tag_lookup.get("peer.service", "")),
                            "http_method": str(tag_lookup.get("http.method", "")),
                            "http_url": str(tag_lookup.get("http.url", "")),
                            "exception_type": str(tag_lookup.get("exception.type", "")),
                            "exception_message": str(tag_lookup.get("exception.message", "")),
                            "pod": str(tag_lookup.get("pod", "")),
                            "container": str(tag_lookup.get("container", "")),
                            "node": str(tag_lookup.get("node", "")),
                            "tags_json": str(tags),
                        }
                    )
        return pd.DataFrame(rows)
