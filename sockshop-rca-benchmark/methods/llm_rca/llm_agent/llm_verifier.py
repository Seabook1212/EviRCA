"""
README
- Purpose: Verification-only LLM module with restricted Prometheus tool-calling.
- Supports OpenAI and Gemini (OpenAI-compatible endpoint) via chat.completions.
- LLM can query Prometheus only for candidate service and abnormal window.
- Max tool calls per candidate: 30.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import requests
from openai import OpenAI


class PrometheusQueryTool:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def query_range(self, promql: str, start: float, end: float, step: int = 30) -> Dict[str, float]:
        if step < 30:
            raise ValueError("step must be >= 30 seconds")
        if end <= start:
            raise ValueError("end must be greater than start")

        url = f"{self.base_url}/api/v1/query_range"
        params = {
            "query": promql,
            "start": str(float(start)),
            "end": str(float(end)),
            "step": str(int(step)),
        }

        resp = requests.get(url, params=params, timeout=60)
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("status") != "success":
            raise ValueError(f"Prometheus query failed: {payload}")

        results = payload.get("data", {}).get("result", [])
        values: List[float] = []
        for series in results:
            for _, v in series.get("values", []):
                try:
                    values.append(float(v))
                except Exception:
                    continue

        if not values:
            return {"mean": 0.0, "max": 0.0, "min": 0.0, "std": 0.0, "count": 0.0}

        n = float(len(values))
        mean_v = sum(values) / n
        var = sum((x - mean_v) ** 2 for x in values) / n
        std_v = math.sqrt(var)
        return {
            "mean": float(mean_v),
            "max": float(max(values)),
            "min": float(min(values)),
            "std": float(std_v),
            "count": n,
        }


class LLMVerifier:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        provider: str = "openai",
        max_tool_calls: Optional[int] = None,
        # Backward compatibility alias used by older pipeline code.
        max_iterations: Optional[int] = None,
        timeout_seconds: int = 60,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: Optional[str] = None,
        enabled: bool = True,
        verbose: bool = False,
        debug_io: bool = True,
        prometheus_base_url: str = "http://34.28.33.102:30990",
        prometheus_namespace: str = "sock-shop",
    ) -> None:
        self.model = model
        self.provider = provider.lower().strip()
        # Hard cap required by design. Accept legacy `max_iterations` as alias.
        requested_tool_calls = (
            max_tool_calls if max_tool_calls is not None else max_iterations
        )
        if requested_tool_calls is None:
            requested_tool_calls = 30
        self.max_tool_calls = max(1, min(30, int(requested_tool_calls)))
        self.timeout_seconds = timeout_seconds
        self.api_key_env = api_key_env
        self.base_url = base_url
        self.enabled = enabled
        self.verbose = verbose
        self.debug_io = debug_io
        self.prometheus_namespace = prometheus_namespace
        self.prom_tool = PrometheusQueryTool(prometheus_base_url)

        self._client: Optional[OpenAI] = None
        if self.enabled:
            api_key = os.environ.get(self.api_key_env, "")
            if api_key:
                client_kwargs: Dict[str, Any] = {
                    "api_key": api_key,
                    "timeout": self.timeout_seconds,
                }
                if self.provider == "gemini":
                    client_kwargs["base_url"] = (
                        self.base_url
                        or "https://generativelanguage.googleapis.com/v1beta/openai/"
                    )
                elif self.base_url:
                    client_kwargs["base_url"] = self.base_url
                self._client = OpenAI(**client_kwargs)

    @staticmethod
    def _service_pod_regex(service: str) -> str:
        # Prometheus regex parser rejects unnecessary escaped '-' (e.g. front\-end),
        # so keep '-' literal while escaping other regex metacharacters.
        safe = re.escape(service.strip()).replace(r"\-", "-")
        # Match only pods that belong to this service name:
        # - Deployment: <service>-<replicaset-hash>-<pod-suffix>
        # - StatefulSet: <service>-<ordinal>
        return f"^(?:{safe}-[a-z0-9]{{8,12}}-[a-z0-9]{{4,6}}|{safe}-[0-9]+)$"

    def _rewrite_synthetic_kpi_query(self, promql: str, candidate_service: str) -> Tuple[str, bool]:
        """
        Rewrite dataset KPI aliases (used in CSV) to real Prometheus queries.
        This avoids frequent no-data results when aliases are not raw TSDB metric names.
        """
        q = promql.strip()
        if not q:
            return q, False

        svc = candidate_service
        ns = self.prometheus_namespace
        pod_re = self._service_pod_regex(svc)

        rewrites = {
            "pod_cpu_cores": (
                'sum by (pod) (rate(container_cpu_usage_seconds_total{namespace="%s",container!="",container!="POD",container!="istio-proxy",pod=~"%s"}[5m]))'
                % (ns, pod_re)
            ),
            "pod_memory_mib": (
                'sum by (pod) (container_memory_working_set_bytes{namespace="%s",container!="",container!="POD",container!="istio-proxy",pod=~"%s"}) / 1024^2'
                % (ns, pod_re)
            ),
            "pod_net_rx_kbps": (
                'sum by (pod) (rate(container_network_receive_bytes_total{namespace="%s",pod=~"%s"}[5m])) / 1024'
                % (ns, pod_re)
            ),
            "pod_net_tx_kbps": (
                'sum by (pod) (rate(container_network_transmit_bytes_total{namespace="%s",pod=~"%s"}[5m])) / 1024'
                % (ns, pod_re)
            ),
            "pod_container_restarts": (
                'sum by (pod) (increase(kube_pod_container_status_restarts_total{namespace="%s",pod=~"%s"}[10m]))'
                % (ns, pod_re)
            ),
            "pod_request_qps": (
                'sum(rate(istio_requests_total{reporter="destination",destination_workload_namespace="%s",destination_workload="%s"}[1m]))'
                % (ns, svc)
            ),
            "pod_request_success_rate": (
                '100 * (sum(rate(istio_requests_total{reporter="destination",destination_workload_namespace="%s",destination_workload="%s",response_code=~"2..|3.."}[1m])) / clamp_min(sum(rate(istio_requests_total{reporter="destination",destination_workload_namespace="%s",destination_workload="%s"}[1m])), 1e-9))'
                % (ns, svc, ns, svc)
            ),
            "pod_request_latency_p90": (
                'histogram_quantile(0.90, sum by (le) (rate(istio_request_duration_milliseconds_bucket{reporter="destination",destination_workload_namespace="%s",destination_workload="%s"}[1m])))'
                % (ns, svc)
            ),
            "pod_request_latency_p95": (
                'histogram_quantile(0.95, sum by (le) (rate(istio_request_duration_milliseconds_bucket{reporter="destination",destination_workload_namespace="%s",destination_workload="%s"}[1m])))'
                % (ns, svc)
            ),
            "pod_request_latency_p99": (
                'histogram_quantile(0.99, sum by (le) (rate(istio_request_duration_milliseconds_bucket{reporter="destination",destination_workload_namespace="%s",destination_workload="%s"}[1m])))'
                % (ns, svc)
            ),
        }

        # 1) Exact alias form: "metric" or "metric{...}".
        m = re.match(r"^\s*([a-zA-Z_:][a-zA-Z0-9_:]*)\s*(\{[\s\S]*\})?\s*$", q)
        metric = m.group(1) if m else ""
        rewritten = rewrites.get(metric)
        if rewritten:
            return rewritten, True

        # 2) Alias appears inside wrapped expression (e.g. avg_over_time(pod_cpu_cores{...}[5m])).
        # In this case we canonicalize to the alias base query to avoid no-data on synthetic metric names.
        for alias, canonical_query in rewrites.items():
            if re.search(rf"\b{re.escape(alias)}\b", q):
                return canonical_query, True

        return q, False

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    def _log_debug(self, title: str, obj: Any) -> None:
        if self.verbose and self.debug_io:
            try:
                payload = json.dumps(obj, ensure_ascii=False, indent=2)
            except Exception:
                payload = str(obj)
            print(f"[LLM][debug] {title}:\n{payload}", flush=True)

    @staticmethod
    def _default_response(reason: str = "LLM disabled or unavailable") -> Dict[str, Any]:
        return {"consistency_score": 0.5, "reasoning": reason}

    @staticmethod
    def _safe_float_score(value: Any, default: float = 0.5) -> float:
        try:
            v = float(value)
        except Exception:
            return default
        return min(1.0, max(0.0, v))

    @staticmethod
    def _extract_json_block(text: str) -> Optional[Dict[str, Any]]:
        candidates: List[str] = []
        stripped = text.strip()

        # 1) Prefer fenced blocks: ```json ... ```
        for m in re.finditer(r"```(?:json)?\s*([\s\S]*?)```", stripped, flags=re.IGNORECASE):
            block = m.group(1).strip()
            if block:
                candidates.append(block)

        # 2) Raw content as-is.
        if stripped:
            candidates.append(stripped)

        # 3) Try strict JSON and tolerant raw_decode.
        decoder = json.JSONDecoder()
        for cand in candidates:
            try:
                obj = json.loads(cand)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass

            idx = cand.find("{")
            while idx != -1:
                try:
                    obj, _ = decoder.raw_decode(cand[idx:])
                    if isinstance(obj, dict):
                        return obj
                except Exception:
                    pass
                idx = cand.find("{", idx + 1)

            # Relaxed attempt: remove trailing commas.
            relaxed = re.sub(r",\s*([}\]])", r"\1", cand)
            try:
                obj = json.loads(relaxed)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass

        # 4) Regex fallback for near-JSON outputs.
        m_score = re.search(r'"?consistency_score"?\s*[:=]\s*([0-9]*\.?[0-9]+)', stripped)
        if m_score:
            try:
                score = float(m_score.group(1))
            except Exception:
                score = 0.5
            m_reason = re.search(r'"?reasoning"?\s*[:=]\s*"([\s\S]*?)"\s*[},]?\s*$', stripped)
            reasoning = m_reason.group(1) if m_reason else stripped[:400]
            return {"consistency_score": score, "reasoning": reasoning}
        return None

    @staticmethod
    def _extract_text_from_message(msg: Any) -> str:
        content = getattr(msg, "content", None)
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            chunks: List[str] = []
            for item in content:
                if isinstance(item, str):
                    chunks.append(item)
                elif isinstance(item, dict):
                    if isinstance(item.get("text"), str):
                        chunks.append(item.get("text", ""))
                    elif isinstance(item.get("content"), str):
                        chunks.append(item.get("content", ""))
            return "\n".join(x for x in chunks if x).strip()
        return ""

    @staticmethod
    def _parse_ts(value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            s = value.strip()
            if not s:
                return None
            try:
                return float(s)
            except Exception:
                pass
            try:
                if s.endswith("Z"):
                    s = s.replace("Z", "+00:00")
                return dt.datetime.fromisoformat(s).timestamp()
            except Exception:
                return None
        return None

    def _extract_abnormal_window(self, candidate_payload: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
        candidates: List[Tuple[Any, Any]] = [
            (
                candidate_payload.get("abnormal_start_timestamp"),
                candidate_payload.get("abnormal_end_timestamp"),
            ),
            (
                candidate_payload.get("abnormal_start"),
                candidate_payload.get("abnormal_end"),
            ),
        ]
        tw = candidate_payload.get("time_window", {})
        if isinstance(tw, dict):
            candidates.append(
                (
                    tw.get("abnormal_start_timestamp") or tw.get("abnormal_start"),
                    tw.get("abnormal_end_timestamp") or tw.get("abnormal_end"),
                )
            )

        for s, e in candidates:
            start = self._parse_ts(s)
            end = self._parse_ts(e)
            if start is not None and end is not None and end > start:
                return start, end
        return None, None

    @staticmethod
    def _extract_label_selectors(promql: str) -> List[str]:
        """Return all `{...}` selector bodies from the query."""
        return [m.group(1).strip() for m in re.finditer(r"\{([^{}]*)\}", promql)]

    def _check_no_wildcard_scan(self, promql: str) -> Tuple[bool, str]:
        """
        Block TSDB-wide scans. In verifier mode, broad scans are dangerous:
        - increase latency/cost
        - leak unrelated service signals
        - undermine deterministic verification
        """
        q = promql.strip()
        if not q:
            return False, "promql is empty"
        if re.search(r"__name__\s*=~", q):
            return False, "metric regex scans via __name__=~ are not allowed"
        if re.search(r"\{\s*__name__\s*=~\s*\"\\.\\*\"\s*\}", q):
            return False, '{__name__=~".*"} is not allowed'
        if re.search(r"\{\s*\}", q):
            return False, "empty selector {} is not allowed"
        # Label-only selectors (no explicit metric name) behave like broad scans.
        if re.search(r"(^|[^\w:])\s*\{[^{}]+\}", q):
            return False, "label-only selectors without metric name are not allowed"
        return True, ""

    def _check_namespace(self, promql: str) -> Tuple[bool, str]:
        """
        Require namespace scoping to keep verification constrained to the target
        application boundary and avoid unrelated-cluster data contamination.
        """
        ns_pat = re.escape(self.prometheus_namespace)
        if re.search(
            rf'namespace\s*=\s*"{ns_pat}"|namespace\s*=\s*\'{ns_pat}\'|'
            rf'destination_workload_namespace\s*=\s*"{ns_pat}"|destination_workload_namespace\s*=\s*\'{ns_pat}\'',
            promql,
        ):
            return True, ""
        return False, f'promql must contain namespace="{self.prometheus_namespace}" or destination_workload_namespace="{self.prometheus_namespace}"'

    def _selector_binds_candidate(self, selector: str, candidate_service: str) -> Tuple[bool, str]:
        """
        Every selector must bind to the candidate service. This prevents
        cross-service aggregation and keeps the module verification-only.
        """
        # Reject service regex and destination_workload regex to avoid cross-service scans.
        if re.search(r"\bservice\s*=~", selector):
            return False, "service=~ regex is not allowed"
        if re.search(r"\bdestination_workload\s*=~", selector):
            return False, "destination_workload=~ regex is not allowed"

        binding_found = False

        service_labels = re.findall(r'service\s*=\s*"([^"]+)"|service\s*=\s*\'([^\']+)\'', selector)
        for svc in [a or b for a, b in service_labels]:
            if svc != candidate_service:
                return False, f'service must be "{candidate_service}"'
            binding_found = True

        dst_labels = re.findall(
            r'destination_workload\s*=\s*"([^"]+)"|destination_workload\s*=\s*\'([^\']+)\'',
            selector,
        )
        for svc in [a or b for a, b in dst_labels]:
            if svc != candidate_service:
                return False, f'destination_workload must be "{candidate_service}"'
            binding_found = True

        pod_eq_labels = re.findall(r'pod\s*=\s*"([^"]+)"|pod\s*=\s*\'([^\']+)\'', selector)
        for pod in [a or b for a, b in pod_eq_labels]:
            if not pod.startswith(f"{candidate_service}-"):
                return False, f'pod must match "{candidate_service}-..."'
            binding_found = True

        pod_re_labels = re.findall(r'pod\s*=~\s*"([^"]+)"|pod\s*=~\s*\'([^\']+)\'', selector)
        for regex_text in [a or b for a, b in pod_re_labels]:
            unescaped = regex_text.replace("\\", "")
            if candidate_service not in unescaped:
                return False, f'pod=~ must include candidate service "{candidate_service}"'
            # Basic guard against OR-based multi-service regex.
            if "|" in regex_text:
                return False, "pod=~ alternation is not allowed"
            binding_found = True

        if not binding_found:
            return False, (
                'each selector must bind candidate service using one of: '
                'service="<svc>", destination_workload="<svc>", pod="<svc>-...", pod=~"...<svc>..."'
            )
        return True, ""

    def _check_service_binding(self, promql: str, candidate_service: str) -> Tuple[bool, str]:
        selectors = self._extract_label_selectors(promql)
        if not selectors:
            return False, "promql must contain at least one selector bound to candidate service"
        for selector in selectors:
            ok, err = self._selector_binds_candidate(selector, candidate_service)
            if not ok:
                return False, err
        return True, ""

    @staticmethod
    def _check_no_offset_or_subquery(promql: str) -> Tuple[bool, str]:
        """
        Disallow constructs that expand query scope or complexity beyond controlled
        verifier behavior.
        """
        if re.search(r"\boffset\b", promql, flags=re.IGNORECASE):
            return False, "offset is not allowed"
        # Any [range:step] is a subquery and is blocked.
        if re.search(r"\[[^\]]*:[^\]]*\]", promql):
            return False, "subqueries are not allowed"
        return True, ""

    @staticmethod
    def _check_time_window(
        start: Optional[float],
        end: Optional[float],
        abnormal_start: Optional[float],
        abnormal_end: Optional[float],
    ) -> Tuple[bool, str]:
        """
        Enforce strict abnormal-window confinement so the LLM can only verify
        candidate behavior during the known incident interval.
        """
        if start is None or end is None:
            return False, "start/end must be numeric or ISO timestamps"
        if end <= start:
            return False, "end must be greater than start"
        if abnormal_start is None or abnormal_end is None:
            return False, "abnormal window is required for verification tool calls"
        if not (abnormal_start <= start < end <= abnormal_end):
            return False, (
                "query time range must be within abnormal window "
                f"[{abnormal_start}, {abnormal_end}]"
            )
        return True, ""

    def _validate_promql(
        self,
        promql: str,
        candidate_service: str,
        start: Optional[float],
        end: Optional[float],
        abnormal_start: Optional[float],
        abnormal_end: Optional[float],
    ) -> Tuple[bool, str]:
        # Validation is intentionally modular for readability and auditability.
        checks = [
            self._check_no_wildcard_scan(promql),
            self._check_namespace(promql),
            self._check_service_binding(promql, candidate_service),
            self._check_no_offset_or_subquery(promql),
            self._check_time_window(start, end, abnormal_start, abnormal_end),
        ]
        for ok, err in checks:
            if not ok:
                return False, err
        return True, ""

    def _tool_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "query_prometheus",
                    "description": "Query Prometheus metrics for RCA verification",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "promql": {"type": "string"},
                            "start": {"type": "number"},
                            "end": {"type": "number"},
                            "step": {"type": "number"},
                        },
                        "required": ["promql", "start", "end"],
                    },
                },
            }
        ]

    def _execute_tool_call(
        self,
        call_id: str,
        arguments_json: str,
        candidate_service: str,
        abnormal_start: Optional[float],
        abnormal_end: Optional[float],
    ) -> Dict[str, Any]:
        try:
            args = json.loads(arguments_json or "{}")
        except Exception:
            return {"ok": False, "error": "invalid JSON arguments"}

        promql = str(args.get("promql", "")).strip()
        start = self._parse_ts(args.get("start"))
        end = self._parse_ts(args.get("end"))
        step_raw = args.get("step", 30)
        try:
            step = int(step_raw)
        except Exception:
            step = 30

        rewritten_promql, was_rewritten = self._rewrite_synthetic_kpi_query(promql, candidate_service)
        if was_rewritten:
            self._log(
                f"[LLM][tool:{call_id}] rewritten synthetic KPI query: "
                f"original={promql} rewritten={rewritten_promql}"
            )
        promql = rewritten_promql

        self._log(
            f"[LLM][tool:{call_id}] query_prometheus "
            f"promql={promql} start={start} end={end} step={step}"
        )

        valid, err = self._validate_promql(
            promql=promql,
            candidate_service=candidate_service,
            start=start,
            end=end,
            abnormal_start=abnormal_start,
            abnormal_end=abnormal_end,
        )
        if not valid:
            return {"ok": False, "error": err}
        if step < 30:
            return {"ok": False, "error": "step must be >= 30 seconds"}

        try:
            stats = self.prom_tool.query_range(promql=promql, start=start, end=end, step=step)

            # If no samples returned, try safe Istio-specific normalizations.
            if float(stats.get("count", 0.0)) <= 0.0:
                fallback_queries: List[str] = []

                # 1) Some clusters expose duration buckets in milliseconds rather than seconds.
                if "istio_request_duration_seconds_bucket" in promql:
                    fallback_queries.append(
                        promql.replace(
                            "istio_request_duration_seconds_bucket",
                            "istio_request_duration_milliseconds_bucket",
                        )
                    )

                # 2) Istio destination metrics often use destination_workload_namespace,
                # not namespace. Normalize label key if needed.
                istio_metric = "istio_" in promql
                has_plain_ns = bool(re.search(r"\bnamespace\s*=", promql))
                has_dest_ns = bool(re.search(r"\bdestination_workload_namespace\s*=", promql))
                if istio_metric and has_plain_ns and not has_dest_ns:
                    normalized_ns = re.sub(
                        r"\bnamespace(\s*=)",
                        r"destination_workload_namespace\1",
                        promql,
                    )
                    fallback_queries.append(normalized_ns)

                    # Also combine with seconds->milliseconds replacement for latency queries.
                    if "istio_request_duration_seconds_bucket" in normalized_ns:
                        fallback_queries.append(
                            normalized_ns.replace(
                                "istio_request_duration_seconds_bucket",
                                "istio_request_duration_milliseconds_bucket",
                            )
                        )

                # De-duplicate while preserving order.
                seen = set()
                ordered_fallbacks: List[str] = []
                for fq in fallback_queries:
                    if fq not in seen and fq != promql:
                        ordered_fallbacks.append(fq)
                        seen.add(fq)

                for idx, fallback_promql in enumerate(ordered_fallbacks, start=1):
                    self._log(
                        f"[LLM][tool:{call_id}] no data, retrying fallback#{idx} promql={fallback_promql}"
                    )
                    fallback_stats = self.prom_tool.query_range(
                        promql=fallback_promql,
                        start=start,
                        end=end,
                        step=step,
                    )
                    if float(fallback_stats.get("count", 0.0)) > 0.0:
                        stats = fallback_stats
                        break

            self._log(f"[LLM][tool:{call_id}] stats={stats}")
            return {"ok": True, "stats": stats}
        except Exception as exc:
            return {"ok": False, "error": f"prometheus query failed: {exc}"}

    def _precompute_candidate_stats(
        self,
        candidate_payload: Dict[str, Any],
        service: str,
        abnormal_start: Optional[float],
        abnormal_end: Optional[float],
    ) -> Dict[str, Any]:
        """
        Pre-query key KPI aliases so LLM has authoritative stats even if it makes
        poor tool-call choices.
        """
        if abnormal_start is None or abnormal_end is None:
            return {}

        metric_summary = candidate_payload.get("anomaly_summary", {}).get("metric_summary", {})
        metric_names: List[str] = []
        for key in ("top_metric_1", "top_metric_2", "top_metric_3"):
            val = str(metric_summary.get(key, "")).strip()
            if val:
                metric_names.append(val)

        # Add common KPI anchors to reduce "no data" hallucination.
        for m in ("pod_request_success_rate", "pod_request_latency_p95", "pod_cpu_cores"):
            metric_names.append(m)

        # Stable dedupe while preserving order.
        seen = set()
        ordered: List[str] = []
        for m in metric_names:
            if m not in seen:
                seen.add(m)
                ordered.append(m)

        precomputed: Dict[str, Any] = {}
        for idx, metric in enumerate(ordered[:6], start=1):
            promql = f'{metric}{{namespace="{self.prometheus_namespace}",service="{service}"}}'
            args = json.dumps(
                {
                    "promql": promql,
                    "start": abnormal_start,
                    "end": abnormal_end,
                    "step": 30,
                },
                ensure_ascii=False,
            )
            result = self._execute_tool_call(
                call_id=f"precheck-{idx}",
                arguments_json=args,
                candidate_service=service,
                abnormal_start=abnormal_start,
                abnormal_end=abnormal_end,
            )
            precomputed[metric] = result
        return precomputed

    def verify_candidate(self, candidate_payload: Dict[str, Any]) -> Dict[str, Any]:
        service = str(candidate_payload.get("service", "unknown"))
        abnormal_start, abnormal_end = self._extract_abnormal_window(candidate_payload)
        precomputed_stats = self._precompute_candidate_stats(
            candidate_payload=candidate_payload,
            service=service,
            abnormal_start=abnormal_start,
            abnormal_end=abnormal_end,
        )

        self._log(f"[LLM] Verifying candidate service: {service}")
        self._log(
            f"[LLM] provider={self.provider} model={self.model} "
            f"max_tool_calls={self.max_tool_calls}"
        )
        if abnormal_start is not None and abnormal_end is not None:
            self._log(f"[LLM] abnormal_window=[{abnormal_start}, {abnormal_end}]")
        if precomputed_stats:
            self._log_debug(f"{service} precomputed_stats", precomputed_stats)

        if not self.enabled:
            return self._default_response("LLM verification disabled by config")
        if self._client is None:
            return self._default_response(f"Missing API key in env `{self.api_key_env}`")

        system_prompt = (
            "You are an RCA verification assistant. "
            "You may call the tool query_prometheus to validate anomaly strength. "
            "You must only query the candidate service and abnormal time window. "
            "You must not explore other services. "
            f"Maximum {self.max_tool_calls} tool calls. "
            "Candidate binding may use service=..., destination_workload=..., or pod selector for that service only. "
            "Use topology_context (upstream/downstream dependencies) as supporting evidence only. "
            "Return strict JSON: "
            '{"consistency_score": float 0..1, "reasoning": string}'
        )

        window_note = (
            f"abnormal_start={abnormal_start}, abnormal_end={abnormal_end}"
            if abnormal_start is not None and abnormal_end is not None
            else "abnormal window not provided in payload; still restrict to candidate service only."
        )
        user_prompt = (
            f"Candidate service: {service}\n"
            f"Constraint: namespace must be \"{self.prometheus_namespace}\".\n"
            f"Constraint: PromQL must target candidate service \"{service}\" only "
            "(service=, destination_workload=, or pod selector).\n"
            f"Time constraint: {window_note}\n"
            f"Candidate context JSON:\n{json.dumps(candidate_payload, ensure_ascii=False)}\n"
            f"Authoritative precomputed Prometheus stats:\n{json.dumps(precomputed_stats, ensure_ascii=False)}\n"
            "If precomputed stats show non-zero count, do not claim 'no data'.\n"
            f"You may call query_prometheus up to {self.max_tool_calls} times, "
            "then return strict JSON only."
        )

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        tool_calls_used = 0
        non_json_retries = 0
        while True:
            try:
                self._log_debug(
                    f"{service} request(turn={len(messages)})",
                    {
                        "model": self.model,
                        "provider": self.provider,
                        "messages": messages,
                        "tools": self._tool_schema(),
                        "tool_choice": "auto",
                        "temperature": 0,
                    },
                )
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self._tool_schema(),
                    tool_choice="auto",
                    temperature=0,
                )
            except Exception as exc:
                self._log(f"[LLM] {service} call failed: {exc}")
                return self._default_response(f"LLM call failed: {exc}")

            msg = response.choices[0].message
            finish_reason = response.choices[0].finish_reason
            self._log_debug(
                f"{service} raw_response(turn={len(messages)})",
                response.model_dump() if hasattr(response, "model_dump") else str(response),
            )
            self._log(f"[LLM] {service} finish_reason={finish_reason}")
            tool_calls = msg.tool_calls or []

            if tool_calls:
                assistant_msg: Dict[str, Any] = {"role": "assistant", "content": msg.content or "", "tool_calls": []}
                for tc in tool_calls:
                    assistant_msg["tool_calls"].append(
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                    )
                messages.append(assistant_msg)

                for tc in tool_calls:
                    if tc.function.name != "query_prometheus":
                        result = {"ok": False, "error": f"unsupported tool: {tc.function.name}"}
                    else:
                        if tool_calls_used >= self.max_tool_calls:
                            self._log(f"[LLM] {service} reached max tool calls ({self.max_tool_calls})")
                            return self._default_response("max tool calls reached before final answer")
                        tool_calls_used += 1
                        result = self._execute_tool_call(
                            call_id=tc.id,
                            arguments_json=tc.function.arguments,
                            candidate_service=service,
                            abnormal_start=abnormal_start,
                            abnormal_end=abnormal_end,
                        )

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps(result, ensure_ascii=False),
                        }
                    )
                continue

            text = self._extract_text_from_message(msg)
            self._log_debug(f"{service} assistant_text(turn={len(messages)})", {"text": text})
            parsed = self._extract_json_block(text)
            if not parsed:
                self._log(f"[LLM] {service} malformed final output: {text[:120]}")
                if not text.strip():
                    # Some providers occasionally return empty text after tool calls; ask once more for final JSON.
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Return final answer now. Output ONLY strict JSON with schema: "
                                '{"consistency_score": float 0..1, "reasoning": string}'
                            ),
                        }
                    )
                    continue
                if non_json_retries < 2:
                    non_json_retries += 1
                    messages.append({"role": "assistant", "content": text})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Your previous response was not valid strict JSON. "
                                "Return ONLY strict JSON now with this exact schema: "
                                '{"consistency_score": float 0..1, "reasoning": string}'
                            ),
                        }
                    )
                    self._log(
                        f"[LLM] {service} retrying due to malformed output "
                        f"(retry={non_json_retries}/2)"
                    )
                    continue
                return self._default_response(
                    f"Malformed LLM output after retries; last text: {text[:200]}"
                )

            score = self._safe_float_score(parsed.get("consistency_score", 0.5), default=0.5)
            reasoning = str(parsed.get("reasoning", "")).strip() or "No reasoning returned"

            # Guardrail: avoid low-score artifacts caused by LLM claiming "no data"
            # when verifier already has non-zero Prometheus samples.
            no_data_claim = bool(re.search(r"\bno data\b|returned no data|all stats.*0", reasoning, re.IGNORECASE))
            has_precomputed_data = any(
                isinstance(v, dict)
                and v.get("ok") is True
                and float(v.get("stats", {}).get("count", 0.0)) > 0.0
                for v in precomputed_stats.values()
            )
            if no_data_claim and has_precomputed_data:
                score = max(score, 0.35)
                # Remove misleading "no data" sentences when verifier has ground-truth samples.
                sentences = re.split(r"(?<=[.!?])\s+", reasoning)
                filtered = [
                    s
                    for s in sentences
                    if not re.search(r"\bno data\b|returned no data|all stats.*0", s, re.IGNORECASE)
                ]
                cleaned_reasoning = " ".join(x.strip() for x in filtered if x.strip())
                reasoning = (
                    (cleaned_reasoning + " ") if cleaned_reasoning else ""
                ) + (
                    "Verifier correction: precomputed Prometheus stats in the abnormal window "
                    "had non-zero sample counts for this service."
                )
                self._log(f"[LLM] {service} corrected false 'no data' claim using precomputed stats.")

            self._log(f"[LLM] {service} consistency_score={score:.4f}")
            self._log_debug(f"{service} parsed_final", parsed)
            return {"consistency_score": score, "reasoning": reasoning}

    def verify_top_k(self, candidate_payloads: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        results: Dict[str, Dict[str, Any]] = {}
        for payload in candidate_payloads:
            service = str(payload.get("service", "unknown"))
            results[service] = self.verify_candidate(payload)
        return results
