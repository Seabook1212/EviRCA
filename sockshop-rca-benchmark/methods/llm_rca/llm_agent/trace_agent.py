"""
README
- Purpose: Compute service-level trace anomaly score.
- Formula components:
  1) latency_ratio = mean_latency_abnormal / (mean_latency_normal + 1e-6)
  2) downstream_failure_count from abnormal `tags_json` (error/5xx).
  trace_score_raw = latency_ratio + lambda_fail * downstream_failure_count
- Output scores are normalized to [0, 1].
"""

from __future__ import annotations

import json
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def minmax_normalize(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    values = np.array(list(scores.values()), dtype=float)
    vmax = float(np.max(values))
    vmin = float(np.min(values))
    if np.isclose(vmax, vmin):
        return {k: 0.0 for k in scores}
    return {k: float((v - vmin) / (vmax - vmin)) for k, v in scores.items()}


class TraceAgent:
    def __init__(self, epsilon: float = 1e-6, lambda_fail: float = 0.05) -> None:
        self.epsilon = epsilon
        self.lambda_fail = lambda_fail

    @staticmethod
    def _is_failure(tags_raw: str) -> bool:
        if not tags_raw or tags_raw == "{}":
            return False
        try:
            tags = json.loads(tags_raw)
        except Exception:
            return False
        if not isinstance(tags, dict):
            return False

        status_keys = ("status", "http.status_code", "http.status", "status_code")
        for key in status_keys:
            value = str(tags.get(key, "")).strip()
            if value.startswith("5"):
                return True

        text_blob = " ".join(str(v).lower() for v in tags.values())
        return any(tok in text_blob for tok in ("error", "exception", "fail"))

    def compute_scores(
        self,
        normal_traces: pd.DataFrame,
        abnormal_traces: pd.DataFrame,
    ) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
        if normal_traces.empty and abnormal_traces.empty:
            return {}, {}

        normal = normal_traces.copy()
        abnormal = abnormal_traces.copy()
        required = {"service", "duration_us"}

        if not required.issubset(normal.columns) and not normal.empty:
            raise ValueError("Normal traces must contain `service`, `duration_us`")
        if not required.issubset(abnormal.columns) and not abnormal.empty:
            raise ValueError("Abnormal traces must contain `service`, `duration_us`")

        normal["duration_us"] = pd.to_numeric(normal["duration_us"], errors="coerce")
        abnormal["duration_us"] = pd.to_numeric(abnormal["duration_us"], errors="coerce")
        normal = normal.dropna(subset=["duration_us"])
        abnormal = abnormal.dropna(subset=["duration_us"])

        normal_latency = normal.groupby("service")["duration_us"].mean().to_dict()
        abnormal_latency = abnormal.groupby("service")["duration_us"].mean().to_dict()

        if "tags_json" not in abnormal.columns:
            abnormal["tags_json"] = "{}"
        abnormal["__is_failure__"] = abnormal["tags_json"].astype(str).apply(self._is_failure)
        failure_counts = abnormal.groupby("service")["__is_failure__"].sum().to_dict()

        services = sorted(set(normal_latency.keys()) | set(abnormal_latency.keys()) | set(failure_counts.keys()))
        raw_scores: Dict[str, float] = {}
        summary: Dict[str, Dict[str, float]] = {}

        for service in services:
            n_lat = float(normal_latency.get(service, 0.0))
            a_lat = float(abnormal_latency.get(service, 0.0))
            latency_ratio = a_lat / (n_lat + self.epsilon)
            failures = float(failure_counts.get(service, 0.0))
            score = latency_ratio + self.lambda_fail * failures
            raw_scores[service] = score

            summary[service] = {
                "normal_mean_latency_us": n_lat,
                "abnormal_mean_latency_us": a_lat,
                "latency_ratio": latency_ratio,
                "downstream_failure_count": failures,
                "raw_trace_score": score,
            }

        return minmax_normalize(raw_scores), summary

