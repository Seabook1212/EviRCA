"""
README
- Purpose: Compute service-level log anomaly score from WARN/ERROR frequency changes.
- Formula:
  normal_error_warn = count(log contains "error|warn")
  abnormal_error_warn = count(log contains "error|warn")
  score = abnormal_error_warn / (normal_error_warn + 1e-6)
- Output scores are normalized to [0, 1].
"""

from __future__ import annotations

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


class LogsAgent:
    def __init__(self, epsilon: float = 1e-6) -> None:
        self.epsilon = epsilon

    def compute_scores(
        self,
        normal_logs: pd.DataFrame,
        abnormal_logs: pd.DataFrame,
    ) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
        if normal_logs.empty and abnormal_logs.empty:
            return {}, {}

        normal = normal_logs.copy()
        abnormal = abnormal_logs.copy()

        if "service" not in normal.columns and not normal.empty:
            raise ValueError("Normal logs must include `service` column")
        if "service" not in abnormal.columns and not abnormal.empty:
            raise ValueError("Abnormal logs must include `service` column")

        if "log" not in normal.columns:
            normal["log"] = ""
        if "log" not in abnormal.columns:
            abnormal["log"] = ""

        issue_pattern = r"\b(?:error|warn|exception|fail(?:ed|ure)?)\b"
        normal["__is_issue__"] = normal["log"].astype(str).str.contains(issue_pattern, case=False, regex=True)
        abnormal["__is_issue__"] = abnormal["log"].astype(str).str.contains(issue_pattern, case=False, regex=True)

        services = sorted(set(normal.get("service", pd.Series(dtype=str)).astype(str).tolist()) | set(abnormal.get("service", pd.Series(dtype=str)).astype(str).tolist()))
        raw_scores: Dict[str, float] = {}
        summary: Dict[str, Dict[str, float]] = {}

        for service in services:
            normal_service = normal[normal["service"] == service]
            abnormal_service = abnormal[abnormal["service"] == service]
            normal_cnt = float(normal_service["__is_issue__"].sum())
            abnormal_cnt = float(abnormal_service["__is_issue__"].sum())
            ratio = abnormal_cnt / (normal_cnt + self.epsilon)
            raw_scores[service] = ratio
            summary[service] = {
                "normal_issue_count": normal_cnt,
                "abnormal_issue_count": abnormal_cnt,
                "issue_ratio": ratio,
                "normal_log_count": float(len(normal_service)),
                "abnormal_log_count": float(len(abnormal_service)),
            }

        return minmax_normalize(raw_scores), summary
