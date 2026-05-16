"""
README
- Purpose: Compute service-level metric anomaly scores.
- Formula per KPI:
  deviation = (mean_abnormal - mean_normal) / (std_normal + 1e-6)
- Service score: max absolute deviation across KPIs.
- Output scores are normalized to [0, 1].
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def minmax_normalize(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    values = np.array(list(scores.values()), dtype=float)
    vmax = float(np.max(values))
    vmin = float(np.min(values))
    if np.isclose(vmax, vmin):
        return {k: 0.0 for k in scores}
    return {k: float((v - vmin) / (vmax - vmin)) for k, v in scores.items()}


class MetricsAgent:
    LATENCY_KPIS = {
        "pod_request_latency_p90",
        "pod_request_latency_p95",
        "pod_request_latency_p99",
    }

    def __init__(
        self,
        epsilon: float = 1e-6,
        allowed_kpis: Optional[Iterable[str]] = None,
        latency_kpi: Optional[str] = None,
    ) -> None:
        self.epsilon = epsilon
        self.allowed_kpis = [str(x) for x in allowed_kpis] if allowed_kpis else None
        self.latency_kpi = str(latency_kpi) if latency_kpi else None
        self.last_used_kpis: List[str] = []

    def _effective_kpis(self) -> Optional[List[str]]:
        if not self.allowed_kpis:
            return None

        kpis = [k for k in self.allowed_kpis]
        if self.latency_kpi:
            # Keep only one latency KPI among p90/p95/p99 if requested.
            kpis = [k for k in kpis if k not in self.LATENCY_KPIS]
            kpis.append(self.latency_kpi)

        # preserve order while deduplicating
        seen = set()
        ordered = []
        for k in kpis:
            if k not in seen:
                ordered.append(k)
                seen.add(k)
        return ordered

    def compute_scores(
        self,
        normal_metrics: pd.DataFrame,
        abnormal_metrics: pd.DataFrame,
    ) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
        if normal_metrics.empty or abnormal_metrics.empty:
            return {}, {}

        required = {"service", "metric", "value"}
        if not required.issubset(normal_metrics.columns) or not required.issubset(abnormal_metrics.columns):
            raise ValueError("Metrics data must contain columns: service, metric, value")

        normal = normal_metrics.copy()
        abnormal = abnormal_metrics.copy()

        effective_kpis = self._effective_kpis()
        if effective_kpis is not None:
            normal = normal[normal["metric"].astype(str).isin(effective_kpis)].copy()
            abnormal = abnormal[abnormal["metric"].astype(str).isin(effective_kpis)].copy()
            self.last_used_kpis = sorted(
                set(normal["metric"].astype(str).unique()) | set(abnormal["metric"].astype(str).unique())
            )
        else:
            self.last_used_kpis = sorted(
                set(normal["metric"].astype(str).unique()) | set(abnormal["metric"].astype(str).unique())
            )

        if normal.empty or abnormal.empty:
            return {}, {}

        normal["value"] = pd.to_numeric(normal["value"], errors="coerce")
        abnormal["value"] = pd.to_numeric(abnormal["value"], errors="coerce")
        normal = normal.dropna(subset=["value"])
        abnormal = abnormal.dropna(subset=["value"])

        normal_stats = (
            normal.groupby(["service", "metric"])["value"]
            .agg(mean_normal="mean", std_normal="std")
            .reset_index()
        )
        abnormal_stats = (
            abnormal.groupby(["service", "metric"])["value"]
            .agg(mean_abnormal="mean")
            .reset_index()
        )

        merged = pd.merge(normal_stats, abnormal_stats, on=["service", "metric"], how="outer").fillna(0.0)
        merged["std_normal"] = merged["std_normal"].replace(0.0, np.nan).fillna(self.epsilon)
        merged["deviation"] = (merged["mean_abnormal"] - merged["mean_normal"]) / (merged["std_normal"] + self.epsilon)

        # keep scipy usage lightweight for a robust outlier indicator in summary only
        merged["z_like"] = merged.groupby("metric")["deviation"].transform(
            lambda s: stats.zscore(s, nan_policy="omit") if len(s) > 1 else np.zeros_like(s)
        )

        service_scores = (
            merged.groupby("service")["deviation"]
            .apply(lambda s: float(np.max(np.abs(s.values))) if len(s) else 0.0)
            .to_dict()
        )
        normalized_scores = minmax_normalize(service_scores)

        service_summary: Dict[str, Dict[str, float]] = {}
        for service, group in merged.groupby("service"):
            top = group.reindex(group["deviation"].abs().sort_values(ascending=False).index).head(3)
            summary = {
                f"top_metric_{idx + 1}": str(row["metric"])
                for idx, (_, row) in enumerate(top.iterrows())
            }
            summary.update(
                {
                    f"top_metric_deviation_{idx + 1}": float(abs(row["deviation"]))
                    for idx, (_, row) in enumerate(top.iterrows())
                }
            )
            service_summary[str(service)] = summary

        return normalized_scores, service_summary
