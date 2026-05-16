from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = low
    return max(low, min(high, numeric))


def safe_log(value: float, eps: float = 1e-12) -> float:
    return math.log(max(float(value), eps))


def _clean_numeric(values: Iterable | pd.Series | np.ndarray | None) -> np.ndarray:
    if values is None:
        return np.array([], dtype=float)
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return series.astype(float).to_numpy()


def empirical_tail_p_value(
    scores: Iterable[float],
    target_score: float,
    smoothing: float = 1.0,
) -> tuple[float, int, int]:
    clean = _clean_numeric(scores)
    if len(clean) == 0:
        return 1.0, 0, 0
    tail_count = int((clean >= float(target_score)).sum())
    smooth = max(float(smoothing), 0.0)
    p_value = (smooth + tail_count) / (len(clean) + smooth)
    return clamp(p_value), tail_count, int(len(clean))


def poisson_tail_probability(k: int, lambda_hat: float) -> float:
    k = int(k)
    lam = max(float(lambda_hat), 0.0)
    if k <= 0:
        return 1.0
    if lam <= 0.0:
        return 0.0
    try:
        from scipy.stats import poisson  # type: ignore

        return clamp(float(poisson.sf(k - 1, lam)))
    except Exception:
        pass

    # Fallback recurrence for moderate counts. For very large values, use a
    # continuity-corrected normal approximation to avoid underflow.
    if k > 512 or lam > 512:
        sigma = math.sqrt(max(lam, 1e-12))
        z = (k - 0.5 - lam) / sigma
        return clamp(0.5 * math.erfc(z / math.sqrt(2.0)))

    probability = math.exp(-lam)
    cdf = probability
    for value in range(1, k):
        probability *= lam / value
        cdf += probability
    return clamp(1.0 - cdf)


def binomial_tail_probability(k: int, n: int, p: float) -> float:
    k = int(k)
    n = int(n)
    p = clamp(p)
    if k <= 0:
        return 1.0
    if n <= 0 or k > n:
        return 0.0
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    try:
        from scipy.stats import binom  # type: ignore

        return clamp(float(binom.sf(k - 1, n, p)))
    except Exception:
        pass

    def log_pmf(value: int) -> float:
        return (
            math.lgamma(n + 1)
            - math.lgamma(value + 1)
            - math.lgamma(n - value + 1)
            + value * safe_log(p)
            + (n - value) * safe_log(1.0 - p)
        )

    if k <= n * p:
        terms = [math.exp(log_pmf(value)) for value in range(0, k)]
        return clamp(1.0 - sum(terms))
    terms = [math.exp(log_pmf(value)) for value in range(k, n + 1)]
    return clamp(sum(terms))


class SeverityCalibrator:
    def __init__(self, settings: dict | None):
        self.settings = settings or {}
        self.enabled = bool(self.settings.get("enabled", False))
        self.keep_raw_severity = bool(self.settings.get("keep_raw_severity", True))
        empirical_cfg = self.settings.get("empirical_tail", {})
        surprise_cfg = self.settings.get("bayesian_surprise", {})
        self.min_p_value = float(empirical_cfg.get("min_p_value", 1.0e-6))
        self.max_p_value = float(empirical_cfg.get("max_p_value", 1.0))
        self.smoothing = float(empirical_cfg.get("smoothing", 1.0))
        self.min_probability = float(surprise_cfg.get("min_probability", 1.0e-12))
        self.poisson_alpha = float(surprise_cfg.get("poisson_alpha", 1.0))
        self.poisson_beta = float(surprise_cfg.get("poisson_beta", 1.0))
        self.beta_alpha = float(surprise_cfg.get("beta_alpha", 1.0))
        self.beta_beta = float(surprise_cfg.get("beta_beta", 1.0))
        self.log_low_count_full_confidence = float(
            surprise_cfg.get("log_low_count_full_confidence", 20.0)
        )
        self.log_low_count_power = float(surprise_cfg.get("log_low_count_power", 0.5))
        self.blend_rule = str(self.settings.get("blend_rule", "max"))
        self.raw_weight = float(self.settings.get("raw_weight", 0.3))

    def _fallback(self, raw_severity: float, method: str, note: str) -> dict[str, Any]:
        raw = clamp(raw_severity)
        return {
            "severity": raw,
            "raw_severity": raw,
            "severity_method": method,
            "calibration_notes": [note],
            "calibration_metadata": {},
        }

    def _blend(self, raw_severity: float, calibrated_severity: float) -> float:
        raw = clamp(raw_severity)
        calibrated = clamp(calibrated_severity)
        if self.blend_rule == "calibrated_only":
            return round(calibrated, 4)
        if self.blend_rule == "weighted":
            raw_weight = clamp(self.raw_weight)
            return round(raw_weight * raw + (1.0 - raw_weight) * calibrated, 4)
        return round(max(raw, calibrated), 4)

    def calibrate_metric_severity(
        self,
        *,
        metric: str,
        baseline_values,
        abnormal_values,
        raw_severity: float,
        zscore: float,
        robust_zscore: float,
        delta_ratio: float,
        persistence: float,
        direction: str | None = None,
        thresholds: dict | None = None,
    ) -> dict[str, Any]:
        if not self.enabled:
            return self._fallback(raw_severity, "heuristic", "severity calibration disabled")
        cfg = self.settings.get("metric", {})
        if cfg.get("mode", "empirical_tail") != "empirical_tail":
            return self._fallback(raw_severity, "heuristic", "metric calibration mode is not empirical_tail")

        baseline = _clean_numeric(baseline_values)
        abnormal = _clean_numeric(abnormal_values)
        min_points = int(cfg.get("min_baseline_points", 20))
        if len(baseline) < min_points or len(abnormal) == 0:
            return self._fallback(raw_severity, "heuristic", "insufficient baseline or abnormal samples")

        result = self._empirical_tail_calibration(
            baseline,
            abnormal,
            raw_severity=raw_severity,
            direction=direction or "both",
        )
        result["severity_method"] = "empirical_tail"
        result["calibration_metadata"].update(
            {
                "metric": metric,
                "zscore": zscore,
                "robust_zscore": robust_zscore,
                "delta_ratio": delta_ratio,
                "persistence": persistence,
            }
        )
        return result

    def calibrate_trace_latency_severity(
        self,
        *,
        baseline_values,
        abnormal_values,
        ratio: float,
        raw_severity: float,
        sample_count: int,
    ) -> dict[str, Any]:
        if not self.enabled:
            return self._fallback(raw_severity, "heuristic", "severity calibration disabled")
        cfg = self.settings.get("trace", {})
        if cfg.get("latency_mode", "empirical_tail") != "empirical_tail":
            return self._fallback(raw_severity, "heuristic", "trace latency calibration mode is not empirical_tail")

        baseline = _clean_numeric(baseline_values)
        abnormal = _clean_numeric(abnormal_values)
        min_points = int(cfg.get("min_baseline_points", 20))
        if len(baseline) < min_points or len(abnormal) == 0:
            return self._fallback(raw_severity, "heuristic", "insufficient trace latency samples")

        result = self._empirical_tail_calibration(
            baseline,
            abnormal,
            raw_severity=raw_severity,
            direction="increase",
        )
        result["severity_method"] = "empirical_tail"
        result["calibration_metadata"].update({"ratio": ratio, "sample_count": sample_count})
        return result

    def _empirical_tail_calibration(
        self,
        baseline: np.ndarray,
        abnormal: np.ndarray,
        *,
        raw_severity: float,
        direction: str,
    ) -> dict[str, Any]:
        center = float(np.median(baseline))
        mad = float(np.median(np.abs(baseline - center)))
        scale = 1.4826 * mad
        if scale <= 1e-12:
            scale = float(np.std(baseline, ddof=0))
        if scale <= 1e-12:
            return self._fallback(raw_severity, "heuristic", "baseline distribution has near-zero scale")

        abnormal_score_value = float(np.mean(abnormal))
        if direction == "increase":
            baseline_scores = np.maximum(0.0, baseline - center) / scale
            target_score = max(0.0, abnormal_score_value - center) / scale
        elif direction == "decrease":
            baseline_scores = np.maximum(0.0, center - baseline) / scale
            target_score = max(0.0, center - abnormal_score_value) / scale
        else:
            baseline_scores = np.abs(baseline - center) / scale
            target_score = abs(abnormal_score_value - center) / scale

        p_tail, tail_count, baseline_points = empirical_tail_p_value(
            baseline_scores, target_score, self.smoothing
        )
        p_tail = clamp(p_tail, self.min_p_value, self.max_p_value)
        calibrated = clamp(1.0 - p_tail)
        return {
            "severity": self._blend(raw_severity, calibrated),
            "raw_severity": clamp(raw_severity),
            "severity_method": "empirical_tail",
            "calibration_notes": ["empirical tail probability calibrated severity"],
            "calibration_metadata": {
                "tail_p_value": p_tail,
                "nonconformity_score": target_score,
                "baseline_tail_count": tail_count,
                "baseline_points": baseline_points,
                "baseline_center": center,
                "baseline_scale": scale,
                "calibrated_severity": calibrated,
                "blend_rule": self.blend_rule,
            },
        }

    def calibrate_log_severity(
        self,
        *,
        pattern_type: str,
        baseline_count: int,
        abnormal_count: int,
        ratio: float,
        raw_severity: float,
        background_noise: bool = False,
    ) -> dict[str, Any]:
        if not self.enabled:
            return self._fallback(raw_severity, "heuristic", "severity calibration disabled")
        cfg = self.settings.get("log", {})
        if cfg.get("mode", "bayesian_surprise") != "bayesian_surprise":
            return self._fallback(raw_severity, "heuristic", "log calibration mode is not bayesian_surprise")

        alpha = self.poisson_alpha
        beta = self.poisson_beta
        lambda_hat = (alpha + max(int(baseline_count), 0)) / (beta + 1.0)
        p_tail = max(
            poisson_tail_probability(max(int(abnormal_count), 0), lambda_hat),
            self.min_probability,
        )
        surprise = -safe_log(p_tail, self.min_probability)
        calibrated = clamp(1.0 - math.exp(-surprise))
        full_confidence_count = max(self.log_low_count_full_confidence, 1.0)
        low_count_confidence = clamp(
            (max(float(abnormal_count), 0.0) / full_confidence_count)
            ** max(self.log_low_count_power, 0.0)
        )
        count_adjusted = clamp(calibrated * low_count_confidence)
        final = self._blend(raw_severity, count_adjusted)
        if background_noise:
            final = min(final, 0.25)
        return {
            "severity": round(final, 4),
            "raw_severity": clamp(raw_severity),
            "severity_method": "bayesian_surprise",
            "calibration_notes": ["Poisson-Gamma posterior predictive tail calibrated severity"],
            "calibration_metadata": {
                "pattern_type": pattern_type,
                "tail_p_value": p_tail,
                "bayesian_surprise": surprise,
                "lambda_hat": lambda_hat,
                "baseline_count": int(baseline_count),
                "abnormal_count": int(abnormal_count),
                "ratio": ratio,
                "background_noise": background_noise,
                "calibrated_severity": calibrated,
                "low_count_confidence": low_count_confidence,
                "count_adjusted_severity": count_adjusted,
                "log_low_count_full_confidence": full_confidence_count,
                "log_low_count_power": self.log_low_count_power,
                "blend_rule": self.blend_rule,
            },
        }

    def calibrate_trace_failure_severity(
        self,
        *,
        baseline_failure_count: int,
        baseline_total_count: int,
        abnormal_failure_count: int,
        abnormal_total_count: int,
        failure_rate: float,
        failure_ratio: float,
        raw_severity: float,
    ) -> dict[str, Any]:
        if not self.enabled:
            return self._fallback(raw_severity, "heuristic", "severity calibration disabled")
        cfg = self.settings.get("trace", {})
        if cfg.get("failure_mode", "bayesian_surprise") != "bayesian_surprise":
            return self._fallback(raw_severity, "heuristic", "trace failure calibration mode is not bayesian_surprise")

        baseline_failures = max(int(baseline_failure_count), 0)
        baseline_total = max(int(baseline_total_count), 0)
        abnormal_failures = max(int(abnormal_failure_count), 0)
        abnormal_total = max(int(abnormal_total_count), 0)
        alpha_post = self.beta_alpha + baseline_failures
        beta_post = self.beta_beta + max(baseline_total - baseline_failures, 0)
        p_hat = alpha_post / max(alpha_post + beta_post, 1e-12)
        p_tail = max(
            binomial_tail_probability(abnormal_failures, abnormal_total, p_hat),
            self.min_probability,
        )
        surprise = -safe_log(p_tail, self.min_probability)
        calibrated = clamp(1.0 - math.exp(-surprise))
        return {
            "severity": self._blend(raw_severity, calibrated),
            "raw_severity": clamp(raw_severity),
            "severity_method": "bayesian_surprise",
            "calibration_notes": ["Beta-Binomial posterior predictive tail calibrated severity"],
            "calibration_metadata": {
                "tail_p_value": p_tail,
                "bayesian_surprise": surprise,
                "baseline_failure_count": baseline_failures,
                "baseline_total_count": baseline_total,
                "abnormal_failure_count": abnormal_failures,
                "abnormal_total_count": abnormal_total,
                "posterior_failure_rate": p_hat,
                "failure_rate": failure_rate,
                "failure_ratio": failure_ratio,
                "calibrated_severity": calibrated,
                "blend_rule": self.blend_rule,
            },
        }
