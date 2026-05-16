from __future__ import annotations

import pandas as pd

from rca_agent_skills.common.severity_calibrator import SeverityCalibrator
from rca_agent_skills.skills.metric_evidence_skill.detector import (
    detect_metric_anomalies,
    to_anomaly_records as metric_to_records,
)


def test_disabled_calibration_returns_raw_heuristic_severity():
    calibrator = SeverityCalibrator({"enabled": False})

    result = calibrator.calibrate_metric_severity(
        metric="cpu_usage_pct",
        baseline_values=[1, 1, 1, 1],
        abnormal_values=[10, 11],
        raw_severity=0.42,
        zscore=4.0,
        robust_zscore=4.0,
        delta_ratio=1.0,
        persistence=1.0,
        direction="increase",
    )

    assert result["severity"] == 0.42
    assert result["raw_severity"] == 0.42
    assert result["severity_method"] == "heuristic"


def test_metric_empirical_tail_boosts_rare_abnormal_values():
    calibrator = SeverityCalibrator(
        {
            "enabled": True,
            "metric": {"mode": "empirical_tail", "min_baseline_points": 20},
            "empirical_tail": {"smoothing": 1.0},
        }
    )

    result = calibrator.calibrate_metric_severity(
        metric="cpu_usage_pct",
        baseline_values=[10.0] * 30 + [11.0, 9.0],
        abnormal_values=[40.0, 42.0, 41.0],
        raw_severity=0.2,
        zscore=6.0,
        robust_zscore=6.0,
        delta_ratio=3.0,
        persistence=1.0,
        direction="increase",
    )

    assert result["severity"] > 0.9
    assert result["raw_severity"] == 0.2
    assert result["severity_method"] == "empirical_tail"
    assert result["calibration_metadata"]["tail_p_value"] < 0.1


def test_metric_empirical_tail_does_not_inflate_normal_abnormal_values():
    calibrator = SeverityCalibrator(
        {
            "enabled": True,
            "metric": {"mode": "empirical_tail", "min_baseline_points": 20},
            "empirical_tail": {"smoothing": 1.0},
        }
    )

    result = calibrator.calibrate_metric_severity(
        metric="cpu_usage_pct",
        baseline_values=list(range(20, 50)),
        abnormal_values=[34.0, 35.0, 36.0],
        raw_severity=0.1,
        zscore=0.2,
        robust_zscore=0.2,
        delta_ratio=0.01,
        persistence=0.1,
        direction="both",
    )

    assert result["severity"] < 0.5


def test_log_bayesian_surprise_is_high_for_new_large_count():
    calibrator = SeverityCalibrator(
        {
            "enabled": True,
            "log": {"mode": "bayesian_surprise"},
            "bayesian_surprise": {"min_probability": 1.0e-12},
        }
    )

    result = calibrator.calibrate_log_severity(
        pattern_type="keyword_spike",
        baseline_count=0,
        abnormal_count=20,
        ratio=20.0,
        raw_severity=0.2,
        background_noise=False,
    )

    assert result["severity"] > 0.99
    assert result["severity_method"] == "bayesian_surprise"


def test_log_background_noise_is_capped_after_calibration():
    calibrator = SeverityCalibrator(
        {
            "enabled": True,
            "log": {"mode": "bayesian_surprise"},
            "bayesian_surprise": {"min_probability": 1.0e-12},
        }
    )

    result = calibrator.calibrate_log_severity(
        pattern_type="template_spike",
        baseline_count=0,
        abnormal_count=50,
        ratio=50.0,
        raw_severity=0.9,
        background_noise=True,
    )

    assert result["severity"] <= 0.25


def test_trace_failure_surprise_high_when_baseline_failures_are_rare():
    calibrator = SeverityCalibrator(
        {
            "enabled": True,
            "trace": {"failure_mode": "bayesian_surprise"},
            "bayesian_surprise": {"min_probability": 1.0e-12},
        }
    )

    result = calibrator.calibrate_trace_failure_severity(
        baseline_failure_count=0,
        baseline_total_count=100,
        abnormal_failure_count=20,
        abnormal_total_count=40,
        failure_rate=0.5,
        failure_ratio=50.0,
        raw_severity=0.3,
    )

    assert result["severity"] > 0.99
    assert result["severity_method"] == "bayesian_surprise"


def test_metric_records_preserve_calibration_metadata():
    calibrator = SeverityCalibrator(
        {
            "enabled": True,
            "metric": {"mode": "empirical_tail", "min_baseline_points": 20},
            "empirical_tail": {"smoothing": 1.0},
        }
    )
    baseline = pd.DataFrame(
        {
            "pod": ["pod-a"] * 30,
            "service": ["orders"] * 30,
            "metric": ["cpu_usage_pct"] * 30,
            "value": [10.0] * 30,
        }
    )
    abnormal = pd.DataFrame(
        {
            "pod": ["pod-a"] * 5,
            "service": ["orders"] * 5,
            "metric": ["cpu_usage_pct"] * 5,
            "value": [40.0, 41.0, 42.0, 43.0, 44.0],
        }
    )

    features = detect_metric_anomalies(
        baseline,
        abnormal,
        {"zscore_threshold": 2.0, "robust_zscore_threshold": 3.0, "min_relative_delta_ratio": 0.2},
        {"cpu_usage_pct": {"direction": "increase"}},
        ["pod", "service"],
        severity_calibrator=calibrator,
    )
    records = metric_to_records(features)

    assert records
    assert records[0].severity >= records[0].metadata["raw_severity"]
    assert records[0].metadata["severity_method"] in {"empirical_tail", "heuristic"}
