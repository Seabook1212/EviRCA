from pathlib import Path

import pandas as pd

from rca_agent_skills.main import build_request
from rca_agent_skills.common.io_utils import read_json
from rca_agent_skills.data_access import build_data_access
from rca_agent_skills.llm import LLMClient
from rca_agent_skills.skills.metric_evidence_skill.detector import detect_in_window_metric_patterns, detect_metric_anomalies
from rca_agent_skills.skills.metric_evidence_skill.skill import MetricEvidenceSkill


def test_metric_skill_output_format():
    root = Path(__file__).resolve().parents[1]
    payload = read_json(root / "examples" / "sample_request.json")
    request = build_request(payload, root)
    settings = request.config_bundle["settings"]
    skill = MetricEvidenceSkill(settings, build_data_access(request, settings), LLMClient())
    result = skill.run(request, state=None)
    assert result.service_evidence
    assert result.pod_evidence
    assert any(item.service == "orders-db" for item in result.service_evidence)
    assert all(item.supporting_evidence for item in result.service_evidence)


def test_restart_count_with_zero_baseline_is_anomalous():
    baseline = pd.DataFrame(
        [
            {"pod": "orders-1", "service": "orders", "metric": "restart_count", "value": 0.0},
            {"pod": "orders-1", "service": "orders", "metric": "restart_count", "value": 0.0},
        ]
    )
    abnormal = pd.DataFrame(
        [
            {"pod": "orders-1", "service": "orders", "metric": "restart_count", "value": 0.0},
            {"pod": "orders-1", "service": "orders", "metric": "restart_count", "value": 1.0},
        ]
    )

    features = detect_metric_anomalies(
        baseline,
        abnormal,
        {"zscore_threshold": 2.0, "robust_zscore_threshold": 3.0, "persistence_ratio_threshold": 0.5},
        {"restart_count": {"direction": "increase"}},
        ["pod", "service"],
    )

    assert len(features) == 1
    assert features[0].metric == "restart_count"
    assert features[0].abnormal_max == 1.0


def test_ready_ratio_drop_is_anomalous_even_when_mean_shift_is_small():
    baseline = pd.DataFrame(
        [
            {"pod": "orders-1", "service": "orders", "metric": "ready_ratio", "value": 1.0},
            {"pod": "orders-1", "service": "orders", "metric": "ready_ratio", "value": 1.0},
            {"pod": "orders-1", "service": "orders", "metric": "ready_ratio", "value": 1.0},
        ]
    )
    abnormal = pd.DataFrame(
        [
            {"pod": "orders-1", "service": "orders", "metric": "ready_ratio", "value": 1.0},
            {"pod": "orders-1", "service": "orders", "metric": "ready_ratio", "value": 0.0},
            {"pod": "orders-1", "service": "orders", "metric": "ready_ratio", "value": 1.0},
        ]
    )

    features = detect_metric_anomalies(
        baseline,
        abnormal,
        {"zscore_threshold": 2.0, "robust_zscore_threshold": 3.0, "persistence_ratio_threshold": 0.5},
        {"ready_ratio": {"direction": "decrease"}},
        ["pod", "service"],
    )

    assert len(features) == 1
    assert features[0].metric == "ready_ratio"
    assert features[0].abnormal_min == 0.0


def test_small_latency_shift_below_ten_percent_is_not_anomalous():
    baseline = pd.DataFrame(
        [
            {"pod": "front-end-1", "service": "front-end", "metric": "latency_p99", "value": 108.038},
            {"pod": "front-end-1", "service": "front-end", "metric": "latency_p99", "value": 108.038},
            {"pod": "front-end-1", "service": "front-end", "metric": "latency_p99", "value": 108.038},
        ]
    )
    abnormal = pd.DataFrame(
        [
            {"pod": "front-end-1", "service": "front-end", "metric": "latency_p99", "value": 116.900},
            {"pod": "front-end-1", "service": "front-end", "metric": "latency_p99", "value": 116.900},
            {"pod": "front-end-1", "service": "front-end", "metric": "latency_p99", "value": 116.900},
        ]
    )

    features = detect_metric_anomalies(
        baseline,
        abnormal,
        {
            "min_relative_delta_ratio": 0.10,
            "zscore_threshold": 2.0,
            "robust_zscore_threshold": 3.0,
            "persistence_ratio_threshold": 0.0,
        },
        {"latency_p99": {"direction": "increase"}},
        ["pod", "service"],
    )

    assert features == []


def test_metric_supporting_evidence_includes_entity_scope():
    root = Path(__file__).resolve().parents[1]
    payload = read_json(root / "examples" / "sample_request.json")
    request = build_request(payload, root)
    settings = request.config_bundle["settings"]
    skill = MetricEvidenceSkill(settings, build_data_access(request, settings), LLMClient())
    result = skill.run(request, state=None)

    all_support = [
        support
        for evidence in result.service_evidence + result.pod_evidence
        for support in evidence.supporting_evidence
    ]

    assert any(support.startswith("pod ") for support in all_support)
    assert all(support.startswith(("pod ", "service ")) for support in all_support)


def test_metric_skill_prefers_pod_latency_data_over_duplicate_service_latency_records():
    class DataAccess:
        def get_metrics(self, window_name):
            if window_name == "baseline":
                return pd.DataFrame(
                    [
                        {
                            "timestamp": "2026-04-27T00:00:00Z",
                            "pod": "catalogue-58bdd4d4f9-48jpd",
                            "service": "catalogue",
                            "metric": "latency_p99",
                            "value": 10.0,
                        },
                        {
                            "timestamp": "2026-04-27T00:00:00Z",
                            "pod": "catalogue-58bdd4d4f9-48jpd",
                            "service": "catalogue",
                            "metric": "latency_p90",
                            "value": 10.0,
                        },
                        {
                            "timestamp": "2026-04-27T00:00:00Z",
                            "pod": None,
                            "service": "catalogue",
                            "metric": "latency_p99",
                            "value": 10.0,
                        },
                        {
                            "timestamp": "2026-04-27T00:00:00Z",
                            "pod": None,
                            "service": "catalogue",
                            "metric": "latency_p90",
                            "value": 10.0,
                        },
                    ]
                )
            return pd.DataFrame(
                [
                    {
                        "timestamp": "2026-04-27T00:10:00Z",
                        "pod": "catalogue-58bdd4d4f9-48jpd",
                        "service": "catalogue",
                            "metric": "latency_p99",
                            "value": 80.0,
                        },
                    {
                        "timestamp": "2026-04-27T00:10:00Z",
                        "pod": "catalogue-58bdd4d4f9-48jpd",
                        "service": "catalogue",
                        "metric": "latency_p90",
                        "value": 10.1,
                    },
                    {
                        "timestamp": "2026-04-27T00:10:00Z",
                        "pod": None,
                        "service": "catalogue",
                        "metric": "latency_p99",
                        "value": 80.0,
                    },
                    {
                        "timestamp": "2026-04-27T00:10:00Z",
                        "pod": None,
                        "service": "catalogue",
                        "metric": "latency_p90",
                        "value": 80.0,
                    },
                ]
            )

    class Request:
        config_bundle = {"metric_kpis": {"kpis": {"latency_p99": {"direction": "increase"}, "latency_p90": {"direction": "increase"}}}}

    skill = MetricEvidenceSkill(
        {
            "debug": {"print_skill_inputs": False, "print_anomaly_records": False, "print_skill_outputs": False},
            "detection": {"min_relative_delta_ratio": 0.05, "persistence_ratio_threshold": 0.0},
        },
        DataAccess(),
        LLMClient(),
    )

    result = skill.run(Request(), state=None)

    assert all(item.pod == "catalogue-58bdd4d4f9-48jpd" for item in result.pod_evidence)
    assert not any(item.pod in {None, "nan", ""} for item in result.pod_evidence)
    service_evidence = next(item for item in result.service_evidence if item.service == "catalogue")
    pod_evidence = next(item for item in result.pod_evidence if item.pod == "catalogue-58bdd4d4f9-48jpd")

    assert any("pod catalogue-58bdd4d4f9-48jpd latency_p99" in item for item in service_evidence.supporting_evidence)
    assert any("pod catalogue-58bdd4d4f9-48jpd latency_p99" in item for item in pod_evidence.supporting_evidence)
    assert not any("service catalogue latency_p99" in item for item in service_evidence.supporting_evidence)
    assert not any("service catalogue latency_p99" in item for item in pod_evidence.supporting_evidence)
    assert not any(record.entity_type == "service" and record.metric_or_pattern == "latency_p90" for record in result.anomaly_records)


def test_metric_skill_prefers_pod_resource_data_over_duplicate_service_resource_records():
    class DataAccess:
        def get_metrics(self, window_name):
            if window_name == "baseline":
                return pd.DataFrame(
                    [
                        {
                            "timestamp": "2026-04-27T00:00:00Z",
                            "pod": "orders-1",
                            "service": "orders",
                            "metric": "cpu_usage_pct",
                            "value": 5.0,
                        },
                        {
                            "timestamp": "2026-04-27T00:00:00Z",
                            "pod": "orders-2",
                            "service": "orders",
                            "metric": "cpu_usage_pct",
                            "value": 5.0,
                        },
                        {
                            "timestamp": "2026-04-27T00:00:00Z",
                            "pod": None,
                            "service": "orders",
                            "metric": "cpu_usage_pct",
                            "value": 5.0,
                        },
                    ]
                )
            return pd.DataFrame(
                [
                    {
                        "timestamp": "2026-04-27T00:10:00Z",
                        "pod": "orders-1",
                        "service": "orders",
                        "metric": "cpu_usage_pct",
                        "value": 36.0,
                    },
                    {
                        "timestamp": "2026-04-27T00:10:00Z",
                        "pod": "orders-2",
                        "service": "orders",
                        "metric": "cpu_usage_pct",
                        "value": 5.0,
                    },
                    {
                        "timestamp": "2026-04-27T00:10:00Z",
                        "pod": None,
                        "service": "orders",
                        "metric": "cpu_usage_pct",
                        "value": 20.0,
                    },
                ]
            )

    class Request:
        config_bundle = {"metric_kpis": {"kpis": {"cpu_usage_pct": {"direction": "increase"}}}}

    skill = MetricEvidenceSkill(
        {
            "debug": {"print_skill_inputs": False, "print_anomaly_records": False, "print_skill_outputs": False},
            "detection": {"min_relative_delta_ratio": 0.05, "persistence_ratio_threshold": 0.0},
        },
        DataAccess(),
        LLMClient(),
    )

    result = skill.run(Request(), state=None)

    assert any(record.entity_type == "pod" and record.metric_or_pattern == "cpu_usage_pct" for record in result.anomaly_records)
    assert not any(record.entity_type == "service" and record.metric_or_pattern == "cpu_usage_pct" for record in result.anomaly_records)
    orders_service = next(item for item in result.service_evidence if item.service == "orders")
    assert any("pod orders-1 cpu_usage_pct" in item for item in orders_service.supporting_evidence)
    assert not any("service orders cpu_usage_pct" in item for item in orders_service.supporting_evidence)


def test_metric_skill_prefers_pod_restart_and_network_data_over_duplicate_service_records():
    class DataAccess:
        def get_metrics(self, window_name):
            if window_name == "baseline":
                return pd.DataFrame(
                    [
                        {
                            "timestamp": "2026-04-27T00:00:00Z",
                            "pod": "user-db-0",
                            "service": "user-db",
                            "metric": "restart_count",
                            "value": 0.0,
                        },
                        {
                            "timestamp": "2026-04-27T00:00:00Z",
                            "pod": "user-db-0",
                            "service": "user-db",
                            "metric": "network_rx",
                            "value": 7000.0,
                        },
                        {
                            "timestamp": "2026-04-27T00:00:00Z",
                            "pod": None,
                            "service": "user-db",
                            "metric": "restart_count",
                            "value": 0.0,
                        },
                        {
                            "timestamp": "2026-04-27T00:00:00Z",
                            "pod": None,
                            "service": "user-db",
                            "metric": "network_rx",
                            "value": 7000.0,
                        },
                    ]
                )
            return pd.DataFrame(
                [
                    {
                        "timestamp": "2026-04-27T00:10:00Z",
                        "pod": "user-db-0",
                        "service": "user-db",
                        "metric": "restart_count",
                        "value": 2.0,
                    },
                    {
                        "timestamp": "2026-04-27T00:10:00Z",
                        "pod": "user-db-0",
                        "service": "user-db",
                        "metric": "network_rx",
                        "value": 10000.0,
                    },
                    {
                        "timestamp": "2026-04-27T00:10:00Z",
                        "pod": None,
                        "service": "user-db",
                        "metric": "restart_count",
                        "value": 2.0,
                    },
                    {
                        "timestamp": "2026-04-27T00:10:00Z",
                        "pod": None,
                        "service": "user-db",
                        "metric": "network_rx",
                        "value": 10000.0,
                    },
                ]
            )

    class Request:
        config_bundle = {
            "metric_kpis": {
                "kpis": {
                    "restart_count": {"direction": "increase"},
                    "network_rx": {"direction": "increase"},
                }
            }
        }

    skill = MetricEvidenceSkill(
        {
            "debug": {"print_skill_inputs": False, "print_anomaly_records": False, "print_skill_outputs": False},
            "detection": {"min_relative_delta_ratio": 0.05, "persistence_ratio_threshold": 0.0},
        },
        DataAccess(),
        LLMClient(),
    )

    result = skill.run(Request(), state=None)

    assert any(record.entity_type == "pod" and record.metric_or_pattern == "restart_count" for record in result.anomaly_records)
    assert any(record.entity_type == "pod" and record.metric_or_pattern == "network_rx" for record in result.anomaly_records)
    assert not any(record.entity_type == "service" and record.metric_or_pattern == "restart_count" for record in result.anomaly_records)
    assert not any(record.entity_type == "service" and record.metric_or_pattern == "network_rx" for record in result.anomaly_records)
    user_db_service = next(item for item in result.service_evidence if item.service == "user-db")
    assert any("pod user-db-0 restart_count" in item for item in user_db_service.supporting_evidence)
    assert any("pod user-db-0 network_rx" in item for item in user_db_service.supporting_evidence)
    assert not any("service user-db restart_count" in item for item in user_db_service.supporting_evidence)
    assert not any("service user-db network_rx" in item for item in user_db_service.supporting_evidence)


def test_in_window_detector_finds_sudden_cpu_increase():
    rows = []
    for idx, value in enumerate([10, 10, 11, 10, 11, 10, 35, 38, 40, 39]):
        rows.append(
            {
                "timestamp": f"2026-04-27T00:{idx:02d}:00Z",
                "pod": "orders-1",
                "service": "orders",
                "metric": "cpu_usage_pct",
                "value": value,
            }
        )
    abnormal = pd.DataFrame(rows)
    baseline = abnormal.assign(value=10)

    features = detect_in_window_metric_patterns(
        baseline,
        abnormal,
        {
            "metric_in_window": {
                "enabled": True,
                "min_points": 6,
                "rolling_points": 4,
                "min_segment_points": 3,
                "sudden_shift_ratio": 0.3,
                "sudden_shift_metrics": ["cpu_usage_pct"],
            }
        },
        {"cpu_usage_pct": {"direction": "increase"}},
        ["pod", "service"],
    )

    assert len(features) == 1
    assert features[0].metric == "cpu_usage_pct"
    assert features[0].in_window_pattern == "sudden_increase"
    assert features[0].segment_mean > features[0].pre_segment_mean


def test_in_window_detector_finds_request_rate_data_gap():
    timestamps = [
        "2026-04-27T00:00:00Z",
        "2026-04-27T00:01:00Z",
        "2026-04-27T00:02:00Z",
        "2026-04-27T00:08:00Z",
        "2026-04-27T00:09:00Z",
        "2026-04-27T00:10:00Z",
    ]
    abnormal = pd.DataFrame(
        [
            {"timestamp": ts, "pod": "orders-1", "service": "orders", "metric": "request_rate", "value": 5.0}
            for ts in timestamps
        ]
    )
    baseline = pd.DataFrame(
        [
            {
                "timestamp": f"2026-04-26T00:{idx:02d}:00Z",
                "pod": "orders-1",
                "service": "orders",
                "metric": "request_rate",
                "value": 5.0,
            }
            for idx in range(6)
        ]
    )

    features = detect_in_window_metric_patterns(
        baseline,
        abnormal,
        {
            "metric_in_window": {
                "enabled": True,
                "min_points": 6,
                "gap_multiplier": 3,
                "min_gap_seconds": 120,
                "gap_metrics": ["request_rate"],
            }
        },
        {"request_rate": {"direction": "both"}},
        ["pod", "service"],
    )

    assert len(features) == 1
    assert features[0].in_window_pattern == "missing_data_gap"
    assert features[0].gap_seconds == 360
