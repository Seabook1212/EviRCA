from __future__ import annotations

from rca_agent_skills.common.models import AnomalyRecord, PodEvidence
from rca_agent_skills.skills.rootcause_reasoning_skill.fault_matcher import (
    match_fault_types,
)


def _memory_evidence(delta_ratio: float, severity: float) -> list[PodEvidence]:
    record = AnomalyRecord(
        source="metric",
        entity_type="pod",
        entity_name="pod-a",
        metric_or_pattern="memory_usage_pct",
        abnormal_value=43.53,
        baseline_value=41.219,
        delta=2.311,
        zscore=4.0,
        severity=severity,
        summary="memory_usage_pct shifted from 41.219 to 43.530",
        metadata={"delta_ratio": delta_ratio, "service": "payment", "pod": "pod-a"},
    )
    return [
        PodEvidence(
            pod="pod-a",
            service="payment",
            score=severity,
            supporting_evidence=[record.summary],
            anomaly_records=[record],
        )
    ]


def _restart_evidence() -> list[PodEvidence]:
    record = AnomalyRecord(
        source="metric",
        entity_type="pod",
        entity_name="orders-1",
        metric_or_pattern="restart_count",
        abnormal_value=0.2,
        baseline_value=0.0,
        delta=0.2,
        zscore=0.0,
        severity=0.9,
        summary="restart_count shifted from 0.000 to 0.200",
        metadata={
            "delta_ratio": 0.0,
            "abnormal_max": 1.0,
            "baseline_max": 0.0,
            "service": "orders",
            "pod": "orders-1",
        },
    )
    return [
        PodEvidence(
            pod="orders-1",
            service="orders",
            score=0.9,
            supporting_evidence=[record.summary],
            anomaly_records=[record],
        )
    ]


def _stateful_restart_latency_evidence() -> list[PodEvidence]:
    restart = AnomalyRecord(
        source="metric",
        entity_type="pod",
        entity_name="carts-db-0",
        metric_or_pattern="restart_count",
        abnormal_value=0.2,
        baseline_value=0.0,
        delta=0.2,
        zscore=0.0,
        severity=0.9,
        summary="restart_count shifted from 0.000 to 0.200",
        metadata={
            "delta_ratio": 0.0,
            "abnormal_max": 2.0,
            "baseline_max": 0.0,
            "service": "carts-db",
            "pod": "carts-db-0",
        },
    )
    latency = AnomalyRecord(
        source="metric",
        entity_type="pod",
        entity_name="carts-db-0",
        metric_or_pattern="latency_p99",
        abnormal_value=134.0,
        baseline_value=44.0,
        delta=90.0,
        zscore=3.0,
        severity=0.9,
        summary="latency_p99 shifted from 44.000 to 134.000",
        metadata={"delta_ratio": 2.0, "service": "carts-db", "pod": "carts-db-0"},
    )
    return [
        PodEvidence(
            pod="carts-db-0",
            service="carts-db",
            score=0.9,
            supporting_evidence=[restart.summary, latency.summary],
            anomaly_records=[restart, latency],
        )
    ]


def _latency_gap_evidence() -> list[PodEvidence]:
    record = AnomalyRecord(
        source="metric",
        entity_type="pod",
        entity_name="payment-664c4d4c74-r8lxz",
        metric_or_pattern="latency_p99",
        abnormal_value=5.0,
        baseline_value=5.0,
        delta=0.0,
        zscore=0.0,
        severity=0.85,
        summary="latency_p99 has an abnormal-window data gap of 185s",
        metadata={
            "delta_ratio": 0.0,
            "in_window_pattern": "missing_data_gap",
            "gap_seconds": 185.0,
            "service": "payment",
            "pod": "payment-664c4d4c74-r8lxz",
        },
    )
    return [
        PodEvidence(
            pod="payment-664c4d4c74-r8lxz",
            service="payment",
            score=0.85,
            supporting_evidence=[record.summary],
            anomaly_records=[record],
        )
    ]


def _multi_gap_evidence_with_ready_gap() -> list[PodEvidence]:
    records = []
    gap_metrics = [
        "latency_p90",
        "latency_p95",
        "latency_p99",
        "success_rate",
        "ready_ratio",
    ]
    for metric in gap_metrics:
        records.append(
            AnomalyRecord(
                source="metric",
                entity_type="pod",
                entity_name="carts-1",
                metric_or_pattern=metric,
                abnormal_value=0.0,
                baseline_value=1.0,
                delta=-1.0,
                zscore=0.0,
                severity=0.9,
                summary=f"{metric} has an abnormal-window data gap of 185s",
                metadata={
                    "delta_ratio": 0.0,
                    "in_window_pattern": "missing_data_gap",
                    "gap_seconds": 185.0,
                    "service": "carts",
                    "pod": "carts-1",
                },
            )
        )
    return [
        PodEvidence(
            pod="carts-1",
            service="carts",
            score=0.9,
            supporting_evidence=[record.summary for record in records],
            anomaly_records=records,
        )
    ]


def _request_rate_shift_evidence() -> list[PodEvidence]:
    record = AnomalyRecord(
        source="metric",
        entity_type="pod",
        entity_name="payment-1",
        metric_or_pattern="request_rate",
        abnormal_value=12.0,
        baseline_value=9.0,
        delta=3.0,
        zscore=2.5,
        severity=0.75,
        summary="request_rate shifted from 9.000 to 12.000",
        metadata={"delta_ratio": 0.33, "service": "payment", "pod": "payment-1"},
    )
    return [
        PodEvidence(
            pod="payment-1",
            service="payment",
            score=0.75,
            supporting_evidence=[record.summary],
            anomaly_records=[record],
        )
    ]


def test_weak_resource_only_signal_does_not_become_network_delay():
    matches = match_fault_types(
        _memory_evidence(delta_ratio=0.056, severity=0.95), topology={}, rules_config={}
    )

    assert matches[0][0] == "memory_stress"
    assert matches[0][1] <= 0.35
    assert "Weak memory shift" in matches[0][3][0]


def test_material_memory_signal_matches_memory_stress_with_higher_confidence():
    matches = match_fault_types(
        _memory_evidence(delta_ratio=0.35, severity=0.95), topology={}, rules_config={}
    )

    assert matches[0][0] == "memory_stress"
    assert matches[0][1] > 0.35
    assert "Memory pressure" in matches[0][3][0]


def test_restart_count_matches_pod_failure_even_with_zero_delta_ratio():
    matches = match_fault_types(_restart_evidence(), topology={}, rules_config={})

    assert matches[0][0] == "pod_failure"
    assert matches[0][1] > 0.35
    assert matches[0][2][0].startswith("pod orders-1 restart_count:")


def test_latency_missing_data_gap_supports_network_partition_and_loss():
    matches = match_fault_types(_latency_gap_evidence(), topology={}, rules_config={})

    fault_types = [item[0] for item in matches]
    assert fault_types[0] == "network_partition"
    assert "network_loss" in fault_types
    assert matches[0][1] > next(
        item[1] for item in matches if item[0] == "network_loss"
    )
    assert matches[0][2][0].startswith("pod payment-664c4d4c74-r8lxz latency_p99:")
    assert "Missing latency/success telemetry" in matches[0][3][0]


def test_multiple_connectivity_gaps_prefer_network_partition_over_pod_failure():
    matches = match_fault_types(
        _multi_gap_evidence_with_ready_gap(), topology={}, rules_config={}
    )

    fault_types = [item[0] for item in matches]
    assert fault_types[0] == "network_partition"
    assert "pod_failure" not in fault_types
    assert "Multiple abnormal-window telemetry gaps" in matches[0][3][0]


def test_request_rate_shift_alone_is_not_network_loss_or_partition():
    matches = match_fault_types(
        _request_rate_shift_evidence(), topology={}, rules_config={}
    )

    fault_types = [item[0] for item in matches]
    assert "network_loss" not in fault_types
    assert "network_partition" not in fault_types


def test_stateful_restart_with_latency_can_match_io_fault():
    matches = match_fault_types(
        _stateful_restart_latency_evidence(), topology={}, rules_config={}
    )

    fault_types = [item[0] for item in matches]
    assert "io_fault" in fault_types
    io_match = next(item for item in matches if item[0] == "io_fault")
    assert io_match[2][0].startswith("pod carts-db-0 restart_count:")
