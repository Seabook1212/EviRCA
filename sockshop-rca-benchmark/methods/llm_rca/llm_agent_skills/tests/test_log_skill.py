from pathlib import Path

import pandas as pd

from rca_agent_skills.main import build_request
from rca_agent_skills.common.io_utils import read_json
from rca_agent_skills.common.models import AnomalyRecord
from rca_agent_skills.data_access import build_data_access
from rca_agent_skills.llm import LLMClient
from rca_agent_skills.skills.log_evidence_skill.detector import detect_log_spikes
from rca_agent_skills.skills.log_evidence_skill.parser import normalize_message
from rca_agent_skills.skills.log_evidence_skill.parser import parse_raw_log
from rca_agent_skills.skills.log_evidence_skill.skill import LogEvidenceSkill


def test_log_skill_output_format():
    root = Path(__file__).resolve().parents[1]
    payload = read_json(root / "examples" / "sample_request.json")
    request = build_request(payload, root)
    settings = request.config_bundle["settings"]
    skill = LogEvidenceSkill(settings, build_data_access(request, settings), LLMClient())
    result = skill.run(request, state=None)
    assert result.service_evidence
    assert any(item.service == "orders-db" for item in result.service_evidence)
    assert result.anomaly_records


def test_parse_raw_log_extracts_json_level_message_and_trace_ids():
    parsed = parse_raw_log(
        '{"level":"error","trace_id":"abc123","span_id":"def456","msg":"payment failed after retry"}',
        container="payment",
    )

    assert parsed["log_level"] == "ERROR"
    assert parsed["trace_id"] == "abc123"
    assert parsed["span_id"] == "def456"
    assert parsed["message"] == "payment failed after retry"
    assert parsed["log_type"] == "exception_log"
    assert "payment failed after retry" in parsed["message_template"]


def test_parse_raw_log_extracts_spring_style_level_and_message():
    parsed = parse_raw_log(
        "2026-04-27T06:10:01.123Z  WARN [orders,traceId:abc,spanId:def] --- [nio] c.orders.Controller : retry payment timeout",
        container="orders",
    )

    assert parsed["log_level"] == "WARN"
    assert parsed["trace_id"] == "abc"
    assert parsed["span_id"] == "def"
    assert parsed["message"] == "retry payment timeout"
    assert parsed["log_type"] == "timeout_log"


def test_normalize_message_collapses_numeric_duration_units():
    assert normalize_message("[RepositoryTracingAspect] Order saved successfully, duration: 5ms") == normalize_message(
        "[RepositoryTracingAspect] Order saved successfully, duration: 7ms"
    )
    assert normalize_message("query completed in 12 ms") == "query completed in <num> ms"
    assert normalize_message("took=480.47µs") == normalize_message("took=12.106µs")
    assert normalize_message("took=480.47us") == "took=<num> us"


def test_normalize_message_collapses_embedded_timestamps():
    assert normalize_message('time="2026-05-01T08:12:21Z" level=info msg="metrics updated" duration=5 ms') == normalize_message(
        'time="2026-05-01T08:13:36Z" level=info msg="metrics updated" duration=9 ms'
    )
    assert (
        normalize_message('time="2026-05-01T08:12:21Z" level=info msg="metrics updated" duration=5 ms')
        == 'time="<timestamp>" level=info msg="metrics updated" duration=<num> ms'
    )


def test_log_keywords_are_configurable():
    baseline = pd.DataFrame(
        [{"timestamp": "2026-04-27T00:00:00Z", "service": "payment", "pod": "payment-1", "log_level": "INFO", "message": "ok", "message_template": "ok"}]
    )
    abnormal = pd.DataFrame(
        [
            {
                "timestamp": "2026-04-27T00:01:00Z",
                "service": "payment",
                "pod": "payment-1",
                "log_level": "INFO",
                "message": "circuit breaker open",
                "message_template": normalize_message("circuit breaker open"),
            },
            {
                "timestamp": "2026-04-27T00:02:00Z",
                "service": "payment",
                "pod": "payment-1",
                "log_level": "INFO",
                "message": "circuit breaker open",
                "message_template": normalize_message("circuit breaker open"),
            },
            {
                "timestamp": "2026-04-27T00:03:00Z",
                "service": "payment",
                "pod": "payment-1",
                "log_level": "INFO",
                "message": "circuit breaker open",
                "message_template": normalize_message("circuit breaker open"),
            },
        ]
    )

    features = detect_log_spikes(
        baseline,
        abnormal,
        "service",
        {"minimum_count": 3, "log_spike_ratio_threshold": 2.0, "log_keywords": ["circuit breaker"]},
    )

    assert any(item.pattern_type == "keyword_spike" and item.pattern_value == "circuit breaker" for item in features)


def test_duration_templates_are_aggregated_across_different_values():
    baseline = pd.DataFrame(
        [
            {
                "timestamp": "2026-04-27T00:00:00Z",
                "service": "orders",
                "pod": "orders-1",
                "log_level": "INFO",
                "message": "[RepositoryTracingAspect] Order saved successfully, duration: 5ms",
                "message_template": normalize_message("[RepositoryTracingAspect] Order saved successfully, duration: 5ms"),
            }
        ]
    )
    abnormal = pd.DataFrame(
        [
            {
                "timestamp": f"2026-04-27T00:0{index}:00Z",
                "service": "orders",
                "pod": "orders-1",
                "log_level": "INFO",
                "message": f"[RepositoryTracingAspect] Order saved successfully, duration: {duration}ms",
                "message_template": normalize_message(f"[RepositoryTracingAspect] Order saved successfully, duration: {duration}ms"),
            }
            for index, duration in enumerate([5, 6, 7, 9], start=1)
        ]
    )

    features = detect_log_spikes(
        baseline,
        abnormal,
        "pod",
        {"minimum_count": 3, "log_spike_ratio_threshold": 2.0},
    )

    template_features = [item for item in features if item.pattern_type == "template_spike"]
    assert len(template_features) == 1
    assert template_features[0].pattern_value == "[repositorytracingaspect] order saved successfully, duration: <num> ms"
    assert template_features[0].baseline_count == 1
    assert template_features[0].abnormal_count == 4


def test_log_skill_prefers_pod_patterns_over_duplicate_service_patterns():
    service_record = AnomalyRecord(
        source="log",
        entity_type="service",
        entity_name="orders",
        metric_or_pattern="template_spike",
        abnormal_value=141,
        baseline_value=0,
        delta=141,
        zscore=None,
        severity=0.84,
        summary="template_spike='java.lang.runtimeexception: null' count 0->141",
        metadata={"service": "orders", "pod": None, "pattern_value": "java.lang.runtimeexception: null"},
    )
    pod_record = AnomalyRecord(
        source="log",
        entity_type="pod",
        entity_name="orders-1",
        metric_or_pattern="template_spike",
        abnormal_value=141,
        baseline_value=0,
        delta=141,
        zscore=None,
        severity=0.84,
        summary="template_spike='java.lang.runtimeexception: null' count 0->141",
        metadata={"service": "orders", "pod": "orders-1", "pattern_value": "java.lang.runtimeexception: null"},
    )
    unrelated_service_record = AnomalyRecord(
        source="log",
        entity_type="service",
        entity_name="front-end",
        metric_or_pattern="template_spike",
        abnormal_value=20,
        baseline_value=0,
        delta=20,
        zscore=None,
        severity=0.7,
        summary="template_spike='request completed with 5xx response' count 0->20",
        metadata={"service": "front-end", "pod": None, "pattern_value": "request completed with 5xx response"},
    )
    skill = LogEvidenceSkill({}, data_access=None, llm_client=None)

    records = skill._remove_service_duplicates_when_pod_data_exists(
        [service_record, pod_record, unrelated_service_record]
    )
    service_evidence = skill._group_service_evidence(records)

    assert service_record not in records
    assert pod_record in records
    assert unrelated_service_record in records
    orders_evidence = next(item for item in service_evidence if item.service == "orders")
    assert orders_evidence.anomaly_records == [pod_record]


def test_background_template_hints_are_configurable():
    baseline = pd.DataFrame(
        [{"timestamp": "2026-04-27T00:00:00Z", "service": "orders-db", "pod": "orders-db-0", "log_level": "INFO", "message": "ok", "message_template": "ok"}]
    )
    abnormal_rows = []
    for idx in range(6):
        message = "cache heartbeat healthy"
        abnormal_rows.append(
            {
                "timestamp": f"2026-04-27T00:0{idx}:00Z",
                "service": "orders-db",
                "pod": "orders-db-0",
                "log_level": "INFO",
                "message": message,
                "message_template": normalize_message(message),
            }
        )
    abnormal = pd.DataFrame(abnormal_rows)

    features = detect_log_spikes(
        baseline,
        abnormal,
        "service",
        {"minimum_count": 3, "log_spike_ratio_threshold": 2.0, "log_background_template_hints": ["cache heartbeat"]},
    )

    template_feature = next(item for item in features if item.pattern_type == "template_spike")
    assert template_feature.background_noise is True
    assert template_feature.severity <= 0.25
