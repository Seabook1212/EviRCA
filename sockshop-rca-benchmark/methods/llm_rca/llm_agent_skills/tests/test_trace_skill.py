from pathlib import Path

import pandas as pd

from rca_agent_skills.main import build_request
from rca_agent_skills.common.io_utils import read_json
from rca_agent_skills.data_access import build_data_access
from rca_agent_skills.llm import LLMClient
from rca_agent_skills.skills.trace_evidence_skill.detector import detect_trace_anomalies
from rca_agent_skills.skills.trace_evidence_skill.skill import TraceEvidenceSkill


def test_trace_skill_output_format():
    root = Path(__file__).resolve().parents[1]
    payload = read_json(root / "examples" / "sample_request.json")
    request = build_request(payload, root)
    settings = request.config_bundle["settings"]
    skill = TraceEvidenceSkill(settings, build_data_access(request, settings), LLMClient())
    result = skill.run(request, state=None)
    assert result.service_evidence
    assert result.anomaly_records
    assert "propagation_hints" in result.metadata


def _client_span(service: str, peer_service: str, duration: float, status: str = "SUCCESS", status_code: str = "200"):
    return {
        "timestamp": "2026-04-27T00:00:00Z",
        "trace_id": "trace-1",
        "span_id": "span-1",
        "parent_span_id": "",
        "service": service,
        "operation": "GET",
        "duration": duration,
        "span_kind": "client",
        "status_code": status_code,
        "status": status,
        "peer_service": peer_service,
        "pod": f"{service}-pod",
    }


def test_zero_baseline_single_trace_failure_does_not_create_edge_failure_spike():
    baseline = pd.DataFrame([_client_span("front-end", "orders", 10.0) for _ in range(3)])
    abnormal = pd.DataFrame(
        [
            _client_span("front-end", "orders", 10.0, "ERROR", "500"),
            _client_span("front-end", "orders", 10.0),
            _client_span("front-end", "orders", 10.0),
        ]
    )

    features = detect_trace_anomalies(
        baseline,
        abnormal,
        {
            "minimum_count": 3,
            "trace_spike_ratio_threshold": 1.8,
            "trace_min_failure_count": 2,
            "trace_min_failure_rate": 0.2,
        },
    )

    assert not any(item.anomaly_type == "edge_failure_spike" for item in features)


def test_zero_baseline_repeated_trace_failures_create_bounded_edge_failure_spike():
    baseline = pd.DataFrame([_client_span("front-end", "orders", 10.0) for _ in range(5)])
    abnormal = pd.DataFrame(
        [
            _client_span("front-end", "orders", 10.0, "ERROR", "500"),
            _client_span("front-end", "orders", 10.0, "ERROR", "500"),
            _client_span("front-end", "orders", 10.0),
            _client_span("front-end", "orders", 10.0),
            _client_span("front-end", "orders", 10.0),
        ]
    )

    features = detect_trace_anomalies(
        baseline,
        abnormal,
        {
            "minimum_count": 3,
            "trace_spike_ratio_threshold": 1.8,
            "trace_min_failure_count": 2,
            "trace_min_failure_rate": 0.2,
        },
    )
    failure = next(item for item in features if item.anomaly_type == "edge_failure_spike")

    assert failure.ratio == 0.0
    assert 0.0 < failure.severity < 1.0
