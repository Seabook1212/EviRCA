from pathlib import Path

from rca_agent_skills.common.io_utils import read_json, to_jsonable
from rca_agent_skills.main import run_rca


def test_orchestrator_end_to_end():
    root = Path(__file__).resolve().parents[1]
    payload = read_json(root / "examples" / "sample_request.json")
    result = run_rca(payload, project_root=root)
    data = to_jsonable(result)
    assert data["incident_id"] == "incident_sample_001"
    assert len(data["service_top5"]) <= 5
    assert len(data["pod_top5"]) <= 5
    assert isinstance(data["warnings"], list)
    assert isinstance(data["errors"], list)
    assert "reasoning_rules" in data["metadata"]
