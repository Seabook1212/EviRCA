from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rca_agent_skills.main import run_rca
from rca_agent_skills.outputs.writer import write_outputs


def main() -> None:
    payload = {
        "incident_id": "pod_network_loss_queue-master_001",
        "backend_mode": "api",
        "abnormal_window": {
            "start": "2026-05-15T10:10:56Z",
            "end": "2026-05-15T10:21:56Z"
        },
        "baseline_window": {
            "start": "2026-05-15T07:24:23Z",
            "end": "2026-05-15T07:39:23Z"
        },
        "api_inputs": {
            "prometheus_url": "http://34.28.33.102:30990",
            "loki_url": "http://34.28.33.102:31300",
            "jaeger_url": "http://34.28.33.102:32614",
            "namespace": "sock-shop"
        }
    }
    result = run_rca(payload, project_root=PROJECT_ROOT)
    write_outputs(result, PROJECT_ROOT / "examples" / "output_api")
    print("API mode RCA completed.")


if __name__ == "__main__":
    main()
