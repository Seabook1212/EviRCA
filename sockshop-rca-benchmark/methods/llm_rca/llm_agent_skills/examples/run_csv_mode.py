from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rca_agent_skills.common.io_utils import read_json
from rca_agent_skills.main import run_rca
from rca_agent_skills.outputs.writer import write_outputs


def main() -> None:
    payload = read_json(PROJECT_ROOT / "examples" / "sample_request.json")
    result = run_rca(payload, project_root=PROJECT_ROOT)
    write_outputs(result, PROJECT_ROOT / "examples" / "output_csv")
    print("CSV mode RCA completed.")


if __name__ == "__main__":
    main()
