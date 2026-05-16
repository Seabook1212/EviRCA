from __future__ import annotations

import argparse
import json

from rca_agent_skills.common.io_utils import to_jsonable
from rca_agent_skills.main import run_request_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the V1 skill-based RCA agent.")
    parser.add_argument("request_file", help="Path to RCA request JSON")
    args = parser.parse_args()
    result = run_request_file(args.request_file)
    print(json.dumps(to_jsonable(result), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
