from __future__ import annotations

from rca_agent_skills.common.io_utils import to_jsonable


def format_result(result) -> dict:
    return to_jsonable(result)

