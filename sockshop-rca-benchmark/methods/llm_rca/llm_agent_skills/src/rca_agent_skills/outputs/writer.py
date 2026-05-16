from __future__ import annotations

from pathlib import Path

from rca_agent_skills.common.io_utils import ensure_dir, write_json
from .formatter import format_result
from .report import render_text_report


def _with_incident_suffix(filename: str, incident_id: str) -> str:
    path = Path(filename)
    return f"{path.stem}_{incident_id}{path.suffix}"


def write_outputs(result, output_dir: str | Path) -> None:
    out = ensure_dir(Path(output_dir))
    result_name = _with_incident_suffix("rca_result.json", result.incident_id)
    report_name = _with_incident_suffix("rca_report.txt", result.incident_id)
    write_json(out / result_name, format_result(result))
    (out / report_name).write_text(render_text_report(result), encoding="utf-8")
