from __future__ import annotations

from pathlib import Path
from typing import Any

from rca_agent_skills.common.io_utils import read_json


def load_topology(inline_topology: dict[str, Any] | None, topology_file: str | None = None) -> dict[str, Any]:
    if inline_topology:
        return inline_topology
    if topology_file:
        return read_json(topology_file)
    raise ValueError("Topology must be provided either inline or via topology_file")


def service_from_pod(pod: str) -> str:
    if not pod:
        return "unknown"
    tokens = pod.split("-")
    if len(tokens) >= 2 and tokens[-1].isdigit():
        return "-".join(tokens[:-1])
    if len(tokens) >= 3:
        return "-".join(tokens[:-2])
    if len(tokens) == 2:
        return tokens[0]
    return pod
