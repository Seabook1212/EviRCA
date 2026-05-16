from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMRankRequest:
    prompt_name: str
    context: dict[str, Any]
    candidates: list[dict[str, Any]]


@dataclass
class LLMRankResponse:
    rankings: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    raw_request: dict[str, Any] | None = None
    raw_response: dict[str, Any] | None = None
