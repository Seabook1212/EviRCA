from __future__ import annotations

import json
import re
from typing import Any


_HEX_RE = re.compile(r"\b[a-f0-9]{8,}\b", re.IGNORECASE)
_ISO_TIMESTAMP_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}[t\s]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:z|[+-]\d{2}:?\d{2})?\b",
    re.IGNORECASE,
)
_TIME_OF_DAY_RE = re.compile(r"\b\d{1,2}:\d{2}:\d{2}(?:\.\d+)?(?:z)?\b", re.IGNORECASE)
_NUMBER_WITH_UNIT_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(ns|us|µs|μs|ms|s|sec|secs|second|seconds|m|minute|minutes|kb|mb|gb|bytes?)\b",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"\b\d+\b")
_IP_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
_UUID_RE = re.compile(r"\b[0-9a-f]{8}-[0-9a-f-]{27,}\b", re.IGNORECASE)
_TRACE_ID_RE = re.compile(r"\btrace[_-]?id\b\s*[:=]\s*\"?([a-f0-9]+)\"?\b", re.IGNORECASE)
_SPAN_ID_RE = re.compile(r"\bspan[_-]?id\b\s*[:=]\s*\"?([a-f0-9]+)\"?\b", re.IGNORECASE)
_LEVEL_RE = re.compile(
    r'(^|[\s,])(?:level|lvl|severity)[:=]"?(trace|debug|info|warn|warning|error|fatal|panic|[tdiwefp])"?',
    re.IGNORECASE,
)
_BACKGROUND_TEMPLATE_HINTS = (
    "connection accepted",
    "wiredtiger message",
    "wt_session.checkpoint",
    "saving checkpoint snapshot",
)
DEFAULT_KEYWORDS = (
    "timeout",
    "error",
    "exception",
    "refused",
    "reset",
    "disconnect",
    "retry",
    "oom",
    "killed",
    "partition",
)
_LEVEL_ALIASES = {
    "T": "TRACE",
    "TRACE": "TRACE",
    "D": "DEBUG",
    "DEBUG": "DEBUG",
    "I": "INFO",
    "INFO": "INFO",
    "W": "WARN",
    "WARN": "WARN",
    "WARNING": "WARN",
    "E": "ERROR",
    "ERR": "ERROR",
    "ERROR": "ERROR",
    "F": "FATAL",
    "FATAL": "FATAL",
    "P": "PANIC",
    "PANIC": "PANIC",
}


def try_parse_json(log: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(str(log))
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def normalize_log_level(level) -> str:
    if level is None:
        return "UNKNOWN"
    level_text = str(level).strip().upper()
    if not level_text:
        return "UNKNOWN"
    return _LEVEL_ALIASES.get(level_text, "UNKNOWN")


def extract_trace_info(log: str) -> tuple[str | None, str | None]:
    text = str(log)
    trace_match = _TRACE_ID_RE.search(text)
    span_match = _SPAN_ID_RE.search(text)
    return (
        trace_match.group(1) if trace_match else None,
        span_match.group(1) if span_match else None,
    )


def classify_log_level(log: str) -> str:
    text = str(log)
    level_match = _LEVEL_RE.search(text)
    if level_match:
        return normalize_log_level(level_match.group(2))

    padded = f" {text.upper()} "
    if " ERROR " in padded:
        return "ERROR"
    if " WARN " in padded or " WARNING " in padded:
        return "WARN"
    if " DEBUG " in padded:
        return "DEBUG"
    if " INFO " in padded:
        return "INFO"
    if " FATAL " in padded:
        return "FATAL"
    if " PANIC " in padded:
        return "PANIC"
    return "UNKNOWN"


def classify_log_source(container: str | None) -> str:
    container_lower = str(container or "").lower()
    if "mongo" in container_lower or "db" in container_lower or "session" in container_lower:
        return "database"
    if "rabbit" in container_lower:
        return "middleware"
    if "istio" in container_lower or "envoy" in container_lower:
        return "infrastructure"
    return "application"


def classify_log_type(message: str) -> str:
    msg = str(message).lower()
    if any(keyword in msg for keyword in ["exception", "error", "failed", "panic"]):
        return "exception_log"
    if any(keyword in msg for keyword in ["timeout", "deadline", "slow", "latency"]):
        return "timeout_log"
    if any(keyword in msg for keyword in ["retry", "reconnecting", "backoff"]):
        return "retry_log"
    if any(keyword in msg for keyword in ["connection", "connected", "disconnected"]):
        return "connection_log"
    if any(keyword in msg for keyword in ["queue", "publish", "consume", "message"]):
        return "queue_log"
    return "general_log"


def clean_message(log: str, parsed_json: dict[str, Any] | None) -> str:
    if parsed_json:
        return str(parsed_json.get("msg") or parsed_json.get("message") or log)

    parts = str(log).split(" : ")
    if len(parts) > 1:
        return parts[-1]
    return str(log)


def parse_raw_log(raw_log: str, container: str | None = None) -> dict[str, Any]:
    parsed_json = try_parse_json(raw_log)
    trace_id, span_id = extract_trace_info(raw_log)
    log_level = classify_log_level(raw_log)
    message = clean_message(raw_log, parsed_json)

    if parsed_json:
        trace_id = parsed_json.get("trace_id") or parsed_json.get("traceId") or parsed_json.get("traceid") or trace_id
        span_id = parsed_json.get("span_id") or parsed_json.get("spanId") or parsed_json.get("spanid") or span_id
        json_level = parsed_json.get("level") or parsed_json.get("lvl") or parsed_json.get("severity") or parsed_json.get("s")
        if json_level:
            log_level = normalize_log_level(json_level)

    return {
        "trace_id": trace_id,
        "span_id": span_id,
        "log_level": log_level,
        "log_source": classify_log_source(container),
        "log_type": classify_log_type(message),
        "message": message,
        "raw_log": str(raw_log),
        "message_template": normalize_message(message),
    }


def normalize_message(message: str) -> str:
    text = str(message).lower().strip()
    text = _UUID_RE.sub("<uuid>", text)
    text = _ISO_TIMESTAMP_RE.sub("<timestamp>", text)
    text = _TIME_OF_DAY_RE.sub("<time>", text)
    text = _IP_RE.sub("<ip>", text)
    text = _HEX_RE.sub("<hex>", text)
    text = _NUMBER_WITH_UNIT_RE.sub(lambda match: f"<num> {match.group(1).lower()}", text)
    text = _NUMBER_RE.sub("<num>", text)
    return text


def extract_keywords(message: str, keywords: list[str] | tuple[str, ...] | None = None) -> list[str]:
    text = normalize_message(message)
    matched = []
    for token in keywords or DEFAULT_KEYWORDS:
        normalized_token = normalize_message(str(token))
        if normalized_token and normalized_token in text:
            matched.append(str(token).lower())
    return matched


def is_background_template(message_template: str, hints: list[str] | tuple[str, ...] | None = None) -> bool:
    text = normalize_message(message_template)
    return any(normalize_message(token) in text for token in hints or _BACKGROUND_TEMPLATE_HINTS)
