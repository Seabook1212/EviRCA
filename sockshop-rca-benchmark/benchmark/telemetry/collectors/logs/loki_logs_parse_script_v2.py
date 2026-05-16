import json
import os
from pathlib import Path

import pandas as pd
import re

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
INPUT_FILE = Path(
    os.environ.get("LOKI_PARSE_INPUT_FILE", str(DATA_DIR / "loki_logs_raw.csv"))
).expanduser()
OUTPUT_FILE = Path(
    os.environ.get("LOKI_PARSE_OUTPUT_FILE", str(DATA_DIR / "loki_logs_parsed.csv"))
).expanduser()


# ---------- Basic utilities ----------
def try_parse_json(log):
    try:
        return json.loads(log)
    except Exception:
        return None


def extract_trace_info(log):
    trace_id = None
    span_id = None

    trace_match = re.search(r"\btrace[_-]?id\b\s*[:=]\s*\"?([a-f0-9]+)\"?\b", str(log), re.IGNORECASE)
    span_match = re.search(r"\bspan[_-]?id\b\s*[:=]\s*\"?([a-f0-9]+)\"?\b", str(log), re.IGNORECASE)

    if trace_match:
        trace_id = trace_match.group(1)

    if span_match:
        span_id = span_match.group(1)

    return trace_id, span_id


def normalize_log_level(level):
    if level is None:
        return "UNKNOWN"

    level_text = str(level).strip().upper()
    if not level_text:
        return "UNKNOWN"

    aliases = {
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
    return aliases.get(level_text, "UNKNOWN")


def classify_log_level(log):
    level_match = re.search(
        r'(^|[\s,])(?:level|lvl|severity)[:=]"?(trace|debug|info|warn|warning|error|fatal|panic|[tdiwefp])"?',
        str(log),
        re.IGNORECASE,
    )
    if level_match:
        return normalize_log_level(level_match.group(2))

    log_upper = str(log).upper()

    if " ERROR " in log_upper:
        return "ERROR"
    if " WARN " in log_upper or " WARNING " in log_upper:
        return "WARN"
    if " DEBUG " in log_upper:
        return "DEBUG"
    if " INFO " in log_upper:
        return "INFO"
    if " FATAL " in log_upper:
        return "FATAL"
    if " PANIC " in log_upper:
        return "PANIC"

    return "UNKNOWN"


def classify_log_source(container):
    container_lower = str(container).lower()

    if "mongo" in container_lower or "db" in container_lower or "session" in container_lower:
        return "database"

    if "rabbit" in container_lower:
        return "middleware"

    if "istio" in container_lower or "envoy" in container_lower:
        return "infrastructure"

    return "application"


def classify_log_type(message):
    msg = message.lower()

    if any(k in msg for k in ["exception", "error", "failed", "panic"]):
        return "exception_log"

    if any(k in msg for k in ["timeout", "deadline", "slow", "latency"]):
        return "timeout_log"

    if any(k in msg for k in ["retry", "reconnecting", "backoff"]):
        return "retry_log"

    if any(k in msg for k in ["connection", "connected", "disconnected"]):
        return "connection_log"

    if any(k in msg for k in ["queue", "publish", "consume", "message"]):
        return "queue_log"

    return "general_log"


def clean_message(log, parsed_json):
    if parsed_json:
        return parsed_json.get("msg", str(log))

    # Spring Boot style
    parts = log.split(" : ")
    if len(parts) > 1:
        return parts[-1]

    return log


# ---------- Main logic ----------
def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    parsed_rows = []

    for _, row in df.iterrows():

        raw_log = str(row["log"])
        parsed_json = try_parse_json(raw_log)

        trace_id, span_id = extract_trace_info(raw_log)
        if isinstance(parsed_json, dict):
            trace_id = (
                parsed_json.get("trace_id")
                or parsed_json.get("traceId")
                or parsed_json.get("traceid")
                or trace_id
            )
            span_id = (
                parsed_json.get("span_id")
                or parsed_json.get("spanId")
                or parsed_json.get("spanid")
                or span_id
            )

        log_level = classify_log_level(raw_log)
        log_source = classify_log_source(row["container"])

        message = clean_message(raw_log, parsed_json)

        if parsed_json:
            json_level = parsed_json.get("level") or parsed_json.get("lvl") or parsed_json.get("severity") or parsed_json.get("s")
            if json_level:
                log_level = normalize_log_level(json_level)
            message = parsed_json.get("msg", message)

        log_type = classify_log_type(message)

        parsed_rows.append(
            {
                "timestamp": row["timestamp"],
                "trace_id": trace_id,
                "span_id": span_id,
                "service": row["container"],   # Can be mapped to a service name later
                "node": row["node"],
                "pod": row["pod"],
                "container": row["container"],
                "log_level": log_level,
                "log_source": log_source,
                "log_type": log_type,
                "message": message,
                "raw_log": raw_log,
            }
        )

    parsed_df = pd.DataFrame(parsed_rows)

    parsed_df.sort_values(["timestamp"], inplace=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    parsed_df.to_csv(OUTPUT_FILE, index=False)

    print(f"[INFO] Logs parsed → {OUTPUT_FILE}")
    print(f"[INFO] Total logs: {len(parsed_df)}")


if __name__ == "__main__":
    main()
