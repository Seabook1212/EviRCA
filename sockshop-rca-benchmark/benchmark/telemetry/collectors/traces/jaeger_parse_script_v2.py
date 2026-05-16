import json
import os
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
INPUT_FILE = Path(
    os.environ.get("JAEGER_PARSE_INPUT_FILE", str(DATA_DIR / "jaeger_traces_raw.csv"))
).expanduser()
OUTPUT_FILE = Path(
    os.environ.get("JAEGER_PARSE_OUTPUT_FILE", str(DATA_DIR / "jaeger_traces_parsed.csv"))
).expanduser()


def safe_json_loads(value):
    if pd.isna(value) or value == "":
        return []
    try:
        return json.loads(value)
    except Exception:
        return []


def extract_tag(tags, key, default=""):
    for tag in tags:
        if tag.get("key") == key:
            return tag.get("value", default)
    return default


def extract_parent_span_id(references):
    for ref in references:
        if ref.get("refType") == "CHILD_OF":
            return ref.get("spanID")
    return None


def extract_exception(tags):
    exc_type = ""
    exc_msg = ""

    for tag in tags:
        if tag.get("key") in ("exception.type", "error.type"):
            exc_type = tag.get("value", "")
        if tag.get("key") in ("exception.message", "error.message"):
            exc_msg = tag.get("value", "")

    return exc_type, exc_msg


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    parsed_rows = []

    for _, row in df.iterrows():
        references = safe_json_loads(row.get("references"))
        tags = safe_json_loads(row.get("tags"))

        parent_span_id = extract_parent_span_id(references)

        span_kind = extract_tag(tags, "span.kind")
        status_code = extract_tag(tags, "http.status_code")
        status = extract_tag(tags, "otel.status_code", "SUCCESS")

        peer_service = extract_tag(tags, "peer.service")

        http_method = extract_tag(tags, "http.method")
        http_url = extract_tag(tags, "http.url")

        pod = extract_tag(tags, "pod")
        container = extract_tag(tags, "container")
        node = extract_tag(tags, "node")

        exc_type, exc_msg = extract_exception(tags)

        parsed_rows.append(
            {
                "timestamp": int(row["start_time"]),
                "trace_id": row["trace_id"],
                "span_id": row["span_id"],
                "parent_span_id": parent_span_id,
                "service": row["service"],
                "operation": row["operation"],
                "duration": row["duration"],
                "span_kind": span_kind,
                "status_code": str(status_code),
                "status": status,
                "peer_service": peer_service,
                "http_method": http_method,
                "http_url": http_url,
                "exception_type": exc_type,
                "exception_message": exc_msg,
                "pod": pod,
                "container": container,
                "node": node,
                "tags_json": row.get("tags"),
            }
        )

    parsed_df = pd.DataFrame(parsed_rows)

    parsed_df.sort_values(
        ["timestamp", "trace_id", "span_id"], inplace=True
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    parsed_df.to_csv(OUTPUT_FILE, index=False)

    print(f"[INFO] Parsed traces saved to {OUTPUT_FILE}")
    print(f"[INFO] Total spans: {len(parsed_df)}")


if __name__ == "__main__":
    main()
