import json
import csv
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_FILE = SCRIPT_DIR / "data" / "raw_data" / "container_level_raw_metrics_name.json"
OUTPUT_FILE = SCRIPT_DIR / "data" / "unique_name" / "container_level_unique_metrics_name.csv"


def extract_metric_rows(payload):
    """
    Support both:
    1) raw list: [ {...}, {...} ]
    2) wrapped object: {"status":"success","data":[ {...}, {...} ]}
    """
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        data_field = payload.get("data")
        if isinstance(data_field, list):
            return data_field
    return []


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    with INPUT_FILE.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    rows = extract_metric_rows(payload)
    metric_names = sorted(
        {
            str(item.get("__name__", "")).strip()
            for item in rows
            if isinstance(item, dict) and item.get("__name__")
        }
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric_name"])
        for name in metric_names:
            writer.writerow([name])

    print(f"Input rows: {len(rows)}")
    print(f"Total unique metrics: {len(metric_names)}")
    print(f"Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
