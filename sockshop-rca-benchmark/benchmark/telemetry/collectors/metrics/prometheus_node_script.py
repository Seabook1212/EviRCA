import os
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from urllib.parse import quote
from pathlib import Path

# ----------- Parameter Configuration -------------
PROM_URL = os.environ.get("PROM_URL", "http://34.28.33.102:30990")
WINDOW_HOURS = float(os.environ.get("PROM_WINDOW_HOURS", 0.25))  # 0.25 hours = 15 minutes
STEP = os.environ.get("PROM_STEP", "5s")
EXP_ID = os.environ.get("EXP_ID", "node_001")
EXP_NOTE = os.environ.get("EXP_NOTE", "node_metrics_exp")

end_time = datetime.now(timezone.utc)
start_time = end_time - timedelta(hours=WINDOW_HOURS)

# ----------- Query Expressions -------------
queries = {
    # CPU Metrics
    "NodeCpuUsageRate": r"""
100 - (avg by (instance) (rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)
""",
    "NodeLoadAverage1m": r"""
node_load1
""",
    "NodeLoadAverage5m": r"""
node_load5
""",
    "NodeLoadAverage15m": r"""
node_load15
""",

    # Memory Metrics
    "NodeMemoryUsageRate": r"""
100 * (
  1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)
)
""",
    "NodeMemoryAvailableBytes": r"""
node_memory_MemAvailable_bytes
""",
    "NodeMemoryTotalBytes": r"""
node_memory_MemTotal_bytes
""",
#     "NodeSwapUsageRate": r"""
# 100 * (
#   1 - (node_memory_SwapFree_bytes / node_memory_SwapTotal_bytes)
# )
# """,

    # Disk Metrics
    "NodeDiskSpaceUsageRate": r"""
100 * (
  1 - (
    node_filesystem_avail_bytes{mountpoint="/",fstype!~"tmpfs|overlay"} /
    node_filesystem_size_bytes{mountpoint="/",fstype!~"tmpfs|overlay"}
  )
)
""",
    "NodeDiskReadBytes": r"""
sum by (instance) (
  rate(node_disk_read_bytes_total[5m])
)
""",
    "NodeDiskWriteBytes": r"""
sum by (instance) (
  rate(node_disk_written_bytes_total[5m])
)
""",
    "NodeDiskIOPS": r"""
sum by (instance) (
  rate(node_disk_reads_completed_total[5m]) + rate(node_disk_writes_completed_total[5m])
)
""",
    "NodeDiskIOUtilization": r"""
100 * rate(node_disk_io_time_seconds_total[5m])
""",

    # Network Metrics
    "NodeNetworkReceiveBytes": r"""
sum by (instance) (
  rate(node_network_receive_bytes_total{device!~"lo"}[5m])
)
""",
    "NodeNetworkTransmitBytes": r"""
sum by (instance) (
  rate(node_network_transmit_bytes_total{device!~"lo"}[5m])
)
""",
    "NodeNetworkReceivePackets": r"""
sum by (instance) (
  rate(node_network_receive_packets_total{device!~"lo"}[5m])
)
""",
    "NodeNetworkTransmitPackets": r"""
sum by (instance) (
  rate(node_network_transmit_packets_total{device!~"lo"}[5m])
)
""",
    "NodeNetworkReceiveErrors": r"""
sum by (instance) (
  rate(node_network_receive_errs_total{device!~"lo"}[5m])
)
""",
    "NodeNetworkTransmitErrors": r"""
sum by (instance) (
  rate(node_network_transmit_errs_total{device!~"lo"}[5m])
)
""",

    # Process and System Metrics
    "NodeProcessesRunning": r"""
node_procs_running
""",
    "NodeProcessesBlocked": r"""
node_procs_blocked
""",
    "NodeContextSwitches": r"""
rate(node_context_switches_total[5m])
""",
    "NodeFileDescriptorsUsed": r"""
node_filefd_allocated
""",
    "NodeFileDescriptorsMax": r"""
node_filefd_maximum
""",

}

def query_range(metric_name, promql_expr):
    url = f"{PROM_URL}/api/v1/query_range"
    params = {
        "query": promql_expr,
        "start": start_time.isoformat().replace('+00:00', 'Z'),
        "end": end_time.isoformat().replace('+00:00', 'Z'),
        "step": STEP,
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()["data"]["result"]
    rows = []
    for series in data:
        instance = series["metric"].get("instance", "unknown")
        for ts, val in series["values"]:
            timestamp = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            rows.append({
                "timestamp": timestamp,
                "instance": instance,
                "metric": metric_name,
                "value": float(val)
            })
    return pd.DataFrame(rows)

# ----------- Execute Queries and Merge Results -------------
frames = []
for name, query in queries.items():
    print(f"Querying {name}...")
    df = query_range(name, query)
    frames.append(df)

# Merge data
result_df = pd.concat(frames)

# ----------- Export Results -------------
# Use absolute path based on script location
script_dir = Path(__file__).parent
output_dir = script_dir
output_dir.mkdir(parents=True, exist_ok=True)

timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = output_dir / "prometheus_node_metrics.csv"
result_df.to_csv(filename, index=False)
print(f"✅ Exported node metrics to {filename}")
