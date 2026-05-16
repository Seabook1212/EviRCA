import os
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ===== 0. Prometheus Basic Configuration =====
PROM_URL = os.environ.get("PROM_URL", "http://34.28.33.102:30990")
WINDOW_HOURS = float(os.environ.get("PROM_WINDOW_HOURS", 0.25))  # 0.25 hours = 15 minutes
STEP = os.environ.get("PROM_STEP", "5s")
EXP_ID = os.environ.get("EXP_ID", "pod_specific_001")

end_time = datetime.now(timezone.utc)
start_time = end_time - timedelta(hours=WINDOW_HOURS)

# ===== 1. Pod Categorization by Service Type =====
POD_CATEGORIES = {
    "java": {
        "pods": ["shipping", "queue-master", "orders", "carts"],
        "metrics_suffix": "_java"
    },
    "go": {
        "pods": ["payment", "catalogue", "user"],
        "metrics_suffix": "_go"
    },
    "nodejs": {
        "pods": ["front-end"],
        "metrics_suffix": "_nodejs"
    },
    "rabbitmq": {
        "pods": ["rabbitmq"],
        "metrics_suffix": "_rabbitmq"
    },
    "mongodb": {
        "pods": ["carts-db", "user-db", "orders-db"],
        "metrics_suffix": "_mongodb"
    },
    "redis": {
        "pods": ["session-db"],
        "metrics_suffix": "_redis"
    },
    "mysql": {
        "pods": ["catalogue-db"],
        "metrics_suffix": "_mysql"
    }
}

# ===== 2. Common Metrics for All Pods =====
COMMON_QUERIES = {
    "pod_cpu_cores": r"""
sum by (namespace, pod) (
  rate(container_cpu_usage_seconds_total{
    namespace="sock-shop",
    container!="POD",
    container!="",
    container!="istio-proxy",
    pod=~"{pod_regex}"
  }[5m])
)
""",
    "pod_memory_mib": r"""
sum by (namespace, pod) (
  container_memory_working_set_bytes{
    namespace="sock-shop",
    container!="POD",
    container!="",
    container!="istio-proxy",
    pod=~"{pod_regex}"
  }
) / 1024^2
""",
    "pod_net_rx_kbps": r"""
sum by (namespace, pod) (
  rate(container_network_receive_bytes_total{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }[5m])
) / 1024
""",
    "pod_net_tx_kbps": r"""
sum by (namespace, pod) (
  rate(container_network_transmit_bytes_total{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }[5m])
) / 1024
""",
    "pod_container_restarts": r"""
sum by (namespace, pod) (
  increase(kube_pod_container_status_restarts_total{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }[10m])
)
"""
}

# ===== 2b. Istio Service Mesh Metrics (for application services) =====
ISTIO_QUERIES = {
    "pod_request_success_rate": r"""
label_replace(
  (
    sum by (destination_workload, destination_canonical_revision, destination_app, destination_service, destination_service_name, destination_workload_namespace, pod) (
      rate(istio_requests_total{
        reporter="destination",
        destination_workload_namespace="sock-shop",
        destination_workload=~"{pod_regex_plain}",
        response_code=~"200|201|202|300|304"
      }[1m])
    )
    /
    sum by (destination_workload, destination_canonical_revision, destination_app, destination_service, destination_service_name, destination_workload_namespace, pod) (
      rate(istio_requests_total{
        reporter="destination",
        destination_workload_namespace="sock-shop",
        destination_workload=~"{pod_regex_plain}"
      }[1m])
    )
  ) * 100,
  "namespace", "$1", "destination_workload_namespace", "(.*)"
)
""",
    "pod_request_latency_p90": r"""
label_replace(
  histogram_quantile(0.90, sum by(le, destination_workload, destination_workload_namespace, pod) (
    rate(istio_request_duration_milliseconds_bucket{
      reporter="destination",
      destination_workload_namespace="sock-shop",
      destination_workload=~"{pod_regex_plain}"
    }[1m])
  )),
  "namespace", "$1", "destination_workload_namespace", "(.*)"
)
""",
    "pod_request_latency_p95": r"""
label_replace(
  histogram_quantile(0.95, sum by(le, destination_workload, destination_workload_namespace, pod) (
    rate(istio_request_duration_milliseconds_bucket{
      reporter="destination",
      destination_workload_namespace="sock-shop",
      destination_workload=~"{pod_regex_plain}"
    }[1m])
  )),
  "namespace", "$1", "destination_workload_namespace", "(.*)"
)
""",
    "pod_request_latency_p99": r"""
label_replace(
  histogram_quantile(0.99, sum by(le, destination_workload, destination_workload_namespace, pod) (
    rate(istio_request_duration_milliseconds_bucket{
      reporter="destination",
      destination_workload_namespace="sock-shop",
      destination_workload=~"{pod_regex_plain}"
    }[1m])
  )),
  "namespace", "$1", "destination_workload_namespace", "(.*)"
)
""",
    "pod_request_qps": r"""
label_replace(
  sum by(destination_workload, destination_workload_namespace, pod) (
    rate(istio_requests_total{
      reporter="destination",
      destination_workload_namespace="sock-shop",
      destination_workload=~"{pod_regex_plain}"
    }[1m])
  ),
  "namespace", "$1", "destination_workload_namespace", "(.*)"
)
"""
}

# ===== 3. Java-Specific Metrics =====
JAVA_QUERIES = {
    # Thread pool executor metrics (CRITICAL - as requested)
    "executor_active_threads": r"""
sum by (namespace, pod, name) (
  executor_active_threads{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }
)
""",
    "executor_pool_size_threads": r"""
sum by (namespace, pod, name) (
  executor_pool_size_threads{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }
)
""",
    "executor_pool_max_threads": r"""
sum by (namespace, pod, name) (
  executor_pool_max_threads{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }
)
""",
    "executor_queued_tasks": r"""
sum by (namespace, pod, name) (
  executor_queued_tasks{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }
)
""",
    "executor_completed_tasks_total": r"""
sum by (namespace, pod, name) (
  rate(executor_completed_tasks_total{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }[5m])
)
""",

    # HTTP server request metrics (application performance)
    "http_server_requests_seconds_count": r"""
sum by (namespace, pod, method, status, outcome) (
  rate(http_server_requests_seconds_count{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }[5m])
)
""",
    "http_server_requests_seconds_sum": r"""
sum by (namespace, pod, method, status, outcome) (
  rate(http_server_requests_seconds_sum{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }[5m])
)
""",

    # JVM thread metrics (important for thread pool monitoring)
    "jvm_threads_live_threads": r"""
sum by (namespace, pod) (
  jvm_threads_live_threads{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }
)
""",
    "jvm_threads_daemon_threads": r"""
sum by (namespace, pod) (
  jvm_threads_daemon_threads{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }
)
""",
    "jvm_threads_peak_threads": r"""
sum by (namespace, pod) (
  jvm_threads_peak_threads{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }
)
""",
    "jvm_threads_states_threads": r"""
sum by (namespace, pod, state) (
  jvm_threads_states_threads{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }
)
""",

    # JVM memory metrics
    "jvm_memory_used_bytes": r"""
sum by (namespace, pod, area) (
  jvm_memory_used_bytes{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }
)
""",
    "jvm_memory_max_bytes": r"""
sum by (namespace, pod, area) (
  jvm_memory_max_bytes{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }
)
""",

    # JVM GC metrics (performance monitoring)
    "jvm_gc_pause_seconds_sum": r"""
sum by (namespace, pod) (
  rate(jvm_gc_pause_seconds_sum{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }[5m])
)
""",
    "jvm_gc_pause_seconds_count": r"""
sum by (namespace, pod) (
  rate(jvm_gc_pause_seconds_count{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }[5m])
)
""",

    # Spring Data repository metrics (database operation performance)
    "spring_data_repository_invocations_seconds_count": r"""
sum by (namespace, pod, repository, method, state) (
  rate(spring_data_repository_invocations_seconds_count{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }[5m])
)
""",
    "spring_data_repository_invocations_seconds_sum": r"""
sum by (namespace, pod, repository, method, state) (
  rate(spring_data_repository_invocations_seconds_sum{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }[5m])
)
""",

    # System metrics (overall system health)
    "system_cpu_usage": r"""
sum by (namespace, pod) (
  system_cpu_usage{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }
)
""",
    "system_load_average_1m": r"""
sum by (namespace, pod) (
  system_load_average_1m{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }
)
""",
    "process_cpu_usage": r"""
sum by (namespace, pod) (
  process_cpu_usage{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }
)
"""
}

# ===== 4. Go-Specific Metrics =====
GO_QUERIES = {
    # Goroutine metrics (critical for Go concurrency)
    "go_goroutines": r"""
sum by (namespace, pod) (
  go_goroutines{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }
)
""",
    "go_threads": r"""
sum by (namespace, pod) (
  go_threads{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }
)
""",

    # Memory allocation metrics
    "go_memstats_alloc_bytes": r"""
sum by (namespace, pod) (
  go_memstats_alloc_bytes{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }
)
""",
    "go_memstats_heap_alloc_bytes": r"""
sum by (namespace, pod) (
  go_memstats_heap_alloc_bytes{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }
)
""",
    "go_memstats_heap_inuse_bytes": r"""
sum by (namespace, pod) (
  go_memstats_heap_inuse_bytes{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }
)
""",
    "go_memstats_heap_idle_bytes": r"""
sum by (namespace, pod) (
  go_memstats_heap_idle_bytes{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }
)
""",
    "go_memstats_heap_sys_bytes": r"""
sum by (namespace, pod) (
  go_memstats_heap_sys_bytes{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }
)
""",
    "go_memstats_sys_bytes": r"""
sum by (namespace, pod) (
  go_memstats_sys_bytes{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }
)
""",

    # GC metrics (important for performance)
    "go_gc_duration_seconds": r"""
sum by (namespace, pod) (
  rate(go_gc_duration_seconds_sum{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }[5m])
)
""",
    "go_gc_duration_seconds_count": r"""
sum by (namespace, pod) (
  rate(go_memstats_alloc_bytes_total{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }[5m])
)
""",

    # Stack and heap objects
    "go_memstats_stack_inuse_bytes": r"""
sum by (namespace, pod) (
  go_memstats_stack_inuse_bytes{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }
)
""",
    "go_memstats_heap_objects": r"""
sum by (namespace, pod) (
  go_memstats_heap_objects{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }
)
"""
}

# ===== 5. Node.js-Specific Metrics =====
NODEJS_QUERIES = {
    # Memory metrics
    "nodejs_nodejs_heap_size_used_bytes": r"""
sum by (namespace, pod) (
  nodejs_nodejs_heap_size_used_bytes{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }
)
""",
    "nodejs_nodejs_heap_size_total_bytes": r"""
sum by (namespace, pod) (
  nodejs_nodejs_heap_size_total_bytes{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }
)
""",
    "nodejs_process_resident_memory_bytes": r"""
sum by (namespace, pod) (
  nodejs_process_resident_memory_bytes{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }
)
""",
    "nodejs_nodejs_external_memory_bytes": r"""
sum by (namespace, pod) (
  nodejs_nodejs_external_memory_bytes{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }
)
""",

    # Event loop metrics (critical for Node.js performance)
    "nodejs_nodejs_eventloop_lag_seconds": r"""
sum by (namespace, pod) (
  nodejs_nodejs_eventloop_lag_seconds{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }
)
""",
    "nodejs_nodejs_eventloop_lag_p99_seconds": r"""
sum by (namespace, pod) (
  nodejs_nodejs_eventloop_lag_p99_seconds{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }
)
""",

    # Active resources
    "nodejs_nodejs_active_handles_total": r"""
sum by (namespace, pod) (
  nodejs_nodejs_active_handles_total{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }
)
""",
    "nodejs_nodejs_active_requests_total": r"""
sum by (namespace, pod) (
  nodejs_nodejs_active_requests_total{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }
)
""",

    # CPU metrics
    "nodejs_process_cpu_seconds_total": r"""
sum by (namespace, pod) (
  rate(nodejs_process_cpu_seconds_total{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }[5m])
)
""",

    # GC metrics
    "nodejs_nodejs_gc_duration_seconds": r"""
sum by (namespace, pod) (
  rate(nodejs_nodejs_gc_duration_seconds_sum{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }[5m])
)
""",

    # HTTP request metrics
    "http_request_duration_seconds": r"""
sum by (namespace, pod) (
  rate(http_request_duration_seconds_sum{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }[5m])
)
""",
    "http_requests_total": r"""
sum by (namespace, pod) (
  rate(http_requests_total{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }[5m])
)
""",
    "http_request_errors_total": r"""
sum by (namespace, pod) (
  rate(http_request_errors_total{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }[5m])
)
"""
}

# ===== 6. RabbitMQ-Specific Metrics =====
RABBITMQ_QUERIES = {
    # Queue depth metrics (critical for monitoring message backlog)
    "rabbitmq_queue_messages": r"""
sum by (namespace, pod, queue) (
  rabbitmq_queue_messages{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }
)
""",
    "rabbitmq_queue_messages_ready": r"""
sum by (namespace, pod, queue) (
  rabbitmq_queue_messages_ready{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }
)
""",
    "rabbitmq_queue_messages_unacknowledged": r"""
sum by (namespace, pod, queue) (
  rabbitmq_queue_messages_unacknowledged{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }
)
""",

    # Queue consumer metrics
    "rabbitmq_queue_consumers": r"""
sum by (namespace, pod, queue) (
  rabbitmq_queue_consumers{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }
)
""",

    # Message throughput metrics
    "rabbitmq_queue_messages_published_total": r"""
sum by (namespace, pod, queue) (
  rate(rabbitmq_queue_messages_published_total{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }[5m])
)
""",
    "rabbitmq_queue_messages_delivered_total": r"""
sum by (namespace, pod, queue) (
  rate(rabbitmq_queue_messages_delivered_total{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }[5m])
)
""",
    "rabbitmq_queue_messages_ack_total": r"""
sum by (namespace, pod, queue) (
  rate(rabbitmq_queue_messages_ack_total{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }[5m])
)
""",

    # Connection metrics (important for monitoring client connections)
    "rabbitmq_connections": r"""
sum by (namespace, pod) (
  rabbitmq_connections{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }
)
""",
    "rabbitmq_channels": r"""
sum by (namespace, pod) (
  rabbitmq_channels{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }
)
""",

    # Node health metrics (critical for RabbitMQ node monitoring)
    "rabbitmq_node_mem_used": r"""
sum by (namespace, pod) (
  rabbitmq_node_mem_used{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }
)
""",
    "rabbitmq_node_mem_limit": r"""
sum by (namespace, pod) (
  rabbitmq_node_mem_limit{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }
)
""",
    "rabbitmq_node_mem_alarm": r"""
sum by (namespace, pod) (
  rabbitmq_node_mem_alarm{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }
)
""",
    "rabbitmq_node_disk_free": r"""
sum by (namespace, pod) (
  rabbitmq_node_disk_free{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }
)
""",
    "rabbitmq_node_disk_free_alarm": r"""
sum by (namespace, pod) (
  rabbitmq_node_disk_free_alarm{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }
)
""",

    # File descriptor metrics (important for connection limits)
    "rabbitmq_fd_used": r"""
sum by (namespace, pod) (
  rabbitmq_fd_used{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }
)
""",
    "rabbitmq_fd_available": r"""
sum by (namespace, pod) (
  rabbitmq_fd_available{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }
)
"""
}

# ===== 7. MongoDB-Specific Metrics =====
MONGODB_QUERIES = {
    # Driver connection pool metrics
    "mongodb_driver_pool_size": r"""
sum by (namespace, pod) (
  mongodb_driver_pool_size{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }
)
""",
    "mongodb_driver_pool_checkedout": r"""
sum by (namespace, pod) (
  mongodb_driver_pool_checkedout{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }
)
""",
    "mongodb_driver_pool_waitqueuesize": r"""
sum by (namespace, pod) (
  mongodb_driver_pool_waitqueuesize{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }
)
""",
    "mongodb_driver_pool_checkoutfailed_total": r"""
sum by (namespace, pod) (
  rate(mongodb_driver_pool_checkoutfailed_total{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }[5m])
)
""",

    # Driver command metrics
    "mongodb_driver_commands_seconds_count": r"""
sum by (namespace, pod) (
  rate(mongodb_driver_commands_seconds_count{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }[5m])
)
""",
    "mongodb_driver_commands_seconds_sum": r"""
sum by (namespace, pod) (
  rate(mongodb_driver_commands_seconds_sum{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }[5m])
)
""",

    # Top metrics - aggregated across all collections
    "mongodb_top_queries_count": r"""
sum by (namespace, pod) (
  rate(mongodb_top_queries_count{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }[5m])
)
""",
    "mongodb_top_queries_time": r"""
sum by (namespace, pod) (
  rate(mongodb_top_queries_time{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }[5m])
)
""",
    "mongodb_top_insert_count": r"""
sum by (namespace, pod) (
  rate(mongodb_top_insert_count{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }[5m])
)
""",
    "mongodb_top_insert_time": r"""
sum by (namespace, pod) (
  rate(mongodb_top_insert_time{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }[5m])
)
""",
    "mongodb_top_update_count": r"""
sum by (namespace, pod) (
  rate(mongodb_top_update_count{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }[5m])
)
""",
    "mongodb_top_update_time": r"""
sum by (namespace, pod) (
  rate(mongodb_top_update_time{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }[5m])
)
""",
    "mongodb_top_remove_count": r"""
sum by (namespace, pod) (
  rate(mongodb_top_remove_count{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }[5m])
)
""",
    "mongodb_top_remove_time": r"""
sum by (namespace, pod) (
  rate(mongodb_top_remove_time{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }[5m])
)
""",

    # Lock metrics
    "mongodb_top_readLock_count": r"""
sum by (namespace, pod) (
  rate(mongodb_top_readLock_count{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }[5m])
)
""",
    "mongodb_top_readLock_time": r"""
sum by (namespace, pod) (
  rate(mongodb_top_readLock_time{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }[5m])
)
""",
    "mongodb_top_writeLock_count": r"""
sum by (namespace, pod) (
  rate(mongodb_top_writeLock_count{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }[5m])
)
""",
    "mongodb_top_writeLock_time": r"""
sum by (namespace, pod) (
  rate(mongodb_top_writeLock_time{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }[5m])
)
""",

    # Overall metrics
    "mongodb_top_total_count": r"""
sum by (namespace, pod) (
  rate(mongodb_top_total_count{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }[5m])
)
""",
    "mongodb_top_total_time": r"""
sum by (namespace, pod) (
  rate(mongodb_top_total_time{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }[5m])
)
"""
}

# ===== 8. Redis-Specific Metrics =====
REDIS_QUERIES = {
    # Connection metrics
    "redis_connected_clients": r"""
sum by (namespace, pod) (
  redis_connected_clients{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }
)
""",
    "redis_blocked_clients": r"""
sum by (namespace, pod) (
  redis_blocked_clients{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }
)
""",
    "redis_connections_received_total": r"""
sum by (namespace, pod) (
  rate(redis_connections_received_total{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }[5m])
)
""",
    "redis_rejected_connections_total": r"""
sum by (namespace, pod) (
  rate(redis_rejected_connections_total{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }[5m])
)
""",

    # Memory metrics
    "redis_memory_used_bytes": r"""
sum by (namespace, pod) (
  redis_memory_used_bytes{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }
)
""",
    "redis_memory_used_rss_bytes": r"""
sum by (namespace, pod) (
  redis_memory_used_rss_bytes{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }
)
""",
    "redis_memory_used_peak_bytes": r"""
sum by (namespace, pod) (
  redis_memory_used_peak_bytes{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }
)
""",
    "redis_memory_max_bytes": r"""
sum by (namespace, pod) (
  redis_memory_max_bytes{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }
)
""",
    "redis_mem_fragmentation_ratio": r"""
sum by (namespace, pod) (
  redis_mem_fragmentation_ratio{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }
)
""",

    # Keyspace metrics
    "redis_keyspace_hits_total": r"""
sum by (namespace, pod) (
  rate(redis_keyspace_hits_total{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }[5m])
)
""",
    "redis_keyspace_misses_total": r"""
sum by (namespace, pod) (
  rate(redis_keyspace_misses_total{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }[5m])
)
""",
    "redis_db_keys": r"""
sum by (namespace, pod, db) (
  redis_db_keys{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }
)
""",
    "redis_db_keys_expiring": r"""
sum by (namespace, pod, db) (
  redis_db_keys_expiring{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }
)
""",
    "redis_expired_keys_total": r"""
sum by (namespace, pod) (
  rate(redis_expired_keys_total{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }[5m])
)
""",
    "redis_evicted_keys_total": r"""
sum by (namespace, pod) (
  rate(redis_evicted_keys_total{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }[5m])
)
""",

    # Command processing metrics
    "redis_commands_processed_total": r"""
sum by (namespace, pod) (
  rate(redis_commands_processed_total{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }[5m])
)
""",
    "redis_commands_duration_seconds_total": r"""
sum by (namespace, pod, cmd) (
  rate(redis_commands_duration_seconds_total{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }[5m])
)
""",

    # CPU metrics
    "redis_cpu_sys_seconds_total": r"""
sum by (namespace, pod) (
  rate(redis_cpu_sys_seconds_total{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }[5m])
)
""",
    "redis_cpu_user_seconds_total": r"""
sum by (namespace, pod) (
  rate(redis_cpu_user_seconds_total{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }[5m])
)
""",

    # Network metrics
    "redis_net_input_bytes_total": r"""
sum by (namespace, pod) (
  rate(redis_net_input_bytes_total{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }[5m])
)
""",
    "redis_net_output_bytes_total": r"""
sum by (namespace, pod) (
  rate(redis_net_output_bytes_total{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }[5m])
)
""",

    # Persistence metrics
    "redis_rdb_bgsave_in_progress": r"""
sum by (namespace, pod) (
  redis_rdb_bgsave_in_progress{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }
)
""",
    "redis_rdb_last_bgsave_status": r"""
sum by (namespace, pod) (
  redis_rdb_last_bgsave_status{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }
)
""",
    "redis_rdb_last_bgsave_duration_sec": r"""
sum by (namespace, pod) (
  redis_rdb_last_bgsave_duration_sec{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }
)
""",
    "redis_rdb_changes_since_last_save": r"""
sum by (namespace, pod) (
  redis_rdb_changes_since_last_save{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }
)
"""
}

# ===== 9. MySQL-Specific Metrics =====
MYSQL_QUERIES = {
    # Connection metrics
    "mysql_global_status_threads_connected": r"""
sum by (namespace, pod) (
  mysql_global_status_threads_connected{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }
)
""",
    "mysql_global_status_threads_running": r"""
sum by (namespace, pod) (
  mysql_global_status_threads_running{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }
)
""",
    "mysql_global_status_max_used_connections": r"""
sum by (namespace, pod) (
  mysql_global_status_max_used_connections{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }
)
""",
    "mysql_global_status_aborted_connects": r"""
sum by (namespace, pod) (
  rate(mysql_global_status_aborted_connects{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }[5m])
)
""",
    "mysql_global_status_aborted_clients": r"""
sum by (namespace, pod) (
  rate(mysql_global_status_aborted_clients{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }[5m])
)
""",

    # Query metrics
    "mysql_global_status_queries": r"""
sum by (namespace, pod) (
  rate(mysql_global_status_queries{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }[5m])
)
""",
    "mysql_global_status_questions": r"""
sum by (namespace, pod) (
  rate(mysql_global_status_questions{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }[5m])
)
""",
    "mysql_global_status_slow_queries": r"""
sum by (namespace, pod) (
  rate(mysql_global_status_slow_queries{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }[5m])
)
""",

    # InnoDB Buffer Pool metrics
    "mysql_global_status_buffer_pool_pages": r"""
sum by (namespace, pod, state) (
  mysql_global_status_buffer_pool_pages{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }
)
""",
    "mysql_global_status_buffer_pool_dirty_pages": r"""
sum by (namespace, pod) (
  mysql_global_status_buffer_pool_dirty_pages{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }
)
""",
    "mysql_global_status_innodb_buffer_pool_read_requests": r"""
sum by (namespace, pod) (
  rate(mysql_global_status_innodb_buffer_pool_read_requests{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }[5m])
)
""",
    "mysql_global_status_innodb_buffer_pool_reads": r"""
sum by (namespace, pod) (
  rate(mysql_global_status_innodb_buffer_pool_reads{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }[5m])
)
""",
    "mysql_global_status_innodb_buffer_pool_write_requests": r"""
sum by (namespace, pod) (
  rate(mysql_global_status_innodb_buffer_pool_write_requests{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }[5m])
)
""",

    # InnoDB Row operations
    "mysql_global_status_innodb_row_ops_total": r"""
sum by (namespace, pod, operation) (
  rate(mysql_global_status_innodb_row_ops_total{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }[5m])
)
""",

    # InnoDB Lock metrics
    "mysql_global_status_innodb_row_lock_waits": r"""
sum by (namespace, pod) (
  rate(mysql_global_status_innodb_row_lock_waits{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }[5m])
)
""",
    "mysql_global_status_innodb_row_lock_time": r"""
sum by (namespace, pod) (
  rate(mysql_global_status_innodb_row_lock_time{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }[5m])
)
""",
    "mysql_global_status_table_locks_waited": r"""
sum by (namespace, pod) (
  rate(mysql_global_status_table_locks_waited{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }[5m])
)
""",

    # Network I/O
    "mysql_global_status_bytes_received": r"""
sum by (namespace, pod) (
  rate(mysql_global_status_bytes_received{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }[5m])
)
""",
    "mysql_global_status_bytes_sent": r"""
sum by (namespace, pod) (
  rate(mysql_global_status_bytes_sent{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }[5m])
)
""",

    # Temporary tables
    "mysql_global_status_created_tmp_tables": r"""
sum by (namespace, pod) (
  rate(mysql_global_status_created_tmp_tables{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }[5m])
)
""",
    "mysql_global_status_created_tmp_disk_tables": r"""
sum by (namespace, pod) (
  rate(mysql_global_status_created_tmp_disk_tables{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }[5m])
)
""",

    # Table cache
    "mysql_global_status_open_tables": r"""
sum by (namespace, pod) (
  mysql_global_status_open_tables{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }
)
""",
    "mysql_global_status_opened_tables": r"""
sum by (namespace, pod) (
  rate(mysql_global_status_opened_tables{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }[5m])
)
""",
    "mysql_global_status_table_open_cache_hits": r"""
sum by (namespace, pod) (
  rate(mysql_global_status_table_open_cache_hits{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }[5m])
)
""",
    "mysql_global_status_table_open_cache_misses": r"""
sum by (namespace, pod) (
  rate(mysql_global_status_table_open_cache_misses{
    namespace="sock-shop",
    pod=~"{pod_regex}"
  }[5m])
)
"""
}

# ===== 10. Map Service Types to Query Sets =====
SERVICE_TYPE_QUERIES = {
    "java": JAVA_QUERIES,
    "go": GO_QUERIES,
    "nodejs": NODEJS_QUERIES,
    "rabbitmq": RABBITMQ_QUERIES,
    "mongodb": MONGODB_QUERIES,
    "redis": REDIS_QUERIES,
    "mysql": MYSQL_QUERIES
}

# ===== 11. Prometheus Query Function =====
def query_range(promql_expr):
    url = f"{PROM_URL}/api/v1/query_range"
    params = {
        "query": promql_expr,
        "start": start_time.isoformat().replace('+00:00', 'Z'),
        "end": end_time.isoformat().replace('+00:00', 'Z'),
        "step": STEP,
    }
    try:
        response = requests.get(url, params=params, timeout=120)
        response.raise_for_status()
        data = response.json()
        if data.get("status") != "success":
            return []
        return data.get("data", {}).get("result", [])
    except Exception as e:
        print(f"[WARN] Query failed: {e}")
        return []

# ===== 12. Main Processing Logic =====
def main():
    # Use absolute path based on script location
    script_dir = Path(__file__).parent
    output_dir = script_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Pod-Specific Metrics Collection ===")
    print(f"PROM_URL: {PROM_URL}")
    print(f"Time Range: {start_time.isoformat()} to {end_time.isoformat()}")
    print(f"Step: {STEP}\n")

    # Dictionary to store rows grouped by individual pod name
    pod_rows = {}

    for service_type, config in POD_CATEGORIES.items():
        print(f"\n=== Processing {service_type.upper()} services ===")
        pod_names = config["pods"]

        # Process each service to collect metrics from all its pods
        for pod_name in pod_names:
            print(f"  Collecting metrics for {pod_name}...")

            # Create regex for this specific pod
            # Match both Deployment pattern (name-hash-hash) and StatefulSet pattern (name-number)
            # Examples:
            #   - Deployment: carts-86cc7969bb-rspqb
            #   - StatefulSet: carts-db-0, rabbitmq-0
            pod_regex = f"{pod_name}-([a-z0-9]+-[a-z0-9]+|[0-9]+)"

            # Common queries
            for metric_name, query_template in COMMON_QUERIES.items():
                query = query_template.replace("{pod_regex}", pod_regex)
                results = query_range(query)

                for series in results:
                    labels = series.get("metric", {})
                    values = series.get("values", [])

                    # Get the actual pod name from the metric labels
                    actual_pod_name = labels.get("pod", "unknown")

                    # Initialize rows list for this actual pod if not exists
                    if actual_pod_name not in pod_rows:
                        pod_rows[actual_pod_name] = []

                    for ts, val in values:
                        try:
                            v = float(val)
                        except (TypeError, ValueError):
                            continue

                        timestamp = datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                        row = {
                            "timestamp": timestamp,
                            "pod": actual_pod_name,
                            "service": pod_name,
                            "namespace": labels.get("namespace", "sock-shop"),
                            "metric": metric_name,
                            "value": v
                        }
                        # Add additional labels if present
                        for label_key in ["area", "queue", "type", "state", "db", "cmd", "command", "operation",
                                          "name", "method", "status", "outcome", "repository"]:
                            if label_key in labels:
                                row[label_key] = labels[label_key]
                        pod_rows[actual_pod_name].append(row)

            # Service-type specific queries
            specific_queries = SERVICE_TYPE_QUERIES.get(service_type, {})
            if specific_queries:
                for metric_name, query_template in specific_queries.items():
                    query = query_template.replace("{pod_regex}", pod_regex)
                    results = query_range(query)

                    for series in results:
                        labels = series.get("metric", {})
                        values = series.get("values", [])

                        # Get the actual pod name from the metric labels
                        actual_pod_name = labels.get("pod", "unknown")

                        # Initialize rows list for this actual pod if not exists
                        if actual_pod_name not in pod_rows:
                            pod_rows[actual_pod_name] = []

                        for ts, val in values:
                            try:
                                v = float(val)
                            except (TypeError, ValueError):
                                continue

                            timestamp = datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                            row = {
                                "timestamp": timestamp,
                                "pod": actual_pod_name,
                                "service": pod_name,
                                "namespace": labels.get("namespace", "sock-shop"),
                                "metric": metric_name,
                                "value": v
                            }
                            # Add additional labels if present
                            for label_key in ["area", "queue", "type", "state", "db", "cmd", "command", "operation",
                                              "name", "method", "status", "outcome", "repository"]:
                                if label_key in labels:
                                    row[label_key] = labels[label_key]
                            pod_rows[actual_pod_name].append(row)

            # Istio service mesh metrics (only for application services)
            if service_type in ["java", "go", "nodejs"]:
                for metric_name, query_template in ISTIO_QUERIES.items():
                    # For Istio metrics, use plain pod name without regex pattern
                    query = query_template.replace("{pod_regex_plain}", pod_name)
                    results = query_range(query)

                    for series in results:
                        labels = series.get("metric", {})
                        values = series.get("values", [])

                        # Get the actual pod name from the metric labels
                        actual_pod_name = labels.get("pod", "unknown")

                        # Initialize rows list for this actual pod if not exists
                        if actual_pod_name not in pod_rows:
                            pod_rows[actual_pod_name] = []

                        for ts, val in values:
                            try:
                                v = float(val)
                            except (TypeError, ValueError):
                                continue

                            timestamp = datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                            row = {
                                "timestamp": timestamp,
                                "pod": actual_pod_name,
                                "service": pod_name,
                                "namespace": labels.get("namespace", "sock-shop"),
                                "metric": metric_name,
                                "value": v
                            }
                            # Add additional labels if present
                            for label_key in ["destination_workload", "destination_service", "destination_app"]:
                                if label_key in labels:
                                    row[label_key] = labels[label_key]
                            pod_rows[actual_pod_name].append(row)

    # Save separate CSV for each pod
    print(f"\n=== Saving CSV files ===")
    total_files = 0
    for pod_name, rows in pod_rows.items():
        if not rows:
            print(f"  ⚠️  No data collected for {pod_name}")
            continue

        df = pd.DataFrame(rows)
        df = df.dropna(axis=1, how='all')
        df.sort_values(["timestamp", "metric"], inplace=True)

        csv_file = output_dir / f"prometheus_pod_metrics_{pod_name}.csv"
        df.to_csv(csv_file, index=False)
        print(f"  ✅ Saved {len(df)} rows to {csv_file}")
        total_files += 1

    print(f"\n=== Collection Complete ===")
    print(f"Total CSV files created: {total_files}")
    print(f"Total unique pods: {len(pod_rows)}")

if __name__ == "__main__":
    main()
