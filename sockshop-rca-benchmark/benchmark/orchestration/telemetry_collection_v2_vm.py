from __future__ import annotations

import csv
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd
import requests
from requests.exceptions import ChunkedEncodingError, ConnectionError, HTTPError, RequestException, Timeout


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATASET_BASE = PROJECT_ROOT / "dataset_v2" / "telemetry"
GO_METRIC_FILE = [
    "go_gc_duration_seconds",
    "go_gc_duration_seconds_count",
    "go_gc_duration_seconds_sum",
    "go_gc_gogc_percent",
    "go_gc_gomemlimit_bytes",
    "go_goroutines",
    "go_info",
    "go_memstats_alloc_bytes",
    "go_memstats_alloc_bytes_total",
    "go_memstats_buck_hash_sys_bytes",
    "go_memstats_frees_total",
    "go_memstats_gc_sys_bytes",
    "go_memstats_heap_alloc_bytes",
    "go_memstats_heap_idle_bytes",
    "go_memstats_heap_inuse_bytes",
    "go_memstats_heap_objects",
    "go_memstats_heap_released_bytes",
    "go_memstats_heap_sys_bytes",
    "go_memstats_last_gc_time_seconds",
    "go_memstats_lookups_total",
    "go_memstats_mallocs_total",
    "go_memstats_mcache_inuse_bytes",
    "go_memstats_mcache_sys_bytes",
    "go_memstats_mspan_inuse_bytes",
    "go_memstats_mspan_sys_bytes",
    "go_memstats_next_gc_bytes",
    "go_memstats_other_sys_bytes",
    "go_memstats_stack_inuse_bytes",
    "go_memstats_stack_sys_bytes",
    "go_memstats_sys_bytes",
    "go_sched_gomaxprocs_threads",
    "go_threads",
    "http_inflight_requests",
    "http_request_body_size_bytes_bucket",
    "http_request_body_size_bytes_count",
    "http_request_body_size_bytes_sum",
    "http_request_duration_seconds_bucket",
    "http_request_duration_seconds_count",
    "http_request_duration_seconds_sum",
    "http_request_size_bytes_bucket",
    "http_request_size_bytes_count",
    "http_request_size_bytes_sum",
    "http_requests_in_flight",
    "http_response_body_size_bytes_bucket",
    "http_response_body_size_bytes_count",
    "http_response_body_size_bytes_sum",
    "http_response_size_bytes_bucket",
    "http_response_size_bytes_count",
    "http_response_size_bytes_sum",
    "prober_probe_duration_seconds_bucket",
    "prober_probe_duration_seconds_count",
    "prober_probe_duration_seconds_sum",
    "prober_probe_total",
    "process_cpu_seconds_total",
    "process_max_fds",
    "process_network_receive_bytes_total",
    "process_network_transmit_bytes_total",
    "process_open_fds",
    "process_resident_memory_bytes",
    "process_start_time_seconds",
    "process_virtual_memory_bytes",
    "process_virtual_memory_max_bytes",
    "up",
]

JAVA_METRIC_FILE = [
    "db_order_findByCustomerId_active_seconds_count",
    "db_order_findByCustomerId_active_seconds_max",
    "db_order_findByCustomerId_active_seconds_sum",
    "db_order_findByCustomerId_seconds_count",
    "db_order_findByCustomerId_seconds_max",
    "db_order_findByCustomerId_seconds_sum",
    "db_order_findById_active_seconds_count",
    "db_order_findById_active_seconds_max",
    "db_order_findById_active_seconds_sum",
    "db_order_findById_seconds_count",
    "db_order_findById_seconds_max",
    "db_order_findById_seconds_sum",
    "db_order_save_active_seconds_count",
    "db_order_save_active_seconds_max",
    "db_order_save_active_seconds_sum",
    "db_order_save_seconds_count",
    "db_order_save_seconds_max",
    "db_order_save_seconds_sum",
    "disk_free_bytes",
    "disk_total_bytes",
    "executor_active_threads",
    "executor_completed_tasks_total",
    "executor_pool_core_threads",
    "executor_pool_max_threads",
    "executor_pool_size_threads",
    "executor_queue_remaining_tasks",
    "executor_queued_tasks",
    "http_client_requests_active_seconds_count",
    "http_client_requests_active_seconds_max",
    "http_client_requests_active_seconds_sum",
    "http_client_requests_seconds_count",
    "http_client_requests_seconds_max",
    "http_client_requests_seconds_sum",
    "http_server_requests_active_seconds_count",
    "http_server_requests_active_seconds_max",
    "http_server_requests_active_seconds_sum",
    "http_server_requests_seconds_count",
    "http_server_requests_seconds_max",
    "http_server_requests_seconds_sum",
    "jvm_buffer_count_buffers",
    "jvm_buffer_memory_used_bytes",
    "jvm_buffer_total_capacity_bytes",
    "jvm_classes_loaded_classes",
    "jvm_classes_unloaded_classes_total",
    "jvm_compilation_time_ms_total",
    "jvm_gc_live_data_size_bytes",
    "jvm_gc_max_data_size_bytes",
    "jvm_gc_memory_allocated_bytes_total",
    "jvm_gc_memory_promoted_bytes_total",
    "jvm_gc_overhead",
    "jvm_gc_pause_seconds_count",
    "jvm_gc_pause_seconds_max",
    "jvm_gc_pause_seconds_sum",
    "jvm_info",
    "jvm_memory_committed_bytes",
    "jvm_memory_max_bytes",
    "jvm_memory_usage_after_gc",
    "jvm_memory_used_bytes",
    "jvm_threads_daemon_threads",
    "jvm_threads_live_threads",
    "jvm_threads_peak_threads",
    "jvm_threads_started_threads_total",
    "jvm_threads_states_threads",
    "logback_events_total",
    "mongodb_driver_commands_seconds_count",
    "mongodb_driver_commands_seconds_max",
    "mongodb_driver_commands_seconds_sum",
    "mongodb_driver_pool_checkedout",
    "mongodb_driver_pool_checkoutfailed_total",
    "mongodb_driver_pool_size",
    "mongodb_driver_pool_waitqueuesize",
    "process_cpu_time_ns_total",
    "process_cpu_usage",
    "process_files_max_files",
    "process_files_open_files",
    "process_start_time_seconds",
    "process_uptime_seconds",
    "rabbitmq_acknowledged_published_total",
    "rabbitmq_acknowledged_total",
    "rabbitmq_channels",
    "rabbitmq_connections",
    "rabbitmq_consumed_total",
    "rabbitmq_failed_to_publish_total",
    "rabbitmq_not_acknowledged_published_total",
    "rabbitmq_published_total",
    "rabbitmq_rejected_total",
    "rabbitmq_unrouted_published_total",
    "spring_data_repository_invocations_seconds_count",
    "spring_data_repository_invocations_seconds_max",
    "spring_data_repository_invocations_seconds_sum",
    "spring_rabbitmq_listener_seconds_count",
    "spring_rabbitmq_listener_seconds_max",
    "spring_rabbitmq_listener_seconds_sum",
    "system_cpu_count",
    "system_cpu_usage",
    "system_load_average_1m",
    "tomcat_sessions_active_current_sessions",
    "tomcat_sessions_active_max_sessions",
    "tomcat_sessions_alive_max_seconds",
    "tomcat_sessions_created_sessions_total",
    "tomcat_sessions_expired_sessions_total",
    "tomcat_sessions_rejected_sessions_total",
    "up",
]

NODEJS_METRIC_FILE = [
    "http_request_duration_seconds_bucket",
    "http_request_duration_seconds_count",
    "http_request_duration_seconds_sum",
    "http_request_errors_total",
    "http_request_size_bytes_bucket",
    "http_request_size_bytes_count",
    "http_request_size_bytes_sum",
    "http_requests_in_progress",
    "http_requests_total",
    "http_response_size_bytes_bucket",
    "http_response_size_bytes_count",
    "http_response_size_bytes_sum",
    "nodejs_nodejs_active_handles",
    "nodejs_nodejs_active_handles_total",
    "nodejs_nodejs_active_requests",
    "nodejs_nodejs_active_requests_total",
    "nodejs_nodejs_active_resources",
    "nodejs_nodejs_active_resources_total",
    "nodejs_nodejs_eventloop_lag_max_seconds",
    "nodejs_nodejs_eventloop_lag_mean_seconds",
    "nodejs_nodejs_eventloop_lag_min_seconds",
    "nodejs_nodejs_eventloop_lag_p50_seconds",
    "nodejs_nodejs_eventloop_lag_p90_seconds",
    "nodejs_nodejs_eventloop_lag_p99_seconds",
    "nodejs_nodejs_eventloop_lag_seconds",
    "nodejs_nodejs_eventloop_lag_stddev_seconds",
    "nodejs_nodejs_external_memory_bytes",
    "nodejs_nodejs_gc_duration_seconds_bucket",
    "nodejs_nodejs_gc_duration_seconds_count",
    "nodejs_nodejs_gc_duration_seconds_sum",
    "nodejs_nodejs_heap_size_total_bytes",
    "nodejs_nodejs_heap_size_used_bytes",
    "nodejs_nodejs_heap_space_size_available_bytes",
    "nodejs_nodejs_heap_space_size_total_bytes",
    "nodejs_nodejs_heap_space_size_used_bytes",
    "nodejs_nodejs_version_info",
    "nodejs_process_cpu_seconds_total",
    "nodejs_process_cpu_system_seconds_total",
    "nodejs_process_cpu_user_seconds_total",
    "nodejs_process_heap_bytes",
    "nodejs_process_max_fds",
    "nodejs_process_open_fds",
    "nodejs_process_resident_memory_bytes",
    "nodejs_process_start_time_seconds",
    "nodejs_process_virtual_memory_bytes",
    "up",
]

CONTAINER_METRIC_FILE = [
    "container_blkio_device_usage_total",
    "container_cpu_cfs_periods_total",
    "container_cpu_cfs_throttled_periods_total",
    "container_cpu_usage_seconds_total",
    "container_fs_reads_bytes_total",
    "container_fs_reads_total",
    "container_fs_writes_bytes_total",
    "container_fs_writes_total",
    "container_last_seen",
    "container_memory_cache",
    "container_memory_failcnt",
    "container_memory_failures_total",
    "container_memory_kernel_usage",
    "container_memory_max_usage_bytes",
    "container_memory_rss",
    "container_memory_usage_bytes",
    "container_memory_working_set_bytes",
    "container_network_receive_bytes_total",
    "container_network_receive_errors_total",
    "container_network_receive_packets_dropped_total",
    "container_network_receive_packets_total",
    "container_network_transmit_bytes_total",
    "container_network_transmit_errors_total",
    "container_network_transmit_packets_dropped_total",
    "container_network_transmit_packets_total",
    "container_oom_events_total",
    "container_processes",
    "container_scrape_error",
    "container_sockets",
    "container_start_time_seconds",
    "container_threads",
    "container_ulimits_soft",
]

KUBE_POD_METRIC_FILE = [
    "kube_pod_completion_time",
    "kube_pod_container_info",
    "kube_pod_container_resource_limits",
    "kube_pod_container_resource_requests",
    "kube_pod_container_state_started",
    "kube_pod_container_status_last_terminated_exitcode",
    "kube_pod_container_status_last_terminated_reason",
    "kube_pod_container_status_last_terminated_timestamp",
    "kube_pod_container_status_ready",
    "kube_pod_container_status_restarts_total",
    "kube_pod_container_status_running",
    "kube_pod_container_status_terminated",
    "kube_pod_container_status_terminated_reason",
    "kube_pod_container_status_waiting",
    "kube_pod_container_status_waiting_reason",
    "kube_pod_created",
    "kube_pod_deletion_timestamp",
    "kube_pod_info",
    "kube_pod_init_container_info",
    "kube_pod_init_container_resource_limits",
    "kube_pod_init_container_resource_requests",
    "kube_pod_init_container_status_ready",
    "kube_pod_init_container_status_restarts_total",
    "kube_pod_init_container_status_running",
    "kube_pod_init_container_status_terminated",
    "kube_pod_init_container_status_terminated_reason",
    "kube_pod_init_container_status_waiting",
    "kube_pod_ips",
    "kube_pod_owner",
    "kube_pod_restart_policy",
    "kube_pod_scheduler",
    "kube_pod_service_account",
    "kube_pod_spec_volumes_persistentvolumeclaims_info",
    "kube_pod_spec_volumes_persistentvolumeclaims_readonly",
    "kube_pod_start_time",
    "kube_pod_status_container_ready_time",
    "kube_pod_status_initialized_time",
    "kube_pod_status_phase",
    "kube_pod_status_qos_class",
    "kube_pod_status_ready",
    "kube_pod_status_ready_time",
    "kube_pod_status_reason",
    "kube_pod_status_scheduled",
    "kube_pod_status_scheduled_time",
    "kube_pod_tolerations",
]

MONGODB_METRIC_FILE = [
    "mongodb_top_commands_count",
    "mongodb_top_commands_time",
    "mongodb_top_getmore_count",
    "mongodb_top_getmore_time",
    "mongodb_top_insert_count",
    "mongodb_top_insert_time",
    "mongodb_top_queries_count",
    "mongodb_top_queries_time",
    "mongodb_top_readLock_count",
    "mongodb_top_readLock_time",
    "mongodb_top_remove_count",
    "mongodb_top_remove_time",
    "mongodb_top_total_count",
    "mongodb_top_total_time",
    "mongodb_top_update_count",
    "mongodb_top_update_time",
    "mongodb_top_writeLock_count",
    "mongodb_top_writeLock_time",
    "mongodb_up",
]

MYSQL_METRIC_FILE = [
    "mysql_exporter_collector_duration_seconds",
    "mysql_exporter_last_scrape_error",
    "mysql_exporter_scrapes_total",
    "mysql_global_status_aborted_clients",
    "mysql_global_status_aborted_connects",
    "mysql_global_status_binlog_cache_disk_use",
    "mysql_global_status_binlog_cache_use",
    "mysql_global_status_binlog_stmt_cache_disk_use",
    "mysql_global_status_binlog_stmt_cache_use",
    "mysql_global_status_buffer_pool_dirty_pages",
    "mysql_global_status_buffer_pool_page_changes_total",
    "mysql_global_status_buffer_pool_pages",
    "mysql_global_status_bytes_received",
    "mysql_global_status_bytes_sent",
    "mysql_global_status_commands_total",
    "mysql_global_status_connection_errors_total",
    "mysql_global_status_connections",
    "mysql_global_status_created_tmp_disk_tables",
    "mysql_global_status_created_tmp_files",
    "mysql_global_status_created_tmp_tables",
    "mysql_global_status_delayed_errors",
    "mysql_global_status_delayed_insert_threads",
    "mysql_global_status_delayed_writes",
    "mysql_global_status_flush_commands",
    "mysql_global_status_handlers_total",
    "mysql_global_status_innodb_available_undo_logs",
    "mysql_global_status_innodb_buffer_pool_bytes_data",
    "mysql_global_status_innodb_buffer_pool_bytes_dirty",
    "mysql_global_status_innodb_buffer_pool_read_ahead",
    "mysql_global_status_innodb_buffer_pool_read_ahead_evicted",
    "mysql_global_status_innodb_buffer_pool_read_ahead_rnd",
    "mysql_global_status_innodb_buffer_pool_read_requests",
    "mysql_global_status_innodb_buffer_pool_reads",
    "mysql_global_status_innodb_buffer_pool_wait_free",
    "mysql_global_status_innodb_buffer_pool_write_requests",
    "mysql_global_status_innodb_data_fsyncs",
    "mysql_global_status_innodb_data_pending_fsyncs",
    "mysql_global_status_innodb_data_pending_reads",
    "mysql_global_status_innodb_data_pending_writes",
    "mysql_global_status_innodb_data_read",
    "mysql_global_status_innodb_data_reads",
    "mysql_global_status_innodb_data_writes",
    "mysql_global_status_innodb_data_written",
    "mysql_global_status_innodb_dblwr_pages_written",
    "mysql_global_status_innodb_dblwr_writes",
    "mysql_global_status_innodb_log_waits",
    "mysql_global_status_innodb_log_write_requests",
    "mysql_global_status_innodb_log_writes",
    "mysql_global_status_innodb_num_open_files",
    "mysql_global_status_innodb_os_log_fsyncs",
    "mysql_global_status_innodb_os_log_pending_fsyncs",
    "mysql_global_status_innodb_os_log_pending_writes",
    "mysql_global_status_innodb_os_log_written",
    "mysql_global_status_innodb_page_size",
    "mysql_global_status_innodb_pages_created",
    "mysql_global_status_innodb_pages_read",
    "mysql_global_status_innodb_pages_written",
    "mysql_global_status_innodb_row_lock_current_waits",
    "mysql_global_status_innodb_row_lock_time",
    "mysql_global_status_innodb_row_lock_time_avg",
    "mysql_global_status_innodb_row_lock_time_max",
    "mysql_global_status_innodb_row_lock_waits",
    "mysql_global_status_innodb_row_ops_total",
    "mysql_global_status_innodb_truncated_status_writes",
    "mysql_global_status_key_blocks_not_flushed",
    "mysql_global_status_key_blocks_unused",
    "mysql_global_status_key_blocks_used",
    "mysql_global_status_key_read_requests",
    "mysql_global_status_key_reads",
    "mysql_global_status_key_write_requests",
    "mysql_global_status_key_writes",
    "mysql_global_status_locked_connects",
    "mysql_global_status_max_execution_time_exceeded",
    "mysql_global_status_max_execution_time_set",
    "mysql_global_status_max_execution_time_set_failed",
    "mysql_global_status_max_used_connections",
    "mysql_global_status_max_used_connections_time",
    "mysql_global_status_not_flushed_delayed_rows",
    "mysql_global_status_ongoing_anonymous_transaction_count",
    "mysql_global_status_open_files",
    "mysql_global_status_open_streams",
    "mysql_global_status_open_table_definitions",
    "mysql_global_status_open_tables",
    "mysql_global_status_opened_files",
    "mysql_global_status_opened_table_definitions",
    "mysql_global_status_opened_tables",
    "mysql_global_status_performance_schema_lost_total",
    "mysql_global_status_prepared_stmt_count",
    "mysql_global_status_qcache_free_blocks",
    "mysql_global_status_qcache_free_memory",
    "mysql_global_status_qcache_hits",
    "mysql_global_status_qcache_inserts",
    "mysql_global_status_qcache_lowmem_prunes",
    "mysql_global_status_qcache_not_cached",
    "mysql_global_status_qcache_queries_in_cache",
    "mysql_global_status_qcache_total_blocks",
    "mysql_global_status_queries",
    "mysql_global_status_questions",
    "mysql_global_status_select_full_join",
    "mysql_global_status_select_full_range_join",
    "mysql_global_status_select_range",
    "mysql_global_status_select_range_check",
    "mysql_global_status_select_scan",
    "mysql_global_status_slave_open_temp_tables",
    "mysql_global_status_slow_launch_threads",
    "mysql_global_status_slow_queries",
    "mysql_global_status_sort_merge_passes",
    "mysql_global_status_sort_range",
    "mysql_global_status_sort_rows",
    "mysql_global_status_sort_scan",
    "mysql_global_status_ssl_accept_renegotiates",
    "mysql_global_status_ssl_accepts",
    "mysql_global_status_ssl_callback_cache_hits",
    "mysql_global_status_ssl_client_connects",
    "mysql_global_status_ssl_connect_renegotiates",
    "mysql_global_status_ssl_ctx_verify_depth",
    "mysql_global_status_ssl_ctx_verify_mode",
    "mysql_global_status_ssl_default_timeout",
    "mysql_global_status_ssl_finished_accepts",
    "mysql_global_status_ssl_finished_connects",
    "mysql_global_status_ssl_session_cache_hits",
    "mysql_global_status_ssl_session_cache_misses",
    "mysql_global_status_ssl_session_cache_overflows",
    "mysql_global_status_ssl_session_cache_size",
    "mysql_global_status_ssl_session_cache_timeouts",
    "mysql_global_status_ssl_sessions_reused",
    "mysql_global_status_ssl_used_session_cache_entries",
    "mysql_global_status_ssl_verify_depth",
    "mysql_global_status_ssl_verify_mode",
    "mysql_global_status_table_locks_immediate",
    "mysql_global_status_table_locks_waited",
    "mysql_global_status_table_open_cache_hits",
    "mysql_global_status_table_open_cache_misses",
    "mysql_global_status_table_open_cache_overflows",
    "mysql_global_status_tc_log_max_pages_used",
    "mysql_global_status_tc_log_page_size",
    "mysql_global_status_tc_log_page_waits",
    "mysql_global_status_threads_cached",
    "mysql_global_status_threads_connected",
    "mysql_global_status_threads_created",
    "mysql_global_status_threads_running",
    "mysql_global_status_uptime",
    "mysql_global_status_uptime_since_flush_status",
    "mysql_global_variables_auto_increment_increment",
    "mysql_global_variables_auto_increment_offset",
    "mysql_global_variables_autocommit",
    "mysql_global_variables_automatic_sp_privileges",
    "mysql_global_variables_avoid_temporal_upgrade",
    "mysql_global_variables_back_log",
    "mysql_global_variables_big_tables",
    "mysql_global_variables_binlog_cache_size",
    "mysql_global_variables_binlog_direct_non_transactional_updates",
    "mysql_global_variables_binlog_group_commit_sync_delay",
    "mysql_global_variables_binlog_group_commit_sync_no_delay_count",
    "mysql_global_variables_binlog_gtid_simple_recovery",
    "mysql_global_variables_binlog_max_flush_queue_time",
    "mysql_global_variables_binlog_order_commits",
    "mysql_global_variables_binlog_rows_query_log_events",
    "mysql_global_variables_binlog_stmt_cache_size",
    "mysql_global_variables_bulk_insert_buffer_size",
    "mysql_global_variables_check_proxy_users",
    "mysql_global_variables_connect_timeout",
    "mysql_global_variables_core_file",
    "mysql_global_variables_default_password_lifetime",
    "mysql_global_variables_default_week_format",
    "mysql_global_variables_delay_key_write",
    "mysql_global_variables_delayed_insert_limit",
    "mysql_global_variables_delayed_insert_timeout",
    "mysql_global_variables_delayed_queue_size",
    "mysql_global_variables_disconnect_on_expired_password",
    "mysql_global_variables_div_precision_increment",
    "mysql_global_variables_end_markers_in_json",
    "mysql_global_variables_enforce_gtid_consistency",
    "mysql_global_variables_eq_range_index_dive_limit",
    "mysql_global_variables_event_scheduler",
    "mysql_global_variables_expire_logs_days",
    "mysql_global_variables_explicit_defaults_for_timestamp",
    "mysql_global_variables_flush",
    "mysql_global_variables_flush_time",
    "mysql_global_variables_foreign_key_checks",
    "mysql_global_variables_ft_max_word_len",
    "mysql_global_variables_ft_min_word_len",
    "mysql_global_variables_ft_query_expansion_limit",
    "mysql_global_variables_general_log",
    "mysql_global_variables_group_concat_max_len",
    "mysql_global_variables_gtid_executed_compression_period",
    "mysql_global_variables_gtid_mode",
    "mysql_global_variables_have_compress",
    "mysql_global_variables_have_crypt",
    "mysql_global_variables_have_dynamic_loading",
    "mysql_global_variables_have_geometry",
    "mysql_global_variables_have_openssl",
    "mysql_global_variables_have_profiling",
    "mysql_global_variables_have_query_cache",
    "mysql_global_variables_have_rtree_keys",
    "mysql_global_variables_have_ssl",
    "mysql_global_variables_have_statement_timeout",
    "mysql_global_variables_have_symlink",
    "mysql_global_variables_host_cache_size",
    "mysql_global_variables_ignore_builtin_innodb",
    "mysql_global_variables_innodb_adaptive_flushing",
    "mysql_global_variables_innodb_adaptive_flushing_lwm",
    "mysql_global_variables_innodb_adaptive_hash_index",
    "mysql_global_variables_innodb_adaptive_hash_index_parts",
    "mysql_global_variables_innodb_adaptive_max_sleep_delay",
    "mysql_global_variables_innodb_api_bk_commit_interval",
    "mysql_global_variables_innodb_api_disable_rowlock",
    "mysql_global_variables_innodb_api_enable_binlog",
    "mysql_global_variables_innodb_api_enable_mdl",
    "mysql_global_variables_innodb_api_trx_level",
    "mysql_global_variables_innodb_autoextend_increment",
    "mysql_global_variables_innodb_autoinc_lock_mode",
    "mysql_global_variables_innodb_buffer_pool_chunk_size",
    "mysql_global_variables_innodb_buffer_pool_dump_at_shutdown",
    "mysql_global_variables_innodb_buffer_pool_dump_now",
    "mysql_global_variables_innodb_buffer_pool_dump_pct",
    "mysql_global_variables_innodb_buffer_pool_instances",
    "mysql_global_variables_innodb_buffer_pool_load_abort",
    "mysql_global_variables_innodb_buffer_pool_load_at_startup",
    "mysql_global_variables_innodb_buffer_pool_load_now",
    "mysql_global_variables_innodb_buffer_pool_size",
    "mysql_global_variables_innodb_change_buffer_max_size",
    "mysql_global_variables_innodb_checksums",
    "mysql_global_variables_innodb_cmp_per_index_enabled",
    "mysql_global_variables_innodb_commit_concurrency",
    "mysql_global_variables_innodb_compression_failure_threshold_pct",
    "mysql_global_variables_innodb_compression_level",
    "mysql_global_variables_innodb_compression_pad_pct_max",
    "mysql_global_variables_innodb_concurrency_tickets",
    "mysql_global_variables_innodb_deadlock_detect",
    "mysql_global_variables_innodb_disable_sort_file_cache",
    "mysql_global_variables_innodb_doublewrite",
    "mysql_global_variables_innodb_fast_shutdown",
    "mysql_global_variables_innodb_file_format_check",
    "mysql_global_variables_innodb_file_per_table",
    "mysql_global_variables_innodb_fill_factor",
    "mysql_global_variables_innodb_flush_log_at_timeout",
    "mysql_global_variables_innodb_flush_log_at_trx_commit",
    "mysql_global_variables_innodb_flush_neighbors",
    "mysql_global_variables_innodb_flush_sync",
    "mysql_global_variables_innodb_flushing_avg_loops",
    "mysql_global_variables_innodb_force_load_corrupted",
    "mysql_global_variables_innodb_force_recovery",
    "mysql_global_variables_innodb_ft_cache_size",
    "mysql_global_variables_innodb_ft_enable_diag_print",
    "mysql_global_variables_innodb_ft_enable_stopword",
    "mysql_global_variables_innodb_ft_max_token_size",
    "mysql_global_variables_innodb_ft_min_token_size",
    "mysql_global_variables_innodb_ft_num_word_optimize",
    "mysql_global_variables_innodb_ft_result_cache_limit",
    "mysql_global_variables_innodb_ft_sort_pll_degree",
    "mysql_global_variables_innodb_ft_total_cache_size",
    "mysql_global_variables_innodb_io_capacity",
    "mysql_global_variables_innodb_io_capacity_max",
    "mysql_global_variables_innodb_large_prefix",
    "mysql_global_variables_innodb_lock_wait_timeout",
    "mysql_global_variables_innodb_locks_unsafe_for_binlog",
    "mysql_global_variables_innodb_log_buffer_size",
    "mysql_global_variables_innodb_log_checksums",
    "mysql_global_variables_innodb_log_compressed_pages",
    "mysql_global_variables_innodb_log_file_size",
    "mysql_global_variables_innodb_log_files_in_group",
    "mysql_global_variables_innodb_log_write_ahead_size",
    "mysql_global_variables_innodb_lru_scan_depth",
    "mysql_global_variables_innodb_max_dirty_pages_pct",
    "mysql_global_variables_innodb_max_dirty_pages_pct_lwm",
    "mysql_global_variables_innodb_max_purge_lag",
    "mysql_global_variables_innodb_max_purge_lag_delay",
    "mysql_global_variables_innodb_max_undo_log_size",
    "mysql_global_variables_innodb_numa_interleave",
    "mysql_global_variables_innodb_old_blocks_pct",
    "mysql_global_variables_innodb_old_blocks_time",
    "mysql_global_variables_innodb_online_alter_log_max_size",
    "mysql_global_variables_innodb_open_files",
    "mysql_global_variables_innodb_optimize_fulltext_only",
    "mysql_global_variables_innodb_page_cleaners",
    "mysql_global_variables_innodb_page_size",
    "mysql_global_variables_innodb_print_all_deadlocks",
    "mysql_global_variables_innodb_purge_batch_size",
    "mysql_global_variables_innodb_purge_rseg_truncate_frequency",
    "mysql_global_variables_innodb_purge_threads",
    "mysql_global_variables_innodb_random_read_ahead",
    "mysql_global_variables_innodb_read_ahead_threshold",
    "mysql_global_variables_innodb_read_io_threads",
    "mysql_global_variables_innodb_read_only",
    "mysql_global_variables_innodb_replication_delay",
    "mysql_global_variables_innodb_rollback_on_timeout",
    "mysql_global_variables_innodb_rollback_segments",
    "mysql_global_variables_innodb_sort_buffer_size",
    "mysql_global_variables_innodb_spin_wait_delay",
    "mysql_global_variables_innodb_stats_auto_recalc",
    "mysql_global_variables_innodb_stats_include_delete_marked",
    "mysql_global_variables_innodb_stats_on_metadata",
    "mysql_global_variables_innodb_stats_persistent",
    "mysql_global_variables_innodb_stats_persistent_sample_pages",
    "mysql_global_variables_innodb_stats_sample_pages",
    "mysql_global_variables_innodb_stats_transient_sample_pages",
    "mysql_global_variables_innodb_status_output",
    "mysql_global_variables_innodb_status_output_locks",
    "mysql_global_variables_innodb_strict_mode",
    "mysql_global_variables_innodb_support_xa",
    "mysql_global_variables_innodb_sync_array_size",
    "mysql_global_variables_innodb_sync_spin_loops",
    "mysql_global_variables_innodb_table_locks",
    "mysql_global_variables_innodb_thread_concurrency",
    "mysql_global_variables_innodb_thread_sleep_delay",
    "mysql_global_variables_innodb_undo_log_truncate",
    "mysql_global_variables_innodb_undo_logs",
    "mysql_global_variables_innodb_undo_tablespaces",
    "mysql_global_variables_innodb_use_native_aio",
    "mysql_global_variables_innodb_write_io_threads",
    "mysql_global_variables_interactive_timeout",
    "mysql_global_variables_join_buffer_size",
    "mysql_global_variables_keep_files_on_create",
    "mysql_global_variables_key_buffer_size",
    "mysql_global_variables_key_cache_age_threshold",
    "mysql_global_variables_key_cache_block_size",
    "mysql_global_variables_key_cache_division_limit",
    "mysql_global_variables_large_files_support",
    "mysql_global_variables_large_page_size",
    "mysql_global_variables_large_pages",
    "mysql_global_variables_local_infile",
    "mysql_global_variables_lock_wait_timeout",
    "mysql_global_variables_locked_in_memory",
    "mysql_global_variables_log_bin",
    "mysql_global_variables_log_bin_trust_function_creators",
    "mysql_global_variables_log_bin_use_v1_row_events",
    "mysql_global_variables_log_builtin_as_identified_by_password",
    "mysql_global_variables_log_error_verbosity",
    "mysql_global_variables_log_queries_not_using_indexes",
    "mysql_global_variables_log_slave_updates",
    "mysql_global_variables_log_slow_admin_statements",
    "mysql_global_variables_log_slow_slave_statements",
    "mysql_global_variables_log_statements_unsafe_for_binlog",
    "mysql_global_variables_log_syslog",
    "mysql_global_variables_log_syslog_include_pid",
    "mysql_global_variables_log_throttle_queries_not_using_indexes",
    "mysql_global_variables_log_warnings",
    "mysql_global_variables_long_query_time",
    "mysql_global_variables_low_priority_updates",
    "mysql_global_variables_lower_case_file_system",
    "mysql_global_variables_lower_case_table_names",
    "mysql_global_variables_master_verify_checksum",
    "mysql_global_variables_max_allowed_packet",
    "mysql_global_variables_max_binlog_cache_size",
    "mysql_global_variables_max_binlog_size",
    "mysql_global_variables_max_binlog_stmt_cache_size",
    "mysql_global_variables_max_connect_errors",
    "mysql_global_variables_max_connections",
    "mysql_global_variables_max_delayed_threads",
    "mysql_global_variables_max_digest_length",
    "mysql_global_variables_max_error_count",
    "mysql_global_variables_max_execution_time",
    "mysql_global_variables_max_heap_table_size",
    "mysql_global_variables_max_insert_delayed_threads",
    "mysql_global_variables_max_join_size",
    "mysql_global_variables_max_length_for_sort_data",
    "mysql_global_variables_max_points_in_geometry",
    "mysql_global_variables_max_prepared_stmt_count",
    "mysql_global_variables_max_relay_log_size",
    "mysql_global_variables_max_seeks_for_key",
    "mysql_global_variables_max_sort_length",
    "mysql_global_variables_max_sp_recursion_depth",
    "mysql_global_variables_max_tmp_tables",
    "mysql_global_variables_max_user_connections",
    "mysql_global_variables_max_write_lock_count",
    "mysql_global_variables_metadata_locks_cache_size",
    "mysql_global_variables_metadata_locks_hash_instances",
    "mysql_global_variables_min_examined_row_limit",
    "mysql_global_variables_multi_range_count",
    "mysql_global_variables_myisam_data_pointer_size",
    "mysql_global_variables_myisam_max_sort_file_size",
    "mysql_global_variables_myisam_mmap_size",
    "mysql_global_variables_myisam_recover_options",
    "mysql_global_variables_myisam_repair_threads",
    "mysql_global_variables_myisam_sort_buffer_size",
    "mysql_global_variables_myisam_use_mmap",
    "mysql_global_variables_mysql_native_password_proxy_users",
    "mysql_global_variables_net_buffer_length",
    "mysql_global_variables_net_read_timeout",
    "mysql_global_variables_net_retry_count",
    "mysql_global_variables_net_write_timeout",
    "mysql_global_variables_new",
    "mysql_global_variables_ngram_token_size",
    "mysql_global_variables_offline_mode",
    "mysql_global_variables_old",
    "mysql_global_variables_old_alter_table",
    "mysql_global_variables_old_passwords",
    "mysql_global_variables_open_files_limit",
    "mysql_global_variables_optimizer_prune_level",
    "mysql_global_variables_optimizer_search_depth",
    "mysql_global_variables_optimizer_trace_limit",
    "mysql_global_variables_optimizer_trace_max_mem_size",
    "mysql_global_variables_optimizer_trace_offset",
    "mysql_global_variables_parser_max_mem_size",
    "mysql_global_variables_performance_schema",
    "mysql_global_variables_performance_schema_accounts_size",
    "mysql_global_variables_performance_schema_digests_size",
    "mysql_global_variables_performance_schema_events_stages_history_long_size",
    "mysql_global_variables_performance_schema_events_stages_history_size",
    "mysql_global_variables_performance_schema_events_statements_history_long_size",
    "mysql_global_variables_performance_schema_events_statements_history_size",
    "mysql_global_variables_performance_schema_events_transactions_history_long_size",
    "mysql_global_variables_performance_schema_events_transactions_history_size",
    "mysql_global_variables_performance_schema_events_waits_history_long_size",
    "mysql_global_variables_performance_schema_events_waits_history_size",
    "mysql_global_variables_performance_schema_hosts_size",
    "mysql_global_variables_performance_schema_max_cond_classes",
    "mysql_global_variables_performance_schema_max_cond_instances",
    "mysql_global_variables_performance_schema_max_digest_length",
    "mysql_global_variables_performance_schema_max_file_classes",
    "mysql_global_variables_performance_schema_max_file_handles",
    "mysql_global_variables_performance_schema_max_file_instances",
    "mysql_global_variables_performance_schema_max_index_stat",
    "mysql_global_variables_performance_schema_max_memory_classes",
    "mysql_global_variables_performance_schema_max_metadata_locks",
    "mysql_global_variables_performance_schema_max_mutex_classes",
    "mysql_global_variables_performance_schema_max_mutex_instances",
    "mysql_global_variables_performance_schema_max_prepared_statements_instances",
    "mysql_global_variables_performance_schema_max_program_instances",
    "mysql_global_variables_performance_schema_max_rwlock_classes",
    "mysql_global_variables_performance_schema_max_rwlock_instances",
    "mysql_global_variables_performance_schema_max_socket_classes",
    "mysql_global_variables_performance_schema_max_socket_instances",
    "mysql_global_variables_performance_schema_max_sql_text_length",
    "mysql_global_variables_performance_schema_max_stage_classes",
    "mysql_global_variables_performance_schema_max_statement_classes",
    "mysql_global_variables_performance_schema_max_statement_stack",
    "mysql_global_variables_performance_schema_max_table_handles",
    "mysql_global_variables_performance_schema_max_table_instances",
    "mysql_global_variables_performance_schema_max_table_lock_stat",
    "mysql_global_variables_performance_schema_max_thread_classes",
    "mysql_global_variables_performance_schema_max_thread_instances",
    "mysql_global_variables_performance_schema_session_connect_attrs_size",
    "mysql_global_variables_performance_schema_setup_actors_size",
    "mysql_global_variables_performance_schema_setup_objects_size",
    "mysql_global_variables_performance_schema_users_size",
    "mysql_global_variables_port",
    "mysql_global_variables_preload_buffer_size",
    "mysql_global_variables_profiling",
    "mysql_global_variables_profiling_history_size",
    "mysql_global_variables_protocol_version",
    "mysql_global_variables_query_alloc_block_size",
    "mysql_global_variables_query_cache_limit",
    "mysql_global_variables_query_cache_min_res_unit",
    "mysql_global_variables_query_cache_size",
    "mysql_global_variables_query_cache_type",
    "mysql_global_variables_query_cache_wlock_invalidate",
    "mysql_global_variables_query_prealloc_size",
    "mysql_global_variables_range_alloc_block_size",
    "mysql_global_variables_range_optimizer_max_mem_size",
    "mysql_global_variables_read_buffer_size",
    "mysql_global_variables_read_only",
    "mysql_global_variables_read_rnd_buffer_size",
    "mysql_global_variables_relay_log_purge",
    "mysql_global_variables_relay_log_recovery",
    "mysql_global_variables_relay_log_space_limit",
    "mysql_global_variables_report_port",
    "mysql_global_variables_require_secure_transport",
    "mysql_global_variables_rpl_stop_slave_timeout",
    "mysql_global_variables_secure_auth",
    "mysql_global_variables_server_id",
    "mysql_global_variables_server_id_bits",
    "mysql_global_variables_session_track_gtids",
    "mysql_global_variables_session_track_schema",
    "mysql_global_variables_session_track_state_change",
    "mysql_global_variables_session_track_transaction_info",
    "mysql_global_variables_sha256_password_proxy_users",
    "mysql_global_variables_show_compatibility_56",
    "mysql_global_variables_show_old_temporals",
    "mysql_global_variables_skip_external_locking",
    "mysql_global_variables_skip_name_resolve",
    "mysql_global_variables_skip_networking",
    "mysql_global_variables_skip_show_database",
    "mysql_global_variables_slave_allow_batching",
    "mysql_global_variables_slave_checkpoint_group",
    "mysql_global_variables_slave_checkpoint_period",
    "mysql_global_variables_slave_compressed_protocol",
    "mysql_global_variables_slave_max_allowed_packet",
    "mysql_global_variables_slave_net_timeout",
    "mysql_global_variables_slave_parallel_workers",
    "mysql_global_variables_slave_pending_jobs_size_max",
    "mysql_global_variables_slave_preserve_commit_order",
    "mysql_global_variables_slave_skip_errors",
    "mysql_global_variables_slave_sql_verify_checksum",
    "mysql_global_variables_slave_transaction_retries",
    "mysql_global_variables_slow_launch_time",
    "mysql_global_variables_slow_query_log",
    "mysql_global_variables_sort_buffer_size",
    "mysql_global_variables_sql_auto_is_null",
    "mysql_global_variables_sql_big_selects",
    "mysql_global_variables_sql_buffer_result",
    "mysql_global_variables_sql_log_off",
    "mysql_global_variables_sql_notes",
    "mysql_global_variables_sql_quote_show_create",
    "mysql_global_variables_sql_safe_updates",
    "mysql_global_variables_sql_select_limit",
    "mysql_global_variables_sql_slave_skip_counter",
    "mysql_global_variables_sql_warnings",
    "mysql_global_variables_stored_program_cache",
    "mysql_global_variables_super_read_only",
    "mysql_global_variables_sync_binlog",
    "mysql_global_variables_sync_frm",
    "mysql_global_variables_sync_master_info",
    "mysql_global_variables_sync_relay_log",
    "mysql_global_variables_sync_relay_log_info",
    "mysql_global_variables_table_definition_cache",
    "mysql_global_variables_table_open_cache",
    "mysql_global_variables_table_open_cache_instances",
    "mysql_global_variables_thread_cache_size",
    "mysql_global_variables_thread_stack",
    "mysql_global_variables_tmp_table_size",
    "mysql_global_variables_transaction_alloc_block_size",
    "mysql_global_variables_transaction_prealloc_size",
    "mysql_global_variables_transaction_write_set_extraction",
    "mysql_global_variables_tx_read_only",
    "mysql_global_variables_unique_checks",
    "mysql_global_variables_updatable_views_with_limit",
    "mysql_global_variables_wait_timeout",
    "mysql_info_schema_innodb_cmp_compress_ops_ok_total",
    "mysql_info_schema_innodb_cmp_compress_ops_total",
    "mysql_info_schema_innodb_cmp_compress_time_seconds_total",
    "mysql_info_schema_innodb_cmp_uncompress_ops_total",
    "mysql_info_schema_innodb_cmp_uncompress_time_seconds_total",
    "mysql_info_schema_innodb_cmpmem_pages_free_total",
    "mysql_info_schema_innodb_cmpmem_pages_used_total",
    "mysql_info_schema_innodb_cmpmem_relocation_ops_total",
    "mysql_info_schema_innodb_cmpmem_relocation_time_seconds_total",
    "mysql_transaction_isolation",
    "mysql_up",
    "mysql_version_info",
    "mysqld_exporter_build_info",
]

RABBITMQ_METRIC_FILE = [
    "rabbitmq_acknowledged_published_total",
    "rabbitmq_acknowledged_total",
    "rabbitmq_channels",
    "rabbitmq_connections",
    "rabbitmq_consumed_total",
    "rabbitmq_consumers",
    "rabbitmq_exchanges",
    "rabbitmq_exporter_build_info",
    "rabbitmq_failed_to_publish_total",
    "rabbitmq_fd_available",
    "rabbitmq_fd_used",
    "rabbitmq_messages_deliver_no_ack_rate",
    "rabbitmq_messages_deliver_rate",
    "rabbitmq_messages_publish_rate",
    "rabbitmq_module_scrape_duration_seconds",
    "rabbitmq_module_up",
    "rabbitmq_node_disk_free",
    "rabbitmq_node_disk_free_alarm",
    "rabbitmq_node_disk_free_limit",
    "rabbitmq_node_mem_alarm",
    "rabbitmq_node_mem_limit",
    "rabbitmq_node_mem_used",
    "rabbitmq_not_acknowledged_published_total",
    "rabbitmq_partitions",
    "rabbitmq_published_total",
    "rabbitmq_queue_consumer_utilisation",
    "rabbitmq_queue_consumers",
    "rabbitmq_queue_disk_reads_total",
    "rabbitmq_queue_disk_writes_total",
    "rabbitmq_queue_gc_collections_before_fullsweep",
    "rabbitmq_queue_gc_min_heap",
    "rabbitmq_queue_gc_min_vheap",
    "rabbitmq_queue_gc_minor_collections_total",
    "rabbitmq_queue_idle_since_seconds",
    "rabbitmq_queue_memory",
    "rabbitmq_queue_message_bytes",
    "rabbitmq_queue_message_bytes_persistent",
    "rabbitmq_queue_message_bytes_ram",
    "rabbitmq_queue_message_bytes_ready",
    "rabbitmq_queue_message_bytes_unacknowledged",
    "rabbitmq_queue_messages",
    "rabbitmq_queue_messages_ack_total",
    "rabbitmq_queue_messages_confirmed_total",
    "rabbitmq_queue_messages_deliver_no_ack_rate",
    "rabbitmq_queue_messages_deliver_rate",
    "rabbitmq_queue_messages_delivered_noack_total",
    "rabbitmq_queue_messages_delivered_total",
    "rabbitmq_queue_messages_get_noack_total",
    "rabbitmq_queue_messages_get_total",
    "rabbitmq_queue_messages_global",
    "rabbitmq_queue_messages_persistent",
    "rabbitmq_queue_messages_publish_rate",
    "rabbitmq_queue_messages_published_total",
    "rabbitmq_queue_messages_ram",
    "rabbitmq_queue_messages_ready",
    "rabbitmq_queue_messages_ready_global",
    "rabbitmq_queue_messages_ready_ram",
    "rabbitmq_queue_messages_redelivered_total",
    "rabbitmq_queue_messages_returned_total",
    "rabbitmq_queue_messages_unacknowledged",
    "rabbitmq_queue_messages_unacknowledged_global",
    "rabbitmq_queue_messages_unacknowledged_ram",
    "rabbitmq_queue_reductions_total",
    "rabbitmq_queue_state",
    "rabbitmq_queues",
    "rabbitmq_rejected_total",
    "rabbitmq_running",
    "rabbitmq_sockets_available",
    "rabbitmq_sockets_used",
    "rabbitmq_unrouted_published_total",
    "rabbitmq_up",
    "rabbitmq_uptime",
    "rabbitmq_version_info",
]

REDIS_METRIC_FILE = [
    "redis_active_defrag_running",
    "redis_allocator_active_bytes",
    "redis_allocator_allocated_bytes",
    "redis_allocator_frag_bytes",
    "redis_allocator_frag_ratio",
    "redis_allocator_resident_bytes",
    "redis_allocator_rss_bytes",
    "redis_allocator_rss_ratio",
    "redis_aof_current_rewrite_duration_sec",
    "redis_aof_enabled",
    "redis_aof_last_bgrewrite_status",
    "redis_aof_last_cow_size_bytes",
    "redis_aof_last_rewrite_duration_sec",
    "redis_aof_last_write_status",
    "redis_aof_rewrite_in_progress",
    "redis_aof_rewrite_scheduled",
    "redis_blocked_clients",
    "redis_client_recent_max_input_buffer_bytes",
    "redis_client_recent_max_output_buffer_bytes",
    "redis_clients_in_timeout_table",
    "redis_cluster_connections",
    "redis_cluster_enabled",
    "redis_commands_duration_seconds_total",
    "redis_commands_failed_calls_total",
    "redis_commands_latencies_usec_bucket",
    "redis_commands_latencies_usec_count",
    "redis_commands_latencies_usec_sum",
    "redis_commands_processed_total",
    "redis_commands_rejected_calls_total",
    "redis_commands_total",
    "redis_config_io_threads",
    "redis_config_maxclients",
    "redis_config_maxmemory",
    "redis_connected_clients",
    "redis_connected_slaves",
    "redis_connections_received_total",
    "redis_cpu_sys_children_seconds_total",
    "redis_cpu_sys_main_thread_seconds_total",
    "redis_cpu_sys_seconds_total",
    "redis_cpu_user_children_seconds_total",
    "redis_cpu_user_main_thread_seconds_total",
    "redis_cpu_user_seconds_total",
    "redis_db0_distrib_strings_sizes",
    "redis_db_avg_ttl_seconds",
    "redis_db_keys",
    "redis_db_keys_cached",
    "redis_db_keys_expiring",
    "redis_defrag_hits",
    "redis_defrag_key_hits",
    "redis_defrag_key_misses",
    "redis_defrag_misses",
    "redis_dump_payload_sanitizations",
    "redis_evicted_keys_total",
    "redis_expired_keys_total",
    "redis_expired_stale_percentage",
    "redis_expired_time_cap_reached_total",
    "redis_exporter_build_info",
    "redis_exporter_last_scrape_connect_time_seconds",
    "redis_exporter_last_scrape_duration_seconds",
    "redis_exporter_last_scrape_error",
    "redis_exporter_scrape_duration_seconds_count",
    "redis_exporter_scrape_duration_seconds_sum",
    "redis_exporter_scrapes_total",
    "redis_instance_info",
    "redis_io_threaded_reads_processed",
    "redis_io_threaded_writes_processed",
    "redis_io_threads_active",
    "redis_keyspace_hits_total",
    "redis_keyspace_misses_total",
    "redis_last_key_groups_scrape_duration_milliseconds",
    "redis_last_slow_execution_duration_seconds",
    "redis_latency_percentiles_usec",
    "redis_latency_percentiles_usec_count",
    "redis_latency_percentiles_usec_sum",
    "redis_latest_fork_seconds",
    "redis_lazyfree_pending_objects",
    "redis_loading_dump_file",
    "redis_master_repl_offset",
    "redis_mem_clients_normal",
    "redis_mem_clients_slaves",
    "redis_mem_fragmentation_bytes",
    "redis_mem_fragmentation_ratio",
    "redis_mem_not_counted_for_eviction_bytes",
    "redis_mem_total_replication_buffers_bytes",
    "redis_memory_max_bytes",
    "redis_memory_used_bytes",
    "redis_memory_used_dataset_bytes",
    "redis_memory_used_lua_bytes",
    "redis_memory_used_overhead_bytes",
    "redis_memory_used_peak_bytes",
    "redis_memory_used_rss_bytes",
    "redis_memory_used_scripts_bytes",
    "redis_memory_used_startup_bytes",
    "redis_migrate_cached_sockets_total",
    "redis_module_fork_in_progress",
    "redis_module_fork_last_cow_size",
    "redis_net_input_bytes_total",
    "redis_net_output_bytes_total",
    "redis_number_of_cached_scripts",
    "redis_process_id",
    "redis_pubsub_channels",
    "redis_pubsub_patterns",
    "redis_pubsubshard_channels",
    "redis_rdb_bgsave_in_progress",
    "redis_rdb_changes_since_last_save",
    "redis_rdb_current_bgsave_duration_sec",
    "redis_rdb_last_bgsave_duration_sec",
    "redis_rdb_last_bgsave_status",
    "redis_rdb_last_cow_size_bytes",
    "redis_rdb_last_load_expired_keys",
    "redis_rdb_last_load_loaded_keys",
    "redis_rdb_last_save_timestamp_seconds",
    "redis_rdb_saves_total",
    "redis_rejected_connections_total",
    "redis_repl_backlog_first_byte_offset",
    "redis_repl_backlog_history_bytes",
    "redis_repl_backlog_is_active",
    "redis_replica_partial_resync_accepted",
    "redis_replica_partial_resync_denied",
    "redis_replica_resyncs_full",
    "redis_replication_backlog_bytes",
    "redis_second_repl_offset",
    "redis_slave_expires_tracked_keys",
    "redis_slowlog_last_id",
    "redis_slowlog_length",
    "redis_start_time_seconds",
    "redis_target_scrape_request_errors_total",
    "redis_total_error_replies",
    "redis_total_reads_processed",
    "redis_total_writes_processed",
    "redis_tracking_clients",
    "redis_tracking_total_items",
    "redis_tracking_total_keys",
    "redis_tracking_total_prefixes",
    "redis_unexpected_error_replies",
    "redis_up",
    "redis_uptime_in_seconds",
]

NGINX_METRIC_FILE = [
    "nginx_ingress_controller_admission_config_size",
    "nginx_ingress_controller_admission_render_duration",
    "nginx_ingress_controller_admission_render_ingresses",
    "nginx_ingress_controller_admission_roundtrip_duration",
    "nginx_ingress_controller_admission_tested_duration",
    "nginx_ingress_controller_admission_tested_ingresses",
    "nginx_ingress_controller_build_info",
    "nginx_ingress_controller_bytes_sent_bucket",
    "nginx_ingress_controller_bytes_sent_count",
    "nginx_ingress_controller_bytes_sent_sum",
    "nginx_ingress_controller_config_hash",
    "nginx_ingress_controller_config_last_reload_successful",
    "nginx_ingress_controller_config_last_reload_successful_timestamp_seconds",
    "nginx_ingress_controller_connect_duration_seconds_bucket",
    "nginx_ingress_controller_connect_duration_seconds_count",
    "nginx_ingress_controller_connect_duration_seconds_sum",
    "nginx_ingress_controller_header_duration_seconds_bucket",
    "nginx_ingress_controller_header_duration_seconds_count",
    "nginx_ingress_controller_header_duration_seconds_sum",
    "nginx_ingress_controller_leader_election_status",
    "nginx_ingress_controller_nginx_process_connections",
    "nginx_ingress_controller_nginx_process_connections_total",
    "nginx_ingress_controller_nginx_process_cpu_seconds_total",
    "nginx_ingress_controller_nginx_process_num_procs",
    "nginx_ingress_controller_nginx_process_oldest_start_time_seconds",
    "nginx_ingress_controller_nginx_process_read_bytes_total",
    "nginx_ingress_controller_nginx_process_requests_total",
    "nginx_ingress_controller_nginx_process_resident_memory_bytes",
    "nginx_ingress_controller_nginx_process_virtual_memory_bytes",
    "nginx_ingress_controller_nginx_process_write_bytes_total",
    "nginx_ingress_controller_orphan_ingress",
    "nginx_ingress_controller_request_duration_seconds_bucket",
    "nginx_ingress_controller_request_duration_seconds_count",
    "nginx_ingress_controller_request_duration_seconds_sum",
    "nginx_ingress_controller_request_size_bucket",
    "nginx_ingress_controller_request_size_count",
    "nginx_ingress_controller_request_size_sum",
    "nginx_ingress_controller_requests",
    "nginx_ingress_controller_response_duration_seconds_bucket",
    "nginx_ingress_controller_response_duration_seconds_count",
    "nginx_ingress_controller_response_duration_seconds_sum",
    "nginx_ingress_controller_response_size_bucket",
    "nginx_ingress_controller_response_size_count",
    "nginx_ingress_controller_response_size_sum",
    "nginx_ingress_controller_ssl_certificate_info",
    "nginx_ingress_controller_ssl_expire_time_seconds",
    "nginx_ingress_controller_success",
]

NODE_METRIC_FILE = [
    "instance:node_cpu:ratio",
    "instance:node_cpu_utilisation:rate5m",
    "instance:node_load1_per_cpu:ratio",
    "instance:node_memory_utilisation:ratio",
    "instance:node_network_receive_bytes_excluding_lo:rate5m",
    "instance:node_network_receive_drop_excluding_lo:rate5m",
    "instance:node_network_transmit_bytes_excluding_lo:rate5m",
    "instance:node_network_transmit_drop_excluding_lo:rate5m",
    "instance:node_num_cpu:sum",
    "instance:node_vmstat_pgmajfault:rate5m",
    "instance_device:node_disk_io_time_seconds:rate5m",
    "instance_device:node_disk_io_time_weighted_seconds:rate5m",
    "node_arp_entries",
    "node_boot_time_seconds",
    "node_context_switches_total",
    "node_cooling_device_cur_state",
    "node_cooling_device_max_state",
    "node_cpu_guest_seconds_total",
    "node_cpu_seconds_total",
    "node_disk_discard_time_seconds_total",
    "node_disk_discarded_sectors_total",
    "node_disk_discards_completed_total",
    "node_disk_discards_merged_total",
    "node_disk_flush_requests_time_seconds_total",
    "node_disk_flush_requests_total",
    "node_disk_info",
    "node_disk_io_now",
    "node_disk_io_time_seconds_total",
    "node_disk_io_time_weighted_seconds_total",
    "node_disk_read_bytes_total",
    "node_disk_read_time_seconds_total",
    "node_disk_reads_completed_total",
    "node_disk_reads_merged_total",
    "node_disk_write_time_seconds_total",
    "node_disk_writes_completed_total",
    "node_disk_writes_merged_total",
    "node_disk_written_bytes_total",
    "node_dmi_info",
    "node_entropy_available_bits",
    "node_entropy_pool_size_bits",
    "node_exporter_build_info",
    "node_filefd_allocated",
    "node_filefd_maximum",
    "node_filesystem_avail_bytes",
    "node_filesystem_device_error",
    "node_filesystem_files",
    "node_filesystem_files_free",
    "node_filesystem_free_bytes",
    "node_filesystem_mount_info",
    "node_filesystem_purgeable_bytes",
    "node_filesystem_readonly",
    "node_filesystem_size_bytes",
    "node_forks_total",
    "node_intr_total",
    "node_load1",
    "node_load15",
    "node_load5",
    "node_memory_Active_anon_bytes",
    "node_memory_Active_bytes",
    "node_memory_Active_file_bytes",
    "node_memory_AnonHugePages_bytes",
    "node_memory_AnonPages_bytes",
    "node_memory_Bounce_bytes",
    "node_memory_Buffers_bytes",
    "node_memory_Cached_bytes",
    "node_memory_CommitLimit_bytes",
    "node_memory_Committed_AS_bytes",
    "node_memory_DirectMap1G_bytes",
    "node_memory_DirectMap2M_bytes",
    "node_memory_DirectMap4k_bytes",
    "node_memory_Dirty_bytes",
    "node_memory_HardwareCorrupted_bytes",
    "node_memory_HugePages_Free",
    "node_memory_HugePages_Rsvd",
    "node_memory_HugePages_Surp",
    "node_memory_HugePages_Total",
    "node_memory_Hugepagesize_bytes",
    "node_memory_Inactive_anon_bytes",
    "node_memory_Inactive_bytes",
    "node_memory_Inactive_file_bytes",
    "node_memory_KernelStack_bytes",
    "node_memory_Mapped_bytes",
    "node_memory_MemAvailable_bytes",
    "node_memory_MemFree_bytes",
    "node_memory_MemTotal_bytes",
    "node_memory_Mlocked_bytes",
    "node_memory_NFS_Unstable_bytes",
    "node_memory_PageTables_bytes",
    "node_memory_Percpu_bytes",
    "node_memory_SReclaimable_bytes",
    "node_memory_SUnreclaim_bytes",
    "node_memory_ShmemHugePages_bytes",
    "node_memory_ShmemPmdMapped_bytes",
    "node_memory_Shmem_bytes",
    "node_memory_Slab_bytes",
    "node_memory_SwapCached_bytes",
    "node_memory_SwapFree_bytes",
    "node_memory_SwapTotal_bytes",
    "node_memory_Unevictable_bytes",
    "node_memory_VmallocChunk_bytes",
    "node_memory_VmallocTotal_bytes",
    "node_memory_VmallocUsed_bytes",
    "node_memory_WritebackTmp_bytes",
    "node_memory_Writeback_bytes",
    "node_netstat_Icmp6_InErrors",
    "node_netstat_Icmp6_InMsgs",
    "node_netstat_Icmp6_OutMsgs",
    "node_netstat_Icmp_InErrors",
    "node_netstat_Icmp_InMsgs",
    "node_netstat_Icmp_OutMsgs",
    "node_netstat_Ip6_InOctets",
    "node_netstat_Ip6_OutOctets",
    "node_netstat_IpExt_InOctets",
    "node_netstat_IpExt_OutOctets",
    "node_netstat_Ip_Forwarding",
    "node_netstat_TcpExt_ListenDrops",
    "node_netstat_TcpExt_ListenOverflows",
    "node_netstat_TcpExt_SyncookiesFailed",
    "node_netstat_TcpExt_SyncookiesRecv",
    "node_netstat_TcpExt_SyncookiesSent",
    "node_netstat_TcpExt_TCPOFOQueue",
    "node_netstat_TcpExt_TCPRcvQDrop",
    "node_netstat_TcpExt_TCPSynRetrans",
    "node_netstat_TcpExt_TCPTimeouts",
    "node_netstat_Tcp_ActiveOpens",
    "node_netstat_Tcp_CurrEstab",
    "node_netstat_Tcp_InErrs",
    "node_netstat_Tcp_InSegs",
    "node_netstat_Tcp_OutRsts",
    "node_netstat_Tcp_OutSegs",
    "node_netstat_Tcp_PassiveOpens",
    "node_netstat_Tcp_RetransSegs",
    "node_netstat_Udp6_InDatagrams",
    "node_netstat_Udp6_InErrors",
    "node_netstat_Udp6_NoPorts",
    "node_netstat_Udp6_OutDatagrams",
    "node_netstat_Udp6_RcvbufErrors",
    "node_netstat_Udp6_SndbufErrors",
    "node_netstat_UdpLite6_InErrors",
    "node_netstat_UdpLite_InErrors",
    "node_netstat_Udp_InDatagrams",
    "node_netstat_Udp_InErrors",
    "node_netstat_Udp_NoPorts",
    "node_netstat_Udp_OutDatagrams",
    "node_netstat_Udp_RcvbufErrors",
    "node_netstat_Udp_SndbufErrors",
    "node_network_address_assign_type",
    "node_network_carrier",
    "node_network_carrier_changes_total",
    "node_network_carrier_down_changes_total",
    "node_network_carrier_up_changes_total",
    "node_network_device_id",
    "node_network_dormant",
    "node_network_flags",
    "node_network_iface_id",
    "node_network_iface_link",
    "node_network_iface_link_mode",
    "node_network_info",
    "node_network_mtu_bytes",
    "node_network_name_assign_type",
    "node_network_net_dev_group",
    "node_network_protocol_type",
    "node_network_receive_bytes_total",
    "node_network_receive_compressed_total",
    "node_network_receive_drop_total",
    "node_network_receive_errs_total",
    "node_network_receive_fifo_total",
    "node_network_receive_frame_total",
    "node_network_receive_multicast_total",
    "node_network_receive_nohandler_total",
    "node_network_receive_packets_total",
    "node_network_speed_bytes",
    "node_network_transmit_bytes_total",
    "node_network_transmit_carrier_total",
    "node_network_transmit_colls_total",
    "node_network_transmit_compressed_total",
    "node_network_transmit_drop_total",
    "node_network_transmit_errs_total",
    "node_network_transmit_fifo_total",
    "node_network_transmit_packets_total",
    "node_network_transmit_queue_length",
    "node_network_up",
    "node_nf_conntrack_entries",
    "node_nf_conntrack_entries_limit",
    "node_nf_conntrack_stat_drop",
    "node_nf_conntrack_stat_early_drop",
    "node_nf_conntrack_stat_found",
    "node_nf_conntrack_stat_ignore",
    "node_nf_conntrack_stat_insert",
    "node_nf_conntrack_stat_insert_failed",
    "node_nf_conntrack_stat_invalid",
    "node_nf_conntrack_stat_search_restart",
    "node_os_info",
    "node_os_version",
    "node_pressure_cpu_waiting_seconds_total",
    "node_pressure_io_stalled_seconds_total",
    "node_pressure_io_waiting_seconds_total",
    "node_pressure_memory_stalled_seconds_total",
    "node_pressure_memory_waiting_seconds_total",
    "node_procs_blocked",
    "node_procs_running",
    "node_schedstat_running_seconds_total",
    "node_schedstat_timeslices_total",
    "node_schedstat_waiting_seconds_total",
    "node_scrape_collector_duration_seconds",
    "node_scrape_collector_success",
    "node_selinux_enabled",
    "node_sockstat_FRAG6_inuse",
    "node_sockstat_FRAG6_memory",
    "node_sockstat_FRAG_inuse",
    "node_sockstat_FRAG_memory",
    "node_sockstat_RAW6_inuse",
    "node_sockstat_RAW_inuse",
    "node_sockstat_TCP6_inuse",
    "node_sockstat_TCP_alloc",
    "node_sockstat_TCP_inuse",
    "node_sockstat_TCP_mem",
    "node_sockstat_TCP_mem_bytes",
    "node_sockstat_TCP_orphan",
    "node_sockstat_TCP_tw",
    "node_sockstat_UDP6_inuse",
    "node_sockstat_UDPLITE6_inuse",
    "node_sockstat_UDPLITE_inuse",
    "node_sockstat_UDP_inuse",
    "node_sockstat_UDP_mem",
    "node_sockstat_UDP_mem_bytes",
    "node_sockstat_sockets_used",
    "node_softnet_backlog_len",
    "node_softnet_cpu_collision_total",
    "node_softnet_dropped_total",
    "node_softnet_flow_limit_count_total",
    "node_softnet_processed_total",
    "node_softnet_received_rps_total",
    "node_softnet_times_squeezed_total",
    "node_textfile_scrape_error",
    "node_time_clocksource_available_info",
    "node_time_clocksource_current_info",
    "node_time_seconds",
    "node_time_zone_offset_seconds",
    "node_timex_estimated_error_seconds",
    "node_timex_frequency_adjustment_ratio",
    "node_timex_loop_time_constant",
    "node_timex_maxerror_seconds",
    "node_timex_offset_seconds",
    "node_timex_pps_calibration_total",
    "node_timex_pps_error_total",
    "node_timex_pps_frequency_hertz",
    "node_timex_pps_jitter_seconds",
    "node_timex_pps_jitter_total",
    "node_timex_pps_shift_seconds",
    "node_timex_pps_stability_exceeded_total",
    "node_timex_pps_stability_hertz",
    "node_timex_status",
    "node_timex_sync_status",
    "node_timex_tai_offset_seconds",
    "node_timex_tick_seconds",
    "node_udp_queues",
    "node_uname_info",
    "node_vmstat_oom_kill",
    "node_vmstat_pgfault",
    "node_vmstat_pgmajfault",
    "node_vmstat_pgpgin",
    "node_vmstat_pgpgout",
    "node_vmstat_pswpin",
    "node_vmstat_pswpout",
]

ISTIO_METRIC_FILE = [
    "istio_build",
    "istio_request_bytes_bucket",
    "istio_request_bytes_count",
    "istio_request_bytes_sum",
    "istio_request_duration_milliseconds_bucket",
    "istio_request_duration_milliseconds_count",
    "istio_request_duration_milliseconds_sum",
    "istio_requests_total",
    "istio_response_bytes_bucket",
    "istio_response_bytes_count",
    "istio_response_bytes_sum",
    "istio_tcp_connections_closed_total",
    "istio_tcp_connections_opened_total",
    "istio_tcp_received_bytes_total",
    "istio_tcp_sent_bytes_total",
]

MONGODB_TARGET_PODS = [
    pod.strip() for pod in os.environ.get("MONGODB_TARGET_PODS", "carts-db-0,user-db-0,orders-db-0").split(",") if pod.strip()
]
MYSQL_TARGET_PODS = [
    pod.strip() for pod in os.environ.get("MYSQL_TARGET_PODS", "catalogue-db-0").split(",") if pod.strip()
]
RABBITMQ_TARGET_PODS = [
    pod.strip() for pod in os.environ.get("RABBITMQ_TARGET_PODS", "rabbitmq-0").split(",") if pod.strip()
]
REDIS_TARGET_PODS = [
    pod.strip() for pod in os.environ.get("REDIS_TARGET_PODS", "session-db-0").split(",") if pod.strip()
]

DATE_TEXT = "2026_02_28"
HOURS: str | list[int] = [16]  # "all" or e.g. [1, 2, 13]
TIMEZONE = timezone.utc
PYTHON_BIN = sys.executable

OUTPUT_ROOT = DATASET_BASE
OUTPUT_DIR_NAMES = {
    "logs": "logs",
    "traces": "traces",
    "metrics": "metrics",
}

PROM_URL = os.environ.get("PROM_URL", "http://34.28.33.102:30990")
PROM_NAMESPACE = os.environ.get("PROM_NAMESPACE", "sock-shop")
LOKI_URL = os.environ.get("LOKI_URL", "http://34.28.33.102:31300")
LOKI_QUERY = os.environ.get("LOKI_QUERY", '{namespace="sock-shop"}')
JAEGER_URL = os.environ.get("JAEGER_URL", "http://34.28.33.102:32614")

PROM_STEP = os.environ.get("PROM_STEP", "5s")
KUBE_POD_STEP = os.environ.get("KUBE_POD_STEP", "5m")
KPI_WINDOW = os.environ.get("KPI_WINDOW", "30s")
RESTART_COUNT_WINDOW = os.environ.get("RESTART_COUNT_WINDOW", "1m")
ISTIO_WINDOW = os.environ.get("ISTIO_WINDOW", "30s")
PROM_TIMEOUT_SEC = int(os.environ.get("PROM_TIMEOUT_SEC", "120"))
NETWORK_KPI_WINDOW = os.environ.get("NETWORK_KPI_WINDOW", "1m") 

RUN_METRICS = True
RUN_LOGS = True
RUN_TRACES = True

METRIC_COLLECTORS = [
    {
        "name": "application",
        "handler": "application",
        "output_name": "prometheus_metrics_application_raw",
        "step": PROM_STEP,
    },
    {
        "name": "container",
        "handler": "container",
        "output_name": "prometheus_metrics_container_raw",
        "step": PROM_STEP,
    },
    {
        "name": "KPI",
        "handler": "KPI",
        "output_name": "prometheus_metrics_KPI",
        "step": PROM_STEP,
    },
    {
        "name": "middleware",
        "handler": "middleware",
        "output_name": "prometheus_metrics_middleware_raw",
        "step": PROM_STEP,
    },
    {
        "name": "network",
        "handler": "network",
        "output_name": "prometheus_metrics_network_raw",
        "step": "30s",
    },
    {
        "name": "node",
        "handler": "node",
        "output_name": "prometheus_metrics_node_raw",
        "step": PROM_STEP,
    },
    {
        "name": "service_proxy",
        "handler": "service_proxy",
        "output_name": "prometheus_metrics_service_proxy_raw",
        "step": "30s",
    },
]

SUMMARY_FILE_NAME = "collection_summary.json"

LOG_RAW_COLUMNS = ["timestamp", "node", "pod", "container", "log"]
TRACE_RAW_COLUMNS = [
    "start_time",
    "trace_id",
    "span_id",
    "service",
    "operation",
    "duration",
    "references",
    "tags",
    "logs",
]
LOG_PARSED_COLUMNS = [
    "timestamp",
    "trace_id",
    "span_id",
    "service",
    "node",
    "pod",
    "container",
    "log_level",
    "log_source",
    "log_type",
    "message",
    "raw_log",
]
TRACE_PARSED_COLUMNS = [
    "timestamp",
    "trace_id",
    "span_id",
    "parent_span_id",
    "service",
    "operation",
    "duration",
    "span_kind",
    "status_code",
    "status",
    "peer_service",
    "http_method",
    "http_url",
    "exception_type",
    "exception_message",
    "pod",
    "container",
    "node",
    "tags_json",
]
JAEGER_EXCLUDED_SERVICES = {"jaeger-all-in-one", "jaeger-query", "jaeger-collector"}
JAEGER_LIMIT = int(os.environ.get("JAEGER_LIMIT", "10000"))
JAEGER_HTTP_TIMEOUT_SECONDS = int(os.environ.get("JAEGER_HTTP_TIMEOUT_SECONDS", "120"))
JAEGER_HTTP_RETRIES = int(os.environ.get("JAEGER_HTTP_RETRIES", "4"))
JAEGER_HTTP_BACKOFF_SECONDS = float(os.environ.get("JAEGER_HTTP_BACKOFF_SECONDS", "2"))
JAEGER_FETCH_MODE = os.environ.get("JAEGER_FETCH_MODE", "per_service").strip() or "per_service"
JAEGER_SPLIT_MIN_SECONDS = int(os.environ.get("JAEGER_SPLIT_MIN_SECONDS", "1"))
LOKI_LIMIT = int(os.environ.get("LOKI_LIMIT", "5000"))
LOKI_DIRECTION = os.environ.get("LOKI_DIRECTION", "forward")
LOKI_SLICE_MINUTES = int(os.environ.get("LOKI_SLICE_MINUTES", "5"))
LOKI_HTTP_TIMEOUT_SECONDS = int(os.environ.get("LOKI_HTTP_TIMEOUT_SECONDS", "120"))
LOKI_HTTP_RETRIES = int(os.environ.get("LOKI_HTTP_RETRIES", "3"))

CONTAINER_TARGETS = {
    "front-end",
    "catalogue",
    "carts",
    "orders",
    "catalogue-db",
    "carts-db",
    "orders-db",
    "payment",
    "shipping",
    "queue-master",
    "user",
    "user-db",
    "rabbitmq",
    "session-db",
}

APPLICATION_SOURCE_CONFIGS = [
    {
        "name": "Go",
        "metric_file": GO_METRIC_FILE,
        "selector": lambda namespace: f'namespace="{namespace}",container=~"user|payment|catalogue"',
        "important_labels": ["le", "method", "path", "status_code", "route"],
    },
    {
        "name": "Java",
        "metric_file": JAVA_METRIC_FILE,
        "selector": lambda namespace: f'namespace="{namespace}",container=~"carts|orders|queue-master|shipping"',
        "important_labels": ["name", "method", "status", "net_peer_name", "uri", "id", "area", "state", "queue"],
    },
    {
        "name": "NodeJS",
        "metric_file": NODEJS_METRIC_FILE,
        "selector": lambda namespace: f'namespace="{namespace}",container="front-end"',
        "important_labels": ["space", "kind", "le", "method", "path", "status_code"],
    },
]

MIDDLEWARE_SOURCE_CONFIGS = [
    {
        "name": "MongoDB",
        "metric_file": MONGODB_METRIC_FILE,
        "namespace": lambda: os.environ.get("MONGODB_NAMESPACE", os.environ.get("MIDDLEWARE_NAMESPACE", PROM_NAMESPACE)).strip(),
        "pods": lambda: MONGODB_TARGET_PODS,
    },
    {
        "name": "MySQL",
        "metric_file": MYSQL_METRIC_FILE,
        "namespace": lambda: os.environ.get("MYSQL_NAMESPACE", os.environ.get("MIDDLEWARE_NAMESPACE", PROM_NAMESPACE)).strip(),
        "pods": lambda: MYSQL_TARGET_PODS,
    },
    {
        "name": "RabbitMQ",
        "metric_file": RABBITMQ_METRIC_FILE,
        "namespace": lambda: os.environ.get("RABBITMQ_NAMESPACE", os.environ.get("MIDDLEWARE_NAMESPACE", PROM_NAMESPACE)).strip(),
        "pods": lambda: RABBITMQ_TARGET_PODS,
    },
    {
        "name": "Redis",
        "metric_file": REDIS_METRIC_FILE,
        "namespace": lambda: os.environ.get("REDIS_NAMESPACE", os.environ.get("MIDDLEWARE_NAMESPACE", PROM_NAMESPACE)).strip(),
        "pods": lambda: REDIS_TARGET_PODS,
    },
]

NETWORK_LABELS = ["status", "method", "ingress", "path", "host"]
SERVICE_PROXY_LABELS = [
    "source_workload",
    "destination_workload",
    "response_code",
    "response_flags",
    "reporter",
    "request_protocol",
]
KPI_NAMES = [
    "request_rate",
    "success_rate",
    "error_count",
    "latency_p50",
    "latency_p90",
    "latency_p95",
    "latency_p99",
    "cpu_usage_pct",
    "memory_usage_pct",
    "restart_count",
    "ready_ratio",
    "network_rx",
    "network_tx",
]


def parse_date(date_text: str) -> datetime:
    return datetime.strptime(date_text, "%Y_%m_%d").replace(tzinfo=TIMEZONE)


def normalize_hours(hours: str | list[int]) -> list[int]:
    if isinstance(hours, str):
        if hours.lower() == "all":
            return list(range(24))
        raise ValueError("HOURS must be 'all' or a list of integers between 0 and 23.")

    normalized = sorted(set(int(hour) for hour in hours))
    invalid = [hour for hour in normalized if hour < 0 or hour > 23]
    if invalid:
        raise ValueError(f"Invalid HOURS entries: {invalid}")
    return normalized


def to_env_time(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def hour_suffix(hour: int) -> str:
    return f"{hour:02d}"


def timestamp_to_str(ts: float) -> str:
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def build_hour_windows(date_text: str, hours: str | list[int]) -> list[dict[str, Any]]:
    base = parse_date(date_text)
    windows = []
    for hour in normalize_hours(hours):
        start_dt = base + timedelta(hours=hour)
        end_dt = start_dt + timedelta(hours=1)
        windows.append(
            {
                "hour": hour,
                "suffix": hour_suffix(hour),
                "start_dt": start_dt,
                "end_dt": end_dt,
                "start_iso": to_env_time(start_dt),
                "end_iso": to_env_time(end_dt),
            }
        )
    return windows


def ensure_output_dirs(date_text: str) -> dict[str, Path]:
    date_dir = OUTPUT_ROOT / date_text
    dirs = {"date": date_dir}
    for key, dirname in OUTPUT_DIR_NAMES.items():
        path = date_dir / dirname
        path.mkdir(parents=True, exist_ok=True)
        dirs[key] = path
    return dirs


def write_empty_csv(path: Path, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)


def write_rows_csv(output_path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(output_path, index=False)


def resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def load_metric_list(metric_source: str | Path | list[str] | tuple[str, ...]) -> list[str]:
    if isinstance(metric_source, (list, tuple)):
        metrics: list[str] = []
        for metric in metric_source:
            metric_text = str(metric).strip()
            if metric_text:
                metrics.append(metric_text)
        return list(dict.fromkeys(metrics))

    metric_list_path = resolve_path(metric_source)
    if not metric_list_path.exists():
        raise FileNotFoundError(f"Metric list CSV not found: {metric_list_path}")

    metrics_df = pd.read_csv(metric_list_path)
    if "metric_name" not in metrics_df.columns:
        raise ValueError(f"`metric_name` column not found in {metric_list_path}")

    metrics = []
    for metric in metrics_df["metric_name"].dropna().astype(str).tolist():
        metric = metric.strip()
        if metric:
            metrics.append(metric)
    return list(dict.fromkeys(metrics))


def prom_query_range(expr: str, start_iso: str, end_iso: str, step: str) -> list[dict[str, Any]]:
    response = requests.get(
        f"{PROM_URL}/api/v1/query_range",
        params={"query": expr, "start": start_iso, "end": end_iso, "step": step},
        timeout=PROM_TIMEOUT_SEC,
    )
    response.raise_for_status()
    data = response.json()
    if data.get("status") != "success":
        return []
    return data.get("data", {}).get("result", [])


def prom_query(expr: str, ts: Optional[str] = None) -> list[dict[str, Any]]:
    params = {"query": expr}
    if ts is not None:
        params["time"] = ts
    response = requests.get(f"{PROM_URL}/api/v1/query", params=params, timeout=PROM_TIMEOUT_SEC)
    response.raise_for_status()
    data = response.json()
    if data.get("status") != "success":
        return []
    return data.get("data", {}).get("result", [])


def safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def dump_selected_labels(metric_labels: dict[str, Any], keys: list[str]) -> str:
    selected = {key: metric_labels[key] for key in keys if key in metric_labels}
    return json.dumps(selected, sort_keys=True, ensure_ascii=False)


def to_unix_ns(dt: datetime) -> int:
    return int(dt.timestamp() * 1_000_000_000)


def to_unix_us(dt: datetime) -> int:
    return int(dt.timestamp() * 1_000_000)


def loki_query_range(start_ns: int, end_ns: int) -> list[dict[str, Any]]:
    url = f"{LOKI_URL.rstrip('/')}/loki/api/v1/query_range"
    params = {
        "query": LOKI_QUERY,
        "start": str(start_ns),
        "end": str(end_ns),
        "direction": LOKI_DIRECTION,
    }
    if LOKI_LIMIT > 0:
        params["limit"] = str(LOKI_LIMIT)

    last_error: Exception | None = None
    for attempt in range(LOKI_HTTP_RETRIES):
        try:
            response = requests.get(url, params=params, timeout=LOKI_HTTP_TIMEOUT_SECONDS)
            response.raise_for_status()
            data = response.json()
            if data.get("status") != "success":
                raise RuntimeError(f"Loki error: {data}")
            return data.get("data", {}).get("result", [])
        except (ChunkedEncodingError, ConnectionError, Timeout, RequestException) as exc:
            last_error = exc
            if attempt >= LOKI_HTTP_RETRIES - 1:
                break
            wait_seconds = 2 ** attempt
            print(f"[Loki] retry {attempt + 1}/{LOKI_HTTP_RETRIES} after error: {exc}")
            time.sleep(wait_seconds)

    raise RuntimeError(f"Loki query failed: {last_error}")


def fetch_loki_slice_rows(start_ns: int, end_ns: int, seen_pods: set[str]) -> list[dict[str, Any]]:
    slice_rows: list[dict[str, Any]] = []
    seen_row_keys: set[tuple[Any, Any, Any, Any, Any]] = set()
    cursor_ns = start_ns
    request_idx = 0

    while cursor_ns < end_ns:
        request_idx += 1
        streams = loki_query_range(cursor_ns, end_ns)

        batch_count = 0
        max_ts_ns = None

        for stream in streams:
            labels = stream.get("stream", {})
            values = stream.get("values", [])

            for ts_ns_str, line in values:
                try:
                    ts_ns = int(ts_ns_str)
                except (TypeError, ValueError):
                    continue

                if max_ts_ns is None or ts_ns > max_ts_ns:
                    max_ts_ns = ts_ns

                row = {
                    "timestamp": datetime.fromtimestamp(ts_ns / 1_000_000_000, tz=timezone.utc).isoformat(),
                    "node": labels.get("node_name"),
                    "pod": labels.get("pod"),
                    "container": labels.get("container"),
                    "log": line,
                }
                row_key = (ts_ns, row["node"], row["pod"], row["container"], row["log"])
                if row_key in seen_row_keys:
                    continue

                seen_row_keys.add(row_key)
                slice_rows.append(row)
                seen_pods.add(labels.get("pod", "unknown"))
                batch_count += 1

        print(f"[INFO]   Batch #{request_idx}: collected {batch_count} log lines")

        if max_ts_ns is None:
            break

        next_cursor_ns = max_ts_ns + 1
        if next_cursor_ns <= cursor_ns:
            print("[WARN] Loki pagination cursor did not advance. Stopping to avoid infinite loop.")
            break
        cursor_ns = next_cursor_ns

    return slice_rows


def fetch_loki_rows(start_dt: datetime, end_dt: datetime) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    current_start = start_dt
    slice_delta = timedelta(minutes=LOKI_SLICE_MINUTES)
    seen_pods: set[str] = set()

    while current_start < end_dt:
        current_end = min(current_start + slice_delta, end_dt)
        slice_rows = fetch_loki_slice_rows(to_unix_ns(current_start), to_unix_ns(current_end), seen_pods)
        rows.extend(slice_rows)

        current_start = current_end

    rows.sort(key=lambda row: (row.get("timestamp", ""), row.get("node") or "", row.get("pod") or "", row.get("container") or ""))
    return rows


def try_parse_json_log(log_text: str) -> Any:
    try:
        return json.loads(log_text)
    except Exception:
        return None


def extract_trace_info_from_log(log_text: str) -> tuple[Optional[str], Optional[str]]:
    trace_match = re.search(r"\btrace[_-]?id\b\s*[:=]\s*\"?([a-f0-9]+)\"?\b", str(log_text), re.IGNORECASE)
    span_match = re.search(r"\bspan[_-]?id\b\s*[:=]\s*\"?([a-f0-9]+)\"?\b", str(log_text), re.IGNORECASE)
    return (
        trace_match.group(1) if trace_match else None,
        span_match.group(1) if span_match else None,
    )


def normalize_log_level(level: Any) -> str:
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


def classify_log_level(log_text: str) -> str:
    level_match = re.search(
        r'(^|[\s,])(?:level|lvl|severity)[:=]"?(trace|debug|info|warn|warning|error|fatal|panic|[tdiwefp])"?',
        str(log_text),
        re.IGNORECASE,
    )
    if level_match:
        return normalize_log_level(level_match.group(2))

    log_upper = str(log_text).upper()
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


def classify_log_source(container: Any) -> str:
    container_lower = str(container).lower()
    if "mongo" in container_lower or "db" in container_lower or "session" in container_lower:
        return "database"
    if "rabbit" in container_lower:
        return "middleware"
    if "istio" in container_lower or "envoy" in container_lower:
        return "infrastructure"
    return "application"


def classify_log_type(message: str) -> str:
    lowered = message.lower()
    if any(token in lowered for token in ["exception", "error", "failed", "panic"]):
        return "exception_log"
    if any(token in lowered for token in ["timeout", "deadline", "slow", "latency"]):
        return "timeout_log"
    if any(token in lowered for token in ["retry", "reconnecting", "backoff"]):
        return "retry_log"
    if any(token in lowered for token in ["connection", "connected", "disconnected"]):
        return "connection_log"
    if any(token in lowered for token in ["queue", "publish", "consume", "message"]):
        return "queue_log"
    return "general_log"


def clean_log_message(log_text: str, parsed_json: Any) -> str:
    if isinstance(parsed_json, dict):
        return parsed_json.get("msg", str(log_text))
    parts = log_text.split(" : ")
    if len(parts) > 1:
        return parts[-1]
    return log_text


def parse_loki_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    parsed_rows: list[dict[str, Any]] = []
    for row in rows:
        raw_log = str(row.get("log", ""))
        parsed_json = try_parse_json_log(raw_log)
        trace_id, span_id = extract_trace_info_from_log(raw_log)
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
        log_source = classify_log_source(row.get("container"))
        message = clean_log_message(raw_log, parsed_json)

        if isinstance(parsed_json, dict):
            json_level = parsed_json.get("level") or parsed_json.get("lvl") or parsed_json.get("severity") or parsed_json.get("s")
            if json_level:
                log_level = normalize_log_level(json_level)
            message = parsed_json.get("msg", message)

        parsed_rows.append(
            {
                "timestamp": row.get("timestamp"),
                "trace_id": trace_id,
                "span_id": span_id,
                "service": row.get("container"),
                "node": row.get("node"),
                "pod": row.get("pod"),
                "container": row.get("container"),
                "log_level": log_level,
                "log_source": log_source,
                "log_type": classify_log_type(message),
                "message": message,
                "raw_log": raw_log,
            }
        )
    parsed_rows.sort(key=lambda row: row.get("timestamp", ""))
    return parsed_rows


def parse_utc_time(text: str) -> datetime:
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def jaeger_get(path: str, params: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    url = f"{JAEGER_URL.rstrip('/')}{path}"
    last_error: Exception | None = None

    for attempt in range(1, max(1, JAEGER_HTTP_RETRIES) + 1):
        try:
            response = requests.get(
                url,
                params=params,
                timeout=JAEGER_HTTP_TIMEOUT_SECONDS,
                headers={"Connection": "close"},
            )
            response.raise_for_status()
            return response.json()
        except HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status is not None and 400 <= status < 500 and status != 429:
                raise RuntimeError(f"Jaeger request rejected with HTTP {status} for {path}") from exc
            last_error = exc
        except (ChunkedEncodingError, ConnectionError, Timeout, RequestException) as exc:
            last_error = exc

        if attempt < max(1, JAEGER_HTTP_RETRIES):
            sleep_seconds = max(0.0, JAEGER_HTTP_BACKOFF_SECONDS) * attempt
            print(f"[Jaeger] retry {attempt}/{JAEGER_HTTP_RETRIES} for {path}: {last_error}")
            time.sleep(sleep_seconds)

    raise RuntimeError(f"Jaeger request failed for {path}: {last_error}")


def list_jaeger_services() -> list[str]:
    data = jaeger_get("/api/services")
    return data.get("data", [])


def query_jaeger_traces(service: Optional[str], start_us: int, end_us: int) -> list[dict[str, Any]]:
    params: dict[str, Any] = {"start": start_us, "end": end_us}
    if service:
        params["service"] = service
    if JAEGER_LIMIT > 0:
        params["limit"] = JAEGER_LIMIT
    data = jaeger_get("/api/traces", params=params)
    return data.get("data", [])


def fetch_jaeger_traces_with_time_splitting(
    service: Optional[str],
    start_us: int,
    end_us: int,
    limit: int,
    min_window_us: int,
    label: str,
    depth: int = 0,
) -> list[dict[str, Any]]:
    traces = query_jaeger_traces(service, start_us, end_us)
    window_us = max(0, end_us - start_us)
    indent = "  " * depth
    print(f"{indent}Window {label}: got {len(traces)} traces")

    if not limit or int(limit) <= 0:
        return traces

    if len(traces) < int(limit):
        return traces

    if window_us <= max(1, min_window_us):
        print(
            f"{indent}[WARN] Window {label} still reached limit={limit} at the minimum split size. "
            "Results may still be truncated."
        )
        return traces

    mid_us = start_us + (window_us // 2)
    if mid_us <= start_us or mid_us >= end_us:
        print(
            f"{indent}[WARN] Cannot split window {label} further even though it reached limit={limit}. "
            "Results may still be truncated."
        )
        return traces

    print(f"{indent}[INFO] Window {label} reached limit={limit}; splitting time range")
    left_traces = fetch_jaeger_traces_with_time_splitting(
        service=service,
        start_us=start_us,
        end_us=mid_us,
        limit=limit,
        min_window_us=min_window_us,
        label=f"{label}.L",
        depth=depth + 1,
    )
    right_traces = fetch_jaeger_traces_with_time_splitting(
        service=service,
        start_us=mid_us + 1,
        end_us=end_us,
        limit=limit,
        min_window_us=min_window_us,
        label=f"{label}.R",
        depth=depth + 1,
    )
    return left_traces + right_traces


def flatten_jaeger_traces_raw(traces: list[dict[str, Any]], fallback_service: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trace in traces:
        trace_id = trace.get("traceID")
        processes = trace.get("processes", {})
        service_by_process = {pid: process.get("serviceName") for pid, process in processes.items()}
        for span in trace.get("spans", []):
            rows.append(
                {
                    "start_time": span.get("startTime"),
                    "trace_id": trace_id,
                    "span_id": span.get("spanID"),
                    "service": service_by_process.get(span.get("processID"), fallback_service),
                    "operation": span.get("operationName"),
                    "duration": span.get("duration"),
                    "references": json.dumps(span.get("references", []), ensure_ascii=False),
                    "tags": json.dumps(span.get("tags", []), ensure_ascii=False),
                    "logs": json.dumps(span.get("logs", []), ensure_ascii=False),
                }
            )
    return rows


def fetch_jaeger_rows(window: dict[str, Any]) -> list[dict[str, Any]]:
    start_dt = parse_utc_time(window["start_iso"])
    end_dt = parse_utc_time(window["end_iso"])
    start_us = to_unix_us(start_dt)
    end_us = to_unix_us(end_dt)
    min_window_us = max(1, JAEGER_SPLIT_MIN_SECONDS) * 1_000_000
    all_services = list_jaeger_services()
    services_of_interest = [service for service in all_services if service not in JAEGER_EXCLUDED_SERVICES]
    seen_span_keys: set[tuple[Any, Any]] = set()
    rows: list[dict[str, Any]] = []

    def append_unique(new_rows: list[dict[str, Any]]) -> None:
        for row in new_rows:
            key = (row.get("trace_id"), row.get("span_id"))
            if key in seen_span_keys:
                continue
            seen_span_keys.add(key)
            rows.append(row)

    for service in services_of_interest:
        try:
            traces = fetch_jaeger_traces_with_time_splitting(
                service=service,
                start_us=start_us,
                end_us=end_us,
                limit=JAEGER_LIMIT,
                min_window_us=min_window_us,
                label=service,
            )
            append_unique(flatten_jaeger_traces_raw(traces, fallback_service=service))
        except Exception as exc:
            print(f"[Jaeger] failed for service {service}: {exc}")

    rows.sort(key=lambda row: (row.get("start_time") or 0, row.get("trace_id") or "", row.get("span_id") or ""))
    return rows


def safe_json_loads(value: Any) -> list[dict[str, Any]]:
    if pd.isna(value) or value == "":
        return []
    try:
        return json.loads(value)
    except Exception:
        return []


def extract_tag(tags: list[dict[str, Any]], key: str, default: Any = "") -> Any:
    for tag in tags:
        if tag.get("key") == key:
            return tag.get("value", default)
    return default


def extract_parent_span_id(references: list[dict[str, Any]]) -> Optional[str]:
    for ref in references:
        if ref.get("refType") == "CHILD_OF":
            return ref.get("spanID")
    return None


def extract_exception(tags: list[dict[str, Any]]) -> tuple[str, str]:
    exc_type = ""
    exc_msg = ""
    for tag in tags:
        if tag.get("key") in ("exception.type", "error.type"):
            exc_type = tag.get("value", "")
        if tag.get("key") in ("exception.message", "error.message"):
            exc_msg = tag.get("value", "")
    return exc_type, exc_msg

def parse_jaeger_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    parsed_rows: list[dict[str, Any]] = []
    for row in rows:
        references = safe_json_loads(row.get("references"))
        tags = safe_json_loads(row.get("tags"))
        exc_type, exc_msg = extract_exception(tags)
        parsed_rows.append(
            {
                "timestamp": int(row["start_time"]) if row.get("start_time") is not None else None,
                "trace_id": row.get("trace_id"),
                "span_id": row.get("span_id"),
                "parent_span_id": extract_parent_span_id(references),
                "service": row.get("service"),
                "operation": row.get("operation"),
                "duration": row.get("duration"),
                "span_kind": extract_tag(tags, "span.kind"),
                "status_code": str(extract_tag(tags, "http.status_code")),
                "status": extract_tag(tags, "otel.status_code", "SUCCESS"),
                "peer_service": extract_tag(tags, "peer.service"),
                "http_method": extract_tag(tags, "http.method"),
                "http_url": extract_tag(tags, "http.url"),
                "exception_type": exc_type,
                "exception_message": exc_msg,
                "pod": extract_tag(tags, "pod"),
                "container": extract_tag(tags, "container"),
                "node": extract_tag(tags, "node"),
                "tags_json": row.get("tags"),
            }
        )
    parsed_rows.sort(key=lambda row: (row.get("timestamp") or 0, row.get("trace_id") or "", row.get("span_id") or ""))
    return parsed_rows


def collect_application_metrics(window: dict[str, Any], output_path: Path, step: str) -> None:
    all_rows: list[dict[str, Any]] = []

    for source in APPLICATION_SOURCE_CONFIGS:
        metrics = load_metric_list(source["metric_file"])
        print(f"Loaded {len(metrics)} {source['name']} metrics")
        selector = source["selector"](PROM_NAMESPACE)

        for metric in metrics:
            print(f"Querying [application/{source['name']}] {metric} ...")
            try:
                results = prom_query_range(f"{metric}{{{selector}}}", window["start_iso"], window["end_iso"], step)
            except Exception as exc:
                print(f"Failed [application/{source['name']}] {metric} -> {exc}")
                continue

            for series in results:
                metric_labels = series.get("metric", {})
                pod = metric_labels.get("pod", "unknown")
                labels_json = dump_selected_labels(metric_labels, source["important_labels"])
                for ts, val in series.get("values", []):
                    value = safe_float(val)
                    if value is None:
                        continue
                    all_rows.append(
                        {
                            "timestamp": timestamp_to_str(float(ts)),
                            "pod": pod,
                            "metric": metric,
                            "value": value,
                            "labels": labels_json,
                        }
                    )

    write_rows_csv(output_path, all_rows, ["timestamp", "pod", "metric", "value", "labels"])


def collect_container_metrics(window: dict[str, Any], output_path: Path, step: str) -> None:
    container_metrics = load_metric_list(CONTAINER_METRIC_FILE)
    kube_pod_metrics = load_metric_list(KUBE_POD_METRIC_FILE)
    all_rows: list[dict[str, Any]] = []
    kube_namespace = os.environ.get("KUBE_POD_NAMESPACE", PROM_NAMESPACE).strip()

    for metric in container_metrics:
        print(f"Querying [container] {metric} ...")
        try:
            results = prom_query_range(metric, window["start_iso"], window["end_iso"], step)
        except Exception as exc:
            print(f"Failed [container] {metric} -> {exc}")
            continue

        for series in results:
            labels = series.get("metric", {})
            pod = labels.get("pod")
            container = labels.get("container")
            if not pod or not container or container not in CONTAINER_TARGETS:
                continue
            for ts, val in series.get("values", []):
                value = safe_float(val)
                if value is None:
                    continue
                all_rows.append(
                    {
                        "timestamp": timestamp_to_str(float(ts)),
                        "pod": pod,
                        "metric": metric,
                        "value": value,
                        "labels": "",
                    }
                )

    for metric in kube_pod_metrics:
        print(f"Querying [kube_pod] {metric} ...")
        expr = f'{metric}{{namespace="{kube_namespace}"}}' if kube_namespace else metric
        try:
            results = prom_query_range(expr, window["start_iso"], window["end_iso"], KUBE_POD_STEP)
        except Exception as exc:
            print(f"Failed [kube_pod] {metric} -> {exc}")
            continue

        for series in results:
            labels = series.get("metric", {})
            pod = labels.get("pod")
            if not pod:
                continue
            labels_json = json.dumps(
                {key: value for key, value in labels.items() if key not in {"__name__", "pod"}},
                sort_keys=True,
                ensure_ascii=False,
            )
            for ts, val in series.get("values", []):
                value = safe_float(val)
                if value is None:
                    continue
                all_rows.append(
                    {
                        "timestamp": timestamp_to_str(float(ts)),
                        "pod": pod,
                        "metric": metric,
                        "value": value,
                        "labels": labels_json,
                    }
                )

    write_rows_csv(output_path, all_rows, ["timestamp", "pod", "metric", "value", "labels"])


def kpi_promql_candidates(kpi: str, pod: str) -> list[str]:
    cpu_sel = f'namespace="{PROM_NAMESPACE}",pod="{pod}",container!="POD",container!="istio-proxy"'
    cpu_limit_sel = (
        f'namespace="{PROM_NAMESPACE}",pod="{pod}",resource="cpu",unit="core",'
        'container!="POD",container!="istio-proxy"'
    )
    mem_limit_sel = (
        f'namespace="{PROM_NAMESPACE}",pod="{pod}",resource="memory",unit="byte",'
        'container!="POD",container!="istio-proxy"'
    )
    cpu_limit_sel_no_unit = (
        f'namespace="{PROM_NAMESPACE}",pod="{pod}",resource="cpu",'
        'container!="POD",container!="istio-proxy"'
    )
    mem_limit_sel_no_unit = (
        f'namespace="{PROM_NAMESPACE}",pod="{pod}",resource="memory",'
        'container!="POD",container!="istio-proxy"'
    )
    cpu_req_sel = (
        f'namespace="{PROM_NAMESPACE}",pod="{pod}",resource="cpu",unit="core",'
        'container!="POD",container!="istio-proxy"'
    )
    mem_req_sel = (
        f'namespace="{PROM_NAMESPACE}",pod="{pod}",resource="memory",unit="byte",'
        'container!="POD",container!="istio-proxy"'
    )
    cpu_req_sel_no_unit = (
        f'namespace="{PROM_NAMESPACE}",pod="{pod}",resource="cpu",'
        'container!="POD",container!="istio-proxy"'
    )
    mem_req_sel_no_unit = (
        f'namespace="{PROM_NAMESPACE}",pod="{pod}",resource="memory",'
        'container!="POD",container!="istio-proxy"'
    )
    pod_info_sel = f'namespace="{PROM_NAMESPACE}",pod="{pod}"'

    istio_base = f'reporter="destination",destination_workload_namespace="{PROM_NAMESPACE}"'
    istio_by_dest_pod = f'{istio_base},destination_pod="{pod}"'
    istio_by_pod = f'{istio_base},pod="{pod}"'
    workload_guess = pod.split("-")[0] if "-" in pod else pod
    istio_by_workload = f'{istio_base},destination_workload="{workload_guess}"'

    if kpi == "request_rate":
        return [
            f'sum by (pod) (rate(istio_requests_total{{{istio_by_dest_pod}}}[{ISTIO_WINDOW}]))',
            f'sum by (pod) (rate(istio_requests_total{{{istio_by_pod}}}[{ISTIO_WINDOW}]))',
            f'sum(rate(istio_requests_total{{{istio_by_workload}}}[{ISTIO_WINDOW}]))',
        ]

    if kpi == "success_rate":
        success_re = 'response_code=~"200|201|202|204|300|301|302|304"'
        return [
            f'(sum(rate(istio_requests_total{{{istio_by_dest_pod},{success_re}}}[{ISTIO_WINDOW}])) / sum(rate(istio_requests_total{{{istio_by_dest_pod}}}[{ISTIO_WINDOW}]))) * 100',
            f'(sum(rate(istio_requests_total{{{istio_by_pod},{success_re}}}[{ISTIO_WINDOW}])) / sum(rate(istio_requests_total{{{istio_by_pod}}}[{ISTIO_WINDOW}]))) * 100',
            f'(sum(rate(istio_requests_total{{{istio_by_workload},{success_re}}}[{ISTIO_WINDOW}])) / sum(rate(istio_requests_total{{{istio_by_workload}}}[{ISTIO_WINDOW}]))) * 100',
        ]

    if kpi == "error_count":
        err_re = 'response_code=~"5.."'
        err_by_dest_pod = f'sum(increase(istio_requests_total{{{istio_by_dest_pod},{err_re}}}[{KPI_WINDOW}]))'
        all_by_dest_pod = f'sum(increase(istio_requests_total{{{istio_by_dest_pod}}}[{KPI_WINDOW}]))'
        err_by_pod = f'sum(increase(istio_requests_total{{{istio_by_pod},{err_re}}}[{KPI_WINDOW}]))'
        all_by_pod = f'sum(increase(istio_requests_total{{{istio_by_pod}}}[{KPI_WINDOW}]))'
        err_by_workload = f'sum(increase(istio_requests_total{{{istio_by_workload},{err_re}}}[{KPI_WINDOW}]))'
        all_by_workload = f'sum(increase(istio_requests_total{{{istio_by_workload}}}[{KPI_WINDOW}]))'
        return [
            f'({err_by_dest_pod}) or (0 * ({all_by_dest_pod}))',
            f'({err_by_pod}) or (0 * ({all_by_pod}))',
            f'({err_by_workload}) or (0 * ({all_by_workload}))',
        ]

    if kpi.startswith("latency_p"):
        quantile = float(kpi.replace("latency_p", "")) / 100.0

        def hist(selector: str) -> str:
            return (
                f'histogram_quantile({quantile}, '
                f'sum by (le) (rate(istio_request_duration_milliseconds_bucket{{{selector}}}[{ISTIO_WINDOW}])))'
            )

        return [hist(istio_by_dest_pod), hist(istio_by_pod), hist(istio_by_workload)]

    if kpi == "cpu_usage_pct":
        usage = f'sum by (pod) (rate(container_cpu_usage_seconds_total{{{cpu_sel}}}[{KPI_WINDOW}]))'
        limit = f'sum by (pod) (kube_pod_container_resource_limits{{{cpu_limit_sel}}})'
        limit_no_unit = f'sum by (pod) (kube_pod_container_resource_limits{{{cpu_limit_sel_no_unit}}})'
        req = f'sum by (pod) (kube_pod_container_resource_requests{{{cpu_req_sel}}})'
        req_no_unit = f'sum by (pod) (kube_pod_container_resource_requests{{{cpu_req_sel_no_unit}}})'
        pod_node = f'max by (pod, node) (kube_pod_info{{{pod_info_sel}}})'
        node_alloc = f'max by (node) (kube_node_status_allocatable{{resource="cpu",unit="core"}})'
        node_alloc_no_unit = f'max by (node) (kube_node_status_allocatable{{resource="cpu"}})'
        node_based = f'sum by (pod) (({pod_node}) * on (node) group_left {node_alloc})'
        node_based_no_unit = f'sum by (pod) (({pod_node}) * on (node) group_left {node_alloc_no_unit})'
        return [
            f'({usage} / clamp_min({limit}, 0.001)) * 100',
            f'({usage} / clamp_min({limit_no_unit}, 0.001)) * 100',
            f'({usage} / clamp_min({req}, 0.001)) * 100',
            f'({usage} / clamp_min({req_no_unit}, 0.001)) * 100',
            f'({usage} / clamp_min({node_based}, 0.001)) * 100',
            f'({usage} / clamp_min({node_based_no_unit}, 0.001)) * 100',
        ]

    if kpi == "memory_usage_pct":
        usage = f'sum by (pod) (container_memory_working_set_bytes{{{cpu_sel}}})'
        limit = f'sum by (pod) (kube_pod_container_resource_limits{{{mem_limit_sel}}})'
        limit_no_unit = f'sum by (pod) (kube_pod_container_resource_limits{{{mem_limit_sel_no_unit}}})'
        req = f'sum by (pod) (kube_pod_container_resource_requests{{{mem_req_sel}}})'
        req_no_unit = f'sum by (pod) (kube_pod_container_resource_requests{{{mem_req_sel_no_unit}}})'
        pod_node = f'max by (pod, node) (kube_pod_info{{{pod_info_sel}}})'
        node_alloc = f'max by (node) (kube_node_status_allocatable{{resource="memory",unit="byte"}})'
        node_alloc_no_unit = f'max by (node) (kube_node_status_allocatable{{resource="memory"}})'
        node_based = f'sum by (pod) (({pod_node}) * on (node) group_left {node_alloc})'
        node_based_no_unit = f'sum by (pod) (({pod_node}) * on (node) group_left {node_alloc_no_unit})'
        return [
            f'({usage} / clamp_min({limit}, 1)) * 100',
            f'({usage} / clamp_min({limit_no_unit}, 1)) * 100',
            f'({usage} / clamp_min({req}, 1)) * 100',
            f'({usage} / clamp_min({req_no_unit}, 1)) * 100',
            f'({usage} / clamp_min({node_based}, 1)) * 100',
            f'({usage} / clamp_min({node_based_no_unit}, 1)) * 100',
        ]

    if kpi in {"restart_count", "restart_rate"}:
        return [
            f'sum(increase(kube_pod_container_status_restarts_total{{namespace="{PROM_NAMESPACE}",pod="{pod}"}}[{RESTART_COUNT_WINDOW}]))'
        ]

    if kpi == "ready_ratio":
        return [
            f'avg_over_time(kube_pod_status_ready{{namespace="{PROM_NAMESPACE}",pod="{pod}",condition="true"}}[{KPI_WINDOW}])'
        ]

    if kpi == "network_rx":
        return [
            f'sum by (pod) (rate(container_network_receive_bytes_total{{namespace="{PROM_NAMESPACE}",pod="{pod}"}}[{NETWORK_KPI_WINDOW}]))'
        ]

    if kpi == "network_tx":
        return [
            f'sum by (pod) (rate(container_network_transmit_bytes_total{{namespace="{PROM_NAMESPACE}",pod="{pod}"}}[{NETWORK_KPI_WINDOW}]))'
        ]

    raise ValueError(f"Unknown KPI: {kpi}")


def list_kpi_pods(end_iso: str) -> list[str]:
    expr = f'kube_pod_info{{namespace="{PROM_NAMESPACE}"}}'
    series = prom_query(expr, ts=end_iso)
    pods: list[str] = []
    for item in series:
        pod = item.get("metric", {}).get("pod")
        if not pod:
            continue
        if pod.startswith("prometheus") or pod.startswith("grafana"):
            continue
        pods.append(pod)

    deduped: list[str] = []
    seen: set[str] = set()
    for pod in pods:
        if pod not in seen:
            seen.add(pod)
            deduped.append(pod)
    return deduped


def collect_kpi_metrics(window: dict[str, Any], output_path: Path, step: str) -> None:
    all_rows: list[dict[str, Any]] = []
    pods = list_kpi_pods(window["end_iso"])
    print(f"Discovered pods for KPI: {len(pods)}")

    for pod in pods:
        for kpi in KPI_NAMES:
            candidates = kpi_promql_candidates(kpi, pod)
            last_err: Optional[str] = None
            found_rows: list[dict[str, Any]] = []

            for index, expr in enumerate(candidates, start=1):
                try:
                    results = prom_query_range(expr, window["start_iso"], window["end_iso"], step)
                except Exception as exc:
                    last_err = str(exc)
                    continue

                if not results:
                    continue

                for series in results:
                    labels = series.get("metric", {})
                    row_pod = labels.get("pod") or labels.get("destination_pod") or pod
                    for ts, val in series.get("values", []):
                        value = safe_float(val)
                        if value is None:
                            continue
                        found_rows.append(
                            {
                                "timestamp": timestamp_to_str(float(ts)),
                                "pod": row_pod,
                                "metric": kpi,
                                "value": value,
                            }
                        )

                if found_rows:
                    print(f"[OK] {kpi} pod={pod} (candidate {index}/{len(candidates)})")
                    break

            if found_rows:
                all_rows.extend(found_rows)
            elif last_err:
                print(f"[WARN] {kpi} pod={pod} failed: {last_err}")
            else:
                print(f"[WARN] {kpi} pod={pod} returned empty (all candidates)")

    write_rows_csv(output_path, all_rows, ["timestamp", "pod", "metric", "value"])


def collect_middleware_metrics(window: dict[str, Any], output_path: Path, step: str) -> None:
    all_rows: list[dict[str, Any]] = []

    for source in MIDDLEWARE_SOURCE_CONFIGS:
        namespace = source["namespace"]()
        pods = source["pods"]()
        if not pods:
            print(f"Skip {source['name']}: no target pods configured")
            continue

        metrics = load_metric_list(source["metric_file"])
        print(f"Loaded {len(metrics)} {source['name']} metrics")

        for metric in metrics:
            print(f"Querying [{source['name']}] {metric} ...")
            for pod in pods:
                filters = []
                if namespace:
                    filters.append(f'namespace="{namespace}"')
                if pod:
                    filters.append(f'pod="{pod}"')
                expr = f'{metric}{{{",".join(filters)}}}' if filters else metric

                try:
                    results = prom_query_range(expr, window["start_iso"], window["end_iso"], step)
                except Exception as exc:
                    print(f"Failed [{source['name']}] {metric} (pod={pod}) -> {exc}")
                    continue

                for series in results:
                    row_pod = series.get("metric", {}).get("pod", pod or "unknown")
                    if pod and row_pod != pod:
                        continue
                    for ts, val in series.get("values", []):
                        value = safe_float(val)
                        if value is None:
                            continue
                        all_rows.append(
                            {
                                "timestamp": timestamp_to_str(float(ts)),
                                "pod": row_pod,
                                "metric": metric,
                                "value": value,
                            }
                        )

    write_rows_csv(output_path, all_rows, ["timestamp", "pod", "metric", "value"])


def collect_network_metrics(window: dict[str, Any], output_path: Path, step: str) -> None:
    metrics = load_metric_list(NGINX_METRIC_FILE)
    namespace = os.environ.get("NGINX_NAMESPACE", PROM_NAMESPACE)
    all_rows: list[dict[str, Any]] = []

    for metric in metrics:
        print(f"Querying [network] {metric} ...")
        expr = f'{metric}{{exported_namespace="{namespace}"}}'
        try:
            results = prom_query_range(expr, window["start_iso"], window["end_iso"], step)
        except Exception as exc:
            print(f"Failed [network] {metric} -> {exc}")
            continue

        for series in results:
            labels = series.get("metric", {})
            service = labels.get("exported_service")
            if not service:
                continue
            labels_json = dump_selected_labels(labels, NETWORK_LABELS)
            for ts, val in series.get("values", []):
                value = safe_float(val)
                if value is None:
                    continue
                all_rows.append(
                    {
                        "timestamp": timestamp_to_str(float(ts)),
                        "service": service,
                        "metric": metric,
                        "value": value,
                        "labels": labels_json,
                    }
                )

    write_rows_csv(output_path, all_rows, ["timestamp", "service", "metric", "value", "labels"])


def collect_node_metrics(window: dict[str, Any], output_path: Path, step: str) -> None:
    metrics = load_metric_list(NODE_METRIC_FILE)
    all_rows: list[dict[str, Any]] = []

    for metric in metrics:
        print(f"Querying [node] {metric} ...")
        try:
            results = prom_query_range(metric, window["start_iso"], window["end_iso"], step)
        except Exception as exc:
            print(f"Failed [node] {metric} -> {exc}")
            continue

        for series in results:
            instance = series.get("metric", {}).get("instance", "unknown")
            for ts, val in series.get("values", []):
                value = safe_float(val)
                if value is None:
                    continue
                all_rows.append(
                    {
                        "timestamp": timestamp_to_str(float(ts)),
                        "instance": instance,
                        "metric": metric,
                        "value": value,
                    }
                )

    write_rows_csv(output_path, all_rows, ["timestamp", "instance", "metric", "value"])


def collect_service_proxy_metrics(window: dict[str, Any], output_path: Path, step: str) -> None:
    metrics = load_metric_list(ISTIO_METRIC_FILE)
    namespace = os.environ.get("ISTIO_NAMESPACE", PROM_NAMESPACE)
    all_rows: list[dict[str, Any]] = []

    for metric in metrics:
        print(f"Querying [service_proxy] {metric} ...")
        expr = f'{metric}{{destination_workload_namespace="{namespace}"}}'
        try:
            results = prom_query_range(expr, window["start_iso"], window["end_iso"], step)
        except Exception as exc:
            print(f"Failed [service_proxy] {metric} -> {exc}")
            continue

        for series in results:
            labels = series.get("metric", {})
            pod = (
                labels.get("pod")
                or labels.get("destination_pod")
                or labels.get("source_pod")
                or labels.get("destination_workload")
            )
            if not pod:
                continue
            labels_json = dump_selected_labels(labels, SERVICE_PROXY_LABELS)
            for ts, val in series.get("values", []):
                value = safe_float(val)
                if value is None:
                    continue
                all_rows.append(
                    {
                        "timestamp": timestamp_to_str(float(ts)),
                        "pod": pod,
                        "metric": metric,
                        "value": value,
                        "labels": labels_json,
                    }
                )

    write_rows_csv(output_path, all_rows, ["timestamp", "pod", "metric", "value", "labels"])


METRIC_HANDLER_MAP: dict[str, Callable[[dict[str, Any], Path, str], None]] = {
    "application": collect_application_metrics,
    "container": collect_container_metrics,
    "KPI": collect_kpi_metrics,
    "middleware": collect_middleware_metrics,
    "network": collect_network_metrics,
    "node": collect_node_metrics,
    "service_proxy": collect_service_proxy_metrics,
}


def collect_logs(window: dict[str, Any], logs_dir: Path) -> list[dict[str, str]]:
    raw_name = f"loki_logs_raw_{window['suffix']}.csv"
    parsed_name = f"loki_logs_parsed_{window['suffix']}.csv"
    raw_path = logs_dir / raw_name
    parsed_path = logs_dir / parsed_name

    rows = fetch_loki_rows(window["start_dt"], window["end_dt"])
    if rows:
        write_rows_csv(raw_path, rows, LOG_RAW_COLUMNS)
    else:
        write_empty_csv(raw_path, LOG_RAW_COLUMNS)

    parsed_rows = parse_loki_rows(rows)
    if parsed_rows:
        write_rows_csv(parsed_path, parsed_rows, LOG_PARSED_COLUMNS)
    else:
        write_empty_csv(parsed_path, LOG_PARSED_COLUMNS)

    return [
        {"type": "logs_raw", "path": str(raw_path)},
        {"type": "logs_parsed", "path": str(parsed_path)},
    ]


def collect_traces(window: dict[str, Any], traces_dir: Path) -> list[dict[str, str]]:
    raw_name = f"jaeger_traces_raw_{window['suffix']}.csv"
    parsed_name = f"jaeger_traces_parsed_{window['suffix']}.csv"
    raw_path = traces_dir / raw_name
    parsed_path = traces_dir / parsed_name

    rows = fetch_jaeger_rows(window)
    if rows:
        write_rows_csv(raw_path, rows, TRACE_RAW_COLUMNS)
    else:
        write_empty_csv(raw_path, TRACE_RAW_COLUMNS)

    parsed_rows = parse_jaeger_rows(rows)
    if parsed_rows:
        write_rows_csv(parsed_path, parsed_rows, TRACE_PARSED_COLUMNS)
    else:
        write_empty_csv(parsed_path, TRACE_PARSED_COLUMNS)

    return [
        {"type": "traces_raw", "path": str(raw_path)},
        {"type": "traces_parsed", "path": str(parsed_path)},
    ]


def collect_metrics(window: dict[str, Any], metrics_dir: Path) -> list[dict[str, str]]:
    outputs: list[dict[str, str]] = []

    for collector in METRIC_COLLECTORS:
        output_name = f"{collector['output_name']}_{window['suffix']}.csv"
        output_path = metrics_dir / output_name
        handler = METRIC_HANDLER_MAP[collector["handler"]]
        handler(window, output_path, collector.get("step", PROM_STEP))
        outputs.append({"type": collector["name"], "path": str(output_path)})

    return outputs


def write_summary(summary_path: Path, summary: dict[str, Any]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    windows = build_hour_windows(DATE_TEXT, HOURS)
    output_dirs = ensure_output_dirs(DATE_TEXT)

    summary: dict[str, Any] = {
        "date": DATE_TEXT,
        "timezone": str(TIMEZONE),
        "python_bin": PYTHON_BIN,
        "output_root": str(output_dirs["date"]),
        "config": {
            "hours": normalize_hours(HOURS),
            "run_metrics": RUN_METRICS,
            "run_logs": RUN_LOGS,
            "run_traces": RUN_TRACES,
            "prom_url": PROM_URL,
            "prom_namespace": PROM_NAMESPACE,
            "loki_url": LOKI_URL,
            "loki_query": LOKI_QUERY,
            "jaeger_url": JAEGER_URL,
            "prom_step": PROM_STEP,
            "kube_pod_step": KUBE_POD_STEP,
            "kpi_window": KPI_WINDOW,
            "restart_count_window": RESTART_COUNT_WINDOW,
            "istio_window": ISTIO_WINDOW,
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "windows": [],
    }

    print(f"Collecting telemetry for {DATE_TEXT}")
    print(f"Hours: {summary['config']['hours']}")
    print(f"Output root: {output_dirs['date']}")

    for window in windows:
        print("\n" + "=" * 72)
        print(f"Hour {window['suffix']}: {window['start_iso']} -> {window['end_iso']}")

        window_result: dict[str, Any] = {
            "hour": window["hour"],
            "suffix": window["suffix"],
            "start_iso": window["start_iso"],
            "end_iso": window["end_iso"],
            "status": "success",
            "outputs": [],
            "errors": [],
        }

        if RUN_METRICS:
            try:
                print("[metrics] collecting...")
                window_result["outputs"].extend(collect_metrics(window, output_dirs["metrics"]))
            except Exception as exc:
                window_result["status"] = "partial_failed"
                window_result["errors"].append({"stage": "metrics", "message": str(exc)})
                print(f"[metrics] failed: {exc}")

        if RUN_LOGS:
            try:
                print("[logs] collecting and parsing...")
                window_result["outputs"].extend(collect_logs(window, output_dirs["logs"]))
            except Exception as exc:
                window_result["status"] = "partial_failed"
                window_result["errors"].append({"stage": "logs", "message": str(exc)})
                print(f"[logs] failed: {exc}")

        if RUN_TRACES:
            try:
                print("[traces] collecting and parsing...")
                window_result["outputs"].extend(collect_traces(window, output_dirs["traces"]))
            except Exception as exc:
                window_result["status"] = "partial_failed"
                window_result["errors"].append({"stage": "traces", "message": str(exc)})
                print(f"[traces] failed: {exc}")

        summary["windows"].append(window_result)
        write_summary(output_dirs["date"] / SUMMARY_FILE_NAME, summary)

    success_count = sum(1 for window in summary["windows"] if window["status"] == "success")
    partial_count = len(summary["windows"]) - success_count
    print("\nCollection finished")
    print(f"Successful windows: {success_count}")
    print(f"Windows with errors: {partial_count}")
    print(f"Summary: {output_dirs['date'] / SUMMARY_FILE_NAME}")


if __name__ == "__main__":
    main()
