# Data Interface

## Backend Modes

### CSV mode

- metrics CSV
- logs CSV
- traces CSV
- topology JSON

### API mode

- Prometheus
- Loki
- Jaeger

## Expected CSV Fields

### Metrics

- `timestamp`
- `pod`
- `metric`
- `value`
- optional `service`

### Logs

- `timestamp`
- `service`
- `node`
- `pod`
- `container`
- `log_level`
- `log_source`
- `log_type`
- `message`
- `raw_log`

### Traces

- `timestamp`
- `trace_id`
- `span_id`
- `parent_span_id`
- `service`
- `operation`
- `duration`
- `span_kind`
- `status_code`
- `status`
- `peer_service`
- `pod`
- `container`
- `node`

