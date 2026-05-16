# Output Schema

The final result is JSON-serializable and includes:

- `incident_id`
- `abnormal_window`
- `baseline_window`
- `service_top5`
- `pod_top5`
- `final_summary`
- `warnings`
- `errors`
- `metadata`

Each service item contains:

- `service`
- `fault_type`
- `score`
- `supporting_evidence`
- `notes`

Each pod item contains:

- `pod`
- `service`
- `fault_type`
- `score`
- `supporting_evidence`
- `notes`

Reasoning metadata may also include:

- `metadata.llm`
- `metadata.reasoning_rules`
- `metadata.service_candidates`
- `metadata.pod_candidates`

Candidate previews in metadata may contain:

- `fault_type_candidates`
- `rule_hints`
- `active_rules`
