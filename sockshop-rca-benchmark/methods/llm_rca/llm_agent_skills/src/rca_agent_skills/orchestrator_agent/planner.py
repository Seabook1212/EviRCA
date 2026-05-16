from __future__ import annotations


class FixedSOPPlanner:
    def get_skill_order(self) -> list[str]:
        return [
            "metric_evidence",
            "log_evidence",
            "trace_evidence",
            "rootcause_reasoning",
        ]

