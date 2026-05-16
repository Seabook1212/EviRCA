"""
README
- Purpose: Orchestrate full RCA flow:
  load data -> anomaly scoring -> graph ranking -> top-k LLM verification -> final probabilities.
- Final score:
  FinalScore = 0.7 * GraphScore + 0.3 * ConsistencyScore
- Outputs list:
  [{"service": "...", "probability": ...}, ...], normalized to sum to 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from data_loader import DataLoader, RCADataBundle
from graph_ranker import GraphRanker
from llm_verifier import LLMVerifier
from logs_agent import LogsAgent
from metrics_agent import MetricsAgent
from trace_agent import TraceAgent


@dataclass
class PipelineResult:
    ranking: List[Dict[str, float]]
    raw_graph_scores: Dict[str, float]
    raw_final_scores: Dict[str, float]
    final_score_breakdown: Dict[str, Dict[str, float]]
    modality_scores: Dict[str, Any]
    llm_results: Dict[str, Dict[str, Any]]
    candidate_payloads: List[Dict[str, Any]]
    endpoints: Dict[str, str]


class RCAPipeline:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.top_k = int(config.get("top_k", 5))
        self.graph_weight = float(config.get("graph_weight", 0.7))
        self.llm_weight = float(config.get("llm_weight", 0.3))
        self.verbose = bool(config.get("verbose", True))

        self.loader = DataLoader(config)
        self.metrics_agent = MetricsAgent(
            epsilon=float(config.get("epsilon", 1e-6)),
            allowed_kpis=config.get("metrics_kpis"),
            latency_kpi=config.get("latency_kpi"),
        )
        self.logs_agent = LogsAgent(epsilon=float(config.get("epsilon", 1e-6)))
        self.trace_agent = TraceAgent(
            epsilon=float(config.get("epsilon", 1e-6)),
            lambda_fail=float(config.get("trace_failure_weight", 0.05)),
        )
        self.graph_ranker = GraphRanker(
            alpha=float(config.get("alpha", 0.5)),
            beta=float(config.get("beta", 0.25)),
            gamma=float(config.get("gamma", 0.25)),
            pagerank_alpha=float(config.get("pagerank_alpha", 0.85)),
            root_score_blend=float(config.get("root_score_blend", 0.0)),
        )
        self.llm_verifier = LLMVerifier(
            model=str(config.get("llm_model", config.get("openai_model", "gpt-4o-mini"))),
            provider=str(config.get("llm_provider", "openai")),
            max_tool_calls=int(config.get("llm_max_iterations", 30)),
            timeout_seconds=int(config.get("llm_timeout_seconds", 60)),
            api_key_env=str(
                config.get("llm_api_key_env", config.get("openai_api_key_env", "OPENAI_API_KEY"))
            ),
            base_url=config.get("llm_base_url"),
            enabled=bool(config.get("use_llm", False)),
            verbose=self.verbose,
        )

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    @staticmethod
    def _normalize_probabilities(score_map: Dict[str, float]) -> Dict[str, float]:
        if not score_map:
            return {}
        values = np.array([max(0.0, float(v)) for v in score_map.values()], dtype=float)
        total = float(values.sum())
        if total <= 0.0:
            values = np.ones_like(values) / len(values)
        else:
            values = values / total
        return {svc: float(p) for svc, p in zip(score_map.keys(), values)}

    @staticmethod
    def _build_anomaly_summary(
        service: str,
        metric_summary: Dict[str, Dict[str, float]],
        log_summary: Dict[str, Dict[str, float]],
        trace_summary: Dict[str, Dict[str, float]],
    ) -> Dict[str, Any]:
        return {
            "metric_summary": metric_summary.get(service, {}),
            "log_summary": log_summary.get(service, {}),
            "trace_summary": trace_summary.get(service, {}),
        }

    def _build_candidate_payloads(
        self,
        bundle: RCADataBundle,
        top_candidates: List[Tuple[str, float]],
        metric_scores: Dict[str, float],
        log_scores: Dict[str, float],
        trace_scores: Dict[str, float],
        metric_summary: Dict[str, Dict[str, float]],
        log_summary: Dict[str, Dict[str, float]],
        trace_summary: Dict[str, Dict[str, float]],
    ) -> List[Dict[str, Any]]:
        payloads: List[Dict[str, Any]] = []
        for service, graph_score in top_candidates:
            downstream = (
                sorted(list(bundle.topology_graph.successors(service)))
                if service in bundle.topology_graph
                else []
            )
            upstream = (
                sorted(list(bundle.topology_graph.predecessors(service)))
                if service in bundle.topology_graph
                else []
            )
            payloads.append(
                {
                    "service": service,
                    "metric_score": float(metric_scores.get(service, 0.0)),
                    "log_score": float(log_scores.get(service, 0.0)),
                    "trace_score": float(trace_scores.get(service, 0.0)),
                    "graph_score": float(graph_score),
                    "downstream_services": downstream,
                    "upstream_services": upstream,
                    "topology_context": {
                        "downstream_services": downstream,
                        "upstream_services": upstream,
                        "in_degree": len(upstream),
                        "out_degree": len(downstream),
                    },
                    "anomaly_summary": self._build_anomaly_summary(service, metric_summary, log_summary, trace_summary),
                    "abnormal_start_timestamp": self.config.get("abnormal_start_timestamp"),
                    "abnormal_end_timestamp": self.config.get("abnormal_end_timestamp"),
                    "time_window": {
                        "abnormal_start_timestamp": self.config.get("abnormal_start_timestamp"),
                        "abnormal_end_timestamp": self.config.get("abnormal_end_timestamp"),
                    },
                    "constraints": {
                        "max_iterations": int(self.config.get("llm_max_iterations", 10)),
                        "discovery_allowed": False,
                        "use_precomputed_stats_only": True,
                    },
                }
            )
        return payloads

    def run(self) -> PipelineResult:
        self._log("[Pipeline] Step 0: Loading data ...")
        bundle = self.loader.load_all()
        self._log(
            "[Pipeline] Loaded rows: "
            f"metrics(normal={len(bundle.normal_metrics)}, abnormal={len(bundle.abnormal_metrics)}), "
            f"logs(normal={len(bundle.normal_logs)}, abnormal={len(bundle.abnormal_logs)}), "
            f"traces(normal={len(bundle.normal_traces)}, abnormal={len(bundle.abnormal_traces)})"
        )

        self._log("[Pipeline] Step 1: Computing anomaly scores ...")
        metric_scores, metric_summary = self.metrics_agent.compute_scores(bundle.normal_metrics, bundle.abnormal_metrics)
        log_scores, log_summary = self.logs_agent.compute_scores(bundle.normal_logs, bundle.abnormal_logs)
        trace_scores, trace_summary = self.trace_agent.compute_scores(bundle.normal_traces, bundle.abnormal_traces)
        self._log(
            f"[Pipeline] Step 1 done. services(metric={len(metric_scores)}, log={len(log_scores)}, trace={len(trace_scores)})"
        )

        self._log("[Pipeline] Step 2: Graph ranking with Reverse PageRank ...")
        graph_scores, top_candidates, _ = self.graph_ranker.rank(
            bundle.topology_graph,
            metric_scores,
            log_scores,
            trace_scores,
            top_k=self.top_k,
        )
        self._log(f"[Pipeline] Step 2 done. Top-{self.top_k} candidates: {top_candidates}")

        candidate_payloads = self._build_candidate_payloads(
            bundle,
            top_candidates,
            metric_scores,
            log_scores,
            trace_scores,
            metric_summary,
            log_summary,
            trace_summary,
        )
        self._log("[Pipeline] Step 3: LLM verification for Top-K candidates ...")
        llm_results = self.llm_verifier.verify_top_k(candidate_payloads)
        self._log("[Pipeline] Step 3 done.")

        final_scores: Dict[str, float] = {}
        final_score_breakdown: Dict[str, Dict[str, float]] = {}
        for service, graph_score in top_candidates:
            consistency = float(llm_results.get(service, {}).get("consistency_score", 0.5))
            final_scores[service] = self.graph_weight * float(graph_score) + self.llm_weight * consistency
            final_score_breakdown[service] = {
                "graph_score": float(graph_score),
                "consistency_score": consistency,
                "final_score": float(final_scores[service]),
            }
            self._log(
                "[Pipeline] FinalScore "
                f"{service}: {self.graph_weight}*{float(graph_score):.6f} + "
                f"{self.llm_weight}*{consistency:.6f} = {final_scores[service]:.6f}"
            )

        final_probs = self._normalize_probabilities(final_scores)
        ranking = [
            {"service": service, "probability": float(prob)}
            for service, prob in sorted(final_probs.items(), key=lambda x: x[1], reverse=True)
        ]
        self._log("[Pipeline] Step 4 done. Final ranking ready.")

        return PipelineResult(
            ranking=ranking,
            raw_graph_scores=graph_scores,
            raw_final_scores=final_scores,
            final_score_breakdown=final_score_breakdown,
            modality_scores={
                "metric_score": metric_scores,
                "metric_kpis_used": self.metrics_agent.last_used_kpis,
                "log_score": log_scores,
                "trace_score": trace_scores,
            },
            llm_results=llm_results,
            candidate_payloads=candidate_payloads,
            endpoints=bundle.endpoints,
        )
