"""
README
- Purpose: Fuse modality scores and propagate via reverse PageRank.
- Formula:
  RootScore_i = alpha*Metric_i + beta*Log_i + gamma*Trace_i
- Propagation:
  - Reverse dependency edges
  - Run personalized PageRank using RootScore as personalization vector
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import networkx as nx
import numpy as np


class GraphRanker:
    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.25,
        gamma: float = 0.25,
        pagerank_alpha: float = 0.85,
        root_score_blend: float = 0.0,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.pagerank_alpha = pagerank_alpha
        self.root_score_blend = min(1.0, max(0.0, float(root_score_blend)))

    def _fuse_scores(
        self,
        graph: nx.DiGraph,
        metric_scores: Dict[str, float],
        log_scores: Dict[str, float],
        trace_scores: Dict[str, float],
    ) -> Dict[str, float]:
        fused: Dict[str, float] = {}
        for node in graph.nodes():
            fused[node] = (
                self.alpha * float(metric_scores.get(node, 0.0))
                + self.beta * float(log_scores.get(node, 0.0))
                + self.gamma * float(trace_scores.get(node, 0.0))
            )
        return fused

    @staticmethod
    def _normalize_vector(vec: Dict[str, float]) -> Dict[str, float]:
        total = float(sum(max(v, 0.0) for v in vec.values()))
        if total <= 0.0:
            n = len(vec) if vec else 1
            return {k: 1.0 / n for k in vec}
        return {k: max(v, 0.0) / total for k, v in vec.items()}

    @staticmethod
    def _to_probabilities(scores: Dict[str, float]) -> Dict[str, float]:
        if not scores:
            return {}
        values = np.array(list(scores.values()), dtype=float)
        values = np.clip(values, a_min=0.0, a_max=None)
        total = float(values.sum())
        if total <= 0.0:
            prob = np.ones_like(values) / len(values)
        else:
            prob = values / total
        return {svc: float(p) for svc, p in zip(scores.keys(), prob)}

    def rank(
        self,
        graph: nx.DiGraph,
        metric_scores: Dict[str, float],
        log_scores: Dict[str, float],
        trace_scores: Dict[str, float],
        top_k: int = 5,
    ) -> Tuple[Dict[str, float], List[Tuple[str, float]], Dict[str, float]]:
        if graph.number_of_nodes() == 0:
            return {}, [], {}

        root_scores = self._fuse_scores(graph, metric_scores, log_scores, trace_scores)
        personalization = self._normalize_vector(root_scores)
        reverse_graph = graph.reverse(copy=True)

        try:
            propagated = nx.pagerank(
                reverse_graph,
                alpha=self.pagerank_alpha,
                personalization=personalization,
                max_iter=200,
                tol=1e-8,
            )
        except Exception:
            propagated = personalization

        # Blend propagated score with original root personalization to reduce
        # over-amplification of highly connected entry services (e.g., front-end).
        if self.root_score_blend > 0.0:
            final_graph_scores = {
                node: (1.0 - self.root_score_blend) * float(propagated.get(node, 0.0))
                + self.root_score_blend * float(personalization.get(node, 0.0))
                for node in graph.nodes()
            }
        else:
            final_graph_scores = propagated

        sorted_candidates = sorted(final_graph_scores.items(), key=lambda x: x[1], reverse=True)[: max(1, top_k)]
        prob_scores = self._to_probabilities(final_graph_scores)
        return final_graph_scores, sorted_candidates, prob_scores
