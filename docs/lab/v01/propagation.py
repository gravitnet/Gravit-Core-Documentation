# propagation.py (v0.2 unified)
"""
Gravit Trust Lab — Propagation Engine v0.2
Adds:
  - node_out_weight: per-node multiplier for outgoing influence
    * QUARANTINED nodes -> 0.0 (no influence)
    * RECOVERING nodes -> e.g. 0.2 (partial influence)
  - stable incoming index cache (rebuilt on edge changes)

Graph model:
  out[src] = list of (dst, weight, semantic_sign)
    - weight >= 0
    - semantic_sign in [-1, 1] (support +1, contradict -1, partials allowed)

Propagation:
  PageRank-like with semantic sign:
    T_new[v] = alpha * sum_u ( (w_uv * s_uv / Z_u) * T[u] * node_out_weight[u] ) + (1-alpha) * B[v]

Where:
  - B[v] is bias (initial_trust)
  - Z_u = sum over outgoing edges of u: w_u*
    (optionally includes node_out_weight[u] as multiplier; we keep it separate in numerator)
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Any
import collections


class TrustGraph:
    def __init__(self) -> None:
        # out[src] -> list of (dst, weight, sign)
        self.out: Dict[str, List[Tuple[str, float, float]]] = collections.defaultdict(list)
        self._incoming_index: Optional[Dict[str, List[Tuple[str, float, float]]]] = None

    def add_edge(self, src: str, dst: str, weight: float = 1.0, sign: float = 1.0) -> None:
        self.out[src].append((dst, float(weight), float(sign)))
        self._incoming_index = None  # invalidate

    def get_outgoing(self, node: str) -> List[Tuple[str, float, float]]:
        return self.out.get(node, [])

    def _build_incoming_index(self) -> None:
        inc: Dict[str, List[Tuple[str, float, float]]] = collections.defaultdict(list)
        for u, outs in self.out.items():
            for (v, w, s) in outs:
                inc[v].append((u, float(w), float(s)))
        self._incoming_index = inc

    def get_incoming(self, node: str) -> List[Tuple[str, float, float]]:
        if self._incoming_index is None:
            self._build_incoming_index()
        return self._incoming_index.get(node, [])

    def nodes(self) -> List[str]:
        nodes = set(self.out.keys())
        for u, outs in self.out.items():
            for (v, _, _) in outs:
                nodes.add(v)
        return sorted(nodes)


def propagate_trust(
    graph: TrustGraph,
    initial_trust: Dict[str, float],
    alpha: float = 0.85,
    max_iter: int = 100,
    eps: float = 1e-6,
    node_out_weight: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    PageRank-like propagation with semantic sign and per-node outgoing multipliers.

    Args:
      graph: TrustGraph
      initial_trust: dict node->bias in [0,1] (also used as starting T)
      alpha: damping
      max_iter: iterations cap
      eps: convergence L1 threshold
      node_out_weight: dict node->multiplier [0,1] (or >1 if you want), default 1.0
        Example: QUARANTINED => 0.0, RECOVERING => 0.2
    """
    if node_out_weight is None:
        node_out_weight = {}

    # Union of known nodes
    nodes = set(initial_trust.keys()) | set(graph.nodes())

    # Initialize current trust (T) and bias (B)
    T: Dict[str, float] = {n: float(initial_trust.get(n, 0.5)) for n in nodes}
    B: Dict[str, float] = {n: float(initial_trust.get(n, 0.5)) for n in nodes}

    for _ in range(max_iter):
        T_new: Dict[str, float] = {}
        delta = 0.0

        for v in nodes:
            incoming_sum = 0.0

            for (u, w_uv, s_uv) in graph.get_incoming(v):
                outs = graph.get_outgoing(u)
                Z_u = sum(w for (_, w, _) in outs) if outs else 0.0
                if Z_u <= 0.0:
                    continue

                out_mul = float(node_out_weight.get(u, 1.0))
                # contribution from u -> v
                incoming_sum += (w_uv * s_uv / Z_u) * T.get(u, 0.0) * out_mul

            T_new[v] = alpha * incoming_sum + (1.0 - alpha) * B.get(v, 0.0)
            delta += abs(T_new[v] - T.get(v, 0.0))

        T = T_new
        if delta < eps:
            break

    # Clamp to [0,1]
    for k in list(T.keys()):
        if T[k] < 0.0:
            T[k] = 0.0
        elif T[k] > 1.0:
            T[k] = 1.0

    return T
