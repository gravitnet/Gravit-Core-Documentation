# propagation.py (v0.2) — unified
from __future__ import annotations

from typing import Dict, Optional, Iterable, Tuple, Any


# --- adapter helpers (to be tolerant to your existing TrustGraph API) ---

def _get_outgoing(graph: Any, u: str) -> Iterable[Tuple[str, float, float]]:
    """
    Returns iterable of (v, w_uv, s_uv).
    Supports:
      - graph.get_outgoing(u)
      - graph.out[u] where items are either (v,w,s) or dict-like
    """
    if hasattr(graph, "get_outgoing"):
        return graph.get_outgoing(u)

    out = None
    if hasattr(graph, "out"):
        out = graph.out.get(u, [])
    elif hasattr(graph, "adj"):
        out = graph.adj.get(u, [])
    else:
        out = []

    # Normalize common edge representations
    norm = []
    for e in out:
        if isinstance(e, tuple) and len(e) == 3:
            v, w, s = e
        elif isinstance(e, tuple) and len(e) == 2:
            v, w = e
            s = 1.0
        elif isinstance(e, dict):
            v = e.get("v") or e.get("dst") or e.get("to")
            w = float(e.get("w", e.get("weight", 1.0)))
            s = float(e.get("s", e.get("sign", 1.0)))
        else:
            continue
        norm.append((str(v), float(w), float(s)))
    return norm


def _get_incoming(graph: Any, v: str) -> Iterable[Tuple[str, float, float]]:
    """
    Returns iterable of (u, w_uv, s_uv).
    Supports:
      - graph.get_incoming(v)
      - graph.in_[v] / graph.inc[v] / graph.incoming[v]
      - fallback: build from outgoing (slower)
    """
    if hasattr(graph, "get_incoming"):
        return graph.get_incoming(v)

    inc = None
    for attr in ("in_", "inc", "incoming", "in_edges"):
        if hasattr(graph, attr):
            inc = getattr(graph, attr)
            if isinstance(inc, dict):
                inc = inc.get(v, [])
            break

    if inc is None:
        # fallback: compute by scanning outgoing
        norm = []
        nodes = []
        if hasattr(graph, "out") and isinstance(graph.out, dict):
            nodes = list(graph.out.keys())
        elif hasattr(graph, "adj") and isinstance(graph.adj, dict):
            nodes = list(graph.adj.keys())

        for u in nodes:
            for (vv, w, s) in _get_outgoing(graph, u):
                if vv == v:
                    norm.append((u, w, s))
        return norm

    # Normalize
    norm = []
    for e in inc:
        if isinstance(e, tuple) and len(e) == 3:
            u, w, s = e
        elif isinstance(e, tuple) and len(e) == 2:
            u, w = e
            s = 1.0
        elif isinstance(e, dict):
            u = e.get("u") or e.get("src") or e.get("from")
            w = float(e.get("w", e.get("weight", 1.0)))
            s = float(e.get("s", e.get("sign", 1.0)))
        else:
            continue
        norm.append((str(u), float(w), float(s)))
    return norm


def propagate_trust(
    graph: Any,
    initial_trust: Dict[str, float],
    alpha: float = 0.85,
    max_iter: int = 100,
    eps: float = 1e-6,
    node_out_weight: Optional[Dict[str, float]] = None,  # NEW v0.2
) -> Dict[str, float]:
    """
    PageRank-like propagation with semantic sign.
    node_out_weight[u] in [0..1] scales outgoing influence of node u (quarantine/recovering hooks).
    """
    node_out_weight = node_out_weight or {}

    # Determine node set
    nodes = set(initial_trust.keys())
    if hasattr(graph, "out") and isinstance(graph.out, dict):
        nodes |= set(graph.out.keys())
        for u, outs in graph.out.items():
            for e in outs:
                if isinstance(e, tuple) and len(e) >= 1:
                    nodes.add(str(e[0]))
                elif isinstance(e, dict):
                    vv = e.get("v") or e.get("dst") or e.get("to")
                    if vv is not None:
                        nodes.add(str(vv))

    T = {n: float(initial_trust.get(n, 0.5)) for n in nodes}
    B = {n: float(initial_trust.get(n, 0.5)) for n in nodes}

    for _ in range(max_iter):
        T_new: Dict[str, float] = {}
        delta = 0.0

        for v in nodes:
            incoming = 0.0
            for (u, w_uv, s_uv) in _get_incoming(graph, v):
                out_scale = float(node_out_weight.get(u, 1.0))
                if out_scale <= 0.0:
                    continue

                outgoing = list(_get_outgoing(graph, u))
                Zv = sum(w for (_, w, _) in outgoing) if outgoing else 0.0
                if Zv == 0.0:
                    continue

                incoming += out_scale * (w_uv * s_uv / Zv) * T.get(u, 0.0)

            T_new[v] = alpha * incoming + (1 - alpha) * B.get(v, 0.0)
            delta += abs(T_new[v] - T.get(v, 0.0))

        T = T_new
        if delta < eps:
            break

    # Clamp 0..1
    for k in list(T.keys()):
        T[k] = max(0.0, min(1.0, float(T[k])))

    return T
