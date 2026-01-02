# MVP Implementation (Python)

This section contains the core Python modules for the Trust Engine MVP.

## 1. `trust_core.py`

```python
from typing import Dict, Any
import math
import time
import uuid

def now_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

# --- Linear trust aggregator ---
def compute_trust_linear(p: float, s: float, r: float, m: float, decay: float, weights: Dict[str, float]) -> float:
    base = weights['p']*p + weights['s']*s + weights['r']*r + weights['m']*m
    return max(0.0, min(1.0, decay * base))

def compute_provenance(evidence: Dict[str, Any]) -> float:
    a_sig = 0.6
    a_merkle = 0.4
    sig_ok = 1.0 if evidence.get("signature_valid", False) else 0.0
    merkle_ok = 1.0 if evidence.get("merkle_valid", False) else 0.0
    return a_sig * sig_ok + a_merkle * merkle_ok

def compute_semantic_consistency(target_id: str, evidence: Dict[str, Any]) -> float:
    sim = evidence.get("semantic_similarity", 0.5)
    penalty = evidence.get("contradiction_penalty", 0.0)
    value = sim - penalty
    return max(0.0, min(1.0, value))

def get_reputation(store: Dict[str, Any], target_id: str) -> float:
    return float(store.get("reputation", {}).get(target_id, 0.5))

def get_model_confidence(store: Dict[str, Any], target_id: str) -> float:
    return float(store.get("model_confidence", {}).get(target_id, 0.5))

def compute_decay(evidence: Dict[str, Any], gamma: float = 0.001) -> float:
    age = evidence.get("age_seconds", 0)
    return math.exp(-gamma * age)

def make_audit_record(target_id: str, previous: float, new: float, evidence: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    record = {
        "record_id": str(uuid.uuid4()),
        "target_id": target_id,
        "previous_trust": float(previous),
        "new_trust": float(new),
        "evidence": evidence,
        "meta": meta,
        "timestamp": now_ts()
    }
    return record
```

## 2. `propagation.py`

```python
from typing import Dict, List, Tuple, Any
import collections

class TrustGraph:
    def __init__(self):
        self.out = collections.defaultdict(list)
        self.incoming_index = None

    def add_edge(self, src: str, dst: str, weight: float = 1.0, sign: float = 1.0):
        self.out[src].append((dst, float(weight), float(sign)))
        self.incoming_index = None

    def get_outgoing(self, node: str):
        return self.out.get(node, [])

    def build_incoming_index(self):
        inc = collections.defaultdict(list)
        for u, outs in self.out.items():
            for (v, w, s) in outs:
                inc[v].append((u, w, s))
        self.incoming_index = inc

    def get_incoming(self, node: str):
        if self.incoming_index is None:
            self.build_incoming_index()
        return self.incoming_index.get(node, [])

def propagate_trust(graph: TrustGraph, initial_trust: Dict[str, float], alpha: float = 0.85,
                    max_iter: int = 100, eps: float = 1e-6) -> Dict[str, float]:
    nodes = set(initial_trust.keys()) | set(graph.out.keys())
    T = {n: initial_trust.get(n, 0.5) for n in nodes}
    B = {n: initial_trust.get(n, 0.5) for n in nodes}

    for _ in range(max_iter):
        T_new = {}
        delta = 0.0
        for v in nodes:
            incoming = 0.0
            for (u, w_uv, s_uv) in graph.get_incoming(v):
                outgoing = graph.get_outgoing(u)
                Zv = sum(w for (_, w, _) in outgoing) if outgoing else 0.0
                if Zv == 0: continue
                incoming += (w_uv * s_uv / Zv) * T.get(u, 0.0)
            T_new[v] = alpha * incoming + (1 - alpha) * B.get(v, 0.0)
            delta += abs(T_new[v] - T.get(v, 0.0))
        T = T_new
        if delta < eps:
            break
    for k in list(T.keys()):
        T[k] = max(0.0, min(1.0, T[k]))
    return T
```

## 3. `audit.py`

```python
from typing import Dict, Any, List
import json
import os

AUDIT_DB_FILE = "audit_log.jsonl"

def append_audit(record: Dict[str, Any]):
    line = json.dumps(record, ensure_ascii=False)
    with open(AUDIT_DB_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def load_audit(limit: int = 100) -> List[Dict[str, Any]]:
    if not os.path.exists(AUDIT_DB_FILE):
        return []
    lines = []
    with open(AUDIT_DB_FILE, "r", encoding="utf-8") as f:
        for i, l in enumerate(f):
            if i >= limit:
                break
            try:
                lines.append(json.loads(l))
            except Exception:
                continue
    return lines
```
