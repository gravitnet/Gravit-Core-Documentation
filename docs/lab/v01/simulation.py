# simulation.py
"""
Gravit Trust Lab — unified simulation v0.2
------------------------------------------
Goal: show not only trust drop, but quarantine isolation + recovery.

Depends on:
  - propagation.py (v0.2 supports node_out_weight in propagate_trust)
  - trust_core.py
  - policy.py
  - audit.py (optional; if present we write audit records)

Outputs:
  - simulation_output.jsonl (metrics per step)
  - audit_log.jsonl (if audit.py exists & imported)
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import propagation as prop
import trust_core as core

# audit is optional
try:
    import audit as audit_mod
except Exception:
    audit_mod = None  # type: ignore

from policy import ThresholdQuarantinePolicy, PolicyConfig, NodeStatus


OUTPUT_FILE = "simulation_output.jsonl"


def now_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_jsonl(path: str, record: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


@dataclass
class SimConfig:
    seed: int = 42
    node_count: int = 60
    m_attach: int = 3                 # preferential-attachment edges per new node
    alpha: float = 0.85               # propagation damping
    steps: int = 20

    # event probabilities
    p_good_evidence: float = 0.85     # probability node emits "good" evidence in a step
    p_attack_event: float = 0.12      # probability an attack-type event happens at a step

    # attack targets
    attack_target: str = "N1"
    shock_target: str = "N2"
    sybil_target: str = "N0"


# -------------------------
# Graph generators
# -------------------------
def generate_powerlaw_graph(num_nodes: int, m: int, seed: int | None = None) -> prop.TrustGraph:
    if seed is not None:
        random.seed(seed)
    g = prop.TrustGraph()
    nodes = [f"N{i}" for i in range(num_nodes)]
    # small initial clique
    core_size = min(3, num_nodes)
    for i in range(core_size):
        for j in range(i + 1, core_size):
            g.add_edge(nodes[i], nodes[j], weight=1.0, sign=1.0)
            g.add_edge(nodes[j], nodes[i], weight=1.0, sign=1.0)

    # preferential-ish attachment via random choices among existing
    for i in range(core_size, num_nodes):
        existing = nodes[:i]
        targets = random.choices(existing, k=m)
        for t in targets:
            if t == nodes[i]:
                continue
            g.add_edge(nodes[i], t, weight=random.uniform(0.7, 1.3), sign=1.0)
            # occasional back edge
            if random.random() < 0.35:
                g.add_edge(t, nodes[i], weight=random.uniform(0.6, 1.1), sign=1.0)
    return g


def graph_nodes(g: prop.TrustGraph) -> List[str]:
    s = set(g.out.keys())
    for u, outs in g.out.items():
        for (v, _, _) in outs:
            s.add(v)
    return sorted(s)


# -------------------------
# Evidence model (simple but effective for “wow demo”)
# -------------------------
def make_good_evidence() -> Dict[str, Any]:
    return {
        "signature_valid": True,
        "merkle_valid": True,
        "semantic_similarity": random.uniform(0.75, 0.95),
        "contradiction_penalty": random.uniform(0.0, 0.06),
        "age_seconds": random.randint(0, 180),
    }


def make_bad_evidence() -> Dict[str, Any]:
    return {
        "signature_valid": False,
        "merkle_valid": False,
        "semantic_similarity": random.uniform(0.0, 0.35),
        "contradiction_penalty": random.uniform(0.25, 0.70),
        "age_seconds": random.randint(0, 10),
    }


def compute_local_trust(store: Dict[str, Any], node_id: str, evidence: Dict[str, Any]) -> float:
    prev = float(store["trust_state"].get(node_id, 0.5))

    p = core.compute_provenance(evidence)
    s = core.compute_semantic_consistency(node_id, evidence)
    r = core.get_reputation(store, node_id)
    m = core.get_model_confidence(store, node_id)
    decay = core.compute_decay(evidence)

    # You can tune weights later; keep interpretable for v0.2 demo
    weights = evidence.get("weights", {"p": 0.25, "s": 0.35, "r": 0.20, "m": 0.20})
    new_score = core.compute_trust_linear(p, s, r, m, decay, weights)

    # audit (optional)
    if audit_mod is not None:
        rec = core.make_audit_record(node_id, prev, new_score, evidence, {"source": "simulation_v0.2", "weights": weights})
        audit_mod.append_audit(rec)

    store["trust_state"][node_id] = float(new_score)
    return float(new_score)


# -------------------------
# Attacks (lightweight, but demonstrative)
# -------------------------
def apply_sybil_injection(g: prop.TrustGraph, store: Dict[str, Any], target: str, n_sybil: int = 20) -> List[str]:
    sybils = [f"SYB{i}" for i in range(n_sybil)]
    for s in sybils:
        # sybil links toward target
        g.add_edge(s, target, weight=1.2, sign=1.0)
        # some sybil clique
        for t in sybils:
            if t != s and random.random() < 0.5:
                g.add_edge(s, t, weight=random.uniform(0.7, 1.2), sign=1.0)
        # artificially elevated start trust (simulating “bought reputation”)
        store["trust_state"][s] = random.uniform(0.70, 0.93)
        store["reputation"][s] = random.uniform(0.55, 0.85)
        store["model_confidence"][s] = random.uniform(0.50, 0.80)
    return sybils


def apply_contradiction_flood(g: prop.TrustGraph, store: Dict[str, Any], target: str, attackers: int = 10) -> List[str]:
    nodes = list(store["trust_state"].keys())
    pool = [n for n in nodes if not n.startswith("SYB")]
    chosen = random.sample(pool, k=min(len(pool), attackers))
    for a in chosen:
        g.add_edge(a, target, weight=random.uniform(0.9, 1.3), sign=-1.0)
        # make attackers credible enough to matter
        store["trust_state"][a] = max(store["trust_state"].get(a, 0.5), random.uniform(0.55, 0.85))
    return chosen


def apply_rapid_trust_drop(store: Dict[str, Any], target: str) -> float:
    ev = make_bad_evidence()
    # emphasize provenance in rapid drop
    ev["weights"] = {"p": 0.45, "s": 0.30, "r": 0.15, "m": 0.10}
    return compute_local_trust(store, target, ev)


# -------------------------
# Metrics
# -------------------------
def count_by_status(policy: ThresholdQuarantinePolicy) -> Dict[str, int]:
    out: Dict[str, int] = {s.value: 0 for s in NodeStatus}
    for st in policy.status.values():
        out[st.value] = out.get(st.value, 0) + 1
    return out


def compute_metrics(store: Dict[str, Any], policy: ThresholdQuarantinePolicy) -> Dict[str, Any]:
    trusts = list(store["trust_state"].values())
    if not trusts:
        return {"avg": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
    avg = sum(trusts) / len(trusts)
    mn = min(trusts)
    mx = max(trusts)
    var = sum((x - avg) ** 2 for x in trusts) / len(trusts)
    return {
        "node_count": len(trusts),
        "avg_trust": round(avg, 6),
        "min_trust": round(mn, 6),
        "max_trust": round(mx, 6),
        "stddev_trust": round(var ** 0.5, 6),
        "status_counts": count_by_status(policy),
    }


# -------------------------
# Main simulation loop (v0.2)
# -------------------------
def run_simulation(cfg: SimConfig | None = None, policy_cfg: PolicyConfig | None = None) -> Dict[str, Any]:
    cfg = cfg or SimConfig()
    random.seed(cfg.seed)

    # fresh outputs
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    # init
    g = generate_powerlaw_graph(cfg.node_count, cfg.m_attach, seed=cfg.seed)
    nodes = graph_nodes(g)

    store: Dict[str, Any] = {
        "reputation": {},
        "model_confidence": {},
        "trust_state": {},
    }

    # baseline reputation & model confidence
    for n in nodes:
        store["reputation"][n] = random.uniform(0.35, 0.90)
        store["model_confidence"][n] = random.uniform(0.40, 0.95)
        store["trust_state"][n] = 0.5

    policy = ThresholdQuarantinePolicy(policy_cfg or PolicyConfig())
    for n in list(store["trust_state"].keys()):
        policy.ensure_node(n, initial_trust=store["trust_state"][n])

    # STEP 0: warm-up propagation
    node_out_weight = {n: policy.node_out_weight(n) for n in store["trust_state"].keys()}
    T = prop.propagate_trust(g, store["trust_state"], node_out_weight=node_out_weight, alpha=cfg.alpha)
    store["trust_state"].update(T)
    for n in list(store["trust_state"].keys()):
        policy.step_update(n, store["trust_state"][n])

    write_jsonl(OUTPUT_FILE, {
        "ts": now_ts(),
        "step": 0,
        "phase": "warmup",
        "metrics": compute_metrics(store, policy),
    })

    # MAIN LOOP
    for step in range(1, cfg.steps + 1):
        events: List[Dict[str, Any]] = []

        # 1) local evidence updates (per node)
        for n in list(store["trust_state"].keys()):
            good = (random.random() < cfg.p_good_evidence)
            ev = make_good_evidence() if good else make_bad_evidence()
            new_local = compute_local_trust(store, n, ev)
            new_status = policy.step_update(n, new_local)
            events.append({"node": n, "kind": "evidence", "good": good, "trust": round(new_local, 6), "status": new_status.value})

        # 2) occasional attacks (to trigger quarantine/recovery dynamics)
        attack_happened = False
        if random.random() < cfg.p_attack_event:
            attack_happened = True
            # rotate attack types to show different "wow" behaviors
            if step % 3 == 1:
                sybils = apply_sybil_injection(g, store, target=cfg.sybil_target, n_sybil=18)
                for s in sybils:
                    policy.ensure_node(s, initial_trust=store["trust_state"][s])
                events.append({"kind": "attack", "type": "sybil_injection", "target": cfg.sybil_target, "count": len(sybils)})
            elif step % 3 == 2:
                attackers = apply_contradiction_flood(g, store, target=cfg.attack_target, attackers=10)
                events.append({"kind": "attack", "type": "contradiction_flood", "target": cfg.attack_target, "attackers": attackers})
            else:
                new_score = apply_rapid_trust_drop(store, target=cfg.shock_target)
                # policy update (rapid drop should quarantine)
                policy.step_update(cfg.shock_target, new_score)
                events.append({"kind": "attack", "type": "rapid_trust_drop", "target": cfg.shock_target, "new_trust": round(new_score, 6)})

        # 3) propagation with policy-aware node_out_weight
        node_out_weight = {n: policy.node_out_weight(n) for n in store["trust_state"].keys()}
        T = prop.propagate_trust(g, store["trust_state"], node_out_weight=node_out_weight, alpha=cfg.alpha)
        store["trust_state"].update(T)

        # 4) apply policy after propagation (so quarantine affects next step influence)
        for n in list(store["trust_state"].keys()):
            policy.step_update(n, store["trust_state"][n])

        rec = {
            "ts": now_ts(),
            "step": step,
            "phase": "main",
            "attack_happened": attack_happened,
            "metrics": compute_metrics(store, policy),
            # keep events compact; can be turned off if too big
            "events_sample": events[:20],
        }
        write_jsonl(OUTPUT_FILE, rec)

    return {
        "output_file": OUTPUT_FILE,
        "final_metrics": compute_metrics(store, policy),
        "status_snapshot": policy.export_status_snapshot(),
        "node_count": len(store["trust_state"]),
    }


if __name__ == "__main__":
    result = run_simulation()
    print("Simulation finished.")
    print("Output:", result["output_file"])
    print("Final metrics:", result["final_metrics"])
