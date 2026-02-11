# simulation.py
"""
Gravit Trust Lab — MVP v0.1
Simulation runner:
- reads scenarios.json
- runs baseline + attack scenarios
- writes simulation_output.jsonl (structured events)
- uses trust_core.py, propagation.py, audit.py
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import propagation as prop
import trust_core as core
import audit as audit_mod

OUT_LOG = "simulation_output.jsonl"


def now_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log_event(obj: Dict[str, Any]) -> None:
    line = json.dumps(obj, ensure_ascii=False)
    with open(OUT_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# -------------------------
# Graph generators
# -------------------------
def generate_powerlaw_graph(num_nodes: int, m: int = 3, seed: int | None = None) -> prop.TrustGraph:
    if seed is not None:
        random.seed(seed)
    g = prop.TrustGraph()
    nodes = [f"N{i}" for i in range(num_nodes)]
    # small core
    core_n = min(3, num_nodes)
    for i in range(core_n):
        for j in range(i + 1, core_n):
            g.add_edge(nodes[i], nodes[j], weight=1.0, sign=1.0)
    for i in range(core_n, num_nodes):
        targets = random.choices(nodes[:i], k=max(1, m))
        for t in targets:
            g.add_edge(nodes[i], t, weight=1.0, sign=1.0)
    return g


def all_nodes(graph: prop.TrustGraph) -> List[str]:
    s = set(graph.out.keys())
    for u, outs in graph.out.items():
        for v, _, _ in outs:
            s.add(v)
    return sorted(list(s))


# -------------------------
# Trust init + metrics
# -------------------------
def init_store(graph: prop.TrustGraph, seed: int) -> Dict[str, Any]:
    random.seed(seed)
    nodes = all_nodes(graph)
    store: Dict[str, Any] = {"reputation": {}, "model_confidence": {}, "trust_state": {}}
    for n in nodes:
        store["reputation"][n] = random.uniform(0.3, 0.9)
        store["model_confidence"][n] = random.uniform(0.4, 0.95)
        store["trust_state"][n] = 0.5
    return store


def apply_initial_evidence(store: Dict[str, Any], target_id: str, good: bool = True) -> float:
    ev = {
        "signature_valid": True if good else False,
        "merkle_valid": True if good else False,
        "semantic_similarity": random.uniform(0.7, 0.95) if good else random.uniform(0.0, 0.4),
        "contradiction_penalty": random.uniform(0.0, 0.1) if good else random.uniform(0.2, 0.6),
        "age_seconds": random.randint(0, 300),
    }
    prev = store["trust_state"].get(target_id, 0.5)
    p = core.compute_provenance(ev)
    s = core.compute_semantic_consistency(target_id, ev)
    r = core.get_reputation(store, target_id)
    m = core.get_model_confidence(store, target_id)
    decay = core.compute_decay(ev)
    weights = {"p": 0.25, "s": 0.35, "r": 0.2, "m": 0.2}
    new_score = core.compute_trust_linear(p, s, r, m, decay, weights)
    store["trust_state"][target_id] = new_score

    audit = core.make_audit_record(target_id, prev, new_score, ev, {"source": "simulation_initial"})
    audit_mod.append_audit(audit)
    return new_score


def collect_metrics(store: Dict[str, Any]) -> Dict[str, Any]:
    vals = list(store["trust_state"].values())
    if not vals:
        return {"node_count": 0, "avg_trust": 0.0, "min_trust": 0.0, "max_trust": 0.0, "stddev_trust": 0.0}
    avg = sum(vals) / len(vals)
    var = sum((x - avg) ** 2 for x in vals) / len(vals)
    return {
        "node_count": len(vals),
        "avg_trust": avg,
        "min_trust": min(vals),
        "max_trust": max(vals),
        "stddev_trust": var ** 0.5,
    }


# -------------------------
# Attack scenarios
# -------------------------
def sybil_injection(graph: prop.TrustGraph, store: Dict[str, Any], n_sybil: int, connect_to: str, seed: int) -> List[str]:
    random.seed(seed)
    sybils = [f"SYB{i}" for i in range(n_sybil)]
    for s in sybils:
        graph.add_edge(s, connect_to, weight=1.0, sign=1.0)
        # light clique
        for t in sybils:
            if t != s and random.random() < 0.4:
                graph.add_edge(s, t, weight=random.uniform(0.6, 1.2), sign=1.0)
        store["trust_state"][s] = random.uniform(0.7, 0.95)  # "bought" initial trust
        store["reputation"][s] = random.uniform(0.2, 0.4)    # low reputation baseline
        store["model_confidence"][s] = random.uniform(0.4, 0.7)
    return sybils


def contradiction_flood(graph: prop.TrustGraph, store: Dict[str, Any], target: str, n_attackers: int, seed: int) -> List[str]:
    random.seed(seed)
    nodes = [n for n in store["trust_state"].keys() if not n.startswith("SYB")]
    attackers = random.sample(nodes, min(len(nodes), n_attackers))
    for a in attackers:
        graph.add_edge(a, target, weight=random.uniform(0.8, 1.2), sign=-1.0)
        store["trust_state"][a] = max(store["trust_state"].get(a, 0.5), random.uniform(0.5, 0.9))
    return attackers


def rapid_trust_drop(store: Dict[str, Any], target: str) -> float:
    prev = store["trust_state"].get(target, 0.5)
    ev = {
        "signature_valid": False,
        "merkle_valid": False,
        "semantic_similarity": random.uniform(0.0, 0.3),
        "contradiction_penalty": random.uniform(0.2, 0.7),
        "age_seconds": 1,
    }
    p = core.compute_provenance(ev)
    s = core.compute_semantic_consistency(target, ev)
    r = core.get_reputation(store, target)
    m = core.get_model_confidence(store, target)
    decay = core.compute_decay(ev)
    weights = {"p": 0.4, "s": 0.3, "r": 0.15, "m": 0.15}
    new_score = core.compute_trust_linear(p, s, r, m, decay, weights)
    store["trust_state"][target] = new_score
    audit = core.make_audit_record(target, prev, new_score, ev, {"scenario": "rapid_trust_drop"})
    audit_mod.append_audit(audit)
    return new_score


# -------------------------
# Scenario runner
# -------------------------
def build_graph(params: dict) -> prop.TrustGraph:
    ginfo = params.get("graph", {"type": "powerlaw", "num_nodes": 50, "m": 3})
    gtype = ginfo.get("type", "powerlaw")
    if gtype != "powerlaw":
        # v0.1: keep canonical powerlaw; other generators can be added later
        gtype = "powerlaw"
    return generate_powerlaw_graph(
        num_nodes=int(ginfo.get("num_nodes", 50)),
        m=int(ginfo.get("m", 3)),
        seed=int(params.get("seed", 42)),
    )


def propagate(graph: prop.TrustGraph, store: Dict[str, Any], alpha: float) -> None:
    T = prop.propagate_trust(graph, store["trust_state"], alpha=alpha)
    store["trust_state"].update(T)


def run_one_scenario(scenario: dict) -> None:
    sid = scenario["id"]
    stype = scenario.get("type", "unknown")
    params = scenario.get("params", {}) or {}
    seed = int(params.get("seed", 42))
    alpha = float(params.get("alpha", 0.85))

    graph = build_graph(params)
    store = init_store(graph, seed=seed)

    # initial evidence for a few nodes
    for n in all_nodes(graph)[:10]:
        apply_initial_evidence(store, n, good=True)

    log_event({"kind": "scenario_start", "scenario": scenario, "timestamp": now_ts()})

    propagate(graph, store, alpha=alpha)
    baseline_metrics = collect_metrics(store)
    log_event({"kind": "scenario_baseline", "scenario_id": sid, "metrics": baseline_metrics, "timestamp": now_ts()})

    # execute scenario action(s)
    if stype == "baseline_run":
        log_event({"kind": "scenario_note", "scenario_id": sid, "note": "Baseline only (no attacks).", "timestamp": now_ts()})

    elif stype == "sybil_injection":
        n_sybil = int(params.get("n_sybil", 10))
        connect_to = str(params.get("connect_to", "N0"))
        sybils = sybil_injection(graph, store, n_sybil=n_sybil, connect_to=connect_to, seed=seed + 1)
        log_event({"kind": "scenario_note", "scenario_id": sid, "note": f"Injected {len(sybils)} sybils -> {connect_to}.", "timestamp": now_ts()})
        propagate(graph, store, alpha=alpha)

    elif stype == "contradiction_flood":
        target = str(params.get("target", "N1"))
        n_attackers = int(params.get("n_attackers", 5))
        attackers = contradiction_flood(graph, store, target=target, n_attackers=n_attackers, seed=seed + 2)
        log_event({"kind": "scenario_note", "scenario_id": sid, "note": f"Added {len(attackers)} contradicting edges -> {target}.", "timestamp": now_ts()})
        propagate(graph, store, alpha=alpha)

    elif stype == "rapid_trust_drop":
        target = str(params.get("target", "N2"))
        new_score = rapid_trust_drop(store, target=target)
        log_event({"kind": "scenario_note", "scenario_id": sid, "note": f"Rapid drop on {target}: new_score={new_score:.4f}.", "timestamp": now_ts()})
        propagate(graph, store, alpha=alpha)

    elif stype == "sequence":
        seq = params.get("sequence", [])
        for step_i, step in enumerate(seq):
            t = step.get("type")
            p = step.get("params", {}) or {}
            if t == "sybil_injection":
                sybils = sybil_injection(graph, store, n_sybil=int(p.get("n_sybil", 20)), connect_to=str(p.get("connect_to", "N0")), seed=seed + 10 + step_i)
                log_event({"kind": "scenario_note", "scenario_id": sid, "note": f"[seq:{step_i}] sybil_injection: {len(sybils)} -> {p.get('connect_to','N0')}", "timestamp": now_ts()})
            elif t == "contradiction_flood":
                attackers = contradiction_flood(graph, store, target=str(p.get("target", "N1")), n_attackers=int(p.get("n_attackers", 10)), seed=seed + 20 + step_i)
                log_event({"kind": "scenario_note", "scenario_id": sid, "note": f"[seq:{step_i}] contradiction_flood: {len(attackers)} -> {p.get('target','N1')}", "timestamp": now_ts()})
            elif t == "rapid_trust_drop":
                target = str(p.get("target", "N2"))
                new_score = rapid_trust_drop(store, target=target)
                log_event({"kind": "scenario_note", "scenario_id": sid, "note": f"[seq:{step_i}] rapid_drop: {target} new_score={new_score:.4f}", "timestamp": now_ts()})
            propagate(graph, store, alpha=alpha)

    else:
        log_event({"kind": "scenario_note", "scenario_id": sid, "note": f"Unknown scenario type: {stype}", "timestamp": now_ts()})

    after_metrics = collect_metrics(store)
    log_event({"kind": "scenario_after", "scenario_id": sid, "metrics": after_metrics, "timestamp": now_ts()})
    log_event({"kind": "scenario_end", "scenario_id": sid, "timestamp": now_ts()})


def load_scenarios(path: str) -> List[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", default="scenarios.json", help="Path to scenarios json")
    args = parser.parse_args()

    # fresh log
    if os.path.exists(OUT_LOG):
        os.remove(OUT_LOG)

    scenarios = load_scenarios(args.scenarios)
    for sc in scenarios:
        run_one_scenario(sc)

    print(f"[OK] Simulation log written: {OUT_LOG}")


if __name__ == "__main__":
    main()
