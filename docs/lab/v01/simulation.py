# simulation.py (v0.2 unified)
"""
Gravit Trust Lab — Simulation Engine v0.2
Adds: Quarantine & Threshold Policy hooks + Recovery ticks
Depends on:
  - propagation.py (v0.2: propagate_trust supports node_out_weight)
  - trust_core.py
  - audit.py (optional but recommended; used here)
  - policy.py (new in v0.2)

Outputs:
  - simulation_output.jsonl (phases + metrics + state_counts)
  - audit_log.jsonl (trust updates written via audit.append_audit)
"""

from __future__ import annotations

import os
import json
import time
import uuid
import random
from typing import Dict, List, Tuple, Any, Optional

import propagation as prop
import trust_core as core
import audit as audit_mod
import policy as pol

OUTPUT_FILE = "simulation_output.jsonl"


# -------------------------
# Helpers
# -------------------------
def now_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_output(record: Dict[str, Any]) -> None:
    line = json.dumps(record, ensure_ascii=False)
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# -------------------------
# Graph generators
# -------------------------
def generate_random_trust_graph(num_nodes: int, avg_deg: float = 2.0, seed: Optional[int] = None) -> prop.TrustGraph:
    if seed is not None:
        random.seed(seed)
    g = prop.TrustGraph()
    nodes = [f"N{i}" for i in range(num_nodes)]
    for u in nodes:
        k = max(1, int(random.expovariate(1.0 / avg_deg)))
        targets = random.sample(nodes, min(k, len(nodes)))
        for v in targets:
            if v == u:
                continue
            weight = random.uniform(0.5, 1.5)
            sign = 1.0
            g.add_edge(u, v, weight=weight, sign=sign)
    return g


def generate_powerlaw_graph(num_nodes: int, m: int = 2, seed: Optional[int] = None) -> prop.TrustGraph:
    """
    Simple preferential attachment-ish generator (not strict BA, but good enough for MVP sims).
    """
    if seed is not None:
        random.seed(seed)

    g = prop.TrustGraph()
    nodes = [f"N{i}" for i in range(num_nodes)]

    core_size = min(3, num_nodes)
    for i in range(core_size):
        for j in range(i + 1, core_size):
            g.add_edge(nodes[i], nodes[j], weight=1.0, sign=1.0)
            g.add_edge(nodes[j], nodes[i], weight=1.0, sign=1.0)

    for i in range(core_size, num_nodes):
        # prefer earlier nodes
        targets = random.choices(nodes[:i], k=m)
        for t in targets:
            if t == nodes[i]:
                continue
            g.add_edge(nodes[i], t, weight=1.0, sign=1.0)

    return g


def graph_nodes(graph: prop.TrustGraph) -> List[str]:
    nodes = set(graph.out.keys())
    for u, outs in graph.out.items():
        for (v, _, _) in outs:
            nodes.add(v)
    return sorted(nodes)


# -------------------------
# Baseline trust init
# -------------------------
def baseline_initial_trust(graph: prop.TrustGraph, baseline: float = 0.5) -> Dict[str, float]:
    return {n: baseline for n in graph_nodes(graph)}


def apply_initial_evidence(store: Dict[str, Any], target_id: str, good: bool = True) -> float:
    """
    Create an evidence packet and compute/store trust for a node.
    Writes an audit record via audit_mod.
    """
    ev = {
        "signature_valid": True if good else False,
        "merkle_valid": True if good else False,
        "semantic_similarity": random.uniform(0.7, 0.95) if good else random.uniform(0.0, 0.4),
        "contradiction_penalty": random.uniform(0.0, 0.1) if good else random.uniform(0.2, 0.6),
        "age_seconds": random.randint(0, 300),
    }

    prev = store.get("trust_state", {}).get(target_id, 0.5)

    p = core.compute_provenance(ev)
    s = core.compute_semantic_consistency(target_id, ev)
    r = core.get_reputation(store, target_id)
    m = core.get_model_confidence(store, target_id)
    decay = core.compute_decay(ev)

    weights = {"p": 0.25, "s": 0.35, "r": 0.2, "m": 0.2}
    new_score = core.compute_trust_linear(p, s, r, m, decay, weights)

    store.setdefault("trust_state", {})[target_id] = new_score
    audit = core.make_audit_record(
        target_id=target_id,
        previous=prev,
        new=new_score,
        evidence=ev,
        meta={"source": "simulation_initial"},
    )
    audit_mod.append_audit(audit)
    return new_score


# -------------------------
# Policy helpers (v0.2)
# -------------------------
def build_out_weight_map(policy_states: Dict[str, pol.NodePolicyState]) -> Dict[str, float]:
    return {nid: ps.out_weight for nid, ps in policy_states.items()}


def collect_state_counts(policy_states: Dict[str, pol.NodePolicyState]) -> Dict[str, int]:
    counts = {"NORMAL": 0, "WARNING": 0, "QUARANTINED": 0, "RECOVERING": 0}
    for ps in policy_states.values():
        counts[ps.state] = counts.get(ps.state, 0) + 1
    return counts


def propagate_with_policy(
    graph: prop.TrustGraph,
    store: Dict[str, Any],
    policy_states: Dict[str, pol.NodePolicyState],
    thresholds: Dict[str, float],
    policy_cfg: Dict[str, float],
    alpha: float = 0.85,
) -> Dict[str, pol.NodePolicyState]:
    """
    One policy-aware propagation step:
      policy_step -> propagate (with node_out_weight) -> policy_step
    """
    # pre-policy update
    policy_states = pol.policy_step(policy_states, store["trust_state"], thresholds, policy_cfg)

    # propagate with quarantine/recovery effect
    out_w = build_out_weight_map(policy_states)
    T = prop.propagate_trust(graph, store["trust_state"], alpha=alpha, node_out_weight=out_w)
    store["trust_state"].update(T)

    # post-policy update
    policy_states = pol.policy_step(policy_states, store["trust_state"], thresholds, policy_cfg)
    return policy_states


def run_recovery_ticks(
    graph: prop.TrustGraph,
    store: Dict[str, Any],
    policy_states: Dict[str, pol.NodePolicyState],
    thresholds: Dict[str, float],
    policy_cfg: Dict[str, float],
    ticks: int = 5,
    alpha: float = 0.85,
) -> Dict[str, pol.NodePolicyState]:
    for _ in range(ticks):
        policy_states = propagate_with_policy(graph, store, policy_states, thresholds, policy_cfg, alpha=alpha)
    return policy_states


# -------------------------
# Attack Scenarios
# -------------------------
def scenario_sybil_injection(
    graph: prop.TrustGraph,
    store: Dict[str, Any],
    n_sybil: int = 20,
    connect_to: str = "N0",
    seed: Optional[int] = None,
) -> List[str]:
    """
    Create sybil nodes that connect strongly to a target and to each other.
    """
    if seed is not None:
        random.seed(seed)

    sybils = [f"SYB{i}" for i in range(n_sybil)]
    for s in sybils:
        # strong support edge to target
        graph.add_edge(s, connect_to, weight=1.0, sign=1.0)

        # interconnect sybils (dense-ish clique)
        for t in sybils:
            if t == s:
                continue
            if random.random() < 0.6:
                graph.add_edge(s, t, weight=random.uniform(0.6, 1.2), sign=1.0)

        # simulate "bought" trust baseline (will be constrained by policy later if needed)
        store.setdefault("trust_state", {})[s] = random.uniform(0.7, 0.95)

    write_output(
        {
            "scenario": "sybil_injection",
            "target": connect_to,
            "sybil_count": n_sybil,
            "timestamp": now_ts(),
        }
    )
    return sybils


def scenario_contradiction_flood(
    graph: prop.TrustGraph,
    store: Dict[str, Any],
    target: str,
    n_attackers: int = 10,
    seed: Optional[int] = None,
) -> List[str]:
    """
    Add negative-sign edges into target from random attackers.
    """
    if seed is not None:
        random.seed(seed)

    pool = graph_nodes(graph)
    attackers = random.sample(pool, min(len(pool), n_attackers))
    for a in attackers:
        if a == target:
            continue
        graph.add_edge(a, target, weight=random.uniform(0.8, 1.2), sign=-1.0)
        store.setdefault("trust_state", {})[a] = random.uniform(0.4, 0.9)

    write_output(
        {
            "scenario": "contradiction_flood",
            "target": target,
            "attackers": attackers,
            "timestamp": now_ts(),
        }
    )
    return attackers


def scenario_rapid_trust_drop(store: Dict[str, Any], target: str) -> float:
    """
    Simulate sudden provenance failure (key compromise).
    Writes audit record.
    """
    prev = store.get("trust_state", {}).get(target, 0.5)
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

    # heavier weight on provenance in this shock scenario
    weights = {"p": 0.4, "s": 0.3, "r": 0.15, "m": 0.15}
    new_score = core.compute_trust_linear(p, s, r, m, decay, weights)

    store.setdefault("trust_state", {})[target] = new_score

    audit = core.make_audit_record(
        target_id=target,
        previous=prev,
        new=new_score,
        evidence=ev,
        meta={"scenario": "rapid_trust_drop", "weights": weights},
    )
    audit_mod.append_audit(audit)

    write_output(
        {
            "scenario": "rapid_trust_drop",
            "target": target,
            "prev": prev,
            "new": new_score,
            "timestamp": now_ts(),
        }
    )
    return new_score


# -------------------------
# Metrics
# -------------------------
def collect_metrics(store: Dict[str, Any], nodes: Optional[List[str]] = None) -> Dict[str, float]:
    if nodes is None:
        nodes = list(store.get("trust_state", {}).keys())

    trusts = [float(store.get("trust_state", {}).get(n, 0.5)) for n in nodes]
    if not trusts:
        return {
            "node_count": 0,
            "avg_trust": 0.0,
            "min_trust": 0.0,
            "max_trust": 0.0,
            "stddev_trust": 0.0,
        }

    avg = sum(trusts) / len(trusts)
    var = sum((x - avg) ** 2 for x in trusts) / len(trusts)
    return {
        "node_count": len(trusts),
        "avg_trust": avg,
        "min_trust": min(trusts),
        "max_trust": max(trusts),
        "stddev_trust": var ** 0.5,
    }


# -------------------------
# Runner
# -------------------------
def run_full_simulation(
    seed: int = 42,
    scenario_sequence: Optional[List[Tuple[str, Dict[str, Any]]]] = None,
    verbose: bool = True,
    graph_type: str = "powerlaw",
    node_count: int = 50,
    alpha: float = 0.85,
    recovery_ticks: int = 5,
) -> Dict[str, Any]:
    """
    scenario_sequence: list of (scenario_name, params)
      supported: "sybil_injection", "contradiction_flood", "rapid_trust_drop"
    """
    random.seed(seed)

    # (re)create outputs
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    # graph
    if graph_type == "random":
        graph = generate_random_trust_graph(node_count, avg_deg=2.0, seed=seed)
    else:
        graph = generate_powerlaw_graph(node_count, m=3, seed=seed)

    nodes = graph_nodes(graph)

    # store
    store: Dict[str, Any] = {"reputation": {}, "model_confidence": {}, "trust_state": {}}

    # init reputations + model confidence
    for n in nodes:
        store["reputation"][n] = random.uniform(0.3, 0.9)
        store["model_confidence"][n] = random.uniform(0.4, 0.95)

    # initial evidence for some nodes
    for n in nodes[:10]:
        apply_initial_evidence(store, n, good=True)

    # baseline fill
    baseline = baseline_initial_trust(graph, baseline=0.5)
    for k, v in baseline.items():
        store["trust_state"].setdefault(k, v)

    # policy state + configs
    policy_states: Dict[str, pol.NodePolicyState] = {}
    thresholds = pol.DEFAULT_THRESHOLDS.copy()
    policy_cfg = pol.DEFAULT_POLICY.copy()

    # initial policy-aware propagation
    policy_states = propagate_with_policy(graph, store, policy_states, thresholds, policy_cfg, alpha=alpha)
    write_output(
        {
            "phase": "initial_propagation",
            "graph_type": graph_type,
            "node_count": node_count,
            "alpha": alpha,
            "metrics": collect_metrics(store, nodes),
            "state_counts": collect_state_counts(policy_states),
            "timestamp": now_ts(),
        }
    )

    if scenario_sequence is None:
        scenario_sequence = [
            ("sybil_injection", {"n_sybil": 20, "connect_to": "N0"}),
            ("contradiction_flood", {"target": "N1", "n_attackers": 8}),
            ("rapid_trust_drop", {"target": "N2"}),
        ]

    # run scenarios
    for sn, params in scenario_sequence:
        if sn == "sybil_injection":
            sybils = scenario_sybil_injection(
                graph,
                store,
                n_sybil=int(params.get("n_sybil", 20)),
                connect_to=str(params.get("connect_to", "N0")),
                seed=params.get("seed"),
            )

            policy_states = propagate_with_policy(graph, store, policy_states, thresholds, policy_cfg, alpha=alpha)

            rec = {
                "phase": "after_sybil",
                "sybil_count": len(sybils),
                "target": params.get("connect_to", "N0"),
                "metrics": collect_metrics(store, nodes + sybils),
                "state_counts": collect_state_counts(policy_states),
                "timestamp": now_ts(),
            }
            write_output(rec)
            if verbose:
                print("After sybil:", rec)

            # recovery ticks
            policy_states = run_recovery_ticks(
                graph, store, policy_states, thresholds, policy_cfg, ticks=recovery_ticks, alpha=alpha
            )
            rec2 = {
                "phase": "recovery_after_sybil",
                "ticks": recovery_ticks,
                "metrics": collect_metrics(store, nodes + sybils),
                "state_counts": collect_state_counts(policy_states),
                "timestamp": now_ts(),
            }
            write_output(rec2)
            if verbose:
                print("Recovery after sybil:", rec2)

        elif sn == "contradiction_flood":
            target = str(params.get("target", "N1"))
            attackers = scenario_contradiction_flood(
                graph,
                store,
                target=target,
                n_attackers=int(params.get("n_attackers", 8)),
                seed=params.get("seed"),
            )

            policy_states = propagate_with_policy(graph, store, policy_states, thresholds, policy_cfg, alpha=alpha)

            rec = {
                "phase": "after_contradiction",
                "target": target,
                "attackers": attackers,
                "metrics": collect_metrics(store, nodes),
                "state_counts": collect_state_counts(policy_states),
                "timestamp": now_ts(),
            }
            write_output(rec)
            if verbose:
                print("After contradiction flood:", rec)

            policy_states = run_recovery_ticks(
                graph, store, policy_states, thresholds, policy_cfg, ticks=recovery_ticks, alpha=alpha
            )
            rec2 = {
                "phase": "recovery_after_contradiction",
                "ticks": recovery_ticks,
                "metrics": collect_metrics(store, nodes),
                "state_counts": collect_state_counts(policy_states),
                "timestamp": now_ts(),
            }
            write_output(rec2)
            if verbose:
                print("Recovery after contradiction:", rec2)

        elif sn == "rapid_trust_drop":
            target = str(params.get("target", "N2"))
            newscore = scenario_rapid_trust_drop(store, target)

            policy_states = propagate_with_policy(graph, store, policy_states, thresholds, policy_cfg, alpha=alpha)

            rec = {
                "phase": "after_rapid_drop",
                "target": target,
                "newscore": float(newscore),
                "metrics": collect_metrics(store, nodes),
                "state_counts": collect_state_counts(policy_states),
                "timestamp": now_ts(),
            }
            write_output(rec)
            if verbose:
                print("After rapid drop:", rec)

            policy_states = run_recovery_ticks(
                graph, store, policy_states, thresholds, policy_cfg, ticks=recovery_ticks, alpha=alpha
            )
            rec2 = {
                "phase": "recovery_after_rapid_drop",
                "ticks": recovery_ticks,
                "metrics": collect_metrics(store, nodes),
                "state_counts": collect_state_counts(policy_states),
                "timestamp": now_ts(),
            }
            write_output(rec2)
            if verbose:
                print("Recovery after rapid drop:", rec2)

        else:
            if verbose:
                print("Unknown scenario:", sn)

    final_metrics = collect_metrics(store, nodes)
    final_state_counts = collect_state_counts(policy_states)
    write_output(
        {
            "phase": "final_snapshot",
            "metrics": final_metrics,
            "state_counts": final_state_counts,
            "timestamp": now_ts(),
        }
    )
    if verbose:
        print("Final metrics:", final_metrics)
        print("Final state_counts:", final_state_counts)

    return {
        "graph": graph,
        "store": store,
        "metrics": final_metrics,
        "policy_states": policy_states,
    }


# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    # optional: clear audit to make runs easier to read (comment out if you want cumulative audit)
    # if os.path.exists("audit_log.jsonl"):
    #     os.remove("audit_log.jsonl")

    run_full_simulation(seed=42, verbose=True, graph_type="powerlaw", node_count=50, alpha=0.85, recovery_ticks=5)
    print("Simulation complete. See:", OUTPUT_FILE)
