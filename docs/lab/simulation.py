# simulation.py (v0.2) — unified
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Callable, Tuple

import propagation as prop
import policy as pol


# ---- Optional imports from your kit (tolerant) ----
# We don't break if your kit has different structure.
try:
    import trust_core  # type: ignore
except Exception:
    trust_core = None  # type: ignore

try:
    import attacks  # type: ignore
except Exception:
    attacks = None  # type: ignore


def _now_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def build_out_weight_map(policy_states: Dict[str, pol.NodePolicyState]) -> Dict[str, float]:
    return {nid: float(ps.out_weight) for nid, ps in policy_states.items()}


def collect_state_counts(policy_states: Dict[str, pol.NodePolicyState]) -> Dict[str, int]:
    counts = {"NORMAL": 0, "WARNING": 0, "QUARANTINED": 0, "RECOVERING": 0}
    for ps in policy_states.values():
        counts[ps.state] = counts.get(ps.state, 0) + 1
    return counts


def run_recovery_ticks(
    graph: Any,
    store: Dict[str, Any],
    policy_states: Dict[str, pol.NodePolicyState],
    thresholds: Dict[str, float],
    policy_cfg: Dict[str, float],
    ticks: int = 5,
    alpha: float = 0.85,
) -> Dict[str, pol.NodePolicyState]:
    for _ in range(int(ticks)):
        policy_states = pol.policy_step(policy_states, store["trust_state"], thresholds, policy_cfg)
        out_w = build_out_weight_map(policy_states)
        T = prop.propagate_trust(graph, store["trust_state"], alpha=alpha, node_out_weight=out_w)
        store["trust_state"].update(T)
        policy_states = pol.policy_step(policy_states, store["trust_state"], thresholds, policy_cfg)
    return policy_states


@dataclass
class PhaseRecord:
    phase: str
    step: int
    ts: str
    metrics: Dict[str, Any]


def _default_nodes_from_graph(graph: Any) -> List[str]:
    # Best-effort: use trust_state keys later; this is for initialization only
    nodes = set()
    if hasattr(graph, "out") and isinstance(graph.out, dict):
        nodes |= set(graph.out.keys())
        for u, outs in graph.out.items():
            for e in outs:
                if isinstance(e, tuple) and len(e) >= 1:
                    nodes.add(str(e[0]))
                elif isinstance(e, dict):
                    v = e.get("v") or e.get("dst") or e.get("to")
                    if v is not None:
                        nodes.add(str(v))
    return sorted(nodes)


def _init_trust_state(graph: Any, initial_trust: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    initial_trust = dict(initial_trust or {})
    if initial_trust:
        return {k: float(v) for k, v in initial_trust.items()}

    nodes = _default_nodes_from_graph(graph)
    if not nodes:
        # fallback: if graph provides nodes()
        if hasattr(graph, "nodes"):
            try:
                nodes = list(graph.nodes())
            except Exception:
                nodes = []
    return {str(n): 0.5 for n in nodes}


def _attack_functions() -> List[Tuple[str, Callable[[Any, Dict[str, Any]], None]]]:
    """
    Try to locate known attack hooks from either attacks.py or trust_core.py.
    Signature expected: fn(graph, store) -> modifies store["trust_state"] or graph.
    """
    candidates: List[Tuple[str, str]] = [
        ("sybil_injection", "sybil_injection"),
        ("contradiction_attack", "contradiction_attack"),
        ("rapid_trust_drop", "rapid_trust_drop"),
    ]

    funcs: List[Tuple[str, Callable[[Any, Dict[str, Any]], None]]] = []

    for phase, name in candidates:
        fn = None
        if attacks is not None and hasattr(attacks, name):
            fn = getattr(attacks, name)
        elif trust_core is not None and hasattr(trust_core, name):
            fn = getattr(trust_core, name)

        if callable(fn):
            funcs.append((phase, fn))
    return funcs


def run_full_simulation(
    graph: Any,
    initial_trust: Optional[Dict[str, float]] = None,
    alpha: float = 0.85,
    baseline_ticks: int = 5,
    recovery_ticks: int = 5,
    thresholds: Optional[Dict[str, float]] = None,
    policy_cfg: Optional[Dict[str, float]] = None,
    out_jsonl_path: Optional[str] = "simulation_output.jsonl",
) -> Dict[str, Any]:
    """
    v0.2 pipeline:
      baseline (policy -> propagate -> policy) ticks
      for each attack:
         apply attack
         recovery ticks (policy/prop/policy loop)
      logs state_counts per phase
    """
    thresholds = dict(thresholds or pol.DEFAULT_THRESHOLDS)
    policy_cfg = dict(policy_cfg or pol.DEFAULT_POLICY)

    store: Dict[str, Any] = {
        "trust_state": _init_trust_state(graph, initial_trust),
        "meta": {},
    }

    policy_states: Dict[str, pol.NodePolicyState] = {}

    records: List[PhaseRecord] = []

    def log_phase(phase: str, step: int) -> None:
        sc = collect_state_counts(policy_states)
        trust_vals = list(store["trust_state"].values())
        avg_trust = (sum(trust_vals) / len(trust_vals)) if trust_vals else 0.0
        metrics = {
            "avg_trust": avg_trust,
            "min_trust": min(trust_vals) if trust_vals else 0.0,
            "max_trust": max(trust_vals) if trust_vals else 0.0,
            "state_counts": sc,
        }
        rec = PhaseRecord(phase=phase, step=step, ts=_now_ts(), metrics=metrics)
        records.append(rec)

        if out_jsonl_path:
            with open(out_jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "ts": rec.ts,
                    "phase": rec.phase,
                    "step": rec.step,
                    **rec.metrics
                }, ensure_ascii=False) + "\n")

    # Clear output file
    if out_jsonl_path:
        with open(out_jsonl_path, "w", encoding="utf-8") as f:
            f.write("")  # truncate

    # ---- baseline ticks ----
    for t in range(int(baseline_ticks)):
        policy_states = pol.policy_step(policy_states, store["trust_state"], thresholds, policy_cfg)
        out_w = build_out_weight_map(policy_states)
        T = prop.propagate_trust(graph, store["trust_state"], alpha=alpha, node_out_weight=out_w)
        store["trust_state"].update(T)
        policy_states = pol.policy_step(policy_states, store["trust_state"], thresholds, policy_cfg)
        log_phase("baseline", t)

    # ---- attack phases (if exist) ----
    attack_fns = _attack_functions()
    if not attack_fns:
        # If no attack hooks found, still demonstrate recovery behavior
        # by applying an optional synthetic "shock" if available in policy thresholds.
        log_phase("no_attacks_found", int(baseline_ticks))
    else:
        step_counter = int(baseline_ticks)
        for phase, fn in attack_fns:
            # apply attack
            try:
                fn(graph, store)
            except TypeError:
                # some kits use signature fn(graph, trust_state)
                fn(graph, store["trust_state"])  # type: ignore
            except Exception as e:
                store["meta"][f"attack_error_{phase}"] = str(e)

            # policy + recovery loop
            policy_states = pol.policy_step(policy_states, store["trust_state"], thresholds, policy_cfg)
            log_phase(phase + "_after_attack", step_counter)
            step_counter += 1

            policy_states = run_recovery_ticks(
                graph, store, policy_states,
                thresholds, policy_cfg,
                ticks=int(recovery_ticks),
                alpha=alpha
            )
            log_phase(phase + "_after_recovery", step_counter)
            step_counter += int(recovery_ticks)

    result = {
        "ts": _now_ts(),
        "alpha": alpha,
        "baseline_ticks": int(baseline_ticks),
        "recovery_ticks": int(recovery_ticks),
        "thresholds": thresholds,
        "policy": policy_cfg,
        "final_state_counts": collect_state_counts(policy_states),
        "final_trust_state": store["trust_state"],
        "meta": store.get("meta", {}),
        "records": [
            {"phase": r.phase, "step": r.step, "ts": r.ts, **r.metrics}
            for r in records
        ],
    }
    return result


# ---- Optional CLI runner ----
if __name__ == "__main__":
    # If your trust_core has a builder, use it; else exit with helpful note.
    graph = None
    if trust_core is not None:
        for builder_name in ("build_demo_graph", "make_demo_graph", "build_graph", "demo_graph"):
            if hasattr(trust_core, builder_name) and callable(getattr(trust_core, builder_name)):
                graph = getattr(trust_core, builder_name)()
                break

    if graph is None:
        print("[simulation v0.2] No graph builder found in trust_core. "
              "Call run_full_simulation(graph=...) from your own entrypoint.")
    else:
        out = run_full_simulation(graph, out_jsonl_path="simulation_output.jsonl")
        with open("simulation_summary.json", "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print("[ok] wrote simulation_output.jsonl and simulation_summary.json")
