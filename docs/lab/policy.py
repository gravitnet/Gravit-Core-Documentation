# policy.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict


DEFAULT_THRESHOLDS = {
    "accept": 0.85,
    "warn": 0.60,
    "quarantine": 0.30,
    "recover": 0.55,
}

DEFAULT_POLICY = {
    "min_stable_steps": 3,          # сколько шагов доверие должно быть стабильным для recovery
    "max_trust_step_delta": 0.05,   # "стабильность" = изменение доверия меньше этого
    "quarantine_out_weight": 0.0,   # влияние карантинного узла наружу
    "recovering_out_weight": 0.2,   # влияние recovering узла наружу
    "warning_out_weight": 0.7,      # optional: смягчаем влияние warning
    "normal_out_weight": 1.0,
    "incoming_quarantine_weight": 0.3,  # optional: входящие влияния в quarantine
}


@dataclass
class NodePolicyState:
    node_id: str
    state: str = "NORMAL"  # NORMAL | WARNING | QUARANTINED | RECOVERING
    stable_steps: int = 0
    last_trust: float = 0.5
    out_weight: float = 1.0
    incoming_weight: float = 1.0


def evaluate_state(trust: float, prev_state: str, thresholds: Dict[str, float]) -> str:
    if trust < thresholds["quarantine"]:
        return "QUARANTINED"
    if trust < thresholds["warn"]:
        return "WARNING"
    # RECOVERING мы не сбрасываем автоматически, только через try_recover→NORMAL
    if trust >= thresholds["accept"] and prev_state in ("NORMAL", "WARNING"):
        return "NORMAL"
    return prev_state


def apply_state_effects(ps: NodePolicyState, policy: Dict[str, float]) -> None:
    if ps.state == "QUARANTINED":
        ps.out_weight = policy["quarantine_out_weight"]
        ps.incoming_weight = policy["incoming_quarantine_weight"]
    elif ps.state == "RECOVERING":
        ps.out_weight = policy["recovering_out_weight"]
        ps.incoming_weight = 1.0
    elif ps.state == "WARNING":
        ps.out_weight = policy["warning_out_weight"]
        ps.incoming_weight = 1.0
    else:  # NORMAL
        ps.out_weight = policy["normal_out_weight"]
        ps.incoming_weight = 1.0


def update_stability(ps: NodePolicyState, current_trust: float, policy: Dict[str, float]) -> None:
    # стабильность = доверие не скачет
    if abs(current_trust - ps.last_trust) <= policy["max_trust_step_delta"]:
        ps.stable_steps += 1
    else:
        ps.stable_steps = 0
    ps.last_trust = current_trust


def try_recover(ps: NodePolicyState, current_trust: float, thresholds: Dict[str, float], policy: Dict[str, float]) -> None:
    """
    Логика восстановления:
    QUARANTINED -> RECOVERING, если trust > recover и стабильность накоплена.
    RECOVERING -> NORMAL, если trust >= warn (или accept) и стабильность накоплена.
    """
    min_steps = int(policy["min_stable_steps"])

    if ps.state == "QUARANTINED":
        if current_trust > thresholds["recover"] and ps.stable_steps >= min_steps:
            ps.state = "RECOVERING"
            ps.stable_steps = 0  # перезапуск стабильности для следующего шага
            apply_state_effects(ps, policy)

    elif ps.state == "RECOVERING":
        # условие возврата: выбрались из warn-зоны и стабильны
        if current_trust >= thresholds["warn"] and ps.stable_steps >= min_steps:
            ps.state = "NORMAL"
            ps.stable_steps = 0
            apply_state_effects(ps, policy)


def policy_step(
    policy_states: Dict[str, NodePolicyState],
    trust_state: Dict[str, float],
    thresholds: Dict[str, float] | None = None,
    policy: Dict[str, float] | None = None,
) -> Dict[str, NodePolicyState]:
    thresholds = thresholds or DEFAULT_THRESHOLDS
    policy = policy or DEFAULT_POLICY

    # ensure policy_state exists per node
    for node_id, t in trust_state.items():
        if node_id not in policy_states:
            ps = NodePolicyState(node_id=node_id, last_trust=t)
            apply_state_effects(ps, policy)
            policy_states[node_id] = ps

    # update stability + state
    for node_id, t in trust_state.items():
        ps = policy_states[node_id]
        update_stability(ps, t, policy)
        new_state = evaluate_state(t, ps.state, thresholds)
        if new_state != ps.state:
            ps.state = new_state
            ps.stable_steps = 0
            apply_state_effects(ps, policy)

        # try recovery transitions (only after stability update)
        try_recover(ps, t, thresholds, policy)

    return policy_states
