# policy.py
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict


class NodeStatus(str, Enum):
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    QUARANTINED = "QUARANTINED"
    RECOVERING = "RECOVERING"


@dataclass
class PolicyConfig:
    # Thresholds
    warning_threshold: float = 0.65
    quarantine_threshold: float = 0.25
    recover_enter_threshold: float = 0.35
    normal_enter_threshold: float = 0.70

    # Dynamics
    rapid_drop_delta: float = 0.25          # if trust drops by >= this per step → quarantine trigger
    quarantine_min_steps: int = 3           # minimum time spent in quarantine before recovery allowed
    recovery_min_steps: int = 4             # minimum time spent recovering before NORMAL allowed

    # Influence multipliers (used by propagation node_out_weight)
    out_weight_normal: float = 1.0
    out_weight_warning: float = 0.6
    out_weight_quarantined: float = 0.0
    out_weight_recovering: float = 0.2


class ThresholdQuarantinePolicy:
    """
    Minimal policy engine:
      - decides node status transitions based on trust score dynamics
      - exposes node_out_weight multipliers for propagation engine
    """

    def __init__(self, cfg: PolicyConfig | None = None):
        self.cfg = cfg or PolicyConfig()

        # per-node state
        self.status: Dict[str, NodeStatus] = {}
        self.quarantine_age: Dict[str, int] = {}
        self.recovery_age: Dict[str, int] = {}
        self.last_trust: Dict[str, float] = {}

    def ensure_node(self, node_id: str, initial_trust: float = 0.5) -> None:
        if node_id not in self.status:
            self.status[node_id] = NodeStatus.NORMAL
            self.quarantine_age[node_id] = 0
            self.recovery_age[node_id] = 0
            self.last_trust[node_id] = float(initial_trust)

    def node_out_weight(self, node_id: str) -> float:
        st = self.status.get(node_id, NodeStatus.NORMAL)
        if st == NodeStatus.NORMAL:
            return self.cfg.out_weight_normal
        if st == NodeStatus.WARNING:
            return self.cfg.out_weight_warning
        if st == NodeStatus.QUARANTINED:
            return self.cfg.out_weight_quarantined
        if st == NodeStatus.RECOVERING:
            return self.cfg.out_weight_recovering
        return 1.0

    def step_update(self, node_id: str, new_trust: float) -> NodeStatus:
        """
        Apply one policy step for node based on trust and delta vs previous.
        Returns updated status.
        """
        self.ensure_node(node_id, initial_trust=new_trust)

        prev = self.last_trust.get(node_id, 0.5)
        delta_drop = max(0.0, prev - float(new_trust))

        st = self.status[node_id]

        # track ages
        if st == NodeStatus.QUARANTINED:
            self.quarantine_age[node_id] += 1
        else:
            self.quarantine_age[node_id] = 0

        if st == NodeStatus.RECOVERING:
            self.recovery_age[node_id] += 1
        else:
            self.recovery_age[node_id] = 0

        # hard triggers into quarantine
        if float(new_trust) <= self.cfg.quarantine_threshold or delta_drop >= self.cfg.rapid_drop_delta:
            self.status[node_id] = NodeStatus.QUARANTINED
            self.last_trust[node_id] = float(new_trust)
            return self.status[node_id]

        # transitions out of quarantine
        if st == NodeStatus.QUARANTINED:
            # only after minimum quarantine time and trust above recover_enter_threshold
            if self.quarantine_age[node_id] >= self.cfg.quarantine_min_steps and float(new_trust) >= self.cfg.recover_enter_threshold:
                self.status[node_id] = NodeStatus.RECOVERING
                self.last_trust[node_id] = float(new_trust)
                return self.status[node_id]
            else:
                self.last_trust[node_id] = float(new_trust)
                return self.status[node_id]

        # transitions within non-quarantine states
        if float(new_trust) < self.cfg.warning_threshold:
            self.status[node_id] = NodeStatus.WARNING
        else:
            self.status[node_id] = NodeStatus.NORMAL

        # exit from recovering to normal (stable & sufficient trust + time)
        if st == NodeStatus.RECOVERING:
            if (self.recovery_age[node_id] >= self.cfg.recovery_min_steps) and (float(new_trust) >= self.cfg.normal_enter_threshold):
                self.status[node_id] = NodeStatus.NORMAL
            else:
                self.status[node_id] = NodeStatus.RECOVERING

        self.last_trust[node_id] = float(new_trust)
        return self.status[node_id]

    def export_status_snapshot(self) -> Dict[str, str]:
        return {k: v.value for k, v in self.status.items()}
