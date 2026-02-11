#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified Simulation v0.2
=======================

What’s new in v0.2:
- Quarantine & Threshold Policy hooks (without full governance)
- Simulations show not only collapse, but also recovery / isolation behavior
- Deterministic seeding, reproducible runs
- JSON config + CLI overrides
- Outputs: metrics.csv + events.jsonl + snapshot.json

Model (high-level):
- Directed graph of agents with edge "trust" weights.
- Each agent has state: HEALTHY, INFECTED, QUARANTINED, RECOVERED.
- "Infection" can be interpreted as rumor/false signal/compromised node.
- Propagation probability depends on: source strength, target susceptibility,
  edge trust, and global "temperature".
- Policy hooks:
  - Threshold quarantine: if local infection pressure exceeds threshold,
    quarantine node (cuts outgoing edges, reduces incoming influence).
  - Network quarantine: when global infected ratio exceeds threshold, enable
    network-level quarantine mode (stronger gating).
  - Recovery: infected nodes can recover with rate; quarantined nodes can rejoin
    if conditions improve (hysteresis supported).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, List, Tuple, Optional, Any


# ----------------------------
# Core data structures
# ----------------------------

class State(str, Enum):
    HEALTHY = "healthy"
    INFECTED = "infected"
    QUARANTINED = "quarantined"
    RECOVERED = "recovered"


@dataclass
class Agent:
    id: str
    susceptibility: float = 0.5   # 0..1
    resilience: float = 0.5       # 0..1 (reduces infection)
    recovery_rate: float = 0.02   # per step
    state: State = State.HEALTHY
    days_infected: int = 0
    days_quarantined: int = 0


@dataclass
class Edge:
    src: str
    dst: str
    trust: float = 0.5            # 0..1


@dataclass
class QuarantinePolicy:
    enabled: bool = True

    # Local quarantine rule: quarantine node if its infection pressure >= local_threshold
    local_threshold: float = 0.65

    # To avoid oscillation: release quarantine only if pressure <= local_release_threshold
    local_release_threshold: float = 0.40

    # If global infected ratio exceeds this, activate stronger network quarantine mode
    global_infected_trigger: float = 0.25

    # When network quarantine mode active, multiply infection probability by factor (<1)
    network_quarantine_factor: float = 0.55

    # When node quarantined, reduce its outgoing influence and incoming exposure
    node_outgoing_factor: float = 0.10
    node_incoming_factor: float = 0.35


@dataclass
class SimulationConfig:
    seed: int = 42
    steps: int = 200
    dt: float = 1.0

    # Graph generation (if no input graph provided)
    n_agents: int = 80
    edge_prob: float = 0.06

    # Initial infection
    initial_infected_frac: float = 0.05

    # Infection dynamics
    base_transmission: float = 0.22     # baseline per-step probability scale
    temperature: float = 1.0            # global multiplier (>=0)
    infected_emission: float = 1.0      # how strongly infected agents emit pressure
    recovered_immunity: float = 0.85    # reduces susceptibility if recovered (0..1)
    reinfection_allowed: bool = False

    # Optional noise and drift
    trust_drift_std: float = 0.00       # per step gaussian drift on edge trust
    susceptibility_drift_std: float = 0.00

    # Quarantine policy
    quarantine: QuarantinePolicy = QuarantinePolicy()

    # Output
    out_dir: str = "out_sim_v02"
    run_name: str = "run"
    write_events: bool = True
    write_snapshot: bool = True

    # Optional: load graph from json (agents + edges)
    graph_path: Optional[str] = None


# ----------------------------
# Utilities
# ----------------------------

def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


def sigmoid(x: float) -> float:
    # stable sigmoid
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def now_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ----------------------------
# Graph + Simulation Engine
# ----------------------------

class TrustGraph:
    def __init__(self) -> None:
        self.agents: Dict[str, Agent] = {}
        self.edges_out: Dict[str, List[Edge]] = {}
        self.edges_in: Dict[str, List[Edge]] = {}

    def add_agent(self, a: Agent) -> None:
        self.agents[a.id] = a
        self.edges_out.setdefault(a.id, [])
        self.edges_in.setdefault(a.id, [])

    def add_edge(self, e: Edge) -> None:
        if e.src not in self.agents or e.dst not in self.agents:
            raise ValueError(f"Edge references unknown agent: {e}")
        self.edges_out[e.src].append(e)
        self.edges_in[e.dst].append(e)

    def agent_ids(self) -> List[str]:
        return list(self.agents.keys())


def generate_random_graph(cfg: SimulationConfig, rng: random.Random) -> TrustGraph:
    g = TrustGraph()
    for i in range(cfg.n_agents):
        aid = f"a{i:03d}"
        a = Agent(
            id=aid,
            susceptibility=clamp(rng.random() * 0.9 + 0.05),
            resilience=clamp(rng.random() * 0.9 + 0.05),
            recovery_rate=clamp(0.01 + rng.random() * 0.05, 0.005, 0.12),
        )
        g.add_agent(a)

    ids = g.agent_ids()
    for src in ids:
        for dst in ids:
            if src == dst:
                continue
            if rng.random() < cfg.edge_prob:
                g.add_edge(Edge(src=src, dst=dst, trust=clamp(rng.random() * 0.85 + 0.10)))
    return g


def load_graph_json(path: str) -> TrustGraph:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    g = TrustGraph()
    for a in data.get("agents", []):
        g.add_agent(Agent(
            id=str(a["id"]),
            susceptibility=float(a.get("susceptibility", 0.5)),
            resilience=float(a.get("resilience", 0.5)),
            recovery_rate=float(a.get("recovery_rate", 0.02)),
            state=State(str(a.get("state", State.HEALTHY))),
        ))
    for e in data.get("edges", []):
        g.add_edge(Edge(
            src=str(e["src"]),
            dst=str(e["dst"]),
            trust=float(e.get("trust", 0.5)),
        ))
    return g


def seed_initial_infections(g: TrustGraph, cfg: SimulationConfig, rng: random.Random) -> None:
    ids = g.agent_ids()
    rng.shuffle(ids)
    k = max(1, int(round(cfg.initial_infected_frac * len(ids))))
    for aid in ids[:k]:
        g.agents[aid].state = State.INFECTED
        g.agents[aid].days_infected = 0


@dataclass
class StepMetrics:
    step: int
    healthy: int
    infected: int
    quarantined: int
    recovered: int
    infected_ratio: float
    quarantine_mode: bool
    avg_pressure: float
    avg_trust: float


class Simulation:
    def __init__(self, cfg: SimulationConfig) -> None:
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)

        if cfg.graph_path:
            self.g = load_graph_json(cfg.graph_path)
        else:
            self.g = generate_random_graph(cfg, self.rng)

        seed_initial_infections(self.g, cfg, self.rng)

        self.quarantine_mode = False
        self._last_pressure: Dict[str, float] = {aid: 0.0 for aid in self.g.agent_ids()}

        # outputs
        ensure_dir(cfg.out_dir)
        self.metrics_path = os.path.join(cfg.out_dir, f"{cfg.run_name}_metrics.csv")
        self.events_path = os.path.join(cfg.out_dir, f"{cfg.run_name}_events.jsonl")
        self.snapshot_path = os.path.join(cfg.out_dir, f"{cfg.run_name}_snapshot.json")

        self._init_metrics_file()

    def _init_metrics_file(self) -> None:
        with open(self.metrics_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "ts", "step",
                "healthy", "infected", "quarantined", "recovered",
                "infected_ratio", "quarantine_mode",
                "avg_pressure", "avg_trust"
            ])

    def _write_event(self, obj: Dict[str, Any]) -> None:
        if not self.cfg.write_events:
            return
        obj = dict(obj)
        obj["ts"] = now_ts()
        with open(self.events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def _avg_trust(self) -> float:
        total = 0.0
        cnt = 0
        for src, edges in self.g.edges_out.items():
            for e in edges:
                total += e.trust
                cnt += 1
        return (total / cnt) if cnt else 0.0

    def _compute_infection_pressure(self) -> Dict[str, float]:
        """
        Infection pressure on node = sum over incoming edges:
        source_infected * emission * trust * modifiers
        """
        cfg = self.cfg
        pol = cfg.quarantine

        pressure: Dict[str, float] = {aid: 0.0 for aid in self.g.agent_ids()}
        for dst, edges in self.g.edges_in.items():
            dst_agent = self.g.agents[dst]

            incoming_factor = 1.0
            if pol.enabled and dst_agent.state == State.QUARANTINED:
                incoming_factor *= pol.node_incoming_factor

            p = 0.0
            for e in edges:
                src_agent = self.g.agents[e.src]
                src_factor = 0.0
                if src_agent.state == State.INFECTED:
                    src_factor = cfg.infected_emission
                # quarantined infected node emits less outward influence
                if pol.enabled and src_agent.state == State.QUARANTINED:
                    # If quarantined, treat as "isolated": zero emission
                    src_factor = 0.0

                # Outgoing reduction if infected but quarantined handled above
                # Additional outgoing factor if source is quarantined (already 0)
                contrib = src_factor * e.trust
                p += contrib

            pressure[dst] = clamp(p * incoming_factor, 0.0, 3.0)  # cap for stability
        return pressure

    def _should_quarantine_node(self, aid: str, pressure: float) -> bool:
        pol = self.cfg.quarantine
        a = self.g.agents[aid]
        if not pol.enabled:
            return False
        if a.state == State.QUARANTINED:
            # keep quarantine unless pressure is clearly low
            return pressure >= pol.local_release_threshold
        # trigger quarantine if pressure high and node not already recovered immune enough
        if a.state in (State.HEALTHY, State.INFECTED):
            return pressure >= pol.local_threshold
        return False

    def _update_quarantine_mode(self) -> None:
        pol = self.cfg.quarantine
        if not pol.enabled:
            self.quarantine_mode = False
            return
        infected = sum(1 for a in self.g.agents.values() if a.state == State.INFECTED)
        total = len(self.g.agents)
        ratio = infected / total if total else 0.0
        self.quarantine_mode = (ratio >= pol.global_infected_trigger)

    def _infection_probability(self, src: Agent, dst: Agent, trust: float, pressure_dst: float) -> float:
        """
        Probability that src infects dst in this step, given edge trust and dst pressure.
        Uses:
        - base_transmission * temperature
        - dst susceptibility * (1 - dst resilience)
        - trust
        - pressure shaping
        """
        cfg = self.cfg
        pol = cfg.quarantine

        if src.state != State.INFECTED:
            return 0.0

        # recovered immunity (if reinfection allowed)
        susceptibility = dst.susceptibility
        if dst.state == State.RECOVERED:
            susceptibility *= (1.0 - clamp(cfg.recovered_immunity, 0.0, 1.0))

        eff = cfg.base_transmission * max(cfg.temperature, 0.0)
        eff *= clamp(susceptibility, 0.0, 1.0)
        eff *= clamp(1.0 - dst.resilience, 0.0, 1.0)
        eff *= clamp(trust, 0.0, 1.0)

        # pressure shaping: more pressure -> higher probability (bounded)
        # map pressure 0..3 to multiplier ~0.6..1.6
        mult = 0.6 + 1.0 * sigmoid((pressure_dst - 0.9) * 2.0)
        eff *= mult

        # if quarantine mode active, reduce overall propagation
        if pol.enabled and self.quarantine_mode:
            eff *= pol.network_quarantine_factor

        # if dst quarantined, further reduce exposure (handled via incoming factor already),
        # but keep a small extra reduction
        if pol.enabled and dst.state == State.QUARANTINED:
            eff *= 0.5

        return clamp(eff, 0.0, 0.95)

    def _drift_parameters(self) -> None:
        cfg = self.cfg
        if cfg.susceptibility_drift_std > 0.0:
            for a in self.g.agents.values():
                a.susceptibility = clamp(a.susceptibility + self.rng.gauss(0.0, cfg.susceptibility_drift_std), 0.0, 1.0)

        if cfg.trust_drift_std > 0.0:
            for src, edges in self.g.edges_out.items():
                for e in edges:
                    e.trust = clamp(e.trust + self.rng.gauss(0.0, cfg.trust_drift_std), 0.0, 1.0)

    def step(self, t: int) -> StepMetrics:
        cfg = self.cfg
        pol = cfg.quarantine

        # (A) update global quarantine mode based on current infected ratio
        self._update_quarantine_mode()

        # (B) compute pressure based on current states
        pressure = self._compute_infection_pressure()

        # (C) quarantine decisions (hook)
        if pol.enabled:
            for aid, p in pressure.items():
                a = self.g.agents[aid]
                should_q = self._should_quarantine_node(aid, p)
                if should_q and a.state != State.QUARANTINED:
                    prev = a.state
                    a.state = State.QUARANTINED
                    a.days_quarantined = 0
                    self._write_event({"event": "quarantine_on", "step": t, "agent": aid, "prev": prev, "pressure": p})
                elif (not should_q) and a.state == State.QUARANTINED:
                    # release quarantine -> return to HEALTHY or RECOVERED? We'll default to HEALTHY
                    a.state = State.HEALTHY
                    a.days_quarantined = 0
                    self._write_event({"event": "quarantine_off", "step": t, "agent": aid, "pressure": p})

        # (D) infection attempts
        newly_infected: List[str] = []
        for src, edges in self.g.edges_out.items():
            src_agent = self.g.agents[src]
            if src_agent.state != State.INFECTED:
                continue
            for e in edges:
                dst_agent = self.g.agents[e.dst]

                # skip if dst is infected already
                if dst_agent.state == State.INFECTED:
                    continue

                # if reinfection not allowed, skip recovered
                if (not cfg.reinfection_allowed) and dst_agent.state == State.RECOVERED:
                    continue

                # quarantine: if src quarantined, it should not emit (we treat as no edges)
                if pol.enabled and src_agent.state == State.QUARANTINED:
                    continue

                # compute infection probability
                p = self._infection_probability(src_agent, dst_agent, e.trust, pressure.get(e.dst, 0.0))
                if p <= 0.0:
                    continue
                if self.rng.random() < p:
                    # if dst quarantined, we still allow infection but with reduced probability already applied
                    newly_infected.append(dst_agent.id)

        # apply infections
        for aid in newly_infected:
            a = self.g.agents[aid]
            prev = a.state
            a.state = State.INFECTED
            a.days_infected = 0
            self._write_event({"event": "infect", "step": t, "agent": aid, "prev": prev})

        # (E) recovery
        for a in self.g.agents.values():
            if a.state == State.INFECTED:
                a.days_infected += 1
                # recover probability based on recovery_rate + resilience
                pr = clamp(a.recovery_rate * (0.6 + 0.8 * a.resilience), 0.0, 0.6)
                if self.rng.random() < pr:
                    a.state = State.RECOVERED
                    a.days_infected = 0
                    self._write_event({"event": "recover", "step": t, "agent": a.id})
            elif a.state == State.QUARANTINED:
                a.days_quarantined += 1

        # (F) optional drifts
        self._drift_parameters()

        # (G) metrics
        counts = {s: 0 for s in State}
        for a in self.g.agents.values():
            counts[a.state] += 1

        total = len(self.g.agents)
        infected_ratio = (counts[State.INFECTED] / total) if total else 0.0
        avg_pressure = sum(pressure.values()) / total if total else 0.0
        avg_trust = self._avg_trust()

        m = StepMetrics(
            step=t,
            healthy=counts[State.HEALTHY],
            infected=counts[State.INFECTED],
            quarantined=counts[State.QUARANTINED],
            recovered=counts[State.RECOVERED],
            infected_ratio=infected_ratio,
            quarantine_mode=self.quarantine_mode,
            avg_pressure=avg_pressure,
            avg_trust=avg_trust,
        )

        # persist metrics row
        with open(self.metrics_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                now_ts(), m.step,
                m.healthy, m.infected, m.quarantined, m.recovered,
                f"{m.infected_ratio:.6f}", int(m.quarantine_mode),
                f"{m.avg_pressure:.6f}", f"{m.avg_trust:.6f}"
            ])

        self._last_pressure = pressure
        return m

    def run(self) -> None:
        self._write_event({"event": "run_start", "cfg": asdict(self.cfg)})
        for t in range(self.cfg.steps):
            self.step(t)
        self._write_event({"event": "run_end", "steps": self.cfg.steps})

        if self.cfg.write_snapshot:
            self.write_snapshot()

    def write_snapshot(self) -> None:
        snap = {
            "version": "v0.2",
            "ts": now_ts(),
            "cfg": asdict(self.cfg),
            "agents": [
                {
                    "id": a.id,
                    "state": a.state.value,
                    "susceptibility": a.susceptibility,
                    "resilience": a.resilience,
                    "recovery_rate": a.recovery_rate,
                    "days_infected": a.days_infected,
                    "days_quarantined": a.days_quarantined,
                    "last_pressure": self._last_pressure.get(a.id, 0.0),
                }
                for a in self.g.agents.values()
            ],
            "edges": [
                {"src": e.src, "dst": e.dst, "trust": e.trust}
                for edges in self.g.edges_out.values()
                for e in edges
            ],
        }
        with open(self.snapshot_path, "w", encoding="utf-8") as f:
            json.dump(snap, f, ensure_ascii=False, indent=2)


# ----------------------------
# Config loading + CLI
# ----------------------------

def load_config(path: str) -> SimulationConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Quarantine policy (nested)
    q = data.get("quarantine", {})
    qp = QuarantinePolicy(
        enabled=bool(q.get("enabled", True)),
        local_threshold=float(q.get("local_threshold", 0.65)),
        local_release_threshold=float(q.get("local_release_threshold", 0.40)),
        global_infected_trigger=float(q.get("global_infected_trigger", 0.25)),
        network_quarantine_factor=float(q.get("network_quarantine_factor", 0.55)),
        node_outgoing_factor=float(q.get("node_outgoing_factor", 0.10)),
        node_incoming_factor=float(q.get("node_incoming_factor", 0.35)),
    )

    cfg = SimulationConfig(
        seed=int(data.get("seed", 42)),
        steps=int(data.get("steps", 200)),
        dt=float(data.get("dt", 1.0)),
        n_agents=int(data.get("n_agents", 80)),
        edge_prob=float(data.get("edge_prob", 0.06)),
        initial_infected_frac=float(data.get("initial_infected_frac", 0.05)),
        base_transmission=float(data.get("base_transmission", 0.22)),
        temperature=float(data.get("temperature", 1.0)),
        infected_emission=float(data.get("infected_emission", 1.0)),
        recovered_immunity=float(data.get("recovered_immunity", 0.85)),
        reinfection_allowed=bool(data.get("reinfection_allowed", False)),
        trust_drift_std=float(data.get("trust_drift_std", 0.0)),
        susceptibility_drift_std=float(data.get("susceptibility_drift_std", 0.0)),
        quarantine=qp,
        out_dir=str(data.get("out_dir", "out_sim_v02")),
        run_name=str(data.get("run_name", "run")),
        write_events=bool(data.get("write_events", True)),
        write_snapshot=bool(data.get("write_snapshot", True)),
        graph_path=data.get("graph_path", None),
    )
    return cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified Simulation v0.2 (trust/contagion + quarantine hooks)")
    p.add_argument("--config", type=str, default=None, help="Path to JSON config (optional).")
    p.add_argument("--out", type=str, default=None, help="Output directory override.")
    p.add_argument("--run", type=str, default=None, help="Run name override.")
    p.add_argument("--seed", type=int, default=None, help="Random seed override.")
    p.add_argument("--steps", type=int, default=None, help="Steps override.")
    p.add_argument("--n", type=int, default=None, help="Number of agents override.")
    p.add_argument("--edge-prob", type=float, default=None, help="Edge probability override.")
    p.add_argument("--init-inf", type=float, default=None, help="Initial infected fraction override.")
    p.add_argument("--beta", type=float, default=None, help="Base transmission override.")
    p.add_argument("--temp", type=float, default=None, help="Temperature override.")
    p.add_argument("--q", action="store_true", help="Enable quarantine (force on).")
    p.add_argument("--no-q", action="store_true", help="Disable quarantine.")
    return p.parse_args()


def apply_overrides(cfg: SimulationConfig, args: argparse.Namespace) -> SimulationConfig:
    if args.out is not None:
        cfg.out_dir = args.out
    if args.run is not None:
        cfg.run_name = args.run
    if args.seed is not None:
        cfg.seed = args.seed
    if args.steps is not None:
        cfg.steps = args.steps
    if args.n is not None:
        cfg.n_agents = args.n
    if args.edge_prob is not None:
        cfg.edge_prob = args.edge_prob
    if args.init_inf is not None:
        cfg.initial_infected_frac = args.init_inf
    if args.beta is not None:
        cfg.base_transmission = args.beta
    if args.temp is not None:
        cfg.temperature = args.temp
    if args.q:
        cfg.quarantine.enabled = True
    if args.no_q:
        cfg.quarantine.enabled = False
    return cfg


def main() -> int:
    args = parse_args()

    if args.config:
        cfg = load_config(args.config)
    else:
        cfg = SimulationConfig()

    cfg = apply_overrides(cfg, args)

    sim = Simulation(cfg)
    sim.run()

    print(f"[ok] v0.2 finished. metrics: {sim.metrics_path}")
    if cfg.write_events:
        print(f"[ok] events: {sim.events_path}")
    if cfg.write_snapshot:
        print(f"[ok] snapshot: {sim.snapshot_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
