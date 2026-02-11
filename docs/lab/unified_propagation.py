#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified Propagation v0.2
========================

Purpose:
- Run multiple simulation experiments (parameter sweeps)
- Aggregate final metrics + peak metrics
- Output a single CSV summary for quick comparison

This file is designed to work with unified_simulation.py v0.2.

Usage examples:
  python unified_propagation.py --runs 30 --out out_prop_v02
  python unified_propagation.py --sweep beta=0.12,0.18,0.24 temp=0.8,1.0,1.2 --runs 10

Outputs:
- summary.csv (one row per experiment)
- per-experiment subfolders containing simulation outputs (optional)
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import random
import sys
import time
from dataclasses import asdict
from typing import Dict, List, Any, Tuple, Optional

# Local import: expects unified_simulation.py in same directory or PYTHONPATH
try:
    from unified_simulation import SimulationConfig, Simulation, QuarantinePolicy
except ImportError as e:
    print("[error] Could not import unified_simulation. Make sure unified_simulation.py is рядом.", file=sys.stderr)
    raise


# ----------------------------
# Helpers
# ----------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def now_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def parse_sweep_args(pairs: List[str]) -> Dict[str, List[float]]:
    """
    Parse --sweep arguments like: ["beta=0.1,0.2", "temp=0.8,1.0"]
    Returns dict: {"beta":[0.1,0.2], "temp":[0.8,1.0]}
    """
    sweep: Dict[str, List[float]] = {}
    for item in pairs:
        if "=" not in item:
            raise ValueError(f"Invalid sweep item '{item}'. Expected key=val1,val2,...")
        k, v = item.split("=", 1)
        k = k.strip()
        vals = [float(x.strip()) for x in v.split(",") if x.strip() != ""]
        if not vals:
            raise ValueError(f"No values parsed for '{k}' in '{item}'")
        sweep[k] = vals
    return sweep

def product_dict(sweep: Dict[str, List[float]]) -> List[Dict[str, float]]:
    keys = list(sweep.keys())
    combos = []
    for vals in itertools.product(*(sweep[k] for k in keys)):
        combos.append({k: float(v) for k, v in zip(keys, vals)})
    return combos

def read_last_metrics_row(metrics_csv: str) -> Dict[str, Any]:
    """
    Read last row of metrics CSV produced by unified_simulation.py
    Columns:
      ts, step, healthy, infected, quarantined, recovered,
      infected_ratio, quarantine_mode, avg_pressure, avg_trust
    """
    last = None
    with open(metrics_csv, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        for line in f:
            if line.strip():
                last = line.strip().split(",")
    if last is None:
        raise RuntimeError(f"Empty metrics file: {metrics_csv}")

    row = dict(zip(header, last))
    # cast numeric
    for k in ["step","healthy","infected","quarantined","recovered"]:
        row[k] = int(float(row[k]))
    for k in ["infected_ratio","avg_pressure","avg_trust"]:
        row[k] = float(row[k])
    row["quarantine_mode"] = bool(int(float(row["quarantine_mode"])))
    return row

def read_peak_infected(metrics_csv: str) -> Tuple[int, float]:
    """
    Returns (step_of_peak, peak_infected_ratio)
    """
    peak_ratio = -1.0
    peak_step = -1
    with open(metrics_csv, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        idx_step = header.index("step")
        idx_ratio = header.index("infected_ratio")
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split(",")
            step = int(float(parts[idx_step]))
            ratio = float(parts[idx_ratio])
            if ratio > peak_ratio:
                peak_ratio = ratio
                peak_step = step
    return peak_step, peak_ratio


# ----------------------------
# Experiment runner
# ----------------------------

def build_cfg(
    base_cfg: Optional[SimulationConfig],
    seed: int,
    out_dir: str,
    run_name: str,
    overrides: Dict[str, float],
    quarantine_enabled: Optional[bool],
) -> SimulationConfig:
    cfg = base_cfg if base_cfg is not None else SimulationConfig()
    # make a shallow copy by reconstructing from dict (safe enough here)
    cfg = SimulationConfig(**{k: v for k, v in asdict(cfg).items() if k != "quarantine"})
    # reattach quarantine
    qdict = asdict(base_cfg.quarantine) if base_cfg is not None else asdict(SimulationConfig().quarantine)
    cfg.quarantine = QuarantinePolicy(**qdict)

    cfg.seed = seed
    cfg.out_dir = out_dir
    cfg.run_name = run_name

    # Apply numeric overrides
    # Supported keys: beta, temp, init_inf, edge_prob, q_local, q_release, q_global, q_factor
    for k, v in overrides.items():
        if k in ("beta", "base_transmission"):
            cfg.base_transmission = float(v)
        elif k in ("temp", "temperature"):
            cfg.temperature = float(v)
        elif k in ("init_inf", "initial_infected_frac"):
            cfg.initial_infected_frac = float(v)
        elif k in ("edge_prob",):
            cfg.edge_prob = float(v)
        elif k in ("n_agents", "n"):
            cfg.n_agents = int(v)
        elif k in ("q_local",):
            cfg.quarantine.local_threshold = float(v)
        elif k in ("q_release",):
            cfg.quarantine.local_release_threshold = float(v)
        elif k in ("q_global",):
            cfg.quarantine.global_infected_trigger = float(v)
        elif k in ("q_factor",):
            cfg.quarantine.network_quarantine_factor = float(v)
        else:
            raise ValueError(f"Unknown override key: {k}")

    if quarantine_enabled is not None:
        cfg.quarantine.enabled = quarantine_enabled

    return cfg


def run_experiment(cfg: SimulationConfig) -> Dict[str, Any]:
    sim = Simulation(cfg)
    sim.run()

    last = read_last_metrics_row(sim.metrics_path)
    peak_step, peak_ratio = read_peak_infected(sim.metrics_path)

    result = {
        "ts": now_ts(),
        "run_name": cfg.run_name,
        "out_dir": cfg.out_dir,
        "seed": cfg.seed,
        "steps": cfg.steps,
        "n_agents": cfg.n_agents,
        "edge_prob": cfg.edge_prob,
        "init_inf": cfg.initial_infected_frac,
        "beta": cfg.base_transmission,
        "temp": cfg.temperature,
        "quarantine_enabled": bool(cfg.quarantine.enabled),
        "q_local": cfg.quarantine.local_threshold,
        "q_release": cfg.quarantine.local_release_threshold,
        "q_global": cfg.quarantine.global_infected_trigger,
        "q_factor": cfg.quarantine.network_quarantine_factor,

        "final_healthy": last["healthy"],
        "final_infected": last["infected"],
        "final_quarantined": last["quarantined"],
        "final_recovered": last["recovered"],
        "final_infected_ratio": last["infected_ratio"],
        "final_quarantine_mode": last["quarantine_mode"],
        "final_avg_pressure": last["avg_pressure"],
        "final_avg_trust": last["avg_trust"],

        "peak_infected_step": peak_step,
        "peak_infected_ratio": peak_ratio,
    }
    return result


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified Propagation v0.2 (sweeps + aggregation)")

    p.add_argument("--config", type=str, default=None, help="Optional JSON base config (same shape as unified_simulation).")
    p.add_argument("--out", type=str, default="out_prop_v02", help="Propagation output directory.")
    p.add_argument("--runs", type=int, default=10, help="Runs per parameter setting.")
    p.add_argument("--steps", type=int, default=None, help="Override steps for all runs.")
    p.add_argument("--n", type=int, default=None, help="Override number of agents for all runs.")
    p.add_argument("--edge-prob", type=float, default=None, help="Override edge probability for all runs.")
    p.add_argument("--init-inf", type=float, default=None, help="Override initial infected fraction for all runs.")
    p.add_argument("--beta", type=float, default=None, help="Override base transmission for all runs.")
    p.add_argument("--temp", type=float, default=None, help="Override temperature for all runs.")

    p.add_argument("--q", action="store_true", help="Force quarantine enabled.")
    p.add_argument("--no-q", action="store_true", help="Force quarantine disabled.")

    p.add_argument("--sweep", nargs="*", default=[], help="Parameter sweep pairs like beta=0.1,0.2 temp=0.8,1.0 q_factor=0.4,0.6")
    p.add_argument("--keep-per-run", action="store_true", help="Keep per-run subfolders; otherwise only summary.csv is kept.")
    p.add_argument("--master-seed", type=int, default=12345, help="Master seed to generate run seeds deterministically.")

    return p.parse_args()


def load_base_cfg(path: str) -> SimulationConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Use unified_simulation loader semantics by building SimulationConfig directly
    # (Keeping this lightweight; if you prefer, you can reuse load_config from unified_simulation)
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


def write_summary_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    ensure_dir(os.path.dirname(path) or ".")
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> int:
    args = parse_args()
    ensure_dir(args.out)

    base_cfg = load_base_cfg(args.config) if args.config else SimulationConfig()

    # global overrides
    global_overrides: Dict[str, float] = {}
    if args.steps is not None:
        base_cfg.steps = args.steps
    if args.n is not None:
        base_cfg.n_agents = args.n
    if args.edge_prob is not None:
        base_cfg.edge_prob = args.edge_prob
    if args.init_inf is not None:
        base_cfg.initial_infected_frac = args.init_inf
    if args.beta is not None:
        base_cfg.base_transmission = args.beta
    if args.temp is not None:
        base_cfg.temperature = args.temp

    quarantine_enabled: Optional[bool] = None
    if args.q and args.no_q:
        print("[error] Choose either --q or --no-q, not both.", file=sys.stderr)
        return 2
    if args.q:
        quarantine_enabled = True
    if args.no_q:
        quarantine_enabled = False

    sweep = parse_sweep_args(args.sweep) if args.sweep else {}
    combos = product_dict(sweep) if sweep else [{}]

    master_rng = random.Random(args.master_seed)

    all_rows: List[Dict[str, Any]] = []
    summary_path = os.path.join(args.out, "summary.csv")

    for ci, combo in enumerate(combos):
        for ri in range(args.runs):
            seed = master_rng.randint(1, 2_000_000_000)
            exp_name = f"exp{ci:03d}_run{ri:03d}"
            exp_out_dir = os.path.join(args.out, exp_name) if args.keep_per_run else os.path.join(args.out, "_tmp")

            # build cfg for this run
            cfg = build_cfg(
                base_cfg=base_cfg,
                seed=seed,
                out_dir=exp_out_dir,
                run_name=exp_name,
                overrides=combo,
                quarantine_enabled=quarantine_enabled,
            )

            # If not keeping per-run folders, keep outputs in _tmp but rewrite each run.
            ensure_dir(cfg.out_dir)

            row = run_experiment(cfg)
            # add combo columns explicitly (for readability)
            for k, v in combo.items():
                row[f"sweep_{k}"] = v
            all_rows.append(row)

            # cleanup tmp if requested
            if not args.keep_per_run:
                # remove tmp directory content lightly (do not delete directory itself)
                # to avoid OS-specific issues; just keep last run artifacts overwritten.
                pass

            print(f"[ok] {exp_name} done. final_infected_ratio={row['final_infected_ratio']:.3f} peak={row['peak_infected_ratio']:.3f}")

    write_summary_csv(summary_path, all_rows)
    print(f"[ok] summary saved: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
