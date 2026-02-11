# report_generator.py
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SIM_LOG = "simulation_output.jsonl"
OUT_JSON = "aggregated_results.json"
OUT_MD = "aggregated_report.md"


def utc_today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


@dataclass
class ScenarioAggregate:
    scenario_id: str
    scenario_type: str
    description: str
    seed: Optional[int]
    alpha: Optional[float]
    graph: Optional[dict]

    before_metrics: Optional[dict] = None
    after_metrics: Optional[dict] = None
    notes: List[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "type": self.scenario_type,
            "description": self.description,
            "seed": self.seed,
            "alpha": self.alpha,
            "graph": self.graph,
            "metrics_before": self.before_metrics,
            "metrics_after": self.after_metrics,
            "notes": self.notes or [],
        }


def read_jsonl(path: str) -> List[dict]:
    p = Path(path)
    if not p.exists():
        return []
    items: List[dict] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            items.append(json.loads(line))
        except Exception:
            continue
    return items


def write_json(path: str, data: Any) -> None:
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_text(path: str, text: str) -> None:
    Path(path).write_text(text, encoding="utf-8")


def md_table_row(cols: List[str]) -> str:
    return "| " + " | ".join(cols) + " |"


def fmt_metrics(m: Optional[dict]) -> Tuple[str, str, str, str]:
    if not m:
        return ("-", "-", "-", "-")
    return (
        f"{m.get('avg_trust', '-'):.4f}" if isinstance(m.get("avg_trust"), (int, float)) else "-",
        f"{m.get('min_trust', '-'):.4f}" if isinstance(m.get("min_trust"), (int, float)) else "-",
        f"{m.get('max_trust', '-'):.4f}" if isinstance(m.get("max_trust"), (int, float)) else "-",
        f"{m.get('stddev_trust', '-'):.4f}" if isinstance(m.get("stddev_trust"), (int, float)) else "-",
    )


def generate_markdown(aggs: List[ScenarioAggregate]) -> str:
    lines: List[str] = []
    a = lines.append

    a("# Gravit Trust Lab — Simulation Report (MVP v0.1)")
    a(f"Date (UTC): {utc_today()}")
    a("")
    a("## Summary")
    a("")
    a("| Scenario ID | Type | Avg Before | Avg After | Std Before | Std After | Notes |")
    a("|---|---|---:|---:|---:|---:|---|")

    for agg in aggs:
        avg_b, _min_b, _max_b, std_b = fmt_metrics(agg.before_metrics)
        avg_a, _min_a, _max_a, std_a = fmt_metrics(agg.after_metrics)
        notes = "; ".join((agg.notes or [])[:2])  # keep short in table
        a(md_table_row([agg.scenario_id, agg.scenario_type, avg_b, avg_a, std_b, std_a, notes]))

    a("")
    a("## Scenario Details")
    a("")
    for agg in aggs:
        a(f"### {agg.scenario_id}")
        a(f"- Type: `{agg.scenario_type}`")
        a(f"- Description: {agg.description}")
        if agg.seed is not None:
            a(f"- Seed: {agg.seed}")
        if agg.alpha is not None:
            a(f"- Alpha: {agg.alpha}")
        if agg.graph:
            a(f"- Graph: `{json.dumps(agg.graph, ensure_ascii=False)}`")
        a("")
        a("**Before metrics**")
        a(f"```json\n{json.dumps(agg.before_metrics or {}, indent=2, ensure_ascii=False)}\n```")
        a("**After metrics**")
        a(f"```json\n{json.dumps(agg.after_metrics or {}, indent=2, ensure_ascii=False)}\n```")
        if agg.notes:
            a("**Notes**")
            for n in agg.notes:
                a(f"- {n}")
        a("")

    return "\n".join(lines)


def build_aggregates(events: List[dict]) -> List[ScenarioAggregate]:
    # We expect simulator to emit:
    #  - {"kind":"scenario_start", "scenario": {...}}
    #  - {"kind":"scenario_baseline", "metrics": {...}}
    #  - {"kind":"scenario_after", "metrics": {...}}
    #  - {"kind":"scenario_note", "note": "..."}
    #  - {"kind":"scenario_end", ...}
    aggs: Dict[str, ScenarioAggregate] = {}

    for e in events:
        kind = e.get("kind")
        if kind == "scenario_start":
            sc = e.get("scenario", {})
            sid = sc.get("id", "unknown")
            aggs[sid] = ScenarioAggregate(
                scenario_id=sid,
                scenario_type=sc.get("type", "unknown"),
                description=sc.get("description", ""),
                seed=(sc.get("params", {}) or {}).get("seed"),
                alpha=(sc.get("params", {}) or {}).get("alpha"),
                graph=(sc.get("params", {}) or {}).get("graph"),
                notes=[],
            )
        elif kind in ("scenario_baseline", "scenario_after"):
            sid = e.get("scenario_id")
            if sid not in aggs:
                # tolerate missing start
                aggs[sid] = ScenarioAggregate(
                    scenario_id=sid,
                    scenario_type="unknown",
                    description="",
                    seed=None,
                    alpha=None,
                    graph=None,
                    notes=[],
                )
            if kind == "scenario_baseline":
                aggs[sid].before_metrics = e.get("metrics")
            else:
                aggs[sid].after_metrics = e.get("metrics")
        elif kind == "scenario_note":
            sid = e.get("scenario_id")
            if sid in aggs:
                aggs[sid].notes.append(str(e.get("note", "")).strip())

    # deterministic ordering by id
    return [aggs[k] for k in sorted(aggs.keys())]


def main() -> None:
    events = read_jsonl(SIM_LOG)
    aggs = build_aggregates(events)

    out = [a.to_dict() for a in aggs]
    write_json(OUT_JSON, out)

    md = generate_markdown(aggs)
    write_text(OUT_MD, md)

    print(f"[OK] Wrote {OUT_JSON}")
    print(f"[OK] Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
