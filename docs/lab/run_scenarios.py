import json
from datetime import datetime
from pathlib import Path
import random

# ============================================================
# 1. LOAD SCENARIOS
# ============================================================
def load_scenarios(path="scenarios.json"):
    with open(path, "r") as f:
        return json.load(f)

# ============================================================
# 2. SIMULATION ENGINE (STUB)
# ============================================================
def run_simulation(scenario):
    """
    Placeholder simulation.
    Replace with real trust propagation logic.
    """
    sid = scenario["id"]
    base = random.uniform(0.55, 0.95)
    delta = random.uniform(-0.25, 0.1)

    result = {
        "avg_before": round(base, 4),
        "avg_after": round(max(0.0, min(1.0, base + delta)), 4),
        "min_before": round(max(0.0, base - random.uniform(0.05, 0.2)), 4),
        "max_before": round(min(1.0, base + random.uniform(0.05, 0.2)), 4),
        "min_after": round(max(0.0, base + delta - random.uniform(0.05, 0.2)), 4),
        "max_after": round(min(1.0, base + delta + random.uniform(0.05, 0.2)), 4),
        "notes": f"Auto-simulated placeholder dynamics for scenario '{sid}'"
    }
    return result

# ============================================================
# 3. GENERATE REPORT
# ============================================================
def generate_reports(scenarios, results, md_path="aggregated_report.md", json_path="aggregated_results.json"):
    lines = []
    append = lines.append

    # -----------------------------
    # Markdown Report Header
    # -----------------------------
    append("# Gravit Simulation — Aggregated Report")
    append(f"Date: {datetime.utcnow().strftime('%Y-%m-%d')}")
    append("Generated automatically from simulation skeleton.\n")

    # -----------------------------
    # Summary Table
    # -----------------------------
    append("## 1. Summary Table of Scenarios\n")
    header = (
        "| Scenario ID | Type | Description | Avg Before | Avg After | Min/Max Before | Min/Max After | Notes |\n"
        "|-------------|------|-------------|------------|-----------|----------------|---------------|-------|"
    )
    append(header)

    for sc in scenarios:
        r = results[sc["id"]]
        append(
            f"| {sc['id']} | {sc['type']} | {sc['description']} | "
            f"{r['avg_before']} | {r['avg_after']} | "
            f"{r['min_before']}/{r['max_before']} | "
            f"{r['min_after']}/{r['max_after']} | "
            f"{r['notes']} |"
        )

    append("\n## 2. Scenario Details\n")

    for sc in scenarios:
        sid = sc["id"]
        r = results[sid]

        append(f"### Scenario ID: {sid}")
        append(f"**Type:** {sc['type']}")
        append(f"**Description:** {sc['description']}\n")

        append("**Parameters:**")
        for k, v in sc.get("params", {}).items():
            append(f"- {k}: {v}")
        append("")

        append("**Pre-Run Metrics:**")
        append(f"- Avg trust: {r['avg_before']}")
        append(f"- Min trust: {r['min_before']}")
        append(f"- Max trust: {r['max_before']}\n")

        append("**Post-Run Metrics:**")
        append(f"- Avg trust: {r['avg_after']}")
        append(f"- Min trust: {r['min_after']}")
        append(f"- Max trust: {r['max_after']}\n")

        append("**Observations:**")
        append(f"- {r['notes']}\n")

    # -----------------------------
    # Write Markdown
    # -----------------------------
    Path(md_path).write_text("\n".join(lines), encoding="utf-8")
    print(f"Markdown report saved to: {md_path}")

    # -----------------------------
    # Write JSON
    # -----------------------------
    Path(json_path).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"JSON results saved to: {json_path}")

# ============================================================
# 4. MAIN
# ============================================================
def main():
    scenarios = load_scenarios()

    results = {}
    for sc in scenarios:
        sid = sc["id"]
        print(f"Running scenario → {sid}")
        results[sid] = run_simulation(sc)

    generate_reports(scenarios, results)


if __name__ == "__main__":
    main()
