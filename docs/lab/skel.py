import json
from datetime import datetime
from pathlib import Path
import random
import numpy as np
import matplotlib.pyplot as plt

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

    return {
        "avg_before": round(base, 4),
        "avg_after": round(max(0.0, min(1.0, base + delta)), 4),
        "min_before": round(max(0.0, base - random.uniform(0.05, 0.2)), 4),
        "max_before": round(min(1.0, base + random.uniform(0.05, 0.2)), 4),
        "min_after": round(max(0.0, base + delta - random.uniform(0.05, 0.2)), 4),
        "max_after": round(min(1.0, base + delta + random.uniform(0.05, 0.2)), 4),
        "notes": f"Auto-simulated placeholder dynamics for scenario '{sid}'",
        "type": scenario.get("type", "N/A"),
        "description": scenario.get("description", "")
    }

# ============================================================
# 3. GENERATE TRUST CHARTS
# ============================================================
def generate_trust_charts(results, output_dir="figures"):
    Path(output_dir).mkdir(exist_ok=True)
    for sid, r in results.items():
        t = np.linspace(0, 10, 20)  # placeholder time steps
        y = r["avg_before"] + (r["avg_after"] - r["avg_before"]) * (t / t[-1])
        plt.figure(figsize=(6,4))
        plt.plot(t, y, label=f"Avg Trust: {sid}")
        plt.title(f"Trust Propagation - Scenario {sid}")
        plt.xlabel("Time Steps")
        plt.ylabel("Trust Value")
        plt.ylim(0, 1)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        file_path = Path(output_dir) / f"{sid}_trust.png"
        plt.savefig(file_path)
        plt.close()

# ============================================================
# 4. GENERATE CASCADE TEXT CHARTS
# ============================================================
def generate_cascade_text(results, output_dir="figures"):
    Path(output_dir).mkdir(exist_ok=True)
    for sid, r in results.items():
        cascade_depth = int(random.randint(1,5))  # placeholder
        text_lines = ["Root Node"]
        for i in range(1, cascade_depth+1):
            text_lines.append("  " * i + f"-> Node {i}")
        file_path = Path(output_dir) / f"{sid}_cascade.txt"
        Path(file_path).write_text("\n".join(text_lines), encoding="utf-8")

# ============================================================
# 5. GENERATE REPORTS (Markdown + JSON)
# ============================================================
def generate_reports(results, md_path="aggregated_report_final.md", json_path="aggregated_results.json", figures_dir="figures"):
    # JSON output
    Path(json_path).write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Markdown output
    lines = []
    append = lines.append
    append("# Gravit Simulation — Aggregated Report with Visuals")
    append(f"Date: {datetime.utcnow().strftime('%Y-%m-%d')}\n")

    append("## 1. Summary Table\n")
    append("| Scenario ID | Type | Description | Avg Before | Avg After | Notes | Graph |")
    append("|-------------|------|-------------|------------|-----------|-------|-------|")

    for sid, r in results.items():
        graph_path = Path(figures_dir) / f"{sid}_trust.png"
        graph_md = f"![{sid}]({graph_path.as_posix()})" if graph_path.exists() else "N/A"
        append(f"| {sid} | {r['type']} | {r['description']} | {r['avg_before']} | {r['avg_after']} | {r['notes']} | {graph_md} |")

    append("\n## 2. Scenario Details\n")
    for sid, r in results.items():
        append(f"### Scenario ID: {sid}")
        append(f"**Type:** {r['type']}")
        append(f"**Description:** {r['description']}\n")
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

        # Insert Trust Graph
        trust_graph_path = Path(figures_dir) / f"{sid}_trust.png"
        if trust_graph_path.exists():
            append(f"![Trust Chart {sid}]({trust_graph_path.as_posix()})\n")

        # Insert Cascade Chart
        cascade_txt_path = Path(figures_dir) / f"{sid}_cascade.txt"
        if cascade_txt_path.exists():
            cascade_text = cascade_txt_path.read_text()
            append("```\n" + cascade_text + "\n```\n")

    Path(md_path).write_text("\n".join(lines), encoding="utf-8")
    print(f"Final Markdown report saved: {md_path}")
    print(f"Final JSON results saved: {json_path}")

# ============================================================
# 6. MAIN
# ============================================================
def main():
    scenarios = load_scenarios()
    results = {}
    for sc in scenarios:
        sid = sc["id"]
        results[sid] = run_simulation(sc)

    generate_trust_charts(results)
    generate_cascade_text(results)
    generate_reports(results)

if __name__ == "__main__":
    main()
