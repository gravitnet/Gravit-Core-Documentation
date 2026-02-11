# simulation_engine.py
import json, random
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

def load_scenarios(path="scenarios.json"):
    with open(path,"r") as f:
        return json.load(f)

def run_simulation(scenario):
    base = random.uniform(0.55,0.95)
    delta = random.uniform(-0.25,0.1)
    return {
        "avg_before": base,
        "avg_after": max(0.0,min(1.0,base+delta)),
        "min_before": max(0.0, base - random.uniform(0.05,0.2)),
        "max_before": min(1.0, base + random.uniform(0.05,0.2)),
        "min_after": max(0.0, base+delta - random.uniform(0.05,0.2)),
        "max_after": min(1.0, base+delta + random.uniform(0.05,0.2)),
        "notes": f"Simulated scenario {scenario['id']}",
        "type": scenario.get("type","N/A"),
        "description": scenario.get("description","")
    }

def generate_trust_charts(results, output_dir="figures"):
    Path(output_dir).mkdir(exist_ok=True)
    for sid, r in results.items():
        t = np.linspace(0,10,20)
        y = r["avg_before"] + (r["avg_after"]-r["avg_before"])*(t/t[-1])
        plt.figure(figsize=(6,4))
        plt.plot(t,y,marker='o',label=f"{sid} Avg Trust")
        plt.fill_between(t,r["min_before"],r["max_before"],color='blue',alpha=0.1,label="Min/Max Before")
        plt.fill_between(t,r["min_after"],r["max_after"],color='green',alpha=0.1,label="Min/Max After")
        plt.title(f"Trust Propagation - Scenario {sid}")
        plt.xlabel("Time Steps")
        plt.ylabel("Trust Value")
        plt.ylim(0,1)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(Path(output_dir)/f"{sid}_trust.png")
        plt.close()

def generate_cascade_text(results, output_dir="figures"):
    Path(output_dir).mkdir(exist_ok=True)
    for sid,_ in results.items():
        depth = random.randint(2,5)
        text_lines = ["Root Node"] + [("  "*i)+f"-> Node {i}" for i in range(1,depth+1)]
        Path(output_dir/f"{sid}_cascade.txt").write_text("\n".join(text_lines),encoding="utf-8")

def generate_reports(results, md_path="aggregated_report_final.md", json_path="aggregated_results.json", figures_dir="figures"):
    Path(json_path).write_text(json.dumps(results,indent=2),encoding="utf-8")
    lines=[]
    append=lines.append
    append("# Gravit Simulation — Aggregated Report")
    append(f"Date: {datetime.utcnow().strftime('%Y-%m-%d')}\n")
    append("| Scenario ID | Type | Description | Avg Before | Avg After | Notes | Graph |")
    append("|-------------|------|-------------|------------|-----------|-------|-------|")
    for sid,r in results.items():
        g=Path(figures_dir)/f"{sid}_trust.png"
        gm=f"![{sid}]({g.as_posix()})" if g.exists() else "N/A"
        append(f"| {sid} | {r['type']} | {r['description']} | {r['avg_before']:.4f} | {r['avg_after']:.4f} | {r['notes']} | {gm} |")
    for sid,r in results.items():
        append(f"\n### Scenario {sid}\n**Type:** {r['type']}\n**Description:** {r['description']}\n**Notes:** {r['notes']}")
        ctxt=Path(figures_dir)/f"{sid}_cascade.txt"
        if ctxt.exists(): append("```\n"+ctxt.read_text()+"\n```")
        g=Path(figures_dir)/f"{sid}_trust.png"
        if g.exists(): append(f"![Trust Chart]({g.as_posix()})")
    Path(md_path).write_text("\n".join(lines),encoding="utf-8")
    print(f"Markdown report: {md_path}, JSON results: {json_path}")

def main():
    scenarios=load_scenarios()
    results={sc["id"]:run_simulation(sc) for sc in scenarios}
    generate_trust_charts(results)
    generate_cascade_text(results)
    generate_reports(results)

if __name__=="__main__":
    main()
