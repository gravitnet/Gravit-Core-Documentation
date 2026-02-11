# demo_run.py
import subprocess
import sys
from pathlib import Path

def run(cmd):
    print("\n>>>", " ".join(cmd))
    p = subprocess.run(cmd, capture_output=True, text=True)
    print(p.stdout)
    if p.returncode != 0:
        print(p.stderr)
        raise SystemExit(p.returncode)

def main():
    # Clean old outputs
    for f in ["simulation_output.jsonl", "aggregated_results.json", "aggregated_report.md"]:
        p = Path(f)
        if p.exists():
            p.unlink()

    # Run simulations from scenarios.json (writes simulation_output.jsonl)
    run([sys.executable, "simulation.py", "--scenarios", "scenarios.json"])

    # Build aggregated report
    run([sys.executable, "report_generator.py"])

    print("\n[READY] Gravit Trust Lab — MVP v0.1 outputs:")
    print(" - simulation_output.jsonl")
    print(" - aggregated_results.json")
    print(" - aggregated_report.md")

if __name__ == "__main__":
    main()
