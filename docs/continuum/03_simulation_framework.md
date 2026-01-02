# Simulation Framework

This document outlines the simulation engine used to stress-test the Trust Engine.

## `simulation.py` Structure

The simulation engine is designed to run specific scenarios (Baseline, Sybil, Contradiction) and generate reports.

### Key Components

1.  **Graph Generator**: Creates random or power-law graphs to simulate network topology.
2.  **Scenarios**:
    *   `sybil_injection`: Adds hostile nodes trying to inflate trust.
    *   `contradiction_flood`: Adds nodes submitting contradictory semantic evidence.
    *   `rapid_trust_drop`: Simulates sudden key compromise or failure.
3.  **Metrics**: Collects average trust, min/max, standard deviation, and node counts.

## Python Logic Skeleton

```python
def run_full_simulation(seed:int=42, scenario_sequence=None, verbose=True):
    # Setup graph & store
    graph = generate_powerlaw_graph(50, m=3, seed=seed)
    store = initialize_store(graph)

    # Initial propagation
    T0 = prop.propagate_trust(graph, store["trust_state"], alpha=0.85)
    store["trust_state"].update(T0)

    # Run scenario sequence
    for sn, params in scenario_sequence:
        if sn == "sybil_injection":
            run_sybil_attack(...)
        elif sn == "contradiction_flood":
            run_flood_attack(...)
        # ... log results ...

    return final_metrics
```

## Output
The simulation produces:
*   `simulation_output.jsonl`: Detailed step-by-step metrics.
*   `audit_log.jsonl`: Trust update records.
