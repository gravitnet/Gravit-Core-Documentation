# Gravit Trust Lab — Developer Guide

**Version:** 0.2
**Last Updated:** February 16, 2026

## Overview

The Gravit Trust Lab is a research and simulation environment for testing trust propagation, policy enforcement, and attack resistance in distributed systems. It provides tools for:

- **Trust computation and propagation** across network graphs
- **Attack scenario simulation** (Sybil, contradiction floods, trust drops)
- **Policy-based quarantine and recovery** mechanisms
- **Audit logging** for trust state transitions
- **REST API** for programmatic trust computation
- **Automated reporting** for simulation results

---

## Quick Start

### One-Command Run

The fastest way to run a simulation:

```bash
make lab
```

This will:
1. Create a Python virtual environment (`.venv`)
2. Install dependencies from `requirements.txt`
3. Run the default simulation (v0.2)
4. Output results to `out/` directory

Alternatively, run the shell script directly:

```bash
bash lab.sh
```

### Manual Setup

If you prefer manual control:

```bash
# 1. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run simulations
python demo_run.py
```

---

## File Structure

### Core Modules

| File | Purpose |
|------|---------|
| `trust_core.py` | Core trust computation algorithms (linear aggregation, decay, provenance) |
| `propagation.py` | Graph-based trust propagation using PageRank-style algorithms |
| `policy.py` | Node state management (NORMAL, WARNING, QUARANTINED, RECOVERING) |
| `audit.py` | JSONL-based audit logging for trust state changes |
| `api.py` | Flask REST API for trust computation and graph management |

### Simulation Components

| File | Purpose |
|------|---------|
| `simulation.py` | Main simulation engine (v0.2) with attack scenarios |
| `unified_simulation.py` | Alternative unified simulation framework |
| `unified_propagation.py` | Extended propagation with policy integration |
| `run_scenarios.py` | Scenario runner for batch execution |
| `demo_run.py` | Demo script for quick testing |

### Configuration & Data

| File | Purpose |
|------|---------|
| `scenarios.json` | Simulation scenario definitions (baseline, Sybil, contradiction) |
| `scenarios_catalog.json` | Extended scenario catalog |
| `requirements.txt` | Python dependencies (pytest, flask, matplotlib, numpy) |

### Reporting & Utilities

| File | Purpose |
|------|---------|
| `report_generator.py` | Aggregates simulation results into JSON and Markdown |
| `skel.py` | Skeleton code for custom simulations with plotting |

### Build & Automation

| File | Purpose |
|------|---------|
| `Makefile` | Build targets (`make lab`, `make clean`) |
| `lab.sh` | Shell script for automated environment setup and execution |

### Testing

| Directory/File | Purpose |
|----------------|---------|
| `tests/test_trust_core.py` | Unit tests for trust computation |
| `tests/test_propagation.py` | Tests for graph propagation |
| `tests/test_integration_api_sim.py` | Integration tests for API + simulation |

### Interactive Notebooks

| File | Purpose |
|------|---------|
| `simulation/colab.ipynb` | Google Colab notebook for interactive experiments |
| `simulation/interactive_simulation.ipynb` | Jupyter notebook for step-by-step simulation |
| `simulation/simulation_engine.py` | Engine for notebook-based simulations |

---

## Working with the Lab

### 1. Running Trust Simulations

#### Basic Simulation (v0.2)

```python
import simulation as s

# Run default simulation
result = s.run_full_simulation()
print(result['final_metrics'])
```

#### Custom Scenario Execution

```bash
# Run specific scenarios from scenarios.json
python simulation.py --scenarios scenarios.json

# Generate aggregated report
python report_generator.py
```

This produces:
- `simulation_output.jsonl` — Per-phase simulation logs
- `aggregated_results.json` — Structured results for all scenarios
- `aggregated_report.md` — Human-readable markdown report

#### Available Attack Scenarios

From `scenarios.json`:

| Scenario ID | Type | Description |
|-------------|------|-------------|
| `baseline_small` | baseline_run | Clean propagation on 50-node power-law graph |
| `sybil_small` | sybil_injection | 10 Sybil nodes attached to N0 |
| `sybil_medium` | sybil_injection | 30 Sybil nodes attached to N0 |
| `contradiction_small` | contradiction_flood | 5 attackers send negative edges to target |
| `rapid_drop_local` | rapid_trust_drop | Simulate sudden trust collapse |

### 2. Using the Trust API

#### Start the API Server

```bash
python api.py
# Server starts on http://127.0.0.1:5000
```

#### API Endpoints

**Compute Trust for Entity**
```bash
POST /trust/compute
Content-Type: application/json

{
  "target_id": "entity123",
  "evidence": {
    "signature_valid": true,
    "merkle_valid": true,
    "semantic_similarity": 0.85,
    "contradiction_penalty": 0.0,
    "age_seconds": 3600
  }
}
```

**Get Trust State**
```bash
GET /trust/state/entity123
```

**Propagate Trust Across Graph**
```bash
POST /trust/propagate
Content-Type: application/json

{
  "alpha": 0.85
}
```

**Add Graph Edge**
```bash
POST /graph/add_edge
Content-Type: application/json

{
  "src": "node1",
  "dst": "node2",
  "weight": 1.0,
  "sign": 1.0
}
```

**Get Audit Log**
```bash
GET /trust/audit
```

### 3. Working with Policy States

The policy module manages node states based on trust thresholds:

```python
import policy as pol

# Default thresholds
thresholds = {
    "accept": 0.85,     # Trust >= 0.85 → NORMAL
    "warn": 0.60,       # Trust < 0.60 → WARNING
    "quarantine": 0.30, # Trust < 0.30 → QUARANTINED
    "recover": 0.55,    # Recovery threshold
}

# Initialize policy states
policy_states = {
    "N0": pol.NodePolicyState("N0", state="NORMAL", last_trust=0.8),
    "N1": pol.NodePolicyState("N1", state="QUARANTINED", last_trust=0.25),
}

# Update states based on current trust
trust_state = {"N0": 0.75, "N1": 0.35}
policy_states = pol.policy_step(policy_states, trust_state, thresholds)
```

**State Transitions:**
- `NORMAL` → `WARNING` (trust drops below 0.60)
- `WARNING` → `QUARANTINED` (trust drops below 0.30)
- `QUARANTINED` → `RECOVERING` (trust rises above 0.55, stable for 3+ steps)
- `RECOVERING` → `NORMAL` (trust >= 0.60, stable for 3+ steps)

**State Effects:**
- `QUARANTINED` nodes have `out_weight = 0.0` (no influence)
- `RECOVERING` nodes have `out_weight = 0.2` (limited influence)
- `WARNING` nodes have `out_weight = 0.7` (reduced influence)

### 4. Trust Propagation

```python
import propagation as prop

# Create trust graph
graph = prop.TrustGraph()
graph.add_edge("N0", "N1", weight=1.0, sign=1.0)
graph.add_edge("N1", "N2", weight=0.8, sign=1.0)

# Initial trust values
initial_trust = {"N0": 1.0, "N1": 0.5, "N2": 0.5}

# Propagate trust (PageRank-style)
updated_trust = prop.propagate_trust(
    graph,
    initial_trust,
    alpha=0.85,  # Damping factor
    iterations=10
)

print(updated_trust)
```

### 5. Audit Logging

```python
import audit

# Append audit record
record = {
    "record_id": "uuid-1234",
    "target_id": "entity123",
    "previous_trust": 0.7,
    "new_trust": 0.85,
    "evidence": {...},
    "timestamp": "2026-02-16T12:00:00Z"
}
audit.append_audit(record)

# Load recent audit entries
recent = audit.load_audit(limit=50)
for entry in recent:
    print(entry['target_id'], entry['new_trust'])
```

### 6. Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_trust_core.py -v

# Run with coverage
pytest --cov=. --cov-report=html
```

### 7. Custom Simulations

Use `skel.py` as a template for custom simulations:

```python
import skel

# Load scenarios
scenarios = skel.load_scenarios("scenarios.json")

# Run simulations
results = {}
for scenario in scenarios:
    results[scenario['id']] = skel.run_simulation(scenario)

# Generate plots
skel.generate_trust_charts(results, output_dir="figures")

# Generate reports
skel.generate_reports(scenarios, results)
```

---

## Output Files

After running simulations, you'll find:

| File | Description |
|------|-------------|
| `simulation_output.jsonl` | JSONL log of every simulation phase |
| `audit_log.jsonl` | Trust state change audit trail |
| `aggregated_results.json` | Structured JSON summary of all scenarios |
| `aggregated_report.md` | Human-readable markdown report |
| `out/` | Directory with simulation artifacts |

---

## Configuration

### Environment Variables

```bash
# Python interpreter (default: python3)
export PYTHON=python3.11

# Virtual environment directory (default: .venv)
export VENV_DIR=".venv"

# Output directory (default: out)
export OUT_DIR="results"
```

### Simulation Parameters

Edit `scenarios.json` to customize:

```json
{
  "id": "custom_scenario",
  "type": "sybil_injection",
  "description": "Custom Sybil attack test",
  "params": {
    "seed": 777,
    "graph": {
      "type": "powerlaw",
      "num_nodes": 100,
      "m": 4
    },
    "alpha": 0.85,
    "n_sybil": 20,
    "connect_to": "N0"
  }
}
```

### Trust Algorithm Weights

In `trust_core.py`, modify the linear trust aggregator:

```python
weights = {
    "p": 0.25,  # Provenance
    "s": 0.35,  # Semantic consistency
    "r": 0.20,  # Reputation
    "m": 0.20   # Model confidence
}
```

---

## Development Workflow

### 1. Research & Experimentation
- Use Jupyter notebooks in `simulation/` for interactive exploration
- Modify `demo_run.py` for quick iterations

### 2. Algorithm Development
- Edit core modules (`trust_core.py`, `propagation.py`, `policy.py`)
- Add unit tests in `tests/`
- Run `pytest` to verify changes

### 3. Scenario Testing
- Add scenarios to `scenarios.json`
- Run `python simulation.py --scenarios scenarios.json`
- Review `aggregated_report.md`

### 4. Integration Testing
- Start API with `python api.py`
- Run integration tests: `pytest tests/test_integration_api_sim.py`

### 5. Production Deployment
- Use `Makefile` for clean builds
- Monitor `audit_log.jsonl` for trust events
- Integrate with external systems via REST API

---

## Architecture Notes

### Trust Computation Flow

```
Evidence Input → Provenance + Semantic + Reputation + Model Confidence
                      ↓
                Linear Aggregation (weighted sum)
                      ↓
                  Decay Function
                      ↓
              Updated Trust Score
                      ↓
                  Audit Log
```

### Propagation & Policy Loop

```
Initial Trust State → Trust Propagation (PageRank)
                             ↓
                      Policy Evaluation
                             ↓
                  State Transitions (NORMAL/WARNING/QUARANTINED/RECOVERING)
                             ↓
                  Apply out_weight Adjustments
                             ↓
                      Repeat N iterations
```

### Attack Scenario Execution

```
Load Scenario Config → Build Graph → Initialize Trust
                             ↓
                      Run Attack Phase
                             ↓
                  Propagate + Policy Step
                             ↓
                      Run Recovery Phase
                             ↓
                  Collect Metrics → Generate Report
```

---

## References

- **Core Documentation:** `docs/core/`
- **Continuum Specs:** `docs/continuum/`
- **Data Models:** `docs/core/10_data_models_overview.md`
- **Interaction Controller:** `docs/core/07_interaction_controller.md`
