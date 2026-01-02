# Attack Scenarios Catalog

This catalog defines the parameters for repeatable tests.

## scenarios.json

```json
[
  {
    "id": "baseline_small",
    "description": "Baseline run on powerlaw graph (50 nodes), no attacks.",
    "type": "baseline_run",
    "params": {}
  },
  {
    "id": "sybil_medium",
    "description": "Medium Sybil injection: 30 sybils attacking node N0.",
    "type": "sybil_injection",
    "params": {"n_sybil": 30, "connect_to": "N0", "seed": 102}
  },
  {
    "id": "contradiction_medium",
    "description": "Contradiction flood: 15 attackers against N1.",
    "type": "contradiction_flood",
    "params": {"target": "N1", "n_attackers": 15, "seed": 202}
  },
  {
    "id": "rapid_drop_local",
    "description": "Rapid trust drop on a single node (N2).",
    "type": "rapid_trust_drop",
    "params": {"target": "N2", "shock": 0.6}
  }
]
```
