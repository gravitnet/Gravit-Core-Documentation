# Reporting Templates

## Aggregated Report Structure

### 1. Summary Table
| Scenario ID | Type | Avg Trust Before | Avg Trust After | Notes |
|---|---|---|---|---|
| baseline | control | 0.65 | 0.65 | Stable |
| sybil_med | attack | 0.65 | 0.52 | Significant drop |

### 2. Detailed Observations
*   **Propagation Dynamics**: How fast did the trust change?
*   **Locality**: Was the effect contained or cascading?
*   **Anomalies**: Any unexpected oscillations?

### 3. Diagnosis & Mitigations
*   **Root Cause**: e.g., "High edge density of Sybils".
*   **Mitigation**: "Increase identity-age weight".
