# Trust Model Theory

## 1. Objective
*   Ensure reliable trust assessment for entities (agents, events, messages).
*   Propagate trust across the multi-layer architecture (L1 → L2 → L3).
*   Provide formal verification and auditability.
*   Support dynamic updates (real-time, decay, attack countermeasures).

## 2. Terminology
*   **Entity**: Agent/Service (L1).
*   **TrustScore(entity, context, time)**: Real value in [0, 1].
*   **TrustVector**: Components [integrity, provenance, semantic_consistency, recency, reputation].

## 3. High-Level Formula
For entity `e` in context `c` at time `t`:

$T_{e,c}(t) = D(t) \cdot f(P_{e,c}, S_{e,c}, R_e, M_{e,c})$

Where:
*   **P (Provenance)**: Crypto signatures, Merkle proofs.
*   **S (Semantic Consistency)**: Cosine similarity of meaning vectors - contradictions.
*   **R (Reputation)**: Historical EWMA (Exponentially Weighted Moving Average).
*   **M (Model Confidence)**: Predictive confidence from Quantum Platform.
*   **D (Decay)**: Time decay factor $e^{-\gamma \Delta t}$.

## 4. Aggregation Logic
**Weighted Linear Aggregator (V1):**
$T = D \cdot (w_P P + w_S S + w_R R + w_M M)$

## 5. Trust Propagation Algorithms
**Iterative Propagation (PageRank-like):**
Trust flows through the graph of relations (L1 interactions, L2 semantic links).
$T_v^{(k+1)} = \alpha \sum_{u} \frac{w_{uv} s_{uv}}{Z_u} T_u^{(k)} + (1-\alpha) B_v$
*   $\alpha$: Damping factor.
*   $s_{uv}$: Semantic sign (supports vs contradicts).
*   $B_v$: Local bias (base check).
