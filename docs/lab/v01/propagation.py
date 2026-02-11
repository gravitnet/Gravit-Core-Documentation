# trust_core.py
from typing import Dict, Any
import math
import time
import uuid

# Simple data classes as dict-like structures
def now_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

# --- Linear trust aggregator ---
def compute_trust_linear(p: float, s: float, r: float, m: float, decay: float, weights: Dict[str, float]) -> float:
    """
    Simple interpretable linear aggregator.
    p,s,r,m in [0,1], decay in (0,1], weights sum to 1.
    """
    base = weights['p']*p + weights['s']*s + weights['r']*r + weights['m']*m
    return max(0.0, min(1.0, decay * base))


# --- Provenance computation (stub / example) ---
def compute_provenance(evidence: Dict[str, Any]) -> float:
    """
    Example provenance computation based on presence/validity of signatures and merkle.
    evidence may contain 'signature_valid' (bool), 'merkle_valid' (bool)
    """
    a_sig = 0.6
    a_merkle = 0.4
    sig_ok = 1.0 if evidence.get("signature_valid", False) else 0.0
    merkle_ok = 1.0 if evidence.get("merkle_valid", False) else 0.0
    return a_sig * sig_ok + a_merkle * merkle_ok


# --- Semantic consistency computation (stub) ---
def compute_semantic_consistency(target_id: str, evidence: Dict[str, Any]) -> float:
    """
    Placeholder for semantic similarity. Evidence may include 'semantic_similarity' float [0,1]
    and 'contradiction_penalty' float [0,1].
    """
    sim = evidence.get("semantic_similarity", 0.5)
    penalty = evidence.get("contradiction_penalty", 0.0)
    value = sim - penalty
    return max(0.0, min(1.0, value))


# --- Reputation (stub, should link to history) ---
def get_reputation(store: Dict[str, Any], target_id: str) -> float:
    """
    Return reputation from a simple store: store['reputation'][target_id] or baseline 0.5
    """
    return float(store.get("reputation", {}).get(target_id, 0.5))


# --- Model confidence (stub: pulled from QPlatform) ---
def get_model_confidence(store: Dict[str, Any], target_id: str) -> float:
    """
    Placeholder: QPlatform may supply a confidence [0,1]
    """
    return float(store.get("model_confidence", {}).get(target_id, 0.5))


# --- Decay function ---
def compute_decay(evidence: Dict[str, Any], gamma: float = 0.001) -> float:
    """
    Exponential decay based on age_seconds (provided in evidence).
    If no age provided, return 1.0.
    """
    age = evidence.get("age_seconds", 0)
    return math.exp(-gamma * age)


# --- Audit record creation ---
def make_audit_record(target_id: str, previous: float, new: float, evidence: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    record = {
        "record_id": str(uuid.uuid4()),
        "target_id": target_id,
        "previous_trust": float(previous),
        "new_trust": float(new),
        "evidence": evidence,
        "meta": meta,
        "timestamp": now_ts()
    }
    return record
