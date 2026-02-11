# tests/test_integration_api_sim.py
# These tests exercise internal functions without running Flask server
import trust_core as core
import propagation as prop

def test_update_and_audit_flow(tmp_path):
    # simulate a minimal store and evidence
    store = {"reputation": {"A": 0.6}, "model_confidence": {"A": 0.7}, "trust_state": {}}
    evidence = {
        "signature_valid": True,
        "merkle_valid": True,
        "semantic_similarity": 0.8,
        "contradiction_penalty": 0.0,
        "age_seconds": 10
    }
    p = core.compute_provenance(evidence)
    s = core.compute_semantic_consistency("A", evidence)
    r = core.get_reputation(store, "A")
    m = core.get_model_confidence(store, "A")
    decay = core.compute_decay(evidence)
    weights = {'p':0.25,'s':0.35,'r':0.2,'m':0.2}
    new_score = core.compute_trust_linear(p, s, r, m, decay, weights)
    assert 0.0 <= new_score <= 1.0
