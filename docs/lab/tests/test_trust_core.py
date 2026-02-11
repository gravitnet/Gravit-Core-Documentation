# tests/test_trust_core.py
import math
import trust_core as core

def test_compute_trust_linear_bounds():
    weights = {'p':0.25,'s':0.35,'r':0.2,'m':0.2}
    # perfect evidence
    p, s, r, m, decay = 1.0, 1.0, 1.0, 1.0, 1.0
    t = core.compute_trust_linear(p, s, r, m, decay, weights)
    assert 0.0 <= t <= 1.0
    assert abs(t - 1.0) < 1e-9

    # zero evidence
    p, s, r, m, decay = 0.0, 0.0, 0.0, 0.0, 1.0
    t = core.compute_trust_linear(p, s, r, m, decay, weights)
    assert 0.0 <= t <= 1.0
    assert abs(t - 0.0) < 1e-9

def test_compute_decay():
    # age 0 -> decay ~= 1
    decay0 = core.compute_decay({'age_seconds': 0}, gamma=0.001)
    assert decay0 > 0.999
    # large age -> decay < 1
    decay1 = core.compute_decay({'age_seconds': 10000}, gamma=0.001)
    assert 0.0 < decay1 < 1.0

def test_provenance_examples():
    ev_good = {"signature_valid": True, "merkle_valid": True}
    p_good = core.compute_provenance(ev_good)
    assert 0.0 <= p_good <= 1.0
    assert p_good > 0.9

    ev_bad = {"signature_valid": False, "merkle_valid": False}
    p_bad = core.compute_provenance(ev_bad)
    assert p_bad == 0.0

def test_semantic_consistency_penalty():
    ev = {"semantic_similarity": 0.9, "contradiction_penalty": 0.2}
    s = core.compute_semantic_consistency("X", ev)
    assert 0.0 <= s <= 1.0
    assert abs(s - 0.7) < 1e-6
