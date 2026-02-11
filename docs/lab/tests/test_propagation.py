# tests/test_propagation.py
import propagation as prop

def make_simple_graph():
    g = prop.TrustGraph()
    # A -> B (support), B -> C (support), C -> A (support)
    g.add_edge("A", "B", weight=1.0, sign=1.0)
    g.add_edge("B", "C", weight=1.0, sign=1.0)
    g.add_edge("C", "A", weight=1.0, sign=1.0)
    return g

def test_propagate_converges_simple():
    g = make_simple_graph()
    # initial trust biases
    initial = {"A": 0.9, "B": 0.5, "C": 0.2}
    T = prop.propagate_trust(g, initial, alpha=0.85, max_iter=200)
    assert set(T.keys()) >= {"A","B","C"}
    # values in [0,1]
    for v in ["A","B","C"]:
        assert 0.0 <= T[v] <= 1.0

def test_propagate_negative_relations():
    g = prop.TrustGraph()
    # A supports B, X contradicts B with negative sign
    g.add_edge("A", "B", weight=1.0, sign=1.0)
    g.add_edge("X", "B", weight=1.0, sign=-1.0)
    initial = {"A": 0.9, "X": 0.9, "B": 0.5}
    T = prop.propagate_trust(g, initial, alpha=0.85, max_iter=200)
    # B should be influenced by both; ensure still bounded
    assert 0.0 <= T["B"] <= 1.0
