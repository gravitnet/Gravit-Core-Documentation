# api.py
from flask import Flask, request, jsonify
from typing import Dict, Any
import trust_core as core
import propagation as prop
import audit as audit_mod

# Simple in-memory store for demo
STORE: Dict[str, Any] = {
    "reputation": {},            # entity_id -> float
    "model_confidence": {},      # entity_id -> float
    "trust_state": {},          # entity_id -> float
}

# Simple trust graph (in-memory)
TG = prop.TrustGraph()

app = Flask(__name__)

# --- Helper: compute and update trust ---
def compute_and_update(target_id: str, evidence: Dict[str, Any]) -> Dict[str, Any]:
    prev = STORE["trust_state"].get(target_id, 0.5)
    p = core.compute_provenance(evidence)
    s = core.compute_semantic_consistency(target_id, evidence)
    r = core.get_reputation(STORE, target_id)
    m = core.get_model_confidence(STORE, target_id)
    decay = core.compute_decay(evidence)
    weights = evidence.get("weights", {"p":0.25,"s":0.35,"r":0.2,"m":0.2})
    new_score = core.compute_trust_linear(p, s, r, m, decay, weights)
    # store
    STORE["trust_state"][target_id] = new_score
    record = core.make_audit_record(target_id, prev, new_score, evidence, {"algorithm":"linear_v1","weights":weights})
    audit_mod.append_audit(record)
    return {"target_id":target_id, "trust":new_score, "audit_id": record["record_id"]}


@app.route("/trust/compute", methods=["POST"])
def trust_compute():
    data = request.get_json(force=True)
    if not data or "target_id" not in data or "evidence" not in data:
        return jsonify({"error":"target_id and evidence required"}), 400
    res = compute_and_update(data["target_id"], data["evidence"])
    return jsonify(res)


@app.route("/trust/state/<target_id>", methods=["GET"])
def trust_state(target_id):
    score = STORE["trust_state"].get(target_id, 0.5)
    rep = STORE["reputation"].get(target_id, 0.5)
    mc = STORE["model_confidence"].get(target_id, 0.5)
    return jsonify({"target_id": target_id, "trust": score, "reputation":rep, "model_confidence":mc})


@app.route("/trust/propagate", methods=["POST"])
def trust_propagate():
    data = request.get_json(force=True) or {}
    alpha = float(data.get("alpha", 0.85))
    # initial trust from STORE
    initial = STORE.get("trust_state", {}).copy()
    T = prop.propagate_trust(TG, initial, alpha=alpha)
    # write back
    STORE["trust_state"].update(T)
    return jsonify({"status":"propagated","nodes":len(T)})


@app.route("/trust/audit", methods=["GET"])
def trust_audit():
    recs = audit_mod.load_audit(limit=200)
    return jsonify(recs)


@app.route("/graph/add_edge", methods=["POST"])
def graph_add_edge():
    data = request.get_json(force=True)
    src = data.get("src")
    dst = data.get("dst")
    w = float(data.get("weight", 1.0))
    s = float(data.get("sign", 1.0))
    if not src or not dst:
        return jsonify({"error":"src and dst required"}), 400
    TG.add_edge(src, dst, weight=w, sign=s)
    return jsonify({"status":"ok"})


if __name__ == "__main__":
    # for quick local demo only
    app.run(host="127.0.0.1", port=5000, debug=True)
