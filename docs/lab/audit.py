# audit.py
from typing import Dict, Any, List
import json
import os

AUDIT_DB_FILE = "audit_log.jsonl"

def append_audit(record: Dict[str, Any]):
    # append as JSON lines
    line = json.dumps(record, ensure_ascii=False)
    with open(AUDIT_DB_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def load_audit(limit: int = 100) -> List[Dict[str, Any]]:
    if not os.path.exists(AUDIT_DB_FILE):
        return []
    lines = []
    with open(AUDIT_DB_FILE, "r", encoding="utf-8") as f:
        for i, l in enumerate(f):
            if i >= limit:
                break
            try:
                lines.append(json.loads(l))
            except Exception:
                continue
    return lines
