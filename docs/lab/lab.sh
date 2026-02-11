#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
OUT_DIR="${OUT_DIR:-out}"

echo "[Lab] Using: $PYTHON"
echo "[Lab] venv:  $VENV_DIR"
echo "[Lab] out:   $OUT_DIR"

# 1) venv
if [ ! -d "$VENV_DIR" ]; then
  "$PYTHON" -m venv "$VENV_DIR"
fi

# 2) activate
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# 3) deps (только если нужен matplotlib и т.п.; для v0.2 quarantine/recovery можно и без)
python -m pip install --upgrade pip >/dev/null
if [ -f requirements.txt ]; then
  pip install -r requirements.txt >/dev/null
fi

# 4) prepare out
mkdir -p "$OUT_DIR"

# 5) run simulation v0.2
echo "[Lab] Running simulation v0.2..."
python -c "import simulation as s; r=s.run_simulation(); print('[Lab] Done:', r['final_metrics'])"

# 6) collect artifacts
for f in simulation_output.jsonl audit_log.jsonl; do
  if [ -f "$f" ]; then
    cp "$f" "$OUT_DIR/$f"
  fi
done

echo "[Lab] Artifacts:"
ls -lah "$OUT_DIR" || true
echo "[Lab] ✅ One-command run complete."
