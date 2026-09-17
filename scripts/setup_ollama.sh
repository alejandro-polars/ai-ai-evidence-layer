#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${1:-evidence-layer}"
GGUF_PATH="${2:-./export/unsloth.Q4_K_M.gguf}"
MODELFILE="${3:-./deployment/Modelfile}"

# --- Check dependencies ---
if ! command -v ollama &>/dev/null; then
  echo "Error: ollama is not installed. Install from https://ollama.com"
  exit 1
fi

if [ ! -f "$GGUF_PATH" ]; then
  echo "Error: GGUF not found at $GGUF_PATH"
  echo "Run 'python scripts/export_gguf.py --quant q4_k_m' first."
  exit 1
fi

if [ ! -f "$MODELFILE" ]; then
  echo "Error: Modelfile not found at $MODELFILE"
  exit 1
fi

# --- Ensure ollama server is running ---
if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
  echo "Starting ollama server..."
  ollama serve >/dev/null 2>&1 &
  sleep 3
fi

# --- Patch FROM line to point to the real GGUF path ---
TMP_MODELFILE=$(mktemp)
trap 'rm -f "$TMP_MODELFILE"' EXIT

sed "s|^FROM .*|FROM $GGUF_PATH|" "$MODELFILE" >"$TMP_MODELFILE"

# --- Create the model in Ollama ---
echo "Creating model '$MODEL_NAME' from $GGUF_PATH..."
ollama create "$MODEL_NAME" -f "$TMP_MODELFILE"

# --- Smoke test ---
echo ""
echo "Running smoke test..."
RESPONSE=$(ollama run "$MODEL_NAME" "2024-01-15 09:30 OPEN EURUSD 1.0 lot
2024-01-15 14:00 CLOSE EURUSD -850 USD
daily_loss_limit: 500 USD" 2>&1)

echo "Response:"
echo "$RESPONSE"
echo ""

# --- Validate JSON ---
if echo "$RESPONSE" | python3 -c "import sys, json; json.load(sys.stdin)" 2>/dev/null; then
  echo "Smoke test passed: valid JSON output."
else
  echo "Warning: output is not valid JSON. Check system_prompt and Modelfile."
fi

echo ""
echo "Done. Run with: ollama run $MODEL_NAME"
