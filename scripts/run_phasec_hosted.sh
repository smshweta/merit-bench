#!/bin/sh
# Phase C (open-weight slice, hosted): gpt-oss:20b on a hosted provider,
# 1 seed x 9-cell grid. Same open weights as the local Ollama run but
# ~100x faster (~2-3 h for the grid vs weeks on a laptop) for a few
# dollars. Memory side stays gpt-4.1-mini via the OpenAI API so memory
# implementations are identical across all Phase C agent models.
#
# Default provider is Groq: put GROQ_API_KEY in .env. To use another
# provider, override HOSTED_MODEL with any litellm model string, e.g.
#   HOSTED_MODEL=openrouter/openai/gpt-oss-20b  (needs OPENROUTER_API_KEY)
set -eu
cd "$(dirname "$0")/.."
[ -f .env ] && { set -a; . ./.env; set +a; }

HOSTED_MODEL=${HOSTED_MODEL:-groq/openai/gpt-oss-20b}
case "$HOSTED_MODEL" in
  groq/*) [ -n "${GROQ_API_KEY:-}" ] || { echo "GROQ_API_KEY not set (add it to .env)"; exit 1; } ;;
  openrouter/*) [ -n "${OPENROUTER_API_KEY:-}" ] || { echo "OPENROUTER_API_KEY not set (add it to .env)"; exit 1; } ;;
esac

# same per-cell retry + stderr capture as run_phasec_api.sh; stderr file
# lives outside the out dir so the attempt-1 traceback survives the wipe
cell() { # $1 out-dir  $2 domain  $3 tier
  out=$1
  for attempt in 1 2; do
    if [ -d "$out" ] && [ "$attempt" = 2 ]; then rm -rf "$out"; fi
    echo "--- attempt $attempt $(date)" >> "$out.stderr.log"
    if env PYTHONPATH=. PYTHONUNBUFFERED=1 .venv/bin/python scripts/run_pilot.py \
        --model "$HOSTED_MODEL" --memory-llm gpt-4.1-mini \
        --domain "$2" --difficulty "$3" --out "$out" 2>> "$out.stderr.log"; then
      return 0
    fi
    echo "!!! cell $out failed (attempt $attempt), see $out.stderr.log"
  done
  return 1
}

for dom in d1 d2 d3; do
  for tier in easy medium hard; do
    echo "=== phasec gptoss-hosted $dom-$tier $(date)"
    cell "runs/phasec/gptoss-$dom-$tier" "$dom" "$tier"
  done
done
echo "=== phasec hosted done $(date)"
