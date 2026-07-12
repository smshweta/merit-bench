#!/bin/sh
# Phase C (third-model slice, hosted): 1 seed x 9-cell grid on a hosted
# provider. Memory side stays gpt-4.1-mini via the OpenAI API so memory
# implementations are identical across all Phase C agent models.
#
# Default: Claude Haiku 4.5 (cheapest cross-vendor model; pinned ID),
# needs ANTHROPIC_API_KEY in .env. ~2-3 h, ~$12 agent-side.
# Other providers via HOSTED_MODEL + SLICE, e.g. the open-weight slice:
#   HOSTED_MODEL=groq/openai/gpt-oss-20b SLICE=gptoss  (needs GROQ_API_KEY)
#   HOSTED_MODEL=openrouter/openai/gpt-oss-20b SLICE=gptoss  (OPENROUTER_API_KEY)
set -eu
cd "$(dirname "$0")/.."
[ -f .env ] && { set -a; . ./.env; set +a; }

HOSTED_MODEL=${HOSTED_MODEL:-anthropic/claude-haiku-4-5-20251001}
SLICE=${SLICE:-haiku45}
case "$HOSTED_MODEL" in
  anthropic/*) [ -n "${ANTHROPIC_API_KEY:-}" ] || { echo "ANTHROPIC_API_KEY not set (add it to .env)"; exit 1; } ;;
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
    echo "=== phasec $SLICE $dom-$tier $(date)"
    cell "runs/phasec/$SLICE-$dom-$tier" "$dom" "$tier"
  done
done
echo "=== phasec hosted ($SLICE) done $(date)"
