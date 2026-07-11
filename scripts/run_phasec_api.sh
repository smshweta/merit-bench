#!/bin/sh
# Phase C (API slice): seed robustness + frontier model, real memory
# implementations throughout. Memory side is pinned to gpt-4.1-mini for
# every agent model so the agent model is the only varying factor.
#   part 1: gpt-4.1-mini x 3 seeds x 9-cell grid  (~$9,  ~7.5 h)
#   part 2: gpt-4.1      x 1 seed  x 9-cell grid  (~$15, ~3 h)
# Requires OPENAI_API_KEY (e.g. via .env).
set -eu
cd "$(dirname "$0")/.."
[ -f .env ] && { set -a; . ./.env; set +a; }

MEMLLM=gpt-4.1-mini

for dom in d1 d2 d3; do
  for tier in easy medium hard; do
    echo "=== phasec mini-3seed $dom-$tier $(date)"
    env PYTHONPATH=. .venv/bin/python scripts/run_pilot.py \
      --model gpt-4.1-mini --memory-llm $MEMLLM --seeds 3 \
      --domain "$dom" --difficulty "$tier" \
      --out "runs/phasec/mini3-$dom-$tier"
  done
done

for dom in d1 d2 d3; do
  for tier in easy medium hard; do
    echo "=== phasec gpt41 $dom-$tier $(date)"
    env PYTHONPATH=. .venv/bin/python scripts/run_pilot.py \
      --model gpt-4.1 --memory-llm $MEMLLM \
      --domain "$dom" --difficulty "$tier" \
      --out "runs/phasec/gpt41-$dom-$tier"
  done
done
echo "=== phasec api done $(date)"
