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

# run one cell; on failure wipe the partial dir and retry once
# (results.jsonl is append-only, so a rerun must start from a fresh dir)
cell() { # $1 model  $2 extra-args  $3 out-dir  $4... run_pilot args
  out=$3
  for attempt in 1 2; do
    if [ -d "$out" ] && [ "$attempt" = 2 ]; then rm -rf "$out"; fi
    if env PYTHONPATH=. .venv/bin/python scripts/run_pilot.py \
        --model "$1" --memory-llm $MEMLLM $2 \
        --domain "$4" --difficulty "$5" --out "$out"; then
      return 0
    fi
    echo "!!! cell $out failed (attempt $attempt)"
  done
  return 1
}

for dom in d1 d2 d3; do
  for tier in easy medium hard; do
    echo "=== phasec mini-3seed $dom-$tier $(date)"
    cell gpt-4.1-mini "--seeds 3" "runs/phasec/mini3-$dom-$tier" "$dom" "$tier"
  done
done

for dom in d1 d2 d3; do
  for tier in easy medium hard; do
    echo "=== phasec gpt41 $dom-$tier $(date)"
    cell gpt-4.1 "" "runs/phasec/gpt41-$dom-$tier" "$dom" "$tier"
  done
done
echo "=== phasec api done $(date)"
