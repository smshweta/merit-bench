#!/bin/sh
# Phase B: rerun the full grid with real memory implementations
# (embedding RAG, LLM summarization, LLM extraction) via --memory-llm.
# Mirrors the pilot structure: 9 clean (domain x difficulty) cells +
# the D1 corruption sweep (which carries its own clean arm for pairing).
# Requires OPENAI_API_KEY (e.g. via .env). ~4,500 episodes, ~$6, ~7 h.
set -eu
cd "$(dirname "$0")/.."
[ -f .env ] && { set -a; . ./.env; set +a; }

MODEL=gpt-4.1-mini
RUN="env PYTHONPATH=. .venv/bin/python scripts/run_pilot.py \
  --model $MODEL --memory-llm $MODEL"

for dom in d1 d2 d3; do
  for tier in easy medium hard; do
    echo "=== phaseb $dom-$tier $(date)"
    $RUN --domain "$dom" --difficulty "$tier" --out "runs/phaseb/$dom-$tier"
  done
done

echo "=== phaseb d1-corrupt $(date)"
$RUN --domain d1 --corrupt --user-mode llm --out runs/phaseb/d1-corrupt
echo "=== phaseb done $(date)"
