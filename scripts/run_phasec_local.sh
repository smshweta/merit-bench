#!/bin/sh
# Phase C (open-weight slice): gpt-oss:20b served locally by Ollama,
# 1 seed x 9-cell grid. Agent-side cost is $0 (local); memory side stays
# gpt-4.1-mini via the OpenAI API so memory implementations are identical
# across all Phase C agent models. Slow on a laptop (roughly a day).
set -eu
cd "$(dirname "$0")/.."
[ -f .env ] && { set -a; . ./.env; set +a; }

for dom in d1 d2 d3; do
  for tier in easy medium hard; do
    echo "=== phasec gptoss $dom-$tier $(date)"
    env PYTHONPATH=. PYTHONUNBUFFERED=1 MERIT_TIMEOUT=1800 .venv/bin/python scripts/run_pilot.py \
      --model openai/gpt-oss:20b --api-base http://localhost:11434/v1 \
      --memory-llm gpt-4.1-mini \
      --domain "$dom" --difficulty "$tier" \
      --out "runs/phasec/gptoss-$dom-$tier" \
      2>> "runs/phasec/gptoss-$dom-$tier.stderr.log"
  done
done
echo "=== phasec local done $(date)"
