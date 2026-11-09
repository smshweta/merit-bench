#!/bin/sh
# Second revision round: implicit-update tier, split by condition group so
# the groups run in parallel. Each group writes its own directory
# (runs/revision/<model>-implicit-<dom>-<group>); results are identical to
# a single run because each (condition, arc) is independent and seeded.
# Usage: scripts/run_rev2_implicit.sh <model> <dom> <group>
#   model: mini | gpt41 | haiku45   group: a | b | c
set -eu
cd "$(dirname "$0")/.."
[ -f .env ] && { set -a; . ./.env; set +a; }
mk=$1; dom=$2; grp=$3
case "$mk" in
  mini) M=gpt-4.1-mini ;;
  gpt41) M=gpt-4.1 ;;
  haiku45) M=anthropic/claude-haiku-4-5-20251001 ;;
  *) echo "unknown model $mk"; exit 2 ;;
esac
case "$grp" in
  a) CONDS=C0,C1,C2 ;;
  b) CONDS=C2T,C2TR,C2C ;;
  c) CONDS=C3,C4,C5 ;;
  *) echo "unknown group $grp"; exit 2 ;;
esac
out=runs/revision/$mk-implicit-$dom-$grp
if [ -f "$out/.done" ]; then echo "skip $out (done)"; exit 0; fi
for attempt in 1 2; do
  if [ -d "$out" ]; then rm -rf "$out"; fi
  echo "--- attempt $attempt $(date)" >> "$out.stderr.log"
  if env PYTHONPATH=. PYTHONUNBUFFERED=1 MERIT_TIMEOUT=1800 \
      .venv/bin/python scripts/run_pilot.py --memory-llm gpt-4.1-mini \
      --model $M --seeds 3 --domain $dom --difficulty implicit \
      --conditions $CONDS --out "$out" 2>> "$out.stderr.log"; then
    touch "$out/.done"; echo "=== implicit $mk $dom $grp done $(date)"; exit 0
  fi
  echo "!!! $out failed (attempt $attempt)"
done
exit 1
