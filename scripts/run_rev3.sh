#!/bin/sh
# Third revision round (Discover AI): more seeds for the single-seed
# stress tests, and conflict-aware retrieval (C2C) in both.
#   long-<dom>-s<k>     20-episode long-history test, seed k (1 or 2),
#                       all 8 conditions + C2C
#   long-<dom>-c2c      C2C on seed 0 of the long-history test
#   corrupt-<dom>-s<k>  corruption sweep, seed k (1 or 2), 8 conditions + C2C
#   corrupt-<dom>-c2c   C2C on seed 0 of the corruption sweep
# gpt-4.1-mini agent, memory side gpt-4.1-mini + text-embedding-3-small.
# Cells are resume-safe: an interrupted cell continues with --resume
# (complete arcs kept), so a credit outage never loses finished work.
# Usage: scripts/run_rev3.sh <part> | all   (all = every part in parallel)
set -eu
cd "$(dirname "$0")/.."
[ -f .env ] && { set -a; . ./.env; set +a; }
export MERIT_MODEL=gpt-4.1-mini
ALL9=C0,C1,C2,C3,C4,C5,C2T,C2TR,C2C
OUT=runs/revision

cell() {
  out=$1; shift
  if [ -f "$out/.done" ]; then echo "skip $out (done)"; return 0; fi
  for attempt in 1 2 3; do
    R=""; [ -f "$out/results.jsonl" ] && R=--resume
    echo "--- attempt $attempt $R $(date)" >> "$out.stderr.log"
    if env PYTHONPATH=. PYTHONUNBUFFERED=1 MERIT_TIMEOUT=1800 \
        .venv/bin/python scripts/run_pilot.py --model gpt-4.1-mini \
        --memory-llm gpt-4.1-mini "$@" $R --out "$out" \
        2>> "$out.stderr.log"; then
      touch "$out/.done"; echo "=== $out done $(date)"; return 0
    fi
    echo "!!! $out failed (attempt $attempt), see $out.stderr.log"
    sleep 30
  done
  return 1
}

run_part() {
  part=$1
  kind=${part%%-*}; rest=${part#*-}; dom=${rest%-*}; sfx=${rest#*-}
  case "$sfx" in
    s1) SEEDS="--seeds 1 --seed-start 1"; CONDS=$ALL9 ;;
    s2) SEEDS="--seeds 1 --seed-start 2"; CONDS=$ALL9 ;;
    c2c) SEEDS=""; CONDS=C2C ;;
    *) echo "unknown part $part"; exit 2 ;;
  esac
  case "$kind" in
    long)
      cell $OUT/long-mini-$dom-$sfx --domain $dom --difficulty hard \
        --episodes 20 --dep-ratio 0.16 --conditions $CONDS $SEEDS ;;
    corrupt)
      cell $OUT/corrupt-$dom-$sfx --domain $dom --corrupt --user-mode llm \
        --conditions $CONDS $SEEDS ;;
    *) echo "unknown part $part"; exit 2 ;;
  esac
}

if [ "${1:?part required}" = all ]; then
  for kind in long corrupt; do for dom in d1 d2 d3; do
    for sfx in s1 s2 c2c; do
      sh "$0" $kind-$dom-$sfx &
    done
  done; done
  wait
  echo "=== rev3 all finished $(date)"
else
  run_part "$1"
fi
