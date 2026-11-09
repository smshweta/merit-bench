#!/bin/sh
# Revision-stage experiments (Discover AI major revision, Nov 2026).
# Not preregistered; reported as revision-stage additions.
#   corrupt     D2/D3 corruption sweep (C0-C5 + C2T/C2TR) and the D1 sweep
#               for the new temporal conditions            (gpt-4.1-mini)
#   temporal    C2T/C2TR on the 9-cell grid, 3 seeds       (gpt-4.1-mini)
#   gpt41-<dom> GPT-4.1: C2T/C2TR on seed 0, all 8 conditions on seeds 1-2
#   long        long-history stress test: hard tier, 20 episodes/arc,
#               3 probes/arc, all 8 conditions              (gpt-4.1-mini)
#   rev2-<model>-<dom>  second round: C2C on the 3x3 grid and the
#               implicit-update tier, 3 seeds (the implicit tier was run
#               split by condition group with run_rev2_implicit.sh, and
#               resumed with resume_rev2.sh after an API-credit outage)
# Usage: scripts/run_revision.sh <part>   (parts run as separate processes)
# Memory side pinned to gpt-4.1-mini + text-embedding-3-small, as in Phase C.
set -eu
cd "$(dirname "$0")/.."
[ -f .env ] && { set -a; . ./.env; set +a; }

MEMLLM=gpt-4.1-mini
ALL=C0,C1,C2,C3,C4,C5,C2T,C2TR
TEMP=C2T,C2TR
OUT=runs/revision
mkdir -p $OUT

# cell <out-dir> <run_pilot args...>; on failure wipe and retry once
# (results.jsonl is append-only, so a retry must start from a fresh dir)
cell() {
  out=$1; shift
  if [ -f "$out/.done" ]; then echo "skip $out (done)"; return 0; fi
  for attempt in 1 2; do
    if [ -d "$out" ]; then rm -rf "$out"; fi
    echo "--- attempt $attempt $(date)" >> "$out.stderr.log"
    if env PYTHONPATH=. PYTHONUNBUFFERED=1 MERIT_TIMEOUT=1800 \
        .venv/bin/python scripts/run_pilot.py --memory-llm $MEMLLM "$@" \
        --out "$out" 2>> "$out.stderr.log"; then
      touch "$out/.done"; return 0
    fi
    echo "!!! cell $out failed (attempt $attempt), see $out.stderr.log"
  done
  return 1
}

part=${1:?part required}
case "$part" in
  corrupt)
    # LLM-paraphrased users (as in the D1 sweep) take their model from
    # MERIT_MODEL; paraphrases are cached in .user_cache
    export MERIT_MODEL=gpt-4.1-mini
    for dom in d2 d3; do
      echo "=== corrupt $dom $(date)"
      cell $OUT/corrupt-$dom --model gpt-4.1-mini --domain $dom --corrupt \
        --user-mode llm --conditions $ALL
    done
    echo "=== corrupt d1 temporal $(date)"
    cell $OUT/corrupt-d1-temporal --model gpt-4.1-mini --domain d1 --corrupt \
      --user-mode llm --conditions $TEMP
    ;;
  temporal)
    for dom in d1 d2 d3; do for tier in easy medium hard; do
      echo "=== temporal mini3 $dom-$tier $(date)"
      cell $OUT/mini3-temporal-$dom-$tier --model gpt-4.1-mini --seeds 3 \
        --domain $dom --difficulty $tier --conditions $TEMP
    done; done
    ;;
  gpt41-d1|gpt41-d2|gpt41-d3)
    dom=${part#gpt41-}
    for tier in easy medium hard; do
      echo "=== gpt41 seed0 temporal $dom-$tier $(date)"
      cell $OUT/gpt41-temporal-$dom-$tier --model gpt-4.1 \
        --domain $dom --difficulty $tier --conditions $TEMP
      echo "=== gpt41 seeds1-2 $dom-$tier $(date)"
      cell $OUT/gpt41-s12-$dom-$tier --model gpt-4.1 --seeds 2 --seed-start 1 \
        --domain $dom --difficulty $tier --conditions $ALL
    done
    ;;
  long)
    for dom in d1 d2 d3; do
      echo "=== long mini $dom $(date)"
      cell $OUT/long-mini-$dom --model gpt-4.1-mini --domain $dom \
        --difficulty hard --episodes 20 --dep-ratio 0.16 --conditions $ALL
    done
    ;;
  long-d1|long-d2|long-d3)
    # one domain of the long-history test (to run domains in parallel)
    dom=${part#long-}
    cell $OUT/long-mini-$dom --model gpt-4.1-mini --domain $dom \
      --difficulty hard --episodes 20 --dep-ratio 0.16 --conditions $ALL
    ;;
  haiku-d1|haiku-d2|haiku-d3)
    # needs a valid ANTHROPIC_API_KEY
    dom=${part#haiku-}
    H=anthropic/claude-haiku-4-5-20251001
    for tier in easy medium hard; do
      cell $OUT/haiku45-temporal-$dom-$tier --model $H \
        --domain $dom --difficulty $tier --conditions $TEMP
      cell $OUT/haiku45-s12-$dom-$tier --model $H --seeds 2 --seed-start 1 \
        --domain $dom --difficulty $tier --conditions $ALL
    done
    ;;
  rev2-*)
    # second revision round: conflict-aware retrieval (C2C) on the 3x3
    # grid, and the implicit-update tier (hard arcs without supersession
    # cues) with all nine conditions; 3 seeds. Usage: rev2-<model>-<dom>
    #   model: mini | gpt41 | haiku45
    rest=${part#rev2-}; mk=${rest%-*}; dom=${rest##*-}
    case "$mk" in
      mini) M=gpt-4.1-mini ;;
      gpt41) M=gpt-4.1 ;;
      haiku45) M=anthropic/claude-haiku-4-5-20251001 ;;
      *) echo "unknown model $mk"; exit 2 ;;
    esac
    for tier in hard easy medium; do
      echo "=== rev2 $mk c2c $dom-$tier $(date)"
      cell $OUT/$mk-c2c-$dom-$tier --model $M --seeds 3 \
        --domain $dom --difficulty $tier --conditions C2C
    done
    echo "=== rev2 $mk implicit $dom $(date)"
    cell $OUT/$mk-implicit-$dom --model $M --seeds 3 --domain $dom \
      --difficulty implicit --conditions $ALL,C2C
    ;;
  *) echo "unknown part $part"; exit 2 ;;
esac
echo "=== $part done $(date)"
