#!/bin/sh
# Resume the second revision round after an interruption (the OpenAI
# account ran out of credits mid-run). Every incomplete C2C or implicit-tier
# directory is resumed with run_pilot.py --resume: complete arcs are kept,
# partial arcs are re-run, and the directory is marked done on success.
# All incomplete directories run in parallel. Usage: scripts/resume_rev2.sh
set -eu
cd "$(dirname "$0")/.."
[ -f .env ] && { set -a; . ./.env; set +a; }

model_of() {
  case "$1" in
    mini) echo gpt-4.1-mini ;;
    gpt41) echo gpt-4.1 ;;
    haiku45) echo anthropic/claude-haiku-4-5-20251001 ;;
  esac
}

resume_one() {
  out=$1; shift
  echo "--- resume $(date)" >> "$out.stderr.log"
  if env PYTHONPATH=. PYTHONUNBUFFERED=1 MERIT_TIMEOUT=1800 \
      .venv/bin/python scripts/run_pilot.py --memory-llm gpt-4.1-mini \
      --seeds 3 --resume --out "$out" "$@" 2>> "$out.stderr.log"; then
    touch "$out/.done"; echo "=== resumed $out done $(date)"
  else
    echo "!!! resume $out failed"
  fi
}

for d in runs/revision/*-c2c-*-* runs/revision/*-implicit-d?-?; do
  [ -d "$d" ] || continue
  [ -f "$d/.done" ] && continue
  name=$(basename "$d")
  mk=${name%%-*}
  M=$(model_of "$mk")
  case "$name" in
    *-c2c-*)
      rest=${name#*-c2c-}; dom=${rest%-*}; tier=${rest#*-}
      resume_one "$d" --model "$M" --domain "$dom" --difficulty "$tier" \
        --conditions C2C &
      ;;
    *-implicit-*)
      rest=${name#*-implicit-}; dom=${rest%-*}; grp=${rest#*-}
      case "$grp" in
        a) C=C0,C1,C2 ;; b) C=C2T,C2TR,C2C ;; c) C=C3,C4,C5 ;;
      esac
      resume_one "$d" --model "$M" --domain "$dom" --difficulty implicit \
        --conditions "$C" &
      ;;
  esac
done
wait
echo "=== resume_rev2 finished $(date)"
