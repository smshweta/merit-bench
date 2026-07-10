# Reproducing the MERIT pilot results

Every number in `docs/paper_draft.md` §5 comes from the commands below.
Worlds, arcs, and users are deterministic given the seed; the only
stochastic component is the model itself (temperature 0, but provider-side
nondeterminism applies to API models).

## 0. Environment

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-lock.txt   # exact pinned versions
python -m pytest tests/                # 56 tests, no API key needed
```

## 1. $0 pipeline validation (no API key)

The deterministic mock model exercises the full grid — generation, leak
checks, memory conditions, corruption, scoring, analysis:

```bash
python scripts/run_pilot.py --corrupt --out runs/mock   # model=mock default
python scripts/analyze.py runs/mock/results.jsonl
```

Mock rows validate the harness only and must never be reported as findings.

## 2. Real pilot (model: gpt-4.1-mini, key via OPENAI_API_KEY)

Each command appends to a fresh `--out` dir; ~$0.25 per 300-episode run.

```bash
# easy tier, per domain (results in paper §5.1, §5.5)
python scripts/run_pilot.py --model gpt-4.1-mini --domain d1 --out runs/pilot-clean
python scripts/run_pilot.py --model gpt-4.1-mini --domain d2 --out runs/pilot-d2-v2
python scripts/run_pilot.py --model gpt-4.1-mini --domain d3 --out runs/sweep/d3-easy

# difficulty sweep (paper §5.2, §5.3)
for spec in d1:medium d1:hard d2:medium d2:hard d3:medium d3:hard; do
  python scripts/run_pilot.py --model gpt-4.1-mini \
    --domain "${spec%%:*}" --difficulty "${spec##*:}" \
    --out "runs/sweep/${spec%%:*}-${spec##*:}"
done

# corruption sweep + LLM-paraphrased users on D1 (paper §5.4)
python scripts/run_pilot.py --model gpt-4.1-mini --domain d1 \
  --corrupt --user-mode llm --out runs/pilot-full
```

## 3. Analysis (CIs, Holm-adjusted p, CAMU)

```bash
python scripts/analyze.py runs/pilot-clean/results.jsonl   # any results file
```

## 4. Local / open-weight models

Any OpenAI-compatible server works, e.g. Ollama:

```bash
export OPENAI_API_KEY=ollama
python scripts/run_pilot.py --model openai/qwen3:8b \
  --api-base http://localhost:11434/v1 --domain d1 --out runs/local
```

## Layout of results

One JSON row per scored episode in `<out>/results.jsonl` (fields include
`success`, `pre_satisfied`, `memory_had_fact`, `memory_utilized`, token and
dollar costs); full transcripts in `<out>/traces/episodes.jsonl`.

## Preregistration

`HYPOTHESES.md` (H1–H4) was committed before any full experiment; the git
history of this repository is the preregistration record.
