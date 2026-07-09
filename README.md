# MERIT — Memory Evaluation for Realistic Instrumented Tasks

A cost-aware benchmark for long-term memory in **tool-using** LLM agents.
Existing memory benchmarks test conversational recall; MERIT tests whether
memory changes what an agent *does* — and what that costs.

**Preregistered hypotheses:** see [HYPOTHESES.md](HYPOTHESES.md) (committed
before full experiments).

## Quickstart

```bash
pip install -r requirements.txt
pytest tests/            # deterministic environment tests, no API key needed

export MERIT_MODEL=gpt-4.1-mini     # any LiteLLM model string
export OPENAI_API_KEY=...           # or ANTHROPIC_API_KEY etc.
python scripts/smoke_episode.py     # runs ONE live episode (~$0.01)
```

## What's implemented (Phase 0/1 starter)

- Domain D1 (customer support): seeded SQLite world, 5 tools, programmatic
  success checkers — all unit-tested and deterministic.
- Memory conditions C0–C5 behind one interface (`merit/memory.py`); C2/C3/C4
  ship with cheap starter implementations and marked upgrade points.
- Corruption injector (stale / contradiction) with ground-truth flags.
- ReAct tool loop + JSONL trace logging + token metering (`merit/runner.py`).
- MUR value-tracer (`merit/metrics.py`).

## Run the pilot

```bash
# 1. Offline validation pilot — $0, no API key, ~10 s. Runs the ENTIRE
#    pipeline (arcs, leak check, all 6 conditions, corruption sweeps,
#    scoring, metering) with a deterministic rule-based mock agent.
python scripts/run_pilot.py --arcs 10 --episodes 5 --seeds 3 --corrupt
python scripts/analyze.py runs/pilot/results.jsonl

# 2. Real pilot (Protocol Phase 2) — same commands, one flag (~$5–20):
export OPENAI_API_KEY=...   # or ANTHROPIC_API_KEY etc.
python scripts/run_pilot.py --model gpt-4.1-mini --arcs 10 --episodes 5 \
    --seeds 1 --corrupt
python scripts/analyze.py runs/pilot/results.jsonl
```

**Mock results validate the harness only** — checkers, leak check, corruption
injector, MUR tracer, bootstrap analysis. They are not evidence about LLM
agents and must never be reported as experimental results.

## Roadmap (tracked as issues)

- [x] Arc generator with parameterized dependent-task ratio + leak check (`merit/arcs.py`)
- [x] Simulated user: scripted deterministic mode + cached LLM mode (`merit/user_sim.py`)
- [x] Corruption injector: stale / contradiction / distractor, all stores (`merit/memory.py`)
- [x] Pilot runner + $0 mock pipeline validation (`scripts/run_pilot.py`, `merit/mockmodel.py`)
- [x] Analysis: paired bootstrap clustered by arc, H1–H4 readout, CAMU (`scripts/analyze.py`)
- [ ] Real pilot: 1 domain × 6 conditions × 1 cheap model × 50 episodes (needs API key)
- [ ] Embedding retrieval for C2; LLM summarization for C3; LLM extraction for C4
- [ ] Domains D2 (IT ops), D3 (personal assistant)
- [ ] Phase 0 calibration: reproduce a Mem0/LoCoMo slice result

## License

MIT
