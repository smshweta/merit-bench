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

## Roadmap (tracked as issues)

- [ ] Arc generator with parameterized dependent-task ratio + leak check
- [ ] LLM-simulated user (temperature 0, cached)
- [ ] Embedding retrieval for C2; LLM summarization for C3; LLM extraction for C4
- [ ] Domains D2 (IT ops), D3 (personal assistant)
- [ ] Pilot: 1 domain × 6 conditions × 1 model × 50 episodes
- [ ] Analysis notebook: paired bootstrap clustered by arc, CAMU

## License

MIT
