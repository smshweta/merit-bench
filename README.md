# MERIT — Memory Evaluation for Realistic Instrumented Tasks

A cost-aware benchmark for long-term memory in **tool-using LLM agents**.
Existing memory benchmarks test conversational recall; MERIT tests whether
memory changes what an agent *does* — and what that costs.

**Paper:** [docs/paper_draft.md](docs/paper_draft.md) (arXiv source in
[paper/main.tex](paper/main.tex); IEEE conference version in
[paper/ieee/main.tex](paper/ieee/main.tex)) · **Preregistered hypotheses:**
[HYPOTHESES.md](HYPOTHESES.md) (committed before the experiments) ·
**Protocol:** [docs/experiment_protocol.md](docs/experiment_protocol.md) ·
**Reproduce:** [REPRODUCE.md](REPRODUCE.md)

**Study complete: 23,440 scored episodes, $44.82 total API cost.** All
results data and full episode traces are in [`runs/`](runs/) in this repo.

## Headline results

![difficulty ladder](docs/figures/fig1_difficulty_ladder.png)

1. **Memory helps, and the floor is real.** With no memory, dependent-task
   success is 0.00 (leak-verified: the facts cannot be re-derived). Every
   memory condition lifts it to 0.55–1.00 (all Holm-adjusted p ≤ 0.001).
2. **The difficulty ladder dissociates architectures.** Single-fact recall
   is at ceiling for everything — existing benchmarks stop here. On
   *updated-fact* recall (the hard tier), embedding retrieval collapses
   while stores that overwrite (LLM summarization, structured fact store)
   stay at 0.70–1.00.
3. **The collapse is unreliable, which is worse.** A 3-model × 3-seed
   replication (gpt-4.1-mini, GPT-4.1, Claude Haiku 4.5; memory side held
   fixed) shows *where* embedding retrieval fails varies by agent model
   (hard-tier success 0.30–0.95) and by seed (max pairwise gap 0.45) —
   while LLM summarization spans 0.80–1.00 across every model, domain, and
   seed. Single-model single-seed rankings of RAG memory are not
   trustworthy.

![cross-model hard tier](docs/figures/fig6_phasec_hard_cross_model.png)

4. **Implementation quality is a first-class variable.** Swapping the
   starter summarizer for LLM summarization rescues updated-fact recall
   (0.00–0.15 → 0.70–1.00); swapping regex extraction for LLM extraction
   *costs* the fact store up to 60 points in one domain.
5. **Full replay is never the economical choice.** With memory-side calls
   metered, the best condition per domain delivers 2.7–3.9× the marginal
   utility per dollar of replaying everything.

## Design

| | |
|---|---|
| **Domains** | D1 commerce support, D2 IT ops, D3 personal assistant — seeded SQLite worlds, real tool calls, programmatic success checkers |
| **Memory conditions** | C0 none · C1 full replay · C2 embedding retrieval · C3 LLM summarization · C4 structured fact store · C5 hybrid (C4+C2) — one interface, [merit/memory.py](merit/memory.py) + [merit/memory_llm.py](merit/memory_llm.py) |
| **Difficulty ladder** | easy (single fact) · medium (multi-fact) · hard (*updated* fact — the value changed mid-arc) |
| **Corruption** | stale / contradiction / distractor records injected at known rates ρ with ground-truth flags |
| **Metrics** | TSR (dependent vs independent), MUR (memory utilization), Ignore Rate, Stale-Memory Harm, CAMU (cost-adjusted marginal utility) |
| **Statistics** | preregistered; paired bootstrap clustered by arc, Holm–Bonferroni within families |
| **Leak check** | automated: a dependent task must be *unsolvable* without memory, enforced per episode (C0 floor = 0.00 is measured, not assumed) |

## Quickstart

```bash
pip install -r requirements.txt
pytest tests/            # 62 deterministic tests, no API key needed

# $0 offline run of the ENTIRE pipeline with a deterministic mock agent
PYTHONPATH=. python scripts/run_pilot.py --arcs 10 --episodes 5 --seeds 3 --corrupt
PYTHONPATH=. python scripts/analyze.py runs/pilot/results.jsonl

# one live episode (~$0.01) to verify your key
export OPENAI_API_KEY=...           # or ANTHROPIC_API_KEY
MERIT_MODEL=gpt-4.1-mini PYTHONPATH=. python scripts/smoke_episode.py
```

Mock-model results validate the harness only — they are never evidence
about LLM agents. Full reproduction commands for every experiment in the
paper: [REPRODUCE.md](REPRODUCE.md).

## Released data (`runs/`)

Every experiment in the paper, committed in full — one directory per run,
each with `results.jsonl` (one row per scored episode: success,
`pre_satisfied`, memory-utilization flags, tokens, dollars, wall time) and
`traces/episodes.jsonl` (full transcripts: memory shown, user turns,
executed tool calls with arguments and outputs).

| Directory | Contents |
|---|---|
| `runs/pilot-*`, `runs/sweep/` | starter-implementation generation (gen 1) + difficulty sweep |
| `runs/phaseb/` | real implementations on gpt-4.1-mini (gen 2, headline pilot numbers) |
| `runs/phasec/mini3-*` | full grid, gpt-4.1-mini × 3 seeds |
| `runs/phasec/gpt41-*` | full grid, GPT-4.1 |
| `runs/phasec/haiku45-*` | full grid, Claude Haiku 4.5 (agent side; memory side stays gpt-4.1-mini) |
| `runs/mur_audit/` | frozen 100-episode stratified sample for the human MUR audit (answer key withheld until labeling completes) |

Analysis: `scripts/analyze.py` (single run, H1–H4 readout),
`scripts/analyze_phasec.py` (multi-model grid: seed robustness +
cross-model tables), `scripts/figures.py` (all paper figures),
`scripts/mur_audit.py` (audit sample + Cohen's κ).

## Repository layout

```
merit/            benchmark package: worlds, tools, memory conditions,
                  runner, metrics, corruption, arc generation, user sim
scripts/          drivers (run_pilot.py, run_phase*.sh), analysis, figures
tests/            62 deterministic tests (no API key)
runs/             all released results + traces (see table above)
docs/             paper draft (source of truth), figures, protocol
paper/            arXiv LaTeX source (mirrors docs/paper_draft.md)
HYPOTHESES.md     preregistration (dated via git history)
```

## Citation

```bibtex
@misc{mishra2026merit,
  title   = {When Does Memory Help? A Cost-Aware Evaluation of Long-Term
             Memory in Tool-Using LLM Agents},
  author  = {Mishra, Shweta and Mishra, Shashank},
  year    = {2026},
  url     = {https://github.com/smshweta/merit-bench}
}
```

## License

MIT (code). Released run data and traces may contain LLM-generated text.
