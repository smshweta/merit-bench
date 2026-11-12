# MERIT benchmark — released runs and traces

One directory per run. Each contains:
- `results.jsonl` — one row per scored episode (success, pre_satisfied,
  memory_had_fact, memory_utilized, tokens, dollars, wall time, ids)
- `traces/episodes.jsonl` — full transcripts: memory block shown to the
  agent, user turns, executed tool calls with arguments and outputs

Run directories:
- `pilot-*`, `sweep/*`     starter-implementation generation (gen 1)
- `phaseb/*`               real implementations (embedding retrieval, LLM
                           summarization/extraction); headline numbers
- `phasec/mini3-*`         gpt-4.1-mini x 3 seeds
- `phasec/gpt41-*`         gpt-4.1, seed 0 (preregistered grid)
- `phasec/haiku45-*`       Claude Haiku 4.5, seed 0 (agent side; memory
                           side stays gpt-4.1-mini across all models)
- `revision/*`             experiments added during peer review (not
                           preregistered): `*-s12-*` seeds 1-2 for GPT-4.1
                           and Haiku 4.5; `*-temporal-*` temporal retrieval
                           conditions C2T/C2TR; `corrupt-*` corruption
                           sweeps; `long-mini-*` 20-episode arcs.
                           Analysis: scripts/analyze_revision.py.
                           Second round: `*-c2c-*` conflict-aware
                           retrieval C2C on the grid; `*-implicit-<dom>-<g>`
                           implicit-update tier (groups a/b/c split the
                           nine conditions). Analysis:
                           scripts/analyze_rev2.py (output in
                           revision/analysis_rev2.txt)
                           Third round: `corrupt-<dom>-s1|s2|c2c` and
                           `long-mini-<dom>-s1|s2|c2c` add seeds 1-2 and
                           C2C to the corruption sweeps and the
                           long-history test (scripts/run_rev3.sh).
                           Analysis: scripts/analyze_rev3.py (output in
                           revision/analysis_rev3.txt)
- `intervention/`          counterfactual replay of C2 hard-tier probes
- `mur_audit/`             frozen 100-episode human-audit sample:
                           labels.csv (two annotators, blind to the tracer)
                           and key.csv (item -> episode and tracer label)
- `gate20*`                early gate runs (kept for the leak-fix record)

Reproduction commands: REPRODUCE.md in the code repository
(https://github.com/smshweta/merit-bench).
