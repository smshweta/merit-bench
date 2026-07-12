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
- `phasec/gpt41-*`         gpt-4.1 x 1 seed
- `phasec/haiku45-*`       Claude Haiku 4.5 x 1 seed (agent side; memory
                           side stays gpt-4.1-mini across all models)
- `mur_audit/`             frozen 100-episode human-audit sample (answer
                           key withheld until labeling completes)
- `gate20*`                early gate runs (kept for the leak-fix record)

Reproduction commands: REPRODUCE.md in the code repository
(https://github.com/smshweta/merit-bench).
