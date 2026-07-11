#!/bin/sh
# Package every run (results + full episode traces) into a versioned
# tarball for archival deposit (e.g. Zenodo), backing the paper's
# "released traces" claim. Run AFTER all experiment phases you intend
# to release are complete.
set -eu
cd "$(dirname "$0")/.."
VERSION=${1:-v1}
OUT="merit-traces-$VERSION.tar.gz"

cat > runs/ARTIFACT_README.md <<'EOF'
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
- `phasec/gptoss-*`        gpt-oss:20b (local, Ollama) x 1 seed
- `mur_audit/`             frozen 100-episode human-audit sample
- `gate20*`                early gate runs (kept for the leak-fix record)

Reproduction commands: REPRODUCE.md in the code repository
(https://github.com/smshweta/merit-bench).
EOF

tar czf "$OUT" runs
shasum -a 256 "$OUT" | tee "$OUT.sha256"
du -h "$OUT"
