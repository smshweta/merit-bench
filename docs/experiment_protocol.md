# MERIT — Experiment Protocol & Execution Plan
*(companion to paper_draft.md — follow phases in order; do not skip the pilot)*

## Phase 0 — Setup (Week 1–2, ~10 hrs, cost ≈ $0)
1. Create public GitHub repo `merit-bench` (MIT license). Public from day 1 — commit history is O1 evidence of authorship.
2. Stack: Python 3.11, LangGraph (agent loop), LiteLLM (unified API + token metering), SQLite (world state), pytest (env determinism tests).
3. Reproduce one existing result to calibrate: run Mem0's open-sourced eval on a LoCoMo slice to confirm your harness measures tokens/accuracy the way published papers do. This also gives you a Related Work sanity check.
4. **Preregister**: commit `HYPOTHESES.md` (H1–H4 from the paper) with a dated commit BEFORE running full experiments. Reviewers increasingly reward this; it also proves you didn't cherry-pick.

## Phase 1 — Build the environment (Week 3–6, ~40 hrs)
1. Implement Domain D1 (customer support) first: 5 tools (get_order, refund, update_address, get_policy, send_message) over a seeded SQLite world.
2. Write the arc generator: script that produces arcs of 4–6 episodes with parameterized dependent-task ratio. Facts needed later are planted in earlier episodes ONLY (verify with an automated "leak check" that the fact string never appears in the later episode's inputs).
3. Simulated user: LLM role-playing from a fixed persona + script skeleton, temperature 0. Cache user turns where possible.
4. Programmatic success checkers: pure functions over final world state. Unit-test every checker.
5. Implement memory conditions C0–C5 behind one interface: `memory.write(episode_trace)`, `memory.read(current_context) -> memory_block`.
6. Implement the corruption injector (stale / contradiction / distractor) as a transform on the memory store, with rate ρ.
7. Gate: 20 hand-run episodes pass manual inspection before scaling.

## Phase 2 — Pilot (Week 7, ~$50–100 API)
- 1 domain × all 6 conditions × 1 model (cheap: GPT-4.1-mini or Claude Haiku) × 50 episodes.
- Purpose: shake out bugs, estimate variance (needed to size the full run), estimate cost per episode, validate the MUR value-tracer on 30 hand-audited episodes.
- Decision point: if variance is high, increase arcs, not seeds (arcs are the clustering unit).

## Phase 3 — Full run (Week 8–10)
- 3 domains × 6 conditions × 3 models × ~500 episodes × 3 seeds.
- Budget estimate (verify in pilot): API models ~$0.02–0.15/episode depending on condition (C1 full-replay is the expensive one) → roughly $800–2,500 total. Open-weight model: rent 1×H100 (~$2–3/hr, ~60–100 hrs) → $150–300. **Total realistic budget: $1,000–3,000.** If tight: drop to 2 models and 2 domains — still publishable.
- Log EVERYTHING: full traces to JSONL, token counts from LiteLLM, wall-clock per call.

## Phase 4 — Analysis & human audit (Week 11–12)
- Paired bootstrap (10k resamples, clustered by arc) for all deltas; Holm–Bonferroni over H1–H4.
- Human audit: stratified sample of 100 episodes for MUR tracer validation; recruit 1 second annotator (a colleague) for 50 of them; report Cohen's κ.
- Produce Figures 1–3 (matplotlib, colorblind-safe).

## Phase 5 — Writing & submission (Week 13–16)
- Convert paper_draft.md to the target template. Post to arXiv immediately on submission.
- Target venues (verify deadlines the week you're ready — they shift):
  1. **NeurIPS Datasets & Benchmarks track** (top-tier; deadline typically ~May — likely next cycle for you)
  2. **ICLR** (deadline ~Sep 2026 — realistic primary target if you start now)
  3. **ICSE 2027 SEIP / FSE industry** (practitioner framing)
  4. Faster fallbacks: NeurIPS/ICLR workshops on agents or memory (deadlines rolling, ~4–6 week reviews), ISSRE industry track.
- Release checklist: repo README with one-command repro, dataset card, traces uploaded (HuggingFace datasets), leaderboard stub (invites others to cite you).

## Commitment device (the psychology part)
- Put the Phase gates on your calendar NOW as non-negotiable appointments.
- Public accountability: post a one-tweet/LinkedIn thread announcing the benchmark at end of Phase 1 — sunk public commitment dramatically raises completion rates, and early followers become your first citers.
- Rule: any week you touch the project ≥5 hrs counts as a win. Consistency beats intensity for a 16-week project run alongside a full-time job.

## Risk register (honest)
- **Scooped risk (MEDIUM-HIGH — this area moves monthly):** mitigation = arXiv the benchmark description as a short preprint after Phase 1, before full results. Priority is established by the preprint date.
- **Null-result risk (LOW-MEDIUM):** even "memory doesn't pay for itself below $X task value" IS the headline — negative results with rigorous cost analysis are highly citable in an over-hyped area.
- **API model deprecation:** pin versions; the open-weight model is your permanence anchor.
