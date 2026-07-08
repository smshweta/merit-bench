# MERIT — Preregistered Hypotheses

**Committed BEFORE running full experiments (Phase 3). Pilot (Phase 2) may refine
operationalizations but not the direction of these hypotheses. Any post-hoc analyses
will be explicitly labeled exploratory in the paper.**

Date of preregistration: <fill in on commit day>
Author: Shweta Mishra

## Primary hypotheses

- **H1 (Memory helps when needed):** Task Success Rate (TSR) of memory conditions
  C2–C5 is higher than the no-memory baseline C0 on *dependent* tasks
  (tasks requiring a fact from a prior episode).

- **H2 (Memory has a distraction cost):** TSR of memory conditions C2–C5 is
  *less than or equal to* C0 on *independent* tasks (tasks solvable entirely
  within-episode). i.e., retrieved-but-irrelevant memory does not help and may hurt.

- **H3 (Corrupted memory harms, structure protects):** Stale-Memory Harm
  (TSR_clean − TSR_corrupted) is positive for all memory conditions, and is
  smaller for the structured fact store (C4, update-on-write semantics) than
  for vector RAG (C2) at the same corruption rate.

- **H4 (Accuracy ranking ≠ economic ranking):** The ranking of conditions by
  Cost-Adjusted Marginal Utility (CAMU = ΔTSR/Δcost vs C0) differs from the
  ranking by raw TSR.

## Analysis plan (fixed in advance)

- Paired comparisons on identical task instances; 95% CIs via paired bootstrap
  (10,000 resamples) clustered at the arc level; ≥3 seeds per condition.
- Holm–Bonferroni correction across H1–H4.
- Primary metric definitions live in `merit/metrics.py` at the commit tagged
  `prereg-v1`; changes after that tag are exploratory.
