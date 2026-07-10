"""Phase 4 analysis over pilot/full-run results.jsonl.

Computes, per condition: TSR split by dependent/independent tasks, MUR,
Ignore Rate, Stale-Memory Harm, mean cost, and CAMU vs C0 — with 95% CIs
from a paired bootstrap (10k resamples) clustered at the ARC level, exactly
as preregistered in HYPOTHESES.md. Prints an H1–H4 readout.

stdlib only (no numpy needed).

Usage: python scripts/analyze.py runs/pilot/results.jsonl
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

N_BOOT = 10_000


def load(path: str) -> list[dict]:
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l]
    if not rows:
        sys.exit("no rows in results file")
    return rows


def mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def by_arc(rows) -> dict[str, list[dict]]:
    d = defaultdict(list)
    for r in rows:
        d[r["arc_id"]].append(r)
    return d


def boot_ci_delta(rows_a: list[dict], rows_b: list[dict], key,
                  n_boot: int = N_BOOT, seed: int = 0):
    """Bootstrap CI + two-sided p for mean(key(a)) - mean(key(b)),
    resampling ARCS with replacement (the preregistered clustering unit)."""
    arcs_a, arcs_b = by_arc(rows_a), by_arc(rows_b)
    shared = sorted(set(arcs_a) & set(arcs_b))
    if not shared:
        nan = float("nan")
        return nan, (nan, nan), nan
    rng = random.Random(seed)
    point = (mean(key(r) for a in shared for r in arcs_a[a])
             - mean(key(r) for a in shared for r in arcs_b[a]))
    deltas = []
    for _ in range(n_boot):
        sample = [shared[rng.randrange(len(shared))] for _ in shared]
        da = mean(key(r) for a in sample for r in arcs_a[a])
        db = mean(key(r) for a in sample for r in arcs_b[a])
        deltas.append(da - db)
    deltas.sort()
    lo = deltas[int(0.025 * n_boot)]
    hi = deltas[int(0.975 * n_boot)]
    # two-sided bootstrap p with the +1 continuity correction
    p_le = (sum(d <= 0 for d in deltas) + 1) / (n_boot + 1)
    p_ge = (sum(d >= 0 for d in deltas) + 1) / (n_boot + 1)
    return point, (lo, hi), min(1.0, 2 * min(p_le, p_ge))


def holm(pvals: dict) -> dict:
    """Holm–Bonferroni step-down adjustment over one hypothesis family
    (preregistered in HYPOTHESES.md). Returns adjusted p per key."""
    items = sorted((p, k) for k, p in pvals.items() if p == p)  # drop NaN
    adjusted, running = {}, 0.0
    m = len(items)
    for i, (p, k) in enumerate(items):
        running = max(running, (m - i) * p)
        adjusted[k] = min(1.0, running)
    return adjusted


def fmt_ci(point, ci) -> str:
    return f"{point:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]"


def sig(ci) -> str:
    return "*" if (ci[0] > 0 or ci[1] < 0) else " "


def main(path: str) -> None:
    rows = load(path)
    clean = [r for r in rows if r["corrupt_mode"] == "none"]
    conditions = sorted({r["condition"] for r in rows})

    def sel(rs, cond=None, dep=None):
        out = rs
        if cond is not None:
            out = [r for r in out if r["condition"] == cond]
        if dep is not None:
            out = [r for r in out if r["dependent"] == dep]
        return out

    # ---------- main table ----------
    print(f"\n{'cond':6} {'TSR_dep':>8} {'TSR_ind':>8} {'MUR':>6} "
          f"{'Ignore':>7} {'tok/ep':>8} {'$/ep':>8}")
    for c in conditions:
        dep = sel(clean, c, True)
        ind = sel(clean, c, False)
        had = [r for r in dep if r["memory_had_fact"]]
        mur = mean(r["memory_utilized"] for r in had) if had else float("nan")
        ign = 1 - mur if had else float("nan")
        toks = mean(r["prompt_tokens"] + r["completion_tokens"]
                    for r in sel(clean, c))
        cost = mean(r["cost_usd"] for r in sel(clean, c))
        print(f"{c:6} {mean(r['success'] for r in dep):8.3f} "
              f"{mean(r['success'] for r in ind):8.3f} {mur:6.3f} "
              f"{ign:7.3f} {toks:8.0f} {cost:8.4f}")

    # ---------- H1 / H2 (Holm–Bonferroni within each family) ----------
    c0_dep = sel(clean, "C0", True)
    c0_ind = sel(clean, "C0", False)
    for label, dep, c0_rows in (
            ("H1 (memory helps on DEPENDENT tasks)", True, c0_dep),
            ("H2 (distraction cost on INDEPENDENT tasks)", False, c0_ind)):
        print(f"\n{label}: ΔTSR vs C0, paired bootstrap 95% CI, "
              f"clustered by arc, Holm-adjusted p")
        results, pvals = {}, {}
        for c in conditions:
            if c == "C0":
                continue
            point, ci, p = boot_ci_delta(sel(clean, c, dep), c0_rows,
                                         key=lambda r: r["success"])
            results[c], pvals[c] = (point, ci, p), p
        adj = holm(pvals)
        for c, (point, ci, p) in results.items():
            a = adj.get(c, float("nan"))
            mark = "*" if a < 0.05 else " "
            print(f"  {c} - C0: {fmt_ci(point, ci)}  p={p:.4f} "
                  f"holm={a:.4f} {mark}")

    # ---------- H3: Stale-Memory Harm (Holm within family) ----------
    corrupted = [r for r in rows if r["corrupt_mode"] != "none"]
    if corrupted:
        print("\nH3 (Stale-Memory Harm = TSR_clean - TSR_corrupted, "
              "dependent tasks, Holm-adjusted p)")
        results, pvals = {}, {}
        for c in conditions:
            if c == "C0":
                continue
            for mode in sorted({r["corrupt_mode"] for r in corrupted}):
                for rate in sorted({r["corrupt_rate"] for r in corrupted}):
                    cor = [r for r in corrupted
                           if r["condition"] == c and r["dependent"]
                           and r["corrupt_mode"] == mode
                           and r["corrupt_rate"] == rate]
                    if not cor:
                        continue
                    point, ci, p = boot_ci_delta(sel(clean, c, True), cor,
                                                 key=lambda r: r["success"])
                    results[(c, mode, rate)] = (point, ci, p)
                    pvals[(c, mode, rate)] = p
        adj = holm(pvals)
        for (c, mode, rate), (point, ci, p) in results.items():
            a = adj.get((c, mode, rate), float("nan"))
            mark = "*" if a < 0.05 else " "
            print(f"  {c} {mode:13} ρ={rate}: SMH = {fmt_ci(point, ci)}  "
                  f"p={p:.4f} holm={a:.4f} {mark}")

    # ---------- H4: CAMU ----------
    print("\nH4 (CAMU = ΔTSR_dep per Δ$ vs C0; rankings TSR vs CAMU)")
    if all(r["cost_usd"] == 0 for r in clean):
        print("  SKIPPED: all costs are $0 (mock model or missing price "
              "data) — CAMU is undefined; H4 requires a real-model run.")
        if any(r["model"] == "mock" for r in rows):
            print("\nNOTE: rows with model='mock' validate the pipeline "
                  "only; they are NOT evidence about LLM agents and must "
                  "not be reported as results.")
        return
    c0_cost = mean(r["cost_usd"] for r in sel(clean, "C0"))
    scores = {}
    for c in conditions:
        if c == "C0":
            continue
        dtsr = (mean(r["success"] for r in sel(clean, c, True))
                - mean(r["success"] for r in c0_dep))
        dcost = mean(r["cost_usd"] for r in sel(clean, c)) - c0_cost
        camu = dtsr / dcost if dcost > 0 else float("inf")
        scores[c] = (mean(r["success"] for r in sel(clean, c, True)), camu)
        be = dcost / dtsr if dtsr > 0 else float("inf")
        print(f"  {c}: ΔTSR={dtsr:+.3f}  Δ$/ep={dcost:+.5f}  "
              f"CAMU={camu:.2f} pts/$  break-even task value=${be:.4f}")
    tsr_rank = sorted(scores, key=lambda c: -scores[c][0])
    camu_rank = sorted(scores, key=lambda c: -scores[c][1])
    print(f"  TSR ranking : {' > '.join(tsr_rank)}")
    print(f"  CAMU ranking: {' > '.join(camu_rank)}")
    print(f"  H4 {'SUPPORTED' if tsr_rank != camu_rank else 'not supported'}"
          f" (rankings {'differ' if tsr_rank != camu_rank else 'identical'})")

    if any(r["model"] == "mock" for r in rows):
        print("\nNOTE: rows with model='mock' validate the pipeline only; "
              "they are NOT evidence about LLM agents and must not be "
              "reported as results.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs/pilot/results.jsonl")
