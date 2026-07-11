"""Phase C analysis: seed robustness + cross-model comparison.

Sweeps every cell (<slice>-<domain>-<tier>/results.jsonl) under a Phase C
runs dir and reports, per agent model:
  1. TSR on dependent tasks per condition x domain x tier, as mean +/- sd
     across seeds (sd only where a cell has >1 seed);
  2. a seed-spread summary for the multi-seed slice (max pairwise TSR gap
     per condition, worst cell called out);
  3. the cross-model hard-tier table (the marquee dissociation: does the
     C2-collapse / C4-hold pattern replicate across agent models?).

Cells still being written are analyzed as-is and flagged, so this is safe
to run mid-grid for a progress readout.

stdlib only (no numpy needed).

Usage: python scripts/analyze_phasec.py [runs/phasec]
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


def mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def sd(xs) -> float:
    xs = list(xs)
    if len(xs) < 2:
        return float("nan")
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def load_cells(root: Path) -> list[dict]:
    rows = []
    for f in sorted(root.glob("*/results.jsonl")):
        cell = f.parent.name
        for line in f.read_text().splitlines():
            if line:
                r = json.loads(line)
                r["cell"] = cell
                rows.append(r)
    if not rows:
        sys.exit(f"no results.jsonl rows under {root}")
    return rows


def tsr_dep(rows) -> float:
    dep = [r for r in rows if r["dependent"] and r["corrupt_mode"] == "none"]
    return mean(r["success"] for r in dep)


def main(root: str) -> None:
    rows = load_cells(Path(root))
    models = sorted({r["model"] for r in rows})
    conditions = sorted({r["condition"] for r in rows})

    # index: (model, domain, difficulty, condition, seed) -> rows
    idx = defaultdict(list)
    for r in rows:
        idx[(r["model"], r["domain"], r["difficulty"],
             r["condition"], r["seed"])].append(r)

    # incomplete-cell warning: a finished cell has equal rows per condition
    per_cell_cond = defaultdict(lambda: defaultdict(int))
    for r in rows:
        per_cell_cond[r["cell"]][r["condition"]] += 1
    for cell, counts in sorted(per_cell_cond.items()):
        if len(set(counts.values())) > 1:
            print(f"NOTE: cell {cell} looks incomplete "
                  f"(rows per condition: {dict(sorted(counts.items()))})")

    # ---------- 1. per-model tables, mean +/- sd across seeds ----------
    for m in models:
        doms = sorted({d for (mm, d, *_ ) in idx if mm == m})
        tiers = sorted({t for (mm, _, t, *_ ) in idx if mm == m})
        print(f"\n=== {m} — TSR_dep, mean±sd across seeds ===")
        print(f"{'cell':12} " + " ".join(f"{c:>12}" for c in conditions))
        for d in doms:
            for t in tiers:
                vals = []
                for c in conditions:
                    seeds = sorted({s for (mm, dd, tt, cc, s) in idx
                                    if (mm, dd, tt, cc) == (m, d, t, c)})
                    per_seed = [tsr_dep(idx[(m, d, t, c, s)]) for s in seeds]
                    if not per_seed:
                        vals.append(f"{'—':>12}")
                    elif len(per_seed) == 1:
                        vals.append(f"{per_seed[0]:12.2f}")
                    else:
                        vals.append(f"{mean(per_seed):6.2f}±{sd(per_seed):.2f}"
                                    .rjust(12))
                if any(v.strip() != "—" for v in vals):
                    print(f"{d}-{t:8} " + " ".join(vals))

    # ---------- 2. seed spread (multi-seed slices only) ----------
    multi = [(m, d, t, c) for m in models
             for d in sorted({dd for (mm, dd, *_ ) in idx if mm == m})
             for t in sorted({tt for (mm, _, tt, *_ ) in idx if mm == m})
             for c in conditions
             if len({s for (mm, dd, tt, cc, s) in idx
                     if (mm, dd, tt, cc) == (m, d, t, c)}) > 1]
    if multi:
        print("\n=== seed robustness: max pairwise TSR_dep gap ===")
        worst = (0.0, None)
        by_cond = defaultdict(list)
        for m, d, t, c in multi:
            seeds = sorted({s for (mm, dd, tt, cc, s) in idx
                            if (mm, dd, tt, cc) == (m, d, t, c)})
            per_seed = [tsr_dep(idx[(m, d, t, c, s)]) for s in seeds]
            gap = max(per_seed) - min(per_seed)
            by_cond[c].append(gap)
            if gap > worst[0]:
                worst = (gap, (m, d, t, c, per_seed))
        for c in conditions:
            if by_cond[c]:
                print(f"  {c}: mean gap {mean(by_cond[c]):.3f}, "
                      f"max {max(by_cond[c]):.3f}")
        if worst[1]:
            m, d, t, c, per_seed = worst[1]
            print(f"  worst cell: {c} {d}-{t} ({m}): "
                  f"per-seed TSR {['%.2f' % v for v in per_seed]}")

    # ---------- 3. cross-model hard tier ----------
    hard_models = [m for m in models
                   if any(tt == "hard" for (mm, _, tt, *_ ) in idx if mm == m)]
    if len(hard_models) > 1:
        print("\n=== hard tier across models (TSR_dep, seed-mean) ===")
        doms = sorted({d for (_, d, tt, *_ ) in idx if tt == "hard"})
        for d in doms:
            print(f"  {d}:")
            print(f"    {'cond':6} " + " ".join(f"{m:>16}" for m in hard_models))
            for c in conditions:
                cells = []
                for m in hard_models:
                    seeds = sorted({s for (mm, dd, tt, cc, s) in idx
                                    if (mm, dd, tt, cc) == (m, d, "hard", c)})
                    per_seed = [tsr_dep(idx[(m, d, "hard", c, s)])
                                for s in seeds]
                    cells.append(f"{mean(per_seed):16.2f}" if per_seed
                                 else f"{'—':>16}")
                print(f"    {c:6} " + " ".join(cells))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs/phasec")
