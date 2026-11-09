"""Analysis of the revision-stage experiments (runs/revision), combined with
the preregistered grid (runs/phasec) and the D1 corruption sweep
(runs/phaseb/d1-corrupt). Not preregistered; reported as revision-stage.

Sections
  A  hard tier by model x domain x condition, seed mean +- SD
  B  pooled, paired, arc-clustered hard-tier contrasts per model
  C  temporal conditions on easy/medium tiers (no cost where not needed?)
  D  seed spread per model and condition; reviewer-cited cells
  E  stale-memory harm in all three domains, Holm within domain and jointly
  F  long-history stress test (20 episodes/arc) vs the 5-episode hard tier

Usage: PYTHONPATH=. .venv/bin/python scripts/analyze_revision.py [section...]
"""
from __future__ import annotations

import importlib.util
import itertools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "analyze", str(Path(__file__).parent / "analyze.py"))
az = importlib.util.module_from_spec(spec)
spec.loader.exec_module(az)

MODELS = {"gpt-4.1-mini": "mini", "gpt-4.1": "gpt41",
          "anthropic/claude-haiku-4-5-20251001": "haiku45"}
GRID_DIRS = ["runs/phasec/mini3-*", "runs/phasec/gpt41-*",
             "runs/phasec/haiku45-*", "runs/revision/mini3-temporal-*",
             "runs/revision/gpt41-temporal-*", "runs/revision/gpt41-s12-*",
             "runs/revision/haiku45-temporal-*", "runs/revision/haiku45-s12-*"]
DOMS, TIERS = ("d1", "d2", "d3"), ("easy", "medium", "hard")


def rows_from(globs, clean_only=True):
    out = []
    for g in globs:
        for d in sorted(Path(".").glob(g)):
            f = d / "results.jsonl"
            if not d.is_dir() or not f.exists():
                continue
            for line in f.read_text().splitlines():
                r = json.loads(line)
                if clean_only and r["corrupt_mode"] != "none":
                    continue
                # arc ids already encode the seed; prefix the domain so
                # pooled bootstraps keep domains distinct
                r["arc_id"] = f"{r['domain']}:{r['arc_id']}"
                out.append(r)
    return out


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def sd(xs):
    xs = list(xs)
    if len(xs) < 2:
        return float("nan")
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def dep(rows, **kw):
    return [r for r in rows if r["dependent"]
            and all(r[k] == v for k, v in kw.items())]


def contrast(a, b):
    pt, ci, p = az.boot_ci_delta(a, b, key=lambda r: r["success"])
    return f"{pt:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}] p={p:.4f}"


GRID = None


def grid():
    global GRID
    if GRID is None:
        GRID = rows_from(GRID_DIRS)
    return GRID


def seeds_of(rows):
    return sorted({r["seed"] for r in rows})


def cell_tsr_by_seed(rows):
    by = defaultdict(list)
    for r in rows:
        by[r["seed"]].append(r["success"])
    return [mean(v) for _, v in sorted(by.items())]


def section_a(conds=("C2", "C2T", "C2TR", "C3", "C4", "C5")):
    print("\n=== A: hard tier, dependent TSR, seed mean +- SD (n seeds) ===")
    g = grid()
    for m, short in MODELS.items():
        mr = [r for r in g if r["model"] == m and r["difficulty"] == "hard"]
        if not mr:
            continue
        print(f"\n{short}")
        print("  dom  " + "  ".join(f"{c:>15}" for c in conds))
        for d in DOMS:
            cells = []
            for c in conds:
                v = cell_tsr_by_seed(dep(mr, domain=d, condition=c))
                cells.append(f"{mean(v):.2f}+-{sd(v):.2f}({len(v)})"
                             if v else "--")
            print(f"  {d}   " + "  ".join(f"{x:>15}" for x in cells))


def section_b():
    print("\n=== B: hard tier, pooled over domains and seeds, paired "
          "arc-clustered bootstrap ===")
    g = grid()
    pairs = [("C2T", "C2"), ("C2TR", "C2"), ("C2TR", "C2T"), ("C3", "C2"),
             ("C4", "C2"), ("C3", "C2T"), ("C4", "C2T"), ("C1", "C2T"),
             ("C4", "C5"), ("C2T", "C5")]
    for m, short in MODELS.items():
        mr = [r for r in g if r["model"] == m and r["difficulty"] == "hard"]
        if not mr:
            continue
        print(f"\n{short} (seeds {seeds_of(mr)})")
        for a, b in pairs:
            ra, rb = dep(mr, condition=a), dep(mr, condition=b)
            if ra and rb:
                print(f"  {a:>4} - {b:<4} {contrast(ra, rb)}  "
                      f"(TSR {mean(r['success'] for r in ra):.2f} vs "
                      f"{mean(r['success'] for r in rb):.2f}, n={len(ra)})")
        # per-domain C2T - C2
        for d in DOMS:
            ra = dep(mr, condition="C2T", domain=d)
            rb = dep(mr, condition="C2", domain=d)
            if ra and rb:
                print(f"    {d} C2T - C2  {contrast(ra, rb)}")


def section_c():
    print("\n=== C: temporal conditions on easy and medium tiers ===")
    g = grid()
    for m, short in MODELS.items():
        for tier in ("easy", "medium"):
            mr = [r for r in g if r["model"] == m and r["difficulty"] == tier]
            if not dep(mr, condition="C2T"):
                continue
            for a in ("C2T", "C2TR"):
                print(f"  {short} {tier:6} {a}-C2  "
                      f"{contrast(dep(mr, condition=a), dep(mr, condition='C2'))}")
            ind = {c: mean(r["success"] for r in mr if not r["dependent"]
                           and r["condition"] == c)
                   for c in ("C0", "C2", "C2T", "C2TR")}
            print(f"    independent TSR " + "  ".join(
                f"{c}={v:.2f}" for c, v in ind.items()))


def section_d():
    print("\n=== D: seed spread (mean over cells of max pairwise gap) ===")
    g = grid()
    for m, short in MODELS.items():
        mr = [r for r in g if r["model"] == m]
        if len(seeds_of(mr)) < 2:
            continue
        line = []
        for c in ("C1", "C2", "C2T", "C2TR", "C3", "C4", "C5"):
            gaps = []
            for d, t in itertools.product(DOMS, TIERS):
                v = cell_tsr_by_seed(dep(mr, domain=d, difficulty=t,
                                         condition=c))
                if len(v) >= 2:
                    gaps.append(max(v) - min(v))
            if gaps:
                line.append(f"{c}={mean(gaps):.2f}(max {max(gaps):.2f})")
        print(f"  {short} seeds {seeds_of(mr)}: " + "  ".join(line))
    print("\n  reviewer-cited cells (per-seed TSR, pooled k/n, Wilson 95%)")
    for short, d, t, c in [("gpt41", "d2", "medium", "C4"),
                           ("haiku45", "d2", "medium", "C4"),
                           ("haiku45", "d3", "hard", "C2"),
                           ("gpt41", "d3", "hard", "C2")]:
        m = next(k for k, v in MODELS.items() if v == short)
        rs = dep([r for r in g if r["model"] == m], domain=d, difficulty=t,
                 condition=c)
        if not rs:
            continue
        k = sum(r["success"] for r in rs)
        lo, hi = wilson(k, len(rs))
        v = cell_tsr_by_seed(rs)
        print(f"  {short} {d}-{t} {c}: seeds {[round(x, 2) for x in v]}  "
              f"{k}/{len(rs)} = {k / len(rs):.2f} [{lo:.2f}, {hi:.2f}]")
    print("\n  hybrid vs fact store per domain (hard; C4 - C5)")
    for m, short in MODELS.items():
        mr = [r for r in g if r["model"] == m and r["difficulty"] == "hard"]
        for d in DOMS:
            ra, rb = dep(mr, condition="C4", domain=d), dep(mr, condition="C5",
                                                             domain=d)
            if ra and rb:
                print(f"  {short} {d}: {contrast(ra, rb)}")


def section_e():
    print("\n=== E: stale-memory harm (dependent TSR clean - corrupted) ===")
    srcs = {"d1": ["runs/phaseb/d1-corrupt", "runs/revision/corrupt-d1-temporal"],
            "d2": ["runs/revision/corrupt-d2"],
            "d3": ["runs/revision/corrupt-d3"]}
    allres, allp = {}, {}
    for d, globs in srcs.items():
        rows = rows_from(globs, clean_only=False)
        if not rows:
            print(f"  {d}: no data yet")
            continue
        clean = [r for r in rows if r["corrupt_mode"] == "none"]
        cor = [r for r in rows if r["corrupt_mode"] != "none"]
        res, pv = {}, {}
        for c in sorted({r["condition"] for r in cor}):
            for mode in ("stale", "contradiction", "distractor"):
                for rate in (0.1, 0.3):
                    cr = dep(cor, condition=c, corrupt_mode=mode,
                             corrupt_rate=rate)
                    cl = dep(clean, condition=c)
                    if not cr or not cl:
                        continue
                    pt, ci, p = az.boot_ci_delta(cl, cr,
                                                 key=lambda r: r["success"])
                    res[(d, c, mode, rate)] = (pt, ci, p)
                    pv[(d, c, mode, rate)] = p
        adj = az.holm(pv)
        allres.update(res)
        allp.update(pv)
        print(f"\n  {d}: clean dependent TSR " + "  ".join(
            f"{c}={mean(r['success'] for r in dep(clean, condition=c)):.2f}"
            for c in sorted({r['condition'] for r in clean})))
        for k, (pt, ci, p) in res.items():
            if k[2] == "stale" or adj[k] < 0.05:
                mark = "*" if adj[k] < 0.05 else " "
                print(f"    {k[1]:4} {k[2]:13} rho={k[3]}: {pt:+.2f} "
                      f"[{ci[0]:+.2f}, {ci[1]:+.2f}] p={p:.4f} "
                      f"holm(domain)={adj[k]:.4f} {mark}")
        sig = [k for k in res if adj[k] < 0.05]
        print(f"    significant after Holm within {d}: {len(sig)} of {len(res)}")
    if allp:
        jadj = az.holm(allp)
        sig = sorted(k for k in allp if jadj[k] < 0.05)
        print(f"\n  joint Holm across all domains ({len(allp)} tests): "
              f"{len(sig)} significant")
        for k in sig:
            pt, ci, p = allres[k]
            print(f"    {k}: {pt:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}] "
                  f"holm={jadj[k]:.4f}")
        # pooled stale harm at rho=0.3 per condition across domains
        print("\n  stale rho=0.3 pooled over domains (paired, arc-clustered)")
        rows = rows_from([g for v in srcs.values() for g in v],
                         clean_only=False)
        for c in ("C1", "C2", "C2T", "C2TR", "C3", "C4", "C5"):
            cl = dep(rows, condition=c, corrupt_mode="none")
            cr = dep(rows, condition=c, corrupt_mode="stale", corrupt_rate=0.3)
            if cl and cr:
                print(f"    {c:4} {contrast(cl, cr)}  "
                      f"domains={sorted({r['domain'] for r in cr})}")


def section_f():
    print("\n=== F: long-history stress test (hard, 20 episodes/arc) ===")
    long_rows = rows_from(["runs/revision/long-mini-*"])
    if not long_rows:
        print("  no data yet")
        return
    short = [r for r in rows_from(["runs/phasec/mini3-*-hard",
                                   "runs/revision/mini3-temporal-*-hard"])
             if r["seed"] == 0]
    conds = ["C0", "C1", "C2", "C2T", "C2TR", "C3", "C4", "C5"]

    def summary(rows, label):
        print(f"\n  {label}")
        c0 = [r for r in rows if r["condition"] == "C0"]
        c0_cost = mean(r["cost_usd"] for r in c0)
        c0_dep = mean(r["success"] for r in dep(c0))
        print(f"  {'cond':5} {'TSR_dep':>8} {'n':>4} {'TSR_ind':>8} "
              f"{'tok/ep':>7} {'$/ep(1e-4)':>10} {'CAMU':>7} {'had':>5} "
              f"{'ignore':>6}")
        for c in conds:
            rs = [r for r in rows if r["condition"] == c]
            if not rs:
                continue
            d = dep(rs)
            had = [r for r in d if r["memory_had_fact"]]
            ign = 1 - mean(r["memory_utilized"] for r in had) if had else float("nan")
            cost = mean(r["cost_usd"] for r in rs)
            tok = mean(r["prompt_tokens"] + r["completion_tokens"] for r in rs)
            dts = mean(r["success"] for r in d) - c0_dep
            camu = dts / (cost - c0_cost) if cost > c0_cost else float("nan")
            print(f"  {c:5} {mean(r['success'] for r in d):8.2f} {len(d):4d} "
                  f"{mean(r['success'] for r in rs if not r['dependent']):8.2f} "
                  f"{tok:7.0f} {cost * 1e4:10.1f} {camu:7.0f} "
                  f"{mean(r['memory_had_fact'] for r in d):5.2f} {ign:6.2f}")

    summary(short, "5-episode arcs, hard tier, gpt-4.1-mini seed 0")
    summary(long_rows, "20-episode arcs, hard tier, gpt-4.1-mini seed 0")
    print("\n  pooled contrasts on long arcs (paired, arc-clustered)")
    for a, b in [("C3", "C2"), ("C4", "C2"), ("C2T", "C2"), ("C2TR", "C2"),
                 ("C1", "C2"), ("C3", "C2T"), ("C4", "C5")]:
        ra, rb = dep(long_rows, condition=a), dep(long_rows, condition=b)
        if ra and rb:
            print(f"    {a:>4} - {b:<4} {contrast(ra, rb)}")
    print("\n  per-domain dependent TSR on long arcs")
    for d in DOMS:
        print(f"    {d}: " + "  ".join(
            f"{c}={mean(r['success'] for r in dep(long_rows, domain=d, condition=c)):.2f}"
            for c in conds if dep(long_rows, domain=d, condition=c)))
    # read-block size as a proxy for store size
    print("\n  C1 prompt tokens per episode by episode index (long arcs)")
    by = defaultdict(list)
    for r in long_rows:
        if r["condition"] == "C1":
            by[r["episode_index"]].append(r["prompt_tokens"])
    print("    " + " ".join(f"{i}:{mean(v):.0f}" for i, v in sorted(by.items())))


SECTIONS = {"A": section_a, "B": section_b, "C": section_c, "D": section_d,
            "E": section_e, "F": section_f}

if __name__ == "__main__":
    for s in (sys.argv[1:] or list(SECTIONS)):
        SECTIONS[s.upper()]()
