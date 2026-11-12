"""Third revision round (Discover AI): three seeds for the corruption sweeps
and the long-history test, with conflict-aware retrieval (C2C) in both.
Not preregistered; reported as revision-stage.

Sections
  I  corruption sweeps, D1-D3, seeds 0-2, nine conditions: stale harm at
     rho=0.3 per domain and pooled, Holm within domain and jointly
  J  long-history test (20 episodes/arc, hard), seeds 0-2, nine conditions:
     TSR, cost, CAMU, pooled contrasts, per-seed spread

Usage: PYTHONPATH=. .venv/bin/python scripts/analyze_rev3.py [I|J ...]
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "rev", str(Path(__file__).parent / "analyze_revision.py"))
rev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rev)
az = rev.az
dep, mean, sd = rev.dep, rev.mean, rev.sd
DOMS = rev.DOMS
CONDS = ("C0", "C1", "C2", "C2T", "C2TR", "C2C", "C3", "C4", "C5")
MODES = ("stale", "contradiction", "distractor")


def pc(a, b):
    return az.boot_ci_delta(a, b, key=lambda r: r["success"])


def fmt(pt, ci, p):
    return f"{pt:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}] p={p:.4f}"


def corrupt_rows(d):
    globs = [f"runs/revision/corrupt-{d}", f"runs/revision/corrupt-{d}-*"]
    if d == "d1":
        globs.append("runs/phaseb/d1-corrupt")
    return rev.rows_from(globs, clean_only=False)


def section_i():
    print("\n=== I: corruption sweeps, seeds 0-2, nine conditions ===")
    allrows, allres, allp = [], {}, {}
    for d in DOMS:
        rows = corrupt_rows(d)
        allrows += rows
        clean = [r for r in rows if r["corrupt_mode"] == "none"]
        print(f"\n  {d}: seeds {rev.seeds_of(rows)}, {len(rows)} episodes, "
              f"${sum(r['cost_usd'] for r in rows):.2f}")
        print("    clean dependent TSR " + "  ".join(
            f"{c}={mean(r['success'] for r in dep(clean, condition=c)):.2f}"
            f"(n={len(dep(clean, condition=c))})" for c in CONDS))
        res, pv = {}, {}
        for c in CONDS[1:]:
            for mode in MODES:
                for rate in (0.1, 0.3):
                    cr = dep(rows, condition=c, corrupt_mode=mode,
                             corrupt_rate=rate)
                    cl = dep(clean, condition=c)
                    if cr and cl:
                        res[(d, c, mode, rate)] = pc(cl, cr)
                        pv[(d, c, mode, rate)] = res[(d, c, mode, rate)][2]
        adj = az.holm(pv)
        allres.update(res)
        allp.update(pv)
        for k, v in res.items():
            if (k[2] == "stale" and k[3] == 0.3) or adj[k] < 0.05:
                mark = "*" if adj[k] < 0.05 else " "
                print(f"    {k[1]:4} {k[2]:13} rho={k[3]}: {fmt(*v)} "
                      f"holm(domain)={adj[k]:.4f} {mark}")
        sig = [k for k in res if adj[k] < 0.05]
        print(f"    significant after Holm within {d}: {len(sig)} of "
              f"{len(res)}: " + ", ".join(
                  f"{k[1]}/{k[2]}/{k[3]}" for k in sig))
    jadj = az.holm(allp)
    sig = sorted(k for k in allp if jadj[k] < 0.05)
    print(f"\n  joint Holm across domains ({len(allp)} tests): "
          f"{len(sig)} significant")
    for k in sig:
        print(f"    {k}: {fmt(*allres[k])} holm={jadj[k]:.4f}")
    for mode, rate in [("stale", 0.3), ("stale", 0.1),
                       ("contradiction", 0.3), ("distractor", 0.3)]:
        print(f"\n  {mode} rho={rate} pooled over domains and seeds "
              "(paired, arc-clustered; Holm over the 8 conditions)")
        fam = {}
        for c in CONDS[1:]:
            cl = dep(allrows, condition=c, corrupt_mode="none")
            cr = dep(allrows, condition=c, corrupt_mode=mode,
                     corrupt_rate=rate)
            if cl and cr:
                fam[c] = pc(cl, cr) + (len(cr), mean(r["success"] for r in cl),
                                       mean(r["success"] for r in cr))
        adj = az.holm({c: v[2] for c, v in fam.items()})
        for c, (pt, ci, p, n, a, b) in fam.items():
            print(f"    {c:4} clean {a:.2f} corrupted {b:.2f}  "
                  f"{fmt(pt, ci, p)} p_Holm={adj[c]:.4f} n={n}")
    print("\n  stale rho=0.3 harm per seed (pooled over domains)")
    for c in CONDS[1:]:
        out = []
        for s in (0, 1, 2):
            cl = [r for r in dep(allrows, condition=c, corrupt_mode="none")
                  if r["seed"] == s]
            cr = [r for r in dep(allrows, condition=c, corrupt_mode="stale",
                                 corrupt_rate=0.3) if r["seed"] == s]
            if cl and cr:
                out.append(mean(r["success"] for r in cl)
                           - mean(r["success"] for r in cr))
        print(f"    {c:4} " + " ".join(f"{x:+.2f}" for x in out))
    ind = [r for r in allrows if not r["dependent"]]
    print("\n  independent-task TSR, clean vs any corruption: "
          f"{mean(r['success'] for r in ind if r['corrupt_mode'] == 'none'):.3f}"
          f" vs {mean(r['success'] for r in ind if r['corrupt_mode'] != 'none'):.3f}")
    print(f"  total: {len(allrows)} episodes, "
          f"${sum(r['cost_usd'] for r in allrows):.2f}")


def section_j():
    print("\n=== J: long-history test, seeds 0-2, nine conditions ===")
    long_rows = rev.rows_from(["runs/revision/long-mini-*"])
    short = rev.rows_from(["runs/phasec/mini3-*-hard",
                           "runs/revision/mini3-temporal-*-hard",
                           "runs/revision/mini-c2c-*-hard"])
    print(f"  long: seeds {rev.seeds_of(long_rows)}, {len(long_rows)} "
          f"episodes, ${sum(r['cost_usd'] for r in long_rows):.2f}")

    def summary(rows, label):
        print(f"\n  {label}")
        c0 = [r for r in rows if r["condition"] == "C0"]
        c0_cost = mean(r["cost_usd"] for r in c0)
        c0_dep = mean(r["success"] for r in dep(c0))
        print(f"  {'cond':5} {'TSR_dep':>8} {'sd':>5} {'n':>4} {'TSR_ind':>8}"
              f" {'tok/ep':>7} {'$/ep(1e-4)':>10} {'CAMU':>7} {'had':>5} {'ign':>5}")
        for c in CONDS:
            rs = [r for r in rows if r["condition"] == c]
            if not rs:
                continue
            d = dep(rs)
            by_seed = [mean(r["success"] for r in d if r["seed"] == s)
                       for s in sorted({r["seed"] for r in d})]
            cost = mean(r["cost_usd"] for r in rs)
            had = [r for r in d if r["memory_had_fact"]]
            ign = (1 - mean(r["memory_utilized"] for r in had) if had
                   else float("nan"))
            tok = mean(r["prompt_tokens"] + r["completion_tokens"]
                       for r in rs)
            dts = mean(r["success"] for r in d) - c0_dep
            camu = dts / (cost - c0_cost) if cost > c0_cost else float("nan")
            print(f"  {c:5} {mean(r['success'] for r in d):8.2f} "
                  f"{sd(by_seed):5.2f} {len(d):4d} "
                  f"{mean(r['success'] for r in rs if not r['dependent']):8.2f}"
                  f" {tok:7.0f} {cost * 1e4:10.1f} {camu:7.0f} "
                  f"{mean(r['memory_had_fact'] for r in d):5.2f} {ign:5.2f}"
                  f"   seeds: " + " ".join(f"{x:.2f}" for x in by_seed))

    summary(short, "5-episode arcs, hard tier, gpt-4.1-mini, seeds 0-2")
    summary(long_rows, "20-episode arcs, hard tier, gpt-4.1-mini, seeds 0-2")
    print("\n  pooled contrasts on long arcs (paired, arc-clustered; Holm)")
    fam = {}
    for a, b in [("C3", "C2"), ("C4", "C2"), ("C2T", "C2"), ("C2TR", "C2"),
                 ("C2C", "C2"), ("C1", "C2"), ("C3", "C2T"), ("C3", "C2C"),
                 ("C4", "C2C"), ("C2C", "C2TR"), ("C4", "C5")]:
        ra, rb = dep(long_rows, condition=a), dep(long_rows, condition=b)
        if ra and rb:
            fam[f"{a} - {b}"] = pc(ra, rb)
    adj = az.holm({k: v[2] for k, v in fam.items()})
    for k, v in fam.items():
        print(f"    {k:<11} {fmt(*v)} p_Holm={adj[k]:.4f}")
    print("\n  long minus short per condition (unpaired point difference)")
    for c in CONDS[1:]:
        a, b = dep(long_rows, condition=c), dep(short, condition=c)
        if a and b:
            print(f"    {c:4} {mean(r['success'] for r in a):.2f} vs "
                  f"{mean(r['success'] for r in b):.2f}")
    print("\n  per-domain dependent TSR on long arcs")
    for d in DOMS:
        print(f"    {d}: " + "  ".join(
            f"{c}={mean(r['success'] for r in dep(long_rows, domain=d, condition=c)):.2f}"
            for c in CONDS if dep(long_rows, domain=d, condition=c)))
    c0 = dep(long_rows, condition="C0")
    print(f"\n  no-memory successes on long arcs: "
          f"{sum(r['success'] for r in c0)} of {len(c0)}")


if __name__ == "__main__":
    for s in (sys.argv[1:] or ["I", "J"]):
        {"I": section_i, "J": section_j}[s.upper()]()
