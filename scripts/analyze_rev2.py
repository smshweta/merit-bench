"""Second revision round (Discover AI): conflict-aware retrieval (C2C) and
the implicit-update tier. Not preregistered; reported as revision-stage.

Sections
  G  C2C on the 3x3 grid: hard-tier pooled contrasts per model, per-domain
     C2C - C2, easy/medium no-harm check, memory-side cost per episode
  H  implicit-update tier: TSR by model x domain x condition, implicit minus
     hard per condition (paired on identical arcs), and within-tier
     contrasts per model with Holm correction per model

Usage: PYTHONPATH=. .venv/bin/python scripts/analyze_rev2.py [G|H ...]
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

MODELS = rev.MODELS
DOMS, TIERS = rev.DOMS, rev.TIERS
C2C_DIRS = ["runs/revision/mini-c2c-*", "runs/revision/gpt41-c2c-*",
            "runs/revision/haiku45-c2c-*"]
IMPL_DIRS = ["runs/revision/mini-implicit-*", "runs/revision/gpt41-implicit-*",
             "runs/revision/haiku45-implicit-*"]
ALL9 = ("C0", "C1", "C2", "C2T", "C2TR", "C2C", "C3", "C4", "C5")

dep, mean, sd = rev.dep, rev.mean, rev.sd


def pc(a, b):
    pt, ci, p = az.boot_ci_delta(a, b, key=lambda r: r["success"])
    return pt, ci, p


def fmt(pt, ci, p):
    return f"{pt:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}] p={p:.4f}"


def tsr(rs):
    return mean(r["success"] for r in rs)


def section_g():
    print("\n=== G: conflict-aware retrieval (C2C), grid ===")
    g = rev.grid() + rev.rows_from(C2C_DIRS)
    pairs = [("C2C", "C2"), ("C2C", "C2T"), ("C2C", "C2TR"),
             ("C3", "C2C"), ("C4", "C2C"), ("C1", "C2C")]
    for m, short in MODELS.items():
        mr = [r for r in g if r["model"] == m and r["difficulty"] == "hard"]
        if not dep(mr, condition="C2C"):
            continue
        print(f"\n{short} hard (seeds {rev.seeds_of(dep(mr, condition='C2C'))})")
        for a, b in pairs:
            ra, rb = dep(mr, condition=a), dep(mr, condition=b)
            if ra and rb:
                print(f"  {a:>4} - {b:<4} {fmt(*pc(ra, rb))}  "
                      f"(TSR {tsr(ra):.2f} vs {tsr(rb):.2f}, n={len(ra)})")
        for d in DOMS:
            ra = dep(mr, condition="C2C", domain=d)
            rb = dep(mr, condition="C2", domain=d)
            v = rev.cell_tsr_by_seed(ra)
            print(f"    {d} C2C {mean(v):.2f}+-{sd(v):.2f}  "
                  f"C2C - C2 {fmt(*pc(ra, rb))}")
        hf = [r["memory_had_fact"] for r in dep(mr, condition="C2C")]
        print(f"    C2C memory held the current value: {mean(hf):.2f}")
        for tier in ("easy", "medium"):
            tr = [r for r in g if r["model"] == m and r["difficulty"] == tier]
            ra, rb = dep(tr, condition="C2C"), dep(tr, condition="C2")
            if ra and rb:
                ind = mean(r["success"] for r in tr if not r["dependent"]
                           and r["condition"] == "C2C")
                print(f"  {tier:6} C2C - C2 {fmt(*pc(ra, rb))}  "
                      f"(TSR {tsr(ra):.2f} vs {tsr(rb):.2f}; "
                      f"independent C2C {ind:.2f})")
        # memory-side + agent cost per episode, all tiers
        allm = [r for r in g if r["model"] == m]
        for c in ("C2", "C2T", "C2C", "C4"):
            rs = [r for r in allm if r["condition"] == c]
            if rs:
                # cost_usd already includes the memory-side cost
                memc = mean(r.get("memory_cost_usd", 0) or 0 for r in rs)
                agent = mean(r.get("cost_usd", 0) for r in rs) - memc
                print(f"    cost/episode {c:>4}: agent ${agent*1000:.3f}m "
                      f"+ memory ${memc*1000:.3f}m (x1e-3 USD)")


def section_h():
    print("\n=== H: implicit-update tier ===")
    imp = rev.rows_from(IMPL_DIRS)
    hard = rev.grid() + rev.rows_from(C2C_DIRS)
    hard = [r for r in hard if r["difficulty"] == "hard"]
    for m, short in MODELS.items():
        mi = [r for r in imp if r["model"] == m]
        if not mi:
            continue
        mh = [r for r in hard if r["model"] == m]
        print(f"\n{short} (seeds {rev.seeds_of(mi)})")
        print("  dom  " + "  ".join(f"{c:>12}" for c in ALL9))
        for d in DOMS:
            cells = []
            for c in ALL9:
                v = rev.cell_tsr_by_seed(dep(mi, domain=d, condition=c))
                cells.append(f"{mean(v):.2f}+-{sd(v):.2f}" if v else "--")
            print(f"  {d}   " + "  ".join(f"{x:>12}" for x in cells))
        print("  pooled implicit TSR / implicit - hard (paired arcs):")
        for c in ALL9[1:]:
            ri, rh = dep(mi, condition=c), dep(mh, condition=c)
            if ri and rh:
                print(f"    {c:>4} {tsr(ri):.2f} (hard {tsr(rh):.2f})  "
                      f"{fmt(*pc(ri, rh))}")
        fam = {}
        for a, b in [("C2T", "C2"), ("C2TR", "C2"), ("C2C", "C2"),
                     ("C3", "C2"), ("C4", "C2"), ("C1", "C2"),
                     ("C4", "C5"), ("C3", "C2C"), ("C4", "C2C")]:
            ra, rb = dep(mi, condition=a), dep(mi, condition=b)
            if ra and rb:
                fam[f"{a} - {b}"] = pc(ra, rb)
        adj = az.holm({k: v[2] for k, v in fam.items()})
        print("  within-tier contrasts (Holm over this family):")
        for k, (pt, ci, p) in fam.items():
            print(f"    {k:<10} {fmt(pt, ci, p)}  p_Holm={adj.get(k, 1):.4f}")
        st = []
        for c in ("C2", "C2T", "C2C", "C4"):
            rs = dep(mi, condition=c)
            st.append(f"{c} held={mean(r['memory_had_fact'] for r in rs):.2f}")
        print("    memory held current value: " + "  ".join(st))
    # pooled over models
    print("\npooled over models, implicit - hard:")
    for c in ALL9[1:]:
        ri = dep(imp, condition=c)
        rh = [r for r in hard if r["dependent"] and r["condition"] == c
              and r["model"] in {x["model"] for x in ri}]
        if ri and rh:
            # pair across models: prefix model into the arc key
            ki = [dict(r, arc_id=f"{r['model']}|{r['arc_id']}") for r in ri]
            kh = [dict(r, arc_id=f"{r['model']}|{r['arc_id']}") for r in rh]
            print(f"  {c:>4} {tsr(ki):.2f} vs {tsr(kh):.2f}  "
                  f"{fmt(*pc(ki, kh))}")


if __name__ == "__main__":
    secs = sys.argv[1:] or ["G", "H"]
    for s in secs:
        {"G": section_g, "H": section_h}[s.upper()]()
