"""Analysis of the counterfactual replay (scripts/run_position_intervention.py).

Reports, per agent model and pooled:
  * replay fidelity: agreement between the 'original' variant's success and
    the grid run it reconstructs;
  * success rate and stale-action rate per presentation variant;
  * paired contrasts on identical probes (chrono vs antichrono, chrono vs
    original, stamped vs original): risk difference with an arc-clustered
    bootstrap 95% CI and an exact McNemar test on discordant pairs.

Contrasts are computed on all probes and on the subset whose block held
both the stale and the latest value. stdlib only.

Usage: python scripts/analyze_intervention.py [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

SLICE_LABEL = {"mini3": "gpt-4.1-mini", "gpt41": "GPT-4.1",
               "haiku45": "Haiku 4.5"}
CONTRASTS = [("chrono", "antichrono"), ("chrono", "original"),
             ("antichrono", "original"),
             ("stamped", "original")]


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def paired(rows_by_probe: dict, a: str, b: str, key, subset=None,
           n_boot: int = 10_000, seed: int = 0) -> dict:
    probes = [p for p, v in rows_by_probe.items() if a in v and b in v
              and (subset is None or subset(v))]
    if not probes:
        return {"n": 0}
    xa = {p: key(rows_by_probe[p][a]) for p in probes}
    xb = {p: key(rows_by_probe[p][b]) for p in probes}
    diff = sum(xa[p] - xb[p] for p in probes) / len(probes)
    b_ = sum(xa[p] and not xb[p] for p in probes)
    c_ = sum(xb[p] and not xa[p] for p in probes)
    clusters = defaultdict(list)
    for p in probes:
        clusters[p[:4]].append(p)  # (slice, domain, seed, arc)
    keys = list(clusters)
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        sample = [p for _ in keys for p in clusters[rng.choice(keys)]]
        boots.append(sum(xa[p] - xb[p] for p in sample) / len(sample))
    boots.sort()
    return {"n": len(probes), "a_rate": sum(xa.values()) / len(probes),
            "b_rate": sum(xb.values()) / len(probes), "diff": diff,
            "ci": (boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]),
            "discordant": (b_, c_), "p_mcnemar": mcnemar_exact(b_, c_)}


def fmt(r: dict) -> str:
    if not r["n"]:
        return "n=0"
    return (f"{r['a_rate']:.2f} vs {r['b_rate']:.2f}  diff {r['diff']:+.2f} "
            f"[{r['ci'][0]:+.2f}, {r['ci'][1]:+.2f}]  discordant "
            f"{r['discordant'][0]}/{r['discordant'][1]}  "
            f"McNemar p={r['p_mcnemar']:.2g}  (n={r['n']})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    rows = [json.loads(l) for l in
            Path("runs/intervention/results.jsonl").open() if l.strip()]
    out = {}
    groups = {**{s: [r for r in rows if r["slice"] == s]
                 for s in SLICE_LABEL}, "pooled": rows}
    for g, rs in groups.items():
        if not rs:
            continue
        by = defaultdict(dict)
        for r in rs:
            by[(r["slice"], r["domain"], r["seed"], r["arc_id"],
                r["task_id"])][r["variant"]] = r
        label = SLICE_LABEL.get(g, g)
        print(f"\n=== {label}: {len(by)} probes, "
              f"${sum(r['cost_usd'] for r in rs):.2f} ===")
        orig = [v["original"] for v in by.values() if "original" in v]
        fid = sum(r["success"] == r["grid_success"] for r in orig)
        print(f"replay fidelity (original vs grid success): {fid}/{len(orig)}"
              f" = {fid / len(orig):.2f}; grid TSR "
              f"{sum(r['grid_success'] for r in orig) / len(orig):.2f}")
        res = {"probes": len(by), "fidelity": [fid, len(orig)]}
        for v in ("original", "chrono", "antichrono", "stamped"):
            vs = [x[v] for x in by.values() if v in x]
            if vs:
                tsr = sum(r["success"] for r in vs) / len(vs)
                st = sum(r["action"] == "stale" for r in vs) / len(vs)
                print(f"  {v:10} TSR {tsr:.2f}  stale action {st:.2f} "
                      f"(n={len(vs)})")
                res[v] = {"tsr": tsr, "stale": st, "n": len(vs)}
        both = lambda v: v.get("original", {}).get("state") == "both"
        for a, b in CONTRASTS:
            for name, keyf in (("success", lambda r: r["success"]),
                               ("stale action",
                                lambda r: r["action"] == "stale")):
                r_all = paired(by, a, b, keyf)
                r_both = paired(by, a, b, keyf, subset=both)
                print(f"  {a} vs {b}, {name}: all {fmt(r_all)}")
                print(f"  {'':>{len(a) + len(b) + 4}}{'':>{len(name)}}   "
                      f"both-in-memory {fmt(r_both)}")
                res[f"{a}-{b}-{name}"] = {"all": r_all, "both": r_both}
        out[label] = res
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
