"""Hard-tier (updated-fact) failure taxonomy from released traces ($0).

For every scored hard-tier probe, regenerates its arc deterministically to
recover BOTH values of the fact: the superseded (stale) value from the plant
episode and the latest value the probe requires. Each probe is then
classified on two axes:

  memory state  what the memory block shown to the agent contained:
                both | latest_only | stale_only | neither
  action        what the agent passed to the domain's target tool
                (D1 refund, D2 deploy, D3 create_event):
                latest | stale | other_value | no_action

and, for probes whose block held both values, whether the stale value was
positioned before or after the latest one (the order signal an agent could
use to arbitrate recency).

Reports, per run group and condition: the memory-state distribution, the
action distribution, the stale-action rate conditional on both values
being present, and Wilson 95% CIs. stdlib only.

Usage: PYTHONPATH=. python scripts/analyze_failures.py [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from merit import arcs as d1, d2, d3
from merit.domains import DOMAINS

TARGET_TOOL = {"d1": "refund", "d2": "deploy", "d3": "create_event"}
MODULES = {"d1": d1, "d2": d2, "d3": d3}

# run group -> glob of hard-tier cell dirs (relative to repo root)
GROUPS = {
    "starter/gpt-4.1-mini": ["runs/sweep/d1-hard", "runs/sweep/d2-hard",
                             "runs/sweep/d3-hard"],
    "real/gpt-4.1-mini (pilot)": ["runs/phaseb/d1-hard", "runs/phaseb/d2-hard",
                                  "runs/phaseb/d3-hard"],
    "grid/gpt-4.1-mini x3 seeds": ["runs/phasec/mini3-d*-hard"],
    "grid/gpt-4.1": ["runs/phasec/gpt41-d*-hard"],
    "grid/claude-haiku-4.5": ["runs/phasec/haiku45-d*-hard"],
    "spot/claude-sonnet-5 (D1)": ["runs/phasec/sonnet5-d1-hard"],
}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def fact_pairs(domain: str, seed: int, n_arcs: int) -> dict:
    """(arc_id, task_id) -> (stale, latest) for every hard probe, recovered by
    recording every TaskSpec passed to the generator's _merge (plants that
    share an episode slot are merged away, so the final arc alone loses them)."""
    mod = MODULES[domain]
    seen: list = []
    orig = mod._merge

    def spy(existing, new):
        # snapshot now: _merge later appends to the object in place
        seen.append((new.task_id, new.user_messages[0],
                     dict(new.checker_args)))
        return orig(existing, new)

    mod._merge = spy
    try:
        arcs = DOMAINS[domain].generate_suite(
            n_arcs=n_arcs, episodes_per_arc=5, dep_ratio=0.5,
            base_seed=seed, difficulty="hard")
    finally:
        mod._merge = orig
    out = {}
    for arc in arcs:
        tasks = [t for t in seen if t[0].startswith(arc.arc_id + "-")]
        for ep in arc.episodes:
            t = ep.task
            if not (t.dependent and t.kind.endswith("_upd")):
                continue
            probe_msg = " ".join(t.user_messages)
            prefix = f"{arc.arc_id}-e{t.plant_episode}-plant"
            plants = [p for p in tasks if p[0].startswith(prefix)]
            # disambiguate plants sharing a slot by the probe's entity
            ent = _entity(domain, t, probe_msg)
            plants = [p for p in plants if ent in p[1]]
            assert len(plants) == 1, (arc.arc_id, t.task_id, ent, plants)
            out[(arc.arc_id, t.task_id)] = (
                plants[0][2]["must_contain"], t.gold_fact_value,
                t.user_messages[0])
    return out


def _entity(domain: str, task, probe_msg: str) -> str:
    if domain == "d1":
        return task.checker_args["order_id"]
    if domain == "d2":
        return task.checker_args["service"]
    m = re.search(r"dinner with (.+?) to the calendar", probe_msg)
    return m.group(1)


def memory_block(transcript: str, first_user_msg: str) -> str:
    """The block sits between the header and the probe's first user turn.
    Anchor on that exact turn: replayed/retrieved transcripts inside the
    block contain '[user]' lines of their own."""
    head = "[memory shown]\n"
    end = transcript.find("\n[user] " + first_user_msg)
    assert transcript.startswith(head) and end >= 0
    return transcript[len(head):end]


def classify(domain: str, stale: str, latest: str, first_msg: str,
             rec: dict) -> dict:
    block = memory_block(rec["transcript"], first_msg)
    has_l, has_s = latest in block, stale in block
    state = ("both" if has_l and has_s else "latest_only" if has_l
             else "stale_only" if has_s else "neither")
    calls = [c for c in rec["tool_calls"] if c["name"] == TARGET_TOOL[domain]]
    blob = " | ".join(json.dumps(c["args"]) for c in calls)
    if not calls:
        action = "no_action"
    elif latest in blob:
        action = "latest"
    elif stale in blob:
        action = "stale"
    else:
        action = "other_value"
    order = None
    if state == "both":
        order = ("stale_last" if block.rfind(stale) > block.rfind(latest)
                 else "latest_last")
    return {"state": state, "action": action, "order": order}


def load_group(patterns: list[str]) -> list[dict]:
    rows = []
    for pat in patterns:
        for cell in sorted(Path(".").glob(pat)):
            res = cell / "results.jsonl"
            if not res.exists():
                continue
            traces = {}
            for line in (cell / "traces" / "episodes.jsonl").open():
                t = json.loads(line)
                traces[t["episode_id"]] = t
            cell_rows = [json.loads(l) for l in res.open() if l.strip()]
            pairs = {}
            for r in cell_rows:
                if not (r["dependent"] and r["difficulty"] == "hard"
                        and r["corrupt_mode"] == "none"):
                    continue
                key = (r["domain"], r["seed"])
                if key not in pairs:
                    n_arcs = len({x["arc_id"] for x in cell_rows
                                  if x["seed"] == r["seed"]})
                    pairs[key] = fact_pairs(r["domain"], r["seed"], n_arcs)
                stale, latest, first_msg = pairs[key][(r["arc_id"],
                                                      r["task_id"])]
                c = classify(r["domain"], stale, latest, first_msg,
                             traces[r["episode_id"]])
                rows.append({**r, **c})
    return rows


def pct(k: int, n: int) -> str:
    return f"{k / n:.2f}" if n else "  - "


def report(name: str, rows: list[dict]) -> dict:
    print(f"\n=== {name}: {len(rows)} hard-tier probes ===")
    print(f"{'cond':4} {'n':>4} | state: both  L-only S-only none | "
          f"action: latest stale other none | P(stale|both) [95% CI]  "
          f"P(stale|both,stale_last) P(stale|both,latest_last) | TSR")
    summary = {}
    by = defaultdict(list)
    for r in rows:
        by[r["condition"]].append(r)
    for cond in sorted(by):
        rs = by[cond]
        n = len(rs)
        st = Counter(r["state"] for r in rs)
        ac = Counter(r["action"] for r in rs)
        both = [r for r in rs if r["state"] == "both"]
        k_both = sum(r["action"] == "stale" for r in both)
        lo, hi = wilson(k_both, len(both))
        sl = [r for r in both if r["order"] == "stale_last"]
        ll = [r for r in both if r["order"] == "latest_last"]
        k_sl = sum(r["action"] == "stale" for r in sl)
        k_ll = sum(r["action"] == "stale" for r in ll)
        tsr = sum(r["success"] for r in rs) / n
        print(f"{cond:4} {n:4} |        {pct(st['both'], n)}  "
              f"{pct(st['latest_only'], n)}   {pct(st['stale_only'], n)}  "
              f"{pct(st['neither'], n)} |        {pct(ac['latest'], n)}   "
              f"{pct(ac['stale'], n)}  {pct(ac['other_value'], n)} "
              f"{pct(ac['no_action'], n)} | {k_both:3}/{len(both):<3} "
              f"= {pct(k_both, len(both))} [{lo:.2f},{hi:.2f}]   "
              f"{k_sl}/{len(sl)} = {pct(k_sl, len(sl))}      "
              f"{k_ll}/{len(ll)} = {pct(k_ll, len(ll))} | {tsr:.2f}")
        summary[cond] = {
            "n": n, "state": dict(st), "action": dict(ac),
            "stale_given_both": [k_both, len(both)],
            "stale_given_both_ci": [lo, hi],
            "stale_given_stale_last": [k_sl, len(sl)],
            "stale_given_latest_last": [k_ll, len(ll)],
            "tsr": tsr,
            "by_domain": {
                d: {"n": len(x),
                    "state": dict(Counter(r["state"] for r in x)),
                    "action": dict(Counter(r["action"] for r in x)),
                    "stale_given_both": [
                        sum(r["action"] == "stale" for r in x
                            if r["state"] == "both"),
                        sum(r["state"] == "both" for r in x)]}
                for d in sorted({r["domain"] for r in rs})
                for x in [[r for r in rs if r["domain"] == d]]},
        }
    return summary


def fisher_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Exact two-sided p for the 2x2 table [[a, b], [c, d]]."""
    n1, n2, k = a + b, c + d, a + c
    def hyp(x):
        return (math.comb(n1, x) * math.comb(n2, k - x)
                / math.comb(n1 + n2, k))
    p_obs = hyp(a)
    lo, hi = max(0, k - n2), min(k, n1)
    return min(1.0, sum(hyp(x) for x in range(lo, hi + 1)
                        if hyp(x) <= p_obs * (1 + 1e-9)))


def order_effect(rows: list[dict], conds: tuple[str, ...],
                 n_boot: int = 10_000, seed: int = 0) -> dict:
    """Exploratory: among probes whose block held both values, is a stale
    action more likely when the stale value is positioned last? Exact
    Fisher test plus a bootstrap CI for the risk difference that resamples
    whole arcs (probes in one arc share a world and a memory history)."""
    import random
    both = [r for r in rows if r["condition"] in conds
            and r["state"] == "both"]
    def rates(rs):
        sl = [r["action"] == "stale" for r in rs if r["order"] == "stale_last"]
        ll = [r["action"] == "stale" for r in rs if r["order"] == "latest_last"]
        return sl, ll
    sl, ll = rates(both)
    a, b = sum(sl), len(sl) - sum(sl)
    c, d = sum(ll), len(ll) - sum(ll)
    clusters = defaultdict(list)
    for r in both:
        clusters[(r["model"], r["seed"], r["domain"], r["arc_id"])].append(r)
    keys = list(clusters)
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        sample = [r for _ in keys for r in clusters[rng.choice(keys)]]
        s1, s2 = rates(sample)
        if s1 and s2:
            diffs.append(sum(s1) / len(s1) - sum(s2) / len(s2))
    diffs.sort()
    ci = (diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs)) - 1])
    return {"conds": conds, "stale_last": [a, a + b],
            "latest_last": [c, c + d],
            "diff": a / (a + b) - c / (c + d), "diff_ci95": ci,
            "fisher_p": fisher_two_sided(a, b, c, d),
            "n_clusters": len(keys)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    out = {}
    grid_rows = []
    for name, pats in GROUPS.items():
        rows = load_group(pats)
        if rows:
            out[name] = report(name, rows)
        if name.startswith("grid/"):
            grid_rows += rows
    print("\n=== exploratory order effect, full grid (3 agent models) ===")
    out["order_effect"] = {}
    for conds in (("C2",), ("C5",), ("C2", "C5")):
        e = order_effect(grid_rows, conds)
        out["order_effect"]["+".join(conds)] = e
        print(f"{'+'.join(conds):6} P(stale | stale last) "
              f"{e['stale_last'][0]}/{e['stale_last'][1]} vs "
              f"P(stale | latest last) {e['latest_last'][0]}/"
              f"{e['latest_last'][1]}: diff {e['diff']:+.2f} "
              f"[{e['diff_ci95'][0]:+.2f}, {e['diff_ci95'][1]:+.2f}] "
              f"arc-clustered, Fisher p = {e['fisher_p']:.2g} "
              f"({e['n_clusters']} arcs)")
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
