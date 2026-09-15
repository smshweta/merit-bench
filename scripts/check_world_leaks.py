"""Residual leak audit over released traces ($0).

The generation-time leak check guarantees that no gold value is in a probe's
inputs or in the INITIAL world state. Agents can still write a fact into the
mutable world during an earlier episode (e.g. set_config) and read it back
later. This script counts, for every scored dependent probe, whether any
read-only tool returned a gold value, and how many no-memory (C0) probes
succeeded at all.

Usage: PYTHONPATH=. python scripts/check_world_leaks.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from merit.domains import DOMAINS

READ_TOOLS = {"get_order", "get_policy", "search_logs", "get_deploy_history",
              "get_config", "get_calendar", "get_preference"}
CELLS = (sorted(Path("runs/phaseb").glob("d?-*"))
         + sorted(Path("runs/phasec").glob("*-d?-*"))
         + sorted(Path("runs/sweep").glob("d?-*")))


def main() -> None:
    cache: dict = {}
    exposed, total, c0_n, c0_succ = Counter(), Counter(), 0, 0
    for cell in CELLS:
        res = cell / "results.jsonl"
        if not res.exists():
            continue
        rows = [json.loads(l) for l in res.open() if l.strip()]
        traces = {}
        for line in (cell / "traces" / "episodes.jsonl").open():
            t = json.loads(line)
            traces[t["episode_id"]] = t
        for r in rows:
            if (not r["dependent"] or r["corrupt_mode"] != "none"
                    or r["episode_id"] not in traces):
                continue
            n_arcs = len({x["arc_id"] for x in rows if x["seed"] == r["seed"]})
            key = (r["domain"], r["difficulty"], r["seed"], n_arcs)
            if key not in cache:
                arcs = DOMAINS[r["domain"]].generate_suite(
                    n_arcs=n_arcs, episodes_per_arc=5, dep_ratio=0.5,
                    base_seed=r["seed"], difficulty=r["difficulty"])
                cache[key] = {(a.arc_id, e.task.task_id): e.task.golds()
                              for a in arcs for e in a.episodes
                              if e.task.dependent}
            golds = cache[key].get((r["arc_id"], r["task_id"]))
            if golds is None:
                continue
            outs = " ".join(c["out"] for c in traces[r["episode_id"]]
                            ["tool_calls"] if c["name"] in READ_TOOLS)
            total[r["condition"]] += 1
            exposed[r["condition"]] += any(g in outs for g in golds)
            if r["condition"] == "C0":
                c0_n += 1
                c0_succ += r["success"]
    n, k = sum(total.values()), sum(exposed.values())
    print(f"dependent probes: {n}; a read-only tool returned a gold value in "
          f"{k} ({k / n:.2%}): {dict(sorted(exposed.items()))}")
    print(f"no-memory (C0) dependent probes: {c0_n}; successes: {c0_succ}")


if __name__ == "__main__":
    main()
