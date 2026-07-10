"""MUR human audit: sample episodes for annotation, then score Cohen's kappa.

The MUR value-tracer (merit/metrics.py:memory_utilized) is conservative
string containment; the preregistered validation is a 100-episode human
audit with two annotators. This script:

  sample  draws a stratified sample (by condition x auto-label, spread over
          domain/tier) from episodes where memory held every gold value,
          and writes to runs/mur_audit/:
            audit_sheet.html  what annotators read (memory + transcript +
                              gold values; NO auto label, so they are blind)
            labels.csv        item_id + empty annotator1/annotator2 columns;
                              fill with yes / no / unclear
            key.csv           item_id -> auto tracer label (keep from
                              annotators until labels.csv is filled)
  score   joins labels.csv with key.csv and prints agreement + Cohen's
          kappa for auto-vs-A1, auto-vs-A2, and A1-vs-A2.

Annotation question: "Did the agent's executed tool calls actually USE the
gold fact value(s) it needed from memory (verbatim or reformatted)?"

Golds are regenerated deterministically from the domain suites (verified:
every task_id in all nine cells is covered).

Usage:
  PYTHONPATH=. python scripts/mur_audit.py sample
  PYTHONPATH=. python scripts/mur_audit.py score runs/mur_audit/labels.csv
"""
from __future__ import annotations

import csv
import html
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from merit.domains import DOMAINS  # noqa: E402

OUT = Path("runs/mur_audit")
N_SAMPLE = 100
SEED = 0

# same run -> cell mapping as scripts/figures.py (verified vs Table 5.2)
CELLS = {
    ("d1", "easy"): "runs/pilot-clean",
    ("d2", "easy"): "runs/pilot-d2-v2",
    ("d3", "easy"): "runs/sweep/d3-easy",
    ("d1", "medium"): "runs/sweep/d1-medium",
    ("d2", "medium"): "runs/sweep/d2-medium",
    ("d3", "medium"): "runs/sweep/d3-medium",
    ("d1", "hard"): "runs/sweep/d1-hard",
    ("d2", "hard"): "runs/sweep/d2-hard",
    ("d3", "hard"): "runs/sweep/d3-hard",
}


def pool() -> list[dict]:
    """All auditable episodes: dependent, uncorrupted, memory held golds."""
    golds_by_cell = {
        (dom, tier): {ep.task.task_id: ep.task.golds()
                      for arc in DOMAINS[dom].generate_suite(difficulty=tier)
                      for ep in arc.episodes}
        for (dom, tier) in CELLS
    }
    out = []
    for (dom, tier), run in CELLS.items():
        traces = {}
        for line in (Path(run) / "traces" / "episodes.jsonl").open():
            t = json.loads(line)
            traces[t["episode_id"]] = t
        for line in (Path(run) / "results.jsonl").open():
            r = json.loads(line)
            if not (r["dependent"] and r["corrupt_mode"] == "none"
                    and r["memory_had_fact"]):
                continue
            t = traces[r["episode_id"]]
            out.append({
                "run": run, "domain": dom, "difficulty": tier,
                "condition": r["condition"], "task_id": r["task_id"],
                "episode_id": r["episode_id"],
                "auto": "yes" if r["memory_utilized"] else "no",
                "golds": golds_by_cell[dom, tier][r["task_id"]],
                "transcript": t["transcript"],
                "tool_calls": t["tool_calls"],
            })
    return out


def sample() -> None:
    eps = pool()
    strata = defaultdict(list)
    for e in eps:
        strata[e["condition"], e["auto"]].append(e)
    print(f"pool: {len(eps)} episodes")
    for k in sorted(strata):
        print(f"  {k[0]} auto={k[1]}: {len(strata[k])}")

    # proportional allocation with largest-remainder rounding to N_SAMPLE
    quotas = {k: len(v) * N_SAMPLE / len(eps) for k, v in strata.items()}
    alloc = {k: int(q) for k, q in quotas.items()}
    for k in sorted(quotas, key=lambda k: quotas[k] - alloc[k],
                    reverse=True)[:N_SAMPLE - sum(alloc.values())]:
        alloc[k] += 1

    rng = random.Random(SEED)
    picked = []
    for k, n in sorted(alloc.items()):
        # spread over (domain, tier) inside the stratum: shuffle, then take
        # round-robin across cells
        by_cell = defaultdict(list)
        for e in strata[k]:
            by_cell[e["domain"], e["difficulty"]].append(e)
        for cell in by_cell.values():
            rng.shuffle(cell)
        order = sorted(by_cell)
        i = 0
        while n > 0:
            cell = by_cell[order[i % len(order)]]
            if cell:
                picked.append(cell.pop())
                n -= 1
            i += 1
    rng.shuffle(picked)  # so sheet order carries no stratum signal

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "labels.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item", "annotator1", "annotator2"])
        for i, _ in enumerate(picked, 1):
            w.writerow([f"A{i:03d}", "", ""])
    with (OUT / "key.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item", "auto", "run", "episode_id", "task_id",
                    "condition", "domain", "difficulty"])
        for i, e in enumerate(picked, 1):
            w.writerow([f"A{i:03d}", e["auto"], e["run"], e["episode_id"],
                        e["task_id"], e["condition"], e["domain"],
                        e["difficulty"]])

    items = []
    for i, e in enumerate(picked, 1):
        golds = "".join(f"<li><code>{html.escape(g)}</code></li>"
                        for g in e["golds"])
        items.append(f"""
<section>
<h2>Item A{i:03d}</h2>
<p class="meta">{e['domain']} / {e['difficulty']} / {e['condition']}
&nbsp;·&nbsp; task <code>{html.escape(e['task_id'])}</code>
&nbsp;·&nbsp; episode <code>{e['episode_id']}</code></p>
<p><strong>Gold value(s) the task needed from memory:</strong></p>
<ul>{golds}</ul>
<details open><summary><strong>Episode transcript</strong> (memory block
shown to the agent, then the conversation and executed tool calls)</summary>
<pre>{html.escape(e['transcript'])}</pre></details>
<p class="q"><strong>Question:</strong> did the agent's executed tool calls
actually <em>use</em> the gold value(s) (verbatim or reformatted)?
Answer <code>yes</code> / <code>no</code> / <code>unclear</code> in
labels.csv.</p>
</section>""")
    (OUT / "audit_sheet.html").write_text(f"""<!doctype html>
<meta charset="utf-8"><title>MERIT MUR audit sheet ({len(picked)} items)</title>
<style>
body{{font:15px/1.5 system-ui;max-width:56rem;margin:2rem auto;padding:0 1rem}}
pre{{white-space:pre-wrap;background:#f6f6f2;padding:.8rem;border-radius:6px;
font-size:13px}}
section{{border-top:2px solid #ddd;margin-top:2rem;padding-top:1rem}}
.meta{{color:#666}} .q{{background:#fdf6e3;padding:.5rem .8rem;
border-radius:6px}}
</style>
<h1>MERIT MUR human audit — {len(picked)} episodes</h1>
<p>For each item: read the memory block and the transcript, then judge
whether the agent's <em>executed tool calls</em> made use of the gold
value(s) it needed from memory (verbatim or reformatted — e.g. units or
formatting changes still count as use; the value merely appearing in the
user's words or in memory does not). Record <code>yes</code>,
<code>no</code>, or <code>unclear</code> per item in
<code>labels.csv</code>. Do not consult <code>key.csv</code> until both
annotators are done.</p>
{''.join(items)}""")
    print(f"\nwrote {OUT}/audit_sheet.html, labels.csv, key.csv "
          f"({len(picked)} items)")


def kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    cats = set(a) | set(b)
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def score(labels_path: str) -> None:
    key = {r["item"]: r for r in csv.DictReader((OUT / "key.csv").open())}
    rows = list(csv.DictReader(open(labels_path)))
    raters = [("auto", lambda r: key[r["item"]]["auto"]),
              ("annotator1", lambda r: r["annotator1"].strip().lower()),
              ("annotator2", lambda r: r["annotator2"].strip().lower())]
    for i in range(len(raters)):
        for j in range(i + 1, len(raters)):
            (na, fa), (nb, fb) = raters[i], raters[j]
            pairs = [(fa(r), fb(r)) for r in rows
                     if fa(r) in ("yes", "no") and fb(r) in ("yes", "no")]
            if not pairs:
                print(f"{na} vs {nb}: no scorable pairs (labels empty?)")
                continue
            a, b = [p[0] for p in pairs], [p[1] for p in pairs]
            agree = sum(x == y for x, y in zip(a, b)) / len(pairs)
            print(f"{na} vs {nb}: n={len(pairs)} agreement={agree:.3f} "
                  f"kappa={kappa(a, b):.3f} "
                  f"(excluded {len(rows) - len(pairs)} unclear/blank)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "score":
        score(sys.argv[2] if len(sys.argv) > 2
              else str(OUT / "labels.csv"))
    else:
        sample()
