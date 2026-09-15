"""Counterfactual replay: does the ORDER or PROVENANCE of retrieved memory
cause stale-value actions? (follow-up to scripts/analyze_failures.py)

For every hard-tier C2 (embedding retrieval) probe in the full grid, this
script rebuilds, from the released traces, the exact state the agent saw:

  * memory: the EmbeddingRAG store is refilled with the original episode
    transcripts in order (embeddings come from the on-disk cache), and the
    probe's read() is reproduced; the reconstructed block must equal the
    block in the trace byte for byte, otherwise the probe is skipped;
  * world: a fresh world is built from the arc seed and every logged tool
    call of the preceding episodes is re-applied.

Only the probe episode is then re-run, with the SAME retrieved chunks
presented four ways:

  original    similarity order, as in the grid (replay-fidelity control)
  chrono      same chunks sorted oldest -> newest (latest value last)
  antichrono  same chunks sorted newest -> oldest (stale value last)
  stamped     similarity order, each chunk prefixed with the session it
              was saved after (provenance, no reordering)

Content, world, prompt, model, and decoding are held fixed; only
presentation changes. Rows append to runs/intervention/results.jsonl and
completed (cell, seed, task, variant) keys are skipped, so the run resumes.

Usage:
  PYTHONPATH=. python scripts/run_position_intervention.py [--limit N]
      [--workers 8] [--slices mini3,gpt41,haiku45]
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import analyze_failures as A  # noqa: E402
from merit import metrics as M  # noqa: E402
from merit.domains import DOMAINS  # noqa: E402
from merit.memory_llm import EmbeddingRAG, default_embed_fn  # noqa: E402
from merit.runner import run_episode  # noqa: E402

OUT = Path("runs/intervention")
VARIANTS = ["original", "chrono", "antichrono", "stamped"]
_lock = threading.Lock()


class FixedBlock:
    """Memory stub that shows one precomputed block and stores nothing."""
    name = "C2_rag_intervention"

    def __init__(self, block: str) -> None:
        self.block = block

    def read(self, current_context: str, budget_chars: int = 4000) -> str:
        return self.block

    def write(self, episode_id: str, transcript: str) -> None:
        pass


def selected_indices(mem: EmbeddingRAG, context: str,
                     budget_chars: int = 4000) -> list[int]:
    """Replicates EmbeddingRAG.read, returning record indices instead of
    text (similarity order)."""
    from merit.memory_llm import _cosine
    if len(mem._vectors) < len(mem.records):
        vecs, _ = mem.embed_fn([r.text for r in
                                mem.records[len(mem._vectors):]])
        mem._vectors.extend(vecs)
    (q,), _ = mem.embed_fn([context])
    ranked = sorted(range(len(mem.records)),
                    key=lambda i: -_cosine(q, mem._vectors[i]))
    out, used = [], 0
    for i in ranked[: mem.top_k]:
        n = len(mem.records[i].text)
        if used + n > budget_chars:
            break
        out.append(i)
        used += n
    return out


def session_of(episode_id: str) -> int:
    return int(episode_id.rsplit("-e", 1)[1]) + 1


def build_blocks(mem: EmbeddingRAG, idx: list[int]) -> dict[str, str]:
    recs = mem.records
    join = "\n---\n".join
    return {
        "original": join(recs[i].text for i in idx),
        "chrono": join(recs[i].text for i in sorted(idx)),
        "antichrono": join(recs[i].text for i in sorted(idx, reverse=True)),
        "stamped": join(f"[Note saved after session "
                        f"{session_of(recs[i].episode_id)}]\n{recs[i].text}"
                        for i in idx),
    }


def replay_world(arc, prior_calls: list[list[dict]], tool_funcs: dict):
    world = arc.make_world()
    for calls in prior_calls:
        for c in calls:
            if c["name"] in tool_funcs:
                try:
                    tool_funcs[c["name"]](world, **c["args"])
                except TypeError:
                    pass  # the original call errored the same way
    return world


def collect_jobs(slices: list[str]) -> tuple[list[dict], dict]:
    emb = default_embed_fn("text-embedding-3-small")
    jobs, stats = [], {"probes": 0, "block_mismatch": 0}
    for slice_ in slices:
        for dom in ("d1", "d2", "d3"):
            cell = Path(f"runs/phasec/{slice_}-{dom}-hard")
            rows = [json.loads(l) for l in (cell / "results.jsonl").open()]
            traces = {}
            for line in (cell / "traces" / "episodes.jsonl").open():
                t = json.loads(line)
                traces[t["episode_id"]] = t
            domain = DOMAINS[dom]
            for seed in sorted({r["seed"] for r in rows}):
                c2 = [r for r in rows if r["seed"] == seed
                      and r["condition"] == "C2"]
                n_arcs = len({r["arc_id"] for r in c2})
                arcs = {a.arc_id: a for a in domain.generate_suite(
                    n_arcs=n_arcs, episodes_per_arc=5, dep_ratio=0.5,
                    base_seed=seed, difficulty="hard")}
                pairs = A.fact_pairs(dom, seed, n_arcs)
                for arc_id, arc in arcs.items():
                    arc_rows = sorted((r for r in c2 if r["arc_id"] == arc_id),
                                      key=lambda r: r["episode_index"])
                    mem = EmbeddingRAG(emb)
                    prior_calls: list[list[dict]] = []
                    for r, ep in zip(arc_rows, arc.episodes):
                        assert r["task_id"] == ep.task.task_id
                        tr = traces[r["episode_id"]]
                        task = ep.task
                        if task.dependent and task.kind.endswith("_upd"):
                            stats["probes"] += 1
                            context = " ".join(task.user_messages)
                            idx = selected_indices(mem, context)
                            blocks = build_blocks(mem, idx)
                            stale, latest, first = pairs[(arc_id,
                                                          task.task_id)]
                            if blocks["original"] != A.memory_block(
                                    tr["transcript"], first):
                                stats["block_mismatch"] += 1
                            else:
                                jobs.append({
                                    "slice": slice_, "domain": dom,
                                    "seed": seed, "arc_id": arc_id,
                                    "task_id": task.task_id,
                                    "model": r["model"],
                                    "grid_success": r["success"],
                                    "stale": stale, "latest": latest,
                                    "first_msg": first,
                                    "blocks": blocks,
                                    "prior_calls": list(prior_calls),
                                    "arc": arc, "task": task,
                                })
                        mem.write(f"{arc_id}-e{ep.index}", tr["transcript"])
                        prior_calls.append(tr["tool_calls"])
    return jobs, stats


def run_job(job: dict, variant: str) -> dict:
    domain = DOMAINS[job["domain"]]
    task = job["task"]
    world = replay_world(job["arc"], job["prior_calls"], domain.tool_funcs)
    checker = getattr(M, task.checker)
    pre = checker(world.snapshot(), **task.checker_args)
    block = job["blocks"][variant]
    res = run_episode(world=world, memory=FixedBlock(block),
                      user_messages=list(task.user_messages),
                      task_id=task.task_id, model=job["model"],
                      log_dir=OUT / "traces", tool_funcs=domain.tool_funcs,
                      tool_schemas=domain.tool_schemas,
                      system_prompt=domain.system_prompt)
    success = (not pre) and checker(world.snapshot(), **task.checker_args)
    rec = {"transcript": f"[memory shown]\n{block}\n[user] "
                         + "\n[user] ".join(task.user_messages),
           "tool_calls": res.tool_calls}
    cls = A.classify(job["domain"], job["stale"], job["latest"],
                     job["first_msg"], rec)
    from run_pilot import episode_cost_usd
    return {
        "slice": job["slice"], "domain": job["domain"], "seed": job["seed"],
        "arc_id": job["arc_id"], "task_id": job["task_id"],
        "model": job["model"], "variant": variant, "success": success,
        "pre_satisfied": pre, "grid_success": job["grid_success"],
        "state": cls["state"], "action": cls["action"], "order": cls["order"],
        "prompt_tokens": res.prompt_tokens,
        "completion_tokens": res.completion_tokens,
        "cost_usd": episode_cost_usd(job["model"], res.prompt_tokens,
                                     res.completion_tokens),
        "episode_id": res.episode_id,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices", default="mini3,gpt41,haiku45")
    ap.add_argument("--limit", type=int, default=0,
                    help="run only the first N probes (smoke test)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    results = OUT / "results.jsonl"
    done = set()
    if results.exists():
        for line in results.open():
            d = json.loads(line)
            done.add((d["slice"], d["domain"], d["seed"], d["task_id"],
                      d["variant"]))
    jobs, stats = collect_jobs(args.slices.split(","))
    print(f"probes {stats['probes']}, reconstructed exactly "
          f"{len(jobs)}, block mismatches {stats['block_mismatch']}",
          flush=True)
    if args.limit:
        jobs = jobs[: args.limit]
    todo = [(j, v) for j in jobs for v in VARIANTS
            if (j["slice"], j["domain"], j["seed"], j["task_id"], v)
            not in done]
    print(f"episodes to run: {len(todo)}", flush=True)
    spent, n = 0.0, 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_job, j, v): (j, v) for j, v in todo}
        for f in as_completed(futs):
            try:
                row = f.result()
            except Exception as e:  # keep going; the key stays undone
                j, v = futs[f]
                print(f"FAILED {j['slice']} {j['task_id']} {v}: {e!r}",
                      flush=True)
                continue
            with _lock, results.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            spent += row["cost_usd"]
            n += 1
            if n % 50 == 0:
                print(f"{n}/{len(todo)} done, ${spent:.2f}", flush=True)
    print(f"finished {n} episodes, ${spent:.2f}", flush=True)


if __name__ == "__main__":
    main()
