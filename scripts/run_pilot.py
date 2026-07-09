"""Phase 2 pilot runner.

Runs arcs × conditions × seeds × corruption settings, scores every episode
programmatically, and appends one JSON row per scored episode to the results
file. Deterministic under model="mock" (default, $0); pass --model to run
the real pilot (e.g. --model gpt-4.1-mini, key via env).

Usage:
  python scripts/run_pilot.py                          # offline mock pilot
  python scripts/run_pilot.py --model gpt-4.1-mini \
      --arcs 10 --episodes 5 --seeds 1 --dep-ratio 0.5 # real pilot (~$5-20)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from merit import metrics as M
from merit.arcs import generate_suite
from merit.memory import CONDITIONS, corrupt_records
from merit.runner import run_episode
from merit.user_sim import SimulatedUser


def episode_cost_usd(model: str, prompt_tokens: int,
                     completion_tokens: int) -> float:
    if model == "mock":
        return 0.0
    try:
        import litellm
        return litellm.model_cost and litellm.completion_cost(
            model=model,
            prompt="x" * 0, completion="x" * 0,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens) or 0.0
    except Exception:
        return 0.0  # tokens are always logged; price can be applied later


def run(args: argparse.Namespace) -> Path:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    t_start = time.time()
    n_rows = 0

    corruption_settings = [("none", 0.0)]
    if args.corrupt:
        corruption_settings += [(m, r) for m in args.corrupt_modes.split(",")
                                for r in (0.1, 0.3)]

    for seed in range(args.seeds):
        arcs = generate_suite(n_arcs=args.arcs, episodes_per_arc=args.episodes,
                              dep_ratio=args.dep_ratio, base_seed=seed)
        for cond_key, cond_cls in CONDITIONS.items():
            if cond_key not in args.conditions.split(","):
                continue
            for corrupt_mode, corrupt_rate in corruption_settings:
                if cond_key == "C0" and corrupt_mode != "none":
                    continue  # nothing to corrupt
                for arc in arcs:
                    world = arc.make_world()
                    memory = cond_cls()
                    crng = random.Random(f"{arc.arc_id}|{corrupt_mode}")
                    for ep in arc.episodes:
                        task = ep.task
                        if corrupt_rate > 0:
                            corrupt_records(memory, corrupt_rate, crng,
                                            mode=corrupt_mode)
                        user = SimulatedUser(
                            script=task.user_messages,
                            persona_idx=ep.index,
                            mode=args.user_mode,
                            forbidden=task.gold_fact_value)
                        result = run_episode(
                            world=world, memory=memory,
                            user_messages=user.turns(),
                            task_id=task.task_id, model=args.model,
                            log_dir=out_dir / "traces")
                        checker = getattr(M, task.checker)
                        success = checker(world.snapshot(), **task.checker_args)
                        mem_had_fact = (task.dependent and task.gold_fact_value
                                        in result.memory_block)
                        row = {
                            "seed": seed, "condition": cond_key,
                            "corrupt_mode": corrupt_mode,
                            "corrupt_rate": corrupt_rate,
                            "arc_id": arc.arc_id, "episode_index": ep.index,
                            "task_id": task.task_id, "kind": task.kind,
                            "dependent": task.dependent, "success": success,
                            "memory_had_fact": mem_had_fact,
                            "memory_utilized": (mem_had_fact and
                                M.memory_utilized(result.tool_calls,
                                                  task.gold_fact_value)),
                            "prompt_tokens": result.prompt_tokens,
                            "completion_tokens": result.completion_tokens,
                            "cost_usd": episode_cost_usd(
                                args.model, result.prompt_tokens,
                                result.completion_tokens),
                            "wall_seconds": result.wall_seconds,
                            "model": args.model,
                            "episode_id": result.episode_id,
                        }
                        with open(results_path, "a") as f:
                            f.write(json.dumps(row) + "\n")
                        n_rows += 1
                        memory.write(f"{arc.arc_id}-e{ep.index}",
                                     result.transcript)

    print(f"wrote {n_rows} scored episodes to {results_path} "
          f"in {time.time() - t_start:.1f}s")
    return results_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="mock",
                   help="'mock' ($0 pipeline validation) or any LiteLLM model")
    p.add_argument("--arcs", type=int, default=10)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--dep-ratio", type=float, default=0.5)
    p.add_argument("--conditions", default="C0,C1,C2,C3,C4,C5")
    p.add_argument("--corrupt", action="store_true",
                   help="also run corrupted-memory sweeps (rates 0.1, 0.3)")
    p.add_argument("--corrupt-modes", default="stale,contradiction,distractor")
    p.add_argument("--user-mode", default="scripted",
                   choices=["scripted", "llm"])
    p.add_argument("--out", default="runs/pilot")
    run(p.parse_args())


if __name__ == "__main__":
    main()
