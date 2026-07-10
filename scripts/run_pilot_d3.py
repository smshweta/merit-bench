"""D3 (personal assistant) pilot runner — same as run_pilot.py but uses merit.d3 instead."""
import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from merit import metrics as M
from merit.d3 import (generate_suite, D3World, TOOL_FUNCS as D3_TOOLS,
                      TOOL_SCHEMAS as D3_SCHEMAS, SYSTEM_PROMPT as D3_PROMPT)
from merit.memory import CONDITIONS, corrupt_records
from merit.runner import run_episode
from merit.user_sim import SimulatedUser


def episode_cost_usd(model: str, prompt_tokens: int,
                     completion_tokens: int) -> float:
    if model == "mock":
        return 0.0
    try:
        import litellm
        in_cost, out_cost = litellm.cost_per_token(
            model=model, prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens)
        return in_cost + out_cost
    except Exception:
        return 0.0


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
                    continue
                for arc in arcs:
                    world = arc.make_world()
                    memory = cond_cls()
                    crng = random.Random(f"{arc.arc_id}|{corrupt_mode}")
                    for ep in arc.episodes:
                        task = ep.task
                        checker = getattr(M, task.checker)
                        pre_satisfied = checker(world.snapshot(),
                                                **task.checker_args)
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
                            log_dir=out_dir / "traces",
                            api_base=args.api_base,
                            tool_funcs=D3_TOOLS, tool_schemas=D3_SCHEMAS,
                            system_prompt=D3_PROMPT)
                        success = (not pre_satisfied and
                                   checker(world.snapshot(),
                                           **task.checker_args))
                        mem_had_fact = (task.dependent and task.gold_fact_value
                                        in result.memory_block)
                        row = {
                            "seed": seed, "condition": cond_key,
                            "corrupt_mode": corrupt_mode,
                            "corrupt_rate": corrupt_rate,
                            "arc_id": arc.arc_id, "episode_index": ep.index,
                            "task_id": task.task_id, "kind": task.kind,
                            "dependent": task.dependent, "success": success,
                            "pre_satisfied": pre_satisfied,
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
    p.add_argument("--model", default="mock")
    p.add_argument("--api-base", default=None)
    p.add_argument("--arcs", type=int, default=10)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--dep-ratio", type=float, default=0.5)
    p.add_argument("--conditions", default="C0,C1,C2,C3,C4,C5")
    p.add_argument("--corrupt", action="store_true")
    p.add_argument("--corrupt-modes", default="stale,contradiction,distractor")
    p.add_argument("--user-mode", default="scripted",
                   choices=["scripted", "llm"])
    p.add_argument("--out", default="runs/pilot-d3")
    run(p.parse_args())


if __name__ == "__main__":
    main()
