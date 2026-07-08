"""ReAct-style tool loop + episode runner with full JSONL trace logging.

Uses LiteLLM so any provider works (set MERIT_MODEL, e.g. 'gpt-4.1-mini' or
'claude-haiku-...'; keys via env). Every LLM call's tokens are metered.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from .tools import TOOL_FUNCS, TOOL_SCHEMAS
from .world import World
from .memory import MemoryBase

SYSTEM_PROMPT = """You are a customer-support agent. Resolve the user's request
by calling tools. Verify facts with tools when unsure. When the task is done,
reply with a short confirmation and no further tool calls.

{memory_block}"""

MAX_TURNS = 12


@dataclass
class EpisodeResult:
    episode_id: str
    task_id: str
    success: bool | None  # filled by checker
    tool_calls: list = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    wall_seconds: float = 0.0
    transcript: str = ""


def run_episode(world: World, memory: MemoryBase, user_messages: list[str],
                task_id: str, model: str | None = None,
                log_dir: str | Path = "runs") -> EpisodeResult:
    import litellm  # imported here so tests that don't call LLMs need no key

    model = model or os.environ.get("MERIT_MODEL", "gpt-4.1-mini")
    episode_id = uuid.uuid4().hex[:8]
    t0 = time.time()

    memory_block = memory.read(current_context=" ".join(user_messages))
    mem_section = (f"Relevant notes from previous sessions:\n{memory_block}"
                   if memory_block else "You have no notes from previous sessions.")
    messages = [{"role": "system",
                 "content": SYSTEM_PROMPT.format(memory_block=mem_section)}]

    result = EpisodeResult(episode_id=episode_id, task_id=task_id, success=None)
    transcript_parts: list[str] = [f"[memory shown]\n{memory_block}"]

    for user_msg in user_messages:
        messages.append({"role": "user", "content": user_msg})
        transcript_parts.append(f"[user] {user_msg}")

        for _ in range(MAX_TURNS):
            resp = litellm.completion(model=model, messages=messages,
                                      tools=TOOL_SCHEMAS, temperature=0)
            usage = resp.usage
            result.prompt_tokens += usage.prompt_tokens
            result.completion_tokens += usage.completion_tokens
            msg = resp.choices[0].message
            messages.append(msg.model_dump(exclude_none=True))

            if not msg.tool_calls:
                transcript_parts.append(f"[assistant] {msg.content}")
                break

            for tc in msg.tool_calls:
                fn = tc.function.name
                args = json.loads(tc.function.arguments or "{}")
                out = (TOOL_FUNCS[fn](world, **args)
                       if fn in TOOL_FUNCS
                       else json.dumps({"error": f"unknown tool {fn}"}))
                result.tool_calls.append({"name": fn, "args": args, "out": out})
                transcript_parts.append(f"[tool {fn}] args={args} -> {out}")
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": out})

    result.wall_seconds = time.time() - t0
    result.transcript = "\n".join(transcript_parts)

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "episodes.jsonl", "a") as f:
        f.write(json.dumps({
            "episode_id": episode_id, "task_id": task_id, "model": model,
            "memory": memory.name, "tool_calls": result.tool_calls,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "wall_seconds": result.wall_seconds,
            "transcript": result.transcript,
        }) + "\n")
    return result
