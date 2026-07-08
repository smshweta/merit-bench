"""Run ONE live episode end-to-end to verify your API key + the loop.
Cost: well under $0.01 with a small model. Requires MERIT_MODEL + provider key.
"""
from merit.world import World
from merit.memory import NoMemory
from merit.runner import run_episode
from merit.metrics import check_refund_issued

world = World.create(seed=42)
oid, cid, amount = world.conn.execute(
    "SELECT id, customer_id, amount_cents FROM orders").fetchone()

result = run_episode(
    world=world,
    memory=NoMemory(),
    user_messages=[f"Hi, I'd like a full refund on my order {oid}."],
    task_id="smoke-refund-1",
)

ok = check_refund_issued(world.snapshot(), oid, amount)
print(f"success={ok}  tool_calls={len(result.tool_calls)}  "
      f"tokens={result.prompt_tokens}+{result.completion_tokens}  "
      f"wall={result.wall_seconds:.1f}s")
print("Trace logged to runs/episodes.jsonl")
