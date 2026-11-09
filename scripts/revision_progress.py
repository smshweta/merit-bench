"""Progress and spend of the revision-stage runs (runs/revision)."""
import json
from pathlib import Path

# expected rows per cell directory prefix
EXPECT = {"corrupt-d2": 2100, "corrupt-d3": 2100, "corrupt-d1-temporal": 700,
          "mini3-temporal": 300, "gpt41-temporal": 100, "gpt41-s12": 800,
          "long-mini": 1600, "haiku45-temporal": 100, "haiku45-s12": 800}
tot_rows = tot_cost = 0.0
for d in sorted(Path("runs/revision").glob("*/")):
    f = d / "results.jsonl"
    rows = f.read_text().splitlines() if f.exists() else []
    cost = sum(json.loads(r)["cost_usd"] for r in rows)
    exp = next((v for k, v in EXPECT.items() if d.name.startswith(k)), None)
    done = "done" if (d / ".done").exists() else ""
    print(f"{d.name:28s} {len(rows):5d}/{exp}  ${cost:6.2f} {done}")
    tot_rows += len(rows); tot_cost += cost
print(f"{'TOTAL':28s} {int(tot_rows):5d}        ${tot_cost:6.2f}")
