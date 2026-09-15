"""Figure: counterfactual replay of retrieval-memory probes
(scripts/run_position_intervention.py). Task success and stale-action rate
per presentation variant of the SAME retrieved chunks, per agent model,
with Wilson 95% intervals.

Usage: PYTHONPATH=. .venv/bin/python scripts/fig_intervention.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from analyze_failures import wilson  # noqa: E402
from figures import OUT, plt, save  # noqa: E402

VARIANTS = [("original", "similarity order\n(as in grid)", "#8a8a85", ""),
            ("antichrono", "newest first\n(stale last)", "#d64545", "//"),
            ("stamped", "similarity order\n+ session labels", "#eda100", ".."),
            ("chrono", "oldest first\n(latest last)", "#2a78d6", "xx")]
MODELS = [("mini3", "gpt-4.1-mini (180 probes)"), ("gpt41", "GPT-4.1 (60 probes)")]


def main() -> None:
    rows = [json.loads(l) for l in
            Path("runs/intervention/results.jsonl").open() if l.strip()]
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.3), sharey=True)
    metrics = [("success", "task success rate", lambda r: r["success"]),
               ("stale", "stale-value action rate",
                lambda r: r["action"] == "stale")]
    width = 0.2
    for ax, (_, ylabel, keyf) in zip(axes, metrics):
        for m, (slice_, mlabel) in enumerate(MODELS):
            for j, (v, _, color, hatch) in enumerate(VARIANTS):
                rs = [r for r in rows if r["slice"] == slice_
                      and r["variant"] == v]
                k = sum(bool(keyf(r)) for r in rs)
                p = k / len(rs)
                lo, hi = wilson(k, len(rs))
                x = m + (j - 1.5) * width
                ax.bar(x, p, width * 0.92, color=color, hatch=hatch,
                       edgecolor="white", linewidth=0.4)
                ax.errorbar(x, p, yerr=[[p - lo], [hi - p]], fmt="none",
                            ecolor="#444", capsize=1.5, linewidth=0.8)
        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels([l for _, l in MODELS])
        ax.set_ylim(0, 1)
        ax.set_title(ylabel)
        ax.grid(axis="x", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, hatch=h,
                             edgecolor="white") for _, _, c, h in VARIANTS]
    fig.legend(handles, [l.replace("\n", " ") for _, l, _, _ in VARIANTS],
               loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.1), fontsize=7)
    fig.tight_layout()
    save(fig, "fig8_intervention")


if __name__ == "__main__":
    main()
