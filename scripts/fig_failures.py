"""Figure 7: anatomy of hard-tier failures (from scripts/analyze_failures.py).

  (a) outcome of every hard-tier probe under C2 (retrieval) and C5 (hybrid),
      per agent model in the full grid: acted on the latest value; acted on
      the stale value with both values in memory; acted on the stale value
      with the latest value missing; other value or no action.
  (b) exploratory position effect, pooled over the grid: stale-action rate
      among probes whose block held both values, split by which value is
      positioned last in the block (Wilson 95% intervals).

Usage: PYTHONPATH=. .venv/bin/python scripts/fig_failures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import analyze_failures as A  # noqa: E402
from figures import COLOR, OUT, plt, save  # noqa: E402

MODELS = [("grid/gpt-4.1-mini x3 seeds", "gpt-4.1-mini"),
          ("grid/gpt-4.1", "GPT-4.1"),
          ("grid/claude-haiku-4.5", "Haiku 4.5")]
CATS = [("latest", "acted on latest value", "#2a78d6", ""),
        ("stale_both", "acted on stale value (both in memory)", "#d64545",
         "//"),
        ("stale_missing", "acted on stale value (latest missing)", "#f0a3a3",
         ".."),
        ("other", "other value or no action", "#b9b9b4", "")]


def category(r: dict) -> str:
    if r["action"] == "latest":
        return "latest"
    if r["action"] == "stale":
        return "stale_both" if r["state"] == "both" else "stale_missing"
    return "other"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = {name: A.load_group(A.GROUPS[name]) for name, _ in MODELS}
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.5, 2.3),
                                 gridspec_kw={"width_ratios": [1.6, 1]})

    labels, y = [], 0
    for cond in ("C2", "C5"):
        for name, mlabel in MODELS:
            rs = [r for r in rows[name] if r["condition"] == cond]
            left = 0.0
            for key, _, color, hatch in CATS:
                share = sum(category(r) == key for r in rs) / len(rs)
                ax.barh(y, share, left=left, color=color, hatch=hatch,
                        edgecolor="white", linewidth=0.4, height=0.75)
                left += share
            labels.append(f"{cond}  {mlabel}")
            y += 1
        y += 0.5
    ypos = [0, 1, 2, 3.5, 4.5, 5.5]
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of hard-tier probes")
    ax.set_title("(a) outcome of updated-fact probes")
    ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, hatch=h,
                             edgecolor="white") for _, _, c, h in CATS]
    fig.legend(handles, [l for _, l, _, _ in CATS], loc="upper center",
               ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.16))

    grid = [r for name, _ in MODELS for r in rows[name]]
    width = 0.36
    for k, cond in enumerate(("C2", "C5")):
        both = [r for r in grid if r["condition"] == cond
                and r["state"] == "both"]
        for j, (order, hatch) in enumerate((("latest_last", ""),
                                            ("stale_last", "//"))):
            rs = [r for r in both if r["order"] == order]
            kk = sum(r["action"] == "stale" for r in rs)
            p = kk / len(rs)
            lo, hi = A.wilson(kk, len(rs))
            x = k + (j - 0.5) * width
            bx.bar(x, p, width * 0.92, color=COLOR[cond], hatch=hatch,
                   edgecolor="white", linewidth=0.4)
            bx.errorbar(x, p, yerr=[[p - lo], [hi - p]], fmt="none",
                        ecolor="#444", capsize=2, linewidth=0.9)
            bx.text(x, hi + 0.02, f"{kk}/{len(rs)}", ha="center",
                    va="bottom", fontsize=6.5)
    bx.set_xticks([0, 1])
    bx.set_xticklabels(["C2 retrieval", "C5 hybrid"])
    bx.set_ylim(0, 1)
    bx.set_ylabel("P(stale action | both in memory)")
    bx.set_title("(b) which value is positioned last")
    bx.grid(axis="x", visible=False)
    bx.legend([plt.Rectangle((0, 0), 1, 1, facecolor="#b9b9b4", hatch=h,
                             edgecolor="white") for h in ("", "//")],
              ["latest value last", "stale value last"], frameon=False,
              loc="upper left")
    fig.tight_layout()
    save(fig, "fig7_failure_anatomy")


if __name__ == "__main__":
    main()
