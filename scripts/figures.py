"""Paper figures from pilot/sweep results.jsonl.

Produces docs/figures/fig{1..4}.{pdf,png}:
  fig1  difficulty ladder: dependent-task TSR by tier, per condition, per domain
  fig2  Ignore Rate on the hard tier, per condition x domain
  fig3  stale-memory harm on D1 with paired-bootstrap 95% CIs (clustered by arc)
  fig4  cost vs dependent-task TSR at the easy tier (the CAMU picture)

Pre-Phase-A runs lack domain/difficulty fields, so RUNS maps each file to its
(domain, difficulty) cell explicitly; sweep rows carry the fields and are
trusted as-is. The clean/corrupt contrast for fig3 stays within pilot-full so
the pairing-by-arc assumption of the bootstrap holds.

Usage: PYTHONPATH=. .venv/bin/python scripts/figures.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from analyze import boot_ci_delta, mean  # noqa: E402

OUT = Path("docs/figures")

# (path, domain, difficulty) — domain/difficulty None means "trust row fields"
RUNS = [
    ("runs/pilot-clean/results.jsonl", "d1", "easy"),
    ("runs/pilot-d2-v2/results.jsonl", "d2", "easy"),
    ("runs/sweep/d3-easy/results.jsonl", None, None),
    ("runs/sweep/d1-medium/results.jsonl", None, None),
    ("runs/sweep/d2-medium/results.jsonl", None, None),
    ("runs/sweep/d3-medium/results.jsonl", None, None),
    ("runs/sweep/d1-hard/results.jsonl", None, None),
    ("runs/sweep/d2-hard/results.jsonl", None, None),
    ("runs/sweep/d3-hard/results.jsonl", None, None),
]
CORRUPT_RUN = ("runs/pilot-full/results.jsonl", "d1", "easy")

CONDS = ["C0", "C1", "C2", "C3", "C4", "C5"]
COND_LABEL = {
    "C0": "C0 none", "C1": "C1 replay", "C2": "C2 retrieval",
    "C3": "C3 summary", "C4": "C4 facts", "C5": "C5 hybrid",
}
# fixed categorical assignment (validated CVD-safe palette); C0 is the
# baseline, not a series identity, so it wears neutral gray
COLOR = {
    "C0": "#8a8a85", "C1": "#2a78d6", "C2": "#1baf7a",
    "C3": "#eda100", "C4": "#008300", "C5": "#4a3aa7",
}
TIERS = ["easy", "medium", "hard"]
DOMAINS = ["d1", "d2", "d3"]
DOMAIN_LABEL = {"d1": "D1 commerce", "d2": "D2 IT ops", "d3": "D3 assistant"}

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e6e2", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "lines.linewidth": 1.8,
    "figure.dpi": 100, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def load(path: str, domain: str | None, difficulty: str | None) -> list[dict]:
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l]
    for r in rows:
        r.setdefault("domain", domain)
        r.setdefault("difficulty", difficulty)
        assert r["domain"] and r["difficulty"], f"unmapped row in {path}"
    return rows


def dep_clean(rows):
    return [r for r in rows if r["dependent"] and r["corrupt_mode"] == "none"]


def save(fig, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"wrote {OUT}/{name}.pdf .png")


def fig1_ladder(cells: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.1), sharey=True)
    for ax, dom in zip(axes, DOMAINS):
        for k, c in enumerate(CONDS):
            ys = [mean(r["success"] for r in dep_clean(cells[dom, t])
                       if r["condition"] == c) for t in TIERS]
            # small x-dodge so conditions tied at the same TSR stay visible
            xs = [i + (k - 2.5) * 0.014 for i in range(len(TIERS))]
            ax.plot(xs, ys, marker="o", markersize=4, color=COLOR[c],
                    label=COND_LABEL[c], zorder=3 if c in ("C2", "C4") else 2,
                    clip_on=False)
        ax.set_xticks(range(len(TIERS)))
        ax.set_xticklabels(TIERS)
        ax.set_title(DOMAIN_LABEL[dom])
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlim(-0.15, 2.15)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("dependent-task TSR")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center",
               ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.14),
               columnspacing=1.2, handlelength=1.4)
    save(fig, "fig1_difficulty_ladder")


def fig2_ignore(cells: dict) -> None:
    conds = [c for c in CONDS if c != "C0"]
    tiers = ["medium", "hard"]
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.0), sharey=True)
    width, n_dom = 0.26, len(DOMAINS)
    for ax, tier in zip(axes, tiers):
        for j, dom in enumerate(DOMAINS):
            rows = [r for r in dep_clean(cells[dom, tier])
                    if r["memory_had_fact"]]
            for i, c in enumerate(conds):
                had = [r for r in rows if r["condition"] == c]
                if not had:
                    continue
                ign = 1 - mean(r["memory_utilized"] for r in had)
                x = i + (j - (n_dom - 1) / 2) * width
                # per-domain shading = ordinal steps of the condition's hue
                ax.bar(x, ign, width=width * 0.92, color=COLOR[c],
                       alpha=(0.45, 0.7, 1.0)[j], edgecolor="none")
                ax.annotate(f"{len(had)}", (x, ign), ha="center",
                            va="bottom", fontsize=6, color="#6b6b66",
                            xytext=(0, 1), textcoords="offset points")
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels(conds)
        ax.set_title(tier)
        ax.set_ylim(0, 1.0)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Ignore Rate")
    handles = [plt.Rectangle((0, 0), 1, 1, color="#555550", alpha=a)
               for a in (0.45, 0.7, 1.0)]
    fig.legend(handles, [DOMAIN_LABEL[d] for d in DOMAINS], frameon=False,
               loc="upper center", ncol=3, handlelength=1.0,
               bbox_to_anchor=(0.5, 1.12))
    save(fig, "fig2_ignore_rate")


def fig3_smh() -> None:
    rows = load(*CORRUPT_RUN)
    clean = [r for r in rows if r["corrupt_mode"] == "none" and r["dependent"]]
    modes = ["stale", "contradiction", "distractor"]
    conds = [c for c in CONDS if c != "C0"]
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.0), sharey=True)
    for ax, mode in zip(axes, modes):
        for i, c in enumerate(conds):
            cc = [r for r in clean if r["condition"] == c]
            for rate, filled, dx in ((0.1, False, -0.13), (0.3, True, 0.13)):
                cor = [r for r in rows
                       if r["condition"] == c and r["dependent"]
                       and r["corrupt_mode"] == mode
                       and r["corrupt_rate"] == rate]
                point, ci, _ = boot_ci_delta(cc, cor,
                                             key=lambda r: r["success"])
                ax.errorbar(i + dx, point,
                            yerr=[[point - ci[0]], [ci[1] - point]],
                            fmt="o", markersize=4, color=COLOR[c],
                            markerfacecolor=COLOR[c] if filled else "white",
                            capsize=2, linewidth=1.2, zorder=3)
        ax.axhline(0, color="#8a8a85", linewidth=0.8, zorder=1)
        ax.set_title(mode)
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels(conds)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("stale-memory harm\n(TSR clean − corrupted)")
    hollow = plt.Line2D([], [], marker="o", markerfacecolor="white",
                        color="#555550", linestyle="", markersize=4)
    solid = plt.Line2D([], [], marker="o", color="#555550", linestyle="",
                       markersize=4)
    axes[-1].legend([hollow, solid], ["ρ = 0.1", "ρ = 0.3"], frameon=False,
                    loc="upper left", handlelength=1.0)
    fig.suptitle("D1, paired bootstrap 95% CI clustered by arc", fontsize=7.5,
                 color="#6b6b66", y=1.06)
    save(fig, "fig3_stale_memory_harm")


def fig4_cost(cells: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.1), sharey=True)
    for ax, dom in zip(axes, DOMAINS):
        rows = [r for r in cells[dom, "easy"] if r["corrupt_mode"] == "none"]
        pts = {}
        for c in CONDS:
            cost = mean(r["cost_usd"] for r in rows if r["condition"] == c)
            tsr = mean(r["success"] for r in rows
                       if r["condition"] == c and r["dependent"])
            pts[c] = (cost * 1000, tsr)
            ax.scatter(cost * 1000, tsr, s=26, color=COLOR[c], zorder=3,
                       clip_on=False)
        span = max(x for x, _ in pts.values()) - min(x for x, _ in pts.values())
        placed = []
        for c, (x, y) in sorted(pts.items(), key=lambda kv: kv[1][0]):
            # flip the label to the left when it would collide with an
            # already-placed right-side label of a nearby point
            left = any(abs(x - px) < 0.18 * span and abs(y - py) < 0.06
                       for px, py in placed)
            ax.annotate(c, (x, y), fontsize=7, color="#3d3d39",
                        xytext=(-4 if left else 4, -2),
                        textcoords="offset points",
                        ha="right" if left else "left")
            placed.append((x, y))
        ax.set_title(DOMAIN_LABEL[dom])
        ax.set_xlabel("cost ($ per 1000 episodes)")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("dependent-task TSR (easy)")
    save(fig, "fig4_cost_frontier")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for path, dom, tier in RUNS:
        for r in load(path, dom, tier):
            cells[r["domain"], r["difficulty"]].append(r)
    for dom in DOMAINS:
        for t in TIERS:
            assert cells[dom, t], f"missing cell {dom}/{t}"
    fig1_ladder(cells)
    fig2_ignore(cells)
    fig3_smh()
    fig4_cost(cells)


if __name__ == "__main__":
    main()
