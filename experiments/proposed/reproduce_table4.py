"""Reproduce the paper's Table 4: job distribution over the layers for different QoS preferences.

The paper varies the user's optimisation parameter (time / energy / cost) and prints, for 12 runs,
the share of jobs on cloud / fog / edge together with makespan, energy, cost and offloading time.
It does not say which preference degrees were used, so we sweep a line from the balanced
preference (1/3, 1/3, 1/3) to the pure preference e_k:

    pref(s) = (1 - s) * (1/3, 1/3, 1/3) + s * e_k,      s = 0, 0.1, 0.2, 0.3, 0.4, 0.6, 1,   k in {time, energy, cost}

Default setting of the paper: MONTAGE, 100 jobs, MIN-MIN.  10 seeds (100-109), mean values.

Outputs
  results/tables/table4_reproduction.csv
  results/figures/table4_reproduction.png   layer shares and metrics (relative to s = 0), published mapping
  results/figures/table4_physical_mapping.png same with the physically consistent mapping (sensitivity)

Run:  python -m experiments.proposed.reproduce_table4
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.env import default_environment
from src.fuzzy.qos_selector import ALGORITHM_LAYER, PHYSICAL_LAYER
from src.policies import BALANCED, FuzzyOffloadingPolicy
from src.scheduler import simulate
from src.workflow import montage

ROOT = Path(__file__).resolve().parents[2]
FIG, TAB, DATA = ROOT / "results" / "figures", ROOT / "results" / "tables", ROOT / "data"
JOBS, SEEDS = 100, range(100, 110)
S_VALUES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 1.0)
AXES = {"time": 0, "energy": 1, "cost": 2}  # position in the (time, energy, cost) preference
MAPPINGS = {"paper": ALGORITHM_LAYER, "physical": PHYSICAL_LAYER}
COLORS = {"Cloud": "#d62728", "Fog": "#2ca02c", "Edge": "#1f77b4"}


def preference(kind: str, s: float) -> tuple[float, float, float]:
    unit = [0.0, 0.0, 0.0]
    unit[AXES[kind]] = 1.0
    return tuple((1 - s) * b + s * u for b, u in zip(BALANCED, unit))


def run() -> pd.DataFrame:
    env, rows = default_environment(), []
    grid = [(m, lm, k, s) for m, lm in MAPPINGS.items() for k in AXES for s in S_VALUES]
    for mapping, layer_map, kind, s in grid:
        pref = preference(kind, s)
        for seed in SEEDS:
            pol = FuzzyOffloadingPolicy(preference=pref, layer_map=layer_map, record=False)
            res = simulate(montage(JOBS, seed=seed), env, pol)
            rows.append({"mapping": mapping, "optimization": kind, "s": s, "algorithm": pol.algorithm, "seed": seed,
                         "cloud_pct": 100 * res.layer_share["Cloud"], "fog_pct": 100 * res.layer_share["Fog"],
                         "edge_pct": 100 * res.layer_share["Edge"], "makespan_s": res.makespan,
                         "energy_J": res.energy_j, "cost_usd": res.cost, "offload_s": res.mean_offload_s})
    return pd.DataFrame(rows)


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    cols = ["cloud_pct", "fog_pct", "edge_pct", "makespan_s", "energy_J", "cost_usd", "offload_s"]
    out = runs.groupby(["mapping", "optimization", "s"], sort=False).agg(
        algorithm=("algorithm", lambda a: a.iloc[0]), **{c: (c, "mean") for c in cols}).reset_index()
    return out.round(3)


def plot(table: pd.DataFrame, mapping: str, fname: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    fig.suptitle(f"Table 4 reproduction - A2/A3/A4 layer mapping: {mapping}"
                 + (" (cost->Cloud, energy->Fog, time->Edge)" if mapping == "paper"
                    else " (cost->Edge, energy->Fog, time->Cloud)"))
    for j, kind in enumerate(("cost", "time", "energy")):
        g = table[(table.optimization == kind) & (table.mapping == mapping)]
        x = np.arange(len(g))
        bottom = np.zeros(len(g))
        ax = axes[0, j]
        for layer in ("Edge", "Fog", "Cloud"):
            v = g[f"{layer.lower()}_pct"].to_numpy()
            ax.bar(x, v, bottom=bottom, color=COLORS[layer], label=layer)
            bottom += v
        ax.set(xticks=x, xticklabels=[f"{s:g}\n{a}" for s, a in zip(g.s, g.algorithm)],
               ylabel="Jobs per layer (%)", title=f"{kind.capitalize()} preference")
        ax.set_xlabel("sweep s  /  selected algorithm", fontsize=8)
        ax = axes[1, j]
        for m, label in (("makespan_s", "makespan"), ("energy_J", "energy"), ("cost_usd", "cost")):
            ax.plot(x, g[m] / g[m].iloc[0], marker="o", label=label)
        ax.axhline(1, color="k", lw=0.6)
        ax.set(xticks=x, xticklabels=[f"{s:g}" for s in g.s], xlabel=f"{kind} preference sweep s",
               ylabel="relative to s = 0")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=150)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    table = summarize(run())
    table.to_csv(TAB / "table4_reproduction.csv", index=False)
    plot(table, "paper", "table4_reproduction.png")
    plot(table, "physical", "table4_physical_mapping.png")
    pd.set_option("display.width", 200)
    print(table.to_string(index=False))
    paper = pd.read_csv(DATA / "reference_paper_table4.csv", comment="#")
    print("\npaper (Table 4, units as printed):")
    print(paper.to_string(index=False))


if __name__ == "__main__":
    main()
