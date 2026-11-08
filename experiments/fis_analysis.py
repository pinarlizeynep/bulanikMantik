"""Static analysis of the reproduced FIS (no simulator needed).

Outputs
  results/figures/membership_functions.png   reproduction of the paper's Fig. 3
  results/figures/control_surfaces.png        decision surface for pairs of inputs
  results/tables/fis_decision_stats.csv       layer distribution over random inputs + time per decision

Run:  python -m experiments.fis_analysis
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.fuzzy.paper_fis import (
    DECISION, EDGE_FOG_THRESHOLD, FOG_CLOUD_THRESHOLD, INPUTS, build_fis, crisp_to_layer,
)

ROOT = Path(__file__).resolve().parents[1]
FIG, TAB = ROOT / "results" / "figures", ROOT / "results" / "tables"
LABELS = {"jl": "Job List Size", "vt": "VM Utilization", "nl": "Network Latency (Delay)",
          "bw": "Bandwidth", "dl": "Deadline", "decision": "Compute Node Decision"}
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]
MID = {v.name: (v.lo + v.hi) / 2 for v in INPUTS}


def plot_membership_functions():
    fig, axes = plt.subplots(3, 2, figsize=(10, 10))
    for ax, var, tag in zip(axes.flat, INPUTS + [DECISION], "ABCDEF"):
        x = var.universe(1001)
        for (term, mf), c in zip(var.terms.items(), COLORS):
            ax.plot(x, mf(x), lw=2, color=c, label=term)
        if var is DECISION:
            for t in (EDGE_FOG_THRESHOLD, FOG_CLOUD_THRESHOLD):
                ax.axvline(t, ls="--", color="grey", lw=1)
        ax.set(title=f"({tag}) {LABELS[var.name]}", ylabel="Membership", ylim=(-0.03, 1.05))
        ax.legend(fontsize=8, loc="center right")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "membership_functions.png", dpi=150)
    plt.close(fig)


def plot_control_surfaces(fis, n=31):
    pairs = [("jl", "vt"), ("nl", "dl"), ("bw", "nl"), ("jl", "dl")]
    var = {v.name: v for v in INPUTS}
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, (a, b) in zip(axes.flat, pairs):
        xa, xb = np.linspace(var[a].lo, var[a].hi, n), np.linspace(var[b].lo, var[b].hi, n)
        z = np.empty((n, n))
        for i, vb in enumerate(xb):
            for j, va in enumerate(xa):
                z[i, j] = fis.compute({**MID, a: va, b: vb})
        im = ax.contourf(xa, xb, z, levels=np.linspace(0, 200, 21), cmap="viridis")
        cs = ax.contour(xa, xb, z, levels=[EDGE_FOG_THRESHOLD, FOG_CLOUD_THRESHOLD], colors="white")
        ax.clabel(cs, fmt={EDGE_FOG_THRESHOLD: "Edge|Fog", FOG_CLOUD_THRESHOLD: "Fog|Cloud"}, fontsize=8)
        ax.set(xlabel=LABELS[a], ylabel=LABELS[b], title=f"{a} vs {b} (others at mid-range)")
        fig.colorbar(im, ax=ax, label="crisp decision")
    fig.tight_layout()
    fig.savefig(FIG / "control_surfaces.png", dpi=150)
    plt.close(fig)


def decision_stats(fis, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    samples = [{v.name: rng.uniform(v.lo, v.hi) for v in INPUTS} for _ in range(n)]
    t0 = time.perf_counter()
    crisp = [fis.compute(s) for s in samples]
    per_decision_ms = (time.perf_counter() - t0) / n * 1e3
    layers = pd.Series([crisp_to_layer(c) for c in crisp]).value_counts(normalize=True) * 100
    df = pd.DataFrame({
        "layer": ["Edge", "Fog", "Cloud"],
        "share_percent": [round(layers.get(k, 0.0), 2) for k in ("Edge", "Fog", "Cloud")],
    })
    df["samples"] = n
    df["mean_ms_per_decision"] = round(per_decision_ms, 3)
    df.to_csv(TAB / "fis_decision_stats.csv", index=False)
    return df


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    fis = build_fis()
    plot_membership_functions()
    plot_control_surfaces(fis)
    print(decision_stats(fis).to_string(index=False))
    print(f"figures -> {FIG}\ntables  -> {TAB}")


if __name__ == "__main__":
    main()
