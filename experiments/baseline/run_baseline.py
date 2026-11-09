"""Baseline experiment: classical two-way fog-cloud offloading + reference policies.

Same setting as the paper's evaluation: MONTAGE-like workflows with 20-300 jobs, MIN-MIN
scheduling, Table 3 nodes.  Every configuration is repeated over several seeds (the paper reports
no repetitions or deviations; we report mean and standard deviation).

Outputs
  results/tables/baseline_runs.csv      one row per (policy, jobs, seed)
  results/tables/baseline_summary.csv   mean and std per (policy, jobs)
  results/figures/baseline_metrics.png  makespan / energy / cost vs. number of jobs

Run:  python -m experiments.baseline.run_baseline [--seeds 10]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.env import default_environment
from src.policies import AllLayersPolicy, SingleLayerPolicy, TwoWayFogCloudPolicy
from src.scheduler import simulate
from src.workflow import montage

ROOT = Path(__file__).resolve().parents[2]
FIG, TAB = ROOT / "results" / "figures", ROOT / "results" / "tables"
JOB_COUNTS = (20, 40, 60, 80, 100, 200, 300)  # paper, Sec. 3
METRICS = {"makespan_s": "Makespan (s)", "energy_J": "Energy (J)", "cost_usd": "Cost ($)"}


def policies():
    return [TwoWayFogCloudPolicy(), AllLayersPolicy(),
            SingleLayerPolicy("Edge"), SingleLayerPolicy("Fog"), SingleLayerPolicy("Cloud")]


def run(seeds: int) -> pd.DataFrame:
    env = default_environment()
    rows = []
    for jobs in JOB_COUNTS:
        for seed in range(seeds):
            wf = montage(jobs, seed=seed)  # every policy sees the same workflow
            for pol in policies():
                rows.append({**simulate(wf, env, pol).summary(), "seed": seed})
    return pd.DataFrame(rows)


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    cols = list(METRICS) + ["deadline_miss_rate", "share_Edge", "share_Fog", "share_Cloud"]
    out = runs.groupby(["policy", "jobs"])[cols].agg(["mean", "std"])
    out.columns = [f"{c}_{s}" for c, s in out.columns]
    return out.reset_index().round(4)


def plot(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (m, label) in zip(axes, METRICS.items()):
        for pol, g in summary.groupby("policy", sort=False):
            ax.errorbar(g["jobs"], g[f"{m}_mean"], yerr=g[f"{m}_std"], marker="o", ms=4,
                        capsize=3, label=pol, lw=2.2 if pol.startswith("Classical") else 1.2)
        ax.set(xlabel="Number of jobs", ylabel=label, title=label)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "baseline_metrics.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    runs = run(args.seeds)
    runs.to_csv(TAB / "baseline_runs.csv", index=False)
    summary = summarize(runs)
    summary.to_csv(TAB / "baseline_summary.csv", index=False)
    plot(summary)
    view = summary[summary["jobs"].isin([20, 100, 300])][
        ["policy", "jobs", "makespan_s_mean", "energy_J_mean", "cost_usd_mean",
         "deadline_miss_rate_mean", "share_Fog_mean", "share_Cloud_mean"]]
    print(view.to_string(index=False))
    print(f"tables -> {TAB}\nfigure -> {FIG / 'baseline_metrics.png'}")


if __name__ == "__main__":
    main()
