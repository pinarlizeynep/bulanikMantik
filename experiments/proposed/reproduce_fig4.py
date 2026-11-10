"""Reproduce the paper's Fig. 4: proposed fuzzy offloading vs. the classical two-way hybrid.

Same setting as the paper (MONTAGE, MIN-MIN, 20-300 jobs) with 10 independent seeds (100-109;
seeds 0-2 were used only to scale the FIS inputs, see docs/gunluk/gun4).  The paper reports
single curves without repetitions; we report mean +- std and a paired comparison.

Outputs
  results/tables/fig4_runs.csv          one row per (policy, jobs, seed)
  results/tables/fig4_summary.csv       mean / std per (policy, jobs)
  results/tables/fig4_improvement.csv   overall improvement (%) per metric: ours vs. paper
  results/figures/fig4_reproduction.png makespan / energy / cost bars + improvement panel

Run:  python -m experiments.proposed.reproduce_fig4
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.env import default_environment
from src.policies import FuzzyOffloadingPolicy, TwoWayFogCloudPolicy
from src.scheduler import simulate
from src.workflow import montage

ROOT = Path(__file__).resolve().parents[2]
FIG, TAB, DATA = ROOT / "results" / "figures", ROOT / "results" / "tables", ROOT / "data"
JOB_COUNTS = (20, 40, 60, 80, 100, 200, 300)
SEEDS = range(100, 110)
CLASSICAL, PROPOSED = "Classical two-way (fog-cloud)", "Proposed fuzzy (Soni & Kumar)"
METRICS = {"makespan_s": "Makespan", "energy_J": "Energy", "cost_usd": "Cost"}
UNITS = {"makespan_s": "s", "energy_J": "J", "cost_usd": "$"}
PAPER_COLS = {"makespan_s": ("makespan_classical_ms", "makespan_proposed_ms"),
              "energy_J": ("energy_classical_J", "energy_proposed_J"),
              "cost_usd": ("cost_classical_usd", "cost_proposed_usd")}


def run(seeds=SEEDS) -> pd.DataFrame:
    env, rows = default_environment(), []
    for jobs in JOB_COUNTS:
        for seed in seeds:
            wf = montage(jobs, seed=seed)  # both policies see the identical workflow
            for pol in (TwoWayFogCloudPolicy(), FuzzyOffloadingPolicy(record=False)):
                rows.append({**simulate(wf, env, pol).summary(), "seed": seed})
    return pd.DataFrame(rows)


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    cols = list(METRICS) + ["deadline_miss_rate", "offload_s", "share_Edge", "share_Fog", "share_Cloud"]
    out = runs.groupby(["policy", "jobs"], sort=False)[cols].agg(["mean", "std"])
    out.columns = [f"{c}_{s}" for c, s in out.columns]
    return out.reset_index().round(4)


def paired_improvement(runs: pd.DataFrame) -> pd.DataFrame:
    """Relative improvement (classical - proposed) / classical, paired per (jobs, seed)."""
    c = runs[runs.policy == CLASSICAL].set_index(["jobs", "seed"])
    p = runs[runs.policy == PROPOSED].set_index(["jobs", "seed"])
    paper = pd.read_csv(DATA / "reference_paper_values.csv", comment="#")
    rows = []
    for m, label in METRICS.items():
        rel = (c[m] - p[m]) / c[m] * 100
        per_jobs = rel.groupby(level="jobs").mean()
        pc, pp = PAPER_COLS[m]
        paper_rel = ((paper[pc] - paper[pp]) / paper[pc] * 100).mean()
        t = stats.ttest_1samp(rel, 0.0)
        rows.append({"metric": label, "ours_mean_pct": rel.mean(), "ours_std_pct": rel.std(),
                     "ours_best_jobs": int(per_jobs.idxmax()), "share_of_runs_better_pct": (rel > 0).mean() * 100,
                     "t_test_p": f"{t.pvalue:.2e}", "paper_pct": paper_rel,
                     "paper_reported_pct": {"Makespan": 7.51, "Energy": 4.63, "Cost": 13.60}[label]})
    return pd.DataFrame(rows)


def plot(summary: pd.DataFrame, imp: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    x, w = np.arange(len(JOB_COUNTS)), 0.38
    for ax, (m, label) in zip(axes.flat, METRICS.items()):
        for k, (pol, color) in enumerate(((CLASSICAL, "#9aa5b1"), (PROPOSED, "#1f77b4"))):
            g = summary[summary.policy == pol].set_index("jobs").loc[list(JOB_COUNTS)]
            ax.bar(x + (k - 0.5) * w, g[f"{m}_mean"], w, yerr=g[f"{m}_std"], capsize=2, color=color, label=pol)
        ax.set(xticks=x, xticklabels=JOB_COUNTS, xlabel="Number of jobs", ylabel=f"{label} ({UNITS[m]})",
               title=f"({'ABC'[list(METRICS).index(m)]}) {label}")
        ax.grid(axis="y", alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    ax = axes[1, 1]
    xi = np.arange(len(imp))
    ax.bar(xi - w / 2, imp["ours_mean_pct"], w, yerr=imp["ours_std_pct"], capsize=3, color="#1f77b4", label="this reproduction")
    ax.bar(xi + w / 2, imp["paper_reported_pct"], w, color="#ff7f0e", label="paper (reported)")
    ax.axhline(0, color="k", lw=0.8)
    ax.set(xticks=xi, xticklabels=imp["metric"], ylabel="Improvement over classical (%)", title="(D) Overall improvement")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_reproduction.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    runs = run()
    runs.to_csv(TAB / "fig4_runs.csv", index=False)
    summary = summarize(runs)
    summary.to_csv(TAB / "fig4_summary.csv", index=False)
    imp = paired_improvement(runs)
    imp.round(4).to_csv(TAB / "fig4_improvement.csv", index=False)
    plot(summary, imp)
    pd.set_option("display.width", 200)
    print(imp.round(2).to_string(index=False))
    print("(ours_mean_pct > 0 means the proposed method is better; paper_pct = from the digitised Fig. 4A-C bars,"
          " paper_reported_pct = numbers printed in the text)")
    show = summary[summary.jobs.isin([20, 100, 300])][
        ["policy", "jobs", "makespan_s_mean", "energy_J_mean", "cost_usd_mean", "deadline_miss_rate_mean",
         "share_Edge_mean", "share_Fog_mean", "share_Cloud_mean"]]
    print(show.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
