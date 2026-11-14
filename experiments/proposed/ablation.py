"""Ablation study: which design choices of the proposed method matter?

Every variant is the proposed policy with exactly one component changed (balanced QoS preference,
hybrid algorithm A5, as in Fig. 4):

  Inference       Mamdani (paper)  vs. zero-order Sugeno (constants 40 / 100 / 160)  vs. IT2 (FOU 0.05)
  Defuzzification centroid (paper, Eq. 21) vs. bisector, mean / smallest / largest of maximum
  Offloading condition  deadline check of Alg. 5 ("dl not met") on (paper) vs. off
  Energy accounting     idle power of used nodes only (ours) vs. of every node (``idle="all"``)

Part 2 continues the Day-5 robustness question: without the deadline check, do the decision flips
caused by measurement noise reach the QoS metrics, and does IT2 help there?  (noise on all five inputs)

Setting: MONTAGE 20-300 jobs (the Fig. 4 job counts), test seeds 100-109, MIN-MIN; all variants run
on identical workflows, so differences are paired per (jobs, seed).

Outputs
  results/tables/ablation_runs.csv            one row per (variant, idle, jobs, seed)
  results/tables/ablation_summary.csv         mean metrics, paired difference to the paper variant,
                                              improvement over the classical baseline
  results/tables/ablation_fis.csv             FIS level: layer agreement with Mamdani-COG on random
                                              inputs, output layer shares, time per decision
  results/tables/ablation_idle.csv            Fig. 4 improvement under both energy accountings
  results/tables/ablation_noise_nodeadline.csv  noise x deadline check, T1 vs IT2
  results/figures/ablation_variants.png
  results/figures/ablation_noise_nodeadline.png

Run:  python -m experiments.proposed.ablation        (~2 min)
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.env import default_environment
from src.fuzzy.paper_fis import INPUTS, build_fis, build_it2_fis, build_sugeno_fis, crisp_to_layer
from src.policies import (FuzzyOffloadingPolicy, InputNoise, IT2OffloadingPolicy, SugenoOffloadingPolicy,
                          TwoWayFogCloudPolicy)
from src.scheduler import simulate
from src.workflow import montage

ROOT = Path(__file__).resolve().parents[2]
FIG, TAB = ROOT / "results" / "figures", ROOT / "results" / "tables"
JOB_COUNTS = (20, 40, 60, 80, 100, 200, 300)
SEEDS = range(100, 110)
REFERENCE, CLASSICAL = "Mamdani-COG (paper)", "Classical two-way"
METRICS = {"makespan_s": "Makespan", "energy_J": "Energy", "cost_usd": "Cost", "deadline_miss_rate": "Deadline miss"}
NOISE_LEVELS = (0.0, 0.05, 0.1, 0.2, 0.3)
LAYERS = ("Edge", "Fog", "Cloud")

# variant label -> policy factory (one fresh policy object per simulation)
VARIANTS = {
    REFERENCE: lambda: FuzzyOffloadingPolicy(record=False),
    "Mamdani-bisector": lambda: FuzzyOffloadingPolicy(defuzz="bisector", record=False),
    "Mamdani-MOM": lambda: FuzzyOffloadingPolicy(defuzz="mom", record=False),
    "Mamdani-SOM": lambda: FuzzyOffloadingPolicy(defuzz="som", record=False),
    "Mamdani-LOM": lambda: FuzzyOffloadingPolicy(defuzz="lom", record=False),
    "Sugeno (zero-order)": lambda: SugenoOffloadingPolicy(record=False),
    "IT2 FOU=0.05": lambda: IT2OffloadingPolicy(fou=0.05, record=False),
    "No deadline check": lambda: FuzzyOffloadingPolicy(verify_deadline=False, record=False),
    CLASSICAL: TwoWayFogCloudPolicy,
}
IDLE_ALL_VARIANTS = (REFERENCE, CLASSICAL)  # Fig. 4 comparison under the other energy accounting


def run(seeds=SEEDS, job_counts=JOB_COUNTS) -> pd.DataFrame:
    env, rows = default_environment(), []
    for jobs in job_counts:
        for seed in seeds:
            wf = montage(jobs, seed=seed)
            for label, factory in VARIANTS.items():
                for idle in ("used", "all") if label in IDLE_ALL_VARIANTS else ("used",):
                    res = simulate(wf, env, factory(), idle=idle)
                    rows.append({**res.summary(), "variant": label, "idle": idle, "seed": seed})
    return pd.DataFrame(rows)


def _paired_pct(base: pd.Series, other: pd.Series, metric: str) -> pd.Series:
    """other vs. base in % of base (percentage points for the miss rate); positive = other is worse."""
    if metric == "deadline_miss_rate":
        return (other - base) * 100
    return (other - base) / base * 100


def _wilcoxon(diff: pd.Series) -> float:
    nz = diff[diff != 0]
    return float(stats.wilcoxon(nz).pvalue) if len(nz) > 0 else 1.0


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    used = runs[runs.idle == "used"]
    idx = ["jobs", "seed"]
    ref = used[used.variant == REFERENCE].set_index(idx).sort_index()
    cls = used[used.variant == CLASSICAL].set_index(idx).sort_index()
    rows = []
    for label in VARIANTS:
        g = used[used.variant == label].set_index(idx).sort_index()
        row = {"variant": label}
        for m in METRICS:
            row[f"{m}_mean"] = g[m].mean()
        for m in METRICS:
            d = _paired_pct(ref[m], g[m], m)
            row[f"{m}_vs_paper_pct"] = d.mean()
            row[f"{m}_vs_paper_p"] = _wilcoxon(d) if label != REFERENCE else np.nan
        for m in ("makespan_s", "energy_J", "cost_usd"):
            row[f"{m}_gain_vs_classical_pct"] = ((cls[m] - g[m]) / cls[m] * 100).mean()
        for layer in LAYERS:
            row[f"share_{layer}"] = g[f"share_{layer}"].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def fis_level(n: int = 3000) -> pd.DataFrame:
    """FIS alone on random inputs: layer agreement with the paper's Mamdani-COG, layer shares, time."""
    rng = np.random.default_rng(0)
    samples = [{v.name: rng.uniform(v.lo, v.hi) for v in INPUTS} for _ in range(n)]
    systems = {REFERENCE: build_fis("centroid"), "Mamdani-bisector": build_fis("bisector"),
               "Mamdani-MOM": build_fis("mom"), "Mamdani-SOM": build_fis("som"), "Mamdani-LOM": build_fis("lom"),
               "Sugeno (zero-order)": build_sugeno_fis(), "IT2 FOU=0.05": build_it2_fis(0.05)}
    ref = None
    rows = []
    for label, fis in systems.items():
        fis.compute(samples[0])  # warm-up
        t = time.perf_counter()
        crisp = np.array([fis.compute(s) for s in samples])
        us = (time.perf_counter() - t) / n * 1e6
        layers = np.array([crisp_to_layer(c) for c in crisp])
        if ref is None:
            ref = layers
        rows.append({"variant": label, "agreement_with_paper": (layers == ref).mean(),
                     **{f"share_{l}": (layers == l).mean() for l in LAYERS},
                     "crisp_std": crisp.std(), "us_per_decision": us})
    return pd.DataFrame(rows)


def idle_accounting(runs: pd.DataFrame) -> pd.DataFrame:
    """Energy improvement of the proposed method over the classical baseline under both accountings."""
    rows = []
    for idle, g in runs[runs.variant.isin(IDLE_ALL_VARIANTS)].groupby("idle"):
        c = g[g.variant == CLASSICAL].set_index(["jobs", "seed"]).sort_index()
        p = g[g.variant == REFERENCE].set_index(["jobs", "seed"]).sort_index()
        rel = (c.energy_J - p.energy_J) / c.energy_J * 100
        rows.append({"idle_power_of": "used nodes" if idle == "used" else "all nodes", "energy_classical_J": c.energy_J.mean(),
                     "energy_proposed_J": p.energy_J.mean(), "energy_gain_pct": rel.mean(),
                     "share_runs_better_pct": (rel > 0).mean() * 100, "wilcoxon_p": _wilcoxon(rel)})
    return pd.DataFrame(rows)


def noise_without_deadline_check(seeds=SEEDS, job_counts=(20, 60, 100, 200, 300)) -> pd.DataFrame:
    """T1 vs IT2 under noise on all five inputs, with and without the deadline check."""
    env, rows = default_environment(), []
    for jobs in job_counts:
        for seed in seeds:
            wf = montage(jobs, seed=seed)
            for verify in (True, False):
                for level in NOISE_LEVELS:
                    noise = InputNoise(level, tuple(v.name for v in INPUTS), seed=seed) if level > 0 else None
                    for fis, pol in (("T1", FuzzyOffloadingPolicy(verify_deadline=verify, noise=noise)),
                                     ("IT2", IT2OffloadingPolicy(fou=0.05, verify_deadline=verify, noise=noise))):
                        res = simulate(wf, env, pol)
                        flips = np.mean([d.fis_layer != d.clean_layer for d in pol.decisions]) if noise else 0.0
                        rows.append({**res.summary(), "fis": fis, "deadline_check": verify, "level": level,
                                     "seed": seed, "flip_rate": flips})
    raw = pd.DataFrame(rows)
    # degradation relative to the noise-free run of the same (fis, check, jobs, seed)
    key = ["fis", "deadline_check", "jobs", "seed"]
    base = raw[raw.level == 0].set_index(key)[list(METRICS)]
    raw = raw.join(base, on=key, rsuffix="_clean")
    for m in ("makespan_s", "energy_J", "cost_usd"):
        raw[f"{m}_deg_pct"] = (raw[m] - raw[f"{m}_clean"]) / raw[f"{m}_clean"] * 100
    raw["deadline_miss_deg_pp"] = (raw.deadline_miss_rate - raw.deadline_miss_rate_clean) * 100
    cols = ["flip_rate", "makespan_s", "makespan_s_deg_pct", "energy_J_deg_pct", "cost_usd_deg_pct",
            "deadline_miss_rate", "deadline_miss_deg_pp"]
    out = raw.groupby(["deadline_check", "level", "fis"])[cols].mean().reset_index()
    # paired IT2 vs T1 makespan (positive = IT2 better)
    p_vals = []
    for (verify, level), g in raw.groupby(["deadline_check", "level"]):
        t1 = g[g.fis == "T1"].set_index(["jobs", "seed"]).sort_index()
        it2 = g[g.fis == "IT2"].set_index(["jobs", "seed"]).sort_index()
        d = (t1.makespan_s - it2.makespan_s) / t1.makespan_s * 100
        p_vals.append({"deadline_check": verify, "level": level, "it2_makespan_gain_pct": d.mean(),
                       "it2_makespan_wilcoxon_p": _wilcoxon(d)})
    return out.merge(pd.DataFrame(p_vals), on=["deadline_check", "level"])


def plot_variants(summary: pd.DataFrame) -> None:
    s = summary[~summary.variant.isin([REFERENCE, CLASSICAL])]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.6), sharey=True)
    y = np.arange(len(s))
    for ax, (m, label) in zip(axes, METRICS.items()):
        vals = s[f"{m}_vs_paper_pct"].to_numpy()
        sig = s[f"{m}_vs_paper_p"].to_numpy() < 0.05
        colors = ["#d62728" if v > 0 else "#2ca02c" for v in vals]
        ax.barh(y, vals, color=colors, alpha=0.85)
        for yi, v, k in zip(y, vals, sig):
            if k:
                ax.text(v, yi, " * " if v >= 0 else " * ", va="center", ha="left" if v >= 0 else "right", fontsize=10)
        ax.axvline(0, color="k", lw=0.8)
        unit = "pp" if m == "deadline_miss_rate" else "%"
        ax.set(title=label, xlabel=f"vs. Mamdani-COG ({unit}), >0 = worse")
        ax.grid(axis="x", alpha=0.3)
    axes[0].set_yticks(y, s.variant)
    axes[0].invert_yaxis()
    fig.suptitle("Ablation: one component changed at a time (balanced QoS, 20-300 jobs, 10 seeds; * = Wilcoxon p < 0.05)")
    fig.tight_layout()
    fig.savefig(FIG / "ablation_variants.png", dpi=150)
    plt.close(fig)


def plot_noise(noise: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    panels = [("flip_rate", "Decision flip rate", 100), ("makespan_s_deg_pct", "Makespan increase vs. noise-free (%)", 1),
              ("deadline_miss_deg_pp", "Deadline-miss increase (pp)", 1)]
    for ax, (col, title, scale) in zip(axes, panels):
        for (verify, fis), g in noise.groupby(["deadline_check", "fis"]):
            ax.plot(g.level, g[col] * scale, marker="o" if fis == "T1" else "s",
                    ls="-" if verify else "--", color="#1f77b4" if fis == "T1" else "#ff7f0e",
                    label=f"{fis}, deadline check {'on' if verify else 'off'}")
        ax.set(title=title, xlabel="Noise level (std / universe width)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("%")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "ablation_noise_nodeadline.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    runs = run()
    runs.to_csv(TAB / "ablation_runs.csv", index=False)
    summary = summarize(runs)
    summary.round(4).to_csv(TAB / "ablation_summary.csv", index=False)
    fis = fis_level()
    fis.round(4).to_csv(TAB / "ablation_fis.csv", index=False)
    idle = idle_accounting(runs)
    idle.round(4).to_csv(TAB / "ablation_idle.csv", index=False)
    noise = noise_without_deadline_check()
    noise.round(4).to_csv(TAB / "ablation_noise_nodeadline.csv", index=False)
    plot_variants(summary)
    plot_noise(noise)
    show = ["variant"] + [f"{m}_vs_paper_pct" for m in METRICS] + [f"{m}_vs_paper_p" for m in METRICS] \
        + ["makespan_s_gain_vs_classical_pct", "energy_J_gain_vs_classical_pct", "cost_usd_gain_vs_classical_pct",
           "share_Edge", "share_Fog", "share_Cloud"]
    print(summary[show].round(3).to_string(index=False))
    print(fis.round(3).to_string(index=False))
    print(idle.round(3).to_string(index=False))
    print(noise.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
