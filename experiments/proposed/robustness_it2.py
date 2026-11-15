"""Extension experiment: robustness of type-1 vs. interval type-2 fuzzy offloading to measurement noise.

The observed FIS inputs are perturbed with Gaussian noise (``InputNoise``): std = level x universe
width, clipped to the universe.  Two scenarios:
  network   nl, bw, vt  (the quantities a fog broker measures over the network / from the VMs)
  all       jl, vt, nl, bw, dl
Policies: the paper's type-1 FIS ("T1") and the IT2 FIS with FOU half-widths 0.025 ... 0.2.  The main
IT2 configuration (``MAIN_FOU`` = 0.05) was fixed beforehand on the tuning seeds 0-2: it is the largest
FOU whose noise-free makespan / energy / cost stay within about 1 % of the type-1 system (technical report, Sec. VII).
The classical two-way baseline does not read the FIS inputs and is run once per workflow as reference.

Decision stability is measured *at the identical simulator state*: at every hybrid (A5) decision the
policy also evaluates the noise-free inputs, so a "flip" is a FIS layer decision (Edge / Fog / Cloud,
before the deadline check) that differs from the noise-free one.  Because a system that says "Fog"
for everything never flips, we also report Cohen's kappa between noisy and clean decisions (agreement
corrected for chance) and the crisp deviation |C_noisy - C_clean| relative to the spread (std) of the
clean crisp outputs.

Balanced QoS preference (hybrid A5, as in Fig. 4), MONTAGE 20-300 jobs, test seeds 100-109; noise draws
depend only on (seed, task), so T1 and IT2 see the same errors (paired comparison).

Outputs
  results/tables/robustness_runs.csv        one row per (policy, scenario, level, jobs, seed)
  results/tables/robustness_summary.csv     mean / std per (policy, scenario, level) + pooled kappa
  results/tables/robustness_paired.csv      IT2 (main FOU) vs T1, paired per (jobs, seed)
  results/tables/it2_decision_time.csv      time per FIS evaluation
  results/figures/it2_membership_functions.png  FOU of every variable (main FOU)
  results/figures/robustness_stability.png  flip rate / kappa / relative crisp deviation vs. noise
  results/figures/robustness_metrics.png    makespan / energy / cost / deadline misses vs. noise

Run:  python -m experiments.proposed.robustness_it2        (~2 min)
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
from src.fuzzy.it2 import IT2Variable
from src.fuzzy.paper_fis import DECISION, INPUTS, build_fis, build_it2_fis
from src.policies import FuzzyOffloadingPolicy, InputNoise, IT2OffloadingPolicy, TwoWayFogCloudPolicy
from src.scheduler import simulate
from src.workflow import montage

ROOT = Path(__file__).resolve().parents[2]
FIG, TAB = ROOT / "results" / "figures", ROOT / "results" / "tables"
JOB_COUNTS = (20, 60, 100, 200, 300)
SEEDS = range(100, 110)
LEVELS = (0.0, 0.05, 0.1, 0.2, 0.3)
SCENARIOS = {"network": ("nl", "bw", "vt"), "all": ("jl", "vt", "nl", "bw", "dl")}
FOUS = (0.025, 0.05, 0.1, 0.2)
MAIN_FOU = 0.05
METRICS = {"makespan_s": "Makespan (s)", "energy_J": "Energy (J)", "cost_usd": "Cost ($)",
           "deadline_miss_rate": "Deadline miss rate"}
LAYERS = ("Edge", "Fog", "Cloud")


def policy_label(fou: float | None) -> str:
    return "T1" if fou is None else f"IT2 FOU={fou:g}"


def make_policy(fou: float | None, noise: InputNoise | None):
    if fou is None:
        return FuzzyOffloadingPolicy(noise=noise)
    return IT2OffloadingPolicy(fou=fou, noise=noise)


def run(seeds=SEEDS, job_counts=JOB_COUNTS, levels=LEVELS, fous=FOUS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (runs, decisions): one row per simulation, one row per hybrid decision with noise."""
    env, rows, dec = default_environment(), [], []
    for jobs in job_counts:
        for seed in seeds:
            wf = montage(jobs, seed=seed)
            rows.append({**simulate(wf, env, TwoWayFogCloudPolicy()).summary(), "policy": "Classical",
                         "scenario": "-", "level": 0.0, "seed": seed})
            for scenario, variables in SCENARIOS.items():
                for level in levels:
                    if level == 0 and scenario != "network":
                        continue  # the noise-free run is the same in both scenarios
                    noise = InputNoise(level, variables, seed=seed) if level > 0 else None
                    for fou in (None, *fous):
                        pol = make_policy(fou, noise)
                        res = simulate(wf, env, pol)
                        label = policy_label(fou)
                        crisp = np.array([d.crisp for d in pol.decisions])
                        row = {**res.summary(), "policy": label, "scenario": scenario, "level": level,
                               "seed": seed, "fis_edge": np.mean([d.fis_layer == "Edge" for d in pol.decisions]),
                               "fis_cloud": np.mean([d.fis_layer == "Cloud" for d in pol.decisions])}
                        if noise is not None:
                            clean = np.array([d.clean_crisp for d in pol.decisions])
                            row["flip_rate"] = np.mean([d.fis_layer != d.clean_layer for d in pol.decisions])
                            row["crisp_dev"] = np.mean(np.abs(crisp - clean))
                            row["crisp_dev_rel"] = row["crisp_dev"] / max(np.std(clean), 1e-9)
                            dec += [{"policy": label, "scenario": scenario, "level": level,
                                     "clean": d.clean_layer, "noisy": d.fis_layer} for d in pol.decisions]
                        else:
                            row.update(flip_rate=0.0, crisp_dev=0.0, crisp_dev_rel=0.0)
                        rows.append(row)
    runs = pd.DataFrame(rows)
    # the noise-free runs belong to both scenarios
    free = runs[(runs.level == 0) & (runs.policy != "Classical")].assign(scenario="all")
    return pd.concat([runs, free], ignore_index=True), pd.DataFrame(dec)


def cohen_kappa(a: pd.Series, b: pd.Series) -> float:
    """Agreement of two categorical labelings corrected for chance (1 = identical, 0 = chance)."""
    po = float((a.values == b.values).mean())
    pe = sum(float((a == c).mean()) * float((b == c).mean()) for c in LAYERS)
    return 1.0 if pe >= 1 else (po - pe) / (1 - pe)


def summarize(runs: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    cols = list(METRICS) + ["flip_rate", "crisp_dev", "crisp_dev_rel", "share_Edge", "share_Fog",
                            "share_Cloud", "fis_edge", "fis_cloud"]
    fuzzy = runs[runs.policy != "Classical"]
    out = fuzzy.groupby(["policy", "scenario", "level"], sort=False)[cols].agg(["mean", "std"])
    out.columns = [f"{c}_{s}" for c, s in out.columns]
    out = out.reset_index()
    kappa = {k: cohen_kappa(g.clean, g.noisy) for k, g in decisions.groupby(["policy", "scenario", "level"])}
    out["kappa"] = [kappa.get((p, s, l), 1.0) for p, s, l in zip(out.policy, out.scenario, out.level)]
    order = {s: i for i, s in enumerate(SCENARIOS)}
    out = out.sort_values(["scenario", "level"], key=lambda c: c.map(order) if c.name == "scenario" else c,
                          kind="stable")  # noise-free rows of "all" were appended last
    return out.reset_index(drop=True).round(4)


def paired(runs: pd.DataFrame, fou: float = MAIN_FOU) -> pd.DataFrame:
    """IT2 vs T1 on the same workflow and the same noise draws.

    Metrics: relative difference (T1 - IT2) / T1 in %, positive = IT2 better (lower).  Flip rate:
    difference in percentage points, positive = IT2 flips less.  p-values: paired t-test on the
    differences and Wilcoxon signed-rank test (non-parametric), as in reproduce_fig4.
    """
    rows = []
    idx = ["jobs", "seed"]
    it2_label = policy_label(fou)
    for (scenario, level), g in runs[runs.policy != "Classical"].groupby(["scenario", "level"], sort=False):
        t1 = g[g.policy == "T1"].set_index(idx).sort_index()
        it2 = g[g.policy == it2_label].set_index(idx).sort_index()
        for m in list(METRICS) + ["flip_rate"]:
            if m in ("flip_rate", "deadline_miss_rate"):
                diff = (t1[m] - it2[m]) * 100  # percentage points
            else:
                diff = (t1[m] - it2[m]) / t1[m] * 100
            if level == 0 and m == "flip_rate":
                continue
            nonzero = diff[diff != 0]
            w = stats.wilcoxon(nonzero).pvalue if len(nonzero) > 0 else 1.0
            rows.append({"scenario": scenario, "level": level, "metric": m, "it2_better_mean": diff.mean(),
                         "std": diff.std(), "share_it2_better_pct": (diff > 0).mean() * 100,
                         "share_equal_pct": (diff == 0).mean() * 100,
                         "t_test_p": f"{stats.ttest_1samp(diff, 0.0).pvalue:.2e}" if diff.std() > 0 else "nan",
                         "wilcoxon_p": f"{w:.2e}"})
    out = pd.DataFrame(rows)
    order = {s: i for i, s in enumerate(SCENARIOS)}
    return out.sort_values(["scenario", "level"], key=lambda c: c.map(order) if c.name == "scenario" else c,
                           kind="stable").reset_index(drop=True)


def decision_time(n: int = 2000) -> pd.DataFrame:
    """Mean time per FIS evaluation on random inputs (type-1 COG vs. IT2 with KM type reduction)."""
    rng = np.random.default_rng(0)
    samples = [{v.name: rng.uniform(v.lo, v.hi) for v in INPUTS} for _ in range(n)]
    rows = []
    for label, fis in [("T1", build_fis())] + [(policy_label(f), build_it2_fis(f)) for f in FOUS]:
        t = time.perf_counter()
        for s in samples:
            fis.compute(s)
        rows.append({"fis": label, "us_per_decision": (time.perf_counter() - t) / n * 1e6})
    return pd.DataFrame(rows).round(1)


# ---------------------------------------------------------------- figures
def plot_fou(fou: float = MAIN_FOU) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    colors = ("#1f77b4", "#2ca02c", "#d62728")
    for ax, var in zip(axes.flat, INPUTS + [DECISION]):
        it2 = IT2Variable(var, fou if var is not DECISION else 0.0)
        x = var.universe(801)
        for (term, m), c in zip(it2.terms.items(), colors):
            ax.fill_between(x, m.lower(x), m.upper(x), color=c, alpha=0.25)
            ax.plot(x, var.terms[term](x), color=c, lw=1.2, label=term)
        ax.set(title=f"{var.name}" + (" (type-1 consequents)" if var is DECISION else ""), ylim=(0, 1.05))
        ax.legend(fontsize=7)
    fig.suptitle(f"Interval type-2 MFs: principal MF = paper's Fig. 3, shaded FOU = +-{fou:g} x universe width")
    fig.tight_layout()
    fig.savefig(FIG / "it2_membership_functions.png", dpi=150)
    plt.close(fig)


def _style(label: str) -> dict:
    if label == "T1":
        return {"color": "k", "lw": 2.2, "marker": "o"}
    fou = float(label.split("=")[1])
    shade = plt.cm.Blues(0.35 + 0.6 * FOUS.index(fou) / (len(FOUS) - 1))
    return {"color": shade, "lw": 2.2 if fou == MAIN_FOU else 1.2, "marker": "s" if fou == MAIN_FOU else "."}


def plot_stability(summary: pd.DataFrame) -> None:
    panels = [("flip_rate_mean", "Layer flips vs. noise-free (%)", 100), ("kappa", "Cohen's kappa (clean vs. noisy)", 1),
              ("crisp_dev_rel_mean", "|C_noisy - C_clean| / std(C_clean)", 1)]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for i, scenario in enumerate(SCENARIOS):
        g = summary[summary.scenario == scenario]
        for j, (col, label, k) in enumerate(panels):
            ax = axes[i, j]
            for pol, gp in g.groupby("policy", sort=False):
                gp = gp.sort_values("level")
                ax.plot(gp.level, gp[col] * k, label=pol, **_style(pol))
            ax.set(xlabel="noise std (fraction of universe)", ylabel=label,
                   title=f"{scenario} noise ({', '.join(SCENARIOS[scenario])})")
            ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "robustness_stability.png", dpi=150)
    plt.close(fig)


def plot_metrics(runs: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    classical = runs[runs.policy == "Classical"][list(METRICS)].mean()
    for i, scenario in enumerate(SCENARIOS):
        g = runs[(runs.scenario == scenario)]
        for j, (m, label) in enumerate(METRICS.items()):
            ax = axes[i, j]
            for pol in ["T1", policy_label(MAIN_FOU), policy_label(max(FOUS))]:
                gp = g[g.policy == pol].groupby("level")[m].agg(["mean", "sem"])
                ax.errorbar(gp.index, gp["mean"], yerr=gp["sem"], capsize=2, label=pol, **_style(pol))
            ax.axhline(classical[m], color="#ff7f0e", ls="--", lw=1.2, label="Classical (no FIS)")
            ax.set(xlabel="noise std (fraction of universe)", ylabel=label, title=f"{scenario} noise")
            ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Mean over 20-300 jobs x 10 seeds (error bar = standard error)")
    fig.tight_layout()
    fig.savefig(FIG / "robustness_metrics.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    runs, decisions = run()
    runs.round(6).to_csv(TAB / "robustness_runs.csv", index=False)
    summary = summarize(runs, decisions)
    summary.to_csv(TAB / "robustness_summary.csv", index=False)
    pair = paired(runs)
    pair.round(4).to_csv(TAB / "robustness_paired.csv", index=False)
    timing = decision_time()
    timing.to_csv(TAB / "it2_decision_time.csv", index=False)
    plot_fou()
    plot_stability(summary)
    plot_metrics(runs)
    pd.set_option("display.width", 220)
    show = summary[["policy", "scenario", "level", "flip_rate_mean", "kappa", "crisp_dev_rel_mean", "fis_edge_mean",
                    "fis_cloud_mean", "makespan_s_mean", "energy_J_mean", "cost_usd_mean",
                    "deadline_miss_rate_mean"]]
    print(show.round(3).to_string(index=False))
    print(pair.round(3).to_string(index=False))
    print(timing.to_string(index=False))
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
