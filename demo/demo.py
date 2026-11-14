"""Live demo for the presentation (menu-driven, runs in a terminal in a few seconds per step).

    python -m demo.demo                 interactive menu
    python -m demo.demo decision 10 50 25 3 8      one decision, all three fuzzy systems
    python -m demo.demo simulate 100 101           one MONTAGE workflow (jobs, seed), all policies
    python -m demo.demo noise 0.1                  decision flips of type-1 vs. IT2 at one noise level

Every step prints *why* the system decided as it did, so that new input values given during the
defence can be explained on the spot.
"""
from __future__ import annotations

import sys

import numpy as np

from src.env import default_environment
from src.fuzzy.paper_fis import (INPUTS, build_fis, build_it2_fis, build_sugeno_fis, crisp_to_layer,
                                 explain)
from src.fuzzy.qos_selector import algorithm_strengths, select_algorithm
from src.policies import (FuzzyOffloadingPolicy, InputNoise, IT2OffloadingPolicy, SugenoOffloadingPolicy,
                          TwoWayFogCloudPolicy)
from src.scheduler import simulate
from src.workflow import montage

RANGES = {v.name: (v.lo, v.hi) for v in INPUTS}
NAMES = {"jl": "job list size", "vt": "VM utilization", "nl": "network latency", "bw": "bandwidth",
         "dl": "deadline"}


def decision(jl: float, vt: float, nl: float, bw: float, dl: float, fou: float = 0.05) -> None:
    values = {"jl": jl, "vt": vt, "nl": nl, "bw": bw, "dl": dl}
    print("\n=== Type-1 Mamdani (paper) ===")
    print(explain(build_fis(), values))
    y = build_sugeno_fis().compute(values)
    print("\n=== Zero-order Sugeno (ablation) ===")
    print(f"weighted average of the rule constants 40 / 100 / 160 = {y:.2f}  ->  {crisp_to_layer(y)}")
    res = build_it2_fis(fou).infer(values)
    yl, yr = res.interval
    print(f"\n=== Interval type-2, FOU = {fou:g} (extension) ===")
    print(f"firing intervals of the strongest rules: "
          + ", ".join(f"[{lo:.2f}, {up:.2f}]" for lo, up in
                      sorted(zip(res.firing_lower, res.firing_upper), key=lambda t: -t[1])[:3]))
    straddles = crisp_to_layer(yl) != crisp_to_layer(yr)
    print(f"type-reduced set [{yl:.2f}, {yr:.2f}] (Karnik-Mendel)  ->  {res.crisp:.2f}  ->  {crisp_to_layer(res.crisp)}"
          + ("   (the interval crosses a threshold: uncertain decision)" if straddles else ""))


def qos(time: float, energy: float, cost: float) -> None:
    s = algorithm_strengths(time, energy, cost)
    print(f"\nQoS preference T={time}, E={energy}, C={cost}")
    print("rule strengths: " + ", ".join(f"{a}={v:.2f}" for a, v in s.items()))
    print(f"selected algorithm: {select_algorithm(time, energy, cost)}"
          "   (A2 cost -> Cloud, A3 energy -> Fog, A4 time -> Edge, A5 hybrid -> 5-input FIS)")


def simulate_one(jobs: int = 100, seed: int = 101) -> None:
    env, wf = default_environment(), montage(jobs, seed=seed)
    print(f"\nMONTAGE workflow with {jobs} tasks, seed {seed}; 2 nodes per layer (paper's Table 3)")
    print(f"{'policy':34s} {'makespan s':>10s} {'energy J':>9s} {'cost $':>8s} {'misses':>7s}   edge/fog/cloud")
    for pol in (TwoWayFogCloudPolicy(), FuzzyOffloadingPolicy(record=False), SugenoOffloadingPolicy(record=False),
                IT2OffloadingPolicy(fou=0.05, record=False), FuzzyOffloadingPolicy(verify_deadline=False, record=False,
                                                                                    name="Proposed, no deadline check")):
        r = simulate(wf, env, pol).summary()
        print(f"{pol.name:34s} {r['makespan_s']:10.1f} {r['energy_J']:9.0f} {r['cost_usd']:8.1f} "
              f"{r['deadline_miss_rate']:7.1%}   {r['share_Edge']:.0%}/{r['share_Fog']:.0%}/{r['share_Cloud']:.0%}")


def noise(level: float = 0.1, jobs: int = 100, seed: int = 101) -> None:
    env, wf = default_environment(), montage(jobs, seed=seed)
    names = tuple(v.name for v in INPUTS)
    print(f"\nGaussian noise std = {level:g} x universe width on all five inputs; {jobs} tasks, seed {seed}")
    for label, pol in (("type-1", FuzzyOffloadingPolicy(noise=InputNoise(level, names, seed=seed))),
                       ("IT2 FOU 0.05", IT2OffloadingPolicy(fou=0.05, noise=InputNoise(level, names, seed=seed)))):
        r = simulate(wf, env, pol).summary()
        flips = np.mean([d.fis_layer != d.clean_layer for d in pol.decisions])
        print(f"{label:13s} layer flips vs. noise-free decision: {flips:6.1%}   makespan {r['makespan_s']:.1f} s")


def _ask(prompt: str, default: float) -> float:
    raw = input(f"{prompt} [{default}]: ").strip()
    return float(raw) if raw else default


def menu() -> None:
    while True:
        print("\n1) one offloading decision   2) QoS selector   3) simulate a workflow   4) noise   q) quit")
        choice = input("> ").strip().lower()
        try:
            if choice == "1":
                vals = [_ask(f"{k} {NAMES[k]} {RANGES[k]}", d) for k, d in
                        zip(RANGES, (10, 50, 25, 3, 8))]
                decision(*vals)
            elif choice == "2":
                qos(_ask("time", 1 / 3), _ask("energy", 1 / 3), _ask("cost", 1 / 3))
            elif choice == "3":
                simulate_one(int(_ask("jobs", 100)), int(_ask("seed", 101)))
            elif choice == "4":
                noise(_ask("noise level", 0.1))
            elif choice in ("q", "quit", "exit"):
                return
        except ValueError as exc:
            print(f"invalid input: {exc}")


def main(argv: list[str]) -> None:
    if not argv:
        menu()
    elif argv[0] == "decision":
        decision(*map(float, argv[1:6]))
    elif argv[0] == "qos":
        qos(*map(float, argv[1:4]))
    elif argv[0] == "simulate":
        simulate_one(*map(int, argv[1:3]))
    elif argv[0] == "noise":
        noise(float(argv[1]) if len(argv) > 1 else 0.1)
    else:
        print(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
