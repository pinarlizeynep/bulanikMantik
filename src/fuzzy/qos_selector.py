"""Second decision layer of the paper: choose the offloading algorithm from the user's QoS preference.

The paper (Sec. 2.2, Table 2, Algorithm 1) lets the user give a preference degree in [0, 1] for
time (T), energy (E) and cost (C).  Each degree is fuzzified into Low / Medium / High and 16 rules
select one of four algorithms:

    A2  cost-only        -> target layer: cloud
    A3  energy-only      -> target layer: fog
    A4  time-only        -> target layer: edge (local execution)
    A5  hybrid           -> the 5-input FIS of ``paper_fis`` decides per job

The target layer is where the policy sends a job; the deadline check of
``FuzzyOffloadingPolicy`` (Alg. 5, "dl not met") can still move single jobs of A2 / A3 elsewhere.

The paper writes the sets as L = (0, 0.5], M = [0.2, 0.8], H = [0.5, 1] without giving shapes;
we use a left shoulder, a triangle and a right shoulder with exactly those supports.
Rule strength = min over the three antecedents; the algorithm with the strongest rule wins.
Table 2 does not cover every combination (e.g. T = E = C = 1/3 fires no rule); then the hybrid
algorithm A5 is used, which is also the sensible default for a balanced preference.
"""
from __future__ import annotations

from .membership import LinguisticVariable, trap, tri


def _preference(name: str) -> LinguisticVariable:
    return LinguisticVariable(name, 0.0, 1.0, {
        "L": trap(0.0, 0.0, 0.2, 0.5), "M": tri(0.2, 0.5, 0.8), "H": trap(0.5, 0.8, 1.0, 1.0)})


TIME, ENERGY, COST = _preference("T"), _preference("E"), _preference("C")

# (time, energy, cost) -> algorithm: Table 2, rules R1-R16 in the paper's order.
TABLE2 = [
    ("L", "L", "H", "A2"), ("L", "M", "H", "A2"), ("M", "M", "H", "A2"),   # R1-R3   cost
    ("L", "H", "L", "A3"), ("L", "H", "M", "A3"), ("M", "H", "M", "A3"),   # R4-R6   energy
    ("H", "L", "L", "A4"), ("H", "L", "M", "A4"), ("H", "M", "M", "A4"),   # R7-R9   time
    ("L", "H", "H", "A5"), ("M", "H", "H", "A5"),                          # R10-R11 energy + cost
    ("H", "H", "L", "A5"), ("H", "H", "M", "A5"),                          # R12-R13 time + energy
    ("H", "L", "H", "A5"), ("H", "M", "H", "A5"), ("H", "H", "H", "A5"),   # R14-R16 time + cost / all
]
ALGORITHM_LAYER = {"A2": "Cloud", "A3": "Fog", "A4": "Edge"}  # as published; A5 has no fixed layer
# Sensitivity analysis: the published mapping contradicts the paper's own Table 3 (the cloud has the
# fastest CPU but the highest prices, the edge is free but slowest).  The physically consistent
# assignment sends the cost-driven algorithm to the edge and the time-driven one to the cloud.
PHYSICAL_LAYER = {"A2": "Edge", "A3": "Fog", "A4": "Cloud"}


def algorithm_strengths(time: float, energy: float, cost: float) -> dict[str, float]:
    mt, me, mc = TIME.fuzzify(time), ENERGY.fuzzify(energy), COST.fuzzify(cost)
    strength = {"A2": 0.0, "A3": 0.0, "A4": 0.0, "A5": 0.0}
    for t, e, c, algo in TABLE2:
        strength[algo] = max(strength[algo], min(mt[t], me[e], mc[c]))
    return strength


def select_algorithm(time: float, energy: float, cost: float) -> str:
    """Algorithm 1: A2 / A3 / A4 / A5 for a preference (time, energy, cost) in [0, 1]^3."""
    strength = algorithm_strengths(time, energy, cost)
    best = max(strength.values())
    if best <= 0.0:
        return "A5"  # no Table-2 rule applies -> hybrid
    winners = [a for a, s in strength.items() if s == best]
    return "A5" if "A5" in winners else winners[0]
