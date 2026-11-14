"""The offloading FIS of Soni & Kumar (2024): 5 inputs -> compute-node decision in [0, 200].

Membership functions are read off the paper's Fig. 3 (the paper gives no numeric values).
The paper states that the full rule base has 3^5 = 243 rules but lists only three examples (R1-R3).
We therefore generate all 243 rules with a transparent scoring scheme and check that it reproduces
the three published rules (see ``PAPER_EXAMPLE_RULES`` and tests/test_paper_fis.py).

Run ``python -m src.fuzzy.paper_fis --jl 10 --vt 50 --nl 25 --bw 3 --dl 8`` for an explained decision.
"""
from __future__ import annotations

import argparse
import itertools

from .it2 import IT2MamdaniFIS
from .mamdani import MamdaniFIS, Rule
from .membership import LinguisticVariable, trap, tri
from .sugeno import SugenoFIS

# ---------------------------------------------------------------- linguistic variables (Fig. 3)
JOB_LIST = LinguisticVariable("jl", 0, 40, {
    "Small": trap(0, 0, 8, 16), "Medium": tri(8, 24, 40), "Large": trap(32, 38, 40, 40)})
VM_UTIL = LinguisticVariable("vt", 0, 200, {
    "Poor": trap(0, 0, 40, 80), "Medium": tri(40, 120, 200), "Heavy": trap(160, 180, 200, 200)})
LATENCY = LinguisticVariable("nl", 0, 30, {
    "Low": trap(0, 0, 4, 8), "Average": tri(4, 17, 30), "High": trap(20, 28, 30, 30)})
BANDWIDTH = LinguisticVariable("bw", 0, 20, {
    "Low": trap(0, 0, 4, 8), "Average": tri(4, 12, 20), "High": trap(16, 18, 20, 20)})
DEADLINE = LinguisticVariable("dl", 0, 50, {
    "Low": trap(0, 0, 10, 20), "Medium": tri(10, 30, 50), "High": trap(40, 48, 50, 50)})
DECISION = LinguisticVariable("decision", 0, 200, {
    "Edge": tri(0, 40, 80), "Fog": tri(40, 100, 160), "Cloud": tri(120, 160, 200)})

INPUTS = [JOB_LIST, VM_UTIL, LATENCY, BANDWIDTH, DEADLINE]

# Crisp thresholds on the defuzzified output (paper, Sec. 2.2): <60 Edge, 60-140 Fog, >140 Cloud.
EDGE_FOG_THRESHOLD = 60.0
FOG_CLOUD_THRESHOLD = 140.0

# ---------------------------------------------------------------- rule base (243 rules)
# Each term "pushes" the job towards the cloud (+1), keeps it neutral (0) or pulls it to the edge (-1).
#   jl  more jobs                -> needs more capacity          -> cloud
#   vt  heavier VM utilization   -> local/edge VMs are saturated -> cloud
#   nl  high latency to upper tiers -> keep the job close        -> edge
#   bw  more bandwidth           -> uploading is cheap           -> cloud
#   dl  looser (higher) deadline -> long WAN trip is acceptable  -> cloud
TERM_SCORE = {
    "jl": {"Small": -1, "Medium": 0, "Large": +1},
    "vt": {"Poor": -1, "Medium": 0, "Heavy": +1},
    "nl": {"Low": +1, "Average": 0, "High": -1},
    "bw": {"Low": -1, "Average": 0, "High": +1},
    "dl": {"Low": -1, "Medium": 0, "High": +1},
}
# Total score s in [-5, 5]:  s <= -2 -> Edge,  -1 <= s <= 1 -> Fog,  s >= 2 -> Cloud.
SCORE_EDGE_MAX = -2
SCORE_CLOUD_MIN = 2

# The three example rules printed in the paper (Sec. 2.2, "Rule Base").
PAPER_EXAMPLE_RULES = [
    Rule({"jl": "Small", "vt": "Poor", "nl": "High", "dl": "Low", "bw": "Low"}, "Edge"),     # R1
    Rule({"jl": "Small", "vt": "Heavy", "nl": "Low", "dl": "Medium", "bw": "Low"}, "Fog"),   # R2
    Rule({"jl": "Large", "vt": "Heavy", "nl": "Average", "dl": "High", "bw": "High"}, "Cloud"),  # R3
]


def score_to_layer(score: int) -> str:
    if score <= SCORE_EDGE_MAX:
        return "Edge"
    if score >= SCORE_CLOUD_MIN:
        return "Cloud"
    return "Fog"


def build_rules() -> list[Rule]:
    names = [v.name for v in INPUTS]
    rules = []
    for combo in itertools.product(*(list(v.terms) for v in INPUTS)):
        antecedent = dict(zip(names, combo))
        score = sum(TERM_SCORE[n][t] for n, t in antecedent.items())
        rules.append(Rule(antecedent, score_to_layer(score)))
    return rules


def build_fis(defuzz: str = "centroid", resolution: int = 1001) -> MamdaniFIS:
    return MamdaniFIS(INPUTS, DECISION, build_rules(), defuzz=defuzz, resolution=resolution)


def build_it2_fis(fou: float = 0.05, output_fou: float = 0.0, type_reduction: str = "km",
                  resolution: int = 1001) -> IT2MamdaniFIS:
    """Interval type-2 version of the paper's FIS (extension, see ``it2.py``): same MFs as principal
    MFs, same 243 rules, KM type reduction; ``fou = 0`` reproduces ``build_fis()``."""
    return IT2MamdaniFIS(build_fis("centroid", resolution), fou, output_fou, type_reduction)


def build_sugeno_fis() -> SugenoFIS:
    """Zero-order Sugeno version of the paper's FIS (ablation, see ``sugeno.py``): same MFs and
    243 rule antecedents, constant consequents Edge = 40, Fog = 100, Cloud = 160."""
    return SugenoFIS(build_fis("centroid"))


def crisp_to_layer(c: float) -> str:
    if c != c:  # NaN: no rule fired, there is no decision to map
        raise ValueError("no rule fired (crisp output is NaN)")
    if c < EDGE_FOG_THRESHOLD:
        return "Edge"
    if c > FOG_CLOUD_THRESHOLD:
        return "Cloud"
    return "Fog"


def explain(fis: MamdaniFIS, values: dict[str, float], k: int = 5) -> str:
    """Human-readable trace of one decision (used in the live demo)."""
    res = fis.infer(values)
    lines = ["Fuzzification:"]
    for var in fis.inputs:
        degs = ", ".join(f"{t}={m:.2f}" for t, m in res.memberships[var.name].items() if m > 0)
        lines.append(f"  {var.name:>2} = {values[var.name]:>6.2f}  ->  {degs}")
    lines.append(f"Active rules: {(res.firing > 0).sum()} / {len(fis.rules)}   (top {k}):")
    for s, r in res.top_rules(fis.rules, k):
        lines.append(f"  [{s:.2f}] {r}")
    lines.append(f"Defuzzified ({fis.defuzz}) = {res.crisp:.2f}  ->  offload to {crisp_to_layer(res.crisp)}")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Explain one offloading decision of the paper's FIS.")
    p.add_argument("--jl", type=float, default=10, help="job list size [0, 40]")
    p.add_argument("--vt", type=float, default=50, help="VM utilization [0, 200]")
    p.add_argument("--nl", type=float, default=25, help="network latency [0, 30]")
    p.add_argument("--bw", type=float, default=3, help="bandwidth [0, 20]")
    p.add_argument("--dl", type=float, default=8, help="deadline [0, 50]")
    p.add_argument("--defuzz", default="centroid", choices=["centroid", "bisector", "mom", "som", "lom"])
    p.add_argument("--fou", type=float, default=None,
                   help="also show the interval type-2 decision with this FOU half-width (fraction of each universe)")
    p.add_argument("--sugeno", action="store_true", help="also show the zero-order Sugeno decision")
    a = p.parse_args()
    values = {"jl": a.jl, "vt": a.vt, "nl": a.nl, "bw": a.bw, "dl": a.dl}
    print(explain(build_fis(a.defuzz), values))
    if a.sugeno:
        y = build_sugeno_fis().compute(values)
        print(f"Zero-order Sugeno (weighted average of 40 / 100 / 160) = {y:.2f}  ->  offload to {crisp_to_layer(y)}")
    if a.fou is not None:
        res = build_it2_fis(a.fou).infer(values)
        yl, yr = res.interval
        print(f"Interval type-2 (FOU {a.fou:g}): type-reduced set [{yl:.2f}, {yr:.2f}] (Karnik-Mendel)"
              f"  ->  {res.crisp:.2f}  ->  offload to {crisp_to_layer(res.crisp)}")


if __name__ == "__main__":
    main()
