"""Mamdani fuzzy inference engine (written from scratch, cross-checked against scikit-fuzzy in tests).

Pipeline (paper, Sec. 2.2 and Algorithm 5):
  1. Fuzzification     crisp input  -> membership degree of each linguistic term
  2. Rule evaluation   AND = min over the antecedents  -> firing strength of each rule
  3. Implication       consequent MF clipped at the firing strength (min)
  4. Aggregation       max over all clipped consequents ("maximum aggregation method", MAM)
  5. Defuzzification   centre of gravity (COG), Eq. (21); other methods kept for the ablation study
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .membership import LinguisticVariable


@dataclass(frozen=True)
class Rule:
    """IF (var1 is term1) AND (var2 is term2) ... THEN output is ``consequent``."""

    antecedent: dict[str, str]
    consequent: str
    weight: float = 1.0

    def __str__(self) -> str:
        cond = " AND ".join(f"{v} is {t}" for v, t in self.antecedent.items())
        return f"IF {cond} THEN {self.consequent}"


def defuzzify(x: np.ndarray, mu: np.ndarray, method: str = "centroid") -> float:
    """Turn an aggregated output fuzzy set into one crisp number."""
    if mu.max() <= 0:
        return float("nan")
    if method == "centroid":  # COG: integral(x*mu) / integral(mu)
        return float(np.trapezoid(x * mu, x) / np.trapezoid(mu, x))
    if method == "bisector":  # vertical line that splits the area in half
        area = np.concatenate([[0.0], np.cumsum((mu[1:] + mu[:-1]) / 2 * np.diff(x))])
        return float(np.interp(area[-1] / 2, area, x))
    peak = x[np.isclose(mu, mu.max())]
    if method == "mom":  # mean of maximum
        return float(peak.mean())
    if method == "som":  # smallest of maximum
        return float(peak.min())
    if method == "lom":  # largest of maximum
        return float(peak.max())
    raise ValueError(f"unknown defuzzification method: {method}")


@dataclass
class InferenceResult:
    crisp: float
    memberships: dict[str, dict[str, float]]
    firing: np.ndarray  # firing strength of every rule
    aggregated: np.ndarray  # aggregated output MF sampled on the output universe

    def top_rules(self, rules: list[Rule], k: int = 5) -> list[tuple[float, Rule]]:
        idx = np.argsort(-self.firing)[:k]
        return [(float(self.firing[i]), rules[i]) for i in idx if self.firing[i] > 0]


@dataclass
class MamdaniFIS:
    inputs: list[LinguisticVariable]
    output: LinguisticVariable
    rules: list[Rule]
    defuzz: str = "centroid"
    resolution: int = 1001
    _x: np.ndarray = field(init=False, repr=False)
    _out_mf: dict[str, np.ndarray] = field(init=False, repr=False)

    def __post_init__(self):
        self._x = self.output.universe(self.resolution)
        self._out_mf = {t: mf(self._x) for t, mf in self.output.terms.items()}
        names = {v.name: v for v in self.inputs}
        for r in self.rules:  # validate the rule base once
            for var, term in r.antecedent.items():
                if var not in names or term not in names[var].terms:
                    raise ValueError(f"bad antecedent in rule: {r}")
            if r.consequent not in self.output.terms:
                raise ValueError(f"bad consequent in rule: {r}")

    def infer(self, values: dict[str, float]) -> InferenceResult:
        mem = {v.name: v.fuzzify(values[v.name]) for v in self.inputs}
        firing = np.array(
            [min(mem[var][term] for var, term in r.antecedent.items()) * r.weight for r in self.rules]
        )
        agg = np.zeros_like(self._x)
        for strength, r in zip(firing, self.rules):
            if strength > 0:
                np.maximum(agg, np.minimum(strength, self._out_mf[r.consequent]), out=agg)
        return InferenceResult(defuzzify(self._x, agg, self.defuzz), mem, firing, agg)

    def compute(self, values: dict[str, float]) -> float:
        return self.infer(values).crisp
