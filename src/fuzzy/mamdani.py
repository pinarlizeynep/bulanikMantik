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

    def __hash__(self) -> int:  # dict field is unhashable; hash its (ordered) items instead
        return hash((tuple(self.antecedent.items()), self.consequent, self.weight))

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


@dataclass(eq=False)  # numpy fields make field-wise == ambiguous; compare by identity
class MamdaniFIS:
    inputs: list[LinguisticVariable]
    output: LinguisticVariable
    rules: list[Rule]
    defuzz: str = "centroid"
    resolution: int = 1001
    _x: np.ndarray = field(init=False, repr=False)
    _out_mf: np.ndarray = field(init=False, repr=False)  # (n_terms, resolution)
    _term_idx: np.ndarray = field(init=False, repr=False)  # (n_rules, n_inputs) term index per rule
    _cons_idx: np.ndarray = field(init=False, repr=False)  # (n_rules,) consequent index per rule
    _weights: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self._x = self.output.universe(self.resolution)
        out_terms = list(self.output.terms)
        self._out_mf = np.array([self.output.terms[t](self._x) for t in out_terms])
        term_pos = [{t: j for j, t in enumerate(v.terms)} for v in self.inputs]
        names = [v.name for v in self.inputs]
        # Compile the rule base once into index arrays.  A rule that does not mention an input
        # gets index -1, which points at an always-1 "don't care" column (min ignores it).
        self._term_idx = np.full((len(self.rules), len(self.inputs)), -1, dtype=int)
        cons = []
        for i, r in enumerate(self.rules):
            for var, term in r.antecedent.items():
                if var not in names or term not in term_pos[names.index(var)]:
                    raise ValueError(f"bad antecedent in rule: {r}")
                self._term_idx[i, names.index(var)] = term_pos[names.index(var)][term]
            if r.consequent not in self.output.terms:
                raise ValueError(f"bad consequent in rule: {r}")
            cons.append(out_terms.index(r.consequent))
        self._cons_idx = np.array(cons, dtype=int)
        self._weights = np.array([r.weight for r in self.rules], dtype=float)

    def infer(self, values: dict[str, float]) -> InferenceResult:
        mem = {v.name: v.fuzzify(values[v.name]) for v in self.inputs}
        width = max(len(v.terms) for v in self.inputs) + 1
        table = np.ones((len(self.inputs), width))  # last column = 1 ("don't care")
        for k, v in enumerate(self.inputs):
            table[k, : len(v.terms)] = list(mem[v.name].values())
        cols = np.arange(len(self.inputs))
        firing = table[cols, self._term_idx].min(axis=1) * self._weights  # AND = min
        # Implication (min) + aggregation (max): rules with the same consequent clip the same MF,
        # so only the strongest of them matters -> one clip per output term instead of per rule.
        strength = np.zeros(len(self._out_mf))
        np.maximum.at(strength, self._cons_idx, firing)
        agg = np.minimum(strength[:, None], self._out_mf).max(axis=0)
        return InferenceResult(defuzzify(self._x, agg, self.defuzz), mem, firing, agg)

    def compute(self, values: dict[str, float]) -> float:
        return self.infer(values).crisp
