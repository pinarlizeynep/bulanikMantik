"""Zero-order Takagi-Sugeno-Kang (TSK) inference -- ablation counterpart of the paper's Mamdani FIS.

Same fuzzification and rule antecedents as ``MamdaniFIS`` (AND = min, 243 rules), but every rule
consequent is a constant instead of a fuzzy set, and the output is the firing-strength weighted
average of those constants (no aggregation, no defuzzification integral):

    y = sum_i w_i * z_i / sum_i w_i

The constants are the peaks of the paper's output sets (Fig. 3): Edge = 40, Fog = 100, Cloud = 160,
so a single fully firing rule gives the same crisp value in both systems and the paper's thresholds
(<60 Edge, 60-140 Fog, >140 Cloud) keep their meaning.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .mamdani import MamdaniFIS
from .membership import MF


def core_centre(mf: MF) -> float:
    """Centre of the region where the MF is 1 (the peak of a triangle)."""
    p = mf.params
    return p[1] if mf.kind == "tri" else (p[1] + p[2]) / 2


@dataclass(eq=False)
class SugenoFIS:
    """Zero-order TSK system that shares the compiled rule base of a type-1 ``MamdaniFIS``."""

    base: MamdaniFIS
    constants: dict[str, float] | None = None  # output term -> constant; default: peak of its MF
    _z: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        terms = list(self.base.output.terms)
        if self.constants is None:
            self.constants = {t: core_centre(self.base.output.terms[t]) for t in terms}
        if set(self.constants) != set(terms):
            raise ValueError(f"constants must cover the output terms {terms}")
        self._z = np.array([self.constants[t] for t in terms])[self.base._cons_idx]  # per rule

    @property
    def inputs(self):
        return self.base.inputs

    @property
    def rules(self):
        return self.base.rules

    def firing(self, values: dict[str, float]) -> np.ndarray:
        mem = {v.name: v.fuzzify(values[v.name]) for v in self.base.inputs}
        return self.base.fire(self.base.membership_table(mem))

    def compute(self, values: dict[str, float]) -> float:
        w = self.firing(values)
        total = w.sum()
        return float(np.dot(w, self._z) / total) if total > 0 else float("nan")
