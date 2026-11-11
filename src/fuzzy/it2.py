"""Interval type-2 (IT2) Mamdani inference -- the project's extension of the paper's type-1 FIS.

Why: the paper motivates fuzzy logic with "uncertain, rapidly changing conditions", but its MFs are
type-1: a measured latency of 7.9 ms is "Low" to exactly one degree.  In an IT2 fuzzy set the degree
itself is an interval [lower, upper]; the band between the lower MF (LMF) and the upper MF (UMF) is
the *footprint of uncertainty* (FOU).

FOU construction ("uncertain position", one parameter).  Every type-1 MF of the paper (Fig. 3) is the
principal MF.  If a crisp reading x is only known up to +-delta, its membership can be anything that
the type-1 MF takes on [x - delta, x + delta] (intersected with the universe):

    UMF(x) = max { mu(z) : z in [x - delta, x + delta] }      (MF widened horizontally by delta)
    LMF(x) = min { mu(z) : z in [x - delta, x + delta] }      (MF narrowed by delta)

with delta = fou * (universe width).  For triangles and trapezoids (unimodal MFs) the max is 1 when
the window touches the core and the min is attained at one of the two window ends, so both are
evaluated in closed form.  fou = 0 gives LMF = UMF = the paper's type-1 MF.  Shoulders keep their
value 1 at the border of the universe because the window is clipped to the universe.

Inference (same operators as the paper, applied to both bounds):
  1. Fuzzification     x -> [LMF(x), UMF(x)] for every term
  2. Firing interval   [f_lower, f_upper] = [min of the LMF degrees, min of the UMF degrees]   (AND = min)
  3. Implication / aggregation (min / max) on both bounds -> an IT2 output set with LMF_out <= UMF_out
  4. Type reduction    centroid of the IT2 set = interval [y_l, y_r], Karnik-Mendel (KM) iterations
  5. Defuzzification   y = (y_l + y_r) / 2, then the paper's thresholds (<60 Edge, 60-140 Fog, >140 Cloud)

The centroid uses the same trapezoidal quadrature as the type-1 COG of ``mamdani.defuzzify``, so with
fou = 0 the IT2 system returns the type-1 crisp output up to floating-point rounding (tested).
The compiled rule base (243 rules) is taken from the type-1 ``MamdaniFIS``; nothing is duplicated.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .mamdani import MamdaniFIS
from .membership import MF, LinguisticVariable


# ---------------------------------------------------------------- IT2 membership functions
@dataclass(frozen=True)
class IT2MF:
    """Lower / upper MF obtained by moving the principal type-1 MF ``mf`` by up to +-``delta``."""

    mf: MF
    delta: float
    lo: float  # universe of the variable (the uncertainty window is clipped to it)
    hi: float

    def __post_init__(self):
        if self.delta < 0:
            raise ValueError("delta must be >= 0")

    @property
    def core(self) -> tuple[float, float]:
        p = self.mf.params
        return (p[1], p[1]) if self.mf.kind == "tri" else (p[1], p[2])

    def bounds(self, x: float) -> tuple[float, float]:
        """(LMF(x), UMF(x)) for a scalar x inside the universe."""
        left, right = max(self.lo, x - self.delta), min(self.hi, x + self.delta)
        mu_l, mu_r = self.mf.degree(left), self.mf.degree(right)
        c0, c1 = self.core
        if right < c0:
            upper = mu_r
        elif left > c1:
            upper = mu_l
        else:
            upper = 1.0  # the window touches the core
        return min(mu_l, mu_r), upper

    def lower(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        left, right = np.maximum(self.lo, x - self.delta), np.minimum(self.hi, x + self.delta)
        return np.minimum(self.mf(left), self.mf(right))

    def upper(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        left, right = np.maximum(self.lo, x - self.delta), np.minimum(self.hi, x + self.delta)
        c0, c1 = self.core
        return np.where(right < c0, self.mf(right), np.where(left > c1, self.mf(left), 1.0))


@dataclass
class IT2Variable:
    """A type-1 linguistic variable whose terms get a FOU of half-width ``fou * (hi - lo)``."""

    base: LinguisticVariable
    fou: float
    terms: dict[str, IT2MF] = field(init=False)

    def __post_init__(self):
        if self.fou < 0:
            raise ValueError("fou must be >= 0")
        delta = self.fou * (self.base.hi - self.base.lo)
        self.terms = {t: IT2MF(mf, delta, self.base.lo, self.base.hi) for t, mf in self.base.terms.items()}

    @property
    def name(self) -> str:
        return self.base.name

    def fuzzify(self, x: float) -> tuple[dict[str, float], dict[str, float]]:
        """Crisp value -> (lower degrees, upper degrees) of every term (input clipped to the universe)."""
        x = self.base.clip(x)
        lower, upper = {}, {}
        for t, m in self.terms.items():
            lower[t], upper[t] = m.bounds(x)
        return lower, upper


# ---------------------------------------------------------------- type reduction
def _quadrature_weights(x: np.ndarray) -> np.ndarray:
    """Trapezoidal-rule weights, so that sum(w * f) == np.trapezoid(f, x)."""
    dx = np.diff(x)
    w = np.zeros_like(x)
    w[:-1] += dx / 2
    w[1:] += dx / 2
    return w


def _km_endpoint(x: np.ndarray, lo: np.ndarray, up: np.ndarray, left: bool, max_iter: int = 100) -> float:
    """One end of the centroid interval with the Karnik-Mendel iterations.

    y_l: points left of the switch point take the upper weight, points right of it the lower weight
    (pulls the centroid left as far as possible); y_r is the mirror case.  ``x`` must be sorted.
    """
    theta = (lo + up) / 2
    y = float(np.dot(x, theta) / theta.sum())
    k = int(np.searchsorted(x, y, side="right"))
    for _ in range(max_iter):
        theta = np.concatenate([up[:k], lo[k:]]) if left else np.concatenate([lo[:k], up[k:]])
        y = float(np.dot(x, theta) / theta.sum())
        k_new = int(np.searchsorted(x, y, side="right"))
        if k_new == k:  # switch point did not move -> converged (KM is exact and terminates)
            return y
        k = k_new
    raise RuntimeError("Karnik-Mendel iterations did not converge")


def _exhaustive_endpoints(x: np.ndarray, lo: np.ndarray, up: np.ndarray) -> tuple[float, float]:
    """Both centroid ends by trying every switch point (O(N) with cumulative sums); reference for KM."""
    zero = np.zeros(1)
    cu_xu, cu_u = np.concatenate([zero, np.cumsum(x * up)]), np.concatenate([zero, np.cumsum(up)])
    cu_xl, cu_l = np.concatenate([zero, np.cumsum(x * lo)]), np.concatenate([zero, np.cumsum(lo)])
    tot_xl, tot_l, tot_xu, tot_u = cu_xl[-1], cu_l[-1], cu_xu[-1], cu_u[-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        left = (cu_xu + (tot_xl - cu_xl)) / (cu_u + (tot_l - cu_l))   # upper for i < k, lower after
        right = (cu_xl + (tot_xu - cu_xu)) / (cu_l + (tot_u - cu_u))  # lower for i < k, upper after
    return float(np.nanmin(left)), float(np.nanmax(right))


def centroid_interval(x: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                      method: str = "km") -> tuple[float, float]:
    """Centroid [y_l, y_r] of an IT2 set sampled on the sorted grid ``x`` (trapezoidal quadrature).

    Returns (nan, nan) if the upper MF is identically zero (no rule fired).
    """
    w = _quadrature_weights(x)
    lo, up = lower * w, upper * w  # KM works with any positive point weights
    if up.sum() <= 0:
        return float("nan"), float("nan")
    if method == "km":
        with np.errstate(invalid="raise", divide="raise"):
            try:
                return _km_endpoint(x, lo, up, left=True), _km_endpoint(x, lo, up, left=False)
            except FloatingPointError:  # degenerate set (all mass on one grid point): enumerate
                return _exhaustive_endpoints(x, lo, up)
    if method == "exhaustive":
        return _exhaustive_endpoints(x, lo, up)
    raise ValueError(f"unknown type-reduction method: {method}")


# ---------------------------------------------------------------- IT2 Mamdani FIS
@dataclass
class IT2Result:
    crisp: float
    interval: tuple[float, float]  # type-reduced set [y_l, y_r]
    lower_memberships: dict[str, dict[str, float]]
    upper_memberships: dict[str, dict[str, float]]
    firing_lower: np.ndarray
    firing_upper: np.ndarray
    aggregated_lower: np.ndarray
    aggregated_upper: np.ndarray


@dataclass(eq=False)
class IT2MamdaniFIS:
    """Interval type-2 version of a type-1 ``MamdaniFIS`` (same variables, rules and operators).

    ``fou``         FOU half-width of every input term, as a fraction of the variable's universe
    ``output_fou``  same for the consequent sets (default 0: type-1 consequents, so that the crisp
                    output keeps the paper's meaning and thresholds)
    """

    base: MamdaniFIS
    fou: float = 0.05
    output_fou: float = 0.0
    type_reduction: str = "km"
    inputs: list[IT2Variable] = field(init=False, repr=False)
    _out_lower: np.ndarray = field(init=False, repr=False)
    _out_upper: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        if self.base.defuzz != "centroid":
            raise ValueError("the IT2 system type-reduces with the centroid; build the base FIS with defuzz='centroid'")
        if self.type_reduction not in ("km", "exhaustive"):
            raise ValueError(f"unknown type-reduction method: {self.type_reduction}")
        self.inputs = [IT2Variable(v, self.fou) for v in self.base.inputs]
        out = IT2Variable(self.base.output, self.output_fou)
        x = self.base._x
        self._out_lower = np.array([m.lower(x) for m in out.terms.values()])
        self._out_upper = np.array([m.upper(x) for m in out.terms.values()])

    @property
    def rules(self):
        return self.base.rules

    def infer(self, values: dict[str, float]) -> IT2Result:
        lower, upper = {}, {}
        for v in self.inputs:
            lower[v.name], upper[v.name] = v.fuzzify(values[v.name])
        f_lo = self.base.fire(self.base.membership_table(lower))
        f_up = self.base.fire(self.base.membership_table(upper))
        agg_lo = np.minimum(self.base.term_strength(f_lo)[:, None], self._out_lower).max(axis=0)
        agg_up = np.minimum(self.base.term_strength(f_up)[:, None], self._out_upper).max(axis=0)
        yl, yr = centroid_interval(self.base._x, agg_lo, agg_up, self.type_reduction)
        return IT2Result((yl + yr) / 2, (yl, yr), lower, upper, f_lo, f_up, agg_lo, agg_up)

    def compute(self, values: dict[str, float]) -> float:
        return self.infer(values).crisp
