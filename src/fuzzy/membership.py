"""Membership functions and linguistic variables.

The reference paper (Soni & Kumar, 2024, Fig. 3 and Algorithm 5) uses triangular MFs and
left/right open-shoulder MFs. A shoulder is a trapezoid whose first (or last) two corners coincide,
e.g. ``trap(0, 0, 8, 16)`` is 1 on [0, 8] and falls linearly to 0 at 16.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


def trimf(x, a: float, b: float, c: float):
    """Triangular MF: 0 at ``a``, peak 1 at ``b``, 0 at ``c``  (paper: MF1(x; p, q, r))."""
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    if b > a:
        rising = (x > a) & (x < b)
        y[rising] = (x[rising] - a) / (b - a)
    if c > b:
        falling = (x > b) & (x < c)
        y[falling] = (c - x[falling]) / (c - b)
    y[x == b] = 1.0
    return y


def trapmf(x, a: float, b: float, c: float, d: float):
    """Trapezoidal MF: 0 at ``a``, 1 on [b, c], 0 at ``d``  (paper: MF2(x; p, q, r, s)).

    ``a == b`` gives a left shoulder, ``c == d`` a right shoulder.
    """
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    y[(x >= b) & (x <= c)] = 1.0
    if b > a:
        rising = (x > a) & (x < b)
        y[rising] = (x[rising] - a) / (b - a)
    if d > c:
        falling = (x > c) & (x < d)
        y[falling] = (d - x[falling]) / (d - c)
    return y


@dataclass(frozen=True)
class MF:
    """A named membership function: ``kind`` is 'tri' or 'trap'."""

    kind: str
    params: tuple[float, ...]

    def __post_init__(self):
        if self.kind not in ("tri", "trap"):
            raise ValueError(f"unknown MF kind: {self.kind}")

    def __call__(self, x):
        """Vectorized evaluation over an array (used for plots and the output universe)."""
        return trimf(x, *self.params) if self.kind == "tri" else trapmf(x, *self.params)

    def degree(self, x: float) -> float:
        """Scalar evaluation without numpy allocation (hot path of fuzzification)."""
        if self.kind == "tri":
            a, b, c = self.params
            if x == b:
                return 1.0
            if a < x < b:
                return (x - a) / (b - a)
            if b < x < c:
                return (c - x) / (c - b)
            return 0.0
        a, b, c, d = self.params
        if b <= x <= c:
            return 1.0
        if a < x < b:
            return (x - a) / (b - a)
        if c < x < d:
            return (d - x) / (d - c)
        return 0.0


def tri(a, b, c) -> MF:
    return MF("tri", (float(a), float(b), float(c)))


def trap(a, b, c, d) -> MF:
    return MF("trap", (float(a), float(b), float(c), float(d)))


@dataclass
class LinguisticVariable:
    """A fuzzy variable: a name, a numeric universe [lo, hi] and named terms (fuzzy sets)."""

    name: str
    lo: float
    hi: float
    terms: dict[str, MF] = field(default_factory=dict)

    def clip(self, x: float) -> float:
        x = float(x)
        if not math.isfinite(x):
            raise ValueError(f"{self.name}: input must be a finite number, got {x}")
        return min(max(x, self.lo), self.hi)

    def fuzzify(self, x: float) -> dict[str, float]:
        """Crisp value -> membership degree of every term (inputs are clipped to the universe)."""
        x = self.clip(x)
        return {term: mf.degree(x) for term, mf in self.terms.items()}

    def universe(self, n: int = 1001) -> np.ndarray:
        return np.linspace(self.lo, self.hi, n)
