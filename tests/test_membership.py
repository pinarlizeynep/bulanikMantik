import numpy as np
import pytest

from src.fuzzy import LinguisticVariable, defuzzify, trap, trapmf, tri, trimf


def test_trimf_key_points():
    y = trimf([0, 5, 10, 15, 20], 0, 10, 20)
    assert np.allclose(y, [0, 0.5, 1, 0.5, 0])


def test_trapmf_shoulders():
    # left shoulder: 1 on [0, 8], 0 from 16
    assert np.allclose(trapmf([0, 8, 12, 16, 20], 0, 0, 8, 16), [1, 1, 0.5, 0, 0])
    # right shoulder: 0 until 32, 1 from 38
    assert np.allclose(trapmf([30, 35, 38, 40], 32, 38, 40, 40), [0, 0.5, 1, 1])


def test_fuzzify_clips_out_of_range_input():
    v = LinguisticVariable("x", 0, 10, {"Low": trap(0, 0, 2, 5), "High": tri(5, 10, 10)})
    assert v.fuzzify(-3) == v.fuzzify(0)
    assert v.fuzzify(99)["High"] == 1.0


@pytest.mark.parametrize("method,expected", [("centroid", 50), ("bisector", 50), ("mom", 50)])
def test_defuzzify_symmetric_triangle(method, expected):
    x = np.linspace(0, 100, 1001)
    assert defuzzify(x, trimf(x, 20, 50, 80), method) == pytest.approx(expected, abs=0.1)


def test_defuzzify_som_lom_on_plateau():
    x = np.linspace(0, 100, 1001)
    mu = np.minimum(0.5, trimf(x, 0, 50, 100))  # plateau 0.5 on [25, 75]
    assert defuzzify(x, mu, "som") == pytest.approx(25, abs=0.2)
    assert defuzzify(x, mu, "lom") == pytest.approx(75, abs=0.2)


def test_scalar_degree_matches_vectorized():
    xs = np.linspace(-5, 55, 241)
    for mf in (tri(10, 30, 50), trap(0, 0, 10, 20), trap(40, 48, 50, 50), tri(0, 0, 20)):
        assert np.allclose([mf.degree(x) for x in xs], mf(xs))
