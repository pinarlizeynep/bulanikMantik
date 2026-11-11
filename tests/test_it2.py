"""Interval type-2 extension: MFs, KM type reduction, IT2 FIS, IT2 policy and the noise model."""
import numpy as np
import pytest

from src.env import default_environment
from src.fuzzy.it2 import IT2MF, IT2MamdaniFIS, IT2Variable, centroid_interval
from src.fuzzy.mamdani import defuzzify
from src.fuzzy.membership import trap, tri
from src.fuzzy.paper_fis import DECISION, INPUTS, build_fis, build_it2_fis, crisp_to_layer
from src.policies import FuzzyOffloadingPolicy, InputNoise, IT2OffloadingPolicy
from src.scheduler import simulate
from src.workflow import montage


def _random_inputs(rng):
    return {v.name: rng.uniform(v.lo, v.hi) for v in INPUTS}


# ------------------------------------------------------------------ IT2 membership functions
@pytest.mark.parametrize("mf", [tri(10, 30, 50), trap(0, 0, 10, 20), trap(40, 48, 50, 50), tri(0, 40, 80)])
def test_lower_below_principal_below_upper(mf):
    m = IT2MF(mf, 4.0, 0.0, 50.0)
    x = np.linspace(0, 50, 501)
    assert np.all(m.lower(x) <= mf(x) + 1e-12)
    assert np.all(mf(x) <= m.upper(x) + 1e-12)
    assert np.all((m.lower(x) >= 0) & (m.upper(x) <= 1))


@pytest.mark.parametrize("mf", [tri(10, 30, 50), trap(0, 0, 10, 20), trap(40, 48, 50, 50)])
def test_closed_form_bounds_match_brute_force_window(mf):
    """UMF / LMF = max / min of the type-1 MF over [x - delta, x + delta] clipped to the universe."""
    delta, lo, hi = 3.0, 0.0, 50.0
    m = IT2MF(mf, delta, lo, hi)
    for x in np.linspace(lo, hi, 101):
        window = np.linspace(max(lo, x - delta), min(hi, x + delta), 2001)
        vals = mf(window)
        lower, upper = m.bounds(x)
        assert lower == pytest.approx(vals.min(), abs=1e-3)
        assert upper == pytest.approx(vals.max(), abs=1e-3)
        assert (lower, upper) == pytest.approx((m.lower(x), m.upper(x)))


def test_zero_fou_is_type1():
    for v in INPUTS:
        it2 = IT2Variable(v, 0.0)
        for x in np.linspace(v.lo, v.hi, 57):
            lower, upper = it2.fuzzify(x)
            assert lower == upper == v.fuzzify(x)


def test_shoulders_keep_full_membership_at_the_universe_border():
    # jl: Small = trap(0, 0, 8, 16), Large = trap(32, 38, 40, 40).  delta = 0.025 * 40 = 1 is narrower
    # than both cores, so the window clipped at the border stays inside the core -> LMF = 1.
    it2 = IT2Variable(INPUTS[0], 0.025)
    assert it2.fuzzify(0)[0]["Small"] == 1.0
    assert it2.fuzzify(40)[0]["Large"] == 1.0
    # A window wider than the core lowers the LMF even at the border (36 -> 4/6), as it should.
    assert IT2Variable(INPUTS[0], 0.1).fuzzify(40)[0]["Large"] == pytest.approx(4 / 6)


def test_fou_grows_with_its_parameter():
    x = np.linspace(0, 200, 401)
    widths = [np.trapezoid(IT2MF(tri(40, 120, 200), f * 200, 0, 200).upper(x)
                           - IT2MF(tri(40, 120, 200), f * 200, 0, 200).lower(x), x) for f in (0, 0.05, 0.1)]
    assert widths[0] == 0 and widths[0] < widths[1] < widths[2]


def test_negative_fou_rejected():
    with pytest.raises(ValueError):
        IT2Variable(INPUTS[0], -0.1)


# ------------------------------------------------------------------ type reduction
def test_type1_set_has_degenerate_centroid_equal_to_cog():
    x = np.linspace(0, 200, 1001)
    mu = np.maximum(np.minimum(0.4, DECISION.terms["Edge"](x)), np.minimum(0.7, DECISION.terms["Fog"](x)))
    yl, yr = centroid_interval(x, mu, mu)
    assert yl == pytest.approx(yr) and yl == pytest.approx(defuzzify(x, mu, "centroid"))


def test_km_matches_exhaustive_switch_point_search():
    x = np.linspace(0, 200, 1001)
    rng = np.random.default_rng(3)
    for _ in range(30):
        upper = np.clip(rng.uniform(0, 1, x.size), 0, 1)
        lower = upper * rng.uniform(0, 1, x.size)
        km = centroid_interval(x, lower, upper, "km")
        ex = centroid_interval(x, lower, upper, "exhaustive")
        assert km == pytest.approx(ex, abs=1e-9)


def test_centroid_interval_bounds_any_embedded_type1_centroid():
    """Every embedded type-1 set lower <= mu <= upper has its centroid inside [y_l, y_r]."""
    x = np.linspace(0, 100, 501)
    upper, lower = np.minimum(1, 1.2 * np.exp(-((x - 40) / 15) ** 2)), 0.5 * np.exp(-((x - 45) / 8) ** 2)
    yl, yr = centroid_interval(x, lower, upper)
    rng = np.random.default_rng(0)
    for _ in range(200):
        mu = lower + rng.uniform(0, 1, x.size) * (upper - lower)
        assert yl - 1e-9 <= defuzzify(x, mu, "centroid") <= yr + 1e-9


def test_symmetric_fou_gives_symmetric_interval():
    x = np.linspace(0, 100, 1001)
    m = IT2MF(tri(20, 50, 80), 5.0, 0, 100)
    yl, yr = centroid_interval(x, m.lower(x), m.upper(x))
    assert (yl + yr) / 2 == pytest.approx(50, abs=1e-6) and yl < 50 < yr


def test_empty_set_gives_nan():
    x = np.linspace(0, 1, 11)
    assert all(np.isnan(centroid_interval(x, np.zeros(11), np.zeros(11))))


def test_unknown_type_reduction_rejected():
    with pytest.raises(ValueError):
        IT2MamdaniFIS(build_fis(), 0.05, type_reduction="nie-tan")
    with pytest.raises(ValueError):
        IT2MamdaniFIS(build_fis("mom"), 0.05)


# ------------------------------------------------------------------ IT2 FIS
def test_it2_with_zero_fou_reproduces_type1_crisp_output():
    t1, it2 = build_fis(), build_it2_fis(0.0)
    rng = np.random.default_rng(1)
    for _ in range(200):
        vals = _random_inputs(rng)
        res = it2.infer(vals)
        assert res.crisp == pytest.approx(t1.compute(vals), abs=1e-9)
        assert res.interval[0] == pytest.approx(res.interval[1], abs=1e-9)


def test_it2_firing_interval_contains_type1_firing():
    t1, it2 = build_fis(), build_it2_fis(0.08)
    rng = np.random.default_rng(2)
    for _ in range(50):
        vals = _random_inputs(rng)
        f = t1.infer(vals).firing
        r = it2.infer(vals)
        assert np.all(r.firing_lower <= f + 1e-12) and np.all(f <= r.firing_upper + 1e-12)
        assert np.all(r.aggregated_lower <= r.aggregated_upper + 1e-12)
        assert r.interval[0] <= r.crisp <= r.interval[1]


def test_it2_type_reduced_interval_widens_with_fou():
    vals = {"jl": 13, "vt": 70, "nl": 6, "bw": 7, "dl": 18}
    widths = [np.diff(build_it2_fis(f).infer(vals).interval)[0] for f in (0.0, 0.03, 0.06, 0.12)]
    assert widths[0] == pytest.approx(0, abs=1e-9)
    assert all(a < b for a, b in zip(widths, widths[1:]))


@pytest.mark.parametrize("values,layer", [
    ({"jl": 2, "vt": 10, "nl": 29, "bw": 1, "dl": 3}, "Edge"),
    ({"jl": 20, "vt": 100, "nl": 15, "bw": 10, "dl": 25}, "Fog"),
    ({"jl": 39, "vt": 195, "nl": 1, "bw": 19, "dl": 49}, "Cloud"),
])
def test_it2_keeps_clear_cut_decisions(values, layer):
    assert crisp_to_layer(build_it2_fis(0.05).compute(values)) == layer


def test_output_fou_is_supported():
    vals = {"jl": 13, "vt": 70, "nl": 6, "bw": 7, "dl": 18}
    a, b = build_it2_fis(0.05).infer(vals), build_it2_fis(0.05, output_fou=0.05).infer(vals)
    assert np.diff(b.interval)[0] > np.diff(a.interval)[0]


# ------------------------------------------------------------------ policies and noise
def test_it2_policy_with_zero_fou_equals_type1_policy():
    env, wf = default_environment(), montage(60, seed=4)
    a, b = simulate(wf, env, FuzzyOffloadingPolicy()), simulate(wf, env, IT2OffloadingPolicy(fou=0.0))
    assert [(x.task, x.node, x.start) for x in a.assignments] == [(x.task, x.node, x.start) for x in b.assignments]
    assert (a.makespan, a.energy_j, a.cost) == pytest.approx((b.makespan, b.energy_j, b.cost))


def test_it2_policy_inherits_everything_but_the_fis():
    pol = IT2OffloadingPolicy(fou=0.07)
    assert isinstance(pol, FuzzyOffloadingPolicy) and isinstance(pol.fis, IT2MamdaniFIS)
    assert pol.fis.fou == 0.07 and pol.algorithm == "A5" and pol.crisp_shift == 0.0
    assert IT2OffloadingPolicy(preference=(0, 0, 1)).algorithm == "A2"
    with pytest.raises(ValueError):
        IT2OffloadingPolicy(defuzz="mom")


def test_noise_is_reproducible_clipped_and_only_on_selected_variables():
    vals = {"jl": 39.0, "vt": 1.0, "nl": 15.0, "bw": 10.0, "dl": 25.0}
    noise = InputNoise(0.3, ("nl", "vt", "jl"), seed=7)
    a, b = noise(5, vals), noise(5, vals)
    assert a == b and a != noise(6, vals)
    assert a["bw"] == vals["bw"] and a["dl"] == vals["dl"]
    for v in INPUTS:
        assert v.lo <= a[v.name] <= v.hi
    assert InputNoise(0.0)(5, vals) == vals


def test_noise_draw_does_not_depend_on_the_variable_set():
    vals = {"jl": 20.0, "vt": 100.0, "nl": 15.0, "bw": 10.0, "dl": 25.0}
    assert InputNoise(0.05, ("nl",), seed=1)(3, vals)["nl"] == InputNoise(0.05, ("nl", "bw"), seed=1)(3, vals)["nl"]


def test_bad_noise_spec_rejected():
    with pytest.raises(ValueError):
        InputNoise(0.1, ("latency",))
    with pytest.raises(ValueError):
        InputNoise(-0.1)


def test_noisy_policy_logs_the_clean_decision_at_the_same_state():
    pol = FuzzyOffloadingPolicy(noise=InputNoise(0.2, seed=3))
    simulate(montage(100, seed=3), default_environment(), pol)
    fis = build_fis()
    flips = 0
    for d in pol.decisions:
        assert d.clean_layer in ("Edge", "Fog", "Cloud") and d.clean_crisp == d.clean_crisp
        assert d.crisp == pytest.approx(fis.compute(d.inputs))  # crisp_shift is 0 for the balanced user
        flips += d.clean_layer != d.fis_layer
    assert 0 < flips < len(pol.decisions)


def test_no_noise_leaves_the_type1_results_unchanged():
    env, wf = default_environment(), montage(100, seed=8)
    a = simulate(wf, env, FuzzyOffloadingPolicy())
    b = simulate(wf, env, FuzzyOffloadingPolicy(noise=InputNoise(0.0)))
    assert (a.makespan, a.energy_j, a.cost) == (b.makespan, b.energy_j, b.cost)
