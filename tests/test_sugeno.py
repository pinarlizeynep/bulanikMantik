"""Zero-order Sugeno FIS and the ablation policies (Day 6)."""
import numpy as np
import pytest

from src.env import default_environment
from src.fuzzy.paper_fis import INPUTS, build_fis, build_sugeno_fis, crisp_to_layer
from src.fuzzy.sugeno import SugenoFIS, core_centre
from src.fuzzy.membership import tri, trap
from src.policies import FuzzyOffloadingPolicy, SugenoOffloadingPolicy
from src.scheduler import simulate
from src.workflow import montage


@pytest.fixture(scope="module")
def sugeno():
    return build_sugeno_fis()


def _random_inputs(n, seed=0):
    rng = np.random.default_rng(seed)
    return [{v.name: rng.uniform(v.lo, v.hi) for v in INPUTS} for _ in range(n)]


def test_constants_are_peaks_of_the_output_sets(sugeno):
    assert sugeno.constants == {"Edge": 40.0, "Fog": 100.0, "Cloud": 160.0}
    assert core_centre(tri(0, 40, 80)) == 40.0
    assert core_centre(trap(0, 10, 30, 40)) == 20.0


def test_output_is_weighted_average_of_rule_constants(sugeno):
    z = {"Edge": 40.0, "Fog": 100.0, "Cloud": 160.0}
    for x in _random_inputs(50):
        w = sugeno.firing(x)
        expected = sum(wi * z[r.consequent] for wi, r in zip(w, sugeno.rules)) / w.sum()
        assert sugeno.compute(x) == pytest.approx(expected, abs=1e-9)


def test_output_stays_between_the_extreme_constants(sugeno):
    for x in _random_inputs(300, seed=1):
        assert 40.0 - 1e-9 <= sugeno.compute(x) <= 160.0 + 1e-9


def test_single_fully_firing_rule_gives_its_constant_like_mamdani(sugeno):
    # every input where only one term is non-zero -> exactly one rule fires, with strength 1
    edge = {"jl": 2, "vt": 10, "nl": 30, "bw": 1, "dl": 2}      # all terms push to the edge
    cloud = {"jl": 40, "vt": 200, "nl": 1, "bw": 20, "dl": 50}  # all terms push to the cloud
    assert (sugeno.firing(edge) > 0).sum() == 1
    assert (sugeno.firing(cloud) > 0).sum() == 1
    assert sugeno.compute(edge) == pytest.approx(40.0)
    assert sugeno.compute(cloud) == pytest.approx(160.0)
    mamdani = build_fis()
    assert crisp_to_layer(mamdani.compute(edge)) == crisp_to_layer(sugeno.compute(edge)) == "Edge"


def test_sugeno_agrees_with_mamdani_on_most_inputs(sugeno):
    mamdani = build_fis()
    xs = _random_inputs(500, seed=2)
    agree = np.mean([crisp_to_layer(mamdani.compute(x)) == crisp_to_layer(sugeno.compute(x)) for x in xs])
    assert agree > 0.85  # same rules, different consequent / defuzzification -> mostly the same layer


def test_constants_must_cover_output_terms():
    with pytest.raises(ValueError):
        SugenoFIS(build_fis(), constants={"Edge": 40.0, "Fog": 100.0})


@pytest.mark.parametrize("policy", [SugenoOffloadingPolicy(), FuzzyOffloadingPolicy(defuzz="bisector"),
                                    FuzzyOffloadingPolicy(defuzz="mom"), FuzzyOffloadingPolicy(verify_deadline=False)])
def test_ablation_policies_schedule_every_task(policy):
    wf = montage(40, seed=5)
    res = simulate(wf, default_environment(), policy)
    assert res.n_tasks == len(wf.tasks)
    assert all(d.layer in ("Edge", "Fog", "Cloud") for d in policy.decisions)


def test_sugeno_policy_rejects_a_defuzzification_method():
    with pytest.raises(ValueError):
        SugenoOffloadingPolicy(defuzz="mom")
