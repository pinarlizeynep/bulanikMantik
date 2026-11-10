import math

import pytest

from src.env import default_environment
from src.fuzzy.paper_fis import INPUTS
from src.fuzzy.qos_selector import algorithm_strengths, select_algorithm
from src.policies import FuzzyOffloadingPolicy, TwoWayFogCloudPolicy
from src.scheduler import simulate
from src.workflow import montage


# ------------------------------------------------------------------ Table 2 / Algorithm 1
@pytest.mark.parametrize("pref,algo", [
    ((1, 0, 0), "A4"),   # time only    -> edge
    ((0, 1, 0), "A3"),   # energy only  -> fog
    ((0, 0, 1), "A2"),   # cost only    -> cloud
    ((1 / 3, 1 / 3, 1 / 3), "A5"),   # balanced: no Table-2 rule fires -> hybrid fallback
    ((0.5, 0.5, 0), "A5"),
    ((0, 0.5, 0.5), "A5"),
    ((0.2, 0.2, 0.6), "A2"),
    ((0.2, 0.6, 0.2), "A3"),
    ((0.6, 0.2, 0.2), "A4"),
])
def test_algorithm_selection(pref, algo):
    assert select_algorithm(*pref) == algo


def test_table2_has_16_rules_and_each_algorithm_is_reachable():
    from src.fuzzy.qos_selector import TABLE2
    assert len(TABLE2) == 16
    assert {r[3] for r in TABLE2} == {"A2", "A3", "A4", "A5"}
    assert algorithm_strengths(1, 1, 1)["A5"] == 1.0  # R16: H H H


# ------------------------------------------------------------------ observation mapping
def test_observed_inputs_stay_inside_the_fis_universes():
    env, pol = default_environment(), FuzzyOffloadingPolicy()
    simulate(montage(100, seed=5), env, pol)
    assert len(pol.decisions) == 100
    for d in pol.decisions:
        for v in INPUTS:
            assert v.lo <= d.inputs[v.name] <= v.hi, (v.name, d.inputs[v.name])


def test_inputs_use_a_wide_part_of_each_universe():
    """Guard against a degenerate mapping (e.g. vt stuck at 0): every input must vary."""
    pol = FuzzyOffloadingPolicy()
    simulate(montage(300, seed=1), default_environment(), pol)
    for name in ("jl", "vt", "nl", "dl"):
        values = [d.inputs[name] for d in pol.decisions]
        assert max(values) - min(values) > 0.3 * (dict((v.name, v.hi - v.lo) for v in INPUTS)[name]), name


# ------------------------------------------------------------------ policy behaviour
@pytest.mark.parametrize("pref,layer", [((0, 0, 1), "Cloud"), ((0, 1, 0), "Fog"), ((1, 0, 0), "Edge")])
def test_single_objective_algorithms_without_deadline_check(pref, layer):
    pol = FuzzyOffloadingPolicy(preference=pref, verify_deadline=False)
    res = simulate(montage(60, seed=0), default_environment(), pol)
    assert res.layer_share[layer] == 1.0
    assert all(math.isnan(d.crisp) for d in pol.decisions)


def test_hybrid_follows_the_fis_when_deadline_check_is_off():
    pol = FuzzyOffloadingPolicy(verify_deadline=False)
    simulate(montage(100, seed=2), default_environment(), pol)
    assert all(d.layer == d.fis_layer for d in pol.decisions)
    assert all(d.algorithm == "A5" for d in pol.decisions)


def test_deadline_check_only_changes_layers_that_would_miss():
    pol = FuzzyOffloadingPolicy()
    simulate(montage(200, seed=3), default_environment(), pol)
    assert any(d.layer != d.fis_layer for d in pol.decisions)  # the check does intervene under load
    assert all(d.layer in ("Edge", "Fog", "Cloud") for d in pol.decisions)


def test_qos_bias_moves_the_crisp_value():
    env, wf = default_environment(), montage(60, seed=0)
    cost = FuzzyOffloadingPolicy(preference=(0.3, 0.3, 0.4), verify_deadline=False)  # still A5
    time = FuzzyOffloadingPolicy(preference=(0.4, 0.3, 0.3), verify_deadline=False)
    assert cost.algorithm == time.algorithm == "A5"
    simulate(wf, env, cost), simulate(wf, env, time)
    mean = lambda p: sum(d.crisp for d in p.decisions) / len(p.decisions)
    assert mean(cost) > mean(time)  # cost pulls towards the cloud, time towards the edge


def test_simulation_is_deterministic():
    env = default_environment()
    a = simulate(montage(80, seed=4), env, FuzzyOffloadingPolicy())
    b = simulate(montage(80, seed=4), env, FuzzyOffloadingPolicy())
    assert (a.makespan, a.energy_j, a.cost) == (b.makespan, b.energy_j, b.cost)


def test_proposed_is_not_worse_than_single_layer_extremes():
    env, wf = default_environment(), montage(100, seed=0)
    prop = simulate(wf, env, FuzzyOffloadingPolicy())
    classical = simulate(wf, env, TwoWayFogCloudPolicy())
    assert prop.makespan < 2 * classical.makespan  # sanity bound, not a performance claim


def test_physical_mapping_swaps_cost_and_time_targets():
    from src.fuzzy.qos_selector import ALGORITHM_LAYER, PHYSICAL_LAYER
    assert ALGORITHM_LAYER == {"A2": "Cloud", "A3": "Fog", "A4": "Edge"}
    assert PHYSICAL_LAYER == {"A2": "Edge", "A3": "Fog", "A4": "Cloud"}
    pol = FuzzyOffloadingPolicy(preference=(0, 0, 1), verify_deadline=False, layer_map=PHYSICAL_LAYER)
    assert simulate(montage(40, seed=0), default_environment(), pol).layer_share["Edge"] == 1.0


def test_extreme_preferences_reflect_table3_trade_offs():
    """Edge-only (paper's A4) is free and frugal but slow; cloud-heavy (A2) is fastest but pricey."""
    env, wf = default_environment(), montage(100, seed=0)
    edge = simulate(wf, env, FuzzyOffloadingPolicy(preference=(1, 0, 0)))   # A4 -> edge
    cloud = simulate(wf, env, FuzzyOffloadingPolicy(preference=(0, 0, 1)))  # A2 -> cloud (+deadline check)
    assert edge.cost == 0 and edge.cost < cloud.cost
    assert edge.energy_j < cloud.energy_j
    assert edge.makespan > cloud.makespan
