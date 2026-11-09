from collections import Counter

import numpy as np
import pytest
import skfuzzy as fuzz
from skfuzzy import control as ctrl

from src.fuzzy.paper_fis import (
    DECISION, INPUTS, PAPER_EXAMPLE_RULES, build_fis, build_rules, crisp_to_layer,
)


def test_rule_base_is_complete():
    rules = build_rules()
    assert len(rules) == 3 ** 5 == 243
    assert len({tuple(sorted(r.antecedent.items())) for r in rules}) == 243
    assert Counter(r.consequent for r in rules) == {"Fog": 141, "Edge": 51, "Cloud": 51}


def test_rule_base_reproduces_paper_examples():
    table = {tuple(sorted(r.antecedent.items())): r.consequent for r in build_rules()}
    for r in PAPER_EXAMPLE_RULES:
        assert table[tuple(sorted(r.antecedent.items()))] == r.consequent, str(r)


def test_thresholds():
    assert [crisp_to_layer(c) for c in (0, 59.9, 60, 140, 140.1, 200)] == [
        "Edge", "Edge", "Fog", "Fog", "Cloud", "Cloud"]


@pytest.mark.parametrize("values,layer", [
    ({"jl": 2, "vt": 10, "nl": 29, "bw": 1, "dl": 3}, "Edge"),     # tiny job, far cloud, tight deadline
    ({"jl": 20, "vt": 100, "nl": 15, "bw": 10, "dl": 25}, "Fog"),  # everything medium
    ({"jl": 39, "vt": 195, "nl": 1, "bw": 19, "dl": 49}, "Cloud"),  # big load, fast link, loose deadline
])
def test_extreme_scenarios(values, layer):
    assert crisp_to_layer(build_fis().compute(values)) == layer


def _skfuzzy_system(method="centroid"):
    """Same FIS built with scikit-fuzzy (min AND, max aggregation, given defuzzification)."""
    ants = {}
    for v in INPUTS:
        a = ctrl.Antecedent(v.universe(1001), v.name)
        for t, mf in v.terms.items():
            fn = fuzz.trimf if mf.kind == "tri" else fuzz.trapmf
            a[t] = fn(a.universe, list(mf.params))
        ants[v.name] = a
    out = ctrl.Consequent(DECISION.universe(1001), "decision", defuzzify_method=method)
    for t, mf in DECISION.terms.items():
        out[t] = fuzz.trimf(out.universe, list(mf.params))
    rules = []
    for r in build_rules():
        cond = None
        for var, term in r.antecedent.items():
            cond = ants[var][term] if cond is None else cond & ants[var][term]
        rules.append(ctrl.Rule(cond, out[r.consequent]))
    return ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules))


@pytest.mark.parametrize("method,n", [
    ("centroid", 40), ("bisector", 10), ("mom", 10), ("som", 10), ("lom", 10)])
def test_matches_scikit_fuzzy(method, n):
    ours, ref = build_fis(method), _skfuzzy_system(method)
    rng = np.random.default_rng(0)
    for _ in range(n):
        vals = {v.name: rng.uniform(v.lo, v.hi) for v in INPUTS}
        for k, x in vals.items():
            ref.input[k] = x
        ref.compute()
        assert ours.compute(vals) == pytest.approx(ref.output["decision"], abs=0.5)


def test_nan_input_and_output_are_rejected():
    fis = build_fis()
    with pytest.raises(ValueError):
        fis.compute({"jl": float("nan"), "vt": 50, "nl": 10, "bw": 5, "dl": 20})
    with pytest.raises(ValueError):
        crisp_to_layer(float("nan"))


def test_rules_are_hashable_and_fis_comparable():
    rules = build_rules()
    assert len(set(rules)) == 243
    fis = build_fis()
    assert fis == fis and fis != build_fis()  # identity comparison, no numpy ambiguity


def test_vectorized_firing_matches_naive_min():
    fis = build_fis()
    vals = {"jl": 13, "vt": 70, "nl": 6, "bw": 7, "dl": 18}
    res = fis.infer(vals)
    naive = [min(res.memberships[v][t] for v, t in r.antecedent.items()) for r in fis.rules]
    assert np.allclose(res.firing, naive)
