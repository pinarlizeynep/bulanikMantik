from collections import Counter

import networkx as nx
import pytest

from src.env import TABLE3, Environment, Node, default_environment
from src.policies import AllLayersPolicy, SingleLayerPolicy, TwoWayFogCloudPolicy
from src.scheduler import simulate
from src.workflow import Task, Workflow, montage


# ------------------------------------------------------------------ environment
def test_table3_parameters():
    assert (TABLE3["Cloud"].mips, TABLE3["Fog"].mips, TABLE3["Edge"].mips) == (1600, 1300, 1000)
    assert (TABLE3["Cloud"].up_bw, TABLE3["Fog"].up_bw, TABLE3["Edge"].up_bw) == (300, 500, 800)
    assert TABLE3["Edge"].task_price == 0 and TABLE3["Cloud"].task_price > TABLE3["Fog"].task_price


def test_transfer_time():
    env = default_environment()
    edge, cloud = env.by_layer["Edge"][0], env.by_layer["Cloud"][0]
    assert env.transfer_time(edge, edge, 100) == 0
    # 10 MB over min(800 up, 500 down) Mbit/s = 0.16 s, plus 2 + 30 ms latency
    assert env.transfer_time(edge, cloud, 10) == pytest.approx(0.16 + 0.032)


# ------------------------------------------------------------------ workflow
@pytest.mark.parametrize("n", [11, 20, 25, 100, 300])
def test_montage_size_and_shape(n):
    wf = montage(n, seed=3)
    kinds = Counter(t.kind for t in wf.tasks.values())
    assert len(wf) == n and nx.is_directed_acyclic_graph(wf.graph)
    assert kinds["mProjectPP"] == kinds["mBackground"] == len(wf.entry_tasks())
    assert all(kinds[k] == 1 for k in ("mConcatFit", "mBgModel", "mImgtbl", "mAdd", "mShrink", "mJPEG"))
    assert all(0 < t.slack_s <= 50 for t in wf.tasks.values())


def test_montage25_matches_pegasus_structure():
    kinds = Counter(t.kind for t in montage(25).tasks.values())
    assert kinds["mProjectPP"] == 5 and kinds["mDiffFit"] == 9


def test_montage_is_seeded():
    a, b = montage(40, seed=7), montage(40, seed=7)
    assert [t.length_mi for t in a.tasks.values()] == [t.length_mi for t in b.tasks.values()]


# ------------------------------------------------------------------ scheduler
def _chain():
    """a -> b on a single edge node: hand-computable schedule."""
    tasks = [Task(0, "a", 1000.0, 1.0, 1.0, 50.0), Task(1, "b", 2000.0, 0.0, 0.0, 50.0)]
    return Workflow(tasks, [(0, 1)], "chain")


def test_hand_computed_schedule():
    env = Environment([Node("Edge-1", TABLE3["Edge"])])
    res = simulate(_chain(), env, SingleLayerPolicy("Edge"))
    up = 0.002 + 8 * 1.0 / 800  # device upload of 1 MB
    a, b = res.assignments
    assert a.start == pytest.approx(up) and a.finish == pytest.approx(up + 1.0)
    assert b.start == pytest.approx(a.finish) and b.finish == pytest.approx(a.finish + 2.0)
    assert res.makespan == pytest.approx(3.0)
    assert res.energy_j == pytest.approx(3.0 * 700 / 1000)  # busy the whole makespan
    assert res.cost == 0


@pytest.mark.parametrize("policy", [TwoWayFogCloudPolicy(), AllLayersPolicy(),
                                    SingleLayerPolicy("Edge"), SingleLayerPolicy("Cloud")])
def test_schedule_respects_dependencies_and_nodes(policy):
    wf, env = montage(60, seed=2), default_environment()
    res = simulate(wf, env, policy)
    by_task = {a.task: a for a in res.assignments}
    assert len(by_task) == len(wf)
    for child in wf.tasks:  # a task never starts before its parents finish
        for p in wf.parents(child):
            assert by_task[child].start >= by_task[p].finish - 1e-9
    for node in env.nodes:  # no two tasks overlap on one node
        runs = sorted((a.start, a.finish) for a in res.assignments if a.node == node.name)
        assert all(s2 >= f1 - 1e-9 for (_, f1), (s2, _) in zip(runs, runs[1:]))
    assert sum(res.layer_share.values()) == pytest.approx(1.0)


def test_single_layer_policy_uses_only_that_layer():
    res = simulate(montage(40), default_environment(), SingleLayerPolicy("Fog"))
    assert res.layer_share["Fog"] == 1.0


def test_baseline_never_uses_edge_and_uses_cloud_under_load():
    res = simulate(montage(200, seed=0), default_environment(), TwoWayFogCloudPolicy())
    assert res.layer_share["Edge"] == 0
    assert res.layer_share["Cloud"] > 0.1  # fog saturates -> spill-over to the cloud


def test_edge_only_is_free():
    assert simulate(montage(40), default_environment(), SingleLayerPolicy("Edge")).cost == 0


def test_idle_power_only_for_used_nodes_by_default():
    wf, env = montage(40, seed=0), default_environment()
    used = simulate(wf, env, SingleLayerPolicy("Edge"))
    everything = simulate(wf, env, SingleLayerPolicy("Edge"), idle="all")
    assert used.makespan == everything.makespan
    assert used.energy_j < everything.energy_j  # unused fog/cloud nodes are not powered
    with pytest.raises(ValueError):
        simulate(wf, env, SingleLayerPolicy("Edge"), idle="nope")


def test_energy_ranking_follows_table3_power_per_mips():
    """mW per MIPS: fog 0.615 < edge 0.700 < cloud 1.000 -> the busy-energy ranking of the layers."""
    wf, env = montage(100, seed=0), default_environment()
    e = {layer: simulate(wf, env, SingleLayerPolicy(layer)).energy_j for layer in ("Edge", "Fog", "Cloud")}
    assert e["Fog"] < e["Edge"] < e["Cloud"]


def test_offloading_time_is_recorded():
    res = simulate(montage(20, seed=0), default_environment(), SingleLayerPolicy("Cloud"))
    assert res.mean_offload_s > 0
