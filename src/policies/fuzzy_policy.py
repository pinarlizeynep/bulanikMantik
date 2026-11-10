"""Proposed method: fuzzy logic-based offloading of Soni & Kumar (2024), Algorithms 1-5.

When a task becomes ready the policy
  1. picks the algorithm from the user's QoS preference (Algorithm 1 + Table 2, ``qos_selector``)
       A2 -> Cloud, A3 -> Fog, A4 -> Edge, A5 -> hybrid (steps 2-4 below)
  2. observes the system and converts it into the five FIS inputs (``observe``, below)
  3. runs the Mamdani FIS -> crisp value C in [0, 200]  (<60 Edge, 60-140 Fog, >140 Cloud)
  4. QoS bias: each preference pulls the crisp value towards the layer of its single-objective
     algorithm in ``layer_map``, ``C' = C + SHIFT * (sum_k pref_k * pos(layer_k)) / (T + E + C)`` with
     pos(Edge) = -1, pos(Fog) = 0, pos(Cloud) = +1.  With the published mapping this is
     ``SHIFT * (cost - time) / (T + E + C)`` (cost -> cloud, time -> edge, Algorithms 2 and 4); with the
     physical mapping the direction flips, so the hybrid and the pure algorithms always agree.
  5. offloading condition (Alg. 5 "dl not met"): if the chosen layer would miss the task's deadline
     and another layer can still meet it, the layer with the earliest expected finish is used.
     Local execution on the edge chosen by a single-objective algorithm is not an offloading decision
     and is not verified (Alg. 2-4: "if the offloading condition fails the job is executed locally").
     The check can be switched off (``verify_deadline=False``) for the ablation study.
     Consequently A2 / A3 send *most*, not all, jobs to their target layer when it is fog or cloud.
MIN-MIN then picks the node inside the chosen layer.

Observation -> FIS input mapping (the paper only names the inputs; the mapping is ours and is
documented in docs/gunluk/gun4):
  jl  job list size       number of tasks waiting in the ready list, clipped to [0, 40]
  vt  VM utilization      load of the local tier (edge + fog): queued + waiting work relative to
                          ``vt_ref_s`` seconds, in percent, clipped to [0, 200]
  nl  network latency     extra delay of reaching the cloud compared with the fog, relative to
                          ``nl_ref_s`` seconds, scaled to [0, 30]
  bw  bandwidth           rate of the link that would carry the task's input data to the best cloud
                          node (``min(src.up_bw, dst.down_bw)`` as in ``Environment.transfer_time``;
                          the device uplink for entry tasks), relative to the fastest effective link
                          of Table 3 (edge uplink, 800 Mbit/s), scaled to [0, 20]
  dl  deadline            slack of the task in seconds, already in [0, 50]
This is the paper's min-max normalisation (Eq. 23) in spirit: raw quantities are divided by a
maximum and stretched to the universe of the fuzzy variable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.fuzzy.paper_fis import build_fis, crisp_to_layer
from src.fuzzy.qos_selector import ALGORITHM_LAYER, select_algorithm

BALANCED = (1 / 3, 1 / 3, 1 / 3)  # (time, energy, cost) preference
LAYER_POSITION = {"Edge": -1.0, "Fog": 0.0, "Cloud": 1.0}  # direction of the QoS shift on the output axis


@dataclass(frozen=True)
class InputScales:
    vt_ref_s: float = 60.0
    nl_ref_s: float = 0.25
    max_rate_mbps: float = 800.0


def _cloud_rate(task_id: int, ctx) -> float:
    """Rate (Mbit/s) of the link that brings the task's input to the best cloud node.

    Entry tasks are uploaded from the device (cloud uplink).  Otherwise the bottleneck is
    ``min(src.up_bw, dst.down_bw)`` over the parents placed on another node (the transfer model of
    ``Environment.transfer_time``); a cloud node that already holds all inputs needs no link.
    """
    parents = ctx.workflow.parents(task_id)
    best = 0.0
    for node in ctx.env.by_layer["Cloud"]:
        if not parents:
            rate = node.spec.up_bw
        else:
            links = [min(ctx.node_of[p].spec.up_bw, node.spec.down_bw)
                     for p in parents if ctx.node_of[p].name != node.name]
            rate = min(links) if links else float("inf")
        best = max(best, rate)
    return best


def observe(task_id: int, ctx, scales: InputScales = InputScales()) -> dict[str, float]:
    """Simulator state -> the 5 crisp FIS inputs.

    Works with partial environments: without edge/fog nodes the local load is 0, without cloud
    nodes the cloud counts as unreachable (maximal latency, no bandwidth).
    """
    wf, env = ctx.workflow, ctx.env
    local = [layer for layer in ("Edge", "Fog") if env.by_layer[layer]]
    backlog = sum(ctx.load(layer) for layer in local) / len(local) if local else 0.0
    if env.by_layer["Cloud"]:
        near = [layer for layer in ("Fog", "Edge") if env.by_layer[layer]]
        cloud_delay = ctx.data_delay(task_id, "Cloud")
        near_delay = ctx.data_delay(task_id, near[0]) if near else cloud_delay
        nl = 30.0 * min(1.0, max(0.0, cloud_delay - near_delay) / scales.nl_ref_s)
        bw = 20.0 * min(1.0, _cloud_rate(task_id, ctx) / scales.max_rate_mbps)
    else:
        nl, bw = 30.0, 0.0
    return {
        "jl": float(min(ctx.n_ready, 40)),
        "vt": min(200.0, max(0.0, 100.0 * backlog / scales.vt_ref_s)),
        "nl": nl,
        "bw": bw,
        "dl": min(50.0, wf.tasks[task_id].slack_s),
    }


@dataclass
class Decision:
    task: int
    algorithm: str
    inputs: dict[str, float]
    crisp: float  # nan for A2-A4
    fis_layer: str
    layer: str


@dataclass
class FuzzyOffloadingPolicy:
    """``preference`` = (time, energy, cost) degrees in [0, 1]; the default is a balanced user.

    Every decision is appended to ``decisions`` (``record=False`` turns the log off).  Use one policy
    object per simulation, or ``decisions.clear()`` in between, so that logs of runs do not mix.
    """

    preference: tuple[float, float, float] = BALANCED
    defuzz: str = "centroid"
    shift: float = 40.0
    verify_deadline: bool = True
    layer_map: dict = field(default_factory=lambda: dict(ALGORITHM_LAYER))  # A2/A3/A4 -> layer
    scales: InputScales = field(default_factory=InputScales)
    name: str = "Proposed fuzzy (Soni & Kumar)"
    record: bool = True
    decisions: list[Decision] = field(default_factory=list, repr=False)

    def __post_init__(self):
        if set(self.layer_map) != {"A2", "A3", "A4"} or not set(self.layer_map.values()) <= set(LAYER_POSITION):
            raise ValueError(f"layer_map must map A2, A3, A4 to Edge/Fog/Cloud, got {self.layer_map}")
        self.fis = build_fis(self.defuzz)
        self.algorithm = select_algorithm(*self.preference)
        t, e, c = self.preference
        pull = (c * LAYER_POSITION[self.layer_map["A2"]] + e * LAYER_POSITION[self.layer_map["A3"]]
                + t * LAYER_POSITION[self.layer_map["A4"]])
        self.crisp_shift = self.shift * pull / max(t + e + c, 1e-9)

    def _hybrid_layer(self, inputs: dict[str, float]) -> tuple[float, str]:
        crisp = self.fis.compute(inputs) + self.crisp_shift
        return crisp, crisp_to_layer(min(200.0, max(0.0, crisp)))

    def allowed_layers(self, task_id: int, ctx) -> tuple[str, ...]:
        if self.algorithm == "A5":
            inputs = observe(task_id, ctx, self.scales)
            crisp, fis_layer = self._hybrid_layer(inputs)
        else:  # fixed target layer: the FIS inputs are not needed
            inputs, crisp, fis_layer = {}, float("nan"), self.layer_map[self.algorithm]
        layer = fis_layer
        present = ctx.env.layers_present()
        local_execution = self.algorithm != "A5" and fis_layer == "Edge"
        if (self.verify_deadline and not local_execution) or layer not in present:
            finish = {l: ctx.expected_eft(task_id, l) for l in present}
            best = min(finish, key=finish.get)
            if layer not in present:  # target layer absent from this environment
                layer = best
            elif finish[layer] > ctx.now + ctx.workflow.tasks[task_id].slack_s >= finish[best]:
                layer = best  # "dl not met" on the chosen layer, but another layer meets it
        if self.record:
            self.decisions.append(Decision(task_id, self.algorithm, inputs, crisp, fis_layer, layer))
        return (layer,)
