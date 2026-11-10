"""Proposed method: fuzzy logic-based offloading of Soni & Kumar (2024), Algorithms 1-5.

When a task becomes ready the policy
  1. picks the algorithm from the user's QoS preference (Algorithm 1 + Table 2, ``qos_selector``)
       A2 -> Cloud, A3 -> Fog, A4 -> Edge, A5 -> hybrid (steps 2-4 below)
  2. observes the system and converts it into the five FIS inputs (``observe``, below)
  3. runs the Mamdani FIS -> crisp value C in [0, 200]  (<60 Edge, 60-140 Fog, >140 Cloud)
  4. QoS bias: the preference moves the crisp value, ``C' = C + SHIFT * (cost - time) / (T + E + C)``
     (cost pulls towards the cloud, time towards the edge, as in Algorithms 2 and 4)
  5. offloading condition (Alg. 5 "dl not met"): if the chosen layer would miss the task's deadline
     and another layer can still meet it, the layer with the earliest expected finish is used.
     This check can be switched off (``verify_deadline=False``) for the ablation study.
MIN-MIN then picks the node inside the chosen layer.

Observation -> FIS input mapping (the paper only names the inputs; the mapping is ours and is
documented in docs/gunluk/gun4):
  jl  job list size       number of tasks waiting in the ready list, clipped to [0, 40]
  vt  VM utilization      load of the local tier (edge + fog): queued + waiting work relative to
                          ``vt_ref_s`` seconds, in percent, clipped to [0, 200]
  nl  network latency     extra delay of reaching the cloud compared with the fog, relative to
                          ``nl_ref_s`` seconds, scaled to [0, 30]
  bw  bandwidth           link rate towards the cloud relative to the fastest link (800 Mbit/s),
                          scaled to [0, 20]
  dl  deadline            slack of the task in seconds, already in [0, 50]
This is the paper's min-max normalisation (Eq. 23) in spirit: raw quantities are divided by a
maximum and stretched to the universe of the fuzzy variable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.fuzzy.paper_fis import build_fis, crisp_to_layer
from src.fuzzy.qos_selector import ALGORITHM_LAYER, select_algorithm

BALANCED = (1 / 3, 1 / 3, 1 / 3)  # (time, energy, cost) preference


@dataclass(frozen=True)
class InputScales:
    vt_ref_s: float = 60.0
    nl_ref_s: float = 0.25
    max_rate_mbps: float = 800.0


def observe(task_id: int, ctx, scales: InputScales = InputScales()) -> dict[str, float]:
    """Simulator state -> the 5 crisp FIS inputs."""
    wf, env = ctx.workflow, ctx.env
    local = [layer for layer in ("Edge", "Fog") if env.by_layer[layer]]
    backlog = sum(ctx.load(layer) for layer in local) / len(local)
    cloud_delay = ctx.data_delay(task_id, "Cloud")
    fog_delay = ctx.data_delay(task_id, "Fog")
    cloud = env.by_layer["Cloud"][0].spec
    rate = cloud.up_bw if wf.graph.in_degree(task_id) == 0 else cloud.down_bw
    return {
        "jl": float(min(ctx.n_ready, 40)),
        "vt": min(200.0, max(0.0, 100.0 * backlog / scales.vt_ref_s)),
        "nl": 30.0 * min(1.0, max(0.0, cloud_delay - fog_delay) / scales.nl_ref_s),
        "bw": 20.0 * min(1.0, rate / scales.max_rate_mbps),
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
    """``preference`` = (time, energy, cost) degrees in [0, 1]; the default is a balanced user."""

    preference: tuple[float, float, float] = BALANCED
    defuzz: str = "centroid"
    shift: float = 40.0
    verify_deadline: bool = True
    layer_map: dict = field(default_factory=lambda: dict(ALGORITHM_LAYER))  # A2/A3/A4 -> layer
    scales: InputScales = field(default_factory=InputScales)
    name: str = "Proposed fuzzy (Soni & Kumar)"
    decisions: list[Decision] = field(default_factory=list, repr=False)

    def __post_init__(self):
        self.fis = build_fis(self.defuzz)
        self.algorithm = select_algorithm(*self.preference)

    def _hybrid_layer(self, inputs: dict[str, float]) -> tuple[float, str]:
        t, e, c = self.preference
        crisp = self.fis.compute(inputs)
        crisp += self.shift * (c - t) / max(t + e + c, 1e-9)
        return crisp, crisp_to_layer(min(200.0, max(0.0, crisp)))

    def allowed_layers(self, task_id: int, ctx) -> tuple[str, ...]:
        inputs = observe(task_id, ctx, self.scales)
        if self.algorithm == "A5":
            crisp, fis_layer = self._hybrid_layer(inputs)
        else:
            crisp, fis_layer = float("nan"), self.layer_map[self.algorithm]
        layer = fis_layer
        # A single-objective algorithm that targets the edge runs locally by design (Alg. 4: if the
        # offloading condition fails the job stays local), so only the other cases are verified.
        local_by_design = self.algorithm != "A5" and fis_layer == "Edge"
        if self.verify_deadline and not local_by_design:
            deadline = ctx.now + ctx.workflow.tasks[task_id].slack_s
            finish = {l: ctx.expected_eft(task_id, l) for l in ctx.env.layers_present()}
            if finish[layer] > deadline:
                best = min(finish, key=finish.get)
                if finish[best] <= deadline:
                    layer = best
        self.decisions.append(Decision(task_id, self.algorithm, inputs, crisp, fis_layer, layer))
        return (layer,)
