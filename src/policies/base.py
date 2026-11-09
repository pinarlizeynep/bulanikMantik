"""Reference policies used as sanity bounds.

* ``SingleLayerPolicy("Edge" | "Fog" | "Cloud")`` sends every task to one layer.  These correspond
  to the paper's single-objective algorithms (Alg. 4 time -> Edge, Alg. 3 energy -> Fog,
  Alg. 2 cost -> Cloud) and show the extreme trade-offs of the environment.
* ``AllLayersPolicy`` lets MIN-MIN choose freely among every node: no offloading logic at all,
  only greedy earliest-finish-time.  It is a strong makespan reference, not a paper method.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.env import LAYERS


@dataclass
class SingleLayerPolicy:
    layer: str

    def __post_init__(self):
        if self.layer not in LAYERS:
            raise ValueError(f"unknown layer {self.layer}")
        self.name = f"{self.layer}-only"

    def allowed_layers(self, task_id, ctx) -> tuple[str, ...]:
        return (self.layer,)


@dataclass
class AllLayersPolicy:
    name: str = "MIN-MIN (all layers)"

    def allowed_layers(self, task_id, ctx) -> tuple[str, ...]:
        return ctx.env.layers_present()
