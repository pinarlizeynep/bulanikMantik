"""Compute nodes and network links of the edge-fog-cloud environment.

Node parameters are taken from the paper's Table 3 (FogWorkflowSim settings):

    layer  MIPS  UpBw  DownBw  busy mW  idle mW  task price  comm. price
    Cloud  1600   300     500     1600     1300        0.96         0.02
    Fog    1300   500     800      800       40        0.48         0.01
    Edge   1000   800    1000      700       30        0            0

The paper does not state units or how prices are charged, so this model fixes them explicitly
(documented in docs/gunluk/gun3 and the report):
  * bandwidth in Mbit/s, data sizes in MB  -> transfer time = 8 * MB / Mbps  [s]
  * task price is charged per second of execution, communication price per MB received
  * one-way network latency per layer (not in Table 3): edge 2 ms, fog 10 ms, cloud 30 ms;
    these values lie inside the FIS latency universe [0, 30] ms.
"""
from __future__ import annotations

from dataclasses import dataclass

LAYERS = ("Edge", "Fog", "Cloud")


@dataclass(frozen=True)
class NodeSpec:
    layer: str
    mips: float
    up_bw: float  # Mbit/s, device -> node and node -> other node
    down_bw: float  # Mbit/s, data arriving at the node from another node
    busy_power_mw: float
    idle_power_mw: float
    task_price: float  # $ per second of execution
    comm_price: float  # $ per MB received
    latency_s: float  # one-way latency between an IoT device and this layer


TABLE3 = {
    "Cloud": NodeSpec("Cloud", 1600, 300, 500, 1600, 1300, 0.96, 0.02, 0.030),
    "Fog": NodeSpec("Fog", 1300, 500, 800, 800, 40, 0.48, 0.01, 0.010),
    "Edge": NodeSpec("Edge", 1000, 800, 1000, 700, 30, 0.0, 0.0, 0.002),
}


@dataclass(frozen=True)
class Node:
    name: str  # e.g. "Fog-1"
    spec: NodeSpec

    @property
    def layer(self) -> str:
        return self.spec.layer

    def exec_time(self, length_mi: float) -> float:
        return length_mi / self.spec.mips


class Environment:
    """A fixed set of nodes plus the network model between them."""

    def __init__(self, nodes: list[Node]):
        if not nodes:
            raise ValueError("environment needs at least one node")
        self.nodes = list(nodes)
        self.by_layer = {layer: [n for n in self.nodes if n.layer == layer] for layer in LAYERS}

    def upload_time(self, dst: Node, mb: float) -> float:
        """IoT device -> node (input data of entry tasks)."""
        return dst.spec.latency_s + 8.0 * mb / dst.spec.up_bw

    def transfer_time(self, src: Node, dst: Node, mb: float) -> float:
        """Output of a parent task on ``src`` -> child task on ``dst`` (0 if same node)."""
        if src == dst or mb <= 0:
            return 0.0
        bw = min(src.spec.up_bw, dst.spec.down_bw)
        return src.spec.latency_s + dst.spec.latency_s + 8.0 * mb / bw

    def layers_present(self) -> tuple[str, ...]:
        return tuple(layer for layer in LAYERS if self.by_layer[layer])


def default_environment(n_edge: int = 2, n_fog: int = 2, n_cloud: int = 2) -> Environment:
    nodes = []
    for layer, count in (("Edge", n_edge), ("Fog", n_fog), ("Cloud", n_cloud)):
        nodes += [Node(f"{layer}-{i + 1}", TABLE3[layer]) for i in range(count)]
    return Environment(nodes)
