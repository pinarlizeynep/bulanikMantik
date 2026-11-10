"""MIN-MIN workflow scheduling with a pluggable offloading policy.

How one simulation works
  1. A task becomes *ready* when all of its parents have been scheduled.  At that moment the
     offloading policy is asked once which layer(s) the task may run on (Edge / Fog / Cloud).
     This mirrors the paper: the offloading technique picks the layer, MIN-MIN schedules inside it.
  2. MIN-MIN round: for every ready task compute its earliest finish time (EFT) on every node of
     its allowed layer(s); pick the (task, node) pair with the smallest EFT and assign it.
  3. Repeat until every task is scheduled.

  EFT(task, node) = max(node free time, data-ready time) + length / MIPS
  data-ready time = max over parents (parent finish + transfer(parent node -> node)),
                    or, for entry tasks, the upload time of the device's input data.

Metrics (paper Sec. 2.1)
  makespan  = finish of the last task - start of the first task                           Eq. (12)
  energy    = sum over nodes: busy_time * P_busy + (makespan - busy_time) * P_idle         [J]
              (idle="used": only nodes that executed at least one task are powered on and draw idle
               power; idle="all": every node of the environment draws idle power all the time)
  cost      = sum over tasks: exec_time * task_price + MB received * comm_price           [$]
  deadline miss rate = share of tasks whose finish > ready time + slack
  offloading time    = mean time a task waits for its input data after it became ready
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Protocol

from src.env import LAYERS, Environment, Node
from src.workflow import Workflow


@dataclass(frozen=True)
class Assignment:
    task: int
    node: str
    layer: str
    ready: float  # time the task became ready (all parents finished)
    start: float
    finish: float
    received_mb: float  # data moved onto the node for this task
    data_wait: float = 0.0  # data-ready time minus ready time (transfer / upload delay)


class Policy(Protocol):
    name: str

    def allowed_layers(self, task_id: int, ctx: "SchedulingContext") -> tuple[str, ...]: ...


@dataclass
class SchedulingContext:
    """What a policy may observe at the moment a task becomes ready."""

    workflow: Workflow
    env: Environment
    now: float  # ready time of the task being decided
    node_free: dict[str, float]  # time at which each node becomes free
    n_ready: int  # tasks waiting in the ready list, including this one (job list size)
    pending_mi: dict[str, float]  # work already sent to each layer but not yet scheduled
    drt: dict[tuple[int, str], float] = field(repr=False, default_factory=dict)

    def eft(self, task_id: int, node: Node) -> float:
        start = max(self.node_free[node.name], self.drt[(task_id, node.name)])
        return start + node.exec_time(self.workflow.tasks[task_id].length_mi)

    def data_delay(self, task_id: int, layer: str) -> float:
        """Shortest wait (s) after ``now`` until the task's input data can be on a node of ``layer``."""
        return max(0.0, min(self.drt[(task_id, n.name)] for n in self.env.by_layer[layer]) - self.now)

    def best_eft(self, task_id: int, layer: str) -> float:
        return min(self.eft(task_id, n) for n in self.env.by_layer[layer])

    def expected_eft(self, task_id: int, layer: str) -> float:
        """Best EFT on ``layer`` plus the layer's not-yet-scheduled queue spread over its nodes.

        ``best_eft`` alone is optimistic: tasks that were already sent to the layer but are still
        waiting in the ready list do not occupy any node yet.
        """
        capacity = sum(n.spec.mips for n in self.env.by_layer[layer])
        return self.best_eft(task_id, layer) + max(0.0, self.pending_mi[layer]) / capacity

    def load(self, layer: str) -> float:
        """Seconds of work ahead of a new task on ``layer``: node queues plus work still waiting."""
        capacity = sum(n.spec.mips for n in self.env.by_layer[layer])
        return self.backlog(layer) + max(0.0, self.pending_mi[layer]) / capacity

    def backlog(self, layer: str) -> float:
        """Mean queued work (s) of a layer's nodes beyond ``now``."""
        nodes = self.env.by_layer[layer]
        return sum(max(0.0, self.node_free[n.name] - self.now) for n in nodes) / len(nodes)


@dataclass
class SimulationResult:
    policy: str
    workflow: str
    n_tasks: int
    makespan: float
    energy_j: float
    cost: float
    deadline_miss_rate: float
    layer_share: dict[str, float]
    assignments: list[Assignment]
    mean_offload_s: float = 0.0

    def summary(self) -> dict:
        row = {"policy": self.policy, "workflow": self.workflow, "jobs": self.n_tasks,
               "makespan_s": self.makespan, "energy_J": self.energy_j, "cost_usd": self.cost,
               "deadline_miss_rate": self.deadline_miss_rate, "offload_s": self.mean_offload_s}
        row.update({f"share_{k}": v for k, v in self.layer_share.items()})
        return row


def simulate(workflow: Workflow, env: Environment, policy: Policy, idle: str = "used") -> SimulationResult:
    if idle not in ("used", "all"):
        raise ValueError("idle must be 'used' or 'all'")
    nodes = {n.name: n for n in env.nodes}
    node_free = {n: 0.0 for n in nodes}
    node_busy = {n: 0.0 for n in nodes}
    placed: dict[int, Assignment] = {}
    remaining_parents = {t: workflow.graph.in_degree(t) for t in workflow.tasks}
    ready: dict[int, tuple] = {}  # task -> (ready time, allowed nodes, allowed layers)
    pending_mi = {layer: 0.0 for layer in LAYERS}
    drt: dict[tuple[int, str], float] = {}
    received: dict[tuple[int, str], float] = {}

    def make_ready(tid: int) -> None:
        task = workflow.tasks[tid]
        parents = workflow.parents(tid)
        t_ready = max((placed[p].finish for p in parents), default=0.0)
        for node in env.nodes:  # data-ready time and data volume for every candidate node
            if parents:
                drt[(tid, node.name)] = max(
                    placed[p].finish + env.transfer_time(nodes[placed[p].node], node,
                                                         workflow.tasks[p].output_mb)
                    for p in parents)
                received[(tid, node.name)] = sum(
                    workflow.tasks[p].output_mb for p in parents if placed[p].node != node.name)
            else:
                drt[(tid, node.name)] = env.upload_time(node, task.input_mb)
                received[(tid, node.name)] = task.input_mb
        ctx = SchedulingContext(workflow, env, t_ready, node_free, len(ready) + 1, pending_mi, drt)
        layers = tuple(policy.allowed_layers(tid, ctx))
        allowed = tuple(n for layer in layers for n in env.by_layer[layer])
        if not allowed:
            raise ValueError(f"policy {policy.name} allowed no available node for task {tid}")
        for layer in layers:  # split the pending work evenly over the allowed layers
            pending_mi[layer] += task.length_mi / len(layers)
        ready[tid] = (t_ready, allowed, layers)

    for tid in workflow.entry_tasks():
        make_ready(tid)

    while ready:
        best = None  # (eft, start, task id, node)
        for tid, (_, allowed, _) in ready.items():
            length = workflow.tasks[tid].length_mi
            for node in allowed:
                start = max(node_free[node.name], drt[(tid, node.name)])
                eft = start + node.exec_time(length)
                if best is None or eft < best[0]:
                    best = (eft, start, tid, node)
        eft, start, tid, node = best
        t_ready, _, layers = ready.pop(tid)
        for layer in layers:
            pending_mi[layer] -= workflow.tasks[tid].length_mi / len(layers)
        placed[tid] = Assignment(tid, node.name, node.layer, t_ready, start, eft,
                                 received[(tid, node.name)], max(0.0, drt[(tid, node.name)] - t_ready))
        node_free[node.name] = eft
        node_busy[node.name] += eft - start
        for child in workflow.children(tid):
            remaining_parents[child] -= 1
            if remaining_parents[child] == 0:
                make_ready(child)

    return _metrics(workflow, env, policy.name, list(placed.values()), node_busy, idle)


def _metrics(workflow, env, policy_name, assignments, node_busy, idle) -> SimulationResult:
    t0 = min(a.start for a in assignments)
    makespan = max(a.finish for a in assignments) - t0
    powered = [n for n in env.nodes if idle == "all" or node_busy[n.name] > 0]
    energy_mj = sum(node_busy[n.name] * n.spec.busy_power_mw
                    + max(0.0, makespan - node_busy[n.name]) * n.spec.idle_power_mw
                    for n in powered)  # mW * s = mJ
    specs = {n.name: n.spec for n in env.nodes}
    cost = sum((a.finish - a.start) * specs[a.node].task_price
               + a.received_mb * specs[a.node].comm_price for a in assignments)
    misses = sum(a.finish > a.ready + workflow.tasks[a.task].slack_s for a in assignments)
    counts = Counter(a.layer for a in assignments)
    share = {layer: counts.get(layer, 0) / len(assignments) for layer in LAYERS}
    return SimulationResult(policy_name, workflow.name, len(assignments), makespan,
                            energy_mj / 1000.0, cost, misses / len(assignments), share,
                            sorted(assignments, key=lambda a: (a.start, a.task)),
                            sum(a.data_wait for a in assignments) / len(assignments))
