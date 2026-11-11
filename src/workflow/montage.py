"""MONTAGE-like workflow generator.

MONTAGE (Pegasus) builds a sky mosaic from n input images. Its DAG has a fixed level structure:

    mProjectPP  x n      re-project every input image             (entry tasks, read device data)
    mDiffFit    x d      fit the overlap of two neighbouring projections  (2 parents each)
    mConcatFit  x 1      collect all fits                          (join)
    mBgModel    x 1      global background model
    mBackground x n      correct each projection   (parents: its mProjectPP + mBgModel)
    mImgtbl     x 1      image table                               (join)
    mAdd        x 1      co-add into the mosaic
    mShrink     x 1      shrink the mosaic
    mJPEG       x 1      render the final image                    (exit task)

Total = 2n + d + 6.  Real Montage_25 has n = 5, d = 9.  For a requested job count N we choose
n = max(2, round((N - 6) / 4)) and d = N - 2n - 6 (so d ~ 2n, overlaps with the next one and two images).

Runtimes are approximate per-type averages of Pegasus MONTAGE traces (seconds on a 1000-MIPS
reference CPU), converted to millions of instructions (MI) and jittered by +-20 % per task.
Each task also gets a deadline slack (s): slack = U(1.2, 4) x reference runtime, capped at 50 s,
which is the deadline universe [0, 50] of the paper's FIS.
Deviation from the paper: Sec. 3.2 uses a single "15 unit deadline" without unit or scope.  A
workflow-level deadline of 15 s (or ms) is infeasible for MONTAGE on Table 3 nodes, and the FIS needs a
per-task deadline in [0, 50], so we use the per-task slack above.
"""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

REFERENCE_MIPS = 1000.0
MAX_SLACK_S = 50.0

# kind: (reference runtime s, output MB)
TASK_TYPES = {
    "mProjectPP": (13.0, 8.0),
    "mDiffFit": (11.0, 0.3),
    "mConcatFit": (40.0, 0.1),
    "mBgModel": (30.0, 0.01),
    "mBackground": (11.0, 8.0),
    "mImgtbl": (3.0, 0.01),
    "mAdd": (60.0, 100.0),
    "mShrink": (10.0, 3.0),
    "mJPEG": (6.0, 0.3),
}
ENTRY_INPUT_MB = 4.0  # raw image uploaded by the IoT device for every mProjectPP


@dataclass(frozen=True)
class Task:
    id: int
    kind: str
    length_mi: float
    output_mb: float
    input_mb: float  # data uploaded from the device (only entry tasks)
    slack_s: float  # deadline relative to the time the task becomes ready


class Workflow:
    """A DAG of tasks; edges carry the parent's output data to the child."""

    def __init__(self, tasks: list[Task], edges: list[tuple[int, int]], name: str = "workflow"):
        self.name = name
        self.tasks = {t.id: t for t in tasks}
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(self.tasks)
        self.graph.add_edges_from(edges)
        if not nx.is_directed_acyclic_graph(self.graph):
            raise ValueError("workflow must be acyclic")

    def __len__(self) -> int:
        return len(self.tasks)

    def parents(self, tid: int) -> list[int]:
        return list(self.graph.predecessors(tid))

    def children(self, tid: int) -> list[int]:
        return list(self.graph.successors(tid))

    def entry_tasks(self) -> list[int]:
        return [t for t in self.tasks if self.graph.in_degree(t) == 0]

    def total_length_mi(self) -> float:
        return sum(t.length_mi for t in self.tasks.values())


def montage(n_jobs: int, seed: int | None = 0) -> Workflow:
    """Generate a MONTAGE-like workflow with exactly ``n_jobs`` tasks (n_jobs >= 11)."""
    if n_jobs < 11:
        raise ValueError("MONTAGE needs at least 11 tasks (2 images, 1 overlap)")
    rng = np.random.default_rng(seed)
    n = max(2, round((n_jobs - 6) / 4))
    d = n_jobs - 2 * n - 6

    tasks: list[Task] = []
    edges: list[tuple[int, int]] = []

    def add(kind: str) -> int:
        runtime, out_mb = TASK_TYPES[kind]
        length = runtime * REFERENCE_MIPS * rng.uniform(0.8, 1.2)
        slack = min(MAX_SLACK_S, rng.uniform(1.2, 4.0) * runtime)
        tid = len(tasks)
        tasks.append(Task(tid, kind, float(length), out_mb,
                          ENTRY_INPUT_MB if kind == "mProjectPP" else 0.0, float(slack)))
        return tid

    project = [add("mProjectPP") for _ in range(n)]
    # overlapping pairs: neighbours at distance 1, 2, 3, ... (cyclic) until d pairs exist
    pairs, gap = [], 1
    while len(pairs) < d:
        for i in range(n):
            pair = tuple(sorted((i, (i + gap) % n)))
            if pair[0] != pair[1] and pair not in pairs:
                pairs.append(pair)
                if len(pairs) == d:
                    break
        gap += 1
        if gap > n:  # all distinct pairs used; allow repeated overlaps
            pairs += [pairs[k % len(pairs)] for k in range(d - len(pairs))]
    diff = []
    for i, j in pairs:
        t = add("mDiffFit")
        edges += [(project[i], t), (project[j], t)]
        diff.append(t)
    concat = add("mConcatFit")
    edges += [(t, concat) for t in diff]
    bgmodel = add("mBgModel")
    edges.append((concat, bgmodel))
    background = []
    for p in project:
        t = add("mBackground")
        edges += [(p, t), (bgmodel, t)]
        background.append(t)
    imgtbl = add("mImgtbl")
    edges += [(t, imgtbl) for t in background]
    madd = add("mAdd")
    edges += [(imgtbl, madd)] + [(t, madd) for t in background]
    shrink = add("mShrink")
    edges.append((madd, shrink))
    jpeg = add("mJPEG")
    edges.append((shrink, jpeg))
    return Workflow(tasks, edges, name=f"MONTAGE_{n_jobs}")
