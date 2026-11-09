"""Baseline: classical two-way fog-cloud offloading (Khan et al., 2023 -- the paper's comparison).

The paper compares against a "classical hybrid" method in which jobs are offloaded between the
fog and the cloud in both directions; edge devices are not used as execution targets.  The
original results come from another simulator (SFogSim).  We re-implement the idea inside *our*
simulator so that both methods run on identical workflows, nodes and network (fair comparison).

Decision rule for a ready task with deadline slack ``s`` at time ``now`` (finish times are
expected EFTs that include work already queued for the layer, see ``SchedulingContext``):
  1. Fog first: if the best fog node can finish it by ``now + s``           -> Fog
  2. Upward:    else if the best cloud node can meet the deadline            -> Cloud
  3. Neither meets it: whichever of fog / cloud finishes earlier             -> Fog or Cloud
Step 1 also realises the "cloud -> fog" direction: as soon as the fog has free capacity again,
tasks that would otherwise queue for the cloud are kept on (returned to) the fog.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TwoWayFogCloudPolicy:
    name: str = "Classical two-way (fog-cloud)"

    def allowed_layers(self, task_id, ctx) -> tuple[str, ...]:
        deadline = ctx.now + ctx.workflow.tasks[task_id].slack_s
        fog, cloud = ctx.expected_eft(task_id, "Fog"), ctx.expected_eft(task_id, "Cloud")
        if fog <= deadline:
            return ("Fog",)
        if cloud <= deadline:
            return ("Cloud",)
        return ("Fog",) if fog <= cloud else ("Cloud",)
