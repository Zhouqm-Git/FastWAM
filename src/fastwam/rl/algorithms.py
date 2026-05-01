"""Algorithm registry for Flow-GSPO ablations on FastWAM."""

from __future__ import annotations

from dataclasses import dataclass

from .advantages import (
    compute_gspo_trajectory_advantages,
    compute_gspo_trajectory_decay_advantages,
)
from .rollout_buffer import RolloutBuffer


@dataclass(frozen=True)
class FlowGSPOVariant:
    """A named Flow-GSPO configuration used in the ablation suite."""

    name: str
    advantage_mode: str
    ratio_mode: str
    description: str


FLOW_GSPO_VARIANTS: dict[str, FlowGSPOVariant] = {
    "traj_chunk": FlowGSPOVariant(
        name="traj_chunk",
        advantage_mode="trajectory",
        ratio_mode="chunk",
        description="Trajectory-level advantage with chunk-level ratio.",
    ),
    "traj_traj": FlowGSPOVariant(
        name="traj_traj",
        advantage_mode="trajectory",
        ratio_mode="trajectory",
        description="Trajectory-level advantage with trajectory-level ratio.",
    ),
}


def resolve_variant(name: str) -> FlowGSPOVariant:
    key = str(name).strip().lower()
    if key not in FLOW_GSPO_VARIANTS:
        raise ValueError(
            f"Unknown Flow-GSPO variant: {name}. "
            f"Expected one of {sorted(FLOW_GSPO_VARIANTS.keys())}."
        )
    return FLOW_GSPO_VARIANTS[key]


def assign_advantages(
    buffer: RolloutBuffer,
    *,
    variant: str,
    trajectory_assignment: str = "uniform",
    gamma: float = 0.99,
) -> None:
    """Assign advantages according to the configured ablation variant."""
    spec = resolve_variant(variant)

    assignment = str(trajectory_assignment).strip().lower()
    if assignment == "uniform":
        compute_gspo_trajectory_advantages(buffer)
        return
    if assignment == "temporal_decay":
        compute_gspo_trajectory_decay_advantages(buffer=buffer, gamma=gamma)
        return

    raise ValueError(
        f"Unsupported trajectory_assignment: {trajectory_assignment}. "
        "Expected one of ['uniform', 'temporal_decay']."
    )
