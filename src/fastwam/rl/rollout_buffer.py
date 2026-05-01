"""Rollout data structures for Flow-GSPO.

Key differences from PPO buffer:
- No value (V(s)) storage — GSPO uses group reward normalization, not a critic.
- No GAE-related data.
- Stores the full denoising chain per action block for log-prob recomputation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class ChunkData:
    """Data from a single model inference (one action block's denoising).

    Attributes:
        obs_image: Observation image, shape [1, 3, H, W].
        obs_proprio: Proprioceptive state, shape [1, proprio_dim] or None.
        context: Text context embeddings, shape [1, L, text_dim] or None.
        context_mask: Context mask, shape [1, L] or None.
        chain: Full denoising trajectory, shape [K+1, H_action, action_dim].
            chain[0] = initial noise, chain[K] = final denoised action.
        old_log_prob: Block log-likelihood log pi_old(A_t|s_t), scalar.
        block_size: H * K, used for GSPO ratio normalization.
        action: Final denoised action, shape [H_action, action_dim].
        chunk_rewards: Per-step rewards within this action block.
        done: Whether the episode terminated during this chunk.
        task_id: Task identifier for group advantage normalization.
        group_id: Reset-aligned rollout group identifier for GSPO normalization.
        reset_id: Reset identifier within a task.
        initial_state_index: Index of the initial state used for this rollout.
        trajectory_id: Parent trajectory identifier.
        chunk_index: Chunk index within the trajectory.
        env_step_start: Inclusive environment-step index where this chunk starts.
        env_step_end: Exclusive environment-step index where this chunk ends.
        rollout_seed: Sampling seed used to start this chunk, if configured.
        rollout_time: Trajectory-relative collection timestamp in seconds.
        task_suite_name: Benchmark suite name.
        reward_components: Named reward breakdowns for later analysis.
        task_description: Natural language task description.
        advantage: Filled by advantages.py. Trajectory-level GSPO advantage.
    """

    obs_image: torch.Tensor
    obs_proprio: Optional[torch.Tensor]
    context: Optional[torch.Tensor]
    context_mask: Optional[torch.Tensor]
    chain: torch.Tensor
    old_log_prob: torch.Tensor
    block_size: int
    action: torch.Tensor
    chunk_rewards: list[float] = field(default_factory=list)
    done: bool = False
    task_id: str = ""
    group_id: str = ""
    reset_id: str = ""
    initial_state_index: int = -1
    trajectory_id: str = ""
    chunk_index: int = -1
    env_step_start: int = 0
    env_step_end: int = 0
    rollout_seed: Optional[int] = None
    rollout_time: float = 0.0
    task_suite_name: str = ""
    reward_components: dict[str, float] = field(default_factory=dict)
    task_description: str = ""
    advantage: float = 0.0


@dataclass
class TrajectoryData:
    """A complete trajectory composed of multiple chunks.

    Attributes:
        task_id: Task identifier (same for all chunks in a group).
        task_description: Natural language task description.
        group_id: Reset-aligned rollout group identifier for GSPO normalization.
        reset_id: Reset identifier within a task.
        initial_state_index: Index of the initial state used for this rollout.
        trajectory_id: Unique trajectory identifier.
        chunks: Ordered list of ChunkData making up this trajectory.
        trajectory_reward: Terminal reward (1.0 success, 0.0 failure for LIBERO).
        success: Whether the task was completed successfully.
        rollout_seed: Base rollout seed, if configured.
        rollout_time: Collection duration in seconds.
        task_suite_name: Benchmark suite name.
        reward_components: Named reward breakdowns for later analysis.
        trajectory_advantage: Trajectory-level advantage assigned by GSPO.
    """

    task_id: str
    task_description: str
    chunks: list[ChunkData]
    trajectory_reward: float
    success: bool
    group_id: str = ""
    reset_id: str = ""
    initial_state_index: int = -1
    trajectory_id: str = ""
    rollout_seed: Optional[int] = None
    rollout_time: float = 0.0
    task_suite_name: str = ""
    reward_components: dict[str, float] = field(default_factory=dict)
    trajectory_advantage: float = 0.0


class RolloutBuffer:
    """Manages a rollout batch for Flow-GSPO training.

    Flow-GSPO data flow:
    1. collect_group() -> G TrajectoryData (same task)
    2. compute_gspo_advantages() -> assign advantage per research plan
    3. get_all_chunks() -> flatten chunks with advantage, for training
    """

    def __init__(self) -> None:
        self.trajectories: list[TrajectoryData] = []

    def add_trajectory(self, traj: TrajectoryData) -> None:
        self.trajectories.append(traj)

    def extend(self, other: "RolloutBuffer") -> None:
        self.trajectories.extend(other.trajectories)

    def get_all_chunks(self) -> list[ChunkData]:
        """Flatten all trajectory chunks, return list with advantage filled."""
        return [c for t in self.trajectories for c in t.chunks]

    def get_chunks_with_advantage(self) -> list[ChunkData]:
        """Return only chunks with non-zero advantage (filters zero-variance groups)."""
        return [c for c in self.get_all_chunks() if c.advantage != 0.0]

    def get_trajectories_with_advantage(self) -> list[TrajectoryData]:
        """Return only trajectories with non-zero trajectory advantage."""
        return [t for t in self.trajectories if t.trajectory_advantage != 0.0]

    def __len__(self) -> int:
        return len(self.trajectories)
