"""Flow-GSPO advantage computation.

Key difference from PPO: No Critic, no GAE. All advantages are computed via
group reward normalization (same task, multiple trajectories).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .rollout_buffer import RolloutBuffer


def _zero_out_trajectory(traj) -> None:
    traj.trajectory_advantage = 0.0
    for chunk in traj.chunks:
        chunk.advantage = 0.0


def compute_gspo_trajectory_advantages(buffer: RolloutBuffer) -> None:
    """Flow-GSPO trajectory-level advantage (Exp-2, primary method).

    For each group (same task_id):
    1. Collect trajectory_reward.
    2. advantage_i = (R_i - mean) / (std + eps).
    3. Uniformly assign to all chunks within the trajectory.

    Corresponds to chat-flow-gspo.md Section 4.1-4.2:
        mu_R = mean_j R_j^traj
        sigma_R = std_j R_j^traj
        A_hat_i^traj = (R_i^traj - mu_R) / (sigma_R + eps)
        A_hat_{i,t} = A_hat_i^traj  (uniform assignment)
    """
    groups: dict[str, list] = defaultdict(list)
    for traj in buffer.trajectories:
        groups[traj.task_id].append(traj)

    for task_id, group in groups.items():
        rewards = [t.trajectory_reward for t in group]
        std_r = np.std(rewards)

        # Zero-variance filter: all success or all failure -> no information
        if std_r < 1e-6:
            for traj in group:
                _zero_out_trajectory(traj)
            continue

        mean_r = np.mean(rewards)
        for traj in group:
            adv = (traj.trajectory_reward - mean_r) / (std_r + 1e-8)
            traj.trajectory_advantage = float(adv)
            for chunk in traj.chunks:
                chunk.advantage = float(adv)


def compute_gspo_block_advantages(buffer: RolloutBuffer, gamma: float = 1.0) -> None:
    """Flow-GSPO block-level advantage (Exp-1, baseline comparison).

    Direct transfer of OmniVLA-RL original approach:
    1. For each (task_id, chunk_index), collect block reward from different trajectories.
    2. R_total(A_{i,t}, s_t) = sum gamma^h * R(s_t, a_{t,i,h}).
    3. Normalize by group.

    Note: This requires block reward to have discernible variance.
    """
    groups: dict[tuple[str, int], list] = defaultdict(list)
    for traj in buffer.trajectories:
        for t, chunk in enumerate(traj.chunks):
            block_reward = sum(gamma**h * r for h, r in enumerate(chunk.chunk_rewards))
            groups[(traj.task_id, t)].append((chunk, block_reward))

    for key, items in groups.items():
        rewards = [r for _, r in items]
        std_r = np.std(rewards)
        if std_r < 1e-6:
            for chunk, _ in items:
                chunk.advantage = 0.0
            continue
        mean_r = np.mean(rewards)
        for chunk, block_reward in items:
            chunk.advantage = float((block_reward - mean_r) / (std_r + 1e-8))

    for traj in buffer.trajectories:
        traj.trajectory_advantage = 0.0


def compute_gspo_trajectory_decay_advantages(
    buffer: RolloutBuffer, gamma: float = 0.99
) -> None:
    """Flow-GSPO temporal decay assignment (Section 4.2 option 2).

    A_{i,t} = gamma^{N_i - t} * A_i^traj
    Later chunks receive larger advantage (closer to task completion).
    """
    groups: dict[str, list] = defaultdict(list)
    for traj in buffer.trajectories:
        groups[traj.task_id].append(traj)

    for task_id, group in groups.items():
        rewards = [t.trajectory_reward for t in group]
        std_r = np.std(rewards)
        if std_r < 1e-6:
            for traj in group:
                _zero_out_trajectory(traj)
            continue
        mean_r = np.mean(rewards)
        for traj in group:
            traj_adv = (traj.trajectory_reward - mean_r) / (std_r + 1e-8)
            traj.trajectory_advantage = float(traj_adv)
            N = len(traj.chunks)
            for t, chunk in enumerate(traj.chunks):
                chunk.advantage = float((gamma ** (N - t)) * traj_adv)
