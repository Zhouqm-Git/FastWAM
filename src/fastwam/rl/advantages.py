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
        groups[traj.group_id or traj.task_id].append(traj)

    for _, group in groups.items():
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


def compute_gspo_trajectory_decay_advantages(
    buffer: RolloutBuffer, gamma: float = 0.99
) -> None:
    """Flow-GSPO temporal decay assignment (Section 4.2 option 2).

    A_{i,t} = gamma^{N_i - t} * A_i^traj
    Later chunks receive larger advantage (closer to task completion).
    """
    groups: dict[str, list] = defaultdict(list)
    for traj in buffer.trajectories:
        groups[traj.group_id or traj.task_id].append(traj)

    for _, group in groups.items():
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
