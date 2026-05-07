"""Rollout and optimization metrics for FastWAM RL."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .rollout_buffer import RolloutBuffer


def compute_rollout_metrics(buffer: RolloutBuffer) -> dict[str, float]:
    trajectories = buffer.trajectories
    if not trajectories:
        return {
            "num_trajectories": 0.0,
            "num_chunks": 0.0,
            "success_rate": 0.0,
            "trajectory_reward_mean": 0.0,
            "trajectory_reward_std": 0.0,
            "trajectory_advantage_mean": 0.0,
            "trajectory_advantage_std": 0.0,
            "chunk_advantage_mean": 0.0,
            "chunk_advantage_std": 0.0,
            "chunk_return_mean": 0.0,
            "chunk_return_std": 0.0,
            "chunks_per_trajectory_mean": 0.0,
            "chunks_per_trajectory_max": 0.0,
            "informative_group_fraction": 0.0,
            "group_reward_std_mean": 0.0,
        }

    traj_rewards = np.asarray([t.trajectory_reward for t in trajectories], dtype=np.float32)
    traj_advantages = np.asarray([t.trajectory_advantage for t in trajectories], dtype=np.float32)
    chunk_advantages = np.asarray(
        [chunk.advantage for traj in trajectories for chunk in traj.chunks],
        dtype=np.float32,
    )
    chunk_returns = np.asarray(
        [sum(chunk.chunk_rewards) for traj in trajectories for chunk in traj.chunks],
        dtype=np.float32,
    )
    chunks_per_trajectory = np.asarray([len(t.chunks) for t in trajectories], dtype=np.float32)

    groups: dict[str, list[float]] = defaultdict(list)
    for traj in trajectories:
        groups[traj.group_id or traj.task_id].append(traj.trajectory_reward)
    group_stds = np.asarray([np.std(v, dtype=np.float32) for v in groups.values()], dtype=np.float32)
    informative_fraction = float((group_stds > 1e-6).mean()) if group_stds.size > 0 else 0.0

    # Per-task success rate breakdown
    per_task: dict[str, list[bool]] = defaultdict(list)
    for traj in trajectories:
        per_task[traj.task_id].append(traj.success)
    per_task_metrics: dict[str, float] = {}
    for tid, successes in sorted(per_task.items()):
        per_task_metrics[f"per_task/{tid}/success_rate"] = float(np.mean(successes))
        per_task_metrics[f"per_task/{tid}/num_trajectories"] = float(len(successes))

    def _safe_stat(array: np.ndarray, reducer, default: float = 0.0) -> float:
        if array.size == 0:
            return default
        return float(reducer(array))

    result = {
        "num_trajectories": float(len(trajectories)),
        "num_chunks": float(sum(len(t.chunks) for t in trajectories)),
        "success_rate": float(np.mean([float(t.success) for t in trajectories])),
        "trajectory_reward_mean": _safe_stat(traj_rewards, np.mean),
        "trajectory_reward_std": _safe_stat(traj_rewards, np.std),
        "trajectory_advantage_mean": _safe_stat(traj_advantages, np.mean),
        "trajectory_advantage_std": _safe_stat(traj_advantages, np.std),
        "chunk_advantage_mean": _safe_stat(chunk_advantages, np.mean),
        "chunk_advantage_std": _safe_stat(chunk_advantages, np.std),
        "chunk_return_mean": _safe_stat(chunk_returns, np.mean),
        "chunk_return_std": _safe_stat(chunk_returns, np.std),
        "chunks_per_trajectory_mean": _safe_stat(chunks_per_trajectory, np.mean),
        "chunks_per_trajectory_max": _safe_stat(chunks_per_trajectory, np.max),
        "informative_group_fraction": informative_fraction,
        "group_reward_std_mean": _safe_stat(group_stds, np.mean),
    }
    result.update(per_task_metrics)
    return result
