"""GSPO objective computation for FastWAM RL ablations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .algorithms import resolve_variant
from .rollout_buffer import RolloutBuffer


@dataclass
class ObjectiveResult:
    loss: torch.Tensor
    metrics: dict[str, float]


def _zero_loss_like(model: torch.nn.Module) -> torch.Tensor:
    return next(model.parameters()).sum() * 0.0


def _approx_kl_from_log_ratio(log_ratio: torch.Tensor) -> torch.Tensor:
    ratio = torch.exp(log_ratio)
    return ratio - 1.0 - log_ratio


def _compute_chunk_level_objective(
    model: torch.nn.Module,
    buffer: RolloutBuffer,
    *,
    clip_range: float,
    kl_coef: float,
    sigma_max: float,
    num_inference_steps: int,
    sigma_shift: Optional[float],
) -> ObjectiveResult:
    objective_terms = []
    ratios = []
    log_ratios = []
    kls = []

    for chunk in buffer.get_chunks_with_advantage():
        old_log_prob = chunk.old_log_prob.to(device=model.device, dtype=torch.float32)
        new_log_prob = model.compute_logprob_from_chain(
            chain=chunk.chain,
            context=chunk.context,
            context_mask=chunk.context_mask,
            input_image=chunk.obs_image,
            proprio=chunk.obs_proprio,
            sigma_max=sigma_max,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            exec_horizon=chunk.exec_horizon,
        ).float()
        log_ratio = (new_log_prob - old_log_prob) / float(chunk.block_size)
        ratio = torch.exp(log_ratio)
        advantage = torch.tensor(float(chunk.advantage), device=model.device, dtype=torch.float32)

        unclipped = ratio * advantage
        clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantage
        objective_terms.append(torch.minimum(unclipped, clipped))
        ratios.append(ratio)
        log_ratios.append(log_ratio)
        kls.append(_approx_kl_from_log_ratio(log_ratio))

    if not objective_terms:
        return ObjectiveResult(
            loss=_zero_loss_like(model),
            metrics={
                "num_objective_terms": 0.0,
                "policy_objective": 0.0,
                "approx_kl": 0.0,
                "clip_fraction": 0.0,
                "ratio_mean": 1.0,
                "ratio_min": 1.0,
                "ratio_max": 1.0,
                "log_ratio_mean": 0.0,
            },
        )

    policy_objective = torch.stack(objective_terms).mean()
    approx_kl = torch.stack(kls).mean()
    ratio_tensor = torch.stack(ratios)
    log_ratio_tensor = torch.stack(log_ratios)
    clipped_mask = (ratio_tensor > (1.0 + clip_range)) | (ratio_tensor < (1.0 - clip_range))
    clip_fraction = clipped_mask.float().mean()
    loss = -(policy_objective - kl_coef * approx_kl)

    return ObjectiveResult(
        loss=loss,
        metrics={
            "num_objective_terms": float(ratio_tensor.numel()),
            "policy_objective": float(policy_objective.detach().item()),
            "approx_kl": float(approx_kl.detach().item()),
            "clip_fraction": float(clip_fraction.detach().item()),
            "ratio_mean": float(ratio_tensor.detach().mean().item()),
            "ratio_min": float(ratio_tensor.detach().min().item()),
            "ratio_max": float(ratio_tensor.detach().max().item()),
            "log_ratio_mean": float(log_ratio_tensor.detach().mean().item()),
        },
    )


def _compute_trajectory_level_objective(
    model: torch.nn.Module,
    buffer: RolloutBuffer,
    *,
    clip_range: float,
    kl_coef: float,
    sigma_max: float,
    num_inference_steps: int,
    sigma_shift: Optional[float],
) -> ObjectiveResult:
    objective_terms = []
    ratios = []
    log_ratios = []
    kls = []
    chunks_per_trajectory = []

    for traj in buffer.get_trajectories_with_advantage():
        if not traj.chunks:
            continue

        new_total = None
        old_total = None
        total_block_size = 0.0
        for chunk in traj.chunks:
            old_log_prob = chunk.old_log_prob.to(device=model.device, dtype=torch.float32)
            new_log_prob = model.compute_logprob_from_chain(
                chain=chunk.chain,
                context=chunk.context,
                context_mask=chunk.context_mask,
                input_image=chunk.obs_image,
                proprio=chunk.obs_proprio,
                sigma_max=sigma_max,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                exec_horizon=chunk.exec_horizon,
            ).float()
            new_total = new_log_prob if new_total is None else new_total + new_log_prob
            old_total = old_log_prob if old_total is None else old_total + old_log_prob
            total_block_size += float(chunk.block_size)

        if new_total is None or old_total is None or total_block_size <= 0.0:
            continue

        log_ratio = (new_total - old_total) / total_block_size
        ratio = torch.exp(log_ratio)
        advantage = torch.tensor(float(traj.trajectory_advantage), device=model.device, dtype=torch.float32)

        unclipped = ratio * advantage
        clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantage
        objective_terms.append(torch.minimum(unclipped, clipped))
        ratios.append(ratio)
        log_ratios.append(log_ratio)
        kls.append(_approx_kl_from_log_ratio(log_ratio))
        chunks_per_trajectory.append(len(traj.chunks))

    if not objective_terms:
        return ObjectiveResult(
            loss=_zero_loss_like(model),
            metrics={
                "num_objective_terms": 0.0,
                "policy_objective": 0.0,
                "approx_kl": 0.0,
                "clip_fraction": 0.0,
                "ratio_mean": 1.0,
                "ratio_min": 1.0,
                "ratio_max": 1.0,
                "log_ratio_mean": 0.0,
                "chunks_per_trajectory_mean": 0.0,
            },
        )

    policy_objective = torch.stack(objective_terms).mean()
    approx_kl = torch.stack(kls).mean()
    ratio_tensor = torch.stack(ratios)
    log_ratio_tensor = torch.stack(log_ratios)
    clipped_mask = (ratio_tensor > (1.0 + clip_range)) | (ratio_tensor < (1.0 - clip_range))
    clip_fraction = clipped_mask.float().mean()
    loss = -(policy_objective - kl_coef * approx_kl)

    return ObjectiveResult(
        loss=loss,
        metrics={
            "num_objective_terms": float(ratio_tensor.numel()),
            "policy_objective": float(policy_objective.detach().item()),
            "approx_kl": float(approx_kl.detach().item()),
            "clip_fraction": float(clip_fraction.detach().item()),
            "ratio_mean": float(ratio_tensor.detach().mean().item()),
            "ratio_min": float(ratio_tensor.detach().min().item()),
            "ratio_max": float(ratio_tensor.detach().max().item()),
            "log_ratio_mean": float(log_ratio_tensor.detach().mean().item()),
            "chunks_per_trajectory_mean": float(sum(chunks_per_trajectory) / len(chunks_per_trajectory)),
        },
    )


def compute_gspo_objective(
    model: torch.nn.Module,
    buffer: RolloutBuffer,
    *,
    variant: str,
    clip_range: float,
    kl_coef: float,
    sigma_max: float,
    num_inference_steps: int,
    sigma_shift: Optional[float],
) -> ObjectiveResult:
    """Compute the configured GSPO objective for one rollout buffer."""
    spec = resolve_variant(variant)
    if spec.ratio_mode == "chunk":
        return _compute_chunk_level_objective(
            model=model,
            buffer=buffer,
            clip_range=clip_range,
            kl_coef=kl_coef,
            sigma_max=sigma_max,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
        )
    if spec.ratio_mode == "trajectory":
        return _compute_trajectory_level_objective(
            model=model,
            buffer=buffer,
            clip_range=clip_range,
            kl_coef=kl_coef,
            sigma_max=sigma_max,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
        )
    raise ValueError(f"Unsupported ratio mode: {spec.ratio_mode}")
