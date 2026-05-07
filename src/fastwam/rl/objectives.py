"""GSPO objective computation for FastWAM RL ablations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

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


def _ratio_value_from_log_ratio(log_ratio: float) -> float:
    """Mirror torch.exp behavior for scalar metrics without Python overflow."""
    ratio = torch.exp(torch.tensor(log_ratio, dtype=torch.float64))
    return float(ratio.item())


def _clip_branch_active(*, ratio: float, advantage: float, clip_range: float) -> bool:
    """Whether PPO-style min(rA, clip(r)A) selects the clipped branch."""
    upper = 1.0 + clip_range
    lower = 1.0 - clip_range
    if advantage >= 0.0:
        return ratio > upper
    return ratio < lower


def _move_tree_to_device(tree, device: str):
    """Recursively move nested tensor containers to the target device."""
    if torch.is_tensor(tree):
        return tree.to(device=device, non_blocking=True)
    if isinstance(tree, list):
        return [_move_tree_to_device(item, device) for item in tree]
    if isinstance(tree, tuple):
        return tuple(_move_tree_to_device(item, device) for item in tree)
    if isinstance(tree, dict):
        return {key: _move_tree_to_device(value, device) for key, value in tree.items()}
    return tree


def _compute_chunk_level_objective(
    model: torch.nn.Module,
    buffer: RolloutBuffer,
    *,
    backward_fn: Callable[[torch.Tensor], None],
    clip_range: float,
    kl_coef: float,
    sigma_max: float,
    num_inference_steps: int,
    sigma_shift: Optional[float],
) -> ObjectiveResult:
    """Chunk-level GSPO objective with per-chunk gradient accumulation.

    Each chunk is processed independently: forward pass -> backward_fn ->
    release computation graph.  This keeps peak memory proportional to a
    single chunk rather than the full rollout buffer.
    """
    chunks = list(buffer.get_chunks_with_advantage())
    if not chunks:
        return ObjectiveResult(
            loss=_zero_loss_like(model),
            metrics={
                "num_objective_terms": 0.0,
                "policy_objective": 0.0,
                "approx_kl": 0.0,
                "clip_fraction": 0.0,
                "clip_branch_active_fraction": 0.0,
                "ratio_mean": 1.0,
                "ratio_min": 1.0,
                "ratio_max": 1.0,
                "log_ratio_mean": 0.0,
            },
        )

    n_valid = len(chunks)
    objective_terms: list[torch.Tensor] = []
    ratios: list[torch.Tensor] = []
    log_ratios: list[torch.Tensor] = []
    kls: list[torch.Tensor] = []
    clip_branch_active_flags: list[float] = []

    for chunk in chunks:
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
        ratio_value = float(ratio.detach().item())
        clip_branch_active_flags.append(
            1.0
            if _clip_branch_active(
                ratio=ratio_value,
                advantage=float(chunk.advantage),
                clip_range=clip_range,
            )
            else 0.0
        )

        unclipped = ratio * advantage
        clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantage
        chunk_objective = torch.minimum(unclipped, clipped)
        chunk_kl = _approx_kl_from_log_ratio(log_ratio)

        # Per-chunk backward via Accelerator for proper mixed-precision scaling.
        micro_loss = -(chunk_objective - kl_coef * chunk_kl) / n_valid
        backward_fn(micro_loss)

        objective_terms.append(chunk_objective.detach())
        ratios.append(ratio.detach())
        log_ratios.append(log_ratio.detach())
        kls.append(chunk_kl.detach())

    # Compute metrics from detached values.
    policy_objective = torch.stack(objective_terms).mean()
    approx_kl = torch.stack(kls).mean()
    ratio_tensor = torch.stack(ratios)
    log_ratio_tensor = torch.stack(log_ratios)
    clip_branch_active_fraction = sum(clip_branch_active_flags) / n_valid
    clipped_mask = (ratio_tensor > (1.0 + clip_range)) | (ratio_tensor < (1.0 - clip_range))
    clip_fraction = clipped_mask.float().mean()

    # Gradients already accumulated via per-chunk backward_fn() calls.
    # Return a dummy zero loss; the caller should NOT call backward() again.
    loss = _zero_loss_like(model)

    return ObjectiveResult(
        loss=loss,
        metrics={
            "num_objective_terms": float(ratio_tensor.numel()),
            "policy_objective": float(policy_objective.item()),
            "approx_kl": float(approx_kl.item()),
            "clip_fraction": float(clip_fraction.item()),
            "clip_branch_active_fraction": float(clip_branch_active_fraction),
            "ratio_mean": float(ratio_tensor.mean().item()),
            "ratio_min": float(ratio_tensor.min().item()),
            "ratio_max": float(ratio_tensor.max().item()),
            "log_ratio_mean": float(log_ratio_tensor.mean().item()),
        },
    )


def _compute_trajectory_level_objective(
    model: torch.nn.Module,
    buffer: RolloutBuffer,
    *,
    backward_fn: Callable[[torch.Tensor], None],
    clip_range: float,
    kl_coef: float,
    sigma_max: float,
    num_inference_steps: int,
    sigma_shift: Optional[float],
) -> ObjectiveResult:
    """Trajectory-level GSPO objective with two-pass gradient accumulation.

    **Memory problem solved:** A failed trajectory may contain ~40 chunks.
    Computing all log-probs with gradients and keeping them alive for the
    ratio computation would require ~70 GB.  Instead we use two passes:

    1. **Pass 1 (no-grad):** Compute all log-probs without gradients to
       obtain detached log-ratios and the analytical gradient coefficient
       ``c_i = d(loss_i) / d(log_ratio_i)`` for each trajectory.
    2. **Pass 2 (with-grad, per chunk):** Re-compute each chunk's log-prob
       with gradients, then immediately call
       ``backward_fn(c_i / block_size_i * new_log_prob_j)`` to accumulate
       gradients and release the computation graph.

    Peak memory stays proportional to a single chunk's forward+backward
    through the ActionDiT, not the full trajectory.
    """

    _empty_metrics = {
        "num_objective_terms": 0.0,
        "policy_objective": 0.0,
        "approx_kl": 0.0,
        "clip_fraction": 0.0,
        "clip_branch_active_fraction": 0.0,
        "ratio_mean": 1.0,
        "ratio_min": 1.0,
        "ratio_max": 1.0,
        "log_ratio_mean": 0.0,
        "chunks_per_trajectory_mean": 0.0,
    }

    trajectories = list(buffer.get_trajectories_with_advantage())
    if not trajectories:
        return ObjectiveResult(loss=_zero_loss_like(model), metrics=_empty_metrics)

    # ----------------------------------------------------------------
    # Pass 1 (no-grad): compute detached metrics & gradient coefficients
    # ----------------------------------------------------------------
    traj_data: list[dict] = []
    for traj in trajectories:
        if not traj.chunks:
            continue

        new_total = 0.0
        old_total = 0.0
        total_block_size = 0.0
        chunk_caches = []

        with torch.no_grad():
            for chunk in traj.chunks:
                cache_bundle = model.build_action_video_kv_cache(
                    input_image=chunk.obs_image,
                    context=chunk.context,
                    context_mask=chunk.context_mask,
                    proprio=chunk.obs_proprio,
                    action_seq_len=chunk.chain.shape[-2],
                )
                new_lp = model.compute_logprob_from_chain(
                    chain=chunk.chain,
                    context=chunk.context,
                    context_mask=chunk.context_mask,
                    input_image=chunk.obs_image,
                    proprio=chunk.obs_proprio,
                    video_kv_cache=cache_bundle,
                    sigma_max=sigma_max,
                    num_inference_steps=num_inference_steps,
                    sigma_shift=sigma_shift,
                    exec_horizon=chunk.exec_horizon,
                ).float().item()
                old_lp = chunk.old_log_prob.to(
                    device=model.device, dtype=torch.float32
                ).item()
                new_total += new_lp
                old_total += old_lp
                total_block_size += float(chunk.block_size)
                chunk_caches.append(_move_tree_to_device(cache_bundle, "cpu"))

        if total_block_size <= 0.0:
            continue

        log_ratio_val = (new_total - old_total) / total_block_size
        ratio_val = _ratio_value_from_log_ratio(log_ratio_val)
        advantage_val = float(traj.trajectory_advantage)
        ratio_out_of_range = ratio_val > (1.0 + clip_range) or ratio_val < (1.0 - clip_range)
        clip_branch_active = _clip_branch_active(
            ratio=ratio_val,
            advantage=advantage_val,
            clip_range=clip_range,
        )

        unclipped_val = ratio_val * advantage_val
        clamped = max(1.0 - clip_range, min(1.0 + clip_range, ratio_val))
        clipped_val = clamped * advantage_val
        obj_val = min(unclipped_val, clipped_val)
        kl_val = ratio_val - 1.0 - log_ratio_val

        # Analytical d(loss_i)/d(log_ratio_i), without the 1/n_valid factor
        # which will be applied in pass 2 once n_valid is known.
        #
        #   loss_i = -(objective_i - kl_coef * kl_i) / n_valid
        #
        # Unclipped: d(obj)/d(lr) = ratio * advantage;  d(kl)/d(lr) = ratio - 1
        #   => dl_dr = -(advantage * ratio - kl_coef * (ratio - 1))
        #
        # Clipped:   d(obj)/d(lr) = 0
        #   => dl_dr = kl_coef * (ratio - 1)
        if clip_branch_active:
            dl_dr_raw = kl_coef * (ratio_val - 1.0)
        else:
            dl_dr_raw = -(advantage_val * ratio_val - kl_coef * (ratio_val - 1.0))

        traj_data.append({
            "chunks": traj.chunks,
            "total_block_size": total_block_size,
            "log_ratio": log_ratio_val,
            "ratio": ratio_val,
            "advantage": advantage_val,
            "objective": obj_val,
            "kl": kl_val,
            "dl_dr_raw": dl_dr_raw,
            "ratio_out_of_range": ratio_out_of_range,
            "clip_branch_active": clip_branch_active,
            "n_chunks": len(traj.chunks),
            "chunk_caches": chunk_caches,
        })

    if not traj_data:
        return ObjectiveResult(loss=_zero_loss_like(model), metrics=_empty_metrics)

    n_valid = len(traj_data)

    # ----------------------------------------------------------------
    # Pass 2 (with-grad, per chunk): accumulate gradients
    # ----------------------------------------------------------------
    for info in traj_data:
        # Per-chunk coefficient:
        #   d(loss_i) / d(new_log_prob_j)
        #     = [d(loss_i) / d(log_ratio_i)] / total_block_size_i
        #     = [dl_dr_raw / n_valid] / total_block_size_i
        chunk_coeff = (info["dl_dr_raw"] / n_valid) / info["total_block_size"]

        for idx, (chunk, cache_cpu) in enumerate(zip(info["chunks"], info["chunk_caches"], strict=True)):
            video_kv_cache = _move_tree_to_device(cache_cpu, model.device)
            new_log_prob = model.compute_logprob_from_chain(
                chain=chunk.chain,
                context=chunk.context,
                context_mask=chunk.context_mask,
                input_image=chunk.obs_image,
                proprio=chunk.obs_proprio,
                video_kv_cache=video_kv_cache,
                sigma_max=sigma_max,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                exec_horizon=chunk.exec_horizon,
            ).float()
            # Accumulate gradient and release computation graph immediately.
            backward_fn(chunk_coeff * new_log_prob)
            info["chunk_caches"][idx] = None
            del video_kv_cache
        info["chunk_caches"].clear()

    # ----------------------------------------------------------------
    # Compute metrics from pass-1 detached values
    # ----------------------------------------------------------------
    objectives = [info["objective"] for info in traj_data]
    kl_values = [info["kl"] for info in traj_data]
    ratio_values = [info["ratio"] for info in traj_data]
    log_ratio_values = [info["log_ratio"] for info in traj_data]
    n_clipped = sum(1 for info in traj_data if info["ratio_out_of_range"])
    n_clip_branch_active = sum(1 for info in traj_data if info["clip_branch_active"])

    return ObjectiveResult(
        loss=_zero_loss_like(model),
        metrics={
            "num_objective_terms": float(n_valid),
            "policy_objective": sum(objectives) / n_valid,
            "approx_kl": sum(kl_values) / n_valid,
            # PPO-style clip_fraction tracks ratio out-of-range incidence,
            # regardless of the advantage sign.
            "clip_fraction": float(n_clipped) / n_valid,
            "clip_branch_active_fraction": float(n_clip_branch_active) / n_valid,
            "ratio_mean": sum(ratio_values) / n_valid,
            "ratio_min": min(ratio_values),
            "ratio_max": max(ratio_values),
            "log_ratio_mean": sum(log_ratio_values) / n_valid,
            "chunks_per_trajectory_mean": (
                sum(info["n_chunks"] for info in traj_data) / n_valid
            ),
        },
    )


def compute_gspo_objective(
    model: torch.nn.Module,
    buffer: RolloutBuffer,
    *,
    backward_fn: Callable[[torch.Tensor], None],
    variant: str,
    clip_range: float,
    kl_coef: float,
    sigma_max: float,
    num_inference_steps: int,
    sigma_shift: Optional[float],
) -> ObjectiveResult:
    """Compute the configured GSPO objective for one rollout buffer.

    Gradients are accumulated into ``model.parameters()`` via per-chunk or
    per-trajectory calls to ``backward_fn`` (which should be
    ``accelerator.backward`` for proper mixed-precision loss scaling).

    The returned ``loss`` is a dummy zero tensor with no grad_fn; the
    caller should **not** call backward() on it — only clip_grad_norm_()
    and optimizer.step().
    """
    spec = resolve_variant(variant)
    if spec.ratio_mode == "chunk":
        return _compute_chunk_level_objective(
            model=model,
            buffer=buffer,
            backward_fn=backward_fn,
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
            backward_fn=backward_fn,
            clip_range=clip_range,
            kl_coef=kl_coef,
            sigma_max=sigma_max,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
        )
    raise ValueError(f"Unsupported ratio mode: {spec.ratio_mode}")
