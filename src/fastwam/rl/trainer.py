"""Online RL trainer for Flow-GSPO ablations on FastWAM."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

import torch
from accelerate import Accelerator
from hydra.utils import instantiate
from omegaconf import DictConfig

from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from fastwam.utils.fs import ensure_dir
from fastwam.utils.logging_config import get_logger
from fastwam.utils.pytorch_utils import set_global_seed

from .algorithms import assign_advantages, resolve_variant
from .metric_logger import RLMetricLogger
from .metrics import compute_rollout_metrics
from .objectives import compute_gspo_objective
from .rollout_buffer import RolloutBuffer
from .rollout_collector import RolloutCollector

logger = get_logger(__name__)


def _resolve_dataset_stats_path(cfg: DictConfig) -> Path:
    explicit = cfg.EVALUATION.get("dataset_stats_path")
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(os.path.expanduser(os.path.expandvars(str(explicit)))))

    ckpt = cfg.get("ckpt")
    if ckpt is not None:
        ckpt_path = Path(os.path.expanduser(os.path.expandvars(str(ckpt))))
        for parent in list(ckpt_path.parents)[:4]:
            candidates.append(parent / "dataset_stats.json")

    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved
    raise FileNotFoundError(
        "Failed to locate dataset_stats.json. "
        "Please set `EVALUATION.dataset_stats_path` explicitly."
    )


def _load_model_checkpoint(model: torch.nn.Module, ckpt: str) -> None:
    model.load_checkpoint(ckpt)
    logger.info("Loaded RL initialization checkpoint: %s", ckpt)


class FastWAMRLTrainer:
    """Single-node Flow-GSPO trainer with clean ablation switches."""

    def __init__(self, model: torch.nn.Module, cfg: DictConfig):
        self.cfg = cfg
        self.model = model
        self.output_dir = str(cfg.output_dir)
        self.variant = resolve_variant(str(cfg.rl.variant))
        self.mixed_precision = str(cfg.mixed_precision).strip().lower()
        self.seed = int(cfg.seed)
        self.max_updates = int(cfg.rl.max_updates)
        self.group_size = int(cfg.rl.group_size)
        self.task_batch_size = int(cfg.rl.task_batch_size)
        self.num_optimization_epochs = int(cfg.rl.num_optimization_epochs)
        self.clip_range = float(cfg.rl.clip_range)
        self.kl_coef = float(cfg.rl.kl_coef)
        self.learning_rate = float(cfg.rl.learning_rate)
        self.weight_decay = float(cfg.rl.weight_decay)
        self.max_grad_norm = float(cfg.rl.max_grad_norm)
        self.sigma_max = float(cfg.rl.sigma_max)
        self.num_inference_steps = int(cfg.rl.num_inference_steps)
        self.trajectory_assignment = str(cfg.rl.trajectory_assignment)
        self.advantage_gamma = float(cfg.rl.advantage_gamma)
        self.trainable_scope = str(cfg.rl.trainable_scope)
        self.log_every = int(cfg.rl.log_every)
        self.save_every = int(cfg.rl.save_every)
        self.resume = cfg.rl.get("resume", None)
        self.global_step = 0
        self.task_cursor = 0
        self._task_state_cursors: dict[int, int] = {}
        self._task_runtime_cache: dict[int, dict[str, Any]] = {}

        self.accelerator = Accelerator(mixed_precision=self.mixed_precision)
        if self.accelerator.num_processes != 1:
            raise NotImplementedError(
                "FastWAM RL rollout is currently single-process only. "
                "Keep `num_processes=1` for the RL trainer."
            )

        if self.seed is not None:
            set_global_seed(self.seed, get_worker_init_fn=False)

        self._init_output_dirs()
        self.processor = self._build_processor(cfg)
        self._load_init_checkpoint_if_needed()
        self._configure_trainable_scope()
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_params:
            raise ValueError(
                f"No trainable parameters found for trainable_scope={self.trainable_scope}."
            )
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
        )
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
        self.metric_logger = RLMetricLogger(cfg, output_dir=self.output_dir)
        self._load_resume_if_needed()
        self.task_suite = self._build_task_suite()
        self.task_ids = self._resolve_task_ids()

        logger.info(
            "Initialized RL trainer: variant=%s (%s), task_suite=%s, tasks=%s, group_size=%d, task_batch_size=%d",
            self.variant.name,
            self.variant.description,
            self.cfg.EVALUATION.task_suite_name,
            self.task_ids,
            self.group_size,
            self.task_batch_size,
        )

    def _init_output_dirs(self) -> None:
        ensure_dir(self.output_dir)
        self.checkpoint_root = os.path.join(self.output_dir, "checkpoints")
        self.weights_dir = os.path.join(self.checkpoint_root, "weights")
        self.state_dir = os.path.join(self.checkpoint_root, "state")
        ensure_dir(self.checkpoint_root)
        ensure_dir(self.weights_dir)
        ensure_dir(self.state_dir)

    def _build_processor(self, cfg: DictConfig) -> FastWAMProcessor:
        dataset_stats_path = _resolve_dataset_stats_path(cfg)
        dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
        processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
        processor.set_normalizer_from_stats(dataset_stats)
        logger.info("Using RL dataset stats: %s", dataset_stats_path)
        return processor

    def _load_init_checkpoint_if_needed(self) -> None:
        ckpt = self.cfg.get("ckpt")
        if ckpt is None:
            logger.warning("No `ckpt` provided. RL training will start from current model weights.")
            return
        _load_model_checkpoint(self.model, str(ckpt))

    def _configure_trainable_scope(self) -> None:
        model = self.model
        model.eval()
        model.requires_grad_(False)

        if self.trainable_scope == "action_expert_only":
            model.action_expert.train()
            model.action_expert.requires_grad_(True)
            if getattr(model, "proprio_encoder", None) is not None:
                model.proprio_encoder.train()
                model.proprio_encoder.requires_grad_(True)
            return

        if self.trainable_scope == "action_expert_and_mot":
            model.action_expert.train()
            model.action_expert.requires_grad_(True)
            model.mot.train()
            model.mot.requires_grad_(True)
            if getattr(model, "proprio_encoder", None) is not None:
                model.proprio_encoder.train()
                model.proprio_encoder.requires_grad_(True)
            return

        if self.trainable_scope == "full":
            model.train()
            model.requires_grad_(True)
            return

        raise ValueError(
            f"Unsupported trainable_scope: {self.trainable_scope}. "
            "Expected one of ['action_expert_only', 'action_expert_and_mot', 'full']."
        )

    def _build_task_suite(self):
        from libero.libero import benchmark

        benchmark_dict = benchmark.get_benchmark_dict()
        return benchmark_dict[self.cfg.EVALUATION.task_suite_name]()

    def _resolve_task_ids(self) -> list[int]:
        configured = self.cfg.rl.get("task_ids")
        if configured is None:
            return list(range(int(self.task_suite.n_tasks)))
        return [int(task_id) for task_id in configured]

    def _get_task_runtime(self, task_id: int) -> dict[str, Any]:
        if task_id in self._task_runtime_cache:
            return self._task_runtime_cache[task_id]

        from experiments.libero.libero_utils import LIBERO_ENV_RESOLUTION, get_libero_env

        task = self.task_suite.get_task(task_id)
        initial_states = list(self.task_suite.get_task_init_states(task_id))
        env, task_description = get_libero_env(
            task=task,
            resolution=LIBERO_ENV_RESOLUTION,
            seed=self.seed,
            env_num=1,
        )

        action_horizon_cfg = self.cfg.EVALUATION.get("action_horizon", None)
        if action_horizon_cfg is None:
            action_horizon = int(self.cfg.data.train.num_frames) - 1
        else:
            action_horizon = int(action_horizon_cfg)

        video_size = self.cfg.data.train.get("video_size", [224, 224])
        input_h = int(video_size[0])
        input_w = int(video_size[1])
        collector = RolloutCollector(
            model=self.accelerator.unwrap_model(self.model),
            env=env,
            processor=self.processor,
            cfg=self.cfg,
            sigma_max=self.sigma_max,
            num_inference_steps=self.num_inference_steps,
            action_horizon=action_horizon,
            replan_steps=int(self.cfg.EVALUATION.get("replan_steps", 10)),
            num_steps_wait=int(self.cfg.EVALUATION.get("num_steps_wait", 30)),
            input_h=input_h,
            input_w=input_w,
            model_device=str(self.accelerator.unwrap_model(self.model).device),
        )
        runtime = {
            "task_description": task_description,
            "initial_states": initial_states,
            "collector": collector,
        }
        self._task_runtime_cache[task_id] = runtime
        return runtime

    def _next_task_batch(self) -> list[int]:
        batch = []
        for _ in range(self.task_batch_size):
            task_id = self.task_ids[self.task_cursor % len(self.task_ids)]
            batch.append(task_id)
            self.task_cursor += 1
        return batch

    def _next_initial_state(self, task_id: int, initial_states: list[Any]) -> Any:
        if not initial_states:
            raise ValueError(f"No initial states found for task_id={task_id}.")
        cursor = self._task_state_cursors.get(task_id, 0)
        state = initial_states[cursor % len(initial_states)]
        self._task_state_cursors[task_id] = cursor + 1
        return state

    def _collect_rollout_buffer(self) -> RolloutBuffer:
        buffer = RolloutBuffer()
        for task_id in self._next_task_batch():
            runtime = self._get_task_runtime(task_id)
            initial_state = self._next_initial_state(task_id, runtime["initial_states"])
            task_buffer = runtime["collector"].collect_group(
                task_description=runtime["task_description"],
                task_id=f"{self.cfg.EVALUATION.task_suite_name}:{task_id}",
                group_size=self.group_size,
                initial_state=initial_state,
            )
            buffer.extend(task_buffer)
        return buffer

    def _training_state_payload(self) -> dict[str, Any]:
        return {
            "global_step": int(self.global_step),
            "task_cursor": int(self.task_cursor),
            "task_state_cursors": {str(k): int(v) for k, v in self._task_state_cursors.items()},
            "variant": self.variant.name,
        }

    def _save_weights_checkpoint(self, step_tag: str) -> str:
        model = self.accelerator.unwrap_model(self.model)
        ckpt_path = os.path.join(self.weights_dir, f"{step_tag}.pt")
        model.save_checkpoint(ckpt_path, optimizer=None, step=self.global_step)
        return ckpt_path

    def save_checkpoint(self) -> dict[str, str]:
        step_tag = f"update_{self.global_step:06d}"
        weights_path = self._save_weights_checkpoint(step_tag=step_tag)
        state_path = os.path.join(self.state_dir, step_tag)
        ensure_dir(state_path)
        torch.save(
            {
                "optimizer": self.optimizer.state_dict(),
                "trainer_state": self._training_state_payload(),
            },
            os.path.join(state_path, "trainer_state.pt"),
        )
        return {"weights_path": weights_path, "state_path": state_path}

    def _load_resume_if_needed(self) -> None:
        if not self.resume:
            return
        resume_dir = Path(str(self.resume))
        state_file = resume_dir / "trainer_state.pt"
        if not state_file.exists():
            raise FileNotFoundError(f"RL resume state not found: {state_file}")
        payload = torch.load(state_file, map_location="cpu")
        self.optimizer.load_state_dict(payload["optimizer"])
        trainer_state = payload["trainer_state"]
        self.global_step = int(trainer_state.get("global_step", 0))
        self.task_cursor = int(trainer_state.get("task_cursor", 0))
        self._task_state_cursors = {
            int(k): int(v) for k, v in trainer_state.get("task_state_cursors", {}).items()
        }
        logger.info("Resumed RL trainer from %s at update=%d", resume_dir, self.global_step)

    def _log_metrics(self, rollout_metrics: dict[str, float], train_metrics: dict[str, float]) -> None:
        payload = {f"rollout/{k}": v for k, v in rollout_metrics.items()}
        payload.update({f"train/{k}": v for k, v in train_metrics.items()})
        payload["train/update"] = float(self.global_step)
        self.metric_logger.log(payload, step=self.global_step)

    def train(self) -> None:
        logger.info("Starting FastWAM RL training for %d updates.", self.max_updates)
        while self.global_step < self.max_updates:
            rollout_buffer = self._collect_rollout_buffer()
            assign_advantages(
                rollout_buffer,
                variant=self.variant.name,
                trajectory_assignment=self.trajectory_assignment,
                gamma=self.advantage_gamma,
            )
            rollout_metrics = compute_rollout_metrics(rollout_buffer)

            unwrapped_model = self.accelerator.unwrap_model(self.model)
            train_metrics = {
                "variant_chunk_ratio": 1.0 if self.variant.ratio_mode == "chunk" else 0.0,
                "variant_trajectory_ratio": 1.0 if self.variant.ratio_mode == "trajectory" else 0.0,
            }
            for _ in range(self.num_optimization_epochs):
                result = compute_gspo_objective(
                    model=unwrapped_model,
                    buffer=rollout_buffer,
                    variant=self.variant.name,
                    clip_range=self.clip_range,
                    kl_coef=self.kl_coef,
                    sigma_max=self.sigma_max,
                    num_inference_steps=self.num_inference_steps,
                    sigma_shift=(
                        None
                        if self.cfg.EVALUATION.get("sigma_shift") is None
                        else float(self.cfg.EVALUATION.sigma_shift)
                    ),
                )
                self.optimizer.zero_grad(set_to_none=True)
                if result.metrics["num_objective_terms"] > 0:
                    self.accelerator.backward(result.loss)
                    grad_norm = self.accelerator.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        self.max_grad_norm,
                    )
                    self.optimizer.step()
                    train_metrics["grad_norm"] = float(grad_norm)
                else:
                    train_metrics["grad_norm"] = 0.0
                train_metrics.update(result.metrics)

            self.global_step += 1

            if self.log_every > 0 and self.global_step % self.log_every == 0:
                self._log_metrics(rollout_metrics=rollout_metrics, train_metrics=train_metrics)
                logger.info(
                    "[rl] update=%d/%d variant=%s success_rate=%.3f num_traj=%d num_chunks=%d loss=%.4f clip_frac=%.3f approx_kl=%.5f",
                    self.global_step,
                    self.max_updates,
                    self.variant.name,
                    rollout_metrics["success_rate"],
                    int(rollout_metrics["num_trajectories"]),
                    int(rollout_metrics["num_chunks"]),
                    float(result.loss.detach().item()) if result.metrics["num_objective_terms"] > 0 else 0.0,
                    train_metrics["clip_fraction"],
                    train_metrics["approx_kl"],
                )

            if self.save_every > 0 and self.global_step % self.save_every == 0:
                ckpt_info = self.save_checkpoint()
                logger.info(
                    "[rl-ckpt] update=%d weights=%s state=%s",
                    self.global_step,
                    ckpt_info["weights_path"],
                    ckpt_info["state_path"],
                )

        ckpt_info = self.save_checkpoint()
        logger.info(
            "[rl-done] updates=%d weights=%s state=%s",
            self.global_step,
            ckpt_info["weights_path"],
            ckpt_info["state_path"],
        )
        self.metric_logger.finish()

        for runtime in self._task_runtime_cache.values():
            env = runtime["collector"].env
            if hasattr(env, "close"):
                env.close()
