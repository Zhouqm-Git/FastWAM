"""Minimal multi-backend metric logger inspired by RLinf."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from omegaconf import DictConfig, OmegaConf


class _TensorboardLogger:
    def __init__(self, log_path: str):
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(log_path)

    def log(self, data: dict[str, float], step: int) -> None:
        for key, value in data.items():
            self.writer.add_scalar(key, value, step)

    def finish(self) -> None:
        self.writer.close()


class RLMetricLogger:
    """Small logger bundle for RL metrics."""

    supported_backends = ("wandb", "tensorboard")

    def __init__(self, cfg: DictConfig, *, output_dir: str):
        self.cfg = cfg
        self.output_dir = output_dir
        logger_cfg = cfg.rl.logging
        configured_backends = logger_cfg.get("backends", [])
        if isinstance(configured_backends, str):
            backends = [configured_backends]
        else:
            backends = list(configured_backends or [])
        if not backends and bool(cfg.wandb.enabled):
            backends = ["wandb"]
        self.backends = [str(backend).strip().lower() for backend in backends]
        for backend in self.backends:
            if backend not in self.supported_backends:
                raise ValueError(
                    f"Unsupported logger backend: {backend}. "
                    f"Expected one of {list(self.supported_backends)}."
                )

        self.loggers: dict[str, object] = {}
        self._init_backends()

    def _init_backends(self) -> None:
        if "wandb" in self.backends:
            try:
                import wandb
            except ImportError as exc:
                raise ImportError(
                    "RL logging requested `wandb`, but the package is not installed."
                ) from exc

            wandb_dir = Path(self.output_dir) / "wandb"
            wandb_dir.mkdir(parents=True, exist_ok=True)
            run = wandb.init(
                entity=self.cfg.wandb.workspace,
                project=self.cfg.wandb.project,
                name=self.cfg.wandb.name,
                group=(
                    None
                    if self.cfg.wandb.group in (None, "null", "")
                    else str(self.cfg.wandb.group)
                ),
                mode=self.cfg.wandb.mode,
                config=OmegaConf.to_container(self.cfg, resolve=True),
                dir=str(wandb_dir),
                reinit=True,
            )
            self.loggers["wandb"] = run

        if "tensorboard" in self.backends:
            tb_dir = Path(self.output_dir) / "tensorboard"
            tb_dir.mkdir(parents=True, exist_ok=True)
            OmegaConf.save(self.cfg, os.path.join(tb_dir, "config.yaml"), resolve=True)
            self.loggers["tensorboard"] = _TensorboardLogger(str(tb_dir))

    def log(self, payload: dict[str, float], step: int) -> None:
        for backend, logger in self.loggers.items():
            if backend == "wandb":
                logger.log(payload, step=step)
            else:
                logger.log(payload, step)

    def finish(self) -> None:
        for backend, logger in self.loggers.items():
            if backend == "wandb":
                logger.finish()
            else:
                logger.finish()

    def __del__(self) -> None:
        self.finish()
