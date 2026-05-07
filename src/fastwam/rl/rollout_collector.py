"""Single-process rollout collector for Flow-GSPO in LIBERO environments.

Reuses FastWAM eval_libero_single.py environment interaction patterns but
replaces ODE denoising with SDE sampling + chain tracking.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image

from ..datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from ..utils.logging_config import get_logger
from .rollout_buffer import ChunkData, RolloutBuffer, TrajectoryData

logger = get_logger(__name__)


def _extract_sim_state(obs: dict) -> np.ndarray:
    """Build proprioceptive state from LIBERO observation."""
    from experiments.libero.libero_utils import quat2axisangle

    state = np.concatenate(
        (
            obs["robot0_eef_pos"],
            quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    ).astype(np.float32)
    return state


def _center_crop_resize(img: np.ndarray, width: int, height: int) -> np.ndarray:
    """Match eval_libero_single.py preprocessing exactly."""
    pil_img = Image.fromarray(img)
    src_w, src_h = pil_img.size
    scale = max(width / src_w, height / src_h)
    resized = pil_img.resize((round(src_w * scale), round(src_h * scale)), resample=Image.BILINEAR)
    rw, rh = resized.size
    left = max((rw - width) // 2, 0)
    top = max((rh - height) // 2, 0)
    cropped = resized.crop((left, top, left + width, top + height))
    return np.asarray(cropped, dtype=np.uint8)


def get_libero_dummy_action() -> list[float]:
    """Return a zero action for LIBERO warm-up steps."""
    return [0.0] * 7


def _get_max_steps(task_suite_name: str) -> int:
    suite_steps = {
        "libero_spatial": 400,
        "libero_object": 400,
        "libero_goal": 400,
        "libero_10": 700,
        "libero_90": 700,
    }
    if task_suite_name not in suite_steps:
        raise ValueError(f"Unknown task suite: {task_suite_name}")
    return suite_steps[task_suite_name]


class RolloutCollector:
    """Collect Flow-GSPO rollout data in a LIBERO environment.

    Single-process sequential execution. Reuses FastWAM eval_libero_single.py
    environment interaction patterns.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        env: Any,
        processor: Any,
        cfg: Any,
        sigma_max: float = 0.1,
        num_inference_steps: int = 10,
        action_horizon: int = 32,
        replan_steps: int = 10,
        num_steps_wait: int = 30,
        input_h: int = 256,
        input_w: int = 256,
        model_device: str = "cuda",
    ):
        """
        Args:
            model: FastWAM model instance.
            env: LIBERO environment (already initialized).
            processor: FastWAMProcessor for action normalization.
            cfg: Hydra DictConfig or similar config object.
            sigma_max: Flow-GSPO SDE noise upper bound.
            num_inference_steps: Denoising steps per action block.
            action_horizon: Number of action steps predicted per block.
            replan_steps: Environment steps before replanning.
            num_steps_wait: LIBERO warm-up steps with dummy actions.
            input_h: Input image height.
            input_w: Input image width.
            model_device: Device for model inference.
        """
        self.model = model
        self.env = env
        self.processor = processor
        self.cfg = cfg
        self.sigma_max = sigma_max
        self.num_inference_steps = num_inference_steps
        self.action_horizon = action_horizon
        self.replan_steps = replan_steps
        self.num_steps_wait = num_steps_wait
        self.input_h = input_h
        self.input_w = input_w
        self.model_device = model_device
        self._seed_base = None if cfg.get("seed") is None else int(cfg.seed)
        self._sample_counter = 0

        # --- Ablation: action horizon & log-prob coverage ---
        # rl.action_horizon overrides the prediction horizon passed to the model.
        #   null  = use default (e.g., 32 from data.train.num_frames - 1)
        #   int   = override to this value (e.g., 10 for predict=exec matching)
        # rl.exec_horizon controls log_prob coverage for partial execution.
        #   null  = log_prob covers all predicted actions
        #   int   = log_prob only covers the first N actions (GR00T-style slicing)
        rl_ah = cfg.rl.get("action_horizon")
        if rl_ah is not None:
            self.action_horizon = int(rl_ah)
            logger.info(
                "RL action_horizon override: %d (default was %d)",
                self.action_horizon, action_horizon,
            )
        self.exec_horizon = (
            int(cfg.rl.get("exec_horizon"))
            if cfg.rl.get("exec_horizon") is not None
            else None
        )
        if self.action_horizon <= 0:
            raise ValueError(f"`action_horizon` must be positive, got {self.action_horizon}.")
        if self.replan_steps <= 0:
            raise ValueError(f"`replan_steps` must be positive, got {self.replan_steps}.")
        if self.action_horizon < self.replan_steps:
            raise ValueError(
                "`action_horizon` must be >= `replan_steps` so each rollout chunk "
                f"can execute the configured {self.replan_steps} actions, got "
                f"action_horizon={self.action_horizon}."
            )
        if self.exec_horizon is not None:
            if self.exec_horizon <= 0:
                raise ValueError(f"`exec_horizon` must be positive, got {self.exec_horizon}.")
            if self.exec_horizon > self.action_horizon:
                raise ValueError(
                    "`exec_horizon` cannot exceed `action_horizon`, got "
                    f"exec_horizon={self.exec_horizon}, action_horizon={self.action_horizon}."
                )
            if self.exec_horizon != self.replan_steps:
                raise ValueError(
                    "`exec_horizon` must equal `replan_steps` so ratio/log-prob coverage "
                    "matches the executed action prefix exactly, got "
                    f"exec_horizon={self.exec_horizon}, replan_steps={self.replan_steps}."
                )

    def _next_inference_seed(self) -> Optional[int]:
        if self._seed_base is None:
            return None
        seed = self._seed_base + self._sample_counter
        self._sample_counter += 1
        return seed

    def _obs_to_model_input(
        self,
        obs: dict,
        task_description: str,
    ) -> dict[str, Any]:
        """Convert LIBERO observation to model input dict.

        Returns dict suitable for model.infer_action_with_logprob(**result).
        """
        # Build image input
        imgs = self._get_libero_image(obs)
        image_meta = self.processor.shape_meta["images"]
        num_cameras = self.processor.num_output_cameras
        if len(image_meta) < int(num_cameras):
            raise ValueError(
                f"shape_meta.images has {len(image_meta)} entries, "
                f"but num_output_cameras={num_cameras}."
            )
        concatenation = getattr(self.cfg, "data", {}).get("train", {}).get("concat_multi_camera", "horizontal")

        if num_cameras == 1:
            primary_h, primary_w = image_meta[0]["shape"][1], image_meta[0]["shape"][2]
            rgb = _center_crop_resize(imgs["image"], width=primary_w, height=primary_h)
        elif num_cameras == 2:
            primary_h, primary_w = image_meta[0]["shape"][1], image_meta[0]["shape"][2]
            wrist_h, wrist_w = image_meta[1]["shape"][1], image_meta[1]["shape"][2]
            primary = _center_crop_resize(imgs["image"], width=primary_w, height=primary_h)
            wrist = _center_crop_resize(imgs["wrist_image"], width=wrist_w, height=wrist_h)
            if concatenation == "horizontal":
                rgb = np.concatenate([primary, wrist], axis=1)
            elif concatenation == "vertical":
                rgb = np.concatenate([primary, wrist], axis=0)
            else:
                raise ValueError(f"Invalid concat_multi_camera: {concatenation}")
        else:
            raise ValueError(f"Unsupported num_output_cameras={num_cameras}")

        actual_h, actual_w = int(rgb.shape[0]), int(rgb.shape[1])
        if actual_h != int(self.input_h) or actual_w != int(self.input_w):
            image_shapes = [meta["shape"] for meta in image_meta]
            raise ValueError(
                "Input image size mismatch after per-camera resize + concat: "
                f"got (H,W)=({actual_h},{actual_w}), expected (H,W)=({self.input_h},{self.input_w}); "
                f"shape_meta.images={image_shapes}, concat_multi_camera={concatenation}."
            )

        x = torch.tensor(rgb).permute(2, 0, 1).unsqueeze(0).to(
            device=self.model_device, dtype=self.model.torch_dtype
        )
        x = x * (2.0 / 255.0) - 1.0

        # Build proprio
        proprio = self._normalize_proprio(_extract_sim_state(obs))

        # Build prompt
        prompt = DEFAULT_PROMPT.format(task=task_description)
        with torch.no_grad():
            context, context_mask = self.model.encode_prompt(prompt)

        return {
            "prompt": prompt,
            "context": context,
            "context_mask": context_mask,
            "input_image": x,
            "proprio": proprio,
        }

    def _normalize_proprio(self, proprio: np.ndarray) -> torch.Tensor:
        """Normalize proprio using FastWAMProcessor."""
        state_meta = self.processor.shape_meta["state"]
        state_key = state_meta[0]["key"]
        state_batch = {"state": {state_key: torch.as_tensor(proprio, dtype=torch.float32).unsqueeze(0)}}
        state_batch = self.processor.action_state_transform(state_batch)
        state_batch = self.processor.normalizer.forward(state_batch)
        return state_batch["state"][state_key]

    def _get_libero_image(self, obs: dict) -> dict[str, np.ndarray]:
        """Extract images from LIBERO observation."""
        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        imgs = {"image": img}
        if "robot0_eye_in_hand_image" in obs:
            wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
            imgs["wrist_image"] = wrist_img
        return imgs

    def _denormalize_action(self, action: torch.Tensor) -> np.ndarray:
        """Denormalize model output action to environment action space."""
        if action.ndim == 2:
            action = action.unsqueeze(0)
        action_meta = self.processor.shape_meta["action"]
        action_key = action_meta[0]["key"]
        normalizer = self.processor.normalizer.normalizers["action"][action_key]
        action = action.to(dtype=torch.float32, device="cpu")
        denorm = normalizer.backward(action)
        return denorm.numpy()

    def _postprocess_actions(self, action: torch.Tensor) -> list[list[float]]:
        """Full action postprocessing: denormalize + gripper conversion.

        Matches eval_libero_single.py behavior.
        """
        from experiments.libero.libero_utils import invert_gripper_action

        action = self._denormalize_action(action)[0]  # [T, D]

        # Flip gripper sign (align with LIBERO convention)
        action[..., -1] = action[..., -1] * 2 - 1
        action = invert_gripper_action(action)

        # Optionally binarize gripper
        if bool(getattr(self.cfg, "EVALUATION", {}).get("binarize_gripper", False)):
            action[..., -1] = np.sign(action[..., -1])

        return action.tolist()

    def collect_trajectory(
        self,
        task_description: str,
        task_id: str,
        initial_state=None,
        group_id: str = "",
        reset_id: str = "",
        initial_state_index: int = -1,
        trajectory_id: str = "",
        max_steps: Optional[int] = None,
    ) -> TrajectoryData:
        """Run a complete trajectory in the environment.

        Returns TrajectoryData containing:
        - All chunks with chain + log_prob (Flow-GSPO core data)
        - Per-step reward + terminal success
        """
        if max_steps is None:
            max_steps = _get_max_steps(str(self.cfg.EVALUATION.task_suite_name))
        trajectory_start_time = time.perf_counter()

        obs = self.env.reset()
        if initial_state is not None:
            obs = self.env.set_init_state(initial_state)

        # Warm-up with dummy actions
        for _ in range(self.num_steps_wait):
            obs, _, _, _ = self.env.step(get_libero_dummy_action())

        chunks: list[ChunkData] = []
        pending_actions: list[list[float]] = []
        first_chunk_seed: Optional[int] = None

        t = 0
        done = False
        while t < max_steps and not done:
            if not pending_actions:
                # Flow-GSPO: SDE sampling + chain tracking
                model_input = self._obs_to_model_input(obs, task_description)
                chunk_index = len(chunks)
                chunk_seed = self._next_inference_seed()
                if first_chunk_seed is None:
                    first_chunk_seed = chunk_seed
                result = self.model.infer_action_with_logprob(
                    prompt=None,
                    context=model_input["context"],
                    context_mask=model_input["context_mask"],
                    input_image=model_input["input_image"],
                    proprio=model_input["proprio"],
                    action_horizon=self.action_horizon,
                    negative_prompt=str(self.cfg.EVALUATION.get("negative_prompt", "")),
                    text_cfg_scale=float(self.cfg.EVALUATION.get("text_cfg_scale", 1.0)),
                    sigma_max=self.sigma_max,
                    num_inference_steps=self.num_inference_steps,
                    sigma_shift=(
                        None
                        if self.cfg.EVALUATION.get("sigma_shift") is None
                        else float(self.cfg.EVALUATION.get("sigma_shift"))
                    ),
                    seed=chunk_seed,
                    rand_device=str(self.cfg.EVALUATION.get("rand_device", "cpu")),
                    tiled=bool(self.cfg.EVALUATION.get("tiled", False)),
                    exec_horizon=self.exec_horizon,
                )

                # Post-process actions for environment execution
                actions = self._postprocess_actions(result["action"])
                pending_actions = list(actions[: self.replan_steps])

                # Store chunk data
                effective_horizon = self.exec_horizon if self.exec_horizon is not None else self.action_horizon
                chunk = ChunkData(
                    obs_image=model_input["input_image"].cpu(),
                    obs_proprio=(
                        model_input["proprio"].cpu()
                        if model_input.get("proprio") is not None
                        else None
                    ),
                    context=model_input["context"].cpu(),
                    context_mask=model_input["context_mask"].cpu(),
                    chain=result["chain"].cpu(),
                    old_log_prob=result["log_prob"].cpu(),
                    block_size=effective_horizon * self.num_inference_steps,
                    exec_horizon=self.exec_horizon,
                    action=result["action"].cpu(),
                    chunk_rewards=[],
                    done=False,
                    task_id=task_id,
                    group_id=group_id,
                    reset_id=reset_id,
                    initial_state_index=initial_state_index,
                    trajectory_id=trajectory_id,
                    chunk_index=chunk_index,
                    env_step_start=t,
                    env_step_end=t,
                    rollout_seed=chunk_seed,
                    rollout_time=time.perf_counter() - trajectory_start_time,
                    task_suite_name=str(self.cfg.EVALUATION.task_suite_name),
                    reward_components={},
                    task_description=task_description,
                )
                chunks.append(chunk)

            # Execute one step
            action = pending_actions.pop(0)
            obs, reward, done, info = self.env.step(action)

            # Record reward to current chunk
            chunks[-1].chunk_rewards.append(float(reward))
            t += 1
            chunks[-1].env_step_end = t

            if done:
                chunks[-1].done = True
                break

        # LIBERO: done=True means task success
        success = done
        trajectory_reward = 1.0 if success else 0.0

        return TrajectoryData(
            task_id=task_id,
            task_description=task_description,
            chunks=chunks,
            trajectory_reward=trajectory_reward,
            success=success,
            group_id=group_id,
            reset_id=reset_id,
            initial_state_index=initial_state_index,
            trajectory_id=trajectory_id,
            rollout_seed=first_chunk_seed,
            rollout_time=time.perf_counter() - trajectory_start_time,
            task_suite_name=str(self.cfg.EVALUATION.task_suite_name),
            reward_components={"success": trajectory_reward},
        )

    def collect_group(
        self,
        task_description: str,
        task_id: str,
        group_size: int = 8,
        initial_state=None,
        group_id: str = "",
        reset_id: str = "",
        initial_state_index: int = -1,
        max_steps: Optional[int] = None,
    ) -> RolloutBuffer:
        """Flow-GSPO: sample G trajectories for the same task.

        Used for group advantage normalization:
            A_hat_i = (R_i - mean) / std
        """
        buffer = RolloutBuffer()
        for traj_idx in range(group_size):
            traj = self.collect_trajectory(
                task_description=task_description,
                task_id=task_id,
                initial_state=initial_state,
                group_id=group_id,
                reset_id=reset_id,
                initial_state_index=initial_state_index,
                trajectory_id=f"{group_id or task_id}:traj_{traj_idx:03d}",
                max_steps=max_steps,
            )
            buffer.add_trajectory(traj)
            logger.info(
                "Collected trajectory %d/%d: success=%s, reward=%.1f, chunks=%d",
                len(buffer.trajectories),
                group_size,
                traj.success,
                traj.trajectory_reward,
                len(traj.chunks),
            )
        return buffer
