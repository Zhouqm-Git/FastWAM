import math

import torch


class WanContinuousFlowMatchScheduler:
    """Continuous-time Flow-Matching scheduler with shift-based sampling."""

    def __init__(self, num_train_timesteps: int = 1000, shift: float = 5.0, eps: float = 1e-10):
        if num_train_timesteps <= 0:
            raise ValueError(f"`num_train_timesteps` must be positive, got {num_train_timesteps}")
        if shift <= 0:
            raise ValueError(f"`shift` must be positive, got {shift}")
        self.num_train_timesteps = int(num_train_timesteps)
        self.shift = float(shift)
        self.eps = float(eps)
        self._y_min, self._weight_norm_const = self._precompute_training_weight_stats()

    @staticmethod
    def _phi(u: torch.Tensor, shift: float) -> torch.Tensor:
        return shift * u / (1.0 + (shift - 1.0) * u)

    def _precompute_training_weight_stats(self) -> tuple[float, float]:
        steps = self.num_train_timesteps
        u_grid = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float64)[:-1]
        t_grid = self._phi(u_grid, self.shift) * float(steps)
        y_grid = torch.exp(-2.0 * ((t_grid - (steps / 2.0)) / steps) ** 2)
        y_min = float(y_grid.min().item())
        y_shifted_grid = y_grid - y_min
        norm_const = float(y_shifted_grid.mean().item())
        return y_min, norm_const

    def sample_training_t(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError(f"`batch_size` must be positive, got {batch_size}")
        u = torch.rand((batch_size,), device=device, dtype=torch.float32)
        sigma = self._phi(u, self.shift)
        timestep = sigma * float(self.num_train_timesteps)
        return timestep.to(dtype=dtype)

    def training_weight(self, timestep: torch.Tensor) -> torch.Tensor:
        t = timestep.to(dtype=torch.float32)
        steps = float(self.num_train_timesteps)
        y = torch.exp(-2.0 * ((t - (steps / 2.0)) / steps) ** 2)
        y_shifted = y - self._y_min
        weight = y_shifted / (self._weight_norm_const + self.eps)
        if weight.numel() == 1:
            return weight.reshape(())
        return weight

    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        sigma = (timestep / float(self.num_train_timesteps)).to(
            original_samples.device, dtype=original_samples.dtype
        )
        if sigma.ndim == 0:
            return (1 - sigma) * original_samples + sigma * noise
        sigma = sigma.view(-1, *([1] * (original_samples.ndim - 1)))
        return (1 - sigma) * original_samples + sigma * noise

    @staticmethod
    def training_target(sample: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        del timestep
        return noise - sample

    def build_inference_schedule(
        self,
        num_inference_steps: int,
        device: torch.device,
        dtype: torch.dtype,
        shift_override: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if num_inference_steps <= 0:
            raise ValueError(f"`num_inference_steps` must be positive, got {num_inference_steps}")
        shift = self.shift if shift_override is None else float(shift_override)
        if shift <= 0:
            raise ValueError(f"`shift` must be positive, got {shift}")

        u_steps = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=device, dtype=torch.float32)
        sigma_steps = self._phi(u_steps, shift)
        timesteps = sigma_steps[:-1] * float(self.num_train_timesteps)
        deltas = sigma_steps[1:] - sigma_steps[:-1]
        return timesteps.to(dtype=dtype), deltas.to(dtype=dtype)

    @staticmethod
    def step(model_output: torch.Tensor, delta: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        delta = delta.to(sample.device, dtype=sample.dtype)
        if delta.ndim == 0:
            return sample + model_output * delta
        delta = delta.view(-1, *([1] * (sample.ndim - 1)))
        return sample + model_output * delta

    @staticmethod
    def step_sde_with_logprob(
        model_output: torch.Tensor,
        delta: torch.Tensor,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        sigma_max: float = 0.1,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Flow-GSPO SDE sampling step with log-probability.

        Based on OmniVLA-RL equations 10-13.
        Noise schedule: sigma_tau = sigma_max * (1 - tau)  (linear decay)
        Transition:     p(x_{tau+delta} | x_tau, s) ~ N(mu_tau, Sigma_tau)
        Where:
          mu_tau = x_tau + [v_theta + sigma_tau^2/2 * (x_tau + (1-tau)*v_theta)] * delta
          Sigma_tau = sigma_tau^2 * delta * I

        When sigma_max=0.1 is small, the score correction term is negligible:
          mu_tau ≈ x_tau + v_theta * delta

        Args:
            model_output: Predicted velocity v_theta, shape [1, T, D].
            delta: sigma_{i+1} - sigma_i (negative scalar).
            sample: Current noisy sample x_t, shape [1, T, D].
            sigma: Current sigma_t (scalar, in [0,1]).
            sigma_max: SDE noise upper bound (default 0.1).

        Returns:
            next_sample: Sampled x_{tau+delta}.
            log_prob:    log p(x_{tau+delta} | x_tau, s), scalar.
            mean:        Deterministic mean mu_tau.
            std:         Noise standard deviation (scalar).
        """
        delta = delta.to(sample.device, dtype=torch.float32)
        sample_f = sample.float()
        model_output_f = model_output.float()

        # tau = 1 - sigma (denoising progress: sigma 1->0 maps to tau 0->1)
        tau = 1.0 - sigma.float()
        # sigma_tau = sigma_max * (1 - tau) = sigma_max * sigma
        sigma_tau = sigma_max * sigma.float()

        # Deterministic mean: mu_tau = x_t + v_theta * delta  (simplified, sigma_max small)
        mean = sample_f + model_output_f * delta

        # Noise scale: sigma_tau * sqrt(|delta|)
        abs_delta = (-delta).float()  # delta is negative, so |delta| = -delta
        std = sigma_tau * torch.sqrt(abs_delta)

        # Sample: x_next = mean + std * randn
        noise = torch.randn(
            sample_f.shape,
            generator=generator,
            device=sample_f.device,
            dtype=sample_f.dtype,
        )
        next_sample = mean + std * noise

        # log p(x_next | mean, std^2 * I) = -0.5 * ((x_next - mean)^2 / std^2).mean() - log(std) - 0.5*log(2pi)
        # .mean() over all dimensions (not .sum()) so log-probs are comparable across chunk sizes
        if std.abs() < 1e-12:
            # sigma_max=0 degenerate case: deterministic step, log_prob=0
            log_prob = torch.tensor(0.0, device=sample.device, dtype=torch.float32)
        else:
            diff = (next_sample - mean) ** 2
            log_prob = -0.5 * diff.mean() / (std ** 2) - torch.log(std) - 0.5 * math.log(2.0 * math.pi)

        return (
            next_sample.to(dtype=sample.dtype),
            log_prob,
            mean.to(dtype=sample.dtype),
            std,
        )
