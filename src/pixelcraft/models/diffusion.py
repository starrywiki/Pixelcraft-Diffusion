from __future__ import annotations

import torch
from torch import nn


def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps)


class GaussianDiffusion(nn.Module):
    def __init__(self, model: nn.Module, timesteps: int = 1000, beta_schedule: str = "linear") -> None:
        super().__init__()
        if beta_schedule != "linear":
            raise ValueError(f"Unsupported beta schedule: {beta_schedule}")

        self.model = model
        self.timesteps = timesteps
        betas = linear_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance.clamp(min=1e-20))

    def q_sample(self, x_start: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return (
            self._extract(self.sqrt_alphas_cumprod, timesteps, x_start.shape) * x_start
            + self._extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x_start.shape) * noise
        )

    def p_losses(self, x_start: torch.Tensor, timesteps: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, timesteps, noise)
        predicted_noise = self.model(x_noisy, timesteps, condition)
        return torch.nn.functional.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def sample(self, shape: tuple[int, int, int, int], condition: torch.Tensor, steps: int | None = None) -> torch.Tensor:
        device = self.betas.device
        image = torch.randn(shape, device=device)
        total_steps = self.timesteps if steps is None else min(steps, self.timesteps)
        indices = torch.linspace(self.timesteps - 1, 0, total_steps, device=device).long()
        for t_value in indices:
            t = torch.full((shape[0],), int(t_value.item()), device=device, dtype=torch.long)
            image = self._p_sample(image, t, condition)
        return image

    @torch.no_grad()
    def _p_sample(self, x: torch.Tensor, timesteps: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        betas_t = self._extract(self.betas, timesteps, x.shape)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x.shape)
        sqrt_recip = self._extract(self.sqrt_recip_alphas, timesteps, x.shape)
        model_mean = sqrt_recip * (x - betas_t * self.model(x, timesteps, condition) / sqrt_one_minus)

        noise = torch.randn_like(x)
        nonzero_mask = (timesteps != 0).float().view(-1, *([1] * (x.ndim - 1)))
        variance = self._extract(self.posterior_variance, timesteps, x.shape)
        return model_mean + nonzero_mask * torch.sqrt(variance) * noise

    @staticmethod
    def _extract(values: torch.Tensor, timesteps: torch.Tensor, shape: torch.Size | tuple[int, ...]) -> torch.Tensor:
        out = values.gather(0, timesteps)
        return out.reshape(timesteps.shape[0], *([1] * (len(shape) - 1)))
