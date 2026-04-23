from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        scale = math.log(10000) / max(half - 1, 1)
        freqs = torch.exp(torch.arange(half, device=timesteps.device) * -scale)
        args = timesteps.float()[:, None] * freqs[None]
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, emb_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.emb = nn.Linear(emb_dim, out_channels)
        self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb(F.silu(emb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, emb_dim: int, dropout: float) -> None:
        super().__init__()
        self.res = ResBlock(in_channels, out_channels, emb_dim, dropout)
        self.down = nn.Conv2d(out_channels, out_channels, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.res(x, emb)
        return self.down(h), h


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, emb_dim: int, dropout: float) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)
        self.res = ResBlock(out_channels + skip_channels, out_channels, emb_dim, dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        return self.res(torch.cat([x, skip], dim=1), emb)


class SimpleUNet(nn.Module):
    def __init__(
        self,
        image_channels: int,
        base_channels: int,
        channel_mults: list[int],
        time_dim: int,
        condition_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        emb_dim = time_dim + condition_dim
        self.time = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.input = nn.Conv2d(image_channels, base_channels, 3, padding=1)

        channels = [base_channels * mult for mult in channel_mults]
        self.downs = nn.ModuleList()
        current = base_channels
        for ch in channels:
            self.downs.append(DownBlock(current, ch, emb_dim, dropout))
            current = ch

        self.mid = nn.ModuleList(
            [
                ResBlock(current, current, emb_dim, dropout),
                ResBlock(current, current, emb_dim, dropout),
            ]
        )

        self.ups = nn.ModuleList()
        for skip_ch in reversed(channels):
            self.ups.append(UpBlock(current, skip_ch, skip_ch, emb_dim, dropout))
            current = skip_ch

        self.output = nn.Sequential(
            nn.GroupNorm(min(8, current), current),
            nn.SiLU(),
            nn.Conv2d(current, image_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        emb = torch.cat([self.time(timesteps), condition], dim=-1)
        h = self.input(x)
        skips = []
        for down in self.downs:
            h, skip = down(h, emb)
            skips.append(skip)

        for block in self.mid:
            h = block(h, emb)

        for up in self.ups:
            h = up(h, skips.pop(), emb)

        return self.output(h)
