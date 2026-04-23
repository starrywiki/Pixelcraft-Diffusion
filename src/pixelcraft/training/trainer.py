from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from pixelcraft.utils.image import save_image_grid


class DiffusionTrainer:
    def __init__(
        self,
        diffusion: torch.nn.Module,
        conditioner: torch.nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        output_dir: str | Path,
        image_size: int,
        image_channels: int,
        device: torch.device,
        grad_clip: float | None = None,
    ) -> None:
        self.diffusion = diffusion
        self.conditioner = conditioner
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.output_dir = Path(output_dir)
        self.image_size = image_size
        self.image_channels = image_channels
        self.device = device
        self.grad_clip = grad_clip
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.sample_dir = self.output_dir / "samples"
        self.log_dir = self.output_dir / "logs"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.sample_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def fit(self, epochs: int, log_every: int, sample_every: int, save_every: int, sample_count: int) -> None:
        global_step = 0
        loss_log = self.log_dir / "loss.csv"
        if not loss_log.exists():
            loss_log.write_text("step,epoch,loss\n", encoding="utf-8")

        for epoch in range(1, epochs + 1):
            self.diffusion.train()
            self.conditioner.train()
            progress = tqdm(self.train_loader, desc=f"epoch {epoch}/{epochs}")
            for batch in progress:
                images = batch["image"].to(self.device)
                condition_ids = batch["condition_id"].to(self.device)
                condition = self.conditioner(condition_ids)
                timesteps = torch.randint(0, self.diffusion.timesteps, (images.shape[0],), device=self.device)

                loss = self.diffusion.p_losses(images, timesteps, condition)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.diffusion.parameters()) + list(self.conditioner.parameters()), self.grad_clip
                    )
                self.optimizer.step()

                global_step += 1
                progress.set_postfix(loss=f"{loss.item():.4f}")
                if global_step % log_every == 0:
                    with loss_log.open("a", encoding="utf-8") as f:
                        f.write(f"{global_step},{epoch},{loss.item():.6f}\n")

            if sample_every and epoch % sample_every == 0:
                self.save_samples(epoch, sample_count)
            if save_every and epoch % save_every == 0:
                self.save_checkpoint(epoch, "latest.pt")
                self.save_checkpoint(epoch, f"epoch_{epoch:04d}.pt")

    @torch.no_grad()
    def save_samples(self, epoch: int, sample_count: int) -> None:
        self.diffusion.eval()
        self.conditioner.eval()
        ids = torch.arange(sample_count, device=self.device) % self.conditioner.embedding.num_embeddings
        condition = self.conditioner(ids)
        samples = self.diffusion.sample(
            (sample_count, self.image_channels, self.image_size, self.image_size),
            condition,
            steps=min(100, self.diffusion.timesteps),
        )
        save_image_grid(samples, self.sample_dir / f"epoch_{epoch:04d}.png", nrow=int(sample_count**0.5))

    def save_checkpoint(self, epoch: int, filename: str) -> None:
        torch.save(
            {
                "epoch": epoch,
                "model": self.diffusion.model.state_dict(),
                "conditioner": self.conditioner.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            self.checkpoint_dir / filename,
        )
