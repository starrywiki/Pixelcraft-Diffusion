from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from pixelcraft.data.dataset import JsonlImageDataset
from pixelcraft.models.conditioning import LabelConditioner
from pixelcraft.models.diffusion import GaussianDiffusion
from pixelcraft.models.unet import SimpleUNet
from pixelcraft.training.trainer import DiffusionTrainer
from pixelcraft.utils.config import load_config, save_config
from pixelcraft.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a lightweight conditional DDPM.")
    parser.add_argument("--config", default="configs/pixelart_32.yaml")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(int(config["experiment"]["seed"]))
    device = resolve_device(config["training"].get("device", "auto"))

    output_dir = Path(config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "config.yaml")
    shutil.copy2(args.config, output_dir / "source_config.yaml")

    dataset = JsonlImageDataset(
        metadata_path=config["data"]["train_metadata"],
        root=config["data"]["root"],
        image_size=int(config["data"]["image_size"]),
        num_classes=int(config["model"]["num_classes"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["data"].get("num_workers", 2)),
        pin_memory=device.type == "cuda",
    )

    conditioner = LabelConditioner(
        num_classes=int(config["model"]["num_classes"]),
        condition_dim=int(config["model"]["condition_dim"]),
    ).to(device)
    model = SimpleUNet(
        image_channels=int(config["data"]["channels"]),
        base_channels=int(config["model"]["base_channels"]),
        channel_mults=list(config["model"]["channel_mults"]),
        time_dim=int(config["model"]["time_dim"]),
        condition_dim=int(config["model"]["condition_dim"]),
        dropout=float(config["model"]["dropout"]),
    ).to(device)
    diffusion = GaussianDiffusion(
        model=model,
        timesteps=int(config["diffusion"]["timesteps"]),
        beta_schedule=config["diffusion"]["beta_schedule"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(conditioner.parameters()),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"].get("weight_decay", 0.0)),
    )

    trainer = DiffusionTrainer(
        diffusion=diffusion,
        conditioner=conditioner,
        train_loader=loader,
        optimizer=optimizer,
        output_dir=output_dir,
        image_size=int(config["data"]["image_size"]),
        image_channels=int(config["data"]["channels"]),
        device=device,
        grad_clip=float(config["training"].get("grad_clip", 0.0)),
    )
    trainer.fit(
        epochs=int(config["training"]["epochs"]),
        log_every=int(config["training"]["log_every"]),
        sample_every=int(config["training"]["sample_every"]),
        save_every=int(config["training"]["save_every"]),
        sample_count=int(config["training"]["sample_count"]),
    )


if __name__ == "__main__":
    main()
