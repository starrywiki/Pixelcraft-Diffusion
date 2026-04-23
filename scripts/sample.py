from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from pixelcraft.data.dataset import stable_label_id
from pixelcraft.models.conditioning import LabelConditioner
from pixelcraft.models.diffusion import GaussianDiffusion
from pixelcraft.models.unet import SimpleUNet
from pixelcraft.utils.config import load_config
from pixelcraft.utils.image import save_image_grid
from pixelcraft.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample images from a trained checkpoint.")
    parser.add_argument("--config", default="configs/pixelart_32.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--prompt", default="red slime")
    parser.add_argument("--num-images", type=int, default=16)
    parser.add_argument("--output", default="outputs/demo/sample.png")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(int(config["experiment"]["seed"]))
    device = resolve_device(args.device)
    checkpoint_path = args.checkpoint or config["sampling"]["checkpoint"]

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

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    conditioner.load_state_dict(checkpoint["conditioner"])
    model.eval()
    conditioner.eval()

    condition_id = stable_label_id(args.prompt, int(config["model"]["num_classes"]))
    ids = torch.full((args.num_images,), condition_id, dtype=torch.long, device=device)
    condition = conditioner(ids)
    samples = diffusion.sample(
        (
            args.num_images,
            int(config["data"]["channels"]),
            int(config["data"]["image_size"]),
            int(config["data"]["image_size"]),
        ),
        condition=condition,
        steps=args.steps or int(config["sampling"].get("steps", config["diffusion"]["timesteps"])),
    )
    save_image_grid(samples, args.output, nrow=int(args.num_images**0.5))
    print(f"Saved samples to {args.output}")


if __name__ == "__main__":
    main()
