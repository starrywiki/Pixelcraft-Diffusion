from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torchvision.utils import make_grid, save_image


def pil_to_rgb(path: str | Path, background: tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    image = Image.open(path).convert("RGBA")
    canvas = Image.new("RGBA", image.size, (*background, 255))
    canvas.alpha_composite(image)
    return canvas.convert("RGB")


def save_image_grid(images: torch.Tensor, path: str | Path, nrow: int = 4) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    images = (images.clamp(-1, 1) + 1) / 2
    grid = make_grid(images, nrow=nrow)
    save_image(grid, path)
