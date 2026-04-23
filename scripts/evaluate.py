from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import numpy as np
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run simple image-folder evaluation.")
    parser.add_argument("--image-dir", required=True, help="Folder containing generated images.")
    parser.add_argument("--output", default="outputs/eval/metrics.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_dir = Path(args.image_dir)
    paths = sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
    if not paths:
        raise SystemExit(f"No images found in {image_dir}")

    tensors = []
    sizes = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        sizes.append(image.size)
        arr = torch.from_numpy(np.asarray(image, dtype=np.float32)) / 255.0
        tensors.append(arr.flatten(0, 1))

    pixels = torch.cat(tensors, dim=0)
    metrics = {
        "num_images": len(paths),
        "unique_sizes": sorted({f"{w}x{h}" for w, h in sizes}),
        "mean_rgb": [round(v, 6) for v in pixels.mean(dim=0).tolist()],
        "std_rgb": [round(v, 6) for v in pixels.std(dim=0).tolist()],
        "note": "This is a lightweight sanity evaluation. Add FID/KID/CLIP score for final experiments.",
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
