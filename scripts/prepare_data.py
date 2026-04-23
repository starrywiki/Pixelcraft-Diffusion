from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from pixelcraft.utils.image import pil_to_rgb


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare image folders into resized images and JSONL metadata.")
    parser.add_argument("--input-dir", required=True, help="Folder containing raw images.")
    parser.add_argument("--output-dir", default="data/processed", help="Folder for resized images.")
    parser.add_argument("--metadata-dir", default="data/metadata", help="Folder for train/val/test JSONL files.")
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source", default="local")
    return parser.parse_args()


def infer_label(path: Path) -> str:
    if path.parent.name not in {"raw", "pixelart", "emoji"}:
        return path.parent.name.replace("_", " ")
    return path.stem.replace("_", " ").replace("-", " ")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    metadata_dir = Path(args.metadata_dir)

    images = sorted(p for p in input_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise SystemExit(f"No images found in {input_dir}")

    random.seed(args.seed)
    random.shuffle(images)

    rows = []
    for index, src in enumerate(images):
        split_name = "all"
        dst = output_dir / split_name / f"{index:06d}.png"
        dst.parent.mkdir(parents=True, exist_ok=True)
        image = pil_to_rgb(src).resize((args.image_size, args.image_size), Image.Resampling.NEAREST)
        image.save(dst)
        label = infer_label(src)
        rows.append(
            {
                "image": str(dst.relative_to(Path("data"))) if str(dst).startswith("data/") else str(dst),
                "text": label,
                "label": label,
                "source": args.source,
            }
        )

    total = len(rows)
    val_count = int(total * args.val_ratio)
    test_count = int(total * args.test_ratio)
    train_count = total - val_count - test_count

    train_rows = rows[:train_count]
    val_rows = rows[train_count : train_count + val_count]
    test_rows = rows[train_count + val_count :]

    write_jsonl(metadata_dir / "train.jsonl", train_rows)
    write_jsonl(metadata_dir / "val.jsonl", val_rows)
    if test_rows:
        write_jsonl(metadata_dir / "test.jsonl", test_rows)

    print(f"Prepared {len(train_rows)} train, {len(val_rows)} val, {len(test_rows)} test images.")


if __name__ == "__main__":
    main()
