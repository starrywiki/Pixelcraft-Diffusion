from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from pixelcraft.data.transforms import build_image_transform
from pixelcraft.utils.image import pil_to_rgb


def stable_label_id(text: str, num_classes: int) -> int:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % num_classes


class JsonlImageDataset(Dataset):
    def __init__(self, metadata_path: str | Path, root: str | Path, image_size: int, num_classes: int) -> None:
        self.metadata_path = Path(metadata_path)
        self.root = Path(root)
        self.transform = build_image_transform(image_size)
        self.num_classes = num_classes
        self.items = self._load_items(self.metadata_path)

        if not self.items:
            raise ValueError(f"No items found in metadata file: {self.metadata_path}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.items[index]
        image_path = Path(item["image"])
        if not image_path.is_absolute():
            image_path = self.root / image_path

        text = str(item.get("text") or item.get("label") or image_path.stem)
        label = str(item.get("label") or text)
        image = self.transform(pil_to_rgb(image_path))

        return {
            "image": image,
            "condition_id": torch.tensor(stable_label_id(label, self.num_classes), dtype=torch.long),
            "text": text,
            "label": label,
            "path": str(image_path),
        }

    @staticmethod
    def _load_items(path: Path) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items
