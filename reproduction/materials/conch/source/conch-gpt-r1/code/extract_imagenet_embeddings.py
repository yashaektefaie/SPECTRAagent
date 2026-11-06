#!/usr/bin/env python3
"""Extract fixed ImageNet MobileNetV3-Small embeddings on the same tiles."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


ROOT = Path(__file__).resolve().parents[1]


class Tiles(Dataset):
    def __init__(self, rows, transform):
        self.rows, self.transform = rows, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        with Image.open(ROOT / row["path"]) as im:
            image = self.transform(im.convert("RGB"))
        return image, row["tile_id"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--balanced-per-class", type=int)
    args = p.parse_args()
    state = json.loads((ROOT / "audit" / "state.json").read_text())
    if not state["target_labels_accessible"]:
        raise SystemExit("controller guard: embeddings are deferred until the frozen contract is released")

    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    model = mobilenet_v3_small(weights=weights)
    model.classifier = torch.nn.Identity()
    model.eval()
    torch.set_num_threads(min(12, torch.get_num_threads()))
    for dataset in args.datasets:
        with (ROOT / "data" / "manifests" / f"{dataset}.blind.csv").open() as f:
            rows = list(csv.DictReader(f))
        with (ROOT / "data" / "manifests" / f"{dataset}.sealed.tsv").open() as f:
            raw = {r["tile_id"]: r["raw_label"] for r in csv.DictReader(f, delimiter="\t")}
        mapping = json.loads((ROOT / "audit" / "label_maps.json").read_text())[dataset]
        rows = [r for r in rows if mapping.get(raw[r["tile_id"]]) is not None]
        if args.balanced_per_class is not None:
            selected = []
            for canonical in sorted({v for v in mapping.values() if v is not None}):
                candidates = [r for r in rows if mapping.get(raw[r["tile_id"]]) == canonical]
                candidates.sort(key=lambda r: __import__("hashlib").sha256(r["tile_id"].encode()).hexdigest())
                selected.extend(candidates[:args.balanced_per_class])
            rows = sorted(selected, key=lambda r: r["tile_id"])
        loader = DataLoader(Tiles(rows, weights.transforms()), batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        embeddings, ids = [], []
        for images, tile_ids in loader:
            with torch.inference_mode():
                embeddings.append(model(images).cpu().numpy().astype(np.float32))
            ids.extend(tile_ids)
        output = ROOT / "features" / f"{dataset}.mobilenet_v3_small_imagenet1k_v1.npz"
        np.savez_compressed(output, tile_id=np.asarray(ids), embedding=np.concatenate(embeddings))
        print(f"{dataset}: {len(ids)} MobileNet embeddings -> {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
