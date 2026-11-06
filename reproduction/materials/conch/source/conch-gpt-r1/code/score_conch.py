#!/usr/bin/env python3
"""Score frozen target cohorts with official CONCH code and prompt ensemble."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external" / "CONCH"))
from conch.downstream.zeroshot_path import zero_shot_classifier  # noqa: E402
from conch.open_clip_custom import create_model_from_pretrained  # noqa: E402


CLASSES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_rows(dataset: str) -> list[dict[str, str]]:
    with (ROOT / "data" / "manifests" / f"{dataset}.blind.csv").open() as f:
        rows = list(csv.DictReader(f))
    with (ROOT / "data" / "manifests" / f"{dataset}.sealed.tsv").open() as f:
        labels = {r["tile_id"]: r["raw_label"] for r in csv.DictReader(f, delimiter="\t")}
    mapping = json.loads((ROOT / "audit" / "label_maps.json").read_text())[dataset]
    for row in rows:
        row["label"] = mapping.get(labels[row["tile_id"]])
    return [row for row in rows if row["label"] is not None]


class Tiles(Dataset):
    def __init__(self, rows, preprocess):
        self.rows, self.preprocess = rows, preprocess

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        with Image.open(ROOT / row["path"]) as im:
            image = self.preprocess(im.convert("RGB"))
        return image, row["tile_id"], row["label"] or "EXCLUDED"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--datasets", nargs="+", default=["crc_val_7k", "stanford_crc_val_small"])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    state = json.loads((ROOT / "audit" / "state.json").read_text())
    if state["phase"] != "scoring" or not state["target_outputs_accessible"]:
        raise SystemExit("controller guard: run score-start only after frozen-axis unblinding")

    checkpoint = Path(args.checkpoint).resolve()
    model, preprocess = create_model_from_pretrained(
        "conch_ViT-B-16", str(checkpoint), device=args.device, force_image_size=224
    )
    model.eval()
    prompt_path = ROOT / "external" / "CONCH" / "prompts" / "crc100k_prompts_all_per_class.json"
    prompts = json.loads(prompt_path.read_text())["0"]
    classnames = [prompts["classnames"][c] for c in CLASSES]
    with torch.inference_mode():
        classifier = zero_shot_classifier(model, classnames, prompts["templates"], device=args.device)
    checkpoint_hash = sha256(checkpoint)

    for dataset in args.datasets:
        rows = load_rows(dataset)
        loader = DataLoader(Tiles(rows, preprocess), batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        output = ROOT / "scores" / f"{dataset}.conch.csv"
        fields = ["tile_id", "dataset", "label", "prediction", "checkpoint_sha256", "prompt_sha256"]
        fields += [f"logit_{c}" for c in CLASSES] + [f"prob_{c}" for c in CLASSES]
        with output.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for images, ids, labels in loader:
                with torch.inference_mode():
                    embedding = model.encode_image(images.to(args.device))
                    logits = embedding @ classifier
                    probs = torch.softmax(logits * model.logit_scale.exp(), dim=1)
                logits = logits.cpu().numpy()
                probs = probs.cpu().numpy()
                pred = probs.argmax(axis=1)
                for i, tile_id in enumerate(ids):
                    row = {
                        "tile_id": tile_id, "dataset": dataset, "label": labels[i],
                        "prediction": CLASSES[pred[i]], "checkpoint_sha256": checkpoint_hash,
                        "prompt_sha256": sha256(prompt_path),
                    }
                    row.update({f"logit_{c}": float(logits[i, j]) for j, c in enumerate(CLASSES)})
                    row.update({f"prob_{c}": float(probs[i, j]) for j, c in enumerate(CLASSES)})
                    writer.writerow(row)
        print(f"{dataset}: {len(rows)} scored -> {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
