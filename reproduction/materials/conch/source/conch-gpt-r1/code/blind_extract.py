#!/usr/bin/env python3
"""Extract image archives under opaque ids while sealing label-bearing paths."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import zipfile
from pathlib import Path, PurePosixPath

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "audit" / "state.json"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--archive", required=True)
    p.add_argument("--dataset", required=True)
    args = p.parse_args()
    state = json.loads(STATE.read_text())
    if state["phase"] not in {"acquired", "features_ready"}:
        raise SystemExit(f"blind extraction only allowed before freeze; phase={state['phase']}")

    archive = (ROOT / args.archive).resolve()
    outdir = ROOT / "data" / "blinded" / args.dataset
    manifest_path = ROOT / "data" / "manifests" / f"{args.dataset}.blind.csv"
    sealed_path = ROOT / "data" / "manifests" / f"{args.dataset}.sealed.tsv"
    outdir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    image_suffixes = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    public_rows = []
    sealed_rows = []
    with zipfile.ZipFile(archive) as zf:
        members = [m for m in zf.infolist() if not m.is_dir() and PurePosixPath(m.filename).suffix.lower() in image_suffixes]
        for member in members:
            posix = PurePosixPath(member.filename)
            if ".." in posix.parts:
                raise RuntimeError("unsafe archive member")
            raw = zf.read(member)
            digest = hashlib.sha256(raw).hexdigest()
            blind_id = hashlib.sha256((args.dataset + "\0" + member.filename).encode()).hexdigest()[:24]
            suffix = posix.suffix.lower()
            destination = outdir / f"{blind_id}{suffix}"
            if not destination.exists():
                destination.write_bytes(raw)
            with Image.open(io.BytesIO(raw)) as im:
                width, height = im.size
            label = posix.parent.name
            public_rows.append({
                "tile_id": blind_id,
                "dataset": args.dataset,
                "path": str(destination.relative_to(ROOT)),
                "bytes": len(raw),
                "sha256": digest,
                "width": width,
                "height": height,
            })
            sealed_rows.append((blind_id, label, member.filename))

    with manifest_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(public_rows[0]))
        w.writeheader()
        w.writerows(public_rows)
    with sealed_path.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["tile_id", "raw_label", "archive_member"])
        w.writerows(sealed_rows)
    sealed_path.chmod(0)
    print(json.dumps({"dataset": args.dataset, "images": len(public_rows), "manifest": str(manifest_path.relative_to(ROOT)), "labels": "sealed"}))


if __name__ == "__main__":
    main()
