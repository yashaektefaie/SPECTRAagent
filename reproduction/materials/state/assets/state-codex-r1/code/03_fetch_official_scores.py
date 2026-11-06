#!/usr/bin/env python3
"""Fetch STATE score CSVs through a mirror and verify official Hub hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path


PREFIX = "zeroshot/state_generalization_zeroshot_X_hvg/eval_best.ckpt/"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="https://alpha.hf-mirror.com/arcinstitute/ST-HVG-Tahoe/resolve/main/")
    args = parser.parse_args()
    tree = json.loads(args.tree.read_text())["files"]
    selected = {
        path: metadata
        for path, metadata in tree.items()
        if path.startswith(PREFIX) and path.endswith(".csv")
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index, (path, metadata) in enumerate(sorted(selected.items()), start=1):
        relative = path.removeprefix(PREFIX)
        output = args.output_dir / relative
        expected = metadata["lfs_sha256"]
        expected_size = int(metadata["lfs_size"])
        if output.exists() and output.stat().st_size == expected_size and sha256(output) == expected:
            status = "verified_existing"
        else:
            temporary = output.with_suffix(output.suffix + ".part")
            url = args.base_url + urllib.parse.quote(path, safe="/")
            for attempt in range(1, 4):
                try:
                    request = urllib.request.Request(url, headers={"User-Agent": "STATE-SPECTRA-audit/1.0"})
                    with urllib.request.urlopen(request, timeout=300) as response, temporary.open("wb") as handle:
                        while chunk := response.read(8 * 1024 * 1024):
                            handle.write(chunk)
                    if temporary.stat().st_size != expected_size:
                        raise RuntimeError(
                            f"size mismatch {temporary.stat().st_size} != {expected_size}"
                        )
                    actual = sha256(temporary)
                    if actual != expected:
                        raise RuntimeError(f"sha256 mismatch {actual} != {expected}")
                    os.replace(temporary, output)
                    status = "downloaded_verified"
                    break
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(3 * attempt)
        print(f"[{index}/{len(selected)}] {status}: {relative} ({expected_size} bytes)", flush=True)
        manifest.append(
            {
                "official_hub_path": path,
                "local_path": str(output),
                "bytes": expected_size,
                "sha256": expected,
                "status": status,
                "transport_base": args.base_url,
            }
        )
    (args.output_dir / "verified_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
