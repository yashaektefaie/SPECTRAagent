"""Build a public download manifest for curated SPECTRA artifacts.

The MCP server is for metadata and previews, not bulk file transfer. This
script selects public result artifacts, records stable HTTPS URLs plus checksums,
and optionally materializes a read-only download tree that Caddy can serve.
"""

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List
from urllib.parse import quote


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"
ARTIFACT_ROOT = DATA_ROOT / "artifacts"
PUBLIC_ROOT = DATA_ROOT / "public_downloads"
STORE_PATH = DATA_ROOT / "store.json"
MANIFEST_PATH = DATA_ROOT / "downloads.json"
DEFAULT_BASE_URL = "https://spectra.yashaektefaie.com/downloads"

PUBLIC_SUFFIXES = {".csv", ".json", ".md", ".txt", ".png", ".pdf"}
EXCLUDED_NAMES = {
    "commands_run.json",
    "controller_prompt.md",
    "controller_session.log",
    "session_contract.json",
    "session_manifest.json",
    "spectra_loop_state.json",
}
EXCLUDED_PARTS = {"__pycache__"}
EXCLUDED_SUBSTRINGS = ("controller_",)


def load_store() -> Dict[str, Any]:
    if not STORE_PATH.exists():
        return {}
    return json.loads(STORE_PATH.read_text(encoding="utf-8"))


def artifact_id_for(path: Path) -> str:
    rel = path.relative_to(ARTIFACT_ROOT).as_posix()
    return rel.replace("/", ":")


def should_publish(path: Path) -> bool:
    rel_parts = path.relative_to(ARTIFACT_ROOT).parts
    if any(part in EXCLUDED_PARTS for part in rel_parts):
        return False
    if path.name in EXCLUDED_NAMES:
        return False
    rel = path.relative_to(ARTIFACT_ROOT).as_posix()
    if any(token in rel for token in EXCLUDED_SUBSTRINGS):
        return False
    return path.suffix.lower() in PUBLIC_SUFFIXES


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def line_count(path: Path) -> int:
    count = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            count += chunk.count(b"\n")
    return count


def build_membership(store: Dict[str, Any]) -> Dict[str, Dict[str, List[str]]]:
    membership: Dict[str, Dict[str, List[str]]] = {}
    for run in store.get("runs", []):
        run_id = run.get("run_id")
        if not run_id:
            continue
        for artifact_id in run.get("artifact_ids", []):
            membership.setdefault(artifact_id, {"run_ids": [], "finding_ids": []})
            membership[artifact_id]["run_ids"].append(run_id)
    for finding in store.get("findings", []):
        finding_id = finding.get("finding_id")
        if not finding_id:
            continue
        for artifact_id in finding.get("key_artifact_ids", []):
            membership.setdefault(artifact_id, {"run_ids": [], "finding_ids": []})
            membership[artifact_id]["finding_ids"].append(finding_id)
    return membership


def unique(values: Iterable[str]) -> List[str]:
    out = []
    seen = set()
    for value in values:
        if value and value not in seen:
            out.append(value)
            seen.add(value)
    return out


def build_manifest(base_url: str) -> Dict[str, Any]:
    store = load_store()
    membership = build_membership(store)
    records = []
    if ARTIFACT_ROOT.exists():
        for path in sorted(ARTIFACT_ROOT.rglob("*")):
            if not path.is_file() or not should_publish(path):
                continue
            rel = path.relative_to(ARTIFACT_ROOT).as_posix()
            artifact_id = artifact_id_for(path)
            parts = rel.split("/")
            info = membership.get(artifact_id, {"run_ids": [], "finding_ids": []})
            download_path = "/downloads/%s" % quote(rel, safe="/")
            rows = None
            header = None
            if path.suffix.lower() == ".csv":
                lines = line_count(path)
                rows = max(0, lines - 1)
                with path.open("r", encoding="utf-8", errors="replace") as handle:
                    header = handle.readline().strip()
            records.append(
                {
                    "download_id": artifact_id,
                    "artifact_id": artifact_id,
                    "model": parts[0] if parts else "",
                    "relative_path": rel,
                    "format": path.suffix.lstrip(".") or "text",
                    "bytes": path.stat().st_size,
                    "rows": rows,
                    "csv_header": header,
                    "sha256": sha256_file(path),
                    "download_path": download_path,
                    "download_url": "%s/%s" % (base_url.rstrip("/"), quote(rel, safe="/")),
                    "finding_ids": unique(info.get("finding_ids", [])),
                    "run_ids": unique(info.get("run_ids", [])),
                }
            )
    records.sort(key=lambda item: (item["model"], item["relative_path"]))
    return {
        "schema_version": "0.1.0",
        "base_url": base_url.rstrip("/"),
        "public_path_prefix": "/downloads",
        "description": (
            "Curated public file downloads for SPECTRA result artifacts. "
            "MCP tools return this manifest; agents should use normal HTTPS "
            "download clients for bulk CSV/JSON/Markdown artifacts."
        ),
        "count": len(records),
        "records": records,
    }


def materialize(manifest: Dict[str, Any], mode: str = "hardlink", clean: bool = False) -> None:
    if clean and PUBLIC_ROOT.exists():
        shutil.rmtree(str(PUBLIC_ROOT))
    PUBLIC_ROOT.mkdir(parents=True, exist_ok=True)
    for record in manifest.get("records", []):
        src = ARTIFACT_ROOT / record["relative_path"]
        dst = PUBLIC_ROOT / record["relative_path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if mode == "copy":
            shutil.copy2(str(src), str(dst))
        elif mode == "symlink":
            target = os.path.relpath(str(src), str(dst.parent))
            os.symlink(target, str(dst))
        else:
            try:
                os.link(str(src), str(dst))
            except OSError:
                shutil.copy2(str(src), str(dst))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SPECTRA public download manifest.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--out", default=str(MANIFEST_PATH))
    parser.add_argument("--materialize", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--mode", choices=("hardlink", "copy", "symlink"), default="hardlink")
    args = parser.parse_args()

    manifest = build_manifest(args.base_url)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.materialize:
        materialize(manifest, mode=args.mode, clean=args.clean)
    print(json.dumps({"out": str(out), "count": manifest["count"]}, indent=2))


if __name__ == "__main__":
    main()
