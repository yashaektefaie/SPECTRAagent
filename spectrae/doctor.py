"""Environment checks for a local SPECTRA installation."""

import argparse
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def _module_status(name: str) -> Dict[str, Any]:
    spec = importlib.util.find_spec(name)
    return {
        "name": name,
        "ok": spec is not None,
        "origin": getattr(spec, "origin", "") if spec is not None else "",
    }


def _command_status(name: str) -> Dict[str, Any]:
    path = shutil.which(name)
    return {"name": name, "ok": path is not None, "path": path or ""}


def _default_scratch_root() -> Path:
    configured = os.environ.get("SPECTRA_SCRATCH_ROOT")
    if configured:
        return Path(configured).expanduser()

    cache_root = os.environ.get("XDG_CACHE_HOME")
    if cache_root:
        return Path(cache_root).expanduser() / "spectra" / "runs"
    return Path.home() / ".cache" / "spectra" / "runs"


def _scratch_status(path: Optional[str] = None) -> Dict[str, Any]:
    root = Path(path).expanduser() if path else _default_scratch_root()
    result: Dict[str, Any] = {"path": str(root), "ok": False, "created": False, "error": ""}
    try:
        existed = root.exists()
        root.mkdir(parents=True, exist_ok=True)
        probe = root / ".spectra_write_test"
        probe.write_text("ok\n", encoding="utf-8")
        probe.unlink()
        result["ok"] = True
        result["created"] = not existed
    except Exception as exc:  # pragma: no cover - defensive command boundary.
        result["error"] = str(exc)
    return result


def collect_status(scratch_root: Optional[str] = None, skip_torch: bool = False) -> Dict[str, Any]:
    required_modules = ["spectrae", "numpy", "pandas", "sklearn", "networkx"]
    optional_modules = ["mcp", "fastmcp"]
    if not skip_torch:
        optional_modules.append("torch")

    modules = {
        "required": [_module_status(name) for name in required_modules],
        "optional": [_module_status(name) for name in optional_modules],
    }
    commands = [_command_status(name) for name in ("spectra", "spectra-mcp", "spectra-doctor")]
    scratch = _scratch_status(scratch_root)

    ok = (
        all(item["ok"] for item in modules["required"])
        and scratch["ok"]
        and any(item["name"] == "spectra" and item["ok"] for item in commands)
    )
    mcp_ready = (
        any(item["name"] == "mcp" and item["ok"] for item in modules["optional"])
        or any(item["name"] == "fastmcp" and item["ok"] for item in modules["optional"])
    )

    return {
        "ok": ok,
        "mcp_ready": mcp_ready,
        "python": sys.executable,
        "python_version": sys.version.split()[0],
        "modules": modules,
        "commands": commands,
        "scratch": scratch,
        "notes": [
            "Install MCP support with `pipx install 'spectrae[mcp]'` or `pip install 'spectrae[mcp]'`."
            if not mcp_ready
            else "MCP dependency is importable.",
            "Set SPECTRA_SCRATCH_ROOT to control where SPECTRA writes run artifacts.",
        ],
    }


def _print_human(status: Dict[str, Any]) -> None:
    print("SPECTRA doctor")
    print(f"  python: {status['python']} ({status['python_version']})")
    print(f"  install ok: {'yes' if status['ok'] else 'no'}")
    print(f"  mcp ready: {'yes' if status['mcp_ready'] else 'no'}")
    print(f"  scratch: {status['scratch']['path']} ({'ok' if status['scratch']['ok'] else 'failed'})")
    if status["scratch"].get("error"):
        print(f"    error: {status['scratch']['error']}")

    print("  commands:")
    for item in status["commands"]:
        print(f"    {item['name']}: {'ok' if item['ok'] else 'missing'} {item['path']}")

    print("  required modules:")
    for item in status["modules"]["required"]:
        print(f"    {item['name']}: {'ok' if item['ok'] else 'missing'}")

    print("  optional modules:")
    for item in status["modules"]["optional"]:
        print(f"    {item['name']}: {'ok' if item['ok'] else 'missing'}")

    for note in status["notes"]:
        print(f"  note: {note}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Check a local SPECTRA installation.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument("--scratch-root", default="", help="Scratch/run root to create and test.")
    parser.add_argument("--skip-torch", action="store_true", help="Skip torch import probing.")
    args = parser.parse_args(argv)

    status = collect_status(scratch_root=args.scratch_root or None, skip_torch=args.skip_torch)
    if args.json:
        print(json.dumps(status, indent=2, sort_keys=True))
    else:
        _print_human(status)
    return 0 if status["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
