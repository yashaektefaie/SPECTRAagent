"""Read-only SPECTRA knowledge MCP server.

This server intentionally exposes SPECTRA protocol guidance, prior findings,
and saved artifacts only. It does not prepare, launch, or execute audits.
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


SERVER_NAME = "spectra-knowledge-mcp"
ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("SPECTRA_KNOWLEDGE_DATA_ROOT", ROOT / "data")).resolve()
STORE_PATH = DATA_ROOT / "store.json"
PROTOCOL_PATH = DATA_ROOT / "protocol" / "current.md"
ARTIFACT_ROOT = DATA_ROOT / "artifacts"


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_store() -> Dict[str, Any]:
    return _load_json(STORE_PATH)


def _read_text(path: Path, max_chars: Optional[int] = None) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    truncated = False
    if max_chars is not None and max_chars >= 0 and len(text) > max_chars:
        text = text[:max_chars]
        truncated = True
    return {
        "path": str(path),
        "chars": path.stat().st_size,
        "truncated": truncated,
        "text": text,
    }


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _matches_filter(value: str, requested: str) -> bool:
    if not requested:
        return True
    return _normalize(requested) in _normalize(value)


def _stringify_record(record: Dict[str, Any]) -> str:
    return json.dumps(record, sort_keys=True, ensure_ascii=False)


def _artifact_id_for(path: Path) -> str:
    rel = path.relative_to(ARTIFACT_ROOT).as_posix()
    return rel.replace("/", ":")


def _artifact_index() -> Dict[str, Dict[str, Any]]:
    artifacts: Dict[str, Dict[str, Any]] = {}
    if not ARTIFACT_ROOT.exists():
        return artifacts
    for path in sorted(ARTIFACT_ROOT.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(ARTIFACT_ROOT).as_posix()
        parts = rel.split("/")
        artifact_id = _artifact_id_for(path)
        artifacts[artifact_id] = {
            "artifact_id": artifact_id,
            "relative_path": rel,
            "path": str(path),
            "model": parts[0] if parts else "",
            "format": path.suffix.lstrip(".") or "text",
            "bytes": path.stat().st_size,
        }
    return artifacts


def _artifact_aliases(store: Dict[str, Any]) -> Dict[str, str]:
    aliases = dict(store.get("artifact_aliases", {}))
    for artifact_id in _artifact_index():
        aliases.setdefault(artifact_id, artifact_id)
    return aliases


def _resolve_artifact_id(artifact_id: str) -> str:
    store = _load_store()
    aliases = _artifact_aliases(store)
    resolved = aliases.get(artifact_id, artifact_id)
    index = _artifact_index()
    if resolved not in index:
        available = sorted(index)[:25]
        raise ValueError(
            "Unknown artifact_id '%s'. Example available ids: %s"
            % (artifact_id, ", ".join(available))
        )
    return resolved


def _filter_findings(
    findings: Iterable[Dict[str, Any]],
    model: str = "",
    domain: str = "",
    decision: str = "",
    axis: str = "",
) -> List[Dict[str, Any]]:
    results = []
    for finding in findings:
        if not _matches_filter(finding.get("model", ""), model):
            continue
        if not _matches_filter(finding.get("domain", ""), domain):
            continue
        if not _matches_filter(finding.get("validity", {}).get("decision", ""), decision):
            continue
        if not _matches_filter(finding.get("axis", {}).get("name", ""), axis):
            continue
        results.append(finding)
    return results


def _summary_finding(finding: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "finding_id": finding.get("finding_id"),
        "model": finding.get("model"),
        "domain": finding.get("domain"),
        "axis": finding.get("axis", {}).get("name"),
        "decision": finding.get("validity", {}).get("decision"),
        "claim_boundary": finding.get("claim_boundary"),
        "summary": finding.get("summary"),
        "key_artifact_ids": finding.get("key_artifact_ids", []),
    }


def _protocol_section(markdown: str, section: str) -> str:
    if not section or section == "current":
        return markdown
    wanted = _normalize(section)
    lines = markdown.splitlines()
    start = None
    level = None
    for idx, line in enumerate(lines):
        if not line.startswith("#"):
            continue
        title = line.lstrip("#").strip()
        if _normalize(title) == wanted:
            start = idx
            level = len(line) - len(line.lstrip("#"))
            break
    if start is None:
        raise ValueError("Unknown protocol section '%s'" % section)
    end = len(lines)
    for idx in range(start + 1, len(lines)):
        line = lines[idx]
        if line.startswith("#"):
            current_level = len(line) - len(line.lstrip("#"))
            if current_level <= level:
                end = idx
                break
    return "\n".join(lines[start:end]).strip() + "\n"


def list_spectra_models() -> Dict[str, Any]:
    """List models with stored SPECTRA findings."""
    store = _load_store()
    return {"models": store.get("models", [])}


def list_spectra_findings(
    model: str = "",
    domain: str = "",
    decision: str = "",
    axis: str = "",
) -> Dict[str, Any]:
    """List stored SPECTRA findings with optional filters."""
    store = _load_store()
    findings = _filter_findings(store.get("findings", []), model, domain, decision, axis)
    return {"count": len(findings), "findings": [_summary_finding(item) for item in findings]}


def search_spectra_findings(
    query: str = "",
    model: str = "",
    domain: str = "",
    axis: str = "",
    decision: str = "",
    top_k: int = 10,
) -> Dict[str, Any]:
    """Search stored finding records. This is metadata search only."""
    store = _load_store()
    findings = _filter_findings(store.get("findings", []), model, domain, decision, axis)
    tokens = [token for token in _normalize(query).split() if token]
    scored = []
    for finding in findings:
        haystack = _normalize(_stringify_record(finding))
        if tokens:
            score = sum(haystack.count(token) for token in tokens)
            if score == 0:
                continue
        else:
            score = 1
        scored.append((score, finding))
    scored.sort(key=lambda item: (-item[0], item[1].get("finding_id", "")))
    limit = max(1, min(int(top_k), 50))
    return {
        "query": query,
        "count": len(scored[:limit]),
        "findings": [_summary_finding(item) for _, item in scored[:limit]],
    }


def get_spectra_finding(finding_id: str) -> Dict[str, Any]:
    """Return one stored SPECTRA finding by id."""
    store = _load_store()
    for finding in store.get("findings", []):
        if finding.get("finding_id") == finding_id:
            return finding
    raise ValueError("Unknown finding_id '%s'" % finding_id)


def list_spectra_runs(model: str = "") -> Dict[str, Any]:
    """List stored SPECTRA run records."""
    store = _load_store()
    runs = []
    for run in store.get("runs", []):
        if _matches_filter(run.get("model", ""), model):
            runs.append(run)
    return {"count": len(runs), "runs": runs}


def get_spectra_run(run_id: str) -> Dict[str, Any]:
    """Return one stored run record by id."""
    store = _load_store()
    for run in store.get("runs", []):
        if run.get("run_id") == run_id:
            return run
    raise ValueError("Unknown run_id '%s'" % run_id)


def list_spectra_artifacts(
    model: str = "",
    finding_id: str = "",
    run_id: str = "",
    query: str = "",
) -> Dict[str, Any]:
    """List readable saved artifacts. This never executes an audit."""
    store = _load_store()
    index = _artifact_index()
    requested_ids = set(index)
    if finding_id:
        finding = get_spectra_finding(finding_id)
        requested_ids &= set(finding.get("key_artifact_ids", []))
    if run_id:
        run = get_spectra_run(run_id)
        requested_ids &= set(run.get("artifact_ids", []))
    results = []
    tokens = [token for token in _normalize(query).split() if token]
    for artifact_id, artifact in index.items():
        if artifact_id not in requested_ids:
            continue
        if model and not _matches_filter(artifact.get("model", ""), model):
            continue
        searchable = _normalize(json.dumps(artifact, sort_keys=True))
        if tokens and not all(token in searchable for token in tokens):
            continue
        results.append(artifact)
    return {"count": len(results), "artifacts": results}


def get_spectra_artifact(artifact_id: str, max_chars: int = 20000) -> Dict[str, Any]:
    """Read a text/CSV/JSON/Markdown artifact by id, truncated by max_chars."""
    resolved = _resolve_artifact_id(artifact_id)
    artifact = _artifact_index()[resolved]
    path = Path(artifact["path"]).resolve()
    if ARTIFACT_ROOT not in path.parents:
        raise ValueError("Artifact path escapes artifact root")
    limit = max(0, min(int(max_chars), 250000))
    payload = _read_text(path, max_chars=limit)
    payload["artifact"] = artifact
    return payload


def get_spectra_protocol(section: str = "current") -> Dict[str, Any]:
    """Return the current SPECTRA protocol or one markdown section."""
    markdown = PROTOCOL_PATH.read_text(encoding="utf-8")
    return {
        "section": section,
        "text": _protocol_section(markdown, section),
    }


def suggest_next_spectra_move(model: str, question: str = "") -> Dict[str, Any]:
    """Return protocol-guided next-step suggestions from stored knowledge only."""
    store = _load_store()
    matches = _filter_findings(store.get("findings", []), model=model)
    if not matches:
        return {
            "model": model,
            "question": question,
            "suggestion": (
                "No stored finding exists for this model. Use the protocol resource "
                "to start a new SPECTRA audit: define the scientific unit, propose "
                "prospective axes, freeze a split-valid SPC, and confirm any "
                "outcome-mined axis on new or adequate evidence."
            ),
            "finding_ids": [],
        }
    axes = [item.get("axis", {}).get("name") for item in matches]
    return {
        "model": model,
        "question": question,
        "finding_ids": [item.get("finding_id") for item in matches],
        "known_axes": axes,
        "suggestion": (
            "Treat stored findings as hypothesis memory and evidence boundaries, "
            "not automatic closure for a new claim. Reuse negative ledgers, "
            "freeze any new axis before fresh target-model scoring, confirm it on "
            "new or explicitly adequate evidence, and report whether the claim is "
            "valid, weak, exploratory, or invalid."
        ),
    }


def create_mcp_server(host: str = "127.0.0.1", port: int = 8000) -> Any:
    """Create the read-only MCP server."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise ImportError(
            "The MCP package is required. Install with: pip install -r requirements.txt"
        ) from exc

    instructions = (
        "Read-only SPECTRA knowledge server. Exposes the current SPECTRA "
        "single-controller protocol, stored finding records, run summaries, and "
        "saved text/CSV/JSON artifacts. It must not run audits, launch agents, "
        "download datasets, call models, or mutate artifacts."
    )
    try:
        mcp = FastMCP(name=SERVER_NAME, instructions=instructions, host=host, port=port)
    except TypeError as exc:
        message = str(exc)
        if "host" not in message and "port" not in message:
            raise
        os.environ.setdefault("FASTMCP_HOST", host)
        os.environ.setdefault("FASTMCP_PORT", str(port))
        mcp = FastMCP(name=SERVER_NAME, instructions=instructions)

    @mcp.resource("spectra://protocol/current")
    def protocol_resource() -> str:
        """Current SPECTRA protocol and controller loop."""
        return PROTOCOL_PATH.read_text(encoding="utf-8")

    @mcp.resource("spectra://findings/index")
    def findings_index_resource() -> Dict[str, Any]:
        """Index of stored SPECTRA findings."""
        return list_spectra_findings()

    @mcp.resource("spectra://models/index")
    def models_index_resource() -> Dict[str, Any]:
        """Index of models with stored SPECTRA findings."""
        return list_spectra_models()

    @mcp.resource("spectra://runs/index")
    def runs_index_resource() -> Dict[str, Any]:
        """Index of stored SPECTRA runs."""
        return list_spectra_runs()

    @mcp.resource("spectra://artifacts/index")
    def artifacts_index_resource() -> Dict[str, Any]:
        """Index of readable saved SPECTRA artifacts."""
        return list_spectra_artifacts()

    mcp.tool(name="list_spectra_models")(list_spectra_models)
    mcp.tool(name="list_spectra_findings")(list_spectra_findings)
    mcp.tool(name="search_spectra_findings")(search_spectra_findings)
    mcp.tool(name="get_spectra_finding")(get_spectra_finding)
    mcp.tool(name="list_spectra_runs")(list_spectra_runs)
    mcp.tool(name="get_spectra_run")(get_spectra_run)
    mcp.tool(name="list_spectra_artifacts")(list_spectra_artifacts)
    mcp.tool(name="get_spectra_artifact")(get_spectra_artifact)
    mcp.tool(name="get_spectra_protocol")(get_spectra_protocol)
    mcp.tool(name="suggest_next_spectra_move")(suggest_next_spectra_move)
    return mcp


try:
    mcp = create_mcp_server()
except ImportError:
    mcp = None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the read-only SPECTRA knowledge MCP server.")
    parser.add_argument("--transport", default="stdio", choices=("stdio", "streamable-http", "sse"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    args = parser.parse_args()
    server = create_mcp_server(host=args.host, port=args.port)
    server.run(transport=args.transport)


if __name__ == "__main__":
    main()
