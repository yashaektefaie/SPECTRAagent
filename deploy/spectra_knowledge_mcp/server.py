"""SPECTRA knowledge MCP server.

This server exposes SPECTRA protocol guidance, prior findings, saved artifacts,
and an authenticated pending-submission queue. It does not prepare, launch, or
execute audits, and submitted findings do not mutate the canonical store until
they are reviewed outside this server.
"""

import argparse
import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


SERVER_NAME = "spectra-knowledge-mcp"
ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("SPECTRA_KNOWLEDGE_DATA_ROOT", ROOT / "data")).resolve()
STORE_PATH = DATA_ROOT / "store.json"
PROTOCOL_PATH = DATA_ROOT / "protocol" / "current.md"
PROVENANCE_PATH = DATA_ROOT / "provenance.json"
PROVENANCE_SCHEMA_PATH = DATA_ROOT / "provenance_schema.json"
DOWNLOADS_PATH = DATA_ROOT / "downloads.json"
SUBMISSION_SCHEMA_PATH = DATA_ROOT / "submission_schema.json"
ARTIFACT_ROOT = DATA_ROOT / "artifacts"
SUBMISSION_ROOT = DATA_ROOT / "submissions"
SUBMISSION_PENDING_ROOT = SUBMISSION_ROOT / "pending"
MAX_SUBMISSION_JSON_BYTES = int(os.environ.get("SPECTRA_MAX_SUBMISSION_JSON_BYTES", "1000000"))
SUBMISSION_STATUS_VALUES = ("pending_review", "accepted", "rejected", "needs_revision")


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_store() -> Dict[str, Any]:
    return _load_json(STORE_PATH)


def _load_json_if_exists(path: Path, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not path.exists():
        return {} if default is None else default
    return _load_json(path)


def _load_provenance() -> Dict[str, Any]:
    return _load_json_if_exists(PROVENANCE_PATH, {"schema_version": "missing", "records": []})


def _load_provenance_schema() -> Dict[str, Any]:
    return _load_json_if_exists(PROVENANCE_SCHEMA_PATH, {"schema_version": "missing"})


def _load_downloads() -> Dict[str, Any]:
    return _load_json_if_exists(DOWNLOADS_PATH, {"schema_version": "missing", "records": []})


def _load_submission_schema() -> Dict[str, Any]:
    return _load_json_if_exists(
        SUBMISSION_SCHEMA_PATH,
        {
            "schema_version": "missing",
            "purpose": "Schema for pending SPECTRA finding submissions.",
            "status_values": list(SUBMISSION_STATUS_VALUES),
        },
    )


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


def _utc_timestamp() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _slugify(value: str, fallback: str = "unknown") -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value.strip()).strip("-._").lower()
    return slug[:64] if slug else fallback


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _coerce_json_object(payload: Any, name: str) -> Dict[str, Any]:
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError("%s must be a JSON object or JSON object string: %s" % (name, exc)) from exc
    if not isinstance(payload, dict):
        raise ValueError("%s must be a JSON object" % name)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    if len(encoded.encode("utf-8")) > MAX_SUBMISSION_JSON_BYTES:
        raise ValueError(
            "%s is too large for MCP submission (%d byte limit). "
            "Reference large artifacts by URL instead."
            % (name, MAX_SUBMISSION_JSON_BYTES)
        )
    return json.loads(encoded)


def _record_list(value: Any) -> List[Dict[str, Any]]:
    if isinstance(value, dict):
        records = value.get("records")
        if records is None:
            return [value]
    else:
        records = value
    if not isinstance(records, list):
        return []
    return [item for item in records if isinstance(item, dict)]


def _require_fields(record: Dict[str, Any], fields: Iterable[str], prefix: str, errors: List[str]) -> None:
    for field in fields:
        if not record.get(field):
            errors.append("%s.%s is required" % (prefix, field))


def _submission_queue_enabled() -> bool:
    return bool(os.environ.get("SPECTRA_SUBMISSION_TOKEN"))


def _require_submission_auth(auth_token: str) -> None:
    expected = os.environ.get("SPECTRA_SUBMISSION_TOKEN")
    if not expected:
        raise PermissionError(
            "SPECTRA submission queue is disabled. Set SPECTRA_SUBMISSION_TOKEN on the MCP host."
        )
    if not auth_token or auth_token != expected:
        raise PermissionError("Invalid SPECTRA submission auth token")


def _submission_id_is_safe(submission_id: str) -> bool:
    return bool(re.match(r"^[A-Za-z0-9_.-]{8,180}$", submission_id or ""))


def _pending_submission_path(submission_id: str) -> Path:
    if not _submission_id_is_safe(submission_id):
        raise ValueError("Invalid submission_id '%s'" % submission_id)
    root = SUBMISSION_PENDING_ROOT.resolve()
    path = (root / submission_id).resolve()
    if root != path.parent:
        raise ValueError("Submission path escapes pending queue")
    return path


def _generate_submission_id(submission: Dict[str, Any]) -> str:
    finding = submission.get("finding", {})
    model_slug = _slugify(str(finding.get("model") or "unknown"))
    finding_slug = _slugify(str(finding.get("finding_id") or submission.get("title") or "finding"))
    digest = hashlib.sha256(_stringify_record(submission).encode("utf-8")).hexdigest()[:10]
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    return "sub_%s_%s_%s_%s" % (timestamp, model_slug, finding_slug[:40], digest)


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


def _filter_provenance_records(
    records: Iterable[Dict[str, Any]],
    model: str = "",
    finding_id: str = "",
    run_id: str = "",
    status: str = "",
) -> List[Dict[str, Any]]:
    results = []
    for record in records:
        if model and not _matches_filter(record.get("model", ""), model):
            continue
        if finding_id and record.get("finding_id") != finding_id:
            continue
        if run_id and record.get("run_id") != run_id:
            continue
        if status and not _matches_filter(record.get("status", ""), status):
            continue
        results.append(record)
    return results


def _summary_provenance_record(record: Dict[str, Any]) -> Dict[str, Any]:
    datasets = record.get("dataset_sources", [])
    metadata = record.get("metadata_sources", [])
    model_source = record.get("model_source", {})
    return {
        "finding_id": record.get("finding_id"),
        "run_id": record.get("run_id"),
        "model": record.get("model"),
        "status": record.get("status"),
        "model_source": model_source.get("name"),
        "execution_mode": model_source.get("execution_mode"),
        "dataset_source_count": len(datasets),
        "metadata_source_count": len(metadata),
        "download_count": len(record.get("download_ids", [])),
        "known_gap_count": len(record.get("known_gaps", [])),
        "provenance_artifact_ids": record.get("provenance_artifact_ids", []),
        "download_ids": record.get("download_ids", []),
    }


def _source_rows(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    model_source = record.get("model_source", {})
    if model_source:
        rows.append(
            {
                "source_type": "model",
                "finding_id": record.get("finding_id"),
                "run_id": record.get("run_id"),
                "model": record.get("model"),
                "name": model_source.get("name"),
                "source_url_or_repo": model_source.get("source_url_or_repo"),
                "access_route": model_source.get("execution_mode"),
                "used_for": "target model",
            }
        )
        weights = model_source.get("weights_or_checkpoint", {})
        if weights:
            rows.append(
                {
                    "source_type": "weights_or_checkpoint",
                    "finding_id": record.get("finding_id"),
                    "run_id": record.get("run_id"),
                    "model": record.get("model"),
                    "name": weights.get("filename") or model_source.get("name"),
                    "source_url_or_repo": weights.get("source_url_or_repo"),
                    "access_route": weights.get("download_command_or_method") or weights.get("load_call"),
                    "used_for": "model weights/checkpoint or official precomputed score provenance",
                }
            )
    for source_type, key in (("dataset", "dataset_sources"), ("metadata", "metadata_sources")):
        for item in record.get(key, []):
            rows.append(
                {
                    "source_type": source_type,
                    "finding_id": record.get("finding_id"),
                    "run_id": record.get("run_id"),
                    "model": record.get("model"),
                    "name": item.get("name"),
                    "source_url_or_repo": item.get("source_url_or_repo"),
                    "access_route": item.get("access_route"),
                    "unit": item.get("unit"),
                    "rows_or_units": item.get("rows_or_units"),
                    "used_for": item.get("used_for"),
                    "local_or_artifact_path": item.get("local_or_artifact_path"),
                }
            )
    for item in record.get("retrieval_and_download", []):
        rows.append(
            {
                "source_type": "retrieval",
                "finding_id": record.get("finding_id"),
                "run_id": record.get("run_id"),
                "model": record.get("model"),
                "name": item.get("phase"),
                "source_url_or_repo": "",
                "access_route": item.get("command_or_method"),
                "used_for": item.get("purpose"),
            }
        )
    return rows


def _validation_status(record: Dict[str, Any]) -> Dict[str, Any]:
    missing: List[str] = []
    warnings: List[str] = []
    if not record.get("finding_id"):
        missing.append("finding_id")
    if not record.get("run_id"):
        missing.append("run_id")
    if not record.get("model"):
        missing.append("model")
    model_source = record.get("model_source")
    if not isinstance(model_source, dict) or not model_source:
        missing.append("model_source")
    else:
        for field in ("name", "execution_mode", "source_url_or_repo", "weights_or_checkpoint"):
            if not model_source.get(field):
                missing.append("model_source.%s" % field)
        weights = model_source.get("weights_or_checkpoint", {})
        if isinstance(weights, dict):
            if not weights.get("source_url_or_repo"):
                missing.append("model_source.weights_or_checkpoint.source_url_or_repo")
            if not (weights.get("download_command_or_method") or weights.get("load_call")):
                missing.append("model_source.weights_or_checkpoint.download_command_or_method")
    if not record.get("dataset_sources"):
        missing.append("dataset_sources")
    for idx, item in enumerate(record.get("dataset_sources", [])):
        for field in ("name", "source_url_or_repo", "access_route", "unit", "local_or_artifact_path"):
            if not item.get(field):
                missing.append("dataset_sources[%d].%s" % (idx, field))
    if not record.get("retrieval_and_download"):
        missing.append("retrieval_and_download")
    if not record.get("execution_environment"):
        missing.append("execution_environment")
    if not record.get("provenance_artifact_ids"):
        missing.append("provenance_artifact_ids")
    if not record.get("download_ids"):
        missing.append("download_ids")
    else:
        downloads = _load_downloads()
        known_downloads = {
            item.get("download_id") for item in downloads.get("records", []) if item.get("download_id")
        }
        for download_id in record.get("download_ids", []):
            if download_id not in known_downloads:
                missing.append("download_ids.%s" % download_id)
    if record.get("status") in {"partial_explicit", "backfill_needed"} and not record.get("known_gaps"):
        warnings.append("partial/backfill records should explain known_gaps")
    if record.get("known_gaps"):
        warnings.extend(record.get("known_gaps", []))
    return {
        "finding_id": record.get("finding_id"),
        "run_id": record.get("run_id"),
        "model": record.get("model"),
        "status": record.get("status"),
        "passes_contract": not missing,
        "missing": missing,
        "warnings": warnings,
    }


def _filter_download_records(
    records: Iterable[Dict[str, Any]],
    model: str = "",
    finding_id: str = "",
    run_id: str = "",
    artifact_id: str = "",
    query: str = "",
    format: str = "",
) -> List[Dict[str, Any]]:
    tokens = [token for token in _normalize(query).split() if token]
    results = []
    for record in records:
        if model and not _matches_filter(record.get("model", ""), model):
            continue
        if artifact_id and record.get("artifact_id") != artifact_id and record.get("download_id") != artifact_id:
            continue
        if finding_id and finding_id not in record.get("finding_ids", []):
            continue
        if run_id and run_id not in record.get("run_ids", []):
            continue
        if format and not _matches_filter(record.get("format", ""), format):
            continue
        haystack = _normalize(_stringify_record(record))
        if tokens and not all(token in haystack for token in tokens):
            continue
        results.append(record)
    return results


def _summary_download_record(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "download_id": record.get("download_id"),
        "artifact_id": record.get("artifact_id"),
        "model": record.get("model"),
        "relative_path": record.get("relative_path"),
        "format": record.get("format"),
        "bytes": record.get("bytes"),
        "rows": record.get("rows"),
        "sha256": record.get("sha256"),
        "download_url": record.get("download_url"),
        "finding_ids": record.get("finding_ids", []),
        "run_ids": record.get("run_ids", []),
    }


def _submission_validation(submission: Dict[str, Any]) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []

    _require_fields(submission, ("title", "submitter", "finding", "provenance"), "submission", errors)

    submitter = submission.get("submitter", {})
    if not isinstance(submitter, dict):
        errors.append("submission.submitter must be an object")
    else:
        _require_fields(submitter, ("name", "contact"), "submitter", errors)
        if not submitter.get("agent"):
            warnings.append("submitter.agent is recommended so reviewers know which client produced the bundle")

    finding = submission.get("finding", {})
    if not isinstance(finding, dict):
        errors.append("submission.finding must be an object")
        finding = {}
    else:
        _require_fields(
            finding,
            ("finding_id", "model", "domain", "task", "summary", "claim_boundary"),
            "finding",
            errors,
        )
        metric = finding.get("metric", {})
        if not isinstance(metric, dict) or not metric.get("primary"):
            errors.append("finding.metric.primary is required")
        axis = finding.get("axis", {})
        if not isinstance(axis, dict):
            errors.append("finding.axis must be an object")
        else:
            _require_fields(axis, ("name", "definition", "unit"), "finding.axis", errors)
        validity = finding.get("validity", {})
        if not isinstance(validity, dict) or not validity.get("decision"):
            errors.append("finding.validity.decision is required")
        if not (finding.get("evidence") or finding.get("spc") or finding.get("results")):
            warnings.append("finding should include evidence, spc, or results with per-level performance")

    provenance_records = _record_list(submission.get("provenance"))
    if not provenance_records:
        errors.append("submission.provenance must contain at least one provenance record")
    for idx, record in enumerate(provenance_records):
        prefix = "provenance[%d]" % idx
        _require_fields(record, ("finding_id", "run_id", "model"), prefix, errors)
        model_source = record.get("model_source", {})
        if not isinstance(model_source, dict) or not model_source:
            errors.append("%s.model_source is required" % prefix)
        else:
            _require_fields(
                model_source,
                ("name", "execution_mode", "source_url_or_repo", "weights_or_checkpoint"),
                "%s.model_source" % prefix,
                errors,
            )
            weights = model_source.get("weights_or_checkpoint", {})
            if isinstance(weights, dict):
                if not weights.get("source_url_or_repo"):
                    errors.append("%s.model_source.weights_or_checkpoint.source_url_or_repo is required" % prefix)
                if not (weights.get("download_command_or_method") or weights.get("load_call")):
                    errors.append(
                        "%s.model_source.weights_or_checkpoint.download_command_or_method or load_call is required"
                        % prefix
                    )
        dataset_sources = record.get("dataset_sources", [])
        if not isinstance(dataset_sources, list) or not dataset_sources:
            errors.append("%s.dataset_sources must contain at least one dataset source" % prefix)
        else:
            for source_idx, source in enumerate(dataset_sources):
                if not isinstance(source, dict):
                    errors.append("%s.dataset_sources[%d] must be an object" % (prefix, source_idx))
                    continue
                _require_fields(
                    source,
                    ("name", "source_url_or_repo", "access_route", "unit", "rows_or_units"),
                    "%s.dataset_sources[%d]" % (prefix, source_idx),
                    errors,
                )
        if not record.get("retrieval_and_download"):
            errors.append("%s.retrieval_and_download is required" % prefix)
        if not record.get("execution_environment"):
            errors.append("%s.execution_environment is required" % prefix)
        if not record.get("known_gaps"):
            warnings.append("%s.known_gaps should explicitly say none or list remaining gaps" % prefix)

    download_records = _record_list(submission.get("downloads"))
    download_ids = {item.get("download_id") for item in download_records if item.get("download_id")}
    if not download_records:
        errors.append("submission.downloads.records must list externally downloadable result artifacts")
    for idx, record in enumerate(download_records):
        prefix = "downloads[%d]" % idx
        _require_fields(record, ("download_id", "download_url", "sha256"), prefix, errors)
        url = str(record.get("download_url") or "")
        if url and not re.match(r"^https?://", url):
            errors.append("%s.download_url must be http(s)" % prefix)
        checksum = str(record.get("sha256") or "")
        if checksum and not re.match(r"^[a-fA-F0-9]{64}$", checksum):
            errors.append("%s.sha256 must be a 64-character hex SHA-256 checksum" % prefix)
        if "bytes" not in record:
            warnings.append("%s.bytes is recommended for reproducibility checks" % prefix)
        if "rows" not in record:
            warnings.append("%s.rows is recommended for per-target or per-example tables" % prefix)

    for idx, record in enumerate(provenance_records):
        for download_id in record.get("download_ids", []):
            if download_id not in download_ids:
                warnings.append("provenance[%d].download_ids.%s is not present in submission downloads" % (idx, download_id))

    artifact_records = _record_list(submission.get("artifact_manifest"))
    if not artifact_records:
        errors.append("submission.artifact_manifest.records must list referenced artifacts")
    forbidden_inline_fields = {"content", "content_base64", "file_bytes", "table_rows"}
    for idx, record in enumerate(artifact_records):
        prefix = "artifact_manifest[%d]" % idx
        _require_fields(record, ("artifact_id", "role"), prefix, errors)
        if not (
            record.get("download_id")
            or record.get("download_url")
            or record.get("url")
            or record.get("external_url")
            or record.get("sha256")
        ):
            errors.append("%s must reference an external URL, download_id, or checksum" % prefix)
        for field in forbidden_inline_fields:
            if field in record:
                errors.append("%s.%s is not allowed; submit large artifacts by URL and checksum" % (prefix, field))

    audit_card = submission.get("audit_card_markdown", "")
    if audit_card and not isinstance(audit_card, str):
        errors.append("submission.audit_card_markdown must be a string")
    if not audit_card:
        warnings.append("audit_card_markdown is recommended for reviewer triage")

    return {
        "passes_contract": not errors,
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "model": finding.get("model"),
            "finding_id": finding.get("finding_id"),
            "download_count": len(download_records),
            "artifact_count": len(artifact_records),
            "provenance_record_count": len(provenance_records),
        },
    }


def _submission_summary_from_manifest(manifest: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "submission_id": manifest.get("submission_id"),
        "status": manifest.get("status"),
        "created_at": manifest.get("created_at"),
        "updated_at": manifest.get("updated_at"),
        "model": manifest.get("model"),
        "finding_id": manifest.get("finding_id"),
        "title": manifest.get("title"),
        "submitter": manifest.get("submitter"),
        "passes_contract": manifest.get("validation", {}).get("passes_contract"),
        "warning_count": len(manifest.get("validation", {}).get("warnings", [])),
    }


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


def list_spectra_provenance(
    model: str = "",
    finding_id: str = "",
    run_id: str = "",
    status: str = "",
) -> Dict[str, Any]:
    """List compact provenance summaries for stored SPECTRA findings."""
    provenance = _load_provenance()
    records = _filter_provenance_records(
        provenance.get("records", []),
        model=model,
        finding_id=finding_id,
        run_id=run_id,
        status=status,
    )
    return {
        "schema_version": provenance.get("schema_version"),
        "count": len(records),
        "provenance": [_summary_provenance_record(item) for item in records],
    }


def get_spectra_provenance(
    model: str = "",
    finding_id: str = "",
    run_id: str = "",
    status: str = "",
) -> Dict[str, Any]:
    """Return normalized model/data/download provenance records."""
    provenance = _load_provenance()
    records = _filter_provenance_records(
        provenance.get("records", []),
        model=model,
        finding_id=finding_id,
        run_id=run_id,
        status=status,
    )
    return {
        "schema_version": provenance.get("schema_version"),
        "count": len(records),
        "records": records,
    }


def list_spectra_sources(
    model: str = "",
    finding_id: str = "",
    run_id: str = "",
    source_type: str = "",
) -> Dict[str, Any]:
    """Flatten model, weight, dataset, metadata, and retrieval sources."""
    provenance = _load_provenance()
    records = _filter_provenance_records(
        provenance.get("records", []),
        model=model,
        finding_id=finding_id,
        run_id=run_id,
    )
    rows: List[Dict[str, Any]] = []
    for record in records:
        for row in _source_rows(record):
            if source_type and not _matches_filter(row.get("source_type", ""), source_type):
                continue
            rows.append(row)
    return {"count": len(rows), "sources": rows}


def validate_spectra_provenance(
    model: str = "",
    finding_id: str = "",
    run_id: str = "",
) -> Dict[str, Any]:
    """Check stored provenance records against the current provenance contract."""
    provenance = _load_provenance()
    records = _filter_provenance_records(
        provenance.get("records", []),
        model=model,
        finding_id=finding_id,
        run_id=run_id,
    )
    validations = [_validation_status(item) for item in records]
    return {
        "schema_version": provenance.get("schema_version"),
        "count": len(validations),
        "all_pass": all(item["passes_contract"] for item in validations),
        "validations": validations,
    }


def get_spectra_provenance_schema() -> Dict[str, Any]:
    """Return the provenance schema and policy for future stored findings."""
    return _load_provenance_schema()


def list_spectra_downloads(
    model: str = "",
    finding_id: str = "",
    run_id: str = "",
    query: str = "",
    format: str = "",
    top_k: int = 100,
) -> Dict[str, Any]:
    """List public HTTPS downloads for curated result artifacts."""
    downloads = _load_downloads()
    records = _filter_download_records(
        downloads.get("records", []),
        model=model,
        finding_id=finding_id,
        run_id=run_id,
        query=query,
        format=format,
    )
    records.sort(key=lambda item: (-int(item.get("bytes") or 0), item.get("relative_path", "")))
    limit = max(1, min(int(top_k), 500))
    return {
        "schema_version": downloads.get("schema_version"),
        "base_url": downloads.get("base_url"),
        "count": len(records[:limit]),
        "total_matches": len(records),
        "downloads": [_summary_download_record(item) for item in records[:limit]],
    }


def get_spectra_download(artifact_id: str) -> Dict[str, Any]:
    """Return one public download record by artifact/download id."""
    downloads = _load_downloads()
    records = _filter_download_records(downloads.get("records", []), artifact_id=artifact_id)
    if not records:
        raise ValueError("Unknown download artifact_id '%s'" % artifact_id)
    return records[0]


def get_spectra_submission_schema() -> Dict[str, Any]:
    """Return the schema and policy for pending contributed findings."""
    schema = _load_submission_schema()
    return {
        "schema": schema,
        "queue": {
            "enabled": _submission_queue_enabled(),
            "status_values": list(SUBMISSION_STATUS_VALUES),
            "max_submission_json_bytes": MAX_SUBMISSION_JSON_BYTES,
            "persistence": "data/submissions/pending/<submission_id>/",
            "canonical_store_mutation": False,
            "artifact_policy": (
                "Do not upload large raw tables through MCP. Reference artifacts "
                "by stable http(s) URL, SHA-256 checksum, byte size, and row count."
            ),
        },
    }


def validate_spectra_submission(submission: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a proposed contributed SPECTRA finding without writing it."""
    try:
        payload = _coerce_json_object(submission, "submission")
    except ValueError as exc:
        return {
            "passes_contract": False,
            "errors": [str(exc)],
            "warnings": [],
            "summary": {},
        }
    return _submission_validation(payload)


def submit_spectra_finding(
    submission: Dict[str, Any],
    auth_token: str = "",
) -> Dict[str, Any]:
    """Submit a SPECTRA finding bundle to the pending review queue.

    This never mutates the canonical finding/provenance/download manifests.
    The MCP host must set SPECTRA_SUBMISSION_TOKEN and clients must pass that
    shared token as auth_token.
    """
    _require_submission_auth(auth_token)
    payload = _coerce_json_object(submission, "submission")
    validation = _submission_validation(payload)
    if not validation["passes_contract"]:
        return {
            "accepted": False,
            "status": "rejected_by_schema",
            "validation": validation,
        }

    submission_id = _generate_submission_id(payload)
    queue_path = _pending_submission_path(submission_id)
    counter = 2
    while queue_path.exists():
        suffix = "-%d" % counter
        queue_path = _pending_submission_path(submission_id + suffix)
        counter += 1
    submission_id = queue_path.name
    queue_path.mkdir(parents=True, exist_ok=False)

    finding = payload.get("finding", {})
    manifest = {
        "submission_id": submission_id,
        "status": "pending_review",
        "created_at": _utc_timestamp(),
        "updated_at": _utc_timestamp(),
        "title": payload.get("title"),
        "model": finding.get("model"),
        "finding_id": finding.get("finding_id"),
        "submitter": payload.get("submitter", {}),
        "validation": validation,
        "files": {
            "submission": "submission.json",
            "finding": "finding.json",
            "provenance": "provenance.json",
            "downloads": "downloads.json",
            "artifact_manifest": "artifact_manifest.json",
            "audit_card": "audit_card.md" if payload.get("audit_card_markdown") else "",
        },
    }

    _write_json(queue_path / "submission.json", payload)
    _write_json(queue_path / "finding.json", payload.get("finding", {}))
    _write_json(queue_path / "provenance.json", {"records": _record_list(payload.get("provenance"))})
    _write_json(queue_path / "downloads.json", {"records": _record_list(payload.get("downloads"))})
    _write_json(queue_path / "artifact_manifest.json", {"records": _record_list(payload.get("artifact_manifest"))})
    if payload.get("audit_card_markdown"):
        (queue_path / "audit_card.md").write_text(payload["audit_card_markdown"], encoding="utf-8")
    _write_json(queue_path / "manifest.json", manifest)

    return {
        "accepted": True,
        "status": "pending_review",
        "submission_id": submission_id,
        "validation": validation,
        "review_queue": "data/submissions/pending/%s" % submission_id,
        "canonical_store_mutated": False,
    }


def get_spectra_submission_status(
    submission_id: str,
    auth_token: str = "",
) -> Dict[str, Any]:
    """Return pending-review status for one contributed finding submission."""
    _require_submission_auth(auth_token)
    path = _pending_submission_path(submission_id)
    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        raise ValueError("Unknown pending submission_id '%s'" % submission_id)
    manifest = _load_json(manifest_path)
    return {
        "submission": _submission_summary_from_manifest(manifest),
        "manifest": manifest,
    }


def list_spectra_submissions(
    status: str = "",
    model: str = "",
    auth_token: str = "",
    top_k: int = 100,
) -> Dict[str, Any]:
    """List pending-review contributed finding submissions for maintainers."""
    _require_submission_auth(auth_token)
    results = []
    if SUBMISSION_PENDING_ROOT.exists():
        for manifest_path in sorted(SUBMISSION_PENDING_ROOT.glob("*/manifest.json")):
            manifest = _load_json(manifest_path)
            if status and not _matches_filter(manifest.get("status", ""), status):
                continue
            if model and not _matches_filter(manifest.get("model", ""), model):
                continue
            results.append(manifest)
    results.sort(key=lambda item: item.get("created_at", ""), reverse=True)
    limit = max(1, min(int(top_k), 500))
    return {
        "count": len(results[:limit]),
        "total_matches": len(results),
        "submissions": [_submission_summary_from_manifest(item) for item in results[:limit]],
    }


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
    """Create the SPECTRA knowledge MCP server."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise ImportError(
            "The MCP package is required. Install with: pip install -r requirements.txt"
        ) from exc

    instructions = (
        "SPECTRA knowledge server. Exposes the current SPECTRA "
        "single-controller protocol, stored finding records, run summaries, and "
        "saved text/CSV/JSON artifacts. It also exposes normalized provenance "
        "for model code/weights, datasets, metadata, download routes, cache "
        "roots, and known gaps. It can accept authenticated contributed finding "
        "bundles into a pending review queue, but those submissions do not mutate "
        "canonical findings until separately reviewed. It must not run audits, "
        "launch agents, download datasets, or call models."
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

    @mcp.resource("spectra://downloads/index")
    def downloads_index_resource() -> Dict[str, Any]:
        """Index of public download URLs for curated result artifacts."""
        return list_spectra_downloads()

    @mcp.resource("spectra://provenance/index")
    def provenance_index_resource() -> Dict[str, Any]:
        """Index of normalized model/data/download provenance records."""
        return list_spectra_provenance()

    @mcp.resource("spectra://provenance/schema")
    def provenance_schema_resource() -> Dict[str, Any]:
        """Current provenance schema and policy for stored findings."""
        return get_spectra_provenance_schema()

    @mcp.resource("spectra://submissions/schema")
    def submission_schema_resource() -> Dict[str, Any]:
        """Schema and policy for pending contributed finding submissions."""
        return get_spectra_submission_schema()

    mcp.tool(name="list_spectra_models")(list_spectra_models)
    mcp.tool(name="list_spectra_findings")(list_spectra_findings)
    mcp.tool(name="search_spectra_findings")(search_spectra_findings)
    mcp.tool(name="get_spectra_finding")(get_spectra_finding)
    mcp.tool(name="list_spectra_runs")(list_spectra_runs)
    mcp.tool(name="get_spectra_run")(get_spectra_run)
    mcp.tool(name="list_spectra_provenance")(list_spectra_provenance)
    mcp.tool(name="get_spectra_provenance")(get_spectra_provenance)
    mcp.tool(name="list_spectra_sources")(list_spectra_sources)
    mcp.tool(name="validate_spectra_provenance")(validate_spectra_provenance)
    mcp.tool(name="get_spectra_provenance_schema")(get_spectra_provenance_schema)
    mcp.tool(name="list_spectra_downloads")(list_spectra_downloads)
    mcp.tool(name="get_spectra_download")(get_spectra_download)
    mcp.tool(name="get_spectra_submission_schema")(get_spectra_submission_schema)
    mcp.tool(name="validate_spectra_submission")(validate_spectra_submission)
    mcp.tool(name="submit_spectra_finding")(submit_spectra_finding)
    mcp.tool(name="get_spectra_submission_status")(get_spectra_submission_status)
    mcp.tool(name="list_spectra_submissions")(list_spectra_submissions)
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
