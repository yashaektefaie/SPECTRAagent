"""Reusable SPECTRA run-memory registry.

The similarity and operating-point registries help an agent decide what to try.
This registry preserves what previous SPECTRA runs already learned: useful
datasets, validated axes, negative results, caveats, and artifact pointers that
future agents should reuse before starting from scratch.
"""

import json
import os
import pkgutil
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Union


REGISTRY_VERSION = "0.1.0"
MEMORY_ROOT = ("spectra_memory", "entries")
SCHEMA_PATH = ("spectra_memory", "schema.json")
_FALLBACK_IDS = [
    "boom_numeric_mini_audit_memory",
    "caduceus_external_perturbational_memory",
    "caduceus_strict_sequence_memory",
    "cross_domain_agent_ablation_memory",
]


def _resource_text(relative_parts: Sequence[str]) -> str:
    relative_path = "/".join(relative_parts)
    local_path = os.path.join(os.path.dirname(__file__), *relative_parts)
    if os.path.exists(local_path):
        with open(local_path, "r", encoding="utf-8") as handle:
            return handle.read()

    data = pkgutil.get_data("spectrae", relative_path)
    if data is None:
        raise FileNotFoundError("Could not load resource: %s" % relative_path)
    return data.decode("utf-8")


def _resource_json(relative_parts: Sequence[str]) -> Dict[str, Any]:
    return json.loads(_resource_text(relative_parts))


def _normalize_token(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _tokens(value: Any) -> Set[str]:
    text = json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else str(value)
    return set(re.findall(r"[a-z0-9_]+", text.lower().replace("-", "_")))


def _memory_dir() -> str:
    return os.path.join(os.path.dirname(__file__), *MEMORY_ROOT)


def _memory_ids() -> List[str]:
    local_dir = _memory_dir()
    if os.path.isdir(local_dir):
        return sorted(
            os.path.splitext(name)[0]
            for name in os.listdir(local_dir)
            if name.endswith(".json")
        )
    return list(_FALLBACK_IDS)


def load_spectra_memory_schema() -> Dict[str, Any]:
    """Load the run-memory schema."""
    return _resource_json(SCHEMA_PATH)


def load_memory_entry(entry_id: str) -> Dict[str, Any]:
    """Load one SPECTRA memory entry by id."""
    normalized = _normalize_token(entry_id)
    if normalized not in _memory_ids():
        raise ValueError("Unknown SPECTRA memory entry: %s" % entry_id)
    return _resource_json(MEMORY_ROOT + ("%s.json" % normalized,))


def _entry_summary(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": item["id"],
        "title": item["title"],
        "status": item.get("status"),
        "domains": item.get("domains", []),
        "data_types": item.get("data_types", []),
        "scientific_units": item.get("scientific_units", []),
        "model_families": item.get("model_families", []),
        "models": item.get("models", []),
        "resource_count": len(item.get("resource_records", [])),
        "axis_count": len(item.get("similarity_axes", [])),
        "finding_count": len(item.get("findings", [])),
        "tags": item.get("tags", []),
    }


def _list_text(values: Sequence[str]) -> str:
    return _normalize_token(" ".join(values))


def list_memory_entries(
    domain: Optional[str] = None,
    data_type: Optional[str] = None,
    model_family: Optional[str] = None,
    status: Optional[str] = None,
) -> Dict[str, Any]:
    """List reusable run-memory entries with optional filters."""
    normalized_domain = _normalize_token(domain) if domain else None
    normalized_data_type = _normalize_token(data_type) if data_type else None
    normalized_family = _normalize_token(model_family) if model_family else None
    normalized_status = _normalize_token(status) if status else None

    entries = []
    for entry_id in _memory_ids():
        item = load_memory_entry(entry_id)
        domains = _list_text(item.get("domains", []))
        data_types = _list_text(item.get("data_types", []))
        families = _list_text(item.get("model_families", []))
        item_status = _normalize_token(item.get("status", ""))
        if normalized_domain and normalized_domain not in domains:
            continue
        if normalized_data_type and normalized_data_type not in data_types:
            continue
        if normalized_family and normalized_family not in families:
            continue
        if normalized_status and normalized_status != item_status:
            continue
        entries.append(_entry_summary(item))

    return {
        "registry_version": REGISTRY_VERSION,
        "count": len(entries),
        "entries": entries,
    }


def search_memory_entries(
    query: str = "",
    domain: str = "",
    data_type: str = "",
    model_family: str = "",
    model_name: str = "",
    tags: Optional[List[str]] = None,
    top_k: int = 8,
) -> Dict[str, Any]:
    """Rank prior SPECTRA memories for a new audit context."""
    query_tokens = _tokens(" ".join([query, domain, data_type, model_family, model_name]))
    tag_tokens = {_normalize_token(tag) for tag in (tags or [])}
    normalized_domain = _normalize_token(domain) if domain else ""
    normalized_data_type = _normalize_token(data_type) if data_type else ""
    normalized_family = _normalize_token(model_family) if model_family else ""
    normalized_model = _normalize_token(model_name) if model_name else ""

    scored = []
    for entry_id in _memory_ids():
        item = load_memory_entry(entry_id)
        haystack = _tokens(item)
        score = len(query_tokens & haystack)
        domains = _list_text(item.get("domains", []))
        data_types = _list_text(item.get("data_types", []))
        families = _list_text(item.get("model_families", []))
        models = _list_text(item.get("models", []))
        item_tags = {_normalize_token(tag) for tag in item.get("tags", [])}

        if normalized_domain and normalized_domain in domains:
            score += 5
        if normalized_data_type and normalized_data_type in data_types:
            score += 4
        if normalized_family and (
            normalized_family in families or families in normalized_family
        ):
            score += 5
        if normalized_model and normalized_model in models:
            score += 6
        if tag_tokens:
            score += 3 * len(tag_tokens & item_tags)

        if score <= 0 and (query_tokens or tag_tokens):
            continue
        summary = _entry_summary(item)
        summary["score"] = score
        summary["reuse_priority"] = item.get("reuse_priority", "medium")
        summary["reuse_guidance"] = item.get("reuse_guidance", [])[:3]
        summary["do_not_repeat"] = item.get("do_not_repeat", [])[:3]
        scored.append(summary)

    scored.sort(key=lambda row: (row["score"], row.get("reuse_priority") == "high"), reverse=True)
    return {
        "registry_version": REGISTRY_VERSION,
        "query": query,
        "domain": domain,
        "data_type": data_type,
        "model_family": model_family,
        "model_name": model_name,
        "results": scored[: max(1, int(top_k))],
        "agent_instruction": (
            "Read the top matching memory entries before proposing new datasets or "
            "similarity axes. Reuse prior validated resources, repeat only the "
            "minimal needed checks, and explicitly avoid listed dead ends."
        ),
    }


def suggest_reusable_memory(
    model_description: str,
    dataset_description: str = "",
    domain: str = "",
    top_k: int = 5,
) -> Dict[str, Any]:
    """Suggest reusable run memories for a new /spectra session."""
    return search_memory_entries(
        query=" ".join([model_description, dataset_description]),
        domain=domain,
        data_type=dataset_description,
        model_family=model_description,
        model_name=model_description,
        top_k=top_k,
    )


def validate_memory_entry(entry: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Validate one memory entry against required top-level fields."""
    if isinstance(entry, str):
        payload = json.loads(entry)
    else:
        payload = entry
    if not isinstance(payload, dict):
        raise ValueError("SPECTRA memory entry must be a JSON object")

    schema = load_spectra_memory_schema()
    missing = []
    empty = []
    for field in schema["required_fields"]:
        if field not in payload:
            missing.append(field)
        elif payload[field] in (None, "", [], {}):
            empty.append(field)

    warnings = []
    if payload.get("status") not in schema.get("status_values", []):
        warnings.append("Entry has non-standard status.")
    if payload.get("reuse_priority") not in schema.get("reuse_priority_values", []):
        warnings.append("Entry has non-standard reuse_priority.")
    if not payload.get("findings"):
        warnings.append("Entry has no findings.")
    for finding in payload.get("findings", []) if isinstance(payload.get("findings"), list) else []:
        if not finding.get("claim"):
            warnings.append("Finding missing claim.")
        if not finding.get("limitations"):
            warnings.append("Finding missing limitations: %s" % finding.get("id", "unknown"))
    for axis in payload.get("similarity_axes", []) if isinstance(payload.get("similarity_axes"), list) else []:
        if not axis.get("quality_gates"):
            warnings.append("Axis missing quality gates: %s" % axis.get("name", "unknown"))
    serialized = json.dumps(payload, sort_keys=True)
    for pattern in schema.get("forbidden_path_patterns", []):
        if pattern in serialized:
            warnings.append("Entry contains forbidden local path pattern: %s" % pattern)

    return {
        "valid": not missing and not empty and not any(
            warning.startswith("Entry contains forbidden local path pattern") for warning in warnings
        ),
        "missing_fields": missing,
        "empty_fields": empty,
        "warnings": warnings,
        "schema_version": schema["schema_version"],
    }


def validate_memory_registry() -> Dict[str, Any]:
    """Validate every bundled SPECTRA memory entry."""
    results = []
    valid = True
    for entry_id in _memory_ids():
        entry = load_memory_entry(entry_id)
        result = validate_memory_entry(entry)
        result["id"] = entry_id
        results.append(result)
        valid = valid and result["valid"]
    return {
        "valid": valid,
        "count": len(results),
        "results": results,
    }


def render_memory_entry(entry_id: str) -> Dict[str, Any]:
    """Render one memory entry as compact Markdown for agent context."""
    item = load_memory_entry(entry_id)
    lines = [
        "# %s" % item["title"],
        "",
        "- id: `%s`" % item["id"],
        "- status: `%s`" % item.get("status"),
        "- domains: %s" % ", ".join("`%s`" % value for value in item.get("domains", [])),
        "- model families: %s" % ", ".join("`%s`" % value for value in item.get("model_families", [])),
        "",
        "## Reuse Guidance",
        "",
    ]
    lines.extend("- %s" % value for value in item.get("reuse_guidance", []))
    lines.extend(["", "## Do Not Repeat", ""])
    lines.extend("- %s" % value for value in item.get("do_not_repeat", []))
    lines.extend(["", "## Findings", ""])
    for finding in item.get("findings", []):
        lines.append("- `%s`: %s" % (finding.get("id", "finding"), finding.get("claim", "")))
    lines.extend(["", "## Useful Resources", ""])
    for resource in item.get("resource_records", []):
        lines.append("- `%s` (%s): %s" % (resource.get("name"), resource.get("kind"), resource.get("source")))
    return {
        "entry_id": item["id"],
        "mime_type": "text/markdown",
        "content": "\n".join(lines).strip() + "\n",
    }
