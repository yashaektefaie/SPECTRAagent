"""Portable dataset catalog for /spectra agents.

Benchmark capsules describe papers. The dataset catalog describes reusable data
resources a fresh SPECTRA call can find, access, and evaluate without knowing
anything about local paths from previous runs.
"""

import json
import os
import pkgutil
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Union


REGISTRY_VERSION = "0.1.0"
DATASET_ROOT = ("dataset_catalog", "entries")
SCHEMA_PATH = ("dataset_catalog", "schema.json")
_FALLBACK_IDS = [
    "admeood_drug_property",
    "bbbc021_cell_painting",
    "boom_benchmark_datasets",
    "causalbench_single_cell_grn",
    "chexphoto_cxr_robustness",
    "coos7_zenodo",
    "crispr_comparison_epcrispr",
    "cross_species_tf_binding_morale",
    "dart_eval_synapse",
    "dc_tap_final_table",
    "drugood_chembl",
    "encode_ccre_screen",
    "eurocropsml_zenodo",
    "flip2_protein_fitness",
    "flip_protein_fitness",
    "geobench_earth_monitoring",
    "gess_qmof",
    "good_graph_ood",
    "landcovernet_radiant",
    "matbench_materials",
    "mimic_eicu_physionet",
    "nabench_assays",
    "open_catalyst_oc20",
    "open_problems_pbmc_dge",
    "perturbench_huggingface",
    "pdebench_sciml",
    "proteingym_aws",
    "rxrx2_cell_painting",
    "scperturb_harmonized",
    "systema_zenodo",
    "tableshift_benchmark",
    "ucr_uea_time_series",
    "umap_virtual_screening_nci60",
    "welqrate_drug_discovery",
    "wild_time_benchmark",
    "wilds_camelyon17",
    "wilds_povertymap",
    "wilds_rxrx1",
    "woods_time_series",
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


def _dataset_dir() -> str:
    return os.path.join(os.path.dirname(__file__), *DATASET_ROOT)


def _dataset_ids() -> List[str]:
    local_dir = _dataset_dir()
    if os.path.isdir(local_dir):
        return sorted(
            os.path.splitext(name)[0]
            for name in os.listdir(local_dir)
            if name.endswith(".json")
        )
    return list(_FALLBACK_IDS)


def load_dataset_catalog_schema() -> Dict[str, Any]:
    """Load the dataset catalog schema."""
    return _resource_json(SCHEMA_PATH)


def load_dataset_entry(dataset_id: str) -> Dict[str, Any]:
    """Load one portable dataset catalog entry."""
    normalized = _normalize_token(dataset_id)
    if normalized not in _dataset_ids():
        raise ValueError("Unknown dataset catalog entry: %s" % dataset_id)
    return _resource_json(DATASET_ROOT + ("%s.json" % normalized,))


def _entry_summary(item: Dict[str, Any]) -> Dict[str, Any]:
    source = item.get("source", {})
    scale = item.get("scale", {})
    return {
        "id": item["id"],
        "title": item["title"],
        "status": item.get("status"),
        "domains": item.get("domains", []),
        "data_types": item.get("data_types", []),
        "scientific_units": item.get("scientific_units", []),
        "access": item.get("access", {}).get("mode"),
        "authentication": item.get("access", {}).get("authentication"),
        "homepage": source.get("homepage"),
        "repository": source.get("repository"),
        "data_url_count": len(source.get("data_urls", [])),
        "download_size": scale.get("download_size"),
        "spectra_default": scale.get("spectra_default"),
        "related_memory_entries": item.get("related_memory_entries", []),
        "tags": item.get("tags", []),
    }


def _list_text(values: Sequence[str]) -> str:
    return _normalize_token(" ".join(values))


def list_dataset_entries(
    domain: Optional[str] = None,
    data_type: Optional[str] = None,
    model_family: Optional[str] = None,
    access: Optional[str] = None,
    status: Optional[str] = None,
) -> Dict[str, Any]:
    """List portable dataset catalog entries with optional filters."""
    normalized_domain = _normalize_token(domain) if domain else None
    normalized_data_type = _normalize_token(data_type) if data_type else None
    normalized_family = _normalize_token(model_family) if model_family else None
    normalized_access = _normalize_token(access) if access else None
    normalized_status = _normalize_token(status) if status else None

    entries = []
    for dataset_id in _dataset_ids():
        item = load_dataset_entry(dataset_id)
        domains = _list_text(item.get("domains", []))
        data_types = _list_text(item.get("data_types", []))
        families = _list_text(item.get("compatible_model_families", []))
        item_access = _normalize_token(item.get("access", {}).get("mode", ""))
        item_status = _normalize_token(item.get("status", ""))
        if normalized_domain and normalized_domain not in domains:
            continue
        if normalized_data_type and normalized_data_type not in data_types:
            continue
        if normalized_family and normalized_family not in families:
            continue
        if normalized_access and normalized_access != item_access:
            continue
        if normalized_status and normalized_status != item_status:
            continue
        entries.append(_entry_summary(item))

    return {
        "registry_version": REGISTRY_VERSION,
        "count": len(entries),
        "datasets": entries,
    }


def search_dataset_entries(
    query: str = "",
    domain: str = "",
    data_type: str = "",
    model_family: str = "",
    required_fields: Optional[List[str]] = None,
    top_k: int = 8,
) -> Dict[str, Any]:
    """Rank portable datasets for a SPECTRA audit context."""
    query_tokens = _tokens(" ".join([query, domain, data_type, model_family]))
    required_tokens = {_normalize_token(value) for value in (required_fields or [])}
    normalized_domain = _normalize_token(domain) if domain else ""
    normalized_data_type = _normalize_token(data_type) if data_type else ""
    normalized_family = _normalize_token(model_family) if model_family else ""

    scored = []
    for dataset_id in _dataset_ids():
        item = load_dataset_entry(dataset_id)
        haystack = _tokens(item)
        score = len(query_tokens & haystack)
        domains = _list_text(item.get("domains", []))
        data_types = _list_text(item.get("data_types", []))
        families = _list_text(item.get("compatible_model_families", []))
        fields = {_normalize_token(value) for value in item.get("expected_fields", [])}

        if normalized_domain and normalized_domain in domains:
            score += 5
        if normalized_data_type and normalized_data_type in data_types:
            score += 4
        if normalized_family and (
            normalized_family in families or families in normalized_family
        ):
            score += 4
        if required_tokens:
            score += 2 * len(required_tokens & fields)

        if score <= 0 and (query_tokens or required_tokens):
            continue
        summary = _entry_summary(item)
        summary["score"] = score
        summary["useful_for"] = item.get("useful_for", [])[:3]
        summary["leakage_risks"] = item.get("leakage_risks", [])[:3]
        scored.append(summary)

    scored.sort(key=lambda row: (row["score"], row.get("status") == "active"), reverse=True)
    return {
        "registry_version": REGISTRY_VERSION,
        "query": query,
        "domain": domain,
        "data_type": data_type,
        "model_family": model_family,
        "results": scored[: max(1, int(top_k))],
        "agent_instruction": (
            "Use this catalog to find portable data resources. Follow the access "
            "instructions and verify licensing, authentication, schema, mapping, "
            "and leakage before constructing SPECTRA splits."
        ),
    }


def suggest_dataset_entries(
    audit_question: str,
    model_description: str = "",
    domain: str = "",
    current_dataset_limitation: str = "",
    top_k: int = 5,
) -> Dict[str, Any]:
    """Suggest portable datasets for a SPECTRA audit or Dataset Scout handoff."""
    return search_dataset_entries(
        query=" ".join([audit_question, model_description, current_dataset_limitation]),
        domain=domain,
        model_family=model_description,
        top_k=top_k,
    )


def validate_dataset_entry(entry: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Validate one dataset catalog entry."""
    if isinstance(entry, str):
        payload = json.loads(entry)
    else:
        payload = entry
    if not isinstance(payload, dict):
        raise ValueError("Dataset catalog entry must be a JSON object")

    schema = load_dataset_catalog_schema()
    missing = []
    empty = []
    optional_empty_fields = {"related_benchmark_capsules", "related_memory_entries"}
    for field in schema["required_fields"]:
        if field not in payload:
            missing.append(field)
        elif field not in optional_empty_fields and payload[field] in (None, "", [], {}):
            empty.append(field)

    warnings = []
    if payload.get("status") not in schema.get("status_values", []):
        warnings.append("Entry has non-standard status.")
    access_mode = payload.get("access", {}).get("mode")
    if access_mode not in schema.get("access_modes", []):
        warnings.append("Entry has non-standard access mode.")
    serialized = json.dumps(payload, sort_keys=True)
    for pattern in schema.get("forbidden_path_patterns", []):
        if pattern in serialized:
            warnings.append("Entry contains forbidden local path pattern: %s" % pattern)
    if not payload.get("recommended_axes"):
        warnings.append("Entry has no recommended SPECTRA axes.")
    if not payload.get("source", {}).get("homepage") and not payload.get("source", {}).get("data_urls"):
        warnings.append("Entry has no portable source URL.")

    return {
        "valid": not missing and not empty and not any(
            warning.startswith("Entry contains forbidden local path pattern") for warning in warnings
        ),
        "missing_fields": missing,
        "empty_fields": empty,
        "warnings": warnings,
        "schema_version": schema["schema_version"],
    }


def validate_dataset_catalog() -> Dict[str, Any]:
    """Validate every bundled dataset catalog entry."""
    results = []
    valid = True
    for dataset_id in _dataset_ids():
        entry = load_dataset_entry(dataset_id)
        result = validate_dataset_entry(entry)
        result["id"] = dataset_id
        results.append(result)
        valid = valid and result["valid"]
    return {
        "valid": valid,
        "count": len(results),
        "results": results,
    }


def render_dataset_entry(dataset_id: str) -> Dict[str, Any]:
    """Render one dataset catalog entry as Markdown for agent context."""
    item = load_dataset_entry(dataset_id)
    source = item.get("source", {})
    lines = [
        "# %s" % item["title"],
        "",
        "- id: `%s`" % item["id"],
        "- status: `%s`" % item.get("status"),
        "- access: `%s`" % item.get("access", {}).get("mode"),
        "- authentication: `%s`" % item.get("access", {}).get("authentication"),
        "- scientific units: %s" % ", ".join("`%s`" % value for value in item.get("scientific_units", [])),
        "- homepage: %s" % (source.get("homepage") or "not specified"),
        "- repository: %s" % (source.get("repository") or "not specified"),
        "",
        "## Access Instructions",
        "",
    ]
    lines.extend("- %s" % value for value in item.get("access", {}).get("instructions", []))
    lines.extend(["", "## Data URLs", ""])
    for data_url in source.get("data_urls", []):
        lines.append("- %s: %s" % (data_url.get("description"), data_url.get("url")))
    lines.extend(["", "## Recommended Axes", ""])
    for axis in item.get("recommended_axes", []):
        lines.append("- `%s`: %s" % (axis.get("name"), axis.get("definition")))
    lines.extend(["", "## Leakage Risks", ""])
    lines.extend("- %s" % value for value in item.get("leakage_risks", []))
    return {
        "dataset_id": item["id"],
        "mime_type": "text/markdown",
        "content": "\n".join(lines).strip() + "\n",
    }
