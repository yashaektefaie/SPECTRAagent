"""Literature-backed targeted operating-point registry for /spectra.

The similarity registry defines what train-test proximity means. The
similarity-computation registry defines how to compute it at scale. This module
defines named split/evaluation operators that sample specific deployment points
on the broader SPECTRA novelty curve.
"""

import json
import os
import pkgutil
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Union


REGISTRY_VERSION = "0.1.0"
METHOD_ROOT = ("operating_points", "methods")
SCHEMA_PATH = ("operating_points", "schema.json")
_FALLBACK_IDS = [
    "bioactivity_step_forward_split",
    "biological_sequence_homology_cluster_split",
    "chromosome_loco_holdout",
    "contiguous_mutational_region_split",
    "cross_species_taxon_holdout",
    "drug_target_cold_start_split",
    "graph_covariate_concept_shift_split",
    "group_k_fold_holdout",
    "leave_assay_out_bioactivity_split",
    "leave_one_domain_out_benchmark",
    "leave_site_out_external_validation",
    "materials_leave_cluster_composition_split",
    "medical_imaging_scanner_site_holdout",
    "molecular_fingerprint_cluster_split",
    "molecular_property_extreme_split",
    "molecular_scaffold_split",
    "molecular_umap_cluster_split",
    "perturbation_systematic_variation_confounder_holdout",
    "perturbation_unseen_cell_type_condition_split",
    "perturbation_unseen_perturbation_split",
    "protein_family_remote_homology_holdout",
    "random_iid_split_baseline",
    "regulatory_cross_cell_type_assay_holdout",
    "rna_structurally_dissimilar_split",
    "spatial_block_buffered_cv",
    "spatiotemporal_forecast_horizon_split",
    "targeted_intended_use_validation",
    "temporal_forward_split",
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


def _method_dir() -> str:
    return os.path.join(os.path.dirname(__file__), *METHOD_ROOT)


def _method_ids() -> List[str]:
    local_dir = _method_dir()
    if os.path.isdir(local_dir):
        return sorted(
            os.path.splitext(name)[0]
            for name in os.listdir(local_dir)
            if name.endswith(".json")
        )
    return list(_FALLBACK_IDS)


def load_operating_point_registry_schema() -> Dict[str, Any]:
    """Load the targeted operating-point registry schema."""
    return _resource_json(SCHEMA_PATH)


def load_operating_point_method(method_id: str) -> Dict[str, Any]:
    """Load one operating-point method by id."""
    normalized = _normalize_token(method_id)
    if normalized not in _method_ids():
        raise ValueError("Unknown operating-point method: %s" % method_id)
    return _resource_json(METHOD_ROOT + ("%s.json" % normalized,))


def _method_summary(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": item["id"],
        "operating_point_name": item["operating_point_name"],
        "method_family": item["method_family"],
        "scientific_unit": item["scientific_unit"],
        "data_types": item.get("data_types", []),
        "target_novelty_axis": item["target_novelty_axis"],
        "curve_region": item["curve_region"],
        "deployment_question": item["deployment_question"],
        "status": item.get("status"),
        "citation_count": len(item.get("citations", [])),
        "tags": item.get("tags", []),
    }


def list_operating_point_methods(
    method_family: Optional[str] = None,
    scientific_unit: Optional[str] = None,
    data_type: Optional[str] = None,
    curve_region: Optional[str] = None,
    status: Optional[str] = None,
) -> Dict[str, Any]:
    """List targeted operating-point methods with optional filters."""
    normalized_family = _normalize_token(method_family) if method_family else None
    normalized_unit = _normalize_token(scientific_unit) if scientific_unit else None
    normalized_data_type = _normalize_token(data_type) if data_type else None
    normalized_region = _normalize_token(curve_region) if curve_region else None
    normalized_status = _normalize_token(status) if status else None

    methods = []
    for method_id in _method_ids():
        item = load_operating_point_method(method_id)
        item_family = _normalize_token(item.get("method_family", ""))
        item_unit = _normalize_token(item.get("scientific_unit", ""))
        item_data_types = _normalize_token(" ".join(item.get("data_types", [])))
        item_region = _normalize_token(item.get("curve_region", ""))
        item_status = _normalize_token(item.get("status", ""))
        if normalized_family and normalized_family not in item_family:
            continue
        if normalized_unit and normalized_unit not in item_unit:
            continue
        if normalized_data_type and normalized_data_type not in item_data_types:
            continue
        if normalized_region and normalized_region != item_region:
            continue
        if normalized_status and normalized_status != item_status:
            continue
        methods.append(_method_summary(item))

    return {
        "registry_version": REGISTRY_VERSION,
        "count": len(methods),
        "methods": methods,
    }


def search_operating_point_methods(
    query: str = "",
    data_type: str = "",
    novelty_axis: str = "",
    deployment_question: str = "",
    required_inputs: Optional[List[str]] = None,
    top_k: int = 8,
) -> Dict[str, Any]:
    """Rank operating-point methods for a targeted SPECTRA evaluation."""
    query_tokens = _tokens(" ".join([query, data_type, novelty_axis, deployment_question]))
    input_tokens = {_normalize_token(value) for value in (required_inputs or [])}
    normalized_data_type = _normalize_token(data_type) if data_type else ""
    normalized_axis = _normalize_token(novelty_axis) if novelty_axis else ""

    scored = []
    for method_id in _method_ids():
        item = load_operating_point_method(method_id)
        haystack = _tokens(
            {
                "id": item.get("id"),
                "operating_point_name": item.get("operating_point_name"),
                "method_family": item.get("method_family"),
                "scientific_unit": item.get("scientific_unit"),
                "data_types": item.get("data_types"),
                "target_novelty_axis": item.get("target_novelty_axis"),
                "deployment_question": item.get("deployment_question"),
                "curve_region": item.get("curve_region"),
                "definition": item.get("definition"),
                "split_operator": item.get("split_operator"),
                "required_inputs": item.get("required_inputs"),
                "recommended_similarity_axes": item.get("recommended_similarity_axes"),
                "implementation_recipe": item.get("implementation_recipe"),
                "quality_gates": item.get("quality_gates"),
                "tags": item.get("tags"),
            }
        )
        score = len(query_tokens & haystack)
        data_type_text = _normalize_token(" ".join(item.get("data_types", [])))
        axis_text = _normalize_token(item.get("target_novelty_axis", ""))
        if normalized_data_type and (
            normalized_data_type in data_type_text or data_type_text in normalized_data_type
        ):
            score += 4
        if normalized_axis and normalized_axis in axis_text:
            score += 4

        required = {_normalize_token(value) for value in item.get("required_inputs", [])}
        if input_tokens:
            score += 2 * len(input_tokens & required)
            missing_inputs = sorted(required - input_tokens)
        else:
            missing_inputs = sorted(required)
        if score <= 0 and query_tokens:
            continue
        summary = _method_summary(item)
        summary["score"] = score
        summary["missing_required_inputs"] = missing_inputs
        scored.append(summary)

    scored.sort(key=lambda row: (row["score"], row["citation_count"]), reverse=True)
    return {
        "registry_version": REGISTRY_VERSION,
        "query": query,
        "data_type": data_type,
        "novelty_axis": novelty_axis,
        "deployment_question": deployment_question,
        "results": scored[: max(1, int(top_k))],
    }


def suggest_operating_point_methods(
    dataset_description: str,
    deployment_question: str = "",
    data_type: str = "",
    novelty_axis: str = "",
    required_inputs: Optional[List[str]] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Suggest targeted operating-point methods for a SPECTRA evaluation."""
    ranked = search_operating_point_methods(
        query=dataset_description,
        data_type=data_type,
        novelty_axis=novelty_axis,
        deployment_question=deployment_question,
        required_inputs=required_inputs,
        top_k=top_k,
    )
    ranked["agent_instruction"] = (
        "Use these methods when the goal is a specific deployment point rather "
        "than the full spectral curve. After constructing the split, measure "
        "where it falls on the SPECTRA similarity axis."
    )
    return ranked


def validate_operating_point_method(
    method: Union[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Validate one operating-point method against required top-level fields."""
    if isinstance(method, str):
        payload = json.loads(method)
    else:
        payload = method
    if not isinstance(payload, dict):
        raise ValueError("Operating-point method must be a JSON object")
    schema = load_operating_point_registry_schema()
    missing = []
    empty = []
    for field in schema["required_fields"]:
        if field not in payload:
            missing.append(field)
        elif payload[field] in (None, "", [], {}):
            empty.append(field)

    warnings = []
    if payload.get("status") != "human_reviewed":
        warnings.append("Method is not marked human_reviewed.")
    if payload.get("curve_region") not in schema.get("curve_region_values", []):
        warnings.append("Method has non-standard curve_region.")
    citations = payload.get("citations", [])
    if not citations:
        warnings.append("Method has no citations.")
    for citation in citations if isinstance(citations, list) else []:
        if not citation.get("evidence"):
            warnings.append("Citation missing evidence: %s" % citation.get("paper_id"))
    output_contract = payload.get("output_contract", {})
    required_columns = set(output_contract.get("required_columns", []))
    for column in ["sample_id", "split", "role"]:
        if column not in required_columns:
            warnings.append("Output contract missing recommended column: %s" % column)

    return {
        "valid": not missing and not empty,
        "missing_fields": missing,
        "empty_fields": empty,
        "warnings": warnings,
        "schema_version": schema["schema_version"],
    }


def validate_operating_point_registry() -> Dict[str, Any]:
    """Validate every bundled targeted operating-point method."""
    results = []
    valid = True
    for method_id in _method_ids():
        method = load_operating_point_method(method_id)
        result = validate_operating_point_method(method)
        result["id"] = method_id
        results.append(result)
        valid = valid and result["valid"]
    return {
        "valid": valid,
        "count": len(results),
        "results": results,
    }


def render_operating_point_method(method_id: str) -> Dict[str, Any]:
    """Render one operating-point method as compact Markdown for agent context."""
    item = load_operating_point_method(method_id)
    lines = [
        "# %s" % item["operating_point_name"],
        "",
        "- id: `%s`" % item["id"],
        "- family: `%s`" % item["method_family"],
        "- scientific unit: `%s`" % item["scientific_unit"],
        "- novelty axis: `%s`" % item["target_novelty_axis"],
        "- curve region: `%s`" % item["curve_region"],
        "",
        "## Deployment Question",
        "",
        item["deployment_question"],
        "",
        "## Definition",
        "",
        item["definition"],
        "",
        "## Required Inputs",
        "",
    ]
    lines.extend("- `%s`" % value for value in item.get("required_inputs", []))
    lines.extend(["", "## Implementation Recipe", ""])
    lines.extend("- %s" % value for value in item.get("implementation_recipe", []))
    lines.extend(["", "## Quality Gates", ""])
    lines.extend("- %s" % value for value in item.get("quality_gates", []))
    lines.extend(["", "## Failure Modes", ""])
    lines.extend("- %s" % value for value in item.get("failure_modes", []))
    lines.extend(["", "## Citations", ""])
    for citation in item.get("citations", []):
        lines.append("- `%s`: %s" % (citation.get("paper_id"), citation.get("title")))
    return {
        "method_id": item["id"],
        "mime_type": "text/markdown",
        "content": "\n".join(lines).strip() + "\n",
    }
