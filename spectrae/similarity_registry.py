"""Literature-backed similarity definition registry for /spectra.

The deterministic SPECTRA audit engine consumes an explicit spectral axis or a
train-eval pairwise similarity file. This registry is the knowledge layer that
helps agents choose a defensible axis before running the engine.
"""

import json
import os
import pkgutil
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Union


REGISTRY_VERSION = "0.1.0"
DEFINITION_ROOT = ("similarity_definitions", "definitions")
SCHEMA_PATH = ("similarity_definitions", "schema.json")
EXAMPLE_ROOT = ("similarity_definitions", "examples")
_FALLBACK_IDS = [
    "atomistic_soap_kernel_similarity",
    "biological_sequence_alignment_homology",
    "biomedical_text_semantic_similarity",
    "cell_morphology_profile_similarity",
    "climate_niche_environmental_similarity",
    "connectome_matrix_correlation_similarity",
    "clinical_temporal_site_domain",
    "clinical_tabular_patient_similarity",
    "domain_distribution_mmd_similarity",
    "drug_target_ligand_target_composite",
    "drug_target_protein_sequence_go_integrated",
    "ehr_concept_relatedness_similarity",
    "environment_confounder_domain_similarity",
    "experimental_batch_protocol_similarity",
    "gdl_shift_factor_distance",
    "gene_expression_signature_connectivity_similarity",
    "geospatial_haversine_region_distance",
    "geospatial_location_embedding_similarity",
    "genotype_ibd_ibs_similarity",
    "graph_neighborhood_jaccard_similarity",
    "graph_wasserstein_wl_kernel_similarity",
    "hyperspectral_spectral_angle_similarity",
    "image_perceptual_embedding_similarity",
    "imaging_histogram_radiomics_similarity",
    "interactome_network_proximity_similarity",
    "longitudinal_patient_trajectory_similarity",
    "mass_spectrometry_spectral_cosine_similarity",
    "materials_composition_formula_similarity",
    "materials_structure_ofm_embedding_similarity",
    "medical_imaging_scanner_protocol_domain",
    "microbiome_beta_diversity_similarity",
    "molecules_bemis_murcko_scaffold",
    "molecules_butina_fingerprint_cluster",
    "molecules_morgan_tanimoto",
    "molecules_umap_cluster",
    "multi_omics_similarity_network_fusion",
    "nucleotide_contiguous_mutation_region",
    "ontology_annotation_jaccard_similarity",
    "phylogenetic_patristic_distance_similarity",
    "point_cloud_point_set_distance_similarity",
    "protein_binding_pocket_similarity",
    "protein_mutational_regime_hamming",
    "protein_positional_mutation_coverage",
    "protein_structure_tm_score_similarity",
    "regulatory_dna_chromosome_holdout",
    "regulatory_dna_embedding_cosine_variant_effect",
    "regulatory_dna_motif_bag_similarity",
    "rna_structural_component_similarity",
    "single_cell_covariate_transfer",
    "single_cell_logfc_cosine",
    "spatial_prediction_horizon_similarity",
    "survival_metric_learned_patient_similarity",
    "time_series_dtw_shape_similarity",
    "topological_persistence_diagram_similarity",
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


def _definition_dir() -> str:
    return os.path.join(os.path.dirname(__file__), *DEFINITION_ROOT)


def _definition_ids() -> List[str]:
    local_dir = _definition_dir()
    if os.path.isdir(local_dir):
        return sorted(
            os.path.splitext(name)[0]
            for name in os.listdir(local_dir)
            if name.endswith(".json")
        )
    return list(_FALLBACK_IDS)


def load_similarity_registry_schema() -> Dict[str, Any]:
    """Load the registry schema used for seed definition validation."""
    return _resource_json(SCHEMA_PATH)


def load_similarity_definition(definition_id: str) -> Dict[str, Any]:
    """Load one similarity definition by id."""
    normalized = _normalize_token(definition_id)
    if normalized not in _definition_ids():
        raise ValueError("Unknown similarity definition: %s" % definition_id)
    return _resource_json(DEFINITION_ROOT + ("%s.json" % normalized,))


def list_similarity_definitions(
    data_type: Optional[str] = None,
    scientific_unit: Optional[str] = None,
    task: Optional[str] = None,
    status: Optional[str] = None,
) -> Dict[str, Any]:
    """List registry entries with optional filters."""
    normalized_data_type = _normalize_token(data_type) if data_type else None
    normalized_unit = _normalize_token(scientific_unit) if scientific_unit else None
    normalized_task = _normalize_token(task) if task else None
    normalized_status = _normalize_token(status) if status else None

    definitions = []
    for definition_id in _definition_ids():
        item = load_similarity_definition(definition_id)
        item_data_type = _normalize_token(item.get("data_type", ""))
        item_unit = _normalize_token(item.get("scientific_unit", ""))
        item_status = _normalize_token(item.get("status", ""))
        if normalized_data_type and normalized_data_type not in item_data_type:
            continue
        if normalized_unit and normalized_unit not in item_unit:
            continue
        if normalized_status and normalized_status != item_status:
            continue
        if normalized_task:
            task_text = " ".join(item.get("task_family", []) + item.get("tags", []))
            if normalized_task not in _normalize_token(task_text):
                continue
        definitions.append(_definition_summary(item))

    return {
        "registry_version": REGISTRY_VERSION,
        "count": len(definitions),
        "definitions": definitions,
    }


def _definition_summary(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": item["id"],
        "data_type": item["data_type"],
        "scientific_unit": item["scientific_unit"],
        "similarity_name": item["similarity_name"],
        "similarity_kind": item["similarity_kind"],
        "spectra_mode": item.get("outputs", {}).get("mode"),
        "axis_name": item.get("spectra_axis", {}).get("name"),
        "status": item.get("status"),
        "citation_count": len(item.get("citations", [])),
        "tags": item.get("tags", []),
    }


def search_similarity_definitions(
    query: str = "",
    data_type: str = "",
    task_description: str = "",
    required_inputs: Optional[List[str]] = None,
    top_k: int = 8,
) -> Dict[str, Any]:
    """Rank similarity definitions for an agent's dataset/task context."""
    query_tokens = _tokens(" ".join([query, data_type, task_description]))
    input_tokens = {_normalize_token(value) for value in (required_inputs or [])}
    normalized_data_type = _normalize_token(data_type) if data_type else ""
    scored = []
    for definition_id in _definition_ids():
        item = load_similarity_definition(definition_id)
        item_data_type = _normalize_token(item.get("data_type", ""))
        data_type_matches = (
            not normalized_data_type
            or normalized_data_type in item_data_type
            or item_data_type in normalized_data_type
        )
        haystack = _tokens(
            {
                "id": item.get("id"),
                "data_type": item.get("data_type"),
                "scientific_unit": item.get("scientific_unit"),
                "task_family": item.get("task_family"),
                "similarity_name": item.get("similarity_name"),
                "definition": item.get("definition"),
                "tags": item.get("tags"),
                "required_inputs": item.get("required_inputs"),
            }
        )
        score = len(query_tokens & haystack)
        if normalized_data_type and data_type_matches:
            score += 3
        elif normalized_data_type and score == 0:
            continue
        elif normalized_data_type:
            score -= 3
            if score < 7:
                continue
        required = {_normalize_token(value) for value in item.get("required_inputs", [])}
        if input_tokens:
            score += 2 * len(input_tokens & required)
            missing_inputs = sorted(required - input_tokens)
        else:
            missing_inputs = sorted(required)
        if score <= 0 and query_tokens:
            continue
        summary = _definition_summary(item)
        summary["score"] = score
        summary["missing_required_inputs"] = missing_inputs
        scored.append(summary)

    scored.sort(key=lambda row: (row["score"], row["citation_count"]), reverse=True)
    return {
        "registry_version": REGISTRY_VERSION,
        "query": query,
        "data_type": data_type,
        "task_description": task_description,
        "results": scored[: max(1, int(top_k))],
    }


def suggest_similarity_definitions(
    dataset_description: str,
    task_description: str = "",
    data_type: str = "",
    required_inputs: Optional[List[str]] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Suggest candidate similarity definitions for a SPECTRA audit."""
    ranked = search_similarity_definitions(
        query=dataset_description,
        data_type=data_type,
        task_description=task_description,
        required_inputs=required_inputs,
        top_k=top_k,
    )
    ranked["agent_instruction"] = (
        "Pick one or more candidate definitions, verify that the required inputs exist, "
        "generate pairwise_similarity.csv or a spectral axis, then run spectra audit."
    )
    return ranked


def get_similarity_example_script(definition_id: str) -> Dict[str, Any]:
    """Return the executable example script associated with a definition."""
    definition = load_similarity_definition(definition_id)
    script_path = definition.get("python_example")
    if not script_path:
        raise ValueError("Definition has no python_example: %s" % definition_id)
    script_name = os.path.basename(script_path)
    content = _resource_text(EXAMPLE_ROOT + (script_name,))
    return {
        "definition_id": definition["id"],
        "script_name": script_name,
        "mime_type": "text/x-python",
        "content": content,
        "usage": definition.get("example_usage", ""),
    }


def validate_similarity_definition(
    definition: Union[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Validate a registry definition against required top-level fields."""
    if isinstance(definition, str):
        payload = json.loads(definition)
    else:
        payload = definition
    if not isinstance(payload, dict):
        raise ValueError("Similarity definition must be a JSON object")
    schema = load_similarity_registry_schema()
    missing = []
    empty = []
    for field in schema["required_fields"]:
        if field not in payload:
            missing.append(field)
        elif payload[field] in (None, "", [], {}):
            empty.append(field)

    warnings = []
    if payload.get("status") != "human_reviewed":
        warnings.append("Definition is not marked human_reviewed.")
    citations = payload.get("citations", [])
    if not citations:
        warnings.append("Definition has no citations.")
    for citation in citations if isinstance(citations, list) else []:
        if not citation.get("evidence"):
            warnings.append("Citation missing evidence spans: %s" % citation.get("paper_id"))

    return {
        "valid": not missing and not empty,
        "missing_fields": missing,
        "empty_fields": empty,
        "warnings": warnings,
        "schema_version": schema["schema_version"],
    }


def validate_similarity_registry() -> Dict[str, Any]:
    """Validate every bundled definition."""
    results = []
    valid = True
    for definition_id in _definition_ids():
        definition = load_similarity_definition(definition_id)
        result = validate_similarity_definition(definition)
        result["id"] = definition_id
        results.append(result)
        valid = valid and result["valid"]
    return {
        "valid": valid,
        "count": len(results),
        "results": results,
    }


def render_similarity_definition(definition_id: str) -> Dict[str, Any]:
    """Render a registry definition as compact Markdown for agent context."""
    item = load_similarity_definition(definition_id)
    lines = [
        "# %s" % item["similarity_name"],
        "",
        "- id: `%s`" % item["id"],
        "- data_type: `%s`" % item["data_type"],
        "- scientific_unit: `%s`" % item["scientific_unit"],
        "- SPECTRA mode: `%s`" % item.get("outputs", {}).get("mode"),
        "- axis: `%s`" % item.get("spectra_axis", {}).get("name"),
        "",
        "## Definition",
        "",
        item["definition"],
        "",
        "## Required Inputs",
        "",
    ]
    lines.extend("- `%s`" % value for value in item.get("required_inputs", []))
    lines.extend(["", "## Limitations", ""])
    lines.extend("- %s" % value for value in item.get("limitations", []))
    lines.extend(["", "## Citations", ""])
    for citation in item.get("citations", []):
        lines.append("- `%s`: %s" % (citation.get("paper_id"), citation.get("title")))
    return {
        "definition_id": item["id"],
        "mime_type": "text/markdown",
        "content": "\n".join(lines).strip() + "\n",
    }
