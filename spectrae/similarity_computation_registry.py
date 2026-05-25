"""Literature-backed strategies for computing SPECTRA pairwise similarity.

The similarity definition registry helps an agent decide what similarity means.
This module helps the agent decide how to compute enough train-eval similarity
edges to run SPECTRA without materializing every possible pair when that is
unnecessary or infeasible.
"""

import json
import os
import pkgutil
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Union


REGISTRY_VERSION = "0.1.0"
STRATEGY_ROOT = ("similarity_computation", "strategies")
SCHEMA_PATH = ("similarity_computation", "schema.json")
EXAMPLE_ROOT = ("similarity_computation", "examples")
_FALLBACK_IDS = [
    "angular_random_projection_lsh",
    "binary_hamming_multi_index_hashing",
    "exact_chunked_all_pairs",
    "dense_embedding_hnsw_topk",
    "disk_backed_vector_ann_search",
    "faiss_ivf_pq_compressed_topk",
    "filtered_ann_attribute_constrained_search",
    "genomic_sketch_fracminhash_search",
    "graph_kernel_random_feature_sketch",
    "gpu_accelerated_similarity_join",
    "hybrid_multimodal_similarity_index",
    "kernel_approximation_feature_map",
    "learned_candidate_filter_similarity_join",
    "lp_stable_lsh_distance",
    "maximum_inner_product_search_transform",
    "minhash_lsh_jaccard_candidates",
    "mass_spectrometry_sparse_cosine_acceleration",
    "multi_vector_late_interaction_search",
    "nn_descent_neighbor_graph",
    "pivot_permutation_metric_index",
    "privacy_preserving_similarity_search",
    "sparse_inverted_index_topk",
    "sequence_seed_prefilter_alignment_rerank",
    "semantic_ontology_similarity_join",
    "sinkhorn_wasserstein_histogram_search",
    "molecular_fingerprint_tanimoto_index",
    "streaming_similarity_join_sketch",
    "string_edit_similarity_join",
    "structured_edit_distance_filter_verify",
    "time_series_matrix_profile_index",
    "dtw_lower_bound_pruning",
    "blocking_candidate_generation",
    "metric_tree_radius_knn",
    "threshold_similarity_join_filter_verify",
    "trajectory_metric_index_filter_refine",
    "two_stage_prefilter_rerank",
    "distributed_partitioned_ann_search",
    "weighted_minhash_generalized_jaccard",
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


def _strategy_dir() -> str:
    return os.path.join(os.path.dirname(__file__), *STRATEGY_ROOT)


def _strategy_ids() -> List[str]:
    local_dir = _strategy_dir()
    if os.path.isdir(local_dir):
        return sorted(
            os.path.splitext(name)[0]
            for name in os.listdir(local_dir)
            if name.endswith(".json")
        )
    return list(_FALLBACK_IDS)


def load_similarity_computation_registry_schema() -> Dict[str, Any]:
    """Load the similarity computation strategy schema."""
    return _resource_json(SCHEMA_PATH)


def load_similarity_computation_strategy(strategy_id: str) -> Dict[str, Any]:
    """Load one computation strategy by id."""
    normalized = _normalize_token(strategy_id)
    if normalized not in _strategy_ids():
        raise ValueError("Unknown similarity computation strategy: %s" % strategy_id)
    return _resource_json(STRATEGY_ROOT + ("%s.json" % normalized,))


def _strategy_summary(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": item["id"],
        "computation_name": item["computation_name"],
        "strategy_family": item["strategy_family"],
        "exactness": item["exactness"],
        "compatible_data_shapes": item.get("compatible_data_shapes", []),
        "scale_summary": item.get("scale_profile", {}).get("best_for", ""),
        "output_mode": item.get("output_contract", {}).get("mode"),
        "status": item.get("status"),
        "citation_count": len(item.get("citations", [])),
        "tags": item.get("tags", []),
    }


def list_similarity_computation_strategies(
    strategy_family: Optional[str] = None,
    exactness: Optional[str] = None,
    data_shape: Optional[str] = None,
    status: Optional[str] = None,
) -> Dict[str, Any]:
    """List computation strategy entries with optional filters."""
    normalized_family = _normalize_token(strategy_family) if strategy_family else None
    normalized_exactness = _normalize_token(exactness) if exactness else None
    normalized_shape = _normalize_token(data_shape) if data_shape else None
    normalized_status = _normalize_token(status) if status else None

    strategies = []
    for strategy_id in _strategy_ids():
        item = load_similarity_computation_strategy(strategy_id)
        item_family = _normalize_token(item.get("strategy_family", ""))
        item_exactness = _normalize_token(item.get("exactness", ""))
        item_status = _normalize_token(item.get("status", ""))
        shape_text = _normalize_token(" ".join(item.get("compatible_data_shapes", [])))
        if normalized_family and normalized_family not in item_family:
            continue
        if normalized_exactness and normalized_exactness != item_exactness:
            continue
        if normalized_shape and normalized_shape not in shape_text:
            continue
        if normalized_status and normalized_status != item_status:
            continue
        strategies.append(_strategy_summary(item))

    return {
        "registry_version": REGISTRY_VERSION,
        "count": len(strategies),
        "strategies": strategies,
    }


def _size_tokens(data_size: str) -> Set[str]:
    text = data_size.lower()
    tokens = _tokens(text)
    if any(word in text for word in ["million", "large", "massive", "1000000", "1e6", "billion"]):
        tokens.update({"large_scale", "approximate", "indexed", "top_k"})
    if any(word in text for word in ["small", "toy", "1000", "10k", "10000"]):
        tokens.update({"exact", "chunked"})
    return tokens


def search_similarity_computation_strategies(
    query: str = "",
    similarity_kind: str = "",
    data_shape: str = "",
    data_size: str = "",
    required_inputs: Optional[List[str]] = None,
    top_k: int = 8,
) -> Dict[str, Any]:
    """Rank computation strategies for an agent's similarity computation context."""
    query_tokens = _tokens(" ".join([query, similarity_kind, data_shape])) | _size_tokens(data_size)
    input_tokens = {_normalize_token(value) for value in (required_inputs or [])}
    normalized_shape = _normalize_token(data_shape) if data_shape else ""
    normalized_kind = _normalize_token(similarity_kind) if similarity_kind else ""

    scored = []
    for strategy_id in _strategy_ids():
        item = load_similarity_computation_strategy(strategy_id)
        haystack = _tokens(
            {
                "id": item.get("id"),
                "computation_name": item.get("computation_name"),
                "strategy_family": item.get("strategy_family"),
                "exactness": item.get("exactness"),
                "applies_to_similarity_kinds": item.get("applies_to_similarity_kinds"),
                "compatible_data_shapes": item.get("compatible_data_shapes"),
                "definition": item.get("definition"),
                "scale_profile": item.get("scale_profile"),
                "required_inputs": item.get("required_inputs"),
                "tunable_parameters": item.get("tunable_parameters"),
                "quality_gates": item.get("quality_gates"),
                "tags": item.get("tags"),
            }
        )
        score = len(query_tokens & haystack)
        shape_text = _normalize_token(" ".join(item.get("compatible_data_shapes", [])))
        kind_text = _normalize_token(" ".join(item.get("applies_to_similarity_kinds", [])))
        if normalized_shape and normalized_shape in shape_text:
            score += 4
        if normalized_kind and normalized_kind in kind_text:
            score += 4
        if "large_scale" in query_tokens and item.get("exactness") != "exact":
            score += 2
        if "exact" in query_tokens and item.get("exactness") == "exact":
            score += 2

        required = {_normalize_token(value) for value in item.get("required_inputs", [])}
        if input_tokens:
            score += 2 * len(input_tokens & required)
            missing_inputs = sorted(required - input_tokens)
        else:
            missing_inputs = sorted(required)
        if score <= 0 and query_tokens:
            continue
        summary = _strategy_summary(item)
        summary["score"] = score
        summary["missing_required_inputs"] = missing_inputs
        scored.append(summary)

    scored.sort(key=lambda row: (row["score"], row["citation_count"]), reverse=True)
    return {
        "registry_version": REGISTRY_VERSION,
        "query": query,
        "similarity_kind": similarity_kind,
        "data_shape": data_shape,
        "data_size": data_size,
        "results": scored[: max(1, int(top_k))],
    }


def suggest_similarity_computation_strategies(
    dataset_description: str,
    similarity_definition: str = "",
    data_shape: str = "",
    data_size: str = "",
    required_inputs: Optional[List[str]] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Suggest computation strategies after an agent defines a similarity axis."""
    ranked = search_similarity_computation_strategies(
        query=dataset_description,
        similarity_kind=similarity_definition,
        data_shape=data_shape,
        data_size=data_size,
        required_inputs=required_inputs,
        top_k=top_k,
    )
    ranked["agent_instruction"] = (
        "Choose the cheapest strategy that preserves the audit question. "
        "For approximate or candidate-generation methods, report recall or compare "
        "against an exact subset before running spectra audit."
    )
    return ranked


def get_similarity_computation_example_script(strategy_id: str) -> Dict[str, Any]:
    """Return the executable example script associated with a strategy."""
    strategy = load_similarity_computation_strategy(strategy_id)
    script_path = strategy.get("python_example")
    if not script_path:
        raise ValueError("Strategy has no python_example: %s" % strategy_id)
    script_name = os.path.basename(script_path)
    content = _resource_text(EXAMPLE_ROOT + (script_name,))
    return {
        "strategy_id": strategy["id"],
        "script_name": script_name,
        "mime_type": "text/x-python",
        "content": content,
        "usage": strategy.get("example_usage", ""),
    }


def validate_similarity_computation_strategy(
    strategy: Union[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Validate a computation strategy against required top-level fields."""
    if isinstance(strategy, str):
        payload = json.loads(strategy)
    else:
        payload = strategy
    if not isinstance(payload, dict):
        raise ValueError("Similarity computation strategy must be a JSON object")
    schema = load_similarity_computation_registry_schema()
    missing = []
    empty = []
    for field in schema["required_fields"]:
        if field not in payload:
            missing.append(field)
        elif payload[field] in (None, "", [], {}):
            empty.append(field)

    warnings = []
    if payload.get("status") != "human_reviewed":
        warnings.append("Strategy is not marked human_reviewed.")
    citations = payload.get("citations", [])
    if not citations:
        warnings.append("Strategy has no citations.")
    for citation in citations if isinstance(citations, list) else []:
        if not citation.get("evidence"):
            warnings.append("Citation missing evidence spans: %s" % citation.get("paper_id"))
    output_contract = payload.get("output_contract", {})
    required_columns = set(output_contract.get("required_columns", []))
    for column in ["sample_id", "train_id", "similarity"]:
        if column not in required_columns:
            warnings.append("Output contract missing recommended column: %s" % column)

    return {
        "valid": not missing and not empty,
        "missing_fields": missing,
        "empty_fields": empty,
        "warnings": warnings,
        "schema_version": schema["schema_version"],
    }


def validate_similarity_computation_registry() -> Dict[str, Any]:
    """Validate every bundled computation strategy."""
    results = []
    valid = True
    for strategy_id in _strategy_ids():
        strategy = load_similarity_computation_strategy(strategy_id)
        result = validate_similarity_computation_strategy(strategy)
        result["id"] = strategy_id
        results.append(result)
        valid = valid and result["valid"]
    return {
        "valid": valid,
        "count": len(results),
        "results": results,
    }


def render_similarity_computation_strategy(strategy_id: str) -> Dict[str, Any]:
    """Render one computation strategy as compact Markdown for agent context."""
    item = load_similarity_computation_strategy(strategy_id)
    lines = [
        "# %s" % item["computation_name"],
        "",
        "- id: `%s`" % item["id"],
        "- family: `%s`" % item["strategy_family"],
        "- exactness: `%s`" % item["exactness"],
        "- compatible data: `%s`" % ", ".join(item.get("compatible_data_shapes", [])),
        "- output mode: `%s`" % item.get("output_contract", {}).get("mode"),
        "",
        "## Strategy",
        "",
        item["definition"],
        "",
        "## Required Inputs",
        "",
    ]
    lines.extend("- `%s`" % value for value in item.get("required_inputs", []))
    lines.extend(["", "## Tunable Parameters", ""])
    for parameter in item.get("tunable_parameters", []):
        lines.append("- `%s`: %s" % (parameter.get("name"), parameter.get("purpose", "")))
    lines.extend(["", "## Quality Gates", ""])
    lines.extend("- %s" % value for value in item.get("quality_gates", []))
    lines.extend(["", "## Failure Modes", ""])
    lines.extend("- %s" % value for value in item.get("failure_modes", []))
    lines.extend(["", "## Citations", ""])
    for citation in item.get("citations", []):
        lines.append("- `%s`: %s" % (citation.get("paper_id"), citation.get("title")))
    return {
        "strategy_id": item["id"],
        "mime_type": "text/markdown",
        "content": "\n".join(lines).strip() + "\n",
    }
