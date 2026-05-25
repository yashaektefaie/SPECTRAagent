"""Utilities for /spectra benchmark capsules and audit cards.

This module intentionally uses only the Python standard library so agents can
inspect capsules, fetch papers/repos, and validate audit artifacts before the
full scientific stack is installed.
"""

import argparse
import json
import math
import os
import pkgutil
import shutil
import subprocess
import sys
import urllib.request
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union


CAPSULE_ROOT = ("benchmarks", "capsules")
SCHEMA_PATH = ("benchmarks", "audit_card_schema.json")
REPORT_TEMPLATE_PATH = ("report_templates", "spectra_audit_report.md")
DEFAULT_OUTPUT_DIR = os.environ.get(
    "SPECTRA_ASSET_DIR",
    "spectra_assets",
)


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


def _capsule_dir() -> str:
    return os.path.join(os.path.dirname(__file__), *CAPSULE_ROOT)


def _normalize_id(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _capsule_ids() -> List[str]:
    local_dir = _capsule_dir()
    if os.path.isdir(local_dir):
        return sorted(
            os.path.splitext(name)[0]
            for name in os.listdir(local_dir)
            if name.endswith(".json")
        )
    # Installed zip-safe fallback. Keep this in sync with capsule filenames.
    return [
        "boom",
        "dart_eval",
        "gess",
        "nabench",
        "perturbench",
        "systema",
        "umap_virtual_screening",
    ]


def load_benchmark_capsule(paper_id: str) -> Dict[str, Any]:
    """Load one benchmark capsule by id."""
    normalized = _normalize_id(paper_id)
    if normalized not in _capsule_ids():
        raise ValueError("Unknown benchmark capsule: %s" % paper_id)
    return _resource_json(CAPSULE_ROOT + ("%s.json" % normalized,))


def list_benchmark_capsules(
    tier: Optional[str] = None,
    community: Optional[str] = None,
) -> Dict[str, Any]:
    """List available /spectra benchmark capsules."""
    normalized_tier = _normalize_id(tier) if tier else None
    normalized_community = _normalize_id(community) if community else None
    capsules = []
    for paper_id in _capsule_ids():
        capsule = load_benchmark_capsule(paper_id)
        if normalized_tier and _normalize_id(capsule.get("tier", "")) != normalized_tier:
            continue
        if normalized_community and _normalize_id(capsule.get("community", "")) != normalized_community:
            continue
        capsules.append(
            {
                "id": capsule["id"],
                "title": capsule["title"],
                "tier": capsule["tier"],
                "community": capsule["community"],
                "status": capsule.get("status", "unknown"),
                "paper_url": capsule.get("paper", {}).get("url"),
                "repo_url": capsule.get("code", {}).get("repo_url"),
            }
        )
    return {"capsules": capsules, "count": len(capsules)}


def load_audit_card_schema() -> Dict[str, Any]:
    """Load the SPECTRA audit-card schema."""
    return _resource_json(SCHEMA_PATH)


def create_audit_card_template(paper_id: str) -> Dict[str, Any]:
    """Create a blank audit card seeded from a benchmark capsule."""
    capsule = load_benchmark_capsule(paper_id)
    return {
        "schema_version": load_audit_card_schema()["schema_version"],
        "capsule_id": capsule["id"],
        "paper": {
            "title": capsule["title"],
            "url": capsule["paper"]["url"],
            "pdf_url": capsule["paper"].get("pdf_url"),
        },
        "scientific_unit": capsule["scientific_unit"],
        "models": [],
        "original_evaluation": capsule["original_evaluation"],
        "discovered_novelty_axes": capsule["recommended_novelty_axes"],
        "property_graphs": [],
        "spectral_split_protocol": "",
        "performance_overlap_curve": [],
        "auspc": None,
        "core_reproduced_claim": "",
        "additional_spectra_finding": "",
        "citation": "",
        "artifacts": {},
    }


def _coerce_json_payload(value: Union[str, Dict[str, Any], List[Dict[str, Any]]]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("Invalid JSON payload: %s" % exc) from exc
    return value


def _as_rows(value: Union[str, List[Dict[str, Any]], Dict[str, Any]]) -> List[Dict[str, Any]]:
    payload = _coerce_json_payload(value)
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        sequence_keys = [key for key, item in payload.items() if isinstance(item, list)]
        if not sequence_keys:
            rows = [payload]
        else:
            length = max(len(payload[key]) for key in sequence_keys)
            rows = []
            for index in range(length):
                row: Dict[str, Any] = {}
                for key, item in payload.items():
                    if isinstance(item, list):
                        row[key] = item[index] if index < len(item) else None
                    else:
                        row[key] = item
                rows.append(row)
    else:
        raise ValueError("Expected JSON object or list")

    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Every curve row must be an object")
    return rows


def _as_float(value: Any, field_name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Field '%s' must be numeric" % field_name) from exc
    if not math.isfinite(result):
        raise ValueError("Field '%s' must be finite" % field_name)
    return result


def compute_auspc(
    curve_points: Union[str, List[Dict[str, Any]], Dict[str, Any]],
    overlap_key: str = "cross_split_overlap",
    performance_key: str = "performance",
    higher_is_better: bool = True,
) -> Dict[str, Any]:
    """Compute area under the spectral performance curve.

    The x-axis is normalized novelty: 0 means highest observed train-test
    overlap, and 1 means lowest observed overlap. The y-axis is the supplied
    performance value. For lower-is-better metrics, the returned transformed
    performance is negated before integration so larger AUSPC remains better.
    """
    rows = _as_rows(curve_points)
    if len(rows) < 2:
        raise ValueError("At least two curve points are required to compute AUSPC")

    parsed = []
    for row in rows:
        overlap = _as_float(row.get(overlap_key), overlap_key)
        performance = _as_float(row.get(performance_key), performance_key)
        parsed.append((overlap, performance))

    overlaps = [overlap for overlap, _ in parsed]
    min_overlap = min(overlaps)
    max_overlap = max(overlaps)
    if max_overlap == min_overlap:
        raise ValueError("Cannot compute AUSPC when all overlap values are equal")

    points = []
    for overlap, performance in parsed:
        novelty = 1.0 - ((overlap - min_overlap) / (max_overlap - min_overlap))
        score = performance if higher_is_better else -performance
        points.append((novelty, score, overlap, performance))
    points.sort(key=lambda item: item[0])

    area = 0.0
    for left, right in zip(points, points[1:]):
        width = right[0] - left[0]
        area += width * ((left[1] + right[1]) / 2.0)

    return {
        "auspc": area,
        "novelty_axis": "1 - normalized_cross_split_overlap",
        "higher_is_better": higher_is_better,
        "point_count": len(points),
        "normalized_points": [
            {
                "novelty": novelty,
                "score": score,
                "cross_split_overlap": overlap,
                "performance": performance,
            }
            for novelty, score, overlap, performance in points
        ],
    }


def validate_audit_card(
    audit_card: Union[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Validate an audit card against required fields and artifact expectations."""
    payload = _coerce_json_payload(audit_card)
    if not isinstance(payload, dict):
        raise ValueError("Audit card must be a JSON object")

    schema = load_audit_card_schema()
    missing = []
    empty = []
    for field in schema["required_fields"]:
        if field not in payload:
            missing.append(field)
        elif payload[field] in (None, "", [], {}):
            empty.append(field)

    warnings = []
    curve = payload.get("performance_overlap_curve")
    if isinstance(curve, list) and len(curve) >= 2 and payload.get("auspc") in (None, ""):
        try:
            warnings.append(
                "performance_overlap_curve is present but auspc is empty; compute_auspc can fill it."
            )
        except ValueError:
            pass

    artifacts = payload.get("artifacts", {})
    if artifacts and isinstance(artifacts, dict):
        for field in schema["artifact_fields"]:
            if field not in artifacts:
                warnings.append("Missing optional artifact pointer: %s" % field)

    return {
        "valid": not missing and not empty,
        "missing_fields": missing,
        "empty_fields": empty,
        "warnings": warnings,
        "schema_version": schema["schema_version"],
    }


def score_agent_audit(
    audit_card: Union[str, Dict[str, Any]],
    rubric_scores: Optional[Union[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Score an agent-produced audit card.

    Deterministic artifact checks are combined with optional 0-2 rubric scores:

    - 0: missing or scientifically invalid
    - 1: partially correct
    - 2: complete and scientifically useful
    """
    payload = _coerce_json_payload(audit_card)
    if not isinstance(payload, dict):
        raise ValueError("Audit card must be a JSON object")
    validation = validate_audit_card(payload)
    schema = load_audit_card_schema()

    def has_numeric_curve(value: Any) -> bool:
        if isinstance(value, list):
            return len(value) >= 2
        if isinstance(value, dict):
            return bool(value.get("computed")) and len(value.get("points") or []) >= 2
        return False

    def has_curve_plan(value: Any) -> bool:
        if has_numeric_curve(value):
            return True
        if isinstance(value, dict):
            return bool(value.get("explanation") or value.get("points") is None)
        return False

    def has_numeric_auspc(value: Any) -> bool:
        if isinstance(value, (int, float)):
            return math.isfinite(float(value))
        if isinstance(value, dict):
            return bool(value.get("computed")) and value.get("value") is not None
        return False

    def has_auspc_plan(value: Any) -> bool:
        if has_numeric_auspc(value):
            return True
        if isinstance(value, dict):
            return bool(value.get("explanation") or value.get("normalization"))
        return False

    artifact_checks = {
        "has_novelty_axes": bool(payload.get("discovered_novelty_axes")),
        "has_property_graphs": bool(payload.get("property_graphs")),
        "has_numeric_curve": has_numeric_curve(payload.get("performance_overlap_curve")),
        "has_curve_plan_or_limitation": has_curve_plan(payload.get("performance_overlap_curve")),
        "has_numeric_auspc": has_numeric_auspc(payload.get("auspc")),
        "has_auspc_plan_or_limitation": has_auspc_plan(payload.get("auspc")),
        "has_report_claims": bool(payload.get("core_reproduced_claim"))
        and bool(payload.get("additional_spectra_finding")),
    }
    artifact_score = sum(1 for passed in artifact_checks.values() if passed)

    rubric_payload: Dict[str, Any] = {}
    if rubric_scores is not None:
        parsed = _coerce_json_payload(rubric_scores)
        if not isinstance(parsed, dict):
            raise ValueError("rubric_scores must be a JSON object")
        rubric_payload = parsed

    rubric_total = 0.0
    rubric_max = 0.0
    rubric_details = {}
    for key in schema["rubric"]:
        if key in rubric_payload:
            score = max(0.0, min(2.0, _as_float(rubric_payload[key], key)))
            rubric_total += score
            rubric_max += 2.0
            rubric_details[key] = score

    deterministic_max = len(artifact_checks)
    deterministic_total = artifact_score
    completeness_penalty = len(validation["missing_fields"]) + len(validation["empty_fields"])
    deterministic_total = max(0, deterministic_total - completeness_penalty)

    total = deterministic_total + rubric_total
    max_score = deterministic_max + rubric_max
    return {
        "score": total,
        "max_score": max_score,
        "score_fraction": total / max_score if max_score else 0.0,
        "deterministic_score": deterministic_total,
        "deterministic_max": deterministic_max,
        "artifact_checks": artifact_checks,
        "rubric_scores": rubric_details,
        "validation": validation,
    }


def create_agent_benchmark_prompt(
    paper_id: str,
    use_spectra_skill: bool = False,
) -> Dict[str, Any]:
    """Create the with-skill or without-skill benchmark prompt."""
    capsule = load_benchmark_capsule(paper_id)
    base_prompt = (
        "Read the target paper and associated repository. Reproduce one core "
        "generalization finding. Then extend the analysis to evaluate whether "
        "model performance degrades under scientifically meaningful train-test "
        "novelty. Return a JSON audit card with the fields from the requested "
        "schema."
    )
    if use_spectra_skill:
        guidance = (
            "You have access to /spectra. Use the SPECTRA protocol: identify the "
            "scientific unit, propose multiple spectral properties, and treat each "
            "property as a similarity hypothesis. Run an iterative, leakage-aware "
            "search: construct or plan each similarity graph, validate cross-split "
            "overlap, score whether the resulting curve is monotonic/localized/weak/"
            "non-explanatory/not-evaluable, use failed axes to choose the next "
            "similarity definition or computation strategy, compute or plan AUSPC "
            "for supported axes, and write a reusable audit that reports both the "
            "selected axis and failed axes."
        )
        resources = {
            "capsule": capsule,
            "audit_card_template": create_audit_card_template(paper_id),
            "schema": load_audit_card_schema(),
        }
    else:
        guidance = (
            "Do not use the /spectra protocol or any SPECTRA-specific checklist. "
            "Use your normal scientific reproduction workflow."
        )
        resources = {
            "paper": capsule["paper"],
            "code": capsule["code"],
            "title": capsule["title"],
        }
    return {
        "paper_id": paper_id,
        "condition": "spectra" if use_spectra_skill else "vanilla",
        "prompt": base_prompt + "\n\n" + guidance,
        "resources": resources,
    }


def render_spectra_report(audit_card: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Render a Markdown SPECTRA audit report from an audit card."""
    payload = _coerce_json_payload(audit_card)
    if not isinstance(payload, dict):
        raise ValueError("Audit card must be a JSON object")
    template = _resource_text(REPORT_TEMPLATE_PATH)

    def format_value(value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            if not value:
                return ""
            return "\n".join("- %s" % json.dumps(item, sort_keys=True) if isinstance(item, dict) else "- %s" % item for item in value)
        if isinstance(value, dict):
            return "```json\n%s\n```" % json.dumps(value, indent=2, sort_keys=True)
        if value is None:
            return ""
        return str(value)

    values = {
        "paper": format_value(payload.get("paper", "")),
        "scientific_unit": format_value(payload.get("scientific_unit", "")),
        "models": format_value(payload.get("models", "")),
        "original_evaluation": format_value(payload.get("original_evaluation", "")),
        "discovered_novelty_axes": format_value(payload.get("discovered_novelty_axes", "")),
        "property_graphs": format_value(payload.get("property_graphs", "")),
        "spectral_split_protocol": format_value(payload.get("spectral_split_protocol", "")),
        "performance_overlap_curve": format_value(payload.get("performance_overlap_curve", "")),
        "auspc": format_value(payload.get("auspc", "")),
        "core_reproduced_claim": format_value(payload.get("core_reproduced_claim", "")),
        "additional_spectra_finding": format_value(payload.get("additional_spectra_finding", "")),
        "citation": format_value(payload.get("citation", "")),
    }
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{{%s}}" % key, value)
    return {"mime_type": "text/markdown", "content": rendered}


def benchmark_download_plan(
    paper_ids: Optional[Sequence[str]] = None,
    include_repos: bool = True,
    include_data: bool = False,
) -> Dict[str, Any]:
    """Return the safe fetch plan for papers, repos, and optional data links."""
    ids = [_normalize_id(item) for item in paper_ids] if paper_ids else _capsule_ids()
    assets = []
    for paper_id in ids:
        capsule = load_benchmark_capsule(paper_id)
        paper = capsule.get("paper", {})
        if paper.get("pdf_url"):
            assets.append(
                {
                    "capsule_id": paper_id,
                    "kind": "paper_pdf",
                    "url": paper["pdf_url"],
                    "filename": "%s.pdf" % paper_id,
                    "default": True,
                }
            )
        else:
            assets.append(
                {
                    "capsule_id": paper_id,
                    "kind": "paper_page",
                    "url": paper.get("open_access_url") or paper.get("url"),
                    "filename": "%s.url.txt" % paper_id,
                    "default": True,
                }
            )
        repo_url = capsule.get("code", {}).get("repo_url")
        if include_repos and repo_url:
            assets.append(
                {
                    "capsule_id": paper_id,
                    "kind": "repo",
                    "url": repo_url,
                    "directory": paper_id,
                    "default": bool(capsule.get("code", {}).get("fetch_default")),
                }
            )
        if include_data:
            for data_item in capsule.get("data", []):
                assets.append(
                    {
                        "capsule_id": paper_id,
                        "kind": "data_pointer",
                        "url": data_item["url"],
                        "name": data_item["name"],
                        "default": bool(data_item.get("fetch_default")),
                        "reason": data_item.get("reason", ""),
                    }
                )
    return {"assets": assets, "count": len(assets)}


def _download_url(url: str, path: str, overwrite: bool = False) -> Dict[str, Any]:
    if os.path.exists(path) and not overwrite:
        return {"path": path, "status": "exists"}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "spectrae-spectra-benchmark-fetcher/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response, open(path, "wb") as handle:
        shutil.copyfileobj(response, handle)
    return {"path": path, "status": "downloaded", "bytes": os.path.getsize(path)}


def _write_pointer(url: str, path: str, overwrite: bool = False) -> Dict[str, Any]:
    if os.path.exists(path) and not overwrite:
        return {"path": path, "status": "exists"}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(url + "\n")
    return {"path": path, "status": "written"}


def _clone_repo(url: str, path: str, overwrite: bool = False) -> Dict[str, Any]:
    if os.path.exists(path):
        if not overwrite:
            return {"path": path, "status": "exists"}
        shutil.rmtree(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", url, path],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return {"path": path, "status": "cloned"}


def fetch_benchmark_assets(
    paper_ids: Optional[Sequence[str]] = None,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    include_repos: bool = False,
    include_data: bool = False,
    overwrite: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Fetch safe benchmark assets.

    By default this downloads paper PDFs/pages only. Repos are opt-in and are
    shallow-cloned. Dataset assets are recorded as pointers unless explicitly
    handled by a future capsule-specific fetcher.
    """
    plan = benchmark_download_plan(
        paper_ids=paper_ids,
        include_repos=include_repos,
        include_data=include_data,
    )
    if dry_run:
        return {"dry_run": True, **plan}

    results = []
    papers_dir = os.path.join(output_dir, "papers")
    repos_dir = os.path.join(output_dir, "repos")
    data_dir = os.path.join(output_dir, "data_pointers")
    for asset in plan["assets"]:
        kind = asset["kind"]
        if kind == "paper_pdf":
            results.append(_download_url(asset["url"], os.path.join(papers_dir, asset["filename"]), overwrite))
        elif kind == "paper_page":
            results.append(_write_pointer(asset["url"], os.path.join(papers_dir, asset["filename"]), overwrite))
        elif kind == "repo":
            results.append(_clone_repo(asset["url"], os.path.join(repos_dir, asset["directory"]), overwrite))
        elif kind == "data_pointer":
            filename = "%s_%s.url.txt" % (asset["capsule_id"], asset["name"].lower().replace(" ", "_"))
            results.append(_write_pointer(asset["url"], os.path.join(data_dir, filename), overwrite))
        else:
            results.append({"status": "skipped", "asset": asset})
    return {"output_dir": output_dir, "results": results, "count": len(results)}


def summarize_experiment_results(result_paths: Sequence[str]) -> Dict[str, Any]:
    """Summarize stored experiment score/summary JSON files."""
    summaries = []
    for path in result_paths:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        item: Dict[str, Any] = {
            "path": path,
            "experiment": payload.get("experiment"),
            "date": payload.get("date"),
            "capsule_id": payload.get("capsule_id"),
        }
        if "conditions" in payload:
            conditions = payload["conditions"]
            item["type"] = "with_without_score"
            item["conditions"] = {}
            for name, score in conditions.items():
                max_total = score.get("max_total")
                item["conditions"][name] = {
                    "total": score.get("total"),
                    "max_total": max_total,
                    "fraction": score.get("total") / max_total if max_total else None,
                }
            if "vanilla" in conditions and "spectra" in conditions:
                item["spectra_delta"] = conditions["spectra"].get("total", 0) - conditions["vanilla"].get("total", 0)
        elif "ood_curve" in payload:
            ood_curve = payload["ood_curve"]
            first = ood_curve[0] if ood_curve else {}
            last = ood_curve[-1] if ood_curve else {}
            id_rmse = payload.get("id", {}).get("rmse")
            item.update(
                {
                    "type": "numeric_audit",
                    "id_rmse": id_rmse,
                    "ood_rmse": first.get("rmse"),
                    "lowest_overlap_rmse": last.get("rmse"),
                    "ood_to_id_rmse_ratio": first.get("rmse") / id_rmse if first.get("rmse") is not None and id_rmse else None,
                    "lowest_overlap_delta_rmse": last.get("rmse") - first.get("rmse") if first.get("rmse") is not None and last.get("rmse") is not None else None,
                    "auspc": payload.get("auspc", {}).get("value"),
                }
            )
        else:
            item["type"] = "unknown"
        summaries.append(item)
    return {"results": summaries, "count": len(summaries)}


def _parse_ids(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    return [_normalize_id(item) for item in value.split(",") if item.strip()]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="/spectra benchmark capsule utility")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List benchmark capsules")
    list_parser.add_argument("--tier")
    list_parser.add_argument("--community")

    show_parser = subparsers.add_parser("show", help="Print one benchmark capsule as JSON")
    show_parser.add_argument("paper_id")

    template_parser = subparsers.add_parser("template", help="Print an audit-card template")
    template_parser.add_argument("paper_id")

    prompt_parser = subparsers.add_parser("prompt", help="Print a benchmark prompt")
    prompt_parser.add_argument("paper_id")
    prompt_parser.add_argument("--spectra", action="store_true")

    score_parser = subparsers.add_parser("score", help="Score an audit-card JSON file")
    score_parser.add_argument("audit_card_json")
    score_parser.add_argument("--rubric-json")

    summarize_parser = subparsers.add_parser("summarize-results", help="Summarize experiment JSON files")
    summarize_parser.add_argument("result_json", nargs="+")

    auspc_parser = subparsers.add_parser("auspc", help="Compute AUSPC from a JSON file")
    auspc_parser.add_argument("curve_json")
    auspc_parser.add_argument("--overlap-key", default="cross_split_overlap")
    auspc_parser.add_argument("--performance-key", default="performance")
    auspc_parser.add_argument("--lower-is-better", action="store_true")

    fetch_parser = subparsers.add_parser("fetch", help="Fetch paper PDFs/pages and optional repos")
    fetch_parser.add_argument("--paper-ids", help="Comma-separated capsule ids. Defaults to all.")
    fetch_parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    fetch_parser.add_argument("--include-repos", action="store_true")
    fetch_parser.add_argument("--include-data", action="store_true")
    fetch_parser.add_argument("--overwrite", action="store_true")
    fetch_parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "list":
        print(json.dumps(list_benchmark_capsules(args.tier, args.community), indent=2, sort_keys=True))
    elif args.command == "show":
        print(json.dumps(load_benchmark_capsule(args.paper_id), indent=2, sort_keys=True))
    elif args.command == "template":
        print(json.dumps(create_audit_card_template(args.paper_id), indent=2, sort_keys=True))
    elif args.command == "prompt":
        print(
            json.dumps(
                create_agent_benchmark_prompt(args.paper_id, use_spectra_skill=args.spectra),
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "score":
        with open(args.audit_card_json, "r", encoding="utf-8") as handle:
            audit_card = json.load(handle)
        rubric = None
        if args.rubric_json:
            with open(args.rubric_json, "r", encoding="utf-8") as handle:
                rubric = json.load(handle)
        print(json.dumps(score_agent_audit(audit_card, rubric), indent=2, sort_keys=True))
    elif args.command == "summarize-results":
        print(json.dumps(summarize_experiment_results(args.result_json), indent=2, sort_keys=True))
    elif args.command == "auspc":
        with open(args.curve_json, "r", encoding="utf-8") as handle:
            curve = json.load(handle)
        print(
            json.dumps(
                compute_auspc(
                    curve,
                    overlap_key=args.overlap_key,
                    performance_key=args.performance_key,
                    higher_is_better=not args.lower_is_better,
                ),
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "fetch":
        print(
            json.dumps(
                fetch_benchmark_assets(
                    paper_ids=_parse_ids(args.paper_ids),
                    output_dir=args.output_dir,
                    include_repos=args.include_repos,
                    include_data=args.include_data,
                    overwrite=args.overwrite,
                    dry_run=args.dry_run,
                ),
                indent=2,
                sort_keys=True,
            )
        )
    else:
        parser.error("Unknown command")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
