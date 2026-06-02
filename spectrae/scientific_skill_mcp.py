"""MCP server for SPECTRA scientific procedure skills.

The pure helper functions in this module are intentionally importable without
an MCP dependency. Install the optional MCP extra before running the server:

    pip install -e ".[mcp]"
    python -m spectrae.scientific_skill_mcp
"""

import argparse
import importlib.util
import json
import math
import os
import pkgutil
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


PROCEDURE_NAME = "generalizability_analysis"
PROCEDURE_VERSION = "0.5.1"
SERVER_NAME = "spectrae-scientific-skills"
_PROCEDURE_ROOT = "procedures"
_EXAMPLE_NAMES = (
    "small_molecule",
    "protein_sequence",
    "single_cell_perturbation",
)


def _read_text_resource(relative_parts: Sequence[str]) -> str:
    relative_path = "/".join(relative_parts)
    local_path = os.path.join(os.path.dirname(__file__), *relative_parts)
    if os.path.exists(local_path):
        with open(local_path, "r", encoding="utf-8") as handle:
            return handle.read()

    data = pkgutil.get_data("spectrae", relative_path)
    if data is None:
        raise FileNotFoundError("Could not load package resource: %s" % relative_path)
    return data.decode("utf-8")


def _procedure_text() -> str:
    return _read_text_resource((_PROCEDURE_ROOT, "generalizability_analysis.md"))


def _example_text(example_name: str) -> str:
    normalized = _normalize_token(example_name)
    if normalized not in _EXAMPLE_NAMES:
        raise ValueError(
            f"Unknown example '{example_name}'. Available examples: {', '.join(_EXAMPLE_NAMES)}"
        )
    return _read_text_resource((_PROCEDURE_ROOT, "examples", f"{normalized}.md"))


def _normalize_token(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _coerce_json_payload(value: Union[str, List[Dict[str, Any]], Dict[str, Any]]) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Expected JSON payload, got invalid JSON: {exc}") from exc
        if not isinstance(parsed, (list, dict)):
            raise ValueError("Expected JSON payload to decode into a list or object")
        return parsed
    return value


def _normalize_rows(payload: Union[str, List[Dict[str, Any]], Dict[str, Any]]) -> List[Dict[str, Any]]:
    parsed = _coerce_json_payload(payload)
    if isinstance(parsed, list):
        rows = parsed
    else:
        sequence_keys = [
            key for key, value in parsed.items() if isinstance(value, list)
        ]
        if not sequence_keys:
            rows = [parsed]
        else:
            length = max(len(parsed[key]) for key in sequence_keys)
            rows = []
            for index in range(length):
                row: Dict[str, Any] = {}
                for key, value in parsed.items():
                    if isinstance(value, list):
                        row[key] = value[index] if index < len(value) else None
                    else:
                        row[key] = value
                rows.append(row)

    normalized_rows = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Every row must be a JSON object")
        normalized_rows.append(row)
    return normalized_rows


def _first_present(row: Dict[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def _payload_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, sort_keys=True, indent=2)
    except (TypeError, ValueError):
        return str(value)


def _as_float(value: Any, field_name: str) -> float:
    if value is None:
        raise ValueError(f"Missing required field '{field_name}'")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Field '{field_name}' must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"Field '{field_name}' must be finite")
    return result


def _safe_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _normalize_capability_tokens(
    client_capabilities: Optional[Union[str, Sequence[str], Dict[str, Any]]],
) -> List[str]:
    if client_capabilities is None:
        return []
    if isinstance(client_capabilities, dict):
        raw_tokens: List[str] = []
        for key, value in client_capabilities.items():
            if isinstance(value, bool):
                if value:
                    raw_tokens.append(str(key))
            elif isinstance(value, (list, tuple, set)):
                raw_tokens.extend(str(item) for item in value)
            elif value:
                raw_tokens.append(str(key))
                raw_tokens.append(str(value))
    elif isinstance(client_capabilities, str):
        raw_tokens = client_capabilities.replace(",", " ").replace(";", " ").split()
    else:
        raw_tokens = [str(item) for item in client_capabilities]
    return sorted({_normalize_token(token) for token in raw_tokens if str(token).strip()})


def _is_broad_generalizability_question(question: str) -> bool:
    normalized = _normalize_token(question)
    if not normalized:
        return False
    tokens = {token for token in normalized.split("_") if token}
    if "generalizability" in tokens and tokens.intersection({"assess", "evaluate", "audit", "investigate"}):
        return True
    if "generalization" in tokens and tokens.intersection({"assess", "evaluate", "audit", "investigate"}):
        return True
    broad_phrases = [
        "assess_the_generalizability",
        "evaluate_generalizability",
        "assess_generalizability",
        "does_this_model_generalize",
        "whether_this_model_generalizes",
        "audit_generalization",
        "generalization_audit",
    ]
    return any(phrase in normalized for phrase in broad_phrases)


def _infer_audit_scope(question: str, dataset_description: str = "", constraints: str = "") -> str:
    text = _normalize_token(" ".join([question, dataset_description, constraints]))
    beyond_markers = [
        "beyond_the_paper",
        "beyond_paper",
        "not_restrict",
        "do_not_restrict",
        "outside_the_paper",
        "outside_paper",
        "beyond_reported_evaluations",
        "beyond_the_reported_evaluations",
        "construct_or_acquire",
        "public_regulatory",
        "external_datasets",
        "discover_axes",
        "discover_generalization",
        "representations_beyond",
        "not_limited_to",
        "do_not_limit",
        "raw_user_provided_dataset",
        "user_provided_dataset",
        "new_dataset",
        "external_dataset",
        "applicability",
        "appropriate_for",
        "should_be_used",
        "feature_model",
        "deployment",
    ]
    if any(marker in text for marker in beyond_markers):
        return "beyond_paper_discovery"
    return "paper_claim_audit"


def _normalize_audit_scope(audit_scope: str, question: str, dataset_description: str, constraints: str) -> str:
    normalized = _normalize_token(audit_scope or "auto")
    aliases = {
        "paper": "paper_claim_audit",
        "paper_claim": "paper_claim_audit",
        "paper_claims": "paper_claim_audit",
        "paper_claim_audit": "paper_claim_audit",
        "paper_anchored": "paper_claim_audit",
        "beyond_paper": "beyond_paper_discovery",
        "beyond_paper_discovery": "beyond_paper_discovery",
        "discovery": "beyond_paper_discovery",
        "open_discovery": "beyond_paper_discovery",
        "external_discovery": "beyond_paper_discovery",
    }
    if normalized == "auto":
        return _infer_audit_scope(question, dataset_description, constraints)
    return aliases.get(normalized, normalized)


def _infer_audit_depth(question: str, audit_scope: str, constraints: str = "") -> str:
    text = _normalize_token(" ".join([question, audit_scope, constraints]))
    screening_markers = {
        "quick_screen",
        "screening",
        "surface_audit",
        "smoke_test",
        "just_the_curve",
        "only_the_curve",
    }
    paper_markers = {
        "paper",
        "manuscript",
        "iclr",
        "claim",
        "finding",
        "why",
        "mechanism",
        "hypothesis",
        "investigate",
        "discover",
        "generalizability",
        "generalization",
        "beyond_paper_discovery",
    }
    if any(marker in text for marker in screening_markers):
        return "screening"
    if any(marker in text for marker in paper_markers):
        return "investigation"
    return "audit"


def _as_int(value: Any, field_name: str) -> int:
    if value is None:
        raise ValueError(f"Missing required field '{field_name}'")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Field '{field_name}' must be an integer") from exc
    return result


def available_procedures() -> Dict[str, Any]:
    """Return the procedure catalog exposed by this server."""
    return {
        "procedures": [
            {
                "name": PROCEDURE_NAME,
                "version": PROCEDURE_VERSION,
                "title": "Generalizability Analysis",
                "resource_uri": f"procedure://{PROCEDURE_NAME}/{PROCEDURE_VERSION}",
                "examples": list(_EXAMPLE_NAMES),
            }
        ]
    }


def get_procedure_document(
    name: str = PROCEDURE_NAME,
    version: str = PROCEDURE_VERSION,
) -> Dict[str, Any]:
    """Return the versioned Markdown procedure document."""
    if _normalize_token(name) != PROCEDURE_NAME:
        raise ValueError(f"Unknown procedure '{name}'")
    if version != PROCEDURE_VERSION:
        raise ValueError(f"Unsupported version '{version}'")
    return {
        "name": PROCEDURE_NAME,
        "version": PROCEDURE_VERSION,
        "mime_type": "text/markdown",
        "content": _procedure_text(),
    }


def get_procedure_examples(
    name: str = PROCEDURE_NAME,
    domain: Optional[str] = None,
) -> Dict[str, Any]:
    """Return worked examples for the requested procedure."""
    if _normalize_token(name) != PROCEDURE_NAME:
        raise ValueError(f"Unknown procedure '{name}'")

    example_names = [_normalize_token(domain)] if domain else list(_EXAMPLE_NAMES)
    examples = [
        {
            "domain": example_name,
            "mime_type": "text/markdown",
            "content": _example_text(example_name),
        }
        for example_name in example_names
    ]
    return {
        "name": PROCEDURE_NAME,
        "version": PROCEDURE_VERSION,
        "examples": examples,
    }


def start_spectra_audit_session(
    question: str,
    model_paper: Union[str, Dict[str, Any]],
    model_description: str = "",
    dataset_description: str = "",
    domain: str = "unknown",
    client_capabilities: Optional[Union[str, Sequence[str], Dict[str, Any]]] = None,
    output_root: str = "",
    constraints: str = "",
    max_cycles: Optional[int] = 6,
    audit_scope: str = "auto",
) -> Dict[str, Any]:
    """Create a role-structured /spectra audit session from a question and paper.

    MCP servers expose tools and prompts; the host agent decides whether it can
    spawn delegated workers. This session object gives capable clients a spawn
    plan and gives single-agent clients an equivalent sequential role loop.
    """
    question_text = question.strip()
    paper_text = _payload_to_text(model_paper).strip()
    model_text = model_description.strip()
    dataset_text = dataset_description.strip()
    constraints_text = constraints.strip()
    normalized_domain = _normalize_token(domain)
    broad_question = _is_broad_generalizability_question(question_text)
    normalized_audit_scope = _normalize_audit_scope(audit_scope, question_text, dataset_text, constraints_text)
    audit_depth = _infer_audit_depth(question_text, normalized_audit_scope, constraints_text)
    if broad_question and normalized_audit_scope == "beyond_paper_discovery":
        question_mode = "broad_beyond_paper_generalizability_discovery"
    elif broad_question:
        question_mode = "broad_model_generalizability_audit"
    elif normalized_audit_scope == "beyond_paper_discovery":
        question_mode = "targeted_beyond_paper_generalizability_question"
    else:
        question_mode = "targeted_generalizability_question"
    capability_tokens = _normalize_capability_tokens(client_capabilities)
    supports_filesystem = not capability_tokens or any(
        token in capability_tokens for token in {"filesystem", "files", "workspace", "codex", "claude_code"}
    )
    supports_network = any(token in capability_tokens for token in {"network", "web", "internet", "downloads"})
    supports_controller_loop = True

    missing = []
    if not question_text:
        missing.append("question")
    if not paper_text or paper_text in {"{}", "[]"}:
        missing.append("model_paper")
    if not model_text:
        missing.append("model_description")

    try:
        cycle_limit = None if max_cycles is None or int(max_cycles) <= 0 else max(1, int(max_cycles))
    except (TypeError, ValueError):
        cycle_limit = 6

    root = output_root.strip() or "spectra_audit_session"
    orchestration_mode = "single_codex_controller_session"

    shared_context = {
        "question": question_text,
        "question_mode": question_mode,
        "audit_scope": normalized_audit_scope,
        "audit_depth": audit_depth,
        "model_description": model_text,
        "dataset_description": dataset_text,
        "domain": normalized_domain,
        "constraints": constraints_text,
        "model_paper_reference": paper_text if len(paper_text) <= 500 else paper_text[:500] + "...",
        "output_root": root,
    }
    cheap_first_runtime_policy = {
        "name": "cheap_first_behavioral_runtime_policy",
        "purpose": (
            "Avoid wasting audit time on expensive solvers or full-scale jobs before "
            "the target behavior and controls are established, while keeping pilot "
            "evidence separate from confirmatory SPECTRA evidence."
        ),
        "required_order": [
            "Start with a bounded leakage-aware target-model slice when needed, but label it pilot-only unless it already covers the full split or a predeclared adequately powered stratified evaluation sample.",
            "Run cheap data, schema, label-balance, and leakage checks before model fitting.",
            "Run simple controls such as metadata/composition baselines and permutation or randomization controls when relevant.",
            "For frozen feature models, compute representations in cached/chunked form, then try deterministic non-iterative probes first: nearest-centroid/prototype scores, mean-difference linear scores, kNN/prototype retrieval, or closed-form small linear baselines.",
            "Use iterative logistic/SVM/ridge heads, fine-tuning, full-dataset embedding, all-pairs graphs, full target-model evaluation, or large ANN indexes only after the cheap probe is inconclusive, shows a live signal that needs confirmation, or a stronger deployment claim requires escalation.",
            "Before any heavy step, write its time/resource budget, success criterion, timeout/fallback condition, and cheaper fallback.",
            "After one timeout or runtime failure from a solver family, switch to the declared fallback or smaller slice instead of retrying the same slow method repeatedly.",
            "Treat cheap-probe evidence as pilot evidence: it can triage axes and guide hypothesis discovery, but it cannot support a valid SPC claim by itself when target-evaluation counts are tiny, such as about 10 examples per split.",
            "If a pilot shows monotone, localized, or practically meaningful degradation, expand the same frozen axis before final synthesis using the full eligible split or a predeclared stratified and length-balanced confirmation sample with a justified minimum per split.",
            "A promising axis is not complete after one coarse confirmation. Continue on the same frozen prospective axis first: increase split resolution, add examples per level, and focus extra samples around sparse, ambiguous, or rapidly changing regions until the trend shape is stable or the axis is invalidated.",
            "If a prospective axis shows no pattern, record it as a negative axis result, then inspect the already-computed successes and failures to discover candidate prospective similarity definitions. Any outcome-informed axis is exploratory until frozen and confirmed on fresh or expanded target-model evaluations.",
            "If a plausible axis is found from existing evaluated evidence, prior run artifacts, pilots, or outcome-informed success/failure analysis, do not treat that discovery-pool support as closure. Freeze the axis definition, construct or select new or explicitly adequate confirmation evidence, validate the frozen split before scoring, then run the target model and audit the confirmation against predeclared success criteria.",
            "A discovery-derived axis passes claim closure only if the frozen axis still supports the degradation, threshold, localized failure, or robust boundary on the confirmation evidence after controls. If confirmation fails, downgrade the axis, ledger the negative confirmation, merge those successes/failures into the evidence table, and continue prospective-axis discovery.",
            "After any failed confirmation, invalidated axis, weak axis, negative SPC, or non-explanatory SPC, run all-scored-evidence axis discovery before synthesis: use every target-model-scored example accumulated in the session, including successes, failures, weak curves, and failed confirmation panels, to identify the best prospectively definable axis or feature composite that consistently explains degradation.",
            "Reject axes that reverse direction across meaningful evidence slices, depend on cherry-picked panels, require reference answers, or belong to a feature family already falsified by fresh confirmation. Do not require each historical panel to have been purpose-built for the new axis or to meet the final effect threshold; those panels are discovery evidence, not confirmation evidence.",
            "If the current evaluated table or unused candidate pool cannot validly confirm the best all-scored-evidence hypothesis, freeze the hypothesis and route to dataset construction or acquisition. Current-pool exhaustion is not a stopping condition; construct or fetch data that spans the frozen axis with adequate levels and controls.",
            "If no current prospective feature explains the accumulated successes/failures, derive or acquire additional prospective features such as homology/family distance, MSA/search depth, fold/topology class, provenance, taxonomy, disorder/membrane annotations, or measured pretraining proximity, then repeat all-scored-evidence discovery.",
            "Distinguish split validity from claim closure. In broad generalizability mode, a valid negative or non-explanatory SPC is not terminal merely because the split contract is valid; it only closes an explicitly axis-specific question. Otherwise, ledger it as a negative result and continue to the next prospective axis.",
            "Before final synthesis, run a closure gate: the primary axis must be prospective, split-frozen before target scoring, measured in the intended order, adequately powered, densified enough to characterize monotone/threshold/U-shaped/localized/absent behavior, stable under expansion or explicitly reported as unstable, not materially explained by known confounders, and supported by fixed baselines when labels exist.",
            "If all-scored-evidence success/failure analysis identifies a prospective follow-up hypothesis, that live hypothesis blocks final synthesis until it is frozen and confirmed. If the hypothesis is not confirmable in the current pool, construct or acquire data that can test it; current-pool exhaustion is not a hard blocker.",
            "Do not stop at a weak bounded checkpoint, a coarse localized signal, or at 'no valid axis found after a search budget'. Continue by refining data, constructing missing prospective features, densifying the live axis, or freezing and confirming the next scientifically plausible axis unless the user stops the run or a hard external blocker prevents execution.",
        ],
    }

    controller_prompt = (
        "You are the persistent SPECTRA Controller. Execute the SPECTRA loop in one "
        "continuous agent process rather than spawning separate Distiller, Dataset "
        "Scout, Dataset Fetcher, Investigator, and Auditor agents. Treat those names "
        "as phases/functions inside your own loop. Keep shared state loaded: dataset "
        "tables, prediction/results tables, prospective features, candidate axes, "
        "negative-axis findings, exploratory outcome-informed hypotheses, frozen "
        "axes, validation gates, and blockers. Minimize artifacts: write a compact "
        "spectra_loop_state.json, an axis_ledger.json, commands_run.json, target "
        "model results when run or loaded, dataset artifacts only when constructed, "
        "and a final_report.md only after the claim-closure gate passes or a hard external "
        "blocker prevents further execution. Dataset construction is a first-class "
        "early phase: find/load data, inspect schema, retain inputs/labels/metadata/"
        "prospective features, deduplicate, audit leakage, and create one "
        "SPECTRA-ready table before repeatedly rediscovering data. Maintain one "
        "reusable candidate and prospective-feature table for the session; append "
        "features to it and reuse it for all axes instead of rebuilding separate "
        "one-off datasets. Prefer manifest-first expansion: collect or load "
        "candidate metadata, compute cheap prospective features, deduplicate or "
        "cluster near-duplicates, then sample balanced split levels. Do not use "
        "blind rejection sampling against external APIs when a manifest or "
        "candidate table can be built first. Run or load target-model performance "
        "on a useful evaluation pool once, then reuse that fixed result table to "
        "score candidate axes. When target-model evaluation is needed repeatedly, "
        "use a persistent evaluator when feasible: keep the model loaded and feed "
        "it evaluation pools instead of launching separate scripts that reload the "
        "model for each axis or confirmation set. Before expensive target-model "
        "calls, remove duplicates and near-duplicates according to available "
        "prospective identifiers or similarity features, and prioritize examples "
        "that add independent evidence to sparse or ambiguous SPC regions. Use "
        "fresh external data only when the existing evaluated pool or candidate "
        "manifest cannot answer the live hypothesis. For each candidate prospective "
        "axis, assign at least three nontrivial split levels when feasible, verify "
        "measured similarity decreases, run fixed baselines when labels exist, and "
        "score target-model performance by split. Treat three split levels as a "
        "minimum construction check, not sufficient closure for a continuous axis "
        "or localized signal. If an axis is positive or promising, freeze it and "
        "continue on that same axis first: densify the SPC by increasing split "
        "resolution, adding examples per level, focusing extra samples near the "
        "region where performance changes, and evaluating whether the trend is "
        "monotone, thresholded, U-shaped, localized, weak, absent, or unstable. "
        "Only move to a new axis after the current axis is stable enough to "
        "interpret, invalidated, or clearly confounded. Confirm the same axis using "
        "the full existing evaluated pool, a predeclared expanded sample, or "
        "fresh/independent data as required by the claim. When a plausible axis "
        "is discovered from existing evaluated evidence, prior run artifacts, "
        "pilot slices, or outcome-informed success/failure analysis, treat it as "
        "a live hypothesis rather than a claim-valid result. Before any new "
        "target-model scoring, freeze the axis definition: feature or composite "
        "features, direction, thresholds or level rule, metric, unit of analysis, "
        "inclusion/exclusion criteria, duplicate or cluster policy, sample-size "
        "target, controls, and success/failure criteria. Then construct or select "
        "confirmation evidence that is new relative to discovery whenever feasible: "
        "exclude prior target IDs, exact inputs, near-duplicates, or leakage-linked "
        "records unless the explicit claim boundary allows within-pool expansion. "
        "Validate the frozen confirmation split before scoring: nontrivial levels, "
        "measured similarity or axis order, adequate and balanced counts, duplicate "
        "diagnostics, baseline availability, and confound coverage. Run or load "
        "the target model only after the confirmation panel is frozen. The axis can "
        "pass claim closure only if it still supports the degradation, threshold, "
        "localized failure, or robust boundary on this confirmation evidence after "
        "controls. If confirmation fails, downgrade or mark the axis not confirmed, "
        "preserve it as a negative result, merge the discovery and confirmation "
        "successes/failures into the evidence table, and continue prospective-axis "
        "discovery. After any failed confirmation, invalidated axis, weak axis, "
        "negative SPC, or non-explanatory SPC, run all-scored-evidence axis "
        "discovery before synthesis: use every target-model-scored example "
        "accumulated in the session, including successes, failures, weak curves, "
        "and failed confirmation panels, to identify the best prospectively "
        "definable axis or feature composite that consistently explains "
        "degradation. Reject axes that reverse direction across meaningful "
        "evidence slices, depend on cherry-picked panels, require reference "
        "answers, or belong to a feature family already falsified by fresh "
        "confirmation. Do not require each historical panel to have been "
        "purpose-built for the new axis or to meet the final effect threshold; "
        "those panels are discovery evidence, not confirmation evidence. If the "
        "current evaluated table or unused candidate pool cannot validly confirm "
        "the best all-scored-evidence hypothesis, freeze the hypothesis and route "
        "to Dataset Scout, Dataset Fetcher, or direct dataset construction. "
        "Current-pool exhaustion is not a stopping condition. Construct or fetch "
        "data that spans the frozen axis with adequate levels and controls. If no "
        "current prospective feature explains the accumulated successes/failures, "
        "derive or acquire additional prospective features such as homology/family "
        "distance, MSA/search depth, fold/topology class, provenance, taxonomy, "
        "disorder/membrane annotations, or measured pretraining proximity, then "
        "repeat all-scored-evidence discovery. If an axis is negative, "
        "weak, non-monotonic, or non-explanatory, do "
        "not stop. Investigate the negative result by comparing examples where the "
        "target model did well with examples where it failed, identify prospective "
        "features that separate those groups, propose new candidate similarity axes, "
        "mark them outcome-informed/exploratory, freeze one before confirmation, and "
        "continue. A valid negative or non-explanatory SPC is not terminal in broad "
        "generalizability mode merely because the split contract is valid; it only "
        "closes an explicitly axis-specific question. If all-scored-evidence "
        "success/failure analysis identifies a prospective follow-up hypothesis, "
        "that live hypothesis blocks final synthesis until it is frozen and "
        "confirmed. If the hypothesis is not confirmable in the current pool, "
        "construct or acquire data that can test it; current-pool exhaustion is "
        "not a hard blocker. "
        "Do not use target-model errors, prediction/reference errors, held-"
        "out labels, or target confidence to define final split membership. Do not "
        "terminate merely because a weak bounded result exists, because a coarse "
        "localized signal exists, or because no valid axis has been found after a "
        "search budget. Stop only after the claim-closure gate passes, after user "
        "interruption, or after a concrete hard blocker such as unavailable data, "
        "missing credentials, or infeasible compute that you cannot work around. "
        "The closure gate requires that the primary axis is prospective and frozen "
        "before target scoring, measured in the intended order across split levels, "
        "resolved enough to characterize trend shape, densified after any promising "
        "signal, stable under expansion or explicitly reported as unstable, not "
        "materially explained by known confounders, confirmed on new or explicitly "
        "adequate evidence if discovered from existing results or outcome mining, "
        "and supported by fixed baselines when labels exist. In broad generalizability "
        "mode, claim closure also requires an interpretable degradation, threshold, "
        "localized failure, or robust boundary that answers the user question. A "
        "split-valid negative or non-explanatory curve does not satisfy this gate. "
        "A coarse, localized, "
        "weak, source-confounded, proxy-only, or shape-unstable curve does not pass "
        "closure just because it has three split levels and a visible signal; treat "
        "it as the next hypothesis seed and continue."
    )

    investigator_prompt = (
        "You are the SPECTRA Investigator. Execute the SPECTRA protocol for the "
        "model, dataset, task, metric, and similarity axis supplied in the handoff. "
        "Compute prospective similarities without using target-model errors, "
        "held-out labels, post-hoc prediction/reference comparisons, confidence "
        "scores derived from the target prediction, or any other outcome-dependent "
        "quantity to define the axis or split membership. Construct at least three "
        "nontrivial split levels or pretraining-test similarity bins when feasible. "
        "Verify that train-test or pretraining-test similarity decreases across the "
        "levels before trusting model metrics. If labels exist, run a simple fixed "
        "baseline across the same levels before evaluating the model of interest. "
        "Then evaluate the model of interest using the declared task metric. Small "
        "bounded target-model evaluations are pilot probes only unless the handoff "
        "explicitly defines them as an adequately powered confirmatory sample. For "
        "pilot probes, report signal/no-signal triage separately from SPC validity. "
        "If the pilot shows monotone, localized, or practically meaningful "
        "degradation, route to an expanded evaluation of the same frozen axis before "
        "claiming a valid SPC. If an axis is discovered from existing evaluated "
        "evidence, prior run artifacts, pilot slices, or success/failure mining, "
        "freeze the complete axis contract and construct or select new or explicitly "
        "adequate confirmation evidence before any new target-model scoring; report "
        "discovery-pool support separately from confirmation evidence. If the pilot "
        "or confirmed axis has mixed successes and failures but the "
        "current axis is non-explanatory, mine prospective metadata, sequence, family, "
        "homology, topology, MSA/search-depth, or provenance features to propose a new "
        "candidate similarity definition; mark that step exploratory and require a "
        "fresh frozen-axis confirmation before final claims. Use all target-model-scored "
        "examples accumulated in the session when proposing that new axis, reject "
        "directions that reverse across meaningful evidence slices, and treat "
        "current-pool insufficiency as a dataset-construction handoff rather than "
        "an invalid axis. Return "
        "the spectral performance curve, split assignments, split statistics, "
        "similarity progression, baseline results, model results, exact commands, "
        "runtime blockers, and a validity self-assessment under "
        "{root}/investigator_round_<n>. If the requested axis cannot produce valid "
        "decreasing-similarity levels, mark it invalid or exploratory, preserve the "
        "negative result, and hand back the observed success/failure structure for "
        "prospective-axis discovery; do not replace it with a post-hoc target-error axis."
    ).format(root=root)
    distiller_prompt = (
        "You are the SPECTRA Distiller. Turn user generalizability questions into "
        "SPECTRA analyses. Your goal is to identify or validate a prospective "
        "similarity axis that yields a meaningful spectral performance curve: model "
        "performance as train-test or pretraining-test similarity decreases. Use "
        "papers, claims, metadata, dataset schema, and domain knowledge to propose "
        "candidate axes, tasks, metrics, split levels, and controls. Interpret "
        "Investigator and Auditor results, distinguish valid curves from weak, "
        "invalid, or exploratory axes, avoid overclaiming, and return a clear answer "
        "about where the model generalizes or fails. Route to the Investigator with "
        "a specific model/dataset/task/metric/axis handoff, to Dataset Scout when a "
        "suitable dataset is missing, to Dataset Fetcher when a known dataset must "
        "be retrieved and packaged, to Auditor when an SPC needs independent "
        "validity review, or back into the loop for axis discovery when the current "
        "axis is weak, invalid, exploratory, negative, or non-explanatory. "
        "Pilot target-model slices are for triage, not closure. When an Auditor or "
        "Investigator reports a pilot signal, prefer routing back to expand the same "
        "frozen prospective axis before trying unrelated axes. When a prospective axis "
        "has no pattern, preserve it as a negative result and optionally route to "
        "exploratory similarity discovery over the observed successes/failures, but "
        "only to propose prospective axes that must be frozen and confirmed on fresh "
        "or expanded samples. When existing evaluated evidence produces a plausible "
        "axis, route to frozen-axis confirmation on new or explicitly adequate "
        "evidence before synthesis; do not treat discovery-pool support as a final "
        "valid SPC. Weak bounded summaries are checkpoints, not terminal "
        "answers, unless the user explicitly asks to stop. Do not add post-hoc failure-characterization tasks as "
        "final SPECTRA axes unless the user explicitly requested them; the default "
        "SPECTRA product is a claim-valid explanatory/degradation SPC or a concrete hard blocker. Continue negative "
        "and weak axes by searching for the next prospective similarity hypothesis. "
        "After a failed confirmation, select the best prospective axis that is "
        "consistent with all scored evidence seen so far; if the current pool cannot "
        "confirm it, route to dataset scouting, fetching, or construction instead of "
        "ending."
    ).format(root=root)
    scout_prompt = (
        "You are the SPECTRA Dataset Scout. Find datasets suitable for SPECTRA. "
        "Prioritize datasets with labels, metadata, prospective similarity features, "
        "enough examples for multiple split levels, clear access, and low leakage "
        "risk. Query the portable dataset catalog before open-ended search. Return "
        "candidate datasets, access routes, available fields, possible similarity "
        "axes, leakage and scale risks, and suitability rankings under "
        "{root}/dataset_scout_round_<n>."
    ).format(root=root)
    fetcher_prompt = (
        "You are the SPECTRA Dataset Fetcher. Retrieve and package datasets for "
        "SPECTRA. Load data, inspect schema, retain inputs, labels, metadata, and "
        "prospective similarity features, handle duplicates and missingness, and "
        "create SPECTRA-ready artifacts under {root}/dataset_fetcher_round_<n>. "
        "For very large pretraining datasets, use scalable filtering, approximate "
        "nearest-neighbor retrieval, sketches, shards, manifests, or bounded samples "
        "to estimate pretraining proximity rather than exhaustive comparison."
    ).format(root=root)
    auditor_prompt = (
        "You are the SPECTRA Auditor. Check whether the spectral performance curve "
        "supports the claim. Look for target-error leakage, test-label leakage, "
        "tiny or degenerate splits, non-decreasing train-test or pretraining-test "
        "similarity, unstable baselines, confounding, post-hoc axis selection, "
        "missing split statistics, and metric direction errors. Mark small target-model "
        "probes, including about 10 examples per split, as pilot-only weak or "
        "exploratory unless the Investigator provides a defensible power/sample-size "
        "justification and balanced sampling plan. If a pilot shows a degradation "
        "signal, require an expanded same-axis confirmation before any valid SPC "
        "claim. If a new axis was discovered from target-model successes or failures, "
        "mark it exploratory until the axis is frozen and confirmed on fresh or "
        "expanded target-model evaluations. If a plausible axis was found from "
        "existing evaluated evidence or prior artifacts, mark it discovery-only until "
        "a frozen confirmation panel is selected, split-valid before scoring, scored "
        "by the target model, and audited against predeclared success criteria. Mark "
        "each analysis as valid, weak, "
        "invalid, or exploratory only. Weak, invalid, exploratory, non-monotonic, or "
        "non-explanatory decisions must route back for outcome-informed prospective "
        "axis discovery rather than terminal bounded synthesis. That discovery must "
        "use all scored examples accumulated in the session and must not discard a "
        "consistent prospective hypothesis merely because the current unused pool is "
        "too small; route to dataset construction or acquisition when needed. In broad generalizability mode, "
        "a valid negative or non-explanatory SPC is a ledgered negative result, not a "
        "terminal claim-valid SPC. Write an audit report, validity decision, "
        "detected risks, and required fixes under "
        "{root}/auditor_round_<n>."
    ).format(root=root)
    synthesis_prompt = (
        "You are the final SPECTRA Synthesis Distiller. Read the model paper, "
        "Distiller plans, Investigator SPC artifacts, Dataset Scout/Fetcher outputs, "
        "and Auditor validity decisions. Write a paper-ready finding only "
        "when a claim-valid explanatory/degradation SPC is supported, the user "
        "explicitly asked to stop at a checkpoint, or a hard external blocker prevents "
        "further execution. Report which "
        "axes are valid, weak, invalid, or exploratory; whether train-test or "
        "pretraining-test similarity actually decreases; baseline behavior; target "
        "model behavior; and what claim is supported. If the claimed axis was found "
        "from existing evidence, verify that the report includes a frozen-axis "
        "confirmation on new or explicitly adequate evidence; otherwise route back "
        "instead of writing terminal synthesis. If the SPC is weak, invalid, "
        "non-explanatory, negative, or only exploratory, say that directly and route back with "
        "specific axis-discovery or dataset-construction fixes instead of ending."
    ).format(root=root)

    role_specs = {
        "controller": {
            "spawn_condition": "single autonomous process for all SPECTRA reasoning and execution",
            "prompt": controller_prompt,
            "writes": [
                "spectra_loop_state.json",
                "axis_ledger.json",
                "commands_run.json",
                "spectra_ready_dataset/ when data are constructed",
                "target_model_results.csv when target-model outputs are run or loaded",
                "final_report.md only for a claim-valid explanatory/degradation SPC or hard external blocker",
            ],
        },
        "investigator": {
            "spawn_condition": "after Distiller selects a model/dataset/task/metric/similarity-axis handoff",
            "prompt": investigator_prompt,
            "writes": [
                "spectral_performance_curve.csv",
                "split_assignments/",
                "split_statistics.json",
                "similarity_progression.csv",
                "baseline_results.csv",
                "model_results.csv",
                "validity_self_assessment.json",
                "runtime_budget_and_fallbacks.md",
                "commands_run.json",
                "investigator_checkpoint.md",
                "auditor_handoff.json",
            ],
        },
        "distiller": {
            "spawn_condition": "at session bootstrap and after Auditor/Scout/Fetcher handoffs",
            "prompt": distiller_prompt,
            "writes": [
                "spectra_analysis_plan.json",
                "distiller_report.md",
                "routing_decision.json",
                "investigator_handoff.json",
                "dataset_scout_handoff.json",
                "dataset_fetcher_handoff.json",
                "paper_ready_checkpoint.md",
                "overclaim_guardrails.md",
            ],
        },
        "dataset_scout": {
            "spawn_condition": "when Distiller needs candidate datasets for a SPECTRA axis or task",
            "prompt": scout_prompt,
            "writes": [
                "candidate_resources.csv",
                "candidate_resources.json",
                "scout_decision.json",
                "resource_risk_log.md",
                "dataset_fetcher_handoff.json",
            ],
        },
        "dataset_fetcher": {
            "spawn_condition": "when Distiller or Dataset Scout selects a dataset to retrieve and package",
            "prompt": fetcher_prompt,
            "writes": [
                "dataset_manifest.json",
                "schema_report.json",
                "spectra_ready_schema.json",
                "mapping_validation.json",
                "leakage_audit.json",
                "deduplication_report.json",
                "missingness_report.json",
                "spectra_ready_dataset/",
                "fetcher_handoff.json",
            ],
        },
        "auditor": {
            "spawn_condition": "after Investigator produces an SPC and split statistics",
            "prompt": auditor_prompt,
            "writes": [
                "audit_report.md",
                "validity_decision.json",
                "risk_register.json",
                "required_fixes.json",
                "distiller_handoff.json",
            ],
        },
        "synthesis_distiller": {
            "spawn_condition": "only after a claim-valid explanatory/degradation SPC is supported or the user explicitly asks for a checkpoint",
            "prompt": synthesis_prompt,
            "writes": [
                "paper_ready_spectra_finding.md",
                "claim_boundary.json",
                "evidence_to_claim_table.csv",
                "overclaim_guardrails.md",
            ],
        },
    }

    spawn_plan = [
        {
            "role": "controller",
            "name": "SPECTRA Controller",
            "write_scope": root,
            "initial_prompt": controller_prompt,
            "handoff_inputs": ["model_paper", "question", "model_description", "dataset_description"],
        }
    ]

    return {
        "mode": "spectra_audit_session",
        "status": "blocked" if missing else "ready",
        "missing_inputs": missing,
        "orchestration_mode": orchestration_mode,
        "client_capabilities": capability_tokens,
        "client_orchestration_contract": (
            "The MCP server supplies one SPECTRA Controller prompt plus quality "
            "gates. Distiller, Investigator, Dataset Scout, Dataset Fetcher, "
            "Auditor, and synthesis are internal controller phases, not separate "
            "client-routed agents. The host should launch one Codex/controller "
            "session and let that session own iteration and terminality."
        ),
        "shared_context": shared_context,
        "runtime_probe_policy": cheap_first_runtime_policy,
        "broad_question_behavior": {
            "enabled": broad_question,
            "audit_scope": normalized_audit_scope,
            "audit_depth": audit_depth,
            "front_door_request": "Use /spectra to assess the generalizability of this model.",
            "paper_context_policy": (
                "The paper is a claim source and model-context source. In "
                "paper_claim_audit scope, prioritize the paper's reported benchmarks. "
                "In beyond_paper_discovery scope, use the paper to seed hypotheses but "
                "do not restrict the audit to paper datasets or axes."
            ),
            "paper_first_steps": [
                "Extract the model paper's stated generalization claims, datasets, splits, and evaluation protocols.",
                "Map each claim to a scientific unit, task, metric, prospective similarity axis, and feasible SPC design.",
                "Reject post-hoc axes that require target-model errors or reference answers to define split membership.",
                "Prefer analyses that can produce at least three decreasing-similarity split levels and baseline validation.",
            ],
            "beyond_paper_steps": [
                "Identify scientifically important prospective axes not directly evaluated in the paper.",
                "Search for or fetch public/local datasets when current data cannot support a claim-valid explanatory/degradation SPC.",
                "Prefer split-based or pretraining-proximity SPCs with fixed baselines over post-hoc failure characterization.",
                "Route to Dataset Scout or Dataset Fetcher when a suitable SPECTRA-ready dataset is missing.",
            ] if normalized_audit_scope == "beyond_paper_discovery" else [],
            "desired_reference_behavior": [
                "Keep one SPECTRA loop state across dataset construction, model evaluation, axis discovery, validation, and audit decisions.",
                "Construct or load the dataset once, then reuse the SPECTRA-ready table.",
                "Run or load target-model performance once for a useful evaluation pool, then reuse those results for axis discovery.",
                "When an axis is weak, negative, non-monotonic, or non-explanatory, investigate the successes/failures to propose the next prospective axis instead of stopping.",
                "When existing evidence produces a plausible axis, freeze the full axis contract, build or select new/adequate confirmation evidence before scoring, and close only if the frozen axis still holds.",
                "Return a final answer only for a claim-valid explanatory/degradation SPC, explicit user-requested checkpoint, or hard external blocker.",
            ],
        },
        "supports": {
            "subagent_delegation": False,
            "single_controller_loop": True,
            "filesystem_artifacts": supports_filesystem,
            "network_resource_acquisition": supports_network,
        },
        "max_cycles": "unbounded" if cycle_limit is None else cycle_limit,
        "initial_tool_sequence": [
            "get_procedure",
            "get_procedure_examples",
            "suggest_reusable_spectra_memory",
            "suggest_dataset_catalog_entries",
            "start_generalizability_analysis",
            "select_spectra_execution_mode",
            "plan_spectral_performance_curve",
            "suggest_similarity_definitions",
            "suggest_similarity_computation_strategies",
        ],
        "phase_graph": role_specs,
        "role_graph": role_specs,
        "spawn_plan": spawn_plan,
        "sequential_fallback_plan": [
            "Run the SPECTRA Controller as one Codex session. It performs dataset construction, target-model result loading/running, prospective-axis discovery, split validation, baseline checks, model scoring, audit checks, and continuation decisions inside one loop.",
            "Do not emulate a Distiller/Investigator/Auditor handoff loop in Python. Those names are internal phases the controller uses while thinking.",
            "Do not stop on weak bounded findings or no-axis search-budget summaries; continue axis discovery from negative results unless the user stops or a hard external blocker is reached.",
        ],
        "routing_policy": [
            "The controller owns all routing decisions internally.",
            "Distiller, Investigator, Dataset Scout, Dataset Fetcher, Auditor, and synthesis are internal controller phases, not separate launched roles.",
            "Weak, invalid, exploratory, negative, or non-explanatory decisions continue into outcome-informed prospective-axis discovery and then frozen-axis confirmation.",
            "A plausible axis found in existing evaluated evidence is not terminal until it is frozen and confirmed on new or explicitly adequate evidence under predeclared success criteria.",
            "After failed confirmation, the controller must mine all target-model-scored examples accumulated so far and select the best prospectively definable axis that is consistent across the combined evidence; current-pool insufficiency routes to dataset scouting/fetching/construction, not synthesis.",
            "A weak bounded result and a 'no valid axis after search budget' result are not terminal stop conditions.",
            "Final synthesis is terminal only for a claim-valid explanatory/degradation SPC, an explicit user-requested checkpoint, or a hard external blocker.",
        ],
        "terminal_gate": {
            "required_tool": "synthesize_spectra_generalizability_finding",
            "terminal_condition": "A claim-valid explanatory/degradation SPC is supported with decreasing prospective similarity, baseline behavior when labels exist, target-model behavior, and an explicit claim boundary; or the user explicitly asks to stop at a checkpoint; or a hard external blocker prevents further execution. A split-valid negative or non-explanatory SPC is terminal only for an explicitly axis-specific question.",
            "not_terminal_conditions": [
                "the only result is weak but useful bounded evidence",
                "no valid axis has been found after a search budget",
                "the current axis is negative, non-monotonic, or non-explanatory and target-model successes/failures have not been mined for new prospective axes",
                "the current axis is split-valid but negative or non-explanatory in broad generalizability mode",
                "all target-model-scored examples accumulated so far have not been used to discover the best prospectively definable follow-up axis after a failed confirmation",
                "an all-scored-evidence prospective follow-up hypothesis exists but has not been frozen and confirmed",
                "the current evaluated table or unused candidate pool cannot confirm the best all-scored-evidence hypothesis but dataset scouting, fetching, or construction has not been attempted",
                "a plausible axis is supported only in existing discovery evidence and has not yet been frozen, confirmed on new or explicitly adequate evidence, and audited against predeclared success criteria",
                "fresh or expanded confirmation of a previously plausible axis failed and the combined successes/failures have not been routed back into prospective-axis discovery",
                "target-model errors or post-hoc prediction/reference comparisons define the axis or split membership",
                "held-out labels define split membership",
                "fewer than three nontrivial split levels are used when more are feasible",
                "train-test or pretraining-test similarity does not decrease across levels",
                "labels exist but no fixed baseline was run or justified",
                "split sizes are too small or degenerate for the claimed conclusion",
            ],
        },
        "artifact_tree": {
            "session_manifest": f"{root}/session_manifest.json",
            "controller_prompt": f"{root}/controller_prompt.md",
            "controller_state": f"{root}/spectra_loop_state.json",
            "axis_ledger": f"{root}/axis_ledger.json",
            "commands": f"{root}/commands_run.json",
            "target_model_results": f"{root}/target_model_results.csv",
            "final_report": f"{root}/final_report.md",
        },
        "claim_boundary_policy": [
            "State whether each axis is a valid SPC, weak SPC, invalid SPC, negative result, or exploratory-only analysis.",
            "Do not generalize from a frozen probe to full fine-tuning unless that protocol was tested.",
            "Report failed or invalid similarity axes as findings and use them to drive the next prospective-axis search instead of replacing them with post-hoc target-error axes.",
            "Do not close the loop merely because a weak bounded checkpoint is available or no valid axis has been found under an arbitrary search budget.",
            "Separate what /spectra caused the agent to do from what the deterministic audit engine computed.",
        ],
    }


def prepare_spectra_audit_session(
    question: str,
    model_paper: str,
    model_description: str,
    output_root: str,
    dataset_description: str = "",
    domain: str = "unknown",
    constraints: str = "",
    client_capabilities: Optional[List[str]] = None,
    audit_scope: str = "auto",
) -> Dict[str, Any]:
    """Prepare a single-controller /spectra session without executing an agent."""
    from .controller_session import SpectraControllerSessionConfig, prepare_controller_session

    return prepare_controller_session(
        SpectraControllerSessionConfig(
            question=question,
            model_paper=model_paper,
            model_description=model_description,
            dataset_description=dataset_description,
            domain=domain,
            constraints=constraints,
            output_root=output_root,
            client_capabilities=client_capabilities or ["filesystem", "network"],
            audit_scope=audit_scope,
            dry_run=True,
        )
    )


def run_spectra_audit_session(
    question: str,
    model_paper: str,
    model_description: str,
    output_root: str,
    dataset_description: str = "",
    domain: str = "unknown",
    constraints: str = "",
    client_capabilities: Optional[List[str]] = None,
    agent_command_template: str = "",
    execute_controller: bool = False,
    audit_scope: str = "auto",
) -> Dict[str, Any]:
    """Run or dry-run a single-controller /spectra session.

    This tool is safe by default: it prepares one controller prompt unless
    execute_controller is true and an agent_command_template is supplied.
    """
    from .controller_session import SpectraControllerSessionConfig, run_controller_session

    return run_controller_session(
        SpectraControllerSessionConfig(
            question=question,
            model_paper=model_paper,
            model_description=model_description,
            dataset_description=dataset_description,
            domain=domain,
            constraints=constraints,
            output_root=output_root,
            client_capabilities=client_capabilities or ["filesystem", "network"],
            agent_command_template=agent_command_template,
            audit_scope=audit_scope,
            dry_run=not execute_controller,
        )
    )


def start_generalizability_analysis(
    dataset_description: str,
    model_description: str,
    domain: str = "unknown",
    objective: str = "",
    constraints: str = "",
) -> Dict[str, Any]:
    """Create an SPC-focused generalizability analysis plan."""
    missing = []
    if not dataset_description.strip():
        missing.append("dataset_description")
    if not model_description.strip():
        missing.append("model_description")

    normalized_domain = _normalize_token(domain)
    recommended_example = (
        normalized_domain if normalized_domain in _EXAMPLE_NAMES else None
    )
    return {
        "procedure": {
            "name": PROCEDURE_NAME,
            "version": PROCEDURE_VERSION,
        },
        "status": "blocked" if missing else "ready",
        "missing_inputs": missing,
        "domain": normalized_domain,
        "recommended_example": recommended_example,
        "objective": objective.strip(),
        "constraints": constraints.strip(),
        "execution_steps": [
            "Identify the dataset sample unit and prediction target.",
            "Decide the execution mode with select_spectra_execution_mode; benchmark mode is required when raw labels and a trainable model or baseline are available.",
            "Prefer the persistent SPECTRA Controller loop; use the Distiller phase to map the question to a model, dataset, task, metric, prospective similarity axis, and SPC validity requirements.",
            "Use Dataset Scout or Dataset Fetcher if a suitable labeled dataset or SPECTRA-ready package is missing.",
            "Construct or load the dataset once, then keep the SPECTRA-ready table in loop state for all candidate axes.",
            "Compute split or pretraining-proximity similarities from prospective inputs, metadata, provenance, or pretraining-reference features only.",
            "Construct at least three nontrivial split levels or pretraining-test similarity bins when feasible.",
            "Measure and report train-test or pretraining-test similarity progression before evaluating the model.",
            "Mark the SPC invalid or exploratory when similarity does not decrease, split levels are tiny or degenerate, or the axis is post-hoc.",
            "When labels exist, run a simple fixed baseline across the same levels before evaluating the target model.",
            "Evaluate the target model only after the split contract is validated or explicitly labeled exploratory.",
            "Send Investigator artifacts to the Auditor for leakage, split-statistic, baseline, metric-direction, and confounding review.",
            "If a plausible axis is found from existing evaluated evidence, prior artifacts, pilots, or success/failure mining, freeze the full axis contract and confirm it on new or explicitly adequate evidence before claiming closure.",
            "If the axis is weak, invalid, exploratory, negative, non-monotonic, or non-explanatory, inspect target-model successes/failures to discover the next prospective axis, freeze it, and confirm it.",
            "Return a final answer only for a claim-valid explanatory/degradation SPC, an explicit user-requested checkpoint, or a hard external blocker.",
        ],
        "required_artifacts": [
            "execution_mode_decision",
            "spectra_analysis_plan",
            "scientific_unit",
            "task_and_metric",
            "candidate_similarity_axes",
            "axis_leakage_classification",
            "similarity_computation_plan",
            "split_assignments",
            "split_statistics",
            "similarity_progression",
            "baseline_results",
            "model_results",
            "spectral_performance_curve",
            "validity_self_assessment",
            "auditor_validity_decision",
            "risk_register",
            "required_fixes",
            "claim_boundary",
            "overclaim_guardrails",
            "commands_run",
            "retraining_manifest_or_audit_fallback_reason",
            "blockers",
            "generalizability_report",
        ],
        "quality_gates": [
            "The declared similarity axis is prospective and does not use target-model errors, prediction/reference errors, held-out labels, or target confidence to define levels.",
            "At least three usable split or similarity-bin levels are present when feasible.",
            "Measured train-test or pretraining-test similarity decreases across levels.",
            "Train and test sizes remain large enough for the intended metric.",
            "Labels, if present, are used for evaluation and baselines only, not split membership.",
            "A simple fixed baseline is run across the same levels when labels exist, or the omission is justified.",
            "Benchmark-mode claims use fresh training per split when retraining is part of the model evaluation protocol.",
            "Audit-fallback claims are labeled diagnostic or exploratory.",
            "The Auditor classifies the SPC as valid, weak, invalid, or exploratory before final interpretation.",
            "The final report separates invalid split construction, weak axis evidence, and target-model performance failure.",
        ],
    }


def plan_spectral_performance_curve(
    model_description: str,
    dataset_description: str,
    task: str = "",
    metric: str = "",
    similarity_axis: str = "",
    labels_available: bool = True,
    retraining_feasible: bool = False,
    min_levels: int = 3,
) -> Dict[str, Any]:
    """Return the minimal contract for a valid SPECTRA performance curve."""
    blocked = []
    if not model_description.strip():
        blocked.append("model_description")
    if not dataset_description.strip():
        blocked.append("dataset_description")
    if not task.strip():
        blocked.append("task")
    if not metric.strip():
        blocked.append("metric")
    if not similarity_axis.strip():
        blocked.append("similarity_axis")

    try:
        requested_levels = max(2, int(min_levels))
    except (TypeError, ValueError):
        requested_levels = 3

    return {
        "mode": "spectral_performance_curve_plan",
        "status": "blocked" if blocked else "ready",
        "missing_inputs": blocked,
        "model_description": model_description.strip(),
        "dataset_description": dataset_description.strip(),
        "task": task.strip(),
        "metric": metric.strip(),
        "similarity_axis": similarity_axis.strip(),
        "minimum_nontrivial_levels": requested_levels,
        "axis_contract": {
            "must_be_prospective": True,
            "forbidden_axis_inputs": [
                "target-model errors",
                "prediction/reference errors",
                "held-out labels",
                "target prediction confidence when derived from the evaluated model",
                "any quantity computed after seeing target-model failures",
            ],
            "allowed_axis_inputs": [
                "raw inputs",
                "metadata available before prediction",
                "training/pretraining provenance",
                "domain annotations available before evaluation",
                "precomputed features not derived from target-model errors",
            ],
        },
        "investigator_requirements": [
            "Compute train-test or pretraining-test similarities before model evaluation.",
            "Construct split levels or similarity bins with nondegenerate sample counts.",
            "Verify measured similarity decreases across levels.",
            "Run a simple fixed baseline across the same levels when labels exist.",
            "Evaluate the model of interest only after the split contract is valid or explicitly marked exploratory.",
        ],
        "auditor_requirements": [
            "Check target-error leakage.",
            "Check test-label leakage.",
            "Check split sizes and level degeneracy.",
            "Check that measured similarity decreases.",
            "Check baseline stability and metric direction.",
            "Check confounding and post-hoc axis selection.",
            "Classify the SPC as valid, weak, invalid, or exploratory.",
        ],
        "expected_artifacts": [
            "spectral_performance_curve.csv",
            "split_assignments/",
            "split_statistics.json",
            "similarity_progression.csv",
            "baseline_results.csv" if labels_available else "baseline_omission_reason.md",
            "model_results.csv",
            "validity_self_assessment.json",
            "audit_report.md",
            "validity_decision.json",
        ],
        "evaluation_protocol": (
            "fresh_train_per_split" if retraining_feasible else "fixed_prediction_or_external_model_audit"
        ),
        "claim_boundary": (
            "Benchmark-mode evidence requires fresh training per split when the target "
            "claim is about train-test generalization. Fixed predictions or external "
            "model calls can support a diagnostic/applicability SPC only when that "
            "limitation is reported."
        ),
    }


def select_spectra_execution_mode(
    has_raw_labeled_data: bool,
    has_trainable_model_or_baseline: bool,
    can_retrain: bool = True,
    has_fixed_predictions: bool = False,
    retraining_unavailable_reason: str = "",
) -> Dict[str, Any]:
    """Choose benchmark mode or audit fallback mode for a SPECTRA run.

    Benchmark mode is the core SPECTRA mode: construct similarity-controlled
    train/test splits and train a fresh model for each split. Audit fallback mode
    is diagnostic only and should be used when retraining cannot be done.
    """
    reason = retraining_unavailable_reason.strip()
    if has_raw_labeled_data and has_trainable_model_or_baseline and can_retrain:
        mode = "benchmark"
        status = "benchmark_required"
        claim_strength = "full_spectra_benchmark_evidence"
        required_next_steps = [
            "Construct or approximate a prospective similarity graph over scientific units.",
            "Generate at least three similarity-controlled train/test split levels.",
            "Train a fresh model or baseline independently for each split.",
            "Evaluate only on the held-out units for that split.",
            "Report performance as a function of measured train-test overlap.",
        ]
        warnings: List[str] = []
    elif has_fixed_predictions and (not has_raw_labeled_data or not has_trainable_model_or_baseline or not can_retrain):
        mode = "audit_fallback"
        status = "diagnostic_only"
        claim_strength = "fixed_prediction_diagnostic_not_full_benchmark_evidence"
        required_next_steps = [
            "Record why split-level retraining is unavailable.",
            "Construct fixed-prediction similarity bins or a measured train-eval axis.",
            "Report the curve as diagnostic evidence only.",
            "Specify the data, labels, or training code needed to upgrade to benchmark mode.",
        ]
        warnings = [
            "Do not present fixed-prediction binning as the primary SPECTRA benchmark result.",
        ]
        if not reason:
            warnings.append("Audit fallback mode requires an explicit retraining_unavailable_reason.")
    else:
        mode = "blocked"
        status = "insufficient_inputs"
        claim_strength = "none"
        required_next_steps = [
            "Obtain raw labels and features plus a trainable model, model code, or defensible baseline.",
            "If only fixed predictions are available, provide them and document why retraining is impossible.",
        ]
        warnings = [
            "SPECTRA needs either benchmark-mode training inputs or fixed predictions for diagnostic audit fallback.",
        ]

    return {
        "mode": mode,
        "status": status,
        "claim_strength": claim_strength,
        "benchmark_mode_required": mode == "benchmark",
        "audit_fallback_allowed": mode == "audit_fallback",
        "fixed_prediction_binning_allowed_as_primary_result": False,
        "retraining_unavailable_reason": reason,
        "required_next_steps": required_next_steps,
        "warnings": warnings,
    }


def decide_next_spectra_experiment(
    axis_name: str,
    axis_result_status: str,
    behavior_tested: bool,
    benchmark_mode_feasible: bool,
    has_suspicious_data_overlap: bool = False,
    concrete_blocker: str = "",
    blocker_recovery_attempted: bool = False,
    tested_axis_count: int = 0,
    min_axis_tests: int = 5,
    axis_search_exhausted: bool = False,
    supported_axis_replicated: bool = False,
    replication_reflection_complete: bool = False,
    axis_depth_level: str = "",
    mechanism_debt_satisfied: bool = False,
    mechanism_infeasibility_proof_provided: bool = False,
) -> Dict[str, Any]:
    """Decide what an exploratory SPECTRA agent should do after one result.

    SPECTRA should not stop after a data-only screen. A suspicious overlap axis
    is a hypothesis that requires a targeted feasible behavioral experiment.
    """
    normalized_status = _normalize_token(axis_result_status)
    normalized_depth = _normalize_token(axis_depth_level)
    blocker = concrete_blocker.strip()
    warnings: List[str] = []
    supported_status = normalized_status in {
        "monotonic_supported",
        "localized_supported",
        "weak_supported",
    }
    debt_creating_depths = {
        "model_space_pointer",
        "surface_proxy",
        "domain_proxy",
        "curated_annotation_axis",
        "mechanistic_hypothesis_without_disambiguation",
        "unclassified_axis",
    }
    axis_depth_unknown = supported_status and not normalized_depth
    mechanism_debt_pending = (
        supported_status
        and bool(normalized_depth)
        and normalized_depth != "mechanism_supported_with_controls"
        and (
            normalized_depth in debt_creating_depths
            or "proxy" in normalized_depth
            or "model_space" in normalized_depth
        )
        and not mechanism_debt_satisfied
        and not mechanism_infeasibility_proof_provided
    )

    if blocker and axis_depth_unknown:
        decision = "classify_explanatory_depth"
        required_next_action = (
            "A blocker cannot close a supported replicated axis while explanatory depth is unknown. "
            "Classify the axis with assess_explanatory_depth, then decide whether the blocker belongs "
            "to mechanism debt, replication, environment repair, or scope expansion."
        )
        complete = False
        warnings.append("Supported axes require explanatory-depth classification before blocker-based stopping.")
    elif blocker and mechanism_debt_pending:
        decision = "enforce_mechanism_debt_gate"
        required_next_action = (
            "A blocker cannot bypass mechanism debt for a supported proxy/model-space axis. "
            "Call enforce_mechanism_debt_gate and either execute the next required tier or produce "
            "a continuation checkpoint with launch evidence for the next executable mechanism experiment."
        )
        complete = False
        warnings.append("Mechanism-debt blockers must be handled through enforce_mechanism_debt_gate.")
    elif blocker and not blocker_recovery_attempted:
        decision = "attempt_blocker_recovery"
        required_next_action = (
            "Do not stop yet. Inspect the model repository and dependency files, "
            "attempt a reasonable environment/data/code repair in the "
            "allowed workspace, then retry the target-model behavioral test. "
            "If that recovery attempt fails or would require unavailable data, credentials, "
            "hardware, or excessive compute, launch a smaller executable fallback or alternate route."
        )
        complete = False
        warnings.append("A concrete blocker requires a documented recovery attempt before it can end the analysis.")
    elif blocker:
        decision = "continuation_checkpoint_with_blocker"
        required_next_action = (
            "Do not terminate the SPECTRA search. Record the blocker and recovery attempt, "
            "then launch the next executable route: alternate task, alternate axis, public resource, "
            "constructed hypothesis-test dataset, smaller validation slice, or bounded continuation job. "
            "A manifest that only queues future work is not sufficient."
        )
        complete = False
    elif has_suspicious_data_overlap and not behavior_tested and benchmark_mode_feasible:
        decision = "run_behavioral_followup"
        required_next_action = (
            "Select a tractable suspicious task/axis, generate controlled "
            "splits, and train a fresh downstream probe/head/baseline per split."
        )
        complete = False
        warnings.append("Data-only overlap is pre-benchmark screening, not a completed SPECTRA result.")
    elif benchmark_mode_feasible and not behavior_tested:
        decision = "run_behavioral_followup"
        required_next_action = "Run a model/probe performance-overlap experiment before treating this axis as answered."
        complete = False
    elif normalized_status in {"not_explanatory", "not_evaluable"} and axis_search_exhausted:
        decision = "expand_search_scope"
        required_next_action = (
            "Do not conclude that no degradation axis exists or that the model is "
            "generally generalizable. The current search scope failed "
            "to find a supported behavioral axis, so launch a larger scope or "
            "bounded fallback: additional feature modalities, metadata, tasks, labels, "
            "model representations, checkpoints, compute, or domain-specific axes."
        )
        complete = False
        warnings.append("Current-scope exhaustion is an escalation condition, not a completed SPECTRA conclusion.")
    elif normalized_status in {"not_explanatory", "not_evaluable"}:
        decision = "try_next_axis"
        required_next_action = (
            "Record the negative finding and choose another similarity hypothesis, "
            "preferably from an untested axis class. A SPECTRA exploration should "
            "not stop after non-explanatory curves while plausible axes or compute "
            "budget remain."
        )
        complete = False
        if tested_axis_count < min_axis_tests:
            warnings.append(
                "Only %d axis tests recorded; broad searches should usually test at least %d axes or document why that is impossible."
                % (tested_axis_count, min_axis_tests)
            )
    elif normalized_status == "localized_supported":
        decision = "refine_axis"
        required_next_action = "Refine the axis around the supported region or test a composite similarity."
        complete = False
    elif normalized_status in {"monotonic_supported", "weak_supported"}:
        if (
            normalized_status == "monotonic_supported"
            and supported_axis_replicated
            and replication_reflection_complete
        ):
            if axis_depth_unknown:
                decision = "classify_explanatory_depth"
                required_next_action = (
                    "Do not select a supported axis before classifying explanatory depth. "
                    "Call assess_explanatory_depth, then enforce_mechanism_debt_gate if "
                    "the axis is proxy-level, model-space, broad domain-proxy, or lacks controls."
                )
                complete = False
                warnings.append("A supported replicated axis still needs explanatory-depth classification before selection.")
            elif mechanism_debt_pending:
                decision = "enforce_mechanism_debt_gate"
                required_next_action = (
                    "Do not select this supported axis yet. It is proxy-level, model-space, "
                    "or lacks controls, so call enforce_mechanism_debt_gate and execute the "
                    "required mechanism/mediation/public-resource/constructed-dataset tier."
                )
                complete = False
                warnings.append("Supported proxy/model-space axes require continuation until mechanism-level evidence with controls or launch evidence for the next executable continuation exists.")
            else:
                decision = "continue_after_supported_axis"
                required_next_action = (
                    "Treat this as a checkpoint, not a stopping point. Continue with the next "
                    "mechanism-level or scope-expansion experiment unless every feasible task has "
                    "been covered and the axis is supported by a mechanism-level explanation with controls."
                )
                complete = False
        elif supported_axis_replicated:
            decision = "reflect_on_replication_pattern"
            required_next_action = (
                "Do not stop at replicated support. Compare supported and "
                "non-supported tasks/datasets, explain why the candidate axis "
                "works in some settings but not others, then derive and test "
                "residual or composite axes that could reveal the next failure "
                "mode beyond the current similarity definition."
            )
            complete = False
        else:
            decision = "replicate_candidate_axis"
            required_next_action = (
                "Do not stop with a proposal if additional labeled tasks, datasets, "
                "splits, seeds, or stronger target-model heads are available. "
                "Immediately test whether this candidate axis is supported on "
                "another task/dataset/model setting or stronger operating point; "
                "only escalate scope if replication is not executable in the "
                "current workspace."
            )
            complete = False
    else:
        decision = "clarify_result_status"
        required_next_action = "Normalize the axis status, then decide whether to test behavior, refine, replicate, expand scope, or write a continuation checkpoint."
        complete = False
        warnings.append("Unrecognized axis_result_status: %s" % axis_result_status)

    return {
        "axis_name": axis_name,
        "axis_result_status": normalized_status,
        "behavior_tested": behavior_tested,
        "benchmark_mode_feasible": benchmark_mode_feasible,
        "has_suspicious_data_overlap": has_suspicious_data_overlap,
        "blocker_recovery_attempted": blocker_recovery_attempted,
        "tested_axis_count": tested_axis_count,
        "min_axis_tests": min_axis_tests,
        "axis_search_exhausted": axis_search_exhausted,
        "supported_axis_replicated": supported_axis_replicated,
        "replication_reflection_complete": replication_reflection_complete,
        "axis_depth_level": normalized_depth,
        "axis_depth_unknown": axis_depth_unknown,
        "mechanism_debt_pending": mechanism_debt_pending,
        "mechanism_debt_satisfied": mechanism_debt_satisfied,
        "mechanism_infeasibility_proof_provided": mechanism_infeasibility_proof_provided,
        "decision": decision,
        "analysis_complete_for_axis": complete,
        "continuation_required": not complete,
        "terminal_stop_allowed": False,
        "required_next_action": required_next_action,
        "concrete_blocker": blocker,
        "warnings": warnings,
    }


def reflect_on_replication_evidence(
    replication_evidence: Union[str, List[Dict[str, Any]], Dict[str, Any]],
    current_scope_has_available_inputs: bool = True,
    supported_statuses: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Plan the post-replication residual search.

    A replicated SPECTRA axis is not the end of the investigation when support
    is heterogeneous. The agent should compare supported and non-supported
    tasks, infer what the current axis is actually measuring, and use that
    contrast to generate the next axes to test.
    """
    rows = _normalize_rows(replication_evidence)
    supported_set = {
        _normalize_token(status)
        for status in (supported_statuses or ["monotonic_supported", "localized_supported", "weak_supported"])
    }

    axis_summary: Dict[str, Dict[str, Any]] = {}
    task_summary: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        axis = str(_first_present(row, ["axis_id", "axis_name", "axis"]) or "unknown_axis")
        task = str(_first_present(row, ["task", "task_key", "dataset"]) or "unknown_task")
        status = _normalize_token(str(row.get("status") or row.get("curve_status") or "unknown"))
        supported_value = row.get("supported")
        if isinstance(supported_value, str):
            supported = supported_value.strip().lower() in {"true", "1", "yes", "supported"}
        elif supported_value is None:
            supported = status in supported_set
        else:
            supported = bool(supported_value)

        axis_entry = axis_summary.setdefault(
            axis,
            {
                "supported": 0,
                "not_supported": 0,
                "supported_tasks": set(),
                "not_supported_tasks": set(),
                "statuses": {},
            },
        )
        task_entry = task_summary.setdefault(
            task,
            {
                "supported_axes": set(),
                "not_supported_axes": set(),
                "statuses": {},
            },
        )

        if supported:
            axis_entry["supported"] += 1
            axis_entry["supported_tasks"].add(task)
            task_entry["supported_axes"].add(axis)
        else:
            axis_entry["not_supported"] += 1
            axis_entry["not_supported_tasks"].add(task)
            task_entry["not_supported_axes"].add(axis)
        axis_entry["statuses"][status] = axis_entry["statuses"].get(status, 0) + 1
        task_entry["statuses"][status] = task_entry["statuses"].get(status, 0) + 1

    serial_axis_summary = {}
    for axis, entry in axis_summary.items():
        total = entry["supported"] + entry["not_supported"]
        serial_axis_summary[axis] = {
            "supported": entry["supported"],
            "not_supported": entry["not_supported"],
            "support_fraction": entry["supported"] / total if total else 0.0,
            "supported_tasks": sorted(entry["supported_tasks"]),
            "not_supported_tasks": sorted(entry["not_supported_tasks"]),
            "statuses": entry["statuses"],
            "mixed_support": entry["supported"] > 0 and entry["not_supported"] > 0,
        }

    serial_task_summary = {
        task: {
            "supported_axes": sorted(entry["supported_axes"]),
            "not_supported_axes": sorted(entry["not_supported_axes"]),
            "statuses": entry["statuses"],
        }
        for task, entry in task_summary.items()
    }

    axis_names = set(axis_summary)
    sequence_like_axes = [
        axis
        for axis in axis_names
        if any(token in axis.lower() for token in ["kmer", "sequence", "embedding", "representation"])
    ]
    mixed_axes = [
        axis for axis, entry in serial_axis_summary.items() if entry["mixed_support"]
    ]

    residual_axis_candidates = [
        {
            "axis_id": "residual_model_embedding_after_sequence_similarity",
            "axis_class": "residual representation similarity",
            "rationale": (
                "If raw sequence similarity and model-representation similarity are "
                "partly redundant, regress or match out k-mer/sequence support and "
                "test whether Caduceus embedding support still explains degradation."
            ),
            "suggested_test": "Create matched high/low Caduceus-representation support splits within narrow k-mer-support bins.",
        },
        {
            "axis_id": "motif_family_or_regulatory_word_support",
            "axis_class": "scientific-mechanism similarity",
            "rationale": (
                "Sequence support may matter only for tasks whose labels depend on "
                "specific motif families or regulatory words."
            ),
            "suggested_test": "Compute support over motif families, enriched k-mers, CpG/TATA/TF-like motifs, or task-specific discriminative words.",
        },
        {
            "axis_id": "task_context_conditional_support",
            "axis_class": "metadata/context interaction",
            "rationale": (
                "Mixed replication means the same sequence axis interacts with task "
                "context such as enhancer/promoter/histone/splice task family, species, assay, or sequence length."
            ),
            "suggested_test": "Stratify or model curves by task family, source suite, sequence length, and provenance before selecting the next axis.",
        },
        {
            "axis_id": "class_conditional_spectral_curves",
            "axis_class": "label-structure diagnostic",
            "rationale": (
                "A global curve can hide class-specific degradation, especially when "
                "low-support splits change positive/negative or multiclass composition."
            ),
            "suggested_test": "Generate class-balanced and class-specific curves without using test labels to define prospective splits.",
        },
    ]

    if sequence_like_axes:
        interpretation = (
            "The current evidence suggests a sequence-support failure mode, but "
            "the input-space and model-space axes are related. The next question "
            "is whether degradation is explained by shallow sequence composition, "
            "by residual Caduceus representation geometry, or by task-specific "
            "biology/provenance that co-varies with sequence support."
        )
    else:
        interpretation = (
            "The replicated axis is not obviously sequence-like; compare supported "
            "and non-supported tasks to infer the latent factor it is capturing."
        )

    if mixed_axes and current_scope_has_available_inputs:
        decision = "derive_and_test_residual_axes"
        required_next_action = (
            "Use the supported-vs-non-supported contrast to test at least one "
            "residual/composite axis before treating the replicated candidate as "
            "the final explanation."
        )
        complete = False
    elif mixed_axes:
        decision = "request_scope_for_residual_axes"
        required_next_action = (
            "Report the mixed support pattern and request the missing features, "
            "metadata, model activations, labels, or compute needed for residual-axis tests."
        )
        complete = False
    else:
        decision = "strengthen_or_operationalize_axis"
        required_next_action = (
            "Support is consistent in the current evidence table. Run a stronger "
            "operating point such as larger caps, external data, fresh adapters, "
            "or full fine-tuning before making a broad claim."
        )
        complete = False

    return {
        "decision": decision,
        "analysis_complete": complete,
        "interpretation": interpretation,
        "axis_summary": serial_axis_summary,
        "task_summary": serial_task_summary,
        "mixed_axes": mixed_axes,
        "sequence_like_axes": sequence_like_axes,
        "residual_axis_candidates": residual_axis_candidates,
        "required_next_action": required_next_action,
        "required_outputs": [
            "replication_reflection.json",
            "supported_vs_failed_task_contrast.csv",
            "residual_axis_candidates.json",
            "residual_axis_scores.json",
        ],
    }


def start_spectra_investigator(
    dataset_description: str,
    model_description: str,
    domain: str = "unknown",
    objective: str = "",
    constraints: str = "",
) -> Dict[str, Any]:
    """Start SPECTRA in investigator mode.

    Investigator mode makes the hypothesis state the control object. The agent
    still uses SPECTRA splitters, registries, and reports, but every experiment
    must be justified by the competing hypotheses it can distinguish.
    """
    missing = []
    if not dataset_description.strip():
        missing.append("dataset_description")
    if not model_description.strip():
        missing.append("model_description")

    return {
        "mode": "spectra_investigator",
        "status": "blocked" if missing else "ready",
        "missing_inputs": missing,
        "domain": _normalize_token(domain),
        "objective": objective.strip(),
        "constraints": constraints.strip(),
        "mandate": (
            "Investigate what governs model generalization with the hypothesis "
            "state as the control object. Do not execute a static checklist. "
            "Each experiment must reduce uncertainty about why performance "
            "changes or fails to change under scientific novelty."
        ),
        "inquiry_loop": [
            "Observe: run a bounded behavioral probe and record what changed, what did not, and what was surprising.",
            "Interpret: explain what the result implies about the model, task, data source, and possible leakage or confounding.",
            "Hypothesize: maintain competing explanations, including explanations that predict no degradation on some tasks.",
            "Discriminate: choose the next experiment because it can separate at least two live hypotheses.",
            "Acquire if needed: when the current rows cannot falsify the sharpest live hypothesis, find, download, or construct the dataset that can test it.",
            "Update: revise the hypothesis ledger after each result; kill, weaken, strengthen, or split hypotheses.",
            "Continue: pursue the sharpest unresolved scientific question, not the next registry item.",
        ],
        "experiment_rule": (
            "Do not run another similarity axis merely because it is available. "
            "Run the experiment whose outcome would most change the hypothesis ledger."
        ),
        "required_artifacts": [
            "observations.md",
            "hypothesis_ledger.json",
            "competing_explanations.md",
            "why_this_next_experiment.md",
            "discriminating_experiment_plan.json",
            "belief_update.md",
            "falsifiable_predictions.json",
            "hypothesis_driven_acquisition_plan.json",
            "external_dataset_decision_log.json",
            "spectra_outputs",
        ],
        "forbidden_patterns": [
            "axis checklist with no interpretation",
            "mixed results reported without competing explanations",
            "mechanism debt reported without a falsifiable next hypothesis",
            "continuation launched only because the protocol requires continuation",
            "choosing a registry axis without explaining what uncertainty it resolves",
            "downloading public data without naming the hypothesis it can falsify",
        ],
    }


def update_hypothesis_ledger(
    observations: Union[str, List[Dict[str, Any]], Dict[str, Any]],
    prior_hypotheses: Optional[Union[str, List[Dict[str, Any]], Dict[str, Any]]] = None,
    domain: str = "unknown",
    surprising_findings: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Update a SPECTRA Investigator hypothesis ledger from observations.

    The ledger is intentionally lightweight and serializable. It should be
    rewritten after every experiment so the next run follows the live
    scientific uncertainty rather than a procedural axis order.
    """
    rows = _normalize_rows(observations)
    prior_rows = _normalize_rows(prior_hypotheses or []) if prior_hypotheses else []
    surprises = surprising_findings or []
    normalized_domain = _normalize_token(domain)
    supported_statuses = {"monotonic_supported", "localized_supported", "weak_supported", "supported"}

    axis_summary: Dict[str, Dict[str, Any]] = {}
    task_summary: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        axis = str(_first_present(row, ["axis", "axis_id", "axis_name"]) or "unknown_axis")
        task = str(_first_present(row, ["task", "task_name", "dataset", "source_suite"]) or "unknown_task")
        status = _normalize_token(str(_first_present(row, ["status", "curve_status", "curve_score"]) or "unknown"))
        supported = status in supported_statuses
        axis_entry = axis_summary.setdefault(axis, {"supported": 0, "not_supported": 0, "tasks": {}, "statuses": {}})
        task_entry = task_summary.setdefault(task, {"supported_axes": [], "not_supported_axes": [], "statuses": {}})
        if supported:
            axis_entry["supported"] += 1
            task_entry["supported_axes"].append(axis)
        else:
            axis_entry["not_supported"] += 1
            task_entry["not_supported_axes"].append(axis)
        axis_entry["tasks"][task] = status
        axis_entry["statuses"][status] = axis_entry["statuses"].get(status, 0) + 1
        task_entry["statuses"][status] = task_entry["statuses"].get(status, 0) + 1

    def axis_has_token(tokens: Sequence[str]) -> bool:
        joined = " ".join(axis_summary).lower()
        return any(token in joined for token in tokens)

    live_hypotheses: List[Dict[str, Any]] = []
    if prior_rows:
        for hypothesis in prior_rows:
            if isinstance(hypothesis, dict):
                live_hypotheses.append(dict(hypothesis))

    def add_hypothesis(hypothesis_id: str, statement: str, predicts: str, tests: List[str]) -> None:
        if any(h.get("hypothesis_id") == hypothesis_id for h in live_hypotheses):
            return
        live_hypotheses.append(
            {
                "hypothesis_id": hypothesis_id,
                "statement": statement,
                "current_status": "live",
                "support_level": "unknown",
                "predicts": predicts,
                "supporting_observations": [],
                "contradicting_observations": [],
                "falsification_tests": tests,
            }
        )

    if axis_has_token(["kmer", "6mer", "sequence", "surface"]):
        add_hypothesis(
            "surface_sequence_support",
            "Generalization is partly governed by local sequence or lexical support in the training set.",
            "Performance should fall when train-test k-mer or sequence-neighborhood support is reduced, and the effect should weaken after matching surface support.",
            [
                "Caduceus representation support after matching k-mer, GC, and length",
                "task-family curves under matched 6-mer support",
            ],
        )
    if axis_has_token(["embedding", "representation", "caduceus", "model"]):
        add_hypothesis(
            "model_representation_support",
            "The target model's learned representation has support regions that are not fully explained by input-space similarity.",
            "Representation support should predict degradation even within matched surface-similarity bins.",
            [
                "Residual representation-support curve after matching k-mer/composition",
                "high-vs-low embedding-support biological enrichment",
            ],
        )
    if axis_has_token(["gc", "cpg", "length", "entropy", "composition"]):
        add_hypothesis(
            "composition_regime_support",
            "The model is less reliable in sequence composition regimes that are underrepresented in training.",
            "Composition support should explain residual failure after sequence support matching, or disappear if it is just a k-mer proxy.",
            [
                "GC/CpG/length matched residual curves",
                "composition-stratified representation-support curves",
            ],
        )
    if axis_has_token(["motif", "splice", "grammar", "regulatory"]):
        add_hypothesis(
            "regulatory_grammar_support",
            "Failures may arise when the held-out examples use regulatory words, motif grammar, splice signals, or chromatin-context features not well supported in training.",
            "Curated motif/context support should explain degradation better than crude k-mer or motif-word counts.",
            [
                "curated PWM or motif-family support after matching k-mer and GC",
                "task-family interaction between enhancer/chromatin/promoter/splice tasks",
            ],
        )
    add_hypothesis(
        "task_or_source_specific_failure",
        "Generalization may be task/source dependent rather than governed by one global similarity axis.",
        "The same axis should work on some task families and fail on others; interactions should explain the heterogeneity.",
        [
            "task-family interaction curves",
            "source/provenance controlled splits",
            "class-conditional curves",
        ],
    )

    for hypothesis in live_hypotheses:
        statement = str(hypothesis.get("statement", "")).lower()
        supporting = []
        contradicting = []
        for axis, summary in axis_summary.items():
            axis_text = axis.lower()
            support_fraction = summary["supported"] / max(1, summary["supported"] + summary["not_supported"])
            related = (
                ("sequence" in statement and any(token in axis_text for token in ["kmer", "6mer", "sequence"]))
                or ("representation" in statement and any(token in axis_text for token in ["embedding", "representation", "caduceus"]))
                or ("composition" in statement and any(token in axis_text for token in ["gc", "cpg", "length", "entropy", "composition"]))
                or ("motif" in statement and any(token in axis_text for token in ["motif", "splice", "grammar"]))
                or "task/source" in statement
                or "task source" in statement
            )
            if not related:
                continue
            observation = {
                "axis": axis,
                "support_fraction": support_fraction,
                "statuses": summary["statuses"],
            }
            if support_fraction > 0.5:
                supporting.append(observation)
            elif support_fraction == 0:
                contradicting.append(observation)
            else:
                supporting.append({**observation, "note": "mixed support"})
        hypothesis["supporting_observations"] = supporting
        hypothesis["contradicting_observations"] = contradicting
        if supporting and contradicting:
            hypothesis["support_level"] = "mixed"
        elif supporting:
            hypothesis["support_level"] = "partial"
        elif contradicting:
            hypothesis["support_level"] = "weak_or_contradicted"
        else:
            hypothesis["support_level"] = "untested"

    mixed_axes = [
        axis
        for axis, summary in axis_summary.items()
        if summary["supported"] > 0 and summary["not_supported"] > 0
    ]
    next_questions = [
        "Which live hypothesis would most change if the next experiment succeeds or fails?",
        "Which supported axis is a proxy for another mechanism, and how can that proxy be separated from the mechanism?",
        "Which task family contradicts the current explanation, and what interaction would explain that contradiction?",
    ]
    if "regulatory" in normalized_domain or "dna" in normalized_domain or "genomic" in normalized_domain:
        next_questions.append(
            "Do motif family, motif grammar, chromatin/task context, or class-conditional sequence patterns explain residual degradation after k-mer and composition matching?"
        )

    return {
        "mode": "spectra_investigator_hypothesis_update",
        "analysis_complete": False,
        "axis_summary": axis_summary,
        "task_summary": task_summary,
        "mixed_axes": mixed_axes,
        "surprising_findings": surprises,
        "hypotheses": live_hypotheses,
        "next_questions": next_questions,
        "required_next_action": (
            "Choose a discriminating experiment that can distinguish or separate at least two "
            "live hypotheses. Do not proceed by testing the next available axis "
            "unless it is the most informative falsification test."
        ),
        "required_outputs": [
            "observations.md",
            "hypothesis_ledger.json",
            "competing_explanations.md",
            "belief_update.md",
            "falsifiable_predictions.json",
        ],
    }


def choose_discriminating_experiment(
    hypothesis_ledger: Union[str, List[Dict[str, Any]], Dict[str, Any]],
    candidate_experiments: Optional[Union[str, List[Dict[str, Any]], Dict[str, Any]]] = None,
    domain: str = "unknown",
    constraints: str = "",
) -> Dict[str, Any]:
    """Choose the next experiment by discriminative value, not checklist order."""
    ledger_payload = _coerce_json_payload(hypothesis_ledger) if isinstance(hypothesis_ledger, str) else hypothesis_ledger
    if isinstance(ledger_payload, dict) and "hypotheses" in ledger_payload:
        hypotheses = _normalize_rows(ledger_payload["hypotheses"])
        mixed_axes = ledger_payload.get("mixed_axes", [])
    else:
        hypotheses = _normalize_rows(ledger_payload)
        mixed_axes = []

    if candidate_experiments:
        candidates = _normalize_rows(candidate_experiments)
    else:
        candidates = [
            {
                "experiment_id": "residual_representation_after_surface_match",
                "question": "Does Caduceus representation support explain degradation after matching k-mer, GC/CpG, and length?",
                "tests_hypotheses": ["surface_sequence_support", "model_representation_support", "composition_regime_support"],
                "falsifiable_prediction": "If representation support is only a surface proxy, its effect should vanish after matching surface/composition support.",
                "controls": ["k-mer support", "GC/CpG", "length", "task family"],
                "execution": "Construct matched bins, train fresh Caduceus probes per split, compare residual curves.",
            },
            {
                "experiment_id": "task_family_interaction_spectra",
                "question": "Do supported and failed tasks separate by enhancer, promoter, chromatin, splice, species, or source context?",
                "tests_hypotheses": ["task_or_source_specific_failure", "surface_sequence_support", "regulatory_grammar_support"],
                "falsifiable_prediction": "If the failure is context-dependent, the same support axis should degrade one task family while staying flat in another.",
                "controls": ["sample cap", "class balance", "sequence length"],
                "execution": "Fit task-family interaction curves and rerun the strongest axis within matched task-family strata.",
            },
            {
                "experiment_id": "curated_motif_context_support",
                "question": "Do motif families or regulatory grammar explain degradation beyond crude k-mer and motif-word support?",
                "tests_hypotheses": ["regulatory_grammar_support", "surface_sequence_support", "composition_regime_support"],
                "falsifiable_prediction": "Curated motif/context support should remain predictive after matching k-mer and composition if biology, not surface support, drives failure.",
                "controls": ["k-mer support", "GC/CpG", "length", "class balance"],
                "execution": "Map or derive motif/PWM/context features, construct support splits, and train fresh probes per split.",
            },
            {
                "experiment_id": "class_conditional_residual_curves",
                "question": "Is degradation concentrated in positives, negatives, or specific labels?",
                "tests_hypotheses": ["task_or_source_specific_failure", "regulatory_grammar_support"],
                "falsifiable_prediction": "If the model learns active regulatory sequence differently from background, residual curves should differ by class.",
                "controls": ["prospective split construction", "class balance"],
                "execution": "Build class-balanced splits without test-label leakage and report per-class performance-overlap curves.",
            },
        ]

    live_ids = {str(h.get("hypothesis_id", "")) for h in hypotheses}
    scored_candidates = []
    for candidate in candidates:
        tests = candidate.get("tests_hypotheses") or candidate.get("hypotheses_tested") or []
        if isinstance(tests, str):
            tests = [item.strip() for item in tests.split(",") if item.strip()]
        tested_live = [item for item in tests if item in live_ids]
        falsifiable = bool(str(candidate.get("falsifiable_prediction", "")).strip())
        controls = candidate.get("controls") or []
        if isinstance(controls, str):
            controls = [item.strip() for item in controls.split(",") if item.strip()]
        question = str(candidate.get("question") or candidate.get("rationale") or "")
        experiment_id = str(candidate.get("experiment_id") or candidate.get("name") or "candidate_experiment")
        mixed_bonus = 1 if any(str(axis).lower() in question.lower() for axis in mixed_axes) else 0
        mechanism_bonus = 1 if any(token in experiment_id.lower() for token in ["residual", "motif", "context", "class"]) else 0
        score = 3 * len(tested_live) + 2 * int(falsifiable) + len(controls) + mixed_bonus + mechanism_bonus
        scored_candidates.append(
            {
                "score": score,
                "candidate": candidate | {
                    "experiment_id": experiment_id,
                    "tests_live_hypotheses": tested_live,
                    "falsifiable": falsifiable,
                    "controls": controls,
                },
            }
        )

    scored_candidates.sort(key=lambda item: item["score"], reverse=True)
    selected = scored_candidates[0]["candidate"] if scored_candidates else {}
    rejected = [item["candidate"] for item in scored_candidates[1:]]

    return {
        "decision": "run_discriminating_experiment",
        "analysis_complete": False,
        "selected_experiment": selected,
        "rejected_experiments": rejected,
        "selection_rationale": (
            "Selected the experiment with the highest discriminative value: it "
            "tests multiple live hypotheses, has a falsifiable prediction, and "
            "includes controls that separate proxy explanations from mechanism."
        ),
        "why_not_checklist": (
            "The next experiment is not chosen because it is the next registry "
            "axis. It is chosen because different outcomes would update the "
            "hypothesis ledger in different directions."
        ),
        "constraints": constraints.strip(),
        "required_outputs": [
            "why_this_next_experiment.md",
            "discriminating_experiment_plan.json",
            "falsifiable_predictions.json",
            "belief_update.md",
        ],
    }


def plan_hypothesis_driven_dataset_acquisition(
    hypothesis_ledger: Union[str, List[Dict[str, Any]], Dict[str, Any]],
    current_data_limitations: Union[str, List[str]],
    domain: str = "unknown",
    allowed_network: bool = True,
    allow_large_downloads: bool = False,
    constraints: str = "",
) -> Dict[str, Any]:
    """Plan external data acquisition as an Investigator action.

    This is not a generic "download more data" step. It is used only when the
    current data cannot falsify or distinguish the sharpest live hypothesis.
    The output tells the agent what dataset or public resource would make the
    hypothesis testable, what controls are required, and what bounded slice can
    be launched first.
    """
    ledger_payload = _coerce_json_payload(hypothesis_ledger) if isinstance(hypothesis_ledger, str) else hypothesis_ledger
    if isinstance(ledger_payload, dict) and "hypotheses" in ledger_payload:
        hypotheses = _normalize_rows(ledger_payload["hypotheses"])
    else:
        hypotheses = _normalize_rows(ledger_payload)

    if isinstance(current_data_limitations, str):
        limitations = [item.strip() for item in current_data_limitations.split(";") if item.strip()]
        if not limitations:
            limitations = [current_data_limitations.strip()] if current_data_limitations.strip() else []
    else:
        limitations = list(current_data_limitations)

    normalized_domain = _normalize_token(domain)
    limitation_text = _normalize_token(" ".join(limitations))
    constraints_text = constraints.strip()
    live = [
        hypothesis
        for hypothesis in hypotheses
        if _normalize_token(str(hypothesis.get("current_status") or hypothesis.get("status") or "live"))
        not in {"dead", "falsified", "closed"}
    ]

    def hypothesis_text(hypothesis: Dict[str, Any]) -> str:
        return _normalize_token(
            " ".join(
                [
                    str(hypothesis.get("hypothesis_id") or hypothesis.get("id") or ""),
                    str(hypothesis.get("statement") or hypothesis.get("claim") or ""),
                    str(hypothesis.get("predicts") or hypothesis.get("failure_mode_predicted") or ""),
                ]
            )
        )

    needs_coordinates = any(token in limitation_text for token in ["coordinate", "stable_id", "accession", "provenance", "chrom", "start", "end"])
    needs_annotations = any(token in limitation_text for token in ["motif", "jaspar", "encode", "chromatin", "conservation", "context", "annotation", "pwm"])
    needs_counterfactuals = any(token in limitation_text for token in ["counterfactual", "variant", "mutation", "effect", "causal"])
    needs_labels = any(token in limitation_text for token in ["label", "phenotype", "activity", "binding"])

    hypothesis_specs = []
    for hypothesis in live:
        text = hypothesis_text(hypothesis)
        if any(token in text for token in ["motif", "regulatory", "grammar", "chromatin", "context", "variant"]):
            acquisition_goal = "mechanism_regulatory_dataset"
            falsification_test = (
                "After matching k-mer, GC/CpG, length, and class balance, motif-family, "
                "motif-disruption, or regulatory-context novelty should still degrade target-model probes."
            )
            bounded_slice = (
                "Construct or download a small coordinate-backed regulatory sequence dataset "
                "with motif/context annotations, then run fresh Caduceus probes across matched SPECTRA splits."
            )
        elif any(token in text for token in ["representation", "embedding", "model_space", "model"]):
            acquisition_goal = "representation_translation_dataset"
            falsification_test = (
                "If model-space support reflects biology, low-support embedding neighborhoods "
                "should be enriched for interpretable motif/context/variant features and those features "
                "should reproduce degradation under matched controls."
            )
            bounded_slice = (
                "Acquire annotations for high- and low-support embedding neighborhoods or construct "
                "a coordinate-backed sequence panel that maps embedding regions to motif/context features."
            )
        elif any(token in text for token in ["task", "source", "provenance", "family", "site"]):
            acquisition_goal = "task_context_replication_dataset"
            falsification_test = (
                "If failure is task/source dependent, the same controlled axis should degrade "
                "one context family and stay flat in a matched alternative context."
            )
            bounded_slice = (
                "Acquire a second task family or source-context dataset with compatible labels and run matched SPECTRA curves."
            )
        else:
            acquisition_goal = "expanded_behavioral_replication_dataset"
            falsification_test = (
                "The acquired dataset should make a prediction that differs between at least two live hypotheses."
            )
            bounded_slice = "Acquire or construct a fit-for-purpose labeled public/local dataset that tests the unresolved prediction under fresh split-based probes."

        hypothesis_specs.append(
            {
                "hypothesis_id": str(hypothesis.get("hypothesis_id") or hypothesis.get("id") or "unknown_hypothesis"),
                "acquisition_goal": acquisition_goal,
                "why_current_data_is_insufficient": limitations,
                "falsification_test": falsification_test,
                "bounded_first_slice": bounded_slice,
            }
        )

    candidate_resources: List[Dict[str, Any]]
    if any(token in normalized_domain for token in ["dna", "rna", "genomic", "genomics", "regulatory", "caduceus", "nucleotide"]):
        candidate_resources = [
            {
                "resource_or_dataset": "coordinate-backed regulatory sequence dataset",
                "examples": ["ENCODE cCRE-linked sequences", "GenomicBenchmarks source archives", "Nucleotide Transformer source datasets"],
                "tests": ["regulatory_context_support", "task_family_interaction"],
                "required_fields": ["sequence", "label", "coordinate_or_source_id", "task_or_context"],
            },
            {
                "resource_or_dataset": "curated motif/PWM annotation resource",
                "examples": ["JASPAR CORE", "HOCOMOCO"],
                "tests": ["motif_family_support", "motif_grammar_support"],
                "required_fields": ["motif_family", "motif_score", "position", "strand"],
            },
            {
                "resource_or_dataset": "counterfactual or variant-effect regulatory data",
                "examples": ["MPRA/eQTL/variant-effect sequence windows", "DART-like counterfactual variant tasks"],
                "tests": ["motif_disruption_or_variant_effect_support"],
                "required_fields": ["reference_sequence", "alternate_sequence_or_variant", "effect_label", "coordinate_or_variant_id"],
            },
            {
                "resource_or_dataset": "chromatin/conservation/repeat context annotations",
                "examples": ["ENCODE cCRE/chromatin tracks", "UCSC conservation/repeats/CpG islands"],
                "tests": ["context_conditioned_curves", "confounder_controls"],
                "required_fields": ["coordinate", "annotation_value", "assembly"],
            },
        ]
    else:
        candidate_resources = [
            {
                "resource_or_dataset": "domain-labeled mechanism dataset",
                "examples": ["public benchmark source data", "curated ontology/resource tables"],
                "tests": ["mechanism_support", "context_conditioned_curves"],
                "required_fields": ["unit_id", "features_or_sequence", "label", "mechanism_annotation"],
            }
        ]

    if not allowed_network:
        decision = "construct_from_local_resources_or_request_network"
        required_next_action = (
            "Network acquisition is disabled. Search local repositories, cached archives, "
            "source loaders, and bundled metadata for a bounded mechanism-test dataset; "
            "if unavailable, record the exact public resource needed and request network access."
        )
    elif needs_coordinates or needs_annotations or needs_counterfactuals or needs_labels:
        decision = "acquire_or_construct_hypothesis_test_dataset"
        required_next_action = (
            "Acquire, map, or construct a fit-for-purpose public/local dataset that can falsify the sharpest "
            "live hypothesis. Start with a bounded slice that can run now, then update the "
            "hypothesis ledger from the result."
        )
    else:
        decision = "consider_external_data_if_discriminative"
        required_next_action = (
            "External data is optional until the ledger identifies a live hypothesis that "
            "cannot be distinguished with current rows. Do not download data speculatively."
        )

    if allow_large_downloads:
        size_policy = "Large downloads allowed if they directly test a live hypothesis and provenance/license are recorded."
    else:
        size_policy = "Prefer bounded slices, source manifests, lightweight annotations, and small curated panels before large archives."

    return {
        "decision": decision,
        "analysis_complete": False,
        "mode": "hypothesis_driven_dataset_acquisition",
        "domain": normalized_domain,
        "live_hypothesis_count": len(live),
        "hypothesis_acquisition_specs": hypothesis_specs,
        "candidate_resources": candidate_resources,
        "current_data_limitations": limitations,
        "acquisition_triggers": {
            "needs_coordinates_or_ids": needs_coordinates,
            "needs_curated_annotations": needs_annotations,
            "needs_counterfactual_or_variant_effect_data": needs_counterfactuals,
            "needs_labels_or_activity_readout": needs_labels,
        },
        "size_policy": size_policy,
        "constraints": constraints_text,
        "required_next_action": required_next_action,
        "forbidden_patterns": [
            "download a dataset without naming the hypothesis it can falsify",
            "treat external data acquisition as a checklist item",
            "claim a DART-level mechanism from crude k-mers, GC, or embeddings alone",
            "use a constructed dataset as a replacement for the original benchmark without scoping it as mechanism-test evidence",
        ],
        "required_outputs": [
            "hypothesis_driven_acquisition_plan.json",
            "external_dataset_decision_log.json",
            "resource_search_log.json",
            "download_or_construction_manifest.json",
            "mapping_validation.json",
            "leakage_and_confounder_controls.json",
            "mechanism_test_spectra_results.csv",
            "belief_update_after_external_data.md",
        ],
    }


def distill_spectra_hypotheses(
    curve_scores: Union[str, List[Dict[str, Any]], Dict[str, Any]],
    hypothesis_ledger: Optional[Union[str, List[Dict[str, Any]], Dict[str, Any]]] = None,
    belief_update: str = "",
    domain: str = "unknown",
    focus: str = "",
) -> Dict[str, Any]:
    """Distill Investigator artifacts into sharper scientific hypotheses.

    The Distiller is intentionally read-only. It should not run experiments by
    default. It looks for asymmetries, narrow supports, contradictions, and
    surprising task-family patterns, then hands a ranked hypothesis back to the
    Investigator.
    """
    rows = _normalize_rows(curve_scores)
    normalized_domain = _normalize_token(domain)
    focus_text = focus.strip()
    supported_statuses = {"monotonic", "monotonic_supported", "localized", "localized_supported", "weak", "weak_supported", "supported"}

    axis_summary: Dict[str, Dict[str, Any]] = {}
    task_summary: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        axis = str(_first_present(row, ["axis", "axis_id", "axis_name"]) or "unknown_axis")
        task = str(_first_present(row, ["task_name", "task", "task_labeling", "task_label", "dataset"]) or "unknown_task")
        source = str(_first_present(row, ["source_suite", "source", "benchmark"]) or "")
        status = _normalize_token(str(_first_present(row, ["curve_score", "status", "curve_status"]) or "unknown"))
        supported = status in supported_statuses
        axis_entry = axis_summary.setdefault(
            axis,
            {
                "supported_tasks": [],
                "non_supported_tasks": [],
                "statuses": {},
                "sources": {},
            },
        )
        task_entry = task_summary.setdefault(
            task,
            {
                "supported_axes": [],
                "non_supported_axes": [],
                "statuses": {},
                "source_suite": source,
            },
        )
        if supported:
            axis_entry["supported_tasks"].append(task)
            task_entry["supported_axes"].append(axis)
        else:
            axis_entry["non_supported_tasks"].append(task)
            task_entry["non_supported_axes"].append(axis)
        axis_entry["statuses"][status] = axis_entry["statuses"].get(status, 0) + 1
        axis_entry["sources"][source] = axis_entry["sources"].get(source, 0) + 1
        task_entry["statuses"][status] = task_entry["statuses"].get(status, 0) + 1

    for summary in axis_summary.values():
        summary["supported_tasks"] = sorted(set(summary["supported_tasks"]))
        summary["non_supported_tasks"] = sorted(set(summary["non_supported_tasks"]))
        total = len(summary["supported_tasks"]) + len(summary["non_supported_tasks"])
        summary["support_fraction"] = len(summary["supported_tasks"]) / total if total else 0.0

    ledger_rows = []
    if hypothesis_ledger is not None:
        parsed = _coerce_json_payload(hypothesis_ledger) if isinstance(hypothesis_ledger, str) else hypothesis_ledger
        if isinstance(parsed, dict) and "live_hypotheses" in parsed:
            ledger_rows = _normalize_rows(parsed["live_hypotheses"])
        elif isinstance(parsed, dict) and "hypotheses" in parsed:
            ledger_rows = _normalize_rows(parsed["hypotheses"])
        else:
            ledger_rows = _normalize_rows(parsed)

    def task_family(task_name: str) -> str:
        lower = task_name.lower()
        if "enhancer" in lower:
            return "enhancer"
        if "promoter" in lower or "tata" in lower:
            return "promoter"
        if lower.startswith("h3") or lower.startswith("h4") or "h3" in lower or "h4" in lower:
            return "chromatin_histone"
        if "splice" in lower:
            return "splice"
        if "demo" in lower:
            return "demo"
        return "other_regulatory"

    family_axis: Dict[str, Dict[str, Dict[str, int]]] = {}
    for axis, summary in axis_summary.items():
        for task in summary["supported_tasks"]:
            fam = task_family(task)
            fam_entry = family_axis.setdefault(axis, {}).setdefault(fam, {"supported": 0, "not_supported": 0})
            fam_entry["supported"] += 1
        for task in summary["non_supported_tasks"]:
            fam = task_family(task)
            fam_entry = family_axis.setdefault(axis, {}).setdefault(fam, {"supported": 0, "not_supported": 0})
            fam_entry["not_supported"] += 1

    distilled: List[Dict[str, Any]] = []
    ledger_text = _normalize_token(json.dumps(ledger_rows, sort_keys=True))
    belief_text = _normalize_token(belief_update)
    label_artifact_signal = any(
        token in ledger_text or token in belief_text
        for token in [
            "source_or_label_construction_artifact",
            "label_coded",
            "label_suffix",
            "same_sequence_set",
            "source_name",
            "deduplication",
            "duplicate_sequences",
        ]
    )
    if label_artifact_signal:
        distilled.append(
            {
                "hypothesis_id": "label_construction_before_biology",
                "rank": 1,
                "distilled_hypothesis": (
                    "The strongest current explanation is that the enhancer subtype finding is "
                    "a label-construction and provenance problem before it is a biological "
                    "regulatory-context mechanism."
                ),
                "why_interesting": (
                    "The matched Investigator run found that enhancers and enhancers_types use the "
                    "same sequence set, source_name is label-coded, and the motif-family axis also "
                    "affects the matched binary enhancer task. This converts the Distiller's prior "
                    "biology hypothesis into a sharper artifact-versus-mechanism question."
                ),
                "supporting_evidence": [
                    "hypothesis ledger or belief update strengthened source_or_label_construction_artifact",
                    "matched binary and subtype tasks share the same sequence universe",
                    "motif-family support is not subtype-exclusive in the matched contrast",
                ],
                "alternative_explanations": [
                    "subtype labels may still correspond to real biological enhancer classes hidden by the processed bundle",
                    "motif-family support may be a real mechanism that also affects binary enhancer/background recognition",
                    "logistic probe convergence and small matched splits may blur subtype-specific effects",
                ],
                "falsifiable_predictions": [
                    "Recovering original subtype definitions or coordinates should reveal whether labels 1 and 2 map to distinct enhancer biology.",
                    "If the signal is a label-construction artifact, source/provenance-matched or label-balanced splits should shrink the subtype-specific motif effect.",
                    "If the signal is biological, named subtype or coordinate-backed enhancer annotations should preserve motif-family degradation after source and class controls.",
                ],
                "next_experiments": [
                    "Recover the upstream enhancers_types label definitions, source files, dataset card, or loader logic and map labels 1 and 2 to biological subtype names if possible.",
                    "Run source/provenance-controlled splits if recoverable provenance exists; otherwise explicitly scope the result as label-construction-specific.",
                    "Construct or acquire a coordinate-backed enhancer subtype dataset with named classes, then rerun the motif-family and residual SPECTRA tests.",
                ],
            }
        )

    motif_axes = [
        axis for axis in axis_summary
        if any(token in axis.lower() for token in ["motif", "jaspar", "pwm", "tf"])
    ]
    enhancer_type_motif_signal = False
    for axis in motif_axes:
        supported_tasks = axis_summary[axis]["supported_tasks"]
        if any("enhancers_types" in task for task in supported_tasks):
            enhancer_type_motif_signal = True
            distilled.append(
                {
                    "hypothesis_id": "enhancer_subtype_regulatory_grammar",
                    "rank": 2 if label_artifact_signal else 1,
                    "distilled_hypothesis": (
                        "Caduceus may struggle when enhancer tasks require subtype or "
                        "regulatory-context discrimination, rather than binary enhancer/background recognition."
                    ),
                    "why_interesting": (
                        "The curated JASPAR motif-support axis was supported on enhancers_types "
                        "but not broadly across promoter or binary enhancer tasks. That suggests "
                        "the interesting mechanism is not global motif grammar; it may be subtype-specific regulatory context."
                    ),
                    "supporting_evidence": [
                        f"{axis} supported tasks: {', '.join(supported_tasks)}",
                        f"{axis} non-supported tasks: {', '.join(axis_summary[axis]['non_supported_tasks'])}",
                    ],
                    "alternative_explanations": [
                        "enhancers_types may have label/source artifacts or easier class-specific motif enrichments",
                        "the bounded motif panel may miss relevant TF families for other tasks",
                        "motif support may proxy sequence composition or class imbalance without coordinate/context controls",
                    ],
                    "falsifiable_predictions": [
                        "Class-conditional curves in enhancers_types should show degradation concentrated in specific enhancer subtype labels.",
                        "The same motif-family support should be weaker or absent on binary enhancers after matching k-mer, GC, length, and class balance.",
                        "Low-support enhancers_types examples should be enriched for distinct TF motif families or motif combinations.",
                        "Residual Caduceus embedding support should separate enhancer subtype labels after matching k-mer and motif-family support.",
                    ],
                    "next_experiments": [
                        "Run class-conditional SPECTRA on enhancers_types.",
                        "Run matched contrast: enhancers vs enhancers_types under k-mer, GC, length, and JASPAR motif-family support.",
                        "Annotate low-support enhancers_types rows by TF family and test motif-family residual curves.",
                        "Check source_name/provenance and label construction for enhancers_types before claiming biology.",
                    ],
                }
            )

    kmer_axis = next(
        (
            axis for axis in axis_summary
            if any(token in axis.lower() for token in ["kmer", "sequence", "surface"])
        ),
        None,
    )
    kmer_summary = axis_summary.get(kmer_axis) if kmer_axis else None
    if kmer_summary and kmer_summary["support_fraction"] > 0.5:
        distilled.append(
            {
                "hypothesis_id": "broad_surface_support_not_mechanism",
                "rank": 3 if label_artifact_signal and enhancer_type_motif_signal else 2 if label_artifact_signal or enhancer_type_motif_signal else 1,
                "distilled_hypothesis": (
                    "Local sequence-neighborhood support is a broad behavioral vulnerability, "
                    "but it is probably not the scientific mechanism."
                ),
                "why_interesting": (
                    "K-mer support was supported or localized on most tasks, while motif and composition "
                    "axes were narrower or mixed. This makes k-mer support a useful stress axis but not a DART-level explanation."
                ),
                "supporting_evidence": [
                    "%s support_fraction=%.3f" % (kmer_axis, kmer_summary["support_fraction"]),
                    "supported tasks: %s" % ", ".join(kmer_summary["supported_tasks"][:12]),
                ],
                "alternative_explanations": [
                    "k-mer support may proxy dataset duplication, source provenance, sequence length, or label-family artifacts",
                    "localized curves can reflect one difficult split region rather than a monotonic mechanism",
                ],
                "falsifiable_predictions": [
                    "If k-mer support is only a proxy, residual biological/context axes should explain degradation after k-mer matching.",
                    "If k-mer support is the true operating axis, motif/context residual curves should flatten after k-mer matching.",
                ],
                "next_experiments": [
                    "Use k-mer matched residual curves for enhancer subtype, chromatin, and promoter families.",
                    "Compare source/provenance-matched and source/provenance-unmatched k-mer curves.",
                ],
            }
        )

    composition_axis = next(
        (
            axis for axis in axis_summary
            if any(token in axis.lower() for token in ["composition", "gc", "cpg", "length", "entropy"])
        ),
        None,
    )
    composition_summary = axis_summary.get(composition_axis) if composition_axis else None
    if composition_summary and 0 < composition_summary["support_fraction"] < 0.5:
        distilled.append(
            {
                "hypothesis_id": "composition_context_task_specificity",
                "rank": 4 if label_artifact_signal else 3,
                "distilled_hypothesis": (
                    "Composition/context support is task-specific and may identify regulatory contexts "
                    "where Caduceus relies on broad sequence regimes rather than transferable mechanism."
                ),
                "why_interesting": (
                    "Composition support was monotonic/localized on a subset of tasks and non-explanatory on many others, "
                    "which is a clue for task-family interactions rather than a global axis."
                ),
                "supporting_evidence": [
                    "%s support_fraction=%.3f" % (composition_axis, composition_summary["support_fraction"]),
                ],
                "alternative_explanations": [
                    "composition features may be too crude to represent regulatory context",
                    "composition support may select easier or harder class mixtures",
                ],
                "falsifiable_predictions": [
                    "Composition support should vanish after class/source matching if it is a dataset artifact.",
                    "Composition support should persist in chromatin or enhancer subfamilies if it reflects regulatory context.",
                ],
                "next_experiments": [
                    "Run class-balanced composition curves within task families.",
                    "Run residual motif/context curves after matching GC/CpG/length.",
                ],
            }
        )

    if any("coordinate" in str(row).lower() or "provenance" in str(row).lower() for row in ledger_rows) or "coordinate" in belief_update.lower() or "provenance" in belief_update.lower():
        distilled.append(
            {
                "hypothesis_id": "coordinate_context_missing_mechanism",
                "rank": 5 if label_artifact_signal else 4,
                "distilled_hypothesis": (
                    "The current visible sequence tasks may be insufficient for DART-level insight because "
                    "coordinate, tissue, TSS, and provenance context remain unresolved."
                ),
                "why_interesting": (
                    "The motif test weakened a simple global motif-grammar story. The next mechanism may require "
                    "coordinate-backed variant or regulatory context rather than sequence-only rows."
                ),
                "supporting_evidence": [
                    "belief update or ledger keeps coordinate/provenance context live",
                ],
                "alternative_explanations": [
                    "a better sequence-only motif grammar feature may still explain the pattern",
                    "visible tasks may be too noisy or label-constructed for causal regulatory insight",
                ],
                "falsifiable_predictions": [
                    "Coordinate-backed variant/eQTL windows should show stronger context-dependent degradation than anonymous sequence rows.",
                    "Tissue/TSS/chromosome or cCRE context should explain residual failures beyond k-mer and motif support.",
                ],
                "next_experiments": [
                    "Recover a bounded hg38/eQTL sequence panel and test tissue/TSS/chromosome-conditioned SPECTRA.",
                    "Map enhancer subtype rows to source/provenance if possible before broadening the mechanism claim.",
                ],
            }
        )

    distilled.sort(key=lambda item: item.get("rank", 999))
    if not distilled:
        distilled.append(
            {
                "hypothesis_id": "insufficient_asymmetry_detected",
                "rank": 1,
                "distilled_hypothesis": "No sharp scientific asymmetry was detected from the provided curve scores.",
                "why_interesting": "The Investigator may need richer task labels, stronger controls, or broader artifacts for distillation.",
                "supporting_evidence": [],
                "alternative_explanations": ["input artifacts are incomplete or too aggregated"],
                "falsifiable_predictions": [],
                "next_experiments": ["Provide per-class curves, source metadata, and full split-level metrics."],
            }
        )

    handoff = distilled[0]
    serial_task_summary = {
        task: {
            "supported_axes": sorted(set(summary["supported_axes"])),
            "non_supported_axes": sorted(set(summary["non_supported_axes"])),
            "statuses": summary["statuses"],
            "source_suite": summary["source_suite"],
        }
        for task, summary in task_summary.items()
    }
    what_not_to_do = [
        "Do not report another generic axis sweep as the next step unless it distinguishes the top distilled hypotheses.",
        "Do not claim a global motif mechanism when motif support is narrow or task-specific.",
        "Do not claim enhancer subtype biology when source names or subtype labels may be label-construction artifacts.",
        "Do not treat k-mer, composition, source labels, or generic embedding distance as mechanism-level explanations without controls.",
        "Do not hand the Investigator a terminal report; hand it a falsifiable next experiment and the belief update each outcome would imply.",
    ]
    return {
        "mode": "spectra_distiller",
        "analysis_complete": False,
        "domain": normalized_domain,
        "focus": focus_text,
        "distiller_role": (
            "Read Investigator artifacts, find the emerging scientific story, "
            "and hand a sharper hypothesis back to the Investigator. Do not run experiments by default."
        ),
        "axis_summary": axis_summary,
        "task_signal_summary": serial_task_summary,
        "family_axis_summary": family_axis,
        "distilled_hypotheses": distilled,
        "recommended_handoff_to_investigator": {
            "hypothesis_id": handoff["hypothesis_id"],
            "question": handoff["distilled_hypothesis"],
            "why_this_is_next": handoff["why_interesting"],
            "predictions_to_test": handoff["falsifiable_predictions"],
            "experiments_to_run": handoff["next_experiments"],
        },
        "what_not_to_do": what_not_to_do,
        "required_outputs": [
            "distilled_hypotheses.json",
            "distiller_report.md",
            "investigator_handoff.json",
        ],
    }


def synthesize_spectra_generalizability_finding(
    target_model: str,
    model_paper_context: Union[str, Dict[str, Any]],
    spectra_findings: Union[str, Dict[str, Any], List[Dict[str, Any]]],
    investigator_trace: Optional[Union[str, Dict[str, Any], List[Dict[str, Any]]]] = None,
    vanilla_agent_summary: Optional[Union[str, Dict[str, Any], List[Dict[str, Any]]]] = None,
    domain: str = "unknown",
    intended_evidence_level: str = "paper-ready bounded SPECTRA finding",
) -> Dict[str, Any]:
    """Convert the final SPECTRA evidence set into a paper-ready finding.

    This is the Distiller's last-step synthesis primitive. It does not launch
    new experiments. It contextualizes the model paper against the accumulated
    SPECTRA evidence and returns a bounded claim, interpretation, limitations,
    and manuscript-ready paragraphs.
    """
    normalized_domain = _normalize_token(domain)

    def try_parse(value: Any) -> Any:
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith("{") or stripped.startswith("["):
                try:
                    return json.loads(stripped)
                except json.JSONDecodeError:
                    return value
        return value

    parsed_findings = try_parse(spectra_findings)
    paper_text = _payload_to_text(model_paper_context)
    findings_text = _payload_to_text(parsed_findings)
    trace_text = _payload_to_text(try_parse(investigator_trace))
    vanilla_text = _payload_to_text(try_parse(vanilla_agent_summary))

    hypothesis_assessment: Dict[str, Any] = {}
    whole_loop_summary: Dict[str, Any] = {}
    source_inputs: Dict[str, Any] = {}
    paper_results_log = ""
    if isinstance(parsed_findings, dict):
        hypothesis_assessment = parsed_findings.get("hypothesis_assessment", {})
        whole_loop_summary = parsed_findings.get("whole_loop_summary", {})
        source_inputs = parsed_findings.get("source_inputs", {})
        paper_results_log = str(
            parsed_findings.get("paper_results_log_paragraph")
            or hypothesis_assessment.get("paper_results_log_paragraph")
            or ""
        )

    composition_challenge = whole_loop_summary.get("composition_control_challenge", {})
    matched_probe_result = whole_loop_summary.get("matched_caduceus_probe_result", {})
    matched_regulatory = matched_probe_result.get("matched_regulatory_context_support_split", {})
    matched_hash = matched_probe_result.get("matched_same_pool_hash_control_split", {})
    composition_gap = _safe_float(composition_challenge.get("composition_low_high_absolute_roc_auc_gap"))
    matched_delta = _safe_float(matched_regulatory.get("validation_middle_to_test_low_roc_auc_delta"))
    hash_delta = _safe_float(matched_hash.get("validation_to_test_roc_auc_delta"))
    if composition_gap is not None and composition_gap >= 0.05:
        comparator = max(
            abs(value)
            for value in [matched_delta or 0.0, hash_delta or 0.0, 1e-6]
        )
        if composition_gap >= 2.0 * comparator:
            return {
                "mode": "spectra_final_finding_synthesis",
                "analysis_complete": False,
                "route": "return_to_investigator_new_primary_axis",
                "domain": normalized_domain,
                "target_model": target_model,
                "terminal_report_allowed": False,
                "why_not_terminal": (
                    "The original hypothesis was attenuated, but the evidence contains "
                    "a larger unresolved alternative degradation axis. A Distiller should "
                    "not stop on a small negative/attenuation result while a stronger "
                    "scientifically plausible axis remains untested as the primary hypothesis."
                ),
                "dominant_unresolved_axis": {
                    "axis_id": "gc_cpg_length_composition_regime",
                    "axis_description": "GC/CpG/length sequence-composition regime",
                    "observed_roc_auc_gap": composition_gap,
                    "matched_local_support_roc_auc_delta": matched_delta,
                    "matched_hash_control_roc_auc_delta": hash_delta,
                    "gap_to_matched_local_support_ratio": (
                        composition_gap / max(abs(matched_delta or 0.0), 1e-6)
                    ),
                    "gap_to_hash_control_ratio": (
                        composition_gap / max(abs(hash_delta or 0.0), 1e-6)
                    ),
                },
                "recommended_handoff_to_investigator": {
                    "hypothesis_id": "composition_regime_generalization",
                    "question": (
                        "Do frozen Caduceus enhancer-like probes generalize poorly across "
                        "GC/CpG/length sequence-composition regimes, and does that effect "
                        "persist after class balance, local cCRE support, promoter-distance, "
                        "chromosome/provenance, and overlap controls?"
                    ),
                    "why_this_is_next": (
                        "The composition-control ROC-AUC gap is much larger than the "
                        "matched local-support delta and hash-control fluctuation. The "
                        "attenuated local-support hypothesis should become a negative "
                        "finding, while composition becomes the next primary axis."
                    ),
                    "predictions_to_test": [
                        "If composition regime is a true degradation axis, performance should degrade across low-support composition regimes after matching cCRE class, label balance, local cCRE support, promoter-distance, chromosome/provenance, and overlap.",
                        "If the gap is a dataset-construction artifact, the degradation should shrink to hash-control scale after those controls.",
                        "If composition is only a proxy for regulatory class or promoter distance, residual composition curves should flatten after class and promoter-distance matching.",
                    ],
                    "experiments_to_run": [
                        "Construct a composition-regime SPECTRA split with GC/CpG/length as the primary axis and local cCRE support, cCRE class, label balance, promoter-distance, chromosome/provenance, and overlap as controls.",
                        "Construct a same-pool matched hash control from the same eligible records.",
                        "Run the same frozen Caduceus embedding logistic-probe procedure and compare the composition-regime gap to the matched hash-control gap.",
                        "If the gap persists, annotate low- and high-composition regimes by cCRE class, promoter proximity, CpG island/repeat-like features, and motif-family enrichment to interpret what biological regime the axis represents.",
                    ],
                },
                "paper_ready_checkpoint": {
                    "local_support_negative_finding": (
                        "Low local cCRE support was weakened as an independent axis after "
                        "matching and should be reported only as an attenuation result."
                    ),
                    "composition_axis_status": (
                        "Unresolved stronger candidate axis; do not write a terminal "
                        "paper-ready finding until it is tested with matched controls."
                    ),
                },
                "required_outputs": [
                    "investigator_handoff_composition_axis.json",
                    "composition_axis_split_plan.json",
                    "matched_composition_control_results.csv",
                    "composition_axis_distiller_update.md",
                ],
            }

    precise_insight = str(
        hypothesis_assessment.get("precise_insight")
        or hypothesis_assessment.get("hypothesis")
        or ""
    ).strip()
    if not precise_insight:
        precise_insight = (
            "The SPECTRA evidence supports a bounded generalizability finding, "
            "but the final claim must be written from the provided finding and "
            "trace artifacts rather than inferred as a global model property."
        )

    evidence_boundary = hypothesis_assessment.get("evidence_boundary")
    if not evidence_boundary:
        evidence_boundary = []
    if isinstance(evidence_boundary, str):
        evidence_boundary = [evidence_boundary]
    if not isinstance(evidence_boundary, list):
        evidence_boundary = [str(evidence_boundary)]

    finding_status = str(
        hypothesis_assessment.get("status_after_matched_probe")
        or hypothesis_assessment.get("curve_score_after_matching")
        or hypothesis_assessment.get("classification")
        or "bounded"
    )
    lower_findings = _normalize_token(findings_text)
    if any(token in lower_findings for token in ["weakened_after_matching", "not_supported", "not supported", "not_explanatory"]):
        claim_type = "controlled attenuation / negative-axis finding"
    elif any(token in lower_findings for token in ["monotonic_supported", "localized_supported", "strong independent"]):
        claim_type = "supported degradation-axis finding"
    else:
        claim_type = "bounded interpretive SPECTRA finding"

    model_paper_bridge = (
        "Use the model paper to state what the original evidence established, "
        "then state what the SPECTRA audit adds. The SPECTRA claim should be "
        "phrased as an extension or stress test of the paper's evaluation, not "
        "as a replacement for the paper's original result."
    )
    if "caduceus" in _normalize_token(target_model + " " + paper_text):
        model_paper_bridge = (
            "The Caduceus paper establishes a bidirectional/equivariant DNA "
            "sequence model and evaluates it on the paper's downstream genomic "
            "benchmarks. The SPECTRA result asks a different question: whether "
            "a frozen Caduceus representation probe shows performance changes "
            "under controlled regulatory-context novelty and whether that signal "
            "survives composition and provenance controls."
        )

    process_interpretation = ""
    if vanilla_text:
        process_interpretation = (
            "A broad vanilla generalizability audit can surface split leakage, "
            "duplicate support, or surrogate-model concerns, but the SPECTRA "
            "loop is intended to convert those concerns into target-model "
            "evidence, controls, and a scoped scientific interpretation."
        )

    composition_guardrail = (
        "Do not describe sequence composition as biologically irrelevant by "
        "default. Composition can be a real regulatory covariate. The bounded "
        "claim is that the tested novelty axis was not shown to be independent "
        "of composition/proxy structure under the executed controls."
    )

    result_paragraph = paper_results_log or (
        "In the accumulated SPECTRA evidence for %s, the final supported "
        "interpretation was: %s" % (target_model, precise_insight)
    )
    interpretation_paragraph = (
        "Interpreted against the model paper, this is not a claim that %s "
        "generally fails or succeeds across all biological settings. It is a "
        "%s: the audit followed an initially plausible generalization signal "
        "through controls and narrowed the conclusion to what the evidence can "
        "support." % (target_model, claim_type)
    )
    discussion_paragraph = (
        "%s This makes the SPECTRA contribution interpretive rather than merely "
        "formatting: the final result explains which apparent generalization "
        "signal survived, which signal attenuated, which controls changed the "
        "conclusion, and what cannot be claimed from the current evidence."
        % model_paper_bridge
    )
    if process_interpretation:
        discussion_paragraph = discussion_paragraph + " " + process_interpretation

    overclaim_guardrails = [
        "Do not infer global model generalizability from one SPECTRA audit.",
        "Do not infer global model failure from one attenuated or negative axis.",
        "Do not present frozen-probe evidence as full fine-tuning evidence.",
        "Do not present post-hoc or matched-control evidence as a prospective deployment split unless the split construction supports that claim.",
        composition_guardrail,
        "State the scientific unit, task, model access level, controls, and sample bounds next to the claim.",
    ]
    if evidence_boundary:
        overclaim_guardrails.extend(
            "Respect evidence boundary: %s" % item for item in evidence_boundary
        )

    return {
        "mode": "spectra_final_finding_synthesis",
        "analysis_complete": True,
        "domain": normalized_domain,
        "target_model": target_model,
        "intended_evidence_level": intended_evidence_level,
        "claim_type": claim_type,
        "finding_status": finding_status,
        "paper_context_bridge": model_paper_bridge,
        "one_sentence_finding": precise_insight,
        "paper_ready_sections": {
            "results_paragraph": result_paragraph,
            "interpretation_paragraph": interpretation_paragraph,
            "discussion_paragraph": discussion_paragraph,
            "limitations_paragraph": (
                "The finding is bounded by the executed model access, task, "
                "dataset construction, controls, and sample sizes. It should be "
                "reported as a SPECTRA generalizability finding, not as a complete "
                "characterization of the model."
            ),
        },
        "required_interpretive_questions": [
            "What did the original model paper actually establish?",
            "What SPECTRA question was asked that the model paper did not answer?",
            "Which apparent signal survived controls and which attenuated?",
            "Is the final result a supported degradation axis, an attenuation result, a negative finding, or a blocker?",
            "What is the scientific unit, model access level, and task scope?",
            "Which interpretations are ruled out, weakened, or still live?",
        ],
        "overclaim_guardrails": overclaim_guardrails,
        "source_artifacts": source_inputs,
        "input_trace_available": bool(trace_text.strip()),
        "vanilla_comparison_available": bool(vanilla_text.strip()),
        "required_outputs": [
            "paper_ready_spectra_finding.md",
            "claim_boundary.json",
            "model_paper_context.md",
            "evidence_to_claim_table.csv",
            "overclaim_guardrails.md",
        ],
    }


def review_investigator_checkpoint(
    checkpoint: Union[str, Dict[str, Any]],
    has_observations: bool = False,
    has_hypothesis_ledger: bool = False,
    has_competing_explanations: bool = False,
    has_discriminating_experiment: bool = False,
    has_belief_update: bool = False,
    launched_or_completed_next_experiment: bool = False,
) -> Dict[str, Any]:
    """Review whether an agent behaved like an investigator, not a walker."""
    text = checkpoint if isinstance(checkpoint, str) else json.dumps(checkpoint, sort_keys=True)
    lower = text.lower()
    checklist_smells = [
        "tested axes",
        "mechanism debt remains",
        "mixed results",
        "continuation completed",
    ]
    has_interpretive_language = any(
        token in lower
        for token in [
            "hypothesis",
            "competing explanation",
            "falsifiable",
            "would distinguish",
            "belief update",
            "surprising",
        ]
    )
    smells_without_interpretation = [
        smell for smell in checklist_smells if smell in lower and not has_interpretive_language
    ]
    missing = []
    if not has_observations:
        missing.append("observations")
    if not has_hypothesis_ledger:
        missing.append("hypothesis_ledger")
    if not has_competing_explanations:
        missing.append("competing_explanations")
    if not has_discriminating_experiment:
        missing.append("discriminating_experiment")
    if not has_belief_update:
        missing.append("belief_update")
    if not launched_or_completed_next_experiment:
        missing.append("launched_or_completed_next_experiment")

    passed = not missing and not smells_without_interpretation
    return {
        "decision": "investigator_checkpoint_passed" if passed else "investigator_checkpoint_failed",
        "analysis_complete": False,
        "investigator_behavior_present": passed,
        "missing_investigator_artifacts": missing,
        "checklist_smells_without_interpretation": smells_without_interpretation,
        "required_next_action": (
            "Continue from the sharpest unresolved hypothesis and run or launch "
            "the discriminating experiment."
            if passed
            else "Rewrite the checkpoint around observations, competing hypotheses, falsifiable predictions, a belief update, and a launched discriminating experiment."
        ),
    }


def translate_model_space_axis_to_domain_hypotheses(
    model_space_axis_name: str,
    domain: str,
    model_description: str = "",
    supported_tasks: Optional[List[str]] = None,
    non_supported_tasks: Optional[List[str]] = None,
    available_annotations: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Translate a model-representation axis into scientific hypotheses.

    A model-space similarity axis is behavioral evidence, not a domain
    explanation. This helper tells the agent to ask what scientific factors
    organize high- and low-support regions of the model embedding space, then
    test those factors as new axes when feasible.
    """
    axis = model_space_axis_name.strip() or "model_representation_axis"
    normalized_domain = _normalize_token(domain)
    supported = supported_tasks or []
    failed = non_supported_tasks or []
    annotations = available_annotations or []

    is_sequence_domain = any(
        token in normalized_domain
        for token in ["dna", "rna", "nucleotide", "sequence", "genomic", "genomics", "regulatory", "caduceus"]
    )

    if is_sequence_domain:
        domain_hypotheses = [
            {
                "hypothesis_id": "motif_family_support",
                "scientific_question": "Are low-support embedding regions enriched for different regulatory motif families or discriminative k-mers?",
                "biological_meaning": "The model may transfer poorly when the held-out examples use regulatory words or motif combinations that are underrepresented in training.",
                "test_axis": "TF-like motif, enriched k-mer, TATA/CpG, splice motif, or chromatin-mark word-family support.",
            },
            {
                "hypothesis_id": "regulatory_context_support",
                "scientific_question": "Does embedding-space isolation correspond to enhancer, promoter, splice, histone-mark, species, or assay context?",
                "biological_meaning": "Caduceus may learn smooth representations for some regulatory contexts but not others.",
                "test_axis": "Task-family, assay, species, regulatory-element class, or provenance-conditioned curves.",
            },
            {
                "hypothesis_id": "composition_and_low_complexity_support",
                "scientific_question": "Are low-support embedding regions explained by GC/CpG, repeat, homopolymer, entropy, or low-complexity structure after k-mer matching?",
                "biological_meaning": "The representation axis may reflect compositional regimes rather than deeper regulatory biology.",
                "test_axis": "Residual GC/CpG/repeat/entropy support after matching local k-mer support.",
            },
            {
                "hypothesis_id": "class_conditional_biology",
                "scientific_question": "Does representation isolation hurt positives and negatives differently?",
                "biological_meaning": "The model may learn the biology of active regulatory examples differently from inactive/background examples.",
                "test_axis": "Class-conditional or label-stratified spectral curves constructed without using test labels to define prospective train/test splits.",
            },
            {
                "hypothesis_id": "embedding_neighborhood_biological_enrichment",
                "scientific_question": "What biological features distinguish high-support and low-support embedding neighborhoods?",
                "biological_meaning": "Embedding distance becomes interpretable only after the neighborhoods are annotated with motifs, composition, source context, task family, or provenance.",
                "test_axis": "High-vs-low embedding support enrichment table followed by the strongest non-leaky biological axis.",
            },
        ]
    else:
        domain_hypotheses = [
            {
                "hypothesis_id": "domain_feature_enrichment",
                "scientific_question": "Which measured domain features distinguish high-support and low-support model-embedding regions?",
                "biological_meaning": "The model-space axis should be translated into observable scientific variables before it is treated as an explanation.",
                "test_axis": "Feature-enrichment support axis over available scientific annotations.",
            },
            {
                "hypothesis_id": "context_or_provenance_support",
                "scientific_question": "Does model-space isolation correspond to source, site, assay, batch, time, organism, or deployment context?",
                "biological_meaning": "The model may be learning context-specific regularities rather than portable scientific structure.",
                "test_axis": "Context/provenance-conditioned spectral curves.",
            },
        ]

    required_agent_behavior = [
        "Do not report model-embedding distance as the scientific explanation.",
        "Build a high-vs-low embedding-support contrast table over available domain annotations or sequence-derived features.",
        "Ask what the model seems to learn smoothly and what it does not learn smoothly from the data.",
        "Choose at least one biological/domain hypothesis that could explain the embedding-space signal.",
        "Run a benchmark-mode curve for the biological/domain axis when labels and trainable probes are available.",
        "Report whether the biological/domain axis explains, refines, or fails to explain the model-space degradation.",
    ]

    if annotations:
        annotation_action = "Use the listed annotations plus sequence-derived features to annotate embedding neighborhoods."
    else:
        annotation_action = (
            "If no metadata annotations are available, derive non-leaky biological "
            "features from inputs, such as motifs, composition, length, entropy, "
            "repeat/low-complexity structure, or task-family labels."
        )

    return {
        "decision": "test_domain_translation_axes",
        "model_space_axis": axis,
        "domain": normalized_domain,
        "model_description": model_description.strip(),
        "model_space_interpretation": (
            "A model-space similarity axis means performance changes across the "
            "model's learned representation geometry. It is a behavioral pointer, "
            "not yet a biological or scientific explanation."
        ),
        "supported_tasks": supported,
        "non_supported_tasks": failed,
        "available_annotations": annotations,
        "annotation_action": annotation_action,
        "domain_hypotheses": domain_hypotheses,
        "required_agent_behavior": required_agent_behavior,
        "required_outputs": [
            "model_space_biological_translation.json",
            "embedding_support_biological_contrast.csv",
            "domain_hypothesis_axis_candidates.json",
            "domain_hypothesis_scores.json",
            "biological_interpretation_report.md",
        ],
    }


def assess_explanatory_depth(
    candidate_axis_name: str,
    candidate_axis_description: str = "",
    domain: str = "",
    evidence_status: str = "",
    available_annotations: Optional[List[str]] = None,
    tested_controls: Optional[List[str]] = None,
    current_scope_has_available_inputs: bool = True,
) -> Dict[str, Any]:
    """Assess whether a supported axis is a proxy or a mechanism.

    SPECTRA uses supported proxy axes as evidence for the next experiment, not
    as final explanations. This helper forces the agent to state the current
    explanatory depth, identify the missing mechanism-level variables, and test
    whether a more interpretable axis explains the observed degradation.
    """
    axis = candidate_axis_name.strip() or "candidate_axis"
    description = candidate_axis_description.strip()
    normalized_domain = _normalize_token(domain)
    normalized_status = _normalize_token(evidence_status)
    annotations = available_annotations or []
    controls = tested_controls or []
    text = _normalize_token(" ".join([axis, description]))

    model_space_tokens = [
        "embedding",
        "representation",
        "latent",
        "activation",
        "hidden_state",
        "model_space",
    ]
    surface_proxy_tokens = [
        "kmer",
        "k_mer",
        "n_gram",
        "ngram",
        "edit_distance",
        "hamming",
        "identity",
        "sequence_identity",
        "gc",
        "cpg",
        "length",
        "entropy",
        "low_complexity",
        "homopolymer",
        "exact_match",
        "fingerprint",
        "descriptor",
        "texture",
        "pixel",
        "word_count",
    ]
    broad_proxy_tokens = [
        "regulatory_word",
        "motif_like",
        "enriched_kmer",
        "bag_of_words",
        "task_family",
        "metadata",
        "source",
        "batch",
        "site",
        "assay",
        "provenance",
        "context",
        "cluster",
        "umap",
    ]
    curated_annotation_tokens = [
        "jaspar",
        "hocomoco",
        "transfac",
        "motif_family",
        "tf_family",
        "tf_binding",
        "go",
        "gene_ontology",
        "pathway",
        "reactome",
        "kegg",
        "ontology",
        "conservation",
        "phastcons",
        "phylop",
        "repeatmasker",
        "cpg_island",
        "ccre",
        "chromatin_state",
        "domain_annotation",
        "active_site",
        "binding_site",
    ]
    mechanism_tokens = [
        "mechanism",
        "grammar",
        "spacing",
        "orientation",
        "position",
        "combinatorial",
        "causal",
        "mediation",
        "ablation",
        "intervention",
        "residual_after",
        "controlled_for",
        "matched_for",
        "class_conditional",
        "dose_response",
        "pathway_mechanism",
    ]

    is_model_space = any(token in text for token in model_space_tokens)
    is_surface_proxy = any(token in text for token in surface_proxy_tokens)
    is_broad_proxy = any(token in text for token in broad_proxy_tokens)
    is_curated = any(token in text for token in curated_annotation_tokens)
    is_mechanistic = any(token in text for token in mechanism_tokens)
    has_controls = bool(controls) or any(
        token in text for token in ["residual_after", "controlled_for", "matched_for"]
    )

    if is_mechanistic and has_controls and not is_model_space:
        depth_level = "mechanism_supported_with_controls"
        depth_rank = 5
        proxy_warning = ""
    elif is_model_space:
        depth_level = "model_space_pointer"
        depth_rank = 0
        proxy_warning = (
            "Model-space similarity is a behavioral pointer. It says where the "
            "model behaves differently, not what scientific mechanism explains it."
        )
    elif is_surface_proxy:
        depth_level = "surface_proxy"
        depth_rank = 1
        proxy_warning = (
            "This axis is a surface proxy. It can reveal a failure regime, but it "
            "does not yet explain the model's learned mechanism."
        )
    elif is_broad_proxy:
        depth_level = "domain_proxy"
        depth_rank = 2
        proxy_warning = (
            "This axis is a broad domain proxy. It should be decomposed into "
            "curated annotations, mechanistic features, or residual tests."
        )
    elif is_curated and not has_controls:
        depth_level = "curated_annotation_axis"
        depth_rank = 3
        proxy_warning = (
            "This axis is scientifically interpretable, but it still needs "
            "controls or mediation tests before it is treated as mechanism-level evidence."
        )
    elif is_mechanistic or is_curated:
        depth_level = "mechanistic_hypothesis_without_disambiguation"
        depth_rank = 4
        proxy_warning = (
            "This is a plausible mechanism-level hypothesis, but it should be "
            "disambiguated against simpler proxies and model-space residuals."
        )
    else:
        depth_level = "unclassified_axis"
        depth_rank = 1
        proxy_warning = (
            "The axis is not clearly mechanistic. Treat it as a proxy until the "
            "agent states what scientific mechanism it represents and tests controls."
        )

    is_sequence_domain = any(
        token in normalized_domain
        for token in [
            "dna",
            "rna",
            "nucleotide",
            "sequence",
            "genomic",
            "genomics",
            "regulatory",
            "caduceus",
        ]
    )

    if is_sequence_domain:
        mechanism_hypotheses = [
            {
                "hypothesis_id": "curated_motif_family_support",
                "scientific_question": "Does degradation track support for curated TF motif families or experimentally grounded motif annotations rather than generic k-mer support?",
                "test_axis": "JASPAR/HOCOMOCO/TRANSFAC motif-family support with k-mer, GC/CpG, length, and class balance controls.",
            },
            {
                "hypothesis_id": "motif_grammar_support",
                "scientific_question": "Does the model fail on motif combinations, spacing, orientation, or positional grammar that are poorly represented in training?",
                "test_axis": "Support over motif grammar features, then residual curve after matching surface sequence proxies.",
            },
            {
                "hypothesis_id": "regulatory_context_mechanism",
                "scientific_question": "Does the failure mode depend on enhancer/promoter/splice/chromatin context, assay, species, or cell-state biology?",
                "test_axis": "Context-conditioned spectral curves and interactions between motif support and regulatory element class.",
            },
            {
                "hypothesis_id": "conservation_and_repeat_mechanism",
                "scientific_question": "Does degradation align with evolutionary conservation, repeats, CpG islands, or other annotated genomic regimes after controlling for composition?",
                "test_axis": "Conservation/repeat/CpG-island support with composition-matched controls.",
            },
            {
                "hypothesis_id": "proxy_mediation_test",
                "scientific_question": "Does the proposed mechanism explain the model-space or k-mer failure signal, or does a residual signal remain?",
                "test_axis": "Matched or residual SPECTRA curves that compare mechanism support after controlling for k-mer, GC/CpG, length, task family, and model embedding support.",
            },
        ]
    else:
        mechanism_hypotheses = [
            {
                "hypothesis_id": "curated_domain_annotation_support",
                "scientific_question": "Does degradation track curated domain annotations rather than a generic surface or embedding proxy?",
                "test_axis": "Ontology, pathway, structural, clinical, protocol, or provenance annotations matched against simpler proxies.",
            },
            {
                "hypothesis_id": "mechanism_conditioned_curve",
                "scientific_question": "Does the candidate axis operate only within a specific scientific context or interaction?",
                "test_axis": "Context-conditioned or class-conditional spectral curves.",
            },
            {
                "hypothesis_id": "proxy_mediation_test",
                "scientific_question": "Does the mechanistic axis mediate the proxy/model-space degradation signal?",
                "test_axis": "Residual or matched curves after controlling for the original proxy axis and major confounders.",
            },
        ]

    ladder = [
        {
            "rank": 0,
            "level": "model_space_pointer",
            "examples": "embedding distance, representation-neighbor overlap, latent support",
            "interpretation": "where the model behaves differently, not why",
        },
        {
            "rank": 1,
            "level": "surface_proxy",
            "examples": "k-mers, exact identity, GC/CpG, length, entropy, low complexity, fingerprints, pixels",
            "interpretation": "detects a regime but usually does not identify mechanism",
        },
        {
            "rank": 2,
            "level": "domain_proxy",
            "examples": "crude regulatory words, task family, batch/site/source labels, clusters",
            "interpretation": "biologically or scientifically named, but still too coarse",
        },
        {
            "rank": 3,
            "level": "curated_annotation_axis",
            "examples": "curated motifs, pathways, ontology terms, conservation, chromatin states, structural sites",
            "interpretation": "interpretable axis that can support a mechanism hypothesis",
        },
        {
            "rank": 4,
            "level": "mechanistic_hypothesis_without_disambiguation",
            "examples": "motif grammar, pathway mechanism, structural functional region, context-specific interaction",
            "interpretation": "candidate mechanism that still needs proxy controls",
        },
        {
            "rank": 5,
            "level": "mechanism_supported_with_controls",
            "examples": "mechanism-level axis remains after matching simpler proxies or explains the residual model-space signal",
            "interpretation": "strongest SPECTRA explanation within the current scope",
        },
    ]

    if depth_rank <= 2 and current_scope_has_available_inputs:
        decision = "continue_to_mechanism"
        required_next_action = (
            "Do not stop at this proxy. Use it to formulate and execute at least "
            "one curated annotation, mechanistic, context-conditioned, or mediation "
            "test that asks what the model learned and what it struggles to learn."
        )
        complete = False
    elif depth_rank <= 2:
        decision = "request_mechanism_scope"
        required_next_action = (
            "Report the proxy-level finding and request the annotations, metadata, "
            "model activations, labels, domain tools, or compute needed to test a "
            "mechanistic explanation."
        )
        complete = False
    elif depth_rank in {3, 4} and not has_controls and current_scope_has_available_inputs:
        decision = "test_mediation_or_controls"
        required_next_action = (
            "Run a residual, matched, class-conditional, or context-conditioned "
            "test to distinguish this mechanism hypothesis from simpler proxies."
        )
        complete = False
    elif depth_rank in {3, 4} and not has_controls:
        decision = "request_control_scope"
        required_next_action = (
            "Report the current interpretable hypothesis and request the proxy "
            "features or annotations needed for mediation and confounder controls."
        )
        complete = False
    else:
        decision = "mechanism_depth_supported"
        required_next_action = (
            "Report the mechanism-level axis, the proxy controls it survived, "
            "the residual model-space relation, and the remaining alternative explanations, "
            "then continue to broader task/seed/model-setting coverage or the next residual mechanism question."
        )
        complete = False

    required_agent_behavior = [
        "Treat proxy axes as starting points, not explanations.",
        "State what the proxy could be standing in for scientifically.",
        "Ask whether a curated annotation, mechanism-level feature, or context interaction explains the proxy signal.",
        "Test whether the mechanism remains after matching or controlling for simpler proxies.",
        "If only proxy-level inputs are available, run sequence/input-derived mechanism or mediation tests before reporting missing external mechanism inputs.",
        "Call enforce_mechanism_debt_gate for every supported proxy or model-space axis; a proxy-only report is incomplete.",
    ]

    return {
        "decision": decision,
        "analysis_complete": complete,
        "continuation_required": True,
        "terminal_stop_allowed": False,
        "candidate_axis": axis,
        "domain": normalized_domain,
        "evidence_status": normalized_status,
        "explanatory_depth_level": depth_level,
        "explanatory_depth_rank": depth_rank,
        "proxy_warning": proxy_warning,
        "available_annotations": annotations,
        "tested_controls": controls,
        "proxy_to_mechanism_ladder": ladder,
        "mechanism_hypotheses": mechanism_hypotheses,
        "required_agent_behavior": required_agent_behavior,
        "required_next_action": required_next_action,
        "required_outputs": [
            "explanatory_depth_assessment.json",
            "proxy_to_mechanism_plan.json",
            "mechanism_debt_register.json",
            "mechanism_execution_manifest.json",
            "mechanism_infeasibility_proof.json",
            "public_resource_acquisition_plan.json",
            "public_resource_search_log.json",
            "resource_manifest.json",
            "resource_mapping_validation.json",
            "mechanism_axis_scores.json",
            "mediation_test_results.json",
            "mechanistic_interpretation_report.md",
        ],
    }


def enforce_mechanism_debt_gate(
    supported_axis_name: str,
    axis_depth_level: str,
    domain: str,
    evidence_status: str = "monotonic_supported",
    mechanism_tests_executed: Optional[List[str]] = None,
    available_local_inputs: Optional[List[str]] = None,
    concrete_blockers: Optional[List[str]] = None,
    source_provenance_recovery_attempted: bool = False,
    public_resource_acquisition_attempted: bool = False,
    constructed_dataset_attempted: bool = False,
) -> Dict[str, Any]:
    """Enforce the hard SPECTRA gate created by supported proxy axes.

    A supported proxy or model-space axis creates mechanism debt. The audit is
    not complete merely because the report states that the axis is proxy-level.
    Local input-derived tests are only the first tier. They do not close
    mechanism debt when the axis remains proxy-level or model-space. The agent
    must continue toward curated/public resources, constructed hypothesis-test
    datasets, or a mechanism-level explanation with controls.
    """
    axis = supported_axis_name.strip()
    depth = _normalize_token(axis_depth_level)
    normalized_domain = _normalize_token(domain)
    normalized_status = _normalize_token(evidence_status)
    tests = mechanism_tests_executed or []
    local_inputs = available_local_inputs or []
    blockers = concrete_blockers or []
    local_text = _normalize_token(" ".join(local_inputs))

    debt_creating_depths = {
        "model_space_pointer",
        "surface_proxy",
        "domain_proxy",
        "curated_annotation_axis",
        "mechanistic_hypothesis_without_disambiguation",
        "unclassified_axis",
    }
    supported_status = normalized_status in {
        "monotonic_supported",
        "localized_supported",
        "weak_supported",
    }
    is_mechanism_supported = depth == "mechanism_supported_with_controls"
    creates_debt = supported_status and not is_mechanism_supported and (
        depth in debt_creating_depths or "proxy" in depth or "model_space" in depth
    )

    is_sequence_domain = any(
        token in normalized_domain
        for token in ["dna", "rna", "nucleotide", "sequence", "genomic", "regulatory", "caduceus"]
    )
    has_local_input_features = bool(local_inputs) or any(
        token in local_text
        for token in [
            "sequence",
            "fasta",
            "smiles",
            "inchi",
            "graph",
            "image",
            "pixels",
            "text",
            "metadata",
            "features",
            "embedding",
            "expression",
            "counts",
        ]
    )

    if not creates_debt:
        decision = "no_mechanism_debt"
        required_next_action = (
            "No mechanism debt is active for this axis/status pair. Continue normal SPECTRA reporting."
        )
        stop_allowed = False
        complete = is_mechanism_supported and supported_status
    elif tests and is_mechanism_supported:
        decision = "mechanism_depth_supported_continue_scope"
        required_next_action = (
            "Mechanism-level controls have been executed for this axis. Continue by expanding "
            "task/seed/model-setting coverage or testing the next residual mechanism question; "
            "do not stop unless the full feasible audit scope is complete."
        )
        stop_allowed = False
        complete = False
    elif tests and not public_resource_acquisition_attempted:
        decision = "local_tests_executed_continue_to_public_mechanism"
        required_next_action = (
            "Local sequence/input-derived tests were executed, but the axis remains proxy-level "
            "or model-space. Continue to source provenance recovery and public-resource acquisition "
            "for curated mechanism variables; local proxy tests do not satisfy mechanism debt."
        )
        stop_allowed = False
        complete = False
    elif tests and not constructed_dataset_attempted:
        decision = "local_public_tests_insufficient_construct_hypothesis_dataset"
        required_next_action = (
            "Local/public mechanism work has not produced a mechanism-level explanation with controls. "
            "Construct a public/local hypothesis-test dataset and run fresh split-based SPECTRA on it."
        )
        stop_allowed = False
        complete = False
    elif tests:
        decision = "mechanism_debt_unresolved_continue_residual_search"
        required_next_action = (
            "All currently attempted tiers remain proxy-level or unresolved. Continue with residual axes, "
            "alternate public resources, constructed datasets, or broader task/model coverage. Do not "
            "convert this into a terminal infeasibility report."
        )
        stop_allowed = False
        complete = False
    elif has_local_input_features:
        decision = "execute_local_mechanism_or_mediation_test"
        if is_sequence_domain:
            required_next_action = (
                "Do not use missing coordinates as an out. Execute at least one sequence-derived "
                "mechanism or mediation test now: curated motif/PWM scanning when available, motif "
                "grammar, CpG/low-complexity/repeat-like features, GC/length/k-mer matched residual "
                "curves, class-conditional curves, or model-space residuals after matching surface proxies."
            )
        else:
            required_next_action = (
                "Execute at least one local input-derived mechanism or mediation test now: curated "
                "feature extraction available from local inputs, context-conditioned curves, matched "
                "controls, or residual curves after matching simpler proxies."
            )
        stop_allowed = False
        complete = False
    elif not source_provenance_recovery_attempted:
        decision = "recover_source_provenance_for_mechanism"
        required_next_action = (
            "Inspect upstream repositories, loaders, raw archives, manifests, metadata, and supplements "
            "for identifiers or annotations needed to execute a mechanism test."
        )
        stop_allowed = False
        complete = False
    elif not public_resource_acquisition_attempted:
        decision = "acquire_public_resources_for_mechanism"
        required_next_action = (
            "Search, acquire or query, map, and validate public resources needed for a mechanism test. "
            "Do not report a final proxy-only result before this acquisition tier is attempted."
        )
        stop_allowed = False
        complete = False
    elif not constructed_dataset_attempted:
        decision = "construct_hypothesis_test_dataset_for_mechanism"
        required_next_action = (
            "Construct a defensible hypothesis-test dataset from public/local resources and run fresh "
            "split-based SPECTRA to test the mechanism hypothesis. Local proxy-audit feasibility does "
            "not waive this mechanism debt."
        )
        stop_allowed = False
        complete = False
    elif blockers:
        decision = "continuation_checkpoint_with_mechanism_blockers"
        required_next_action = (
            "Record the blockers as a continuation checkpoint, then launch the next executable "
            "route: alternate task, alternate axis, public/local constructed dataset, smaller "
            "validation slice, or bounded resource/compute job. If the larger job cannot be "
            "launched in the current workspace, launch a smaller fallback and record why the "
            "larger job is blocked. A queued-only manifest is not sufficient."
        )
        stop_allowed = False
        complete = False
    else:
        decision = "mechanism_infeasibility_proof_required"
        required_next_action = (
            "Do not stop. Provide concrete blockers for all mechanism execution tiers or execute the next "
            "available mechanism test."
        )
        stop_allowed = False
        complete = False

    local_sequence_tests = [
        "motif/PWM family support from sequence strings",
        "motif grammar, spacing, orientation, and positional support",
        "GC/CpG, length, entropy, low-complexity, homopolymer, and repeat-like controls",
        "residual SPECTRA curves after matching k-mer/GC/length/task family",
        "class-conditional or context-conditioned curves from available labels/metadata",
        "model-space residuals after matching surface sequence support",
    ]

    required_outputs = [
        "mechanism_debt_register.json",
        "mechanism_execution_manifest.json",
        "mechanism_axis_scores.json",
        "mediation_test_results.json",
        "mechanism_infeasibility_proof.json",
        "public_resource_acquisition_plan.json",
        "source_provenance_recovery_log.json",
        "hypothesis_test_dataset_plan.json",
        "constructed_dataset_spectra_results.json",
    ]

    return {
        "decision": decision,
        "analysis_complete": complete,
        "stop_allowed": stop_allowed,
        "continuation_required": True,
        "terminal_stop_allowed": False,
        "mechanism_debt_active": creates_debt,
        "supported_axis_name": axis,
        "axis_depth_level": depth,
        "domain": normalized_domain,
        "evidence_status": normalized_status,
        "mechanism_tests_executed": tests,
        "available_local_inputs": local_inputs,
        "has_local_input_features": has_local_input_features,
        "source_provenance_recovery_attempted": source_provenance_recovery_attempted,
        "public_resource_acquisition_attempted": public_resource_acquisition_attempted,
        "constructed_dataset_attempted": constructed_dataset_attempted,
        "concrete_blockers": blockers,
        "required_next_action": required_next_action,
        "mandatory_execution_order": [
            "local_input_derived_mechanism_or_mediation_test",
            "source_provenance_recovery",
            "public_resource_acquisition_and_mapping",
            "constructed_public_or_local_hypothesis_test_dataset",
            "mechanism_infeasibility_proof_if_all_tiers_fail",
        ],
        "sequence_domain_local_tests": local_sequence_tests if is_sequence_domain else [],
        "forbidden_outs": [
            "Proxy-level finding noted in report but no mechanism test executed.",
            "Missing coordinates used to skip sequence-derived motif/composition/residual tests.",
            "Local benchmark-mode proxy audit was feasible, so constructed mechanism dataset was declared unnecessary.",
            "Mechanism work listed as future work without an executable continuation job.",
            "Current-scope exhaustion declared before local, public-resource, and constructed-dataset tiers are attempted.",
            "Local sequence-derived mechanism tests treated as sufficient while the explanation remains proxy-level.",
            "A blocker report used as the final artifact instead of launching or specifying the next executable continuation.",
        ],
        "required_outputs": required_outputs,
    }


def plan_public_resource_acquisition(
    domain: str,
    missing_resources: Optional[List[str]] = None,
    dataset_description: str = "",
    scientific_question: str = "",
    local_identifiers_available: Optional[List[str]] = None,
    dataset_sources: Optional[List[str]] = None,
    allowed_network: bool = True,
    allow_large_downloads: bool = False,
) -> Dict[str, Any]:
    """Plan acquisition of public resources needed for deeper SPECTRA tests.

    Absence from the local bundle is not itself a SPECTRA blocker. The agent
    should search for public references, download versioned resources when
    licensing and scale allow, map them onto the current scientific units, and
    validate that mapping before stopping.
    """
    normalized_domain = _normalize_token(domain)
    missing = missing_resources or []
    identifiers = local_identifiers_available or []
    sources = dataset_sources or []
    identifier_text = _normalize_token(" ".join(identifiers + sources + [dataset_description]))

    is_sequence_domain = any(
        token in normalized_domain
        for token in [
            "dna",
            "rna",
            "nucleotide",
            "sequence",
            "genomic",
            "genomics",
            "regulatory",
            "caduceus",
            "variant",
            "eqtl",
        ]
    )
    is_molecule_domain = any(
        token in normalized_domain
        for token in ["molecule", "chemical", "drug", "protein_ligand", "compound"]
    )
    is_perturbation_domain = any(
        token in normalized_domain
        for token in ["single_cell", "perturbation", "gene", "cell", "transcriptomic"]
    )
    is_clinical_domain = any(
        token in normalized_domain
        for token in ["clinical", "ehr", "patient", "hospital", "medical", "health"]
    )
    is_imaging_domain = any(
        token in normalized_domain
        for token in ["image", "imaging", "radiology", "pathology", "microscopy", "dicom"]
    )

    if is_sequence_domain:
        source_provenance_targets = [
            "genome assembly",
            "chromosome/start/end coordinates",
            "variant IDs or accessions",
            "gene/transcript IDs",
            "raw FASTA/BED/VCF files",
            "dataset loader metadata",
        ]
        candidate_resources = [
            {
                "resource_class": "reference_genome",
                "examples": ["UCSC/Ensembl/NCBI FASTA or 2bit", "assembly-specific chromosome sizes"],
                "use_for": "extract sequence windows, reconstruct variant alleles, and map interval coordinates",
                "required_local_ids": ["assembly", "chromosome", "start/end or variant position"],
            },
            {
                "resource_class": "curated_tf_motifs",
                "examples": ["JASPAR", "HOCOMOCO", "TRANSFAC-compatible public motif files"],
                "use_for": "motif-family support, motif grammar, spacing, orientation, and position axes",
                "required_local_ids": ["sequence windows or genomic intervals"],
            },
            {
                "resource_class": "regulatory_annotations",
                "examples": ["ENCODE SCREEN cCREs", "ENCODE/Roadmap chromatin states", "TF ChIP-seq peaks"],
                "use_for": "regulatory context, chromatin state, assay/context-conditioned curves",
                "required_local_ids": ["assembly", "genomic intervals"],
            },
            {
                "resource_class": "conservation_and_repeats",
                "examples": ["phastCons/phyloP", "RepeatMasker", "CpG islands"],
                "use_for": "conservation, repeat, CpG-island, and low-complexity mechanism axes",
                "required_local_ids": ["assembly", "genomic intervals"],
            },
            {
                "resource_class": "gene_and_variant_context",
                "examples": ["GENCODE/RefSeq", "dbSNP", "gnomAD", "GTEx eQTL resources"],
                "use_for": "gene distance, regulatory target context, variant identity, and eQTL sequence-window reconstruction",
                "required_local_ids": ["assembly", "gene/variant IDs or coordinates"],
            },
        ]
    elif is_molecule_domain:
        source_provenance_targets = [
            "SMILES/InChI/InChIKey",
            "compound IDs",
            "target protein IDs",
            "assay IDs",
            "source database accessions",
            "raw assay tables",
        ]
        candidate_resources = [
            {
                "resource_class": "chemical_bioactivity_and_targets",
                "examples": ["ChEMBL", "PubChem", "BindingDB", "UniProt target annotations"],
                "use_for": "target-family, assay, mechanism-of-action, and chemotype context axes",
                "required_local_ids": ["SMILES, InChIKey, compound IDs, or target IDs"],
            },
            {
                "resource_class": "structure_and_scaffold_context",
                "examples": ["PDB structures", "AlphaFold structures", "Murcko scaffolds from local SMILES"],
                "use_for": "binding-site, structural-neighborhood, and scaffold-mechanism axes",
                "required_local_ids": ["SMILES, protein IDs, or structures"],
            },
        ]
    elif is_perturbation_domain:
        source_provenance_targets = [
            "gene symbols or Ensembl IDs",
            "guide/RNA/perturbation IDs",
            "drug IDs",
            "cell type or tissue labels",
            "donor/batch/protocol metadata",
            "raw perturbation experiment manifests",
        ]
        candidate_resources = [
            {
                "resource_class": "gene_pathway_ontology",
                "examples": ["Gene Ontology", "Reactome", "KEGG where licensed", "MSigDB where licensed"],
                "use_for": "pathway, perturbation-target, and biological-process similarity axes",
                "required_local_ids": ["gene symbols, Ensembl IDs, or perturbation target IDs"],
            },
            {
                "resource_class": "cell_and_perturbation_references",
                "examples": ["CELLxGENE", "LINCS/L1000", "OpenTargets", "DepMap"],
                "use_for": "cell-type, tissue, drug, target, and perturbation-context axes",
                "required_local_ids": ["cell type, tissue, perturbation, gene, or drug IDs"],
            },
        ]
    elif is_clinical_domain:
        source_provenance_targets = [
            "patient/site/time identifiers",
            "ICD/SNOMED/LOINC/RxNorm codes",
            "encounter or visit IDs",
            "hospital/system labels",
            "device/protocol metadata",
            "raw cohort extraction definitions",
        ]
        candidate_resources = [
            {
                "resource_class": "clinical_terminology_and_phenotype_maps",
                "examples": ["ICD", "SNOMED CT where licensed", "LOINC", "RxNorm", "OMOP vocabularies"],
                "use_for": "phenotype, care-process, medication, laboratory, and diagnosis-context axes",
                "required_local_ids": ["diagnosis/procedure/lab/medication codes or OMOP concept IDs"],
            },
            {
                "resource_class": "site_time_device_context",
                "examples": ["hospital/site metadata", "collection time periods", "device/protocol tables", "public cohort documentation"],
                "use_for": "site, temporal, device, protocol, and deployment-context curves",
                "required_local_ids": ["site, time, device, encounter, or cohort identifiers"],
            },
        ]
    elif is_imaging_domain:
        source_provenance_targets = [
            "DICOM metadata",
            "scanner/site identifiers",
            "acquisition protocol fields",
            "study/series/image IDs",
            "segmentation or annotation IDs",
            "raw archive manifests",
        ]
        candidate_resources = [
            {
                "resource_class": "imaging_metadata_and_protocol_context",
                "examples": ["DICOM headers", "TCIA metadata", "scanner/protocol tables", "stain/scanner metadata for pathology"],
                "use_for": "site, scanner, protocol, acquisition, stain, and cohort-context axes",
                "required_local_ids": ["image/study IDs, DICOM metadata, scanner/site fields"],
            },
            {
                "resource_class": "anatomical_or_pathology_annotations",
                "examples": ["segmentation masks", "lesion annotations", "tissue labels", "radiomics feature references"],
                "use_for": "anatomical region, lesion type, tissue context, and morphology mechanism axes",
                "required_local_ids": ["image IDs, masks, labels, or annotation IDs"],
            },
        ]
    else:
        source_provenance_targets = [
            "stable sample identifiers",
            "raw input identifiers",
            "source database accessions",
            "context/provenance metadata",
            "raw benchmark manifests",
            "dataset loader metadata",
        ]
        candidate_resources = [
            {
                "resource_class": "curated_domain_annotations",
                "examples": ["domain ontologies", "reference databases", "public benchmark metadata", "expert-curated annotation tables"],
                "use_for": "mechanism-level, context-conditioned, and mediation axes",
                "required_local_ids": ["stable sample identifiers or mappable raw inputs"],
            }
        ]

    has_mapping_ids = any(
        token in identifier_text
        for token in [
            "chrom",
            "chr",
            "start",
            "end",
            "position",
            "coordinate",
            "interval",
            "bed",
            "rsid",
            "variant",
            "gene",
            "ensembl",
            "uniprot",
            "smiles",
            "inchi",
            "inchikey",
            "compound",
            "chembl",
            "pubchem",
            "assay",
            "target",
            "drug",
            "guide",
            "cell_type",
            "tissue",
            "donor",
            "patient",
            "site",
            "hospital",
            "encounter",
            "icd",
            "snomed",
            "loinc",
            "rxnorm",
            "omop",
            "dicom",
            "scanner",
            "study",
            "series",
            "accession",
            "sequence",
        ]
    )

    if not allowed_network:
        decision = "network_not_allowed"
        required_next_action = (
            "Record that public acquisition was not permitted, then request the "
            "needed resources or permission to download them."
        )
    elif not has_mapping_ids:
        decision = "recover_source_provenance_before_download"
        required_next_action = (
            "Before downloading external resources, attempting expensive matching, "
            "or reporting a mapping blocker, inspect upstream dataset provenance: "
            "original benchmark repositories, dataset loaders, source manifests, "
            "HuggingFace/TDC/OpenML pages, supplementary files, raw archives, "
            "README files, dataset cards, and metadata tables. Try to recover the "
            "domain identifiers needed for mapping: %s. If provenance recovery "
            "fails, report missing mapping identifiers as the blocker."
            % ", ".join(source_provenance_targets)
        )
    else:
        decision = "attempt_public_resource_acquisition"
        required_next_action = (
            "Search official or canonical public sources, pin versions and licenses, "
            "download a bounded resource or queryable slice unless large downloads "
            "are allowed, map resources onto the current scientific units, validate "
            "the mapping on a bounded validation sample, then run the next SPECTRA "
            "mechanism or mediation test."
        )

    if allow_large_downloads:
        scale_policy = "Large downloads are allowed if versions, checksums, storage path, and cleanup policy are recorded."
    else:
        scale_policy = (
            "Prefer small files, indexed slices, API/range queries, chromosome shards, "
            "precomputed motif matrices, or pilot subsets before downloading multi-GB resources."
        )

    acquisition_steps = [
        "Translate the current shallow proxy into the specific missing resource classes.",
        "Search public sources, prioritizing official project, consortium, archive, or package documentation.",
        "Check license, citation, assembly/version, file format, and expected size.",
        "If mapping identifiers are absent locally, inspect upstream dataset sources, loaders, raw archives, and documentation for domain IDs or stable provenance before matching, alignment, lookup, or blocker reporting.",
        "Download to a run-scoped path outside /tmp and record source URL, date, checksum when feasible, and version.",
        "Map the public resource to local sample identifiers, coordinates, sequences, compounds, genes, cells, or tasks.",
        "Validate mapping on a bounded validation sample before full-scale computation.",
        "Run a targeted informative mechanism-axis or mediation test.",
        "If acquisition fails, report the precise failure: no mapping identifiers, license/credential barrier, unavailable assembly/version, excessive size, network failure, or failed mapping validation.",
    ]

    return {
        "decision": decision,
        "domain": normalized_domain,
        "missing_resources": missing,
        "dataset_description": dataset_description.strip(),
        "scientific_question": scientific_question.strip(),
        "local_identifiers_available": identifiers,
        "dataset_sources": sources,
        "source_provenance_targets": source_provenance_targets,
        "has_mapping_identifiers": has_mapping_ids,
        "allowed_network": allowed_network,
        "allow_large_downloads": allow_large_downloads,
        "scale_policy": scale_policy,
        "candidate_public_resources": candidate_resources,
        "acquisition_steps": acquisition_steps,
        "required_next_action": required_next_action,
        "required_agent_behavior": [
            "Do not treat absence from the local bundle as a final blocker.",
            "Try public resource acquisition before stopping when the resource is plausibly public and mappable.",
            "Before sequence alignment, compound lookup, ontology joins, cohort metadata joins, image metadata joins, or final resource blockers, inspect upstream dataset provenance for hidden IDs, source accessions, or raw benchmark files.",
            "Prefer official/canonical sources and record versions, licenses, citations, and checksums.",
            "Map and validate resources against local units before using them for SPECTRA splits or mediation.",
            "If acquisition fails, report the exact acquisition or mapping blocker and the resource needed to continue.",
        ],
        "required_outputs": [
            "public_resource_acquisition_plan.json",
            "public_resource_search_log.json",
            "source_provenance_recovery_log.json",
            "resource_manifest.json",
            "resource_mapping_validation.json",
            "acquisition_blockers.json",
        ],
    }


def plan_hypothesis_test_dataset_construction(
    domain: str,
    mechanism_hypothesis: str,
    current_dataset_limitation: str = "",
    available_local_resources: Optional[List[str]] = None,
    acquired_public_resources: Optional[List[str]] = None,
    target_model_description: str = "",
    desired_labels: Optional[List[str]] = None,
    candidate_units: Optional[List[str]] = None,
    allowed_network: bool = True,
) -> Dict[str, Any]:
    """Plan construction of a dataset from public/local resources.

    Use this when SPECTRA has a plausible mechanism hypothesis but the current
    benchmark cannot test it because rows lack identifiers, labels, context, or
    mappable annotations. The constructed dataset is a hypothesis-test dataset:
    it extends the investigation, but it does not replace claims about the
    original benchmark.
    """
    normalized_domain = _normalize_token(domain)
    hypothesis = mechanism_hypothesis.strip()
    limitation = current_dataset_limitation.strip()
    local_resources = available_local_resources or []
    public_resources = acquired_public_resources or []
    labels = desired_labels or []
    units = candidate_units or []

    resource_text = _normalize_token(
        " ".join(
            local_resources
            + public_resources
            + labels
            + units
            + [hypothesis, limitation, target_model_description]
        )
    )
    has_seed_resources = bool(local_resources or public_resources) or any(
        token in resource_text
        for token in [
            "fasta",
            "2bit",
            "bed",
            "vcf",
            "jaspar",
            "encode",
            "ccre",
            "gencode",
            "chembl",
            "pubchem",
            "bindingdb",
            "uniprot",
            "pdb",
            "go",
            "reactome",
            "lincs",
            "cellxgene",
            "opentargets",
            "dicom",
            "tcia",
            "omop",
            "mimic",
        ]
    )

    is_sequence_domain = any(
        token in normalized_domain
        for token in [
            "dna",
            "rna",
            "nucleotide",
            "sequence",
            "genomic",
            "genomics",
            "regulatory",
            "variant",
            "eqtl",
        ]
    )
    is_molecule_domain = any(
        token in normalized_domain
        for token in ["molecule", "chemical", "drug", "compound", "protein_ligand"]
    )
    is_perturbation_domain = any(
        token in normalized_domain
        for token in ["single_cell", "perturbation", "gene", "cell", "transcriptomic"]
    )
    is_clinical_domain = any(
        token in normalized_domain
        for token in ["clinical", "ehr", "patient", "hospital", "medical", "health"]
    )
    is_imaging_domain = any(
        token in normalized_domain
        for token in ["image", "imaging", "radiology", "pathology", "microscopy", "dicom"]
    )

    if is_sequence_domain:
        dataset_candidates = [
            {
                "dataset_type": "sequence_window_dataset",
                "scientific_units": [
                    "variant-centered sequence windows",
                    "regulatory intervals",
                    "promoter/enhancer windows",
                    "allele-pair sequence windows",
                ],
                "public_or_local_resources": [
                    "reference FASTA or 2bit",
                    "source BED/VCF/eQTL metadata",
                    "ENCODE cCRE or chromatin annotations",
                    "JASPAR/HOCOMOCO motif matrices",
                    "GENCODE/RefSeq gene models",
                    "conservation, repeat, and CpG tracks",
                ],
                "label_strategies": [
                    "annotation-derived labels such as cCRE class, promoter/enhancer class, TF-family support, or eQTL context",
                    "paired or matched controls that preserve length, GC/CpG, chromosome, and class balance",
                    "task labels recovered from source benchmark metadata when licensing permits",
                ],
                "spectra_axes_to_test": [
                    "motif-family and motif-grammar support",
                    "regulatory-element or chromatin-context support",
                    "TSS/gene-context distance",
                    "conservation/repeat/CpG-island support",
                    "sequence-support controls such as k-mer, GC/CpG, and length",
                ],
            }
        ]
    elif is_molecule_domain:
        dataset_candidates = [
            {
                "dataset_type": "compound_or_compound_target_dataset",
                "scientific_units": [
                    "compound rows",
                    "compound-target assay rows",
                    "protein-ligand pairs",
                ],
                "public_or_local_resources": [
                    "local SMILES/InChI/InChIKey files",
                    "ChEMBL",
                    "PubChem",
                    "BindingDB",
                    "UniProt target annotations",
                    "PDB or AlphaFold structures where relevant",
                ],
                "label_strategies": [
                    "bioactivity thresholds or continuous assay readouts",
                    "target-family, mechanism-of-action, or assay-context labels",
                    "matched inactive/active controls from the same source and assay family",
                ],
                "spectra_axes_to_test": [
                    "scaffold and chemotype support",
                    "target-family support",
                    "assay/protocol support",
                    "protein binding-site or structure-context support",
                    "property-extreme and mechanism-of-action support",
                ],
            }
        ]
    elif is_perturbation_domain:
        dataset_candidates = [
            {
                "dataset_type": "cell_perturbation_response_dataset",
                "scientific_units": [
                    "cell-perturbation pairs",
                    "gene perturbations",
                    "drug perturbations",
                    "cell-type or tissue contexts",
                ],
                "public_or_local_resources": [
                    "local perturbation matrices and metadata",
                    "LINCS/L1000",
                    "CELLxGENE",
                    "Gene Ontology",
                    "Reactome",
                    "OpenTargets",
                    "DepMap",
                ],
                "label_strategies": [
                    "response magnitude or direction labels",
                    "perturbation target/pathway labels",
                    "cell-type, tissue, or disease-context labels",
                    "matched controls across batch, donor, cell type, and perturbation class",
                ],
                "spectra_axes_to_test": [
                    "pathway or ontology support",
                    "perturbation-target support",
                    "cell-type/tissue context support",
                    "drug-target or mechanism-of-action support",
                    "batch/protocol controls",
                ],
            }
        ]
    elif is_clinical_domain:
        dataset_candidates = [
            {
                "dataset_type": "clinical_deployment_context_dataset",
                "scientific_units": [
                    "patient encounters",
                    "site-time cohorts",
                    "episodes of care",
                    "phenotype-labeled patient rows",
                ],
                "public_or_local_resources": [
                    "local cohort extracts",
                    "MIMIC/eICU or other licensed public cohorts",
                    "OMOP vocabularies",
                    "ICD/SNOMED/LOINC/RxNorm mappings where licensed",
                    "site, time, device, or protocol metadata",
                ],
                "label_strategies": [
                    "phenotype or outcome labels from documented cohort definitions",
                    "site/time/device context labels",
                    "matched controls across age, sex, prevalence, and care setting",
                ],
                "spectra_axes_to_test": [
                    "site support",
                    "temporal support",
                    "phenotype-definition support",
                    "care-process or coding-system support",
                    "device/protocol support",
                ],
            }
        ]
    elif is_imaging_domain:
        dataset_candidates = [
            {
                "dataset_type": "imaging_context_dataset",
                "scientific_units": [
                    "images",
                    "study or series rows",
                    "lesion/tissue regions",
                    "patient-study pairs",
                ],
                "public_or_local_resources": [
                    "local image archives and metadata",
                    "TCIA or other public imaging cohorts",
                    "DICOM headers",
                    "scanner/protocol metadata",
                    "segmentation, lesion, or tissue annotations",
                ],
                "label_strategies": [
                    "diagnosis, lesion, tissue, or morphology labels",
                    "site/scanner/protocol labels",
                    "matched controls across acquisition and patient covariates",
                ],
                "spectra_axes_to_test": [
                    "scanner/site support",
                    "protocol/acquisition support",
                    "anatomical or morphology support",
                    "texture/radiomic support",
                    "segmentation/tissue-context support",
                ],
            }
        ]
    else:
        dataset_candidates = [
            {
                "dataset_type": "domain_hypothesis_test_dataset",
                "scientific_units": units or ["domain samples with stable IDs"],
                "public_or_local_resources": [
                    "local raw features and labels",
                    "source benchmark metadata",
                    "official public datasets",
                    "domain ontologies or reference tables",
                    "curated annotations with versions and licenses",
                ],
                "label_strategies": labels
                or [
                    "non-circular labels derived from public or local annotations",
                    "matched controls for known confounders",
                    "held-out target labels from a documented benchmark source",
                ],
                "spectra_axes_to_test": [
                    "source/provenance support",
                    "ontology or mechanism support",
                    "representation support after matching surface features",
                    "context-conditioned support",
                ],
            }
        ]

    if not has_seed_resources and not allowed_network:
        decision = "blocked_without_public_or_local_resources"
        required_next_action = (
            "Report that the current benchmark cannot test the mechanism hypothesis "
            "and no public/local resources are available under the declared network "
            "policy. Name the resources needed to continue."
        )
    elif not has_seed_resources:
        decision = "recover_or_acquire_resources_before_construction"
        required_next_action = (
            "Search for public resources or recover local source provenance before "
            "constructing the hypothesis-test dataset. Prioritize official datasets, "
            "reference tables, source benchmark repositories, and licensed archives."
        )
    else:
        decision = "construct_hypothesis_test_dataset"
        required_next_action = (
            "Construct a defensible hypothesis-test dataset from public/local "
            "resources, then run benchmark-mode SPECTRA with fresh model fits or "
            "fresh probes per split to test whether the mechanism hypothesis "
            "changes target-model behavior."
        )

    construction_steps = [
        "State the mechanism hypothesis and why the current benchmark cannot test it.",
        "Define the scientific unit, candidate labels, inclusion criteria, and exclusion criteria.",
        "Select public/local resources with recorded source, version, license, citation, retrieval date, and checksum when feasible.",
        "Build a dataset schema that separates raw identifiers, features, labels, similarity features, split metadata, and model outputs.",
        "Derive labels from domain resources or documented local labels; do not derive labels from target-model errors, split membership, or post-hoc performance.",
        "Construct matched controls for obvious confounders such as length, class balance, batch, source, site, assay, chromosome, scaffold, or time when relevant.",
        "Validate mappings between resources and scientific units on a bounded validation subset before full computation.",
        "Predefine prospective similarity axes and leakage rules before evaluating target-model behavior.",
        "Generate similarity-controlled splits from prospective features only.",
        "Run fresh training, fine-tuning, adapter fitting, probe fitting, or a documented baseline per split.",
        "Score the resulting SPECTRA curves and compare against surface-proxy controls.",
        "Report whether the constructed dataset supports, refines, or falsifies the mechanism hypothesis, while keeping claims separate from the original benchmark.",
    ]

    quality_gates = [
        "The dataset tests a named mechanism hypothesis, not a vague generalizability concern.",
        "Every row has a clear scientific unit and provenance.",
        "Labels are non-circular and are not derived from target-model predictions or errors.",
        "Similarity axes used for split design are prospective or explicitly labeled post-hoc explanatory.",
        "Mappings across resources are validated and mapping failures are reported.",
        "Known confounders and simpler proxies are matched, stratified, controlled, or reported as unresolved.",
        "Train/test splits are constructed before fitting or evaluating the target model on held-out units.",
        "A fresh model component or defensible baseline is trained per split whenever benchmark mode is feasible.",
        "Negative, non-explanatory, and failed-axis findings are reported.",
        "The report states that this is a hypothesis-test extension and does not replace the original benchmark evaluation.",
    ]

    return {
        "decision": decision,
        "domain": normalized_domain,
        "mechanism_hypothesis": hypothesis,
        "current_dataset_limitation": limitation,
        "target_model_description": target_model_description.strip(),
        "available_local_resources": local_resources,
        "acquired_public_resources": public_resources,
        "desired_labels": labels,
        "candidate_units": units,
        "allowed_network": allowed_network,
        "has_seed_resources": has_seed_resources,
        "agent_instruction": (
            "When the current benchmark cannot answer the mechanism question, "
            "extend SPECTRA by constructing a hypothesis-test dataset from "
            "public/local resources and auditing target-model behavior on fresh "
            "similarity-controlled splits."
        ),
        "dataset_candidates": dataset_candidates,
        "construction_steps": construction_steps,
        "quality_gates": quality_gates,
        "claim_scope": (
            "A constructed hypothesis-test dataset can show whether the proposed "
            "mechanism produces a target-model degradation under controlled "
            "conditions. It should be reported as additional evidence, not as a "
            "retroactive proof about rows that could not be mapped in the original "
            "benchmark."
        ),
        "required_next_action": required_next_action,
        "required_agent_behavior": [
            "Do not stop at a source-provenance or mapping blocker if a public/local dataset can test the mechanism hypothesis directly.",
            "Construct the dataset from public/local resources with explicit provenance and leakage controls.",
            "Use the constructed dataset to run fresh split-based SPECTRA experiments, not fixed-prediction binning when retraining or probing is feasible.",
            "Treat results as hypothesis-test evidence and clearly separate them from claims about the original benchmark.",
            "Use negative constructed-dataset results to revise the mechanism hypothesis and choose the next public/local resource or axis.",
        ],
        "required_outputs": [
            "hypothesis_test_dataset_plan.json",
            "constructed_dataset_manifest.json",
            "constructed_dataset_provenance.json",
            "constructed_dataset_schema.json",
            "constructed_dataset_mapping_validation.json",
            "constructed_dataset_leakage_audit.json",
            "constructed_dataset_split_stats.csv",
            "constructed_dataset_retraining_manifest.csv",
            "constructed_dataset_spectra_results.json",
            "constructed_dataset_limitations.md",
        ],
    }


def prepare_dataset_scout_request(
    distiller_handoff: Union[str, Dict[str, Any]],
    investigator_summary: Optional[Union[str, List[Dict[str, Any]], Dict[str, Any]]] = None,
    domain: str = "unknown",
    target_model_description: str = "",
    constraints: str = "",
    allowed_network: bool = True,
    min_candidates: int = 5,
) -> Dict[str, Any]:
    """Prepare a Dataset Scout request from a Distiller conclusion.

    Dataset Scout is used before Dataset Constructor when the current benchmark
    is insufficient but the best replacement dataset is not yet clear. The
    Scout preserves the inconsistency, compares candidate datasets/resources,
    and recommends what should be constructed next.
    """

    def payload_to_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        return json.dumps(value, sort_keys=True)

    if isinstance(distiller_handoff, str):
        try:
            handoff_payload: Union[str, Dict[str, Any]] = json.loads(distiller_handoff)
        except json.JSONDecodeError:
            handoff_payload = distiller_handoff
    else:
        handoff_payload = distiller_handoff

    if isinstance(investigator_summary, str):
        try:
            investigator_payload: Any = json.loads(investigator_summary)
        except json.JSONDecodeError:
            investigator_payload = investigator_summary
    else:
        investigator_payload = investigator_summary

    normalized_domain = _normalize_token(domain)
    handoff_text = payload_to_text(handoff_payload)
    investigator_text = payload_to_text(investigator_payload)
    combined = _normalize_token(" ".join([handoff_text, investigator_text, constraints]))

    hypothesis_id = "dataset_scout_hypothesis"
    question = ""
    if isinstance(handoff_payload, dict):
        hypothesis_id = str(handoff_payload.get("hypothesis_id") or hypothesis_id)
        question = str(handoff_payload.get("question") or handoff_payload.get("distilled_hypothesis") or "")
    elif isinstance(handoff_payload, str):
        question = handoff_payload.strip()

    is_enhancer_case = any(
        token in combined
        for token in [
            "enhancers_types",
            "enhancer_subtype",
            "tissue_specific",
            "tissue_invariant",
            "weak_enhancer",
            "strong_enhancer",
            "motif_family",
        ]
    )

    if is_enhancer_case:
        inconsistency = (
            "Old enhancer-type motif support looked explanatory, but the revised "
            "coordinate-backed enhancer-type follow-up was non-explanatory. The "
            "next dataset search should ask what changed between old weak/strong "
            "enhancer construction and revised tissue-specific/tissue-invariant "
            "construction, and which external dataset can distinguish artifact "
            "from biology."
        )
        resource_classes = [
            {
                "resource_class": "current_and_revised_nt_enhancer_tasks",
                "examples": [
                    "InstaDeepAI/nucleotide_transformer_downstream_tasks/enhancers_types",
                    "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised/enhancers_types",
                ],
                "why_relevant": "Directly explains the old-versus-revised inconsistency and establishes what the current task can and cannot test.",
            },
            {
                "resource_class": "original_weak_strong_enhancer_sources",
                "examples": ["iEnhancer-2L source files", "weak/strong enhancer benchmark supplements"],
                "why_relevant": "Could recover whether old labels 1/2 have biological or synthetic/provenance structure.",
            },
            {
                "resource_class": "coordinate_backed_enhancer_annotations",
                "examples": ["ENCODE SCREEN cCRE", "FANTOM5 enhancers", "VISTA enhancers"],
                "why_relevant": "Can provide named enhancer context and coordinates independent of the old benchmark construction.",
            },
            {
                "resource_class": "chromatin_or_tissue_context",
                "examples": ["Roadmap/ENCODE chromatin states", "tissue/cell-type enhancer annotations"],
                "why_relevant": "Tests whether tissue or chromatin context, not motif-family support alone, drives generalization.",
            },
            {
                "resource_class": "functional_regulatory_assays",
                "examples": ["MPRA/STARR-seq enhancer activity datasets", "variant-effect regulatory datasets"],
                "why_relevant": "Can test functional regulatory generalization with labels less tied to old weak/strong benchmark construction.",
            },
        ]
        candidate_requirements = [
            "coordinates or stable source IDs",
            "named enhancer subtype or regulatory-context labels",
            "label provenance and class dictionary",
            "enough rows for label-balanced spectral levels",
            "compatibility with Caduceus sequence-window inputs",
            "ability to test at least one explanation for the old-vs-revised inconsistency",
        ]
    else:
        inconsistency = (
            "The current benchmark is insufficient or inconsistent with the live "
            "hypothesis. Scout public/local datasets that can distinguish artifact, "
            "proxy, and mechanism explanations before construction."
        )
        resource_classes = [
            {
                "resource_class": "source_benchmark_variants",
                "examples": ["original benchmark source", "revised benchmark", "raw source archives"],
                "why_relevant": "Explains whether the current inconsistency is caused by benchmark construction.",
            },
            {
                "resource_class": "independent_public_datasets",
                "examples": ["official public datasets with stable IDs and labels"],
                "why_relevant": "Tests whether the mechanism holds outside the current benchmark family.",
            },
            {
                "resource_class": "curated_annotation_resources",
                "examples": ["domain ontologies, reference annotations, metadata tables"],
                "why_relevant": "Adds mechanism-level labels or confounder controls.",
            },
        ]
        candidate_requirements = [
            "stable row identifiers",
            "documented labels",
            "provenance",
            "enough rows for controlled splits",
            "compatibility with the target model or baseline",
            "ability to distinguish at least two live hypotheses",
        ]

    return {
        "mode": "spectra_dataset_scout_request",
        "decision": "scout_before_construction",
        "analysis_complete": False,
        "domain": normalized_domain,
        "dataset_scout_role": (
            "Search, compare, and rank possible datasets/resources before any "
            "Dataset Constructor builds a package. Preserve inconsistencies; do "
            "not collapse to the first plausible dataset."
        ),
        "hypothesis_id": hypothesis_id,
        "question": question,
        "inconsistency_to_explain": inconsistency,
        "target_model_description": target_model_description.strip(),
        "constraints": constraints.strip(),
        "allowed_network": allowed_network,
        "min_candidates": max(1, int(min_candidates)),
        "candidate_resource_classes": resource_classes,
        "candidate_requirements": candidate_requirements,
        "scout_steps": [
            "Restate the inconsistency and live alternative explanations.",
            "Query the portable SPECTRA dataset catalog before general public search; record matching catalog IDs, scale defaults, authentication, and rejection reasons.",
            "Search local artifacts first: previous runs, loaders, cached metadata, source manifests, dataset cards, and downloaded public resources.",
            "Search public/canonical sources when network is allowed; record URLs, versions, licenses, file sizes, and whether data are downloadable or queryable.",
            "Build a candidate table with at least min_candidates when feasible; include rejected candidates and rejection reasons.",
            "Score each candidate by whether it can distinguish the inconsistency, not by convenience alone.",
            "Identify the best construction target and the best backup target, or explain why scouting must continue.",
            "Do not construct the dataset package unless explicitly promoted to Dataset Constructor.",
        ],
        "candidate_table_required_columns": [
            "candidate_id",
            "dataset_or_resource",
            "source_url_or_path",
            "local_or_public",
            "labels_available",
            "coordinates_or_stable_ids",
            "provenance_quality",
            "sample_size_estimate",
            "model_compatibility",
            "what_inconsistency_it_tests",
            "construction_feasibility",
            "known_blockers",
            "recommendation",
        ],
        "quality_gates": [
            "At least min_candidates are compared unless the scout proves fewer are available.",
            "The current/revised benchmark family is compared against independent alternatives.",
            "Every recommended candidate states exactly which inconsistency it can distinguish.",
            "Rejected candidates include concrete rejection reasons.",
            "The scout does not construct a final dataset package.",
        ],
        "required_outputs": [
            "dataset_scout_request_used.json",
            "inconsistency_ledger.json",
            "dataset_candidate_table.csv",
            "dataset_candidate_table.json",
            "candidate_resource_search_log.md",
            "candidate_resource_search_log.json",
            "rejected_candidates.md",
            "recommended_constructor_handoff.json",
            "scout_report.md",
        ],
    }


def distill_dataset_scout_output(
    candidate_table: Union[str, List[Dict[str, Any]], Dict[str, Any]],
    inconsistency_ledger: Optional[Union[str, List[Dict[str, Any]], Dict[str, Any]]] = None,
    scout_report: str = "",
    domain: str = "unknown",
    min_candidates: int = 3,
) -> Dict[str, Any]:
    """Distill Dataset Scout output into continue-scouting or construct decision."""
    rows = _normalize_rows(candidate_table)
    ledger_rows = _normalize_rows(inconsistency_ledger or []) if inconsistency_ledger else []
    normalized_domain = _normalize_token(domain)

    def truthy(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        return _normalize_token(str(value)) in {"true", "yes", "y", "1", "available", "present", "high", "medium", "good"}

    scored = []
    for row in rows:
        candidate_id = str(_first_present(row, ["candidate_id", "id", "dataset_or_resource", "dataset"]) or "candidate")
        labels = str(_first_present(row, ["labels_available", "labels", "label_quality"]) or "")
        coords = str(_first_present(row, ["coordinates_or_stable_ids", "coordinates", "stable_ids"]) or "")
        provenance = str(_first_present(row, ["provenance_quality", "provenance"]) or "")
        feasibility = str(_first_present(row, ["construction_feasibility", "feasibility"]) or "")
        tests = str(_first_present(row, ["what_inconsistency_it_tests", "tests_inconsistency", "question"]) or "")
        blockers = str(_first_present(row, ["known_blockers", "blockers"]) or "")
        recommendation = _normalize_token(str(_first_present(row, ["recommendation", "recommended_action"]) or ""))
        text = _normalize_token(" ".join([labels, coords, provenance, feasibility, tests, blockers, recommendation]))

        score = 0.0
        score += 2.0 if any(token in _normalize_token(labels) for token in ["named", "documented", "class", "label", "available"]) else 0.0
        score += 2.0 if any(token in _normalize_token(coords) for token in ["coordinate", "stable", "chrom", "bed", "id", "accession"]) else 0.0
        score += 2.0 if any(token in _normalize_token(provenance) for token in ["high", "good", "source", "row", "provenance", "version"]) else 0.0
        score += 2.0 if any(token in _normalize_token(feasibility) for token in ["high", "ready", "downloaded", "local", "bounded", "feasible"]) else 0.0
        score += 3.0 if tests.strip() else 0.0
        score -= 2.0 if any(token in text for token in ["blocked", "credential", "license_blocked", "too_large", "not_available"]) else 0.0
        if "top_construction_candidate" in recommendation or "top_candidate" in recommendation:
            score += 5.0
        elif any(token in recommendation for token in ["top", "primary", "construct", "promote"]):
            score += 2.0
        if any(token in recommendation for token in ["backup", "secondary"]):
            score -= 1.0
        if any(token in recommendation for token in ["comparator", "reject", "do_not", "not_promote", "not_top"]):
            score -= 6.0
        scored.append({"candidate_id": candidate_id, "score": score, "row": row})

    scored.sort(key=lambda item: item["score"], reverse=True)
    top = scored[0] if scored else None
    candidate_count = len(rows)
    unique_candidate_count = len({item["candidate_id"] for item in scored})
    evidence_of_search = candidate_count >= max(1, int(min_candidates))
    report_text = _normalize_token(scout_report)
    report_mentions_multiple_sources = sum(
        token in report_text
        for token in ["encode", "fantom", "vista", "roadmap", "mpra", "revised", "old", "jaspar", "screen"]
    ) >= 2
    ledger_has_inconsistency = bool(ledger_rows) or any(token in report_text for token in ["inconsistency", "old", "revised", "non_explanatory"])

    blocking_gaps = []
    if not evidence_of_search:
        blocking_gaps.append(
            "Scout compared only %d candidates; compare at least %d or prove fewer are available."
            % (candidate_count, max(1, int(min_candidates)))
        )
    if not report_mentions_multiple_sources:
        blocking_gaps.append("Scout report does not show comparison across multiple source families.")
    if not ledger_has_inconsistency:
        blocking_gaps.append("Inconsistency ledger is missing or does not preserve the old-vs-new question.")
    if top and any(
        token in _normalize_token(str(_first_present(top["row"], ["recommendation", "recommended_action"]) or ""))
        for token in ["comparator", "reject", "do_not", "not_promote", "not_top"]
    ):
        blocking_gaps.append("Top-scoring candidate is marked comparator/rejected; Scout ranking must identify a constructable primary candidate.")
    if not top or top["score"] < 6.0:
        blocking_gaps.append("No candidate is strong enough for construction.")

    if blocking_gaps:
        decision = "continue_scouting"
        required_next_action = (
            "Return to Dataset Scout. The output did not yet compare enough candidates "
            "or preserve the inconsistency strongly enough to justify construction."
        )
    else:
        decision = "promote_to_dataset_constructor"
        required_next_action = (
            "Promote the top candidate to Dataset Constructor. Construct a package "
            "for the candidate that best distinguishes the inconsistency, while "
            "keeping backup candidates in the ledger."
        )

    top_row = top["row"] if top else {}
    backup_rows = [item["row"] for item in scored[1:4]]
    return {
        "mode": "spectra_dataset_scout_distiller",
        "decision": decision,
        "analysis_complete": False,
        "domain": normalized_domain,
        "candidate_count": candidate_count,
        "unique_candidate_count": unique_candidate_count,
        "scored_candidates": [
            {
                "candidate_id": item["candidate_id"],
                "score": item["score"],
                "dataset_or_resource": _first_present(item["row"], ["dataset_or_resource", "dataset", "source_url_or_path"]),
                "recommendation": _first_present(item["row"], ["recommendation", "recommended_action"]),
            }
            for item in scored
        ],
        "blocking_gaps": blocking_gaps,
        "required_next_action": required_next_action,
        "constructor_handoff": {
            "enabled": decision == "promote_to_dataset_constructor",
            "top_candidate": top_row,
            "backup_candidates": backup_rows,
            "constructor_instruction": (
                "Build the top candidate as a SPECTRA-ready hypothesis-test dataset. "
                "Do not ignore the inconsistency ledger; the package should test why "
                "the old benchmark and revised/alternative datasets differ."
            ),
        },
        "continue_scouting_request": {
            "enabled": decision == "continue_scouting",
            "required_fixes": blocking_gaps,
        },
        "what_not_to_do": [
            "Do not construct from the first plausible dataset without comparing alternatives.",
            "Do not discard the old-versus-revised inconsistency when ranking candidates.",
            "Do not treat a packaging-ready dataset as scientifically best unless it tests the live inconsistency.",
        ],
        "required_outputs": [
            "dataset_scout_distillation.json",
            "constructor_handoff.json" if decision == "promote_to_dataset_constructor" else "continue_scouting_request.json",
        ],
    }


def prepare_dataset_constructor_request(
    distiller_handoff: Union[str, Dict[str, Any]],
    investigator_summary: Optional[Union[str, List[Dict[str, Any]], Dict[str, Any]]] = None,
    domain: str = "unknown",
    target_model_description: str = "",
    constraints: str = "",
    allowed_network: bool = True,
) -> Dict[str, Any]:
    """Prepare a Dataset Constructor handoff from a Distiller conclusion.

    The Dataset Constructor is a separate SPECTRA role. It does not decide
    whether a model generalizes; it constructs or recovers the dataset needed to
    test the current distilled hypothesis when the existing benchmark is not
    sufficient.
    """

    def payload_to_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        return json.dumps(value, sort_keys=True)

    if isinstance(distiller_handoff, str):
        try:
            handoff_payload: Union[str, Dict[str, Any]] = json.loads(distiller_handoff)
        except json.JSONDecodeError:
            handoff_payload = distiller_handoff
    else:
        handoff_payload = distiller_handoff

    if isinstance(investigator_summary, str):
        try:
            investigator_payload: Any = json.loads(investigator_summary)
        except json.JSONDecodeError:
            investigator_payload = investigator_summary
    else:
        investigator_payload = investigator_summary

    normalized_domain = _normalize_token(domain)
    handoff_text = payload_to_text(handoff_payload)
    investigator_text = payload_to_text(investigator_payload)
    combined = _normalize_token(" ".join([handoff_text, investigator_text, constraints]))

    hypothesis_id = "dataset_construction_hypothesis"
    question = ""
    if isinstance(handoff_payload, dict):
        hypothesis_id = str(handoff_payload.get("hypothesis_id") or hypothesis_id)
        question = str(handoff_payload.get("question") or handoff_payload.get("distilled_hypothesis") or "")
    elif isinstance(handoff_payload, str):
        question = handoff_payload.strip()

    is_enhancer_case = any(
        token in combined
        for token in ["enhancers_types", "enhancer_subtype", "tissue_specific", "tissue_invariant", "weak_enhancer", "strong_enhancer"]
    )
    is_sequence_domain = is_enhancer_case or any(
        token in normalized_domain
        for token in ["dna", "rna", "nucleotide", "sequence", "genomic", "regulatory", "caduceus"]
    )
    provenance_limited = any(
        token in combined
        for token in ["label_construction", "provenance", "source_name", "coordinate", "masked", "same_sequence_set", "not_sufficient"]
    )

    if is_enhancer_case:
        construction_goal = (
            "Construct a coordinate-backed enhancer subtype hypothesis-test dataset "
            "with named class semantics, row-level provenance, leakage checks, and "
            "SPECTRA-ready split candidates for testing Caduceus generalization across "
            "enhancer regulatory context."
        )
        dataset_spec = {
            "scientific_unit": "coordinate-backed regulatory DNA sequence interval",
            "required_row_fields": [
                "row_id",
                "sequence",
                "chrom",
                "start",
                "end",
                "assembly",
                "label_id",
                "label_name",
                "source_dataset",
                "source_split",
                "source_record_id_or_header",
                "provenance_notes",
            ],
            "desired_labels": [
                "none/non-enhancer",
                "named enhancer subtype such as weak/strong or tissue-specific/tissue-invariant",
            ],
            "candidate_public_or_local_resources": [
                "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised enhancer-type files",
                "old nucleotide_transformer_downstream_tasks enhancer-type files for comparison when provenance is recoverable",
                "JASPAR or another versioned motif matrix collection",
                "ENCODE SCREEN cCRE or other coordinate-backed regulatory annotations when mappable",
                "reference genome metadata only if sequence extraction or coordinate validation is required",
            ],
            "split_candidates": [
                "chromosome-held-out splits",
                "source/provenance-held-out splits if source IDs are recoverable",
                "motif-family support spectral splits",
                "residual motif-family support after matching k-mer, GC/CpG, and length",
                "coordinate/context splits such as cCRE class, distance to TSS, or chromatin context if annotations are mappable",
            ],
            "confounder_controls": [
                "label balance",
                "sequence length",
                "GC/CpG composition",
                "k-mer support",
                "chromosome/source split leakage",
                "duplicate and reverse-complement duplicate leakage",
            ],
        }
        current_benchmark_limitation = (
            "The old enhancers_types benchmark masks row provenance and coordinates, "
            "shares its sequence universe with binary enhancers, and has label-coded "
            "source fields; it cannot by itself support a biological enhancer-subtype claim."
        )
    elif is_sequence_domain:
        construction_goal = (
            "Construct a coordinate-backed sequence hypothesis-test dataset with "
            "named labels, stable row identifiers, mappable annotations, and SPECTRA-ready splits."
        )
        dataset_spec = {
            "scientific_unit": "sequence interval or sequence-centered biological unit",
            "required_row_fields": [
                "row_id",
                "sequence",
                "label_id",
                "label_name",
                "source_dataset",
                "source_record_id",
                "provenance_notes",
            ],
            "desired_labels": ["documented biological or mechanism labels"],
            "candidate_public_or_local_resources": [
                "source benchmark metadata",
                "reference sequence resources",
                "curated annotations or ontologies",
                "versioned motif or regulatory resources where relevant",
            ],
            "split_candidates": [
                "source/provenance-held-out splits",
                "mechanism-support spectral splits",
                "surface-proxy matched residual splits",
            ],
            "confounder_controls": ["class balance", "length", "composition", "duplicates", "source leakage"],
        }
        current_benchmark_limitation = (
            "The current benchmark lacks the provenance, identifiers, labels, or annotations "
            "needed to test the distilled mechanism hypothesis cleanly."
        )
    else:
        construction_goal = (
            "Construct a hypothesis-test dataset from public/local resources that can "
            "distinguish the distilled mechanism hypothesis from artifact or proxy explanations."
        )
        dataset_spec = {
            "scientific_unit": "domain sample with stable provenance",
            "required_row_fields": [
                "row_id",
                "features_or_raw_input",
                "label_id",
                "label_name",
                "source_dataset",
                "source_record_id",
                "provenance_notes",
            ],
            "desired_labels": ["documented non-circular mechanism or outcome labels"],
            "candidate_public_or_local_resources": [
                "source benchmark metadata",
                "official public datasets",
                "curated annotations or ontologies",
                "local raw data with stable IDs",
            ],
            "split_candidates": [
                "source/provenance-held-out splits",
                "mechanism-support spectral splits",
                "proxy-matched residual splits",
            ],
            "confounder_controls": ["class balance", "known source or batch variables", "duplicate leakage"],
        }
        current_benchmark_limitation = (
            "The current benchmark cannot test the distilled mechanism hypothesis with sufficient provenance and controls."
        )

    if not provenance_limited:
        current_benchmark_limitation = (
            current_benchmark_limitation
            + " The constructor must verify this limitation from the actual artifacts before replacing or extending the benchmark."
        )

    constructor_steps = [
        "Read the Distiller handoff and Investigator artifacts; restate the dataset-construction hypothesis.",
        "Inspect local benchmark files, loaders, dataset cards, cache metadata, and source repositories before downloading new data.",
        "Recover label semantics and row-level provenance when possible; otherwise record exactly what is masked or missing.",
        "Acquire or construct only the public/local resources needed for the hypothesis-test dataset.",
        "Build a SPECTRA-ready table with stable row IDs, labels, provenance, raw inputs, and split metadata.",
        "Validate mappings, label semantics, coordinate parsing, duplicate leakage, reverse-complement leakage when relevant, and class balance.",
        "Create split candidates that directly distinguish the distilled hypothesis from artifact/proxy explanations.",
        "Return the dataset package to the Investigator with a dataset card, schema, leakage audit, and recommended first SPECTRA run.",
    ]

    decision = "construct_dataset_request"
    if not allowed_network and "public" in _normalize_token(" ".join(dataset_spec["candidate_public_or_local_resources"])):
        decision = "construct_from_local_or_report_needed_public_resources"

    return {
        "mode": "spectra_dataset_constructor_request",
        "decision": decision,
        "analysis_complete": False,
        "domain": normalized_domain,
        "dataset_constructor_role": (
            "Recover or construct the dataset needed to test a distilled SPECTRA "
            "hypothesis. Do not make the model-generalization claim; return a "
            "validated dataset package and recommended SPECTRA splits."
        ),
        "hypothesis_id": hypothesis_id,
        "question": question,
        "construction_goal": construction_goal,
        "current_benchmark_limitation": current_benchmark_limitation,
        "target_model_description": target_model_description.strip(),
        "constraints": constraints.strip(),
        "allowed_network": allowed_network,
        "dataset_spec": dataset_spec,
        "constructor_steps": constructor_steps,
        "quality_gates": [
            "Labels have documented semantics and are not inferred from model behavior.",
            "Every row has a stable row ID and explicit provenance.",
            "Coordinates or source identifiers are parsed and validated when the hypothesis needs them.",
            "Duplicate and reverse-complement leakage are audited when the data are sequences.",
            "Known proxy variables are available for controls or listed as unresolved limitations.",
            "Split candidates are prospective and do not use target-model errors.",
            "The package clearly separates original benchmark evidence from constructed hypothesis-test evidence.",
        ],
        "required_outputs": [
            "dataset_card.md",
            "construction_manifest.json",
            "label_semantics.json",
            "provenance_table.csv",
            "sequence_table.parquet",
            "spectra_ready_schema.json",
            "split_candidates/",
            "mapping_validation.json",
            "leakage_audit.json",
            "confounder_audit.json",
            "recommended_spectra_run.json",
            "constructor_report.md",
        ],
        "handoff_back_to_investigator": {
            "must_include": [
                "dataset_card.md",
                "spectra_ready_schema.json",
                "split candidate descriptions and files",
                "label/provenance confidence",
                "leakage and confounder audits",
                "recommended first SPECTRA experiment",
            ],
            "claim_scope": (
                "This constructed dataset tests the distilled hypothesis prospectively. "
                "It does not retroactively fix provenance limitations in the original benchmark."
            ),
        },
    }


def distill_dataset_constructor_output(
    construction_manifest: Union[str, Dict[str, Any]],
    mapping_validation: Optional[Union[str, Dict[str, Any]]] = None,
    leakage_audit: Optional[Union[str, Dict[str, Any]]] = None,
    label_semantics: Optional[Union[str, Dict[str, Any]]] = None,
    confounder_audit: Optional[Union[str, Dict[str, Any]]] = None,
    recommended_spectra_run: Optional[Union[str, Dict[str, Any]]] = None,
    domain: str = "unknown",
    focus: str = "",
) -> Dict[str, Any]:
    """Distill a Dataset Constructor package into the next routing decision.

    This is the Distiller's gate after dataset construction. It decides whether
    the constructed package is ready for an Investigator SPECTRA run, should go
    back to the Dataset Constructor, or is blocked by missing provenance.
    """
    manifest = _coerce_json_payload(construction_manifest) if isinstance(construction_manifest, str) else construction_manifest
    if not isinstance(manifest, dict):
        raise ValueError("construction_manifest must decode to a JSON object")

    mapping = (
        _coerce_json_payload(mapping_validation)
        if isinstance(mapping_validation, str)
        else (mapping_validation or {})
    )
    leakage = (
        _coerce_json_payload(leakage_audit)
        if isinstance(leakage_audit, str)
        else (leakage_audit or {})
    )
    labels = (
        _coerce_json_payload(label_semantics)
        if isinstance(label_semantics, str)
        else (label_semantics or {})
    )
    confounders = (
        _coerce_json_payload(confounder_audit)
        if isinstance(confounder_audit, str)
        else (confounder_audit or {})
    )
    recommended = (
        _coerce_json_payload(recommended_spectra_run)
        if isinstance(recommended_spectra_run, str)
        else (recommended_spectra_run or {})
    )
    if not all(isinstance(item, dict) for item in [mapping, leakage, labels, confounders, recommended]):
        raise ValueError("dataset-constructor artifacts must decode to JSON objects")

    normalized_domain = _normalize_token(domain)
    artifacts = dict(manifest.get("artifacts") or manifest.get("main_artifacts") or {})
    if manifest.get("main_artifacts"):
        alias_map = {
            "dataset_card": ["dataset_card", "report", "limitations"],
            "construction_manifest": ["construction_manifest", "manifest"],
            "label_semantics": ["label_semantics", "labels"],
            "provenance_table": ["provenance_table", "provenance", "full_coordinate_index"],
            "sequence_table": ["sequence_table", "sequence_or_coordinate_table"],
            "spectra_ready_schema": ["spectra_ready_schema", "schema"],
            "mapping_validation": ["mapping_validation"],
            "leakage_audit": ["leakage_audit"],
            "confounder_audit": ["confounder_audit"],
            "recommended_spectra_run": ["recommended_spectra_run"],
        }
        for canonical, aliases in alias_map.items():
            if artifacts.get(canonical):
                continue
            for alias in aliases:
                if artifacts.get(alias):
                    artifacts[canonical] = artifacts[alias]
                    break
        artifacts.setdefault("construction_manifest", "constructed_dataset_manifest.json")

    primary_dataset = dict(manifest.get("primary_dataset") or {})
    records = manifest.get("records") or {}
    if not primary_dataset and records:
        primary_dataset = {
            "name": manifest.get("dataset_name") or "constructed_dataset",
            "rows": records.get("bounded_sequence_records")
            or records.get("bounded_coordinate_records")
            or records.get("rows")
            or 0,
            "coordinate_parse_fraction": 1.0
            if records.get("full_registry_coordinate_records") or records.get("bounded_sequence_records")
            else 0.0,
            "assembly_status": manifest.get("assembly_status") or "documented_by_constructor",
        }
    primary_input_files = manifest.get("primary_input_files") or {}
    motif_resources = manifest.get("motif_resources") or {}

    required_artifacts = [
        "dataset_card",
        "construction_manifest",
        "label_semantics",
        "provenance_table",
        "sequence_table",
        "spectra_ready_schema",
        "mapping_validation",
        "leakage_audit",
        "confounder_audit",
        "recommended_spectra_run",
    ]
    missing_artifacts = [key for key in required_artifacts if not artifacts.get(key)]

    rows = int(primary_dataset.get("rows") or 0)
    coordinate_parse_fraction = float(
        mapping.get("coordinate_parse_fraction")
        or primary_dataset.get("coordinate_parse_fraction")
        or (
            (
                float(
                    (mapping.get("sequence_extraction") or {}).get("sample_records_with_sequence")
                    or records.get("bounded_sequence_records_with_sequence")
                    or 0
                )
                / max(
                    1.0,
                    float(
                        records.get("bounded_sequence_records")
                        or (mapping.get("bounded_sample_rule") or {}).get("n")
                        or 0
                    ),
                )
            )
            if (mapping.get("sequence_extraction") or records)
            else 0.0
        )
    )
    required_fields = mapping.get("required_fields_present") or {}
    if not required_fields and artifacts.get("sequence_table") and rows > 0:
        required_fields = {
            "row_id": True,
            "sequence_or_coordinates": True,
            "label": bool(labels.get("labels") or labels.get("classes")),
            "provenance": bool(artifacts.get("provenance_table")),
            "split_metadata": bool(manifest.get("split_candidates")),
        }
    missing_required_fields = [
        field for field, present in required_fields.items() if not present
    ]
    primary_validated = bool(
        mapping.get("primary_dataset_validated")
        or manifest.get("ready_for_investigator")
        or (
            rows > 0
            and coordinate_parse_fraction >= 0.95
            and bool(artifacts.get("sequence_table"))
            and bool(artifacts.get("mapping_validation"))
        )
    )

    cross_exact = int(leakage.get("cross_source_split_exact_duplicate_rows") or 0)
    cross_rc = int(leakage.get("cross_source_split_reverse_complement_duplicate_rows") or 0)
    cross_coord = int(leakage.get("cross_source_split_coordinate_duplicate_rows") or 0)
    overlap_info = leakage.get("coordinate_overlap") or {}
    cross_overlap = int(overlap_info.get("cross_source_split_overlapping_interval_pairs") or 0)

    label_entries = labels.get("labels") or {}
    if not label_entries and isinstance(labels.get("classes"), dict):
        label_entries = {
            label_id: {"label_name": label_name, "semantic_confidence": "documented"}
            for label_id, label_name in labels.get("classes", {}).items()
        }
    missing_label_names = [
        str(label_id)
        for label_id, entry in label_entries.items()
        if not isinstance(entry, dict) or not entry.get("label_name")
    ]
    label_caveats = []
    if labels.get("loader_proven_class_dictionary_available") is False:
        label_caveats.append("Label semantics are not loader-dictionary-proven.")
    for label_id, entry in label_entries.items():
        confidence = _normalize_token(str(entry.get("semantic_confidence", ""))) if isinstance(entry, dict) else ""
        if "inferred" in confidence or "medium" in confidence:
            label_caveats.append(f"Label {label_id} semantics are inferred or medium-confidence.")

    assembly_status = _normalize_token(str(mapping.get("assembly_status") or primary_dataset.get("assembly_status") or ""))
    nonblocking_caveats = []
    if "unknown" in assembly_status or "not_loader_proven" in assembly_status:
        nonblocking_caveats.append("Genome assembly is not loader-proven.")
    nonblocking_caveats.extend(sorted(set(label_caveats)))
    if int(leakage.get("exact_duplicate_rows") or 0) or int(leakage.get("reverse_complement_duplicate_rows") or 0):
        nonblocking_caveats.append("Within-split exact/reverse-complement duplicates exist; run duplicate-collapsed sensitivity.")
    if "ccre" in normalized_domain and "biosample" not in _normalize_token(json.dumps(manifest, sort_keys=True)):
        nonblocking_caveats.append(
            "Per-biosample regulatory activity was not joined; context-support claims are limited to constructed cCRE context features."
        )
    for note in confounders.get("confounder_notes") or []:
        nonblocking_caveats.append(str(note))

    first_run = recommended.get("first_run") or {}
    first_split = recommended.get("recommended_first_split_file") or first_run.get("primary_split_file")
    first_axis = recommended.get("recommended_first_axis") or first_run.get("primary_spectra_axis")
    secondary_axis = recommended.get("secondary_axis") or {}
    secondary_runs = recommended.get("secondary_runs") or []
    if not secondary_axis and secondary_runs:
        secondary_axis = {"name": str(secondary_runs[0]), "split_file": ""}
    test_levels = recommended.get("test_levels") or []
    if not test_levels and first_split and manifest.get("split_candidates"):
        for split_info in manifest.get("split_candidates", {}).values():
            if split_info.get("path") == first_split or split_info.get("recommended"):
                test_levels = list((split_info.get("split_counts") or {}).keys())
                break
    has_recommended_run = bool(first_split and first_axis and len(test_levels) >= 3)
    motif_axis_required = bool(first_axis and "motif" in str(first_axis).lower())
    has_motif_features = (not motif_axis_required) or bool(motif_resources.get("available"))

    blocking_gaps = []
    if missing_artifacts:
        blocking_gaps.append("Missing required artifacts: %s" % ", ".join(missing_artifacts))
    if rows <= 0:
        blocking_gaps.append("Primary dataset has no rows.")
    if not primary_validated:
        blocking_gaps.append("Mapping validation did not mark the primary dataset validated.")
    if coordinate_parse_fraction < 0.95 and any(token in normalized_domain for token in ["dna", "genomic", "regulatory", "sequence", "enhancer"]):
        blocking_gaps.append("Coordinate parse fraction is too low for coordinate-backed regulatory testing.")
    if missing_required_fields:
        blocking_gaps.append("Missing required validated fields: %s" % ", ".join(missing_required_fields))
    if missing_label_names:
        blocking_gaps.append("Missing label names for labels: %s" % ", ".join(missing_label_names))
    if cross_exact or cross_rc or cross_coord or cross_overlap:
        blocking_gaps.append(
            "Cross-source leakage is nonzero: exact=%d rc=%d coordinate_duplicates=%d interval_overlaps=%d"
            % (cross_exact, cross_rc, cross_coord, cross_overlap)
        )
    if not has_recommended_run:
        blocking_gaps.append("No usable recommended first SPECTRA run with at least three levels.")
    if motif_axis_required and not has_motif_features:
        blocking_gaps.append("Recommended motif axis lacks usable motif resources or feature evidence.")

    if blocking_gaps:
        decision = "return_to_dataset_constructor"
        required_next_action = (
            "Return the package to the Dataset Constructor to resolve blocking validation gaps before "
            "an Investigator spends compute on model behavior."
        )
    else:
        decision = "handoff_to_investigator"
        followup_clause = (
            "run the secondary residual motif axis if the first curve is non-explanatory"
            if motif_axis_required
            else "run the secondary/control axis if the first curve is non-explanatory or confounded"
        )
        required_next_action = (
            "Send the package to the Investigator. Run the recommended first SPECTRA split candidate, "
            "validate measured overlap before interpreting metrics, fit fresh Caduceus probes per split, "
            f"and {followup_clause}."
        )

    readiness_checks = {
        "required_artifacts_present": not missing_artifacts,
        "rows_positive": rows > 0,
        "primary_dataset_validated": primary_validated,
        "coordinate_parse_fraction": coordinate_parse_fraction,
        "required_fields_validated": not missing_required_fields,
        "label_names_present": not missing_label_names,
        "cross_source_leakage_zero": not (cross_exact or cross_rc or cross_coord or cross_overlap),
        "recommended_run_available": has_recommended_run,
        "motif_resources_available": has_motif_features,
    }
    readiness_score = sum(1 for passed in readiness_checks.values() if passed is True) / len(readiness_checks)

    package_dir = manifest.get("output_directory")
    if not package_dir:
        for artifact_path in [
            artifacts.get("sequence_table"),
            artifacts.get("spectra_ready_schema"),
            artifacts.get("recommended_spectra_run"),
        ]:
            if artifact_path:
                parent = os.path.dirname(str(artifact_path))
                package_dir = (
                    os.path.dirname(parent)
                    if os.path.basename(parent) in {"processed", "split_candidates"}
                    else parent
                )
                break

    primary_dataset_name = primary_dataset.get("name") or manifest.get("dataset_name") or "constructed_dataset"
    target_question = (
        "Does target-model performance degrade along %s on %s after leakage, overlap, and proxy-control checks pass?"
        % (first_axis or "the recommended SPECTRA axis", primary_dataset_name)
    )
    required_investigator_steps = [
        "Validate measured train-test overlap decreases across spectral levels.",
        "Fit fresh frozen-Caduceus probes or the declared trainable component per split.",
        "Report the curve as non-explanatory if performance improves or stays flat as support decreases.",
    ]
    if int(leakage.get("exact_duplicate_rows") or 0) or int(leakage.get("reverse_complement_duplicate_rows") or 0):
        required_investigator_steps.append("Run duplicate-collapsed sensitivity because within-split duplicates exist.")
    if motif_axis_required:
        required_investigator_steps.append("If motif-family support is non-explanatory, run the residual motif-support split candidate.")
    elif secondary_axis:
        required_investigator_steps.append("Run the secondary/control split candidate to distinguish regulatory-context effects from leakage or surface-composition effects.")
    required_investigator_steps.append("Keep old unrevised enhancers_types claims separate from this constructed coordinate-backed dataset.")

    constructor_followup_queue = [
        "Optionally join per-biosample SCREEN, Roadmap, or FANTOM activity/context annotations before making tissue-specific regulatory-context claims.",
        "Create stricter non-overlapping-window split variants if the Investigator finds cross-split window overlap affects the recommended split.",
        "Add motif or k-mer control features only as controls against the regulatory-context hypothesis, not as the primary mechanism.",
    ]

    return {
        "mode": "spectra_dataset_package_distiller",
        "decision": decision,
        "analysis_complete": False,
        "domain": normalized_domain,
        "focus": focus.strip(),
        "required_next_action": required_next_action,
        "readiness_score": readiness_score,
        "readiness_checks": readiness_checks,
        "blocking_gaps": blocking_gaps,
        "nonblocking_caveats": sorted(set(nonblocking_caveats)),
        "distilled_assessment": (
            "The constructed package is sufficient for a bounded Investigator run, "
            "but its caveats constrain claim strength."
            if not blocking_gaps
            else "The constructed package is not yet ready for an Investigator run."
        ),
        "handoff_to_investigator": {
            "enabled": decision == "handoff_to_investigator",
            "dataset_package": package_dir,
            "primary_table": artifacts.get("sequence_table"),
            "primary_dataset_name": primary_dataset_name,
            "rows": rows,
            "recommended_first_axis": first_axis,
            "recommended_first_split_file": first_split,
            "secondary_axis": secondary_axis.get("name"),
            "secondary_split_file": secondary_axis.get("split_file"),
            "target_question": target_question,
            "required_investigator_steps": required_investigator_steps,
        },
        "return_to_constructor": {
            "enabled": decision == "return_to_dataset_constructor",
            "required_fixes": blocking_gaps,
        },
        "constructor_followup_queue": constructor_followup_queue,
        "what_not_to_do": [
            "Do not keep constructing indefinitely before the first bounded Investigator run if only nonblocking caveats remain.",
            "Do not claim label 1/2 biological semantics are loader-proven.",
            "Do not mix old unrevised provenance-limited rows into the primary coordinate-backed dataset.",
            "Do not interpret model metrics until measured overlap validation passes for the split candidate.",
        ],
        "required_outputs": [
            "dataset_package_distillation.json",
            "dataset_package_distiller_report.md",
            "investigator_dataset_handoff.json" if decision == "handoff_to_investigator" else "dataset_constructor_revision_request.json",
        ],
    }


def recommend_spectral_property(
    domain: str,
    dataset_description: str,
    candidate_property: str = "",
    property_type: str = "auto",
) -> Dict[str, Any]:
    """Recommend and validate a spectral property for generalizability analysis."""
    normalized_domain = _normalize_token(domain)
    normalized_type = _normalize_token(property_type)
    candidate = candidate_property.strip()

    recommendations = {
        "small_molecule": {
            "property": "Morgan fingerprint Tanimoto similarity",
            "graph_type": "weighted",
            "fallbacks": [
                "binary Tanimoto threshold",
                "scaffold or chemotype membership",
            ],
            "risk": "A fingerprint similarity split may not control scaffold leakage unless explicitly checked.",
        },
        "protein_sequence": {
            "property": "sequence identity or alignment score",
            "graph_type": "weighted",
            "fallbacks": [
                "binary identity threshold",
                "k-mer similarity",
                "validated protein embedding similarity",
            ],
            "risk": "Embedding similarity should not replace homology checks unless validated for the task.",
        },
        "single_cell_perturbation": {
            "property": "perturbation target, pathway, or response similarity",
            "graph_type": "binary_or_weighted",
            "fallbacks": [
                "shared pathway membership",
                "expression response similarity",
                "perturbation embedding similarity",
            ],
            "risk": "Cell-type, donor, and batch leakage may need separate controls.",
        },
    }

    default = {
        "property": candidate or "task-specific sample similarity",
        "graph_type": "weighted" if normalized_type == "auto" else normalized_type,
        "fallbacks": [
            "expert-defined binary similarity",
            "validated learned embedding similarity",
            "metadata-derived similarity with leakage checks",
        ],
        "risk": "The property must be justified as a plausible source of model failure.",
    }
    recommendation = recommendations.get(normalized_domain, default)
    chosen_property = candidate or recommendation["property"]
    graph_type = (
        recommendation["graph_type"]
        if normalized_type == "auto"
        else normalized_type
    )

    warnings = []
    if not dataset_description.strip():
        warnings.append("Dataset description is empty, so the recommendation is generic.")
    if normalized_type not in {"auto", "binary", "weighted", "binary_or_weighted"}:
        warnings.append(
            "property_type should usually be auto, binary, weighted, or binary_or_weighted."
        )
    if candidate and len(candidate.split()) < 3:
        warnings.append("Candidate property is terse; require a stronger scientific rationale.")

    return {
        "domain": normalized_domain,
        "recommended_property": chosen_property,
        "recommended_graph_type": graph_type,
        "fallback_properties": recommendation["fallbacks"],
        "scientific_rationale_required": True,
        "validation_questions": [
            "Does this property capture a known or plausible deployment shift?",
            "Should the graph be binary, weighted, or both?",
            "What evidence shows that lower similarity should make the task harder?",
            "Which leakage modes are not captured by this property?",
        ],
        "warnings": warnings + [recommendation["risk"]],
    }


def plan_similarity_computation(
    dataset_size: int,
    property_type: str,
    exact_required: bool = False,
    available_memory_gb: float = 16.0,
) -> Dict[str, Any]:
    """Plan an efficient spectral property graph construction strategy."""
    n = _as_int(dataset_size, "dataset_size")
    if n < 2:
        raise ValueError("dataset_size must be at least 2")
    pairs = n * (n - 1) // 2
    normalized_type = _normalize_token(property_type)
    bytes_per_edge = 1 if normalized_type == "binary" else 2
    estimated_storage_gb = pairs * bytes_per_edge / 1_000_000_000

    if exact_required or n <= 20_000:
        method = "exact_pairwise"
        notes = [
            "Use exact all-pairs similarity.",
            "Chunk computation if the pair count does not fit memory.",
        ]
    elif n <= 250_000:
        method = "chunked_pairwise_or_sparse_index"
        notes = [
            "Use chunked pairwise computation when exactness matters.",
            "Prefer sparse thresholded storage for binary graphs.",
        ]
    else:
        method = "approximate_or_domain_indexed"
        notes = [
            "Use approximate nearest neighbors, hashing, sketches, or domain-specific indexes.",
            "Report approximation limits because they affect split interpretation.",
        ]

    warnings = []
    if estimated_storage_gb > available_memory_gb:
        warnings.append(
            "Estimated flattened adjacency storage exceeds available memory; use chunking or sparse/approximate storage."
        )
    if pairs > 100_000_000 and method == "exact_pairwise":
        warnings.append("Exact all-pairs computation may be slow; schedule chunked jobs.")

    return {
        "dataset_size": n,
        "pair_count": pairs,
        "property_type": normalized_type,
        "estimated_flattened_adjacency_gb": round(estimated_storage_gb, 3),
        "recommended_method": method,
        "notes": notes,
        "warnings": warnings,
    }


def validate_split_stats(
    split_stats: Union[str, List[Dict[str, Any]], Dict[str, Any]],
    allow_minor_nonmonotonicity: bool = True,
    min_train_size: int = 10,
    min_test_size: int = 10,
) -> Dict[str, Any]:
    """Validate whether generated splits satisfy the SPECTRA overlap contract."""
    rows = _normalize_rows(split_stats)
    if len(rows) < 3:
        return {
            "valid": False,
            "status": "insufficient_split_levels",
            "reason": "At least three split difficulty levels are required.",
            "row_count": len(rows),
            "warnings": [],
        }

    normalized: List[Dict[str, Any]] = []
    warnings = []
    for row in rows:
        spectral_parameter = _as_float(
            _first_present(row, ("SPECTRA_parameter", "spectral_parameter", "sp")),
            "SPECTRA_parameter",
        )
        overlap = _as_float(
            _first_present(row, ("cross_split_overlap", "overlap", "css")),
            "cross_split_overlap",
        )
        train_size = _as_int(
            _first_present(row, ("train_size", "n_train")),
            "train_size",
        )
        test_size = _as_int(
            _first_present(row, ("test_size", "n_test")),
            "test_size",
        )
        if train_size < min_train_size:
            warnings.append(
                f"Train size {train_size} at spectral parameter {spectral_parameter} is below minimum {min_train_size}."
            )
        if test_size < min_test_size:
            warnings.append(
                f"Test size {test_size} at spectral parameter {spectral_parameter} is below minimum {min_test_size}."
            )
        normalized.append(
            {
                "spectral_parameter": spectral_parameter,
                "cross_split_overlap": overlap,
                "train_size": train_size,
                "test_size": test_size,
            }
        )

    normalized.sort(key=lambda row: row["spectral_parameter"])
    increases = []
    decreases = 0
    comparisons = 0
    for previous, current in zip(normalized, normalized[1:]):
        delta = current["cross_split_overlap"] - previous["cross_split_overlap"]
        comparisons += 1
        if delta <= 0:
            decreases += 1
        else:
            increases.append(
                {
                    "from_spectral_parameter": previous["spectral_parameter"],
                    "to_spectral_parameter": current["spectral_parameter"],
                    "overlap_increase": delta,
                }
            )

    monotonic = len(increases) == 0
    mostly_monotonic = comparisons > 0 and decreases / comparisons >= 0.75
    valid = monotonic or (allow_minor_nonmonotonicity and mostly_monotonic)
    if increases:
        warnings.append("Cross-split overlap increases at one or more adjacent spectral parameters.")

    status = "valid"
    if not valid:
        status = "overlap_contract_failed"
    elif not monotonic:
        status = "valid_with_minor_nonmonotonicity"

    return {
        "valid": valid,
        "status": status,
        "monotonic_nonincreasing": monotonic,
        "mostly_monotonic": mostly_monotonic,
        "decreasing_fraction": decreases / comparisons if comparisons else 0.0,
        "normalized_stats": normalized,
        "overlap_increases": increases,
        "warnings": warnings,
    }


def review_generalizability_report(
    metrics: Union[str, List[Dict[str, Any]], Dict[str, Any]],
    overlap_key: str = "cross_split_overlap",
    performance_key: str = "performance",
    higher_is_better: bool = True,
) -> Dict[str, Any]:
    """Review model metrics against measured overlap and flag report gaps."""
    rows = _normalize_rows(metrics)
    if len(rows) < 3:
        return {
            "status": "insufficient_metric_rows",
            "valid": False,
            "reason": "At least three evaluated split levels are required.",
            "warnings": [],
        }

    values: List[Tuple[float, float]] = []
    warnings = []
    for row in rows:
        overlap = _as_float(row.get(overlap_key), overlap_key)
        performance = _as_float(row.get(performance_key), performance_key)
        values.append((overlap, performance))

    mean_overlap = sum(overlap for overlap, _ in values) / len(values)
    mean_performance = sum(performance for _, performance in values) / len(values)
    covariance = sum(
        (overlap - mean_overlap) * (performance - mean_performance)
        for overlap, performance in values
    )
    overlap_variance = sum((overlap - mean_overlap) ** 2 for overlap, _ in values)
    performance_variance = sum(
        (performance - mean_performance) ** 2 for _, performance in values
    )
    if overlap_variance == 0 or performance_variance == 0:
        correlation = None
        warnings.append("Correlation is undefined because overlap or performance is constant.")
    else:
        correlation = covariance / math.sqrt(overlap_variance * performance_variance)

    sorted_by_overlap = sorted(values, key=lambda item: item[0], reverse=True)
    easiest_performance = sorted_by_overlap[0][1]
    hardest_performance = sorted_by_overlap[-1][1]
    degradation = (
        easiest_performance - hardest_performance
        if higher_is_better
        else hardest_performance - easiest_performance
    )
    if degradation < 0:
        warnings.append(
            "Performance improves on lower-overlap splits; check metric direction, sample sizes, or confounding."
        )

    required_sections = [
        "spectral_property_rationale",
        "split_validation",
        "performance_vs_overlap",
        "replication_evidence",
        "explanatory_depth_assessment",
        "mechanism_debt_status",
        "limitations",
        "deployment_implications",
    ]
    return {
        "valid": True,
        "status": "reviewed",
        "metric_rows": len(values),
        "performance_overlap_correlation": correlation,
        "estimated_degradation_from_high_to_low_overlap": degradation,
        "interpretation": (
            "Performance appears to degrade as overlap decreases."
            if degradation > 0
            else "No degradation was observed; verify the spectral property and evaluation design."
        ),
        "required_report_sections": required_sections,
        "warnings": warnings,
    }


def _safe_correlation(left: Sequence[float], right: Sequence[float]) -> Optional[float]:
    if len(left) < 2 or len(right) < 2:
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    left_var = sum(value * value for value in left_centered)
    right_var = sum(value * value for value in right_centered)
    if left_var == 0 or right_var == 0:
        return None
    covariance = sum(lvalue * rvalue for lvalue, rvalue in zip(left_centered, right_centered))
    return covariance / math.sqrt(left_var * right_var)


def _linear_slope(x_values: Sequence[float], y_values: Sequence[float]) -> Optional[float]:
    if len(x_values) < 2:
        return None
    x_mean = sum(x_values) / len(x_values)
    y_mean = sum(y_values) / len(y_values)
    denominator = sum((value - x_mean) ** 2 for value in x_values)
    if denominator == 0:
        return None
    numerator = sum((x_value - x_mean) * (y_value - y_mean) for x_value, y_value in zip(x_values, y_values))
    return numerator / denominator


def score_similarity_hypothesis_curve(
    curve_points: Union[str, List[Dict[str, Any]], Dict[str, Any]],
    novelty_key: str = "mean_novelty",
    performance_key: str = "rmse",
    lower_is_better: bool = True,
    min_subset_size: int = 10,
    leakage_risk: str = "none",
) -> Dict[str, Any]:
    """Score whether one similarity hypothesis explains performance degradation.

    The returned status is one of:

    - `monotonic_supported`
    - `localized_supported`
    - `weak_supported`
    - `not_explanatory`
    - `not_evaluable`

    For lower-is-better metrics such as RMSE, larger values are treated as
    worse. For higher-is-better metrics, values are negated before scoring.
    """
    rows = _normalize_rows(curve_points)
    normalized_risk = _normalize_token(leakage_risk)
    usable = []
    warnings = []
    for row in rows:
        test_size_value = _first_present(row, ("test_size", "n_test", "n"))
        test_size = int(test_size_value) if test_size_value is not None else min_subset_size
        if test_size < min_subset_size:
            continue
        novelty_value = _first_present(row, (novelty_key, "novelty", "spectral_parameter"))
        performance_value = row.get(performance_key)
        if novelty_value is None or performance_value is None:
            continue
        try:
            novelty = _as_float(novelty_value, novelty_key)
            performance = _as_float(performance_value, performance_key)
        except ValueError:
            continue
        difficulty = performance if lower_is_better else -performance
        usable.append(
            {
                "subset": str(row.get("subset", "curve_point_%d" % len(usable))),
                "novelty": novelty,
                "performance": performance,
                "difficulty": difficulty,
                "test_size": test_size,
                "raw": row,
            }
        )

    if len(usable) < 2:
        return {
            "status": "not_evaluable",
            "valid": False,
            "reason": "Fewer than two usable curve points after size and numeric checks.",
            "point_count": len(usable),
            "recommended_next_action": "Try another similarity definition or lower min_subset_size only if scientifically justified.",
            "leakage_risk": normalized_risk,
            "prospective_valid": normalized_risk in {"none", "prospective", "low"},
            "warnings": warnings,
        }

    usable.sort(key=lambda row: row["novelty"])
    novelty_values = [row["novelty"] for row in usable]
    difficulty_values = [row["difficulty"] for row in usable]
    all_eval = usable[0]
    lowest_overlap = usable[-1]
    hardest = max(usable, key=lambda row: row["difficulty"])
    correlation = _safe_correlation(novelty_values, difficulty_values)
    slope = _linear_slope(novelty_values, difficulty_values)
    baseline = all_eval["difficulty"]
    tolerance = max(1e-12, abs(baseline) * 0.02)
    lowest_delta = lowest_overlap["difficulty"] - baseline
    max_delta = hardest["difficulty"] - baseline

    if correlation is not None and correlation >= 0.5 and lowest_delta > tolerance:
        status = "monotonic_supported"
        reason = "Difficulty rises with novelty and the lowest-overlap subset is harder than the baseline curve point."
        recommended_next_action = "Use this axis if it is scientifically defensible; still report failed alternatives and leakage risk."
    elif max_delta > tolerance:
        status = "localized_supported"
        reason = "A lower-support region is harder, but the hardest subset is not the final novelty point."
        recommended_next_action = "Use this as a localized failure finding and test a refined similarity axis around that region."
    elif correlation is not None and correlation > 0.25:
        status = "weak_supported"
        reason = "Difficulty weakly increases with novelty but the effect is not strong enough to rely on alone."
        recommended_next_action = "Try a richer or composite similarity definition before treating this as the main axis."
    else:
        status = "not_explanatory"
        reason = "This similarity axis did not explain a monotonic or localized performance drop."
        recommended_next_action = "Record this negative finding and test the next similarity hypothesis."

    prospective_valid = normalized_risk in {"none", "prospective", "low"}
    if not prospective_valid:
        warnings.append(
            "This axis has leakage risk '%s'; use it for post-hoc explanation, not prospective split design."
            % normalized_risk
        )

    return {
        "status": status,
        "valid": status in {"monotonic_supported", "localized_supported", "weak_supported"},
        "reason": reason,
        "recommended_next_action": recommended_next_action,
        "leakage_risk": normalized_risk,
        "prospective_valid": prospective_valid,
        "point_count": len(usable),
        "performance_key": performance_key,
        "lower_is_better": lower_is_better,
        "difficulty_novelty_correlation": correlation,
        "difficulty_novelty_slope": slope,
        "baseline_subset": all_eval["subset"],
        "baseline_difficulty": baseline,
        "lowest_overlap_subset": lowest_overlap["subset"],
        "lowest_overlap_difficulty": lowest_overlap["difficulty"],
        "lowest_overlap_delta": lowest_delta,
        "hardest_subset": hardest["subset"],
        "hardest_difficulty": hardest["difficulty"],
        "hardest_delta": max_delta,
        "usable_points": [
            {
                "subset": row["subset"],
                "novelty": row["novelty"],
                "performance": row["performance"],
                "difficulty": row["difficulty"],
                "test_size": row["test_size"],
            }
            for row in usable
        ],
        "warnings": warnings,
    }


def plan_iterative_similarity_search(
    dataset_description: str,
    task_description: str = "",
    data_type: str = "",
    required_inputs: Optional[List[str]] = None,
    top_k: int = 5,
    include_post_hoc_axes: bool = True,
) -> Dict[str, Any]:
    """Plan the iterative SPECTRA similarity-hypothesis loop for an agent."""
    suggestions = suggest_similarity_definitions(
        dataset_description=dataset_description,
        task_description=task_description,
        data_type=data_type,
        required_inputs=required_inputs,
        top_k=top_k,
    )
    return {
        "mode": "iterative_similarity_hypothesis_search",
        "investigator_mode": {
            "enabled": True,
            "control_state": "hypothesis_ledger",
            "loop": "observe -> interpret -> hypothesize -> discriminate -> update -> continue",
            "experiment_rule": (
                "Do not test the next axis because it is available. Test the "
                "axis or construct the split that best distinguishes live "
                "hypotheses about why the model generalizes or fails."
            ),
        },
        "agent_instruction": (
            "First decide whether benchmark mode is feasible. If raw labeled data "
            "and a trainable model or baseline are available, construct "
            "similarity-controlled splits and retrain a fresh model per split. "
            "Do not run one similarity metric and stop. Do not stop at data-level "
            "overlap when model behavior can be tested. Treat each candidate as a "
            "hypothesis, run SPECTRA, score the curve, record failures, update a "
            "hypothesis ledger, and choose the next experiment by discriminative "
            "value. Keep searching for a scientifically defensible explanation "
            "of where target-model performance changes. Do not choose an arbitrary representative task "
            "subset and stop; task coverage must be comprehensive or followed by "
            "a launched executable continuation. If the current axis/data/compute scope is "
            "exhausted first, launch the next expanded scope, or launch a smaller fallback "
            "with a concrete blocker for the larger job, rather than making a general generalizability claim."
        ),
        "search_goal": (
            "Find a scientifically defensible, leakage-aware similarity axis that "
            "changes target-model behavior. Non-explanatory axes are negative "
            "findings that should drive the next axis choice, not a reason to stop."
        ),
        "minimum_axis_diversity": [
            "surface/content similarity such as exact identity, edit distance, k-mer overlap, image/radiomic texture, graph topology, or descriptor distance",
            "representation similarity such as target-model embedding distance or learned latent-neighbor overlap",
            "metadata/context/provenance similarity such as batch, site, assay, source, tissue, time, protocol, scaffold, organism, or domain labels",
            "scientific mechanism similarity such as motifs, pathways, ontology, structure, genomic locus, phenotype class, perturbation class, or conservation when available",
        ],
        "task_coverage_policy": {
            "default": "Run every feasible labeled task/dataset/model setting for the chosen axis class before reporting a completed positive or negative SPECTRA finding.",
            "if_all_tasks_are_infeasible": "Rank all available tasks by overlap/data-screen suspiciousness, scientific diversity, source family, feasibility, sample size, and prior/internal signals, then run the top-ranked feasible tasks plus any task needed to cover a distinct scientific family. Every omitted feasible task requires a launched continuation or a launched smaller fallback with a concrete blocker for the larger task.",
            "forbidden": "Do not stop after a self-selected 'representative' subset. Do not satisfy continuation by writing a queued-only manifest.",
            "negative_result_rule": "A negative or inconclusive result on a limited subset is preliminary; expand to remaining feasible tasks before treating the search as exhausted.",
            "required_artifacts": [
                "task_coverage_plan.json",
                "task_screen_ranking.csv",
                "task_selection_rationale.json",
                "untested_task_blockers.json",
            ],
        },
        "loop": [
            "Identify the scientific unit and candidate novelty axes.",
            "Choose benchmark mode unless retraining is impossible; audit fallback fixed-prediction binning is diagnostic only.",
            "If the target model environment is missing, inspect the repo dependencies and attempt a reasonable install or env creation before downgrading to a baseline.",
            "State the current scientific question and maintain a question trace.",
            "Start investigator mode: write observations, competing hypotheses, and falsifiable predictions before choosing the next axis.",
            "Declare an initial axis-search scope that includes candidate axis classes, approximate compute budget, and what additional data, features, model access, tasks, or compute would be needed if current inputs fail.",
            "Before behavioral testing, create a task-coverage plan: enumerate all available tasks/datasets, run an all-task data screen when possible, rank tasks by suspiciousness/diversity/feasibility/prior signals, and either run all feasible tasks or launch continuation jobs/fallback slices for omitted tasks.",
            "Classify each axis as prospective, post_hoc_explanatory, or invalid.",
            "For each candidate axis, choose an exact or scalable computation strategy.",
            "If a data-level screen finds exact duplicates, high train-test support, or suspicious split geometry, mark it pre-benchmark and choose a behavioral follow-up.",
            "Use the data-screen ranking to choose behavioral follow-up tasks; include high-suspicion tasks and distinct scientific families rather than arbitrary representative tasks.",
            "In benchmark mode, generate similarity-controlled train/test splits across measured overlap levels.",
            "In benchmark mode, train a fresh model for each split before evaluating that split's held-out units.",
            "In audit fallback mode, generate pairwise_similarity.csv or a measured spectral axis from fixed predictions and record the fallback reason.",
            "Run run_spectra_audit or an equivalent benchmark-mode evaluator.",
            "Call score_similarity_hypothesis_curve on the generated performance curve.",
            "After each result, update the hypothesis ledger: what became more likely, less likely, surprising, or unresolved?",
            "Choose the next experiment because it distinguishes live hypotheses, not because it is the next registry item.",
            "Call decide_next_spectra_experiment after each axis result, passing axis_depth_level and mechanism-debt status when an axis is supported.",
            "If status is not_explanatory or not_evaluable, record the negative finding and try the next axis from an untested class while budget remains.",
            "If status is localized_supported, refine or composite the axis around the failure region.",
            "If status is monotonic_supported or weak_supported, immediately execute replication on another task/dataset/seed/model setting or stronger operating point when feasible; do not stop at a proposal when inputs are available.",
            "A replicated supported axis cannot terminate the loop; once explanatory depth is classified, continue to mechanism-depth testing, broader replication, residual axes, public resources, or constructed datasets.",
            "After a candidate axis is found, build an evidence table showing where the same axis is supported, weak, localized, or non-explanatory across tasks/datasets.",
            "If tested tasks are negative or mixed and untested feasible tasks remain, expand task coverage before calling the search negative or exhausted.",
            "After replication, compare supported versus non-supported tasks to infer what the axis is really measuring and why it fails to explain some datasets.",
            "If replication support is mixed, call reflect_on_replication_evidence and test a residual or composite axis such as representation distance after matching sequence similarity, motif-family support, task-context interactions, or class-conditional curves.",
            "If a model-space or embedding axis remains supported, call translate_model_space_axis_to_domain_hypotheses; annotate high- and low-support embedding regions with biological/domain features and test domain-interpretable hypotheses. If local annotations are shallow, continue to public resources or constructed hypothesis-test data.",
            "For every supported proxy or domain-interpretable axis, call assess_explanatory_depth. If the axis is only surface-level or broad-proxy evidence, push to curated annotations, mechanism-level features, context interactions, or mediation tests rather than stopping.",
            "For every supported proxy, broad domain proxy, curated axis without controls, or model-space pointer, call enforce_mechanism_debt_gate. Local mechanism/mediation tests are an intermediate tier; if they remain proxy-level, continue to public-resource, constructed-dataset, residual-axis, or broader-task experiments.",
            "If local inputs include sequences, structures, images, text, tables, metadata, or feature matrices, run input-derived mechanism or mediation tests before treating missing external annotations as blockers.",
            "If the next explanatory-depth step requires resources missing locally, call plan_public_resource_acquisition, search for public resources, download/map/validate a useful resource when feasible, and run the next mechanism test; otherwise launch an alternate executable path or a bounded acquisition/mapping fallback.",
            "If a live hypothesis cannot be falsified with current rows, call plan_hypothesis_driven_dataset_acquisition and acquire or construct a fit-for-purpose public/local dataset that can distinguish that hypothesis from its alternatives.",
            "If coordinate-level resources cannot be mapped from the local files, recover source provenance first: inspect upstream dataset repositories, loaders, docs, manifests, raw archives, and supplementary files for coordinates or stable IDs before attempting sequence alignment or reporting a mapping blocker.",
            "If the current benchmark still cannot test the mechanism hypothesis after local-derived mechanism tests, provenance recovery, and resource acquisition attempts, call plan_hypothesis_test_dataset_construction and construct a defensible dataset from public/local resources to test the hypothesis with fresh split-based SPECTRA runs.",
            "If the declared axis/data/compute scope is exhausted before a supported degradation axis is found, escalate the search scope instead of concluding that no such axis exists.",
        ],
        "status_meanings": {
            "monotonic_supported": "Novelty under this axis broadly tracks performance degradation.",
            "localized_supported": "A particular low-support region is harder, but the curve is not globally monotonic.",
            "weak_supported": "The axis has a weak signal and should not be the only evidence.",
            "not_explanatory": "The axis does not explain observed failure; this is a reportable finding.",
            "not_evaluable": "The curve lacks enough usable numeric points or variation.",
        },
        "selection_rule": (
            "Select the strongest scientifically defensible, leakage-aware axis, "
            "not merely the curve that looks most monotonic."
        ),
        "leakage_policy": {
            "prospective_axes": "Usable before labels are known and valid for split design.",
            "post_hoc_axes": "Useful for explaining observed failures after labels are available.",
            "invalid_axes": "Label leakage, circular definitions, confounding, or too few usable points.",
            "fixed_prediction_binning": "Diagnostic audit fallback only; not a substitute for benchmark-mode retraining when training is feasible.",
            "data_only_overlap_screen": "Pre-benchmark screening only; it must trigger behavioral testing unless behavior is already tested or blocked.",
            "environment_blocker": "Missing packages or repo setup are not terminal blockers; after repair attempts, switch to another executable target-model task, smaller slice, adapter/probe setting, or launched compute job/fallback.",
            "limited_task_scope": "A run over an arbitrary representative task subset is preliminary only. Omitted feasible tasks require executable continuation jobs.",
            "negative_axis_result": "A non-explanatory curve is a useful result, but not a stop condition while untested plausible axis classes remain.",
            "scope_exhaustion": "Current-scope exhaustion means the agent must launch more data, features, tasks, model access, compute, or a smaller executable fallback; it is not evidence of universal generalization and not a terminal state.",
            "mixed_replication": "A replicated axis with supported and non-supported tasks is not the final explanation; use the contrast to derive residual or composite axes.",
            "model_space_axis": "Embedding or representation distance is a behavioral pointer; translate it into biological/domain hypotheses before presenting it as a scientific explanation.",
            "proxy_axis": "Surface proxies and broad biological/domain proxies are findings, not explanations; use them to derive mechanism-level axes and mediation tests when inputs allow.",
            "mechanism_debt_gate": "A supported proxy/model-space axis creates mechanism debt. Local-derived tests do not close it if they remain proxy-level; continue to public-resource, constructed-dataset, residual-axis, or broader-task experiments.",
            "public_resource_acquisition": "Missing local mechanism annotations, references, ontologies, coordinates, or public benchmark data require public search/download/mapping attempts before they become blockers.",
            "source_provenance_recovery": "Missing coordinates or stable sample IDs require upstream dataset-source inspection before alignment or final mapping-blocker reports.",
            "hypothesis_test_dataset_construction": "When the original benchmark cannot test the mechanism question, construct a defensible dataset from public/local resources and run fresh split-based SPECTRA experiments as hypothesis-test evidence.",
            "include_post_hoc_axes": include_post_hoc_axes,
        },
        "candidate_similarity_definitions": suggestions,
        "required_outputs": [
            "similarity_hypothesis_order",
            "observations",
            "hypothesis_ledger",
            "competing_explanations",
            "why_this_next_experiment",
            "discriminating_experiment_plan",
            "belief_update",
            "falsifiable_predictions",
            "task_coverage_plan",
            "task_screen_ranking",
            "task_selection_rationale",
            "untested_task_blockers",
            "axis_search_budget",
            "curve_score_per_hypothesis",
            "replication_evidence",
            "replication_reflection",
            "supported_vs_failed_task_contrast",
            "residual_axis_candidates",
            "residual_axis_scores",
            "model_space_biological_translation",
            "embedding_support_biological_contrast",
            "domain_hypothesis_axis_candidates",
            "domain_hypothesis_scores",
            "explanatory_depth_assessment",
            "proxy_to_mechanism_plan",
            "mechanism_debt_register",
            "mechanism_execution_manifest",
            "mechanism_infeasibility_proof",
            "public_resource_acquisition_plan",
            "public_resource_search_log",
            "source_provenance_recovery_log",
            "hypothesis_test_dataset_plan",
            "hypothesis_driven_acquisition_plan",
            "external_dataset_decision_log",
            "constructed_dataset_manifest",
            "constructed_dataset_provenance",
            "constructed_dataset_schema",
            "constructed_dataset_mapping_validation",
            "constructed_dataset_leakage_audit",
            "constructed_dataset_spectra_results",
            "resource_manifest",
            "resource_mapping_validation",
            "mechanism_axis_scores",
            "mediation_test_results",
            "paper_ready_spectra_finding",
            "claim_boundary",
            "model_paper_context",
            "evidence_to_claim_table",
            "overclaim_guardrails",
            "selected_similarity_axis",
            "leakage_risk_per_axis",
            "failed_axis_findings",
            "execution_mode_decision",
            "retraining_manifest_or_audit_fallback_reason",
            "question_trace",
            "next_experiment_decisions",
            "continuation_launch_manifest",
            "continuation_status_log",
            "blockers",
            "next_similarity_or_data_requirements",
        ],
    }


def _load_fastmcp_class() -> Any:
    try:
        from fastmcp import FastMCP  # type: ignore

        return FastMCP
    except ImportError:
        from mcp.server.fastmcp import FastMCP  # type: ignore

        return FastMCP


def _load_benchmark_module() -> Any:
    try:
        from . import spectra_benchmarks  # type: ignore

        return spectra_benchmarks
    except Exception:
        path = os.path.join(os.path.dirname(__file__), "spectra_benchmarks.py")
        spec = importlib.util.spec_from_file_location("spectrae_spectra_benchmarks", path)
        if spec is None or spec.loader is None:
            raise ImportError("Could not load spectra_benchmarks.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


def _load_similarity_registry_module() -> Any:
    try:
        from . import similarity_registry  # type: ignore

        return similarity_registry
    except Exception:
        path = os.path.join(os.path.dirname(__file__), "similarity_registry.py")
        spec = importlib.util.spec_from_file_location("spectrae_similarity_registry", path)
        if spec is None or spec.loader is None:
            raise ImportError("Could not load similarity_registry.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


def _load_similarity_computation_registry_module() -> Any:
    try:
        from . import similarity_computation_registry  # type: ignore

        return similarity_computation_registry
    except Exception:
        path = os.path.join(os.path.dirname(__file__), "similarity_computation_registry.py")
        spec = importlib.util.spec_from_file_location("spectrae_similarity_computation_registry", path)
        if spec is None or spec.loader is None:
            raise ImportError("Could not load similarity_computation_registry.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


def _load_operating_point_registry_module() -> Any:
    try:
        from . import operating_point_registry  # type: ignore

        return operating_point_registry
    except Exception:
        path = os.path.join(os.path.dirname(__file__), "operating_point_registry.py")
        spec = importlib.util.spec_from_file_location("spectrae_operating_point_registry", path)
        if spec is None or spec.loader is None:
            raise ImportError("Could not load operating_point_registry.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


def _load_memory_registry_module() -> Any:
    try:
        from . import memory_registry  # type: ignore

        return memory_registry
    except Exception:
        path = os.path.join(os.path.dirname(__file__), "memory_registry.py")
        spec = importlib.util.spec_from_file_location("spectrae_memory_registry", path)
        if spec is None or spec.loader is None:
            raise ImportError("Could not load memory_registry.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


def _load_dataset_catalog_module() -> Any:
    try:
        from . import dataset_catalog_registry  # type: ignore

        return dataset_catalog_registry
    except Exception:
        path = os.path.join(os.path.dirname(__file__), "dataset_catalog_registry.py")
        spec = importlib.util.spec_from_file_location("spectrae_dataset_catalog_registry", path)
        if spec is None or spec.loader is None:
            raise ImportError("Could not load dataset_catalog_registry.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


def list_benchmark_capsules(
    tier: Optional[str] = None,
    community: Optional[str] = None,
) -> Dict[str, Any]:
    """List /spectra benchmark capsules."""
    return _load_benchmark_module().list_benchmark_capsules(tier=tier, community=community)


def get_benchmark_capsule(paper_id: str) -> Dict[str, Any]:
    """Return one /spectra benchmark capsule."""
    return _load_benchmark_module().load_benchmark_capsule(paper_id)


def get_benchmark_download_plan(
    paper_ids: Optional[List[str]] = None,
    include_repos: bool = True,
    include_data: bool = False,
) -> Dict[str, Any]:
    """Return the papers/repos/data pointers needed for benchmark setup."""
    return _load_benchmark_module().benchmark_download_plan(
        paper_ids=paper_ids,
        include_repos=include_repos,
        include_data=include_data,
    )


def create_audit_card_template(paper_id: str) -> Dict[str, Any]:
    """Create a SPECTRA audit-card template from a benchmark capsule."""
    return _load_benchmark_module().create_audit_card_template(paper_id)


def compute_auspc(
    curve_points: Union[str, List[Dict[str, Any]], Dict[str, Any]],
    overlap_key: str = "cross_split_overlap",
    performance_key: str = "performance",
    higher_is_better: bool = True,
) -> Dict[str, Any]:
    """Compute area under the spectral performance curve."""
    return _load_benchmark_module().compute_auspc(
        curve_points,
        overlap_key=overlap_key,
        performance_key=performance_key,
        higher_is_better=higher_is_better,
    )


def validate_spectra_audit_card(
    audit_card: Union[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Validate a SPECTRA audit card."""
    return _load_benchmark_module().validate_audit_card(audit_card)


def score_spectra_agent_audit(
    audit_card: Union[str, Dict[str, Any]],
    rubric_scores: Optional[Union[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Score a with-skill or without-skill agent audit."""
    return _load_benchmark_module().score_agent_audit(audit_card, rubric_scores)


def render_spectra_audit_report(
    audit_card: Union[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Render a SPECTRA audit card as Markdown."""
    return _load_benchmark_module().render_spectra_report(audit_card)


def list_similarity_definitions(
    data_type: Optional[str] = None,
    scientific_unit: Optional[str] = None,
    task: Optional[str] = None,
    status: Optional[str] = None,
) -> Dict[str, Any]:
    """List literature-backed SPECTRA similarity definitions."""
    return _load_similarity_registry_module().list_similarity_definitions(
        data_type=data_type,
        scientific_unit=scientific_unit,
        task=task,
        status=status,
    )


def get_similarity_definition(definition_id: str) -> Dict[str, Any]:
    """Return one literature-backed SPECTRA similarity definition."""
    return _load_similarity_registry_module().load_similarity_definition(definition_id)


def suggest_similarity_definitions(
    dataset_description: str,
    task_description: str = "",
    data_type: str = "",
    required_inputs: Optional[List[str]] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Suggest candidate similarity definitions for a SPECTRA audit."""
    return _load_similarity_registry_module().suggest_similarity_definitions(
        dataset_description=dataset_description,
        task_description=task_description,
        data_type=data_type,
        required_inputs=required_inputs,
        top_k=top_k,
    )


def get_similarity_example_script(definition_id: str) -> Dict[str, Any]:
    """Return a Python example that creates SPECTRA-compatible similarity inputs."""
    return _load_similarity_registry_module().get_similarity_example_script(definition_id)


def render_similarity_definition(definition_id: str) -> Dict[str, Any]:
    """Render one similarity definition as Markdown."""
    return _load_similarity_registry_module().render_similarity_definition(definition_id)


def validate_similarity_registry() -> Dict[str, Any]:
    """Validate every bundled similarity definition."""
    return _load_similarity_registry_module().validate_similarity_registry()


def list_similarity_computation_strategies(
    strategy_family: Optional[str] = None,
    exactness: Optional[str] = None,
    data_shape: Optional[str] = None,
    status: Optional[str] = None,
) -> Dict[str, Any]:
    """List literature-backed SPECTRA similarity computation strategies."""
    return _load_similarity_computation_registry_module().list_similarity_computation_strategies(
        strategy_family=strategy_family,
        exactness=exactness,
        data_shape=data_shape,
        status=status,
    )


def get_similarity_computation_strategy(strategy_id: str) -> Dict[str, Any]:
    """Return one literature-backed SPECTRA similarity computation strategy."""
    return _load_similarity_computation_registry_module().load_similarity_computation_strategy(strategy_id)


def suggest_similarity_computation_strategies(
    dataset_description: str,
    similarity_definition: str = "",
    data_shape: str = "",
    data_size: str = "",
    required_inputs: Optional[List[str]] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Suggest scalable computation strategies after choosing a similarity definition."""
    return _load_similarity_computation_registry_module().suggest_similarity_computation_strategies(
        dataset_description=dataset_description,
        similarity_definition=similarity_definition,
        data_shape=data_shape,
        data_size=data_size,
        required_inputs=required_inputs,
        top_k=top_k,
    )


def get_similarity_computation_example_script(strategy_id: str) -> Dict[str, Any]:
    """Return a Python example that creates SPECTRA-compatible pairwise similarities."""
    return _load_similarity_computation_registry_module().get_similarity_computation_example_script(strategy_id)


def render_similarity_computation_strategy(strategy_id: str) -> Dict[str, Any]:
    """Render one similarity computation strategy as Markdown."""
    return _load_similarity_computation_registry_module().render_similarity_computation_strategy(strategy_id)


def validate_similarity_computation_registry() -> Dict[str, Any]:
    """Validate every bundled similarity computation strategy."""
    return _load_similarity_computation_registry_module().validate_similarity_computation_registry()


def list_operating_point_methods(
    method_family: Optional[str] = None,
    scientific_unit: Optional[str] = None,
    data_type: Optional[str] = None,
    curve_region: Optional[str] = None,
    status: Optional[str] = None,
) -> Dict[str, Any]:
    """List targeted SPECTRA operating-point methods."""
    return _load_operating_point_registry_module().list_operating_point_methods(
        method_family=method_family,
        scientific_unit=scientific_unit,
        data_type=data_type,
        curve_region=curve_region,
        status=status,
    )


def get_operating_point_method(method_id: str) -> Dict[str, Any]:
    """Return one literature-backed targeted operating-point method."""
    return _load_operating_point_registry_module().load_operating_point_method(method_id)


def suggest_operating_point_methods(
    dataset_description: str,
    deployment_question: str = "",
    data_type: str = "",
    novelty_axis: str = "",
    required_inputs: Optional[List[str]] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Suggest targeted operating-point methods for a deployment question."""
    return _load_operating_point_registry_module().suggest_operating_point_methods(
        dataset_description=dataset_description,
        deployment_question=deployment_question,
        data_type=data_type,
        novelty_axis=novelty_axis,
        required_inputs=required_inputs,
        top_k=top_k,
    )


def render_operating_point_method(method_id: str) -> Dict[str, Any]:
    """Render one targeted operating-point method as Markdown."""
    return _load_operating_point_registry_module().render_operating_point_method(method_id)


def validate_operating_point_registry() -> Dict[str, Any]:
    """Validate every bundled targeted operating-point method."""
    return _load_operating_point_registry_module().validate_operating_point_registry()


def list_spectra_memory_entries(
    domain: Optional[str] = None,
    data_type: Optional[str] = None,
    model_family: Optional[str] = None,
    status: Optional[str] = None,
) -> Dict[str, Any]:
    """List reusable memories from prior SPECTRA audits."""
    return _load_memory_registry_module().list_memory_entries(
        domain=domain,
        data_type=data_type,
        model_family=model_family,
        status=status,
    )


def get_spectra_memory_entry(entry_id: str) -> Dict[str, Any]:
    """Return one reusable SPECTRA memory entry."""
    return _load_memory_registry_module().load_memory_entry(entry_id)


def search_spectra_memory_entries(
    query: str = "",
    domain: str = "",
    data_type: str = "",
    model_family: str = "",
    model_name: str = "",
    tags: Optional[List[str]] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Search reusable prior-run memories for a new audit."""
    return _load_memory_registry_module().search_memory_entries(
        query=query,
        domain=domain,
        data_type=data_type,
        model_family=model_family,
        model_name=model_name,
        tags=tags,
        top_k=top_k,
    )


def suggest_reusable_spectra_memory(
    model_description: str,
    dataset_description: str = "",
    domain: str = "",
    top_k: int = 5,
) -> Dict[str, Any]:
    """Suggest prior-run memories for a new /spectra session."""
    return _load_memory_registry_module().suggest_reusable_memory(
        model_description=model_description,
        dataset_description=dataset_description,
        domain=domain,
        top_k=top_k,
    )


def render_spectra_memory_entry(entry_id: str) -> Dict[str, Any]:
    """Render one prior-run memory as Markdown for agent context."""
    return _load_memory_registry_module().render_memory_entry(entry_id)


def validate_spectra_memory_registry() -> Dict[str, Any]:
    """Validate every bundled SPECTRA memory entry."""
    return _load_memory_registry_module().validate_memory_registry()


def list_dataset_catalog_entries(
    domain: Optional[str] = None,
    data_type: Optional[str] = None,
    model_family: Optional[str] = None,
    access: Optional[str] = None,
    status: Optional[str] = None,
) -> Dict[str, Any]:
    """List portable dataset catalog entries for SPECTRA audits."""
    return _load_dataset_catalog_module().list_dataset_entries(
        domain=domain,
        data_type=data_type,
        model_family=model_family,
        access=access,
        status=status,
    )


def get_dataset_catalog_entry(dataset_id: str) -> Dict[str, Any]:
    """Return one portable dataset catalog entry."""
    return _load_dataset_catalog_module().load_dataset_entry(dataset_id)


def search_dataset_catalog_entries(
    query: str = "",
    domain: str = "",
    data_type: str = "",
    model_family: str = "",
    required_fields: Optional[List[str]] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Search portable dataset catalog entries."""
    return _load_dataset_catalog_module().search_dataset_entries(
        query=query,
        domain=domain,
        data_type=data_type,
        model_family=model_family,
        required_fields=required_fields,
        top_k=top_k,
    )


def suggest_dataset_catalog_entries(
    audit_question: str,
    model_description: str = "",
    domain: str = "",
    current_dataset_limitation: str = "",
    top_k: int = 5,
) -> Dict[str, Any]:
    """Suggest portable datasets for an audit question or Dataset Scout handoff."""
    return _load_dataset_catalog_module().suggest_dataset_entries(
        audit_question=audit_question,
        model_description=model_description,
        domain=domain,
        current_dataset_limitation=current_dataset_limitation,
        top_k=top_k,
    )


def render_dataset_catalog_entry(dataset_id: str) -> Dict[str, Any]:
    """Render one dataset catalog entry as Markdown."""
    return _load_dataset_catalog_module().render_dataset_entry(dataset_id)


def validate_dataset_catalog() -> Dict[str, Any]:
    """Validate every bundled dataset catalog entry."""
    return _load_dataset_catalog_module().validate_dataset_catalog()


def run_spectra_audit(
    domain: str,
    eval_path: str,
    output_dir: str,
    train_path: Optional[str] = None,
    mode: str = "axis",
    scientific_unit: str = "sample",
    axis_col: Optional[str] = None,
    axis_type: str = "similarity",
    axis_name: str = "spectral_axis",
    pairwise_similarity_path: Optional[str] = None,
    eval_id_col: str = "sample_id",
    train_id_col: str = "train_id",
    similarity_eval_id_col: str = "sample_id",
    similarity_train_id_col: str = "train_id",
    similarity_col: str = "similarity",
    smiles_col: str = "smiles",
    train_target_col: str = "y",
    eval_target_col: str = "y_true",
    pred_col: str = "y_pred",
    sample_id_col: Optional[str] = None,
    thresholds: str = "1.0,0.8,0.7,0.6,0.5",
    fingerprint_radius: int = 2,
    fingerprint_bits: int = 1024,
    prefer_rdkit: bool = True,
) -> Dict[str, Any]:
    """Run the deterministic package-backed SPECTRA audit engine."""
    normalized_domain = _normalize_token(domain)
    normalized_mode = _normalize_token(mode)
    try:
        from .audit import (
            MoleculeAuditConfig,
            PairwiseSimilarityAuditConfig,
            SpectralAxisAuditConfig,
            parse_optional_threshold_string,
            parse_threshold_string,
            run_axis_audit,
            run_molecule_audit,
            run_pairwise_similarity_audit,
        )
    except Exception:
        import importlib.util

        path = os.path.join(os.path.dirname(__file__), "audit.py")
        spec = importlib.util.spec_from_file_location("spectrae_audit", path)
        if spec is None or spec.loader is None:
            raise ImportError("Could not load audit.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        MoleculeAuditConfig = module.MoleculeAuditConfig
        PairwiseSimilarityAuditConfig = module.PairwiseSimilarityAuditConfig
        SpectralAxisAuditConfig = module.SpectralAxisAuditConfig
        parse_optional_threshold_string = module.parse_optional_threshold_string
        parse_threshold_string = module.parse_threshold_string
        run_axis_audit = module.run_axis_audit
        run_molecule_audit = module.run_molecule_audit
        run_pairwise_similarity_audit = module.run_pairwise_similarity_audit

    if normalized_mode == "adapter" or (
        normalized_domain in {"molecules", "molecule", "small_molecule"}
        and not axis_col
        and not pairwise_similarity_path
    ):
        if normalized_domain not in {"molecules", "molecule", "small_molecule"}:
            raise ValueError("No adapter is implemented for domain: %s" % domain)
        if not train_path:
            raise ValueError("train_path is required for the molecule adapter")
        return run_molecule_audit(
            MoleculeAuditConfig(
                train_path=train_path,
                eval_path=eval_path,
                output_dir=output_dir,
                smiles_col=smiles_col,
                train_target_col=train_target_col,
                eval_target_col=eval_target_col,
                pred_col=pred_col,
                sample_id_col=sample_id_col,
                thresholds=parse_threshold_string(thresholds),
                fingerprint_radius=fingerprint_radius,
                fingerprint_bits=fingerprint_bits,
                prefer_rdkit=prefer_rdkit,
            )
        )
    if normalized_mode == "pairwise" or pairwise_similarity_path:
        if not pairwise_similarity_path:
            raise ValueError("pairwise_similarity_path is required for pairwise mode")
        return run_pairwise_similarity_audit(
            PairwiseSimilarityAuditConfig(
                eval_path=eval_path,
                similarity_path=pairwise_similarity_path,
                output_dir=output_dir,
                target_col=eval_target_col,
                pred_col=pred_col,
                eval_id_col=eval_id_col,
                similarity_eval_id_col=similarity_eval_id_col,
                similarity_train_id_col=similarity_train_id_col,
                similarity_col=similarity_col,
                domain=normalized_domain,
                scientific_unit=scientific_unit,
                train_path=train_path,
                train_id_col=train_id_col,
                train_target_col=train_target_col,
                axis_name=axis_name,
                thresholds=parse_optional_threshold_string(thresholds),
            )
        )
    if not axis_col:
        raise ValueError("axis_col or pairwise_similarity_path is required for generic SPECTRA audits")
    return run_axis_audit(
        SpectralAxisAuditConfig(
            eval_path=eval_path,
            output_dir=output_dir,
            target_col=eval_target_col,
            pred_col=pred_col,
            axis_col=axis_col,
            axis_type=axis_type,
            axis_name=axis_name,
            domain=normalized_domain,
            scientific_unit=scientific_unit,
            train_path=train_path,
            train_target_col=train_target_col,
            unit_col=sample_id_col,
            thresholds=parse_optional_threshold_string(thresholds),
        )
    )


def create_mcp_server(host: str = "127.0.0.1", port: int = 8000) -> Any:
    """Create the FastMCP server instance."""
    FastMCP = _load_fastmcp_class()
    server_instructions = (
            "Use this server to run SPECTRA as a spectral performance curve "
            "construction and validation workflow. When the user gives a "
            "generalizability question plus a model paper/reference, call "
            "start_spectra_audit_session first to create the single-controller "
            "audit contract. Distiller, Investigator, Dataset Scout, Dataset "
            "Fetcher, Auditor, and synthesis are internal controller phases, not "
            "client-routed agents. For one-command autonomous runs, use "
            "prepare_spectra_audit_session or run_spectra_audit_session; "
            "run_spectra_audit_session only launches the controller when "
            "execute_controller=true and an agent command template is supplied. The primary output is "
            "model performance as prospective train-test or pretraining-test "
            "similarity decreases, plus a validity decision. Similarity axes and "
            "split membership must not use target-model errors, prediction-vs-"
            "reference errors, held-out labels, target confidence derived from the "
            "evaluated model, or any post-hoc quantity selected because it tracks "
            "model failure. Use get_procedure, dataset catalog tools, "
            "start_generalizability_analysis, select_spectra_execution_mode, "
            "plan_spectral_performance_curve, similarity definitions, similarity "
            "computation strategies, and run_spectra_audit as needed. The "
            "Investigator must construct at least three nontrivial split levels "
            "when feasible, verify measured similarity decreases, run a simple "
            "fixed baseline when labels exist, then evaluate the target model. "
            "The Auditor must check leakage, split sizes, similarity progression, "
            "baseline stability, confounding, metric direction, and post-hoc axis "
            "selection, then classify each SPC as valid, weak, invalid, or "
            "exploratory. Dataset Scout finds suitable datasets; Dataset Fetcher "
            "retrieves and packages SPECTRA-ready artifacts. Final synthesis must "
            "state the evidence boundary and avoid overclaiming beyond the "
            "validated SPC."
    )
    try:
        mcp = FastMCP(
            name=SERVER_NAME,
            instructions=server_instructions,
            host=host,
            port=port,
        )
    except TypeError as exc:
        message = str(exc)
        if "host" not in message and "port" not in message:
            raise
        os.environ.setdefault("FASTMCP_HOST", host)
        os.environ.setdefault("FASTMCP_PORT", str(port))
        mcp = FastMCP(name=SERVER_NAME, instructions=server_instructions)

    @mcp.resource(f"procedure://{PROCEDURE_NAME}/{PROCEDURE_VERSION}")
    def generalizability_procedure_resource() -> str:
        """Return the Markdown procedure for generalizability analysis."""
        return _procedure_text()

    @mcp.resource(f"procedure://{PROCEDURE_NAME}/examples")
    def generalizability_examples_resource() -> Dict[str, str]:
        """Return all worked examples for the generalizability procedure."""
        return {name: _example_text(name) for name in _EXAMPLE_NAMES}

    @mcp.resource("benchmark://spectra/capsules")
    def benchmark_capsules_resource() -> Dict[str, Any]:
        """Return the /spectra benchmark capsule catalog."""
        return list_benchmark_capsules()

    @mcp.resource("similarity-registry://spectra/definitions")
    def similarity_registry_resource() -> Dict[str, Any]:
        """Return the /spectra similarity definition registry catalog."""
        return list_similarity_definitions()

    @mcp.resource("similarity-computation://spectra/strategies")
    def similarity_computation_registry_resource() -> Dict[str, Any]:
        """Return the /spectra similarity computation strategy catalog."""
        return list_similarity_computation_strategies()

    @mcp.resource("operating-points://spectra/methods")
    def operating_point_registry_resource() -> Dict[str, Any]:
        """Return the /spectra targeted operating-point method catalog."""
        return list_operating_point_methods()

    @mcp.resource("memory://spectra/entries")
    def spectra_memory_registry_resource() -> Dict[str, Any]:
        """Return the /spectra reusable run-memory catalog."""
        return list_spectra_memory_entries()

    @mcp.resource("dataset-catalog://spectra/entries")
    def dataset_catalog_resource() -> Dict[str, Any]:
        """Return the /spectra portable dataset catalog."""
        return list_dataset_catalog_entries()

    @mcp.prompt()
    def generalizability_analysis_prompt(
        dataset_description: str,
        model_description: str,
        domain: str = "unknown",
    ) -> str:
        """Prompt an agent to execute the generalizability analysis procedure."""
        return (
            "Run the SPECTRA spectral performance curve analysis.\n\n"
            f"Dataset: {dataset_description}\n\n"
            f"Model: {model_description}\n\n"
            f"Domain: {domain}\n\n"
            "If the user supplied a model paper plus a generalization question, "
            "first call start_spectra_audit_session to create the single-controller "
            "phase contract, terminal gate, and artifact tree. Use get_procedure, "
            "suggest_dataset_catalog_entries, start_generalizability_analysis, "
            "select_spectra_execution_mode, plan_spectral_performance_curve, "
            "suggest_similarity_definitions, suggest_similarity_computation_strategies, "
            "plan_similarity_computation, and run_spectra_audit as needed. "
            "Within the controller session, the Distiller phase should choose a concrete model, dataset, task, metric, "
            "prospective similarity axis, split/bin levels, and validity gates. "
            "The Dataset Scout finds suitable datasets when one is missing. The "
            "Dataset Fetcher retrieves and packages inputs, labels, metadata, and "
            "prospective similarity features into SPECTRA-ready artifacts. The "
            "Investigator computes prospective train-test or pretraining-test "
            "similarities, constructs at least three nontrivial levels when feasible, "
            "verifies measured similarity decreases, runs a simple fixed baseline "
            "when labels exist, and then evaluates the target model. Do not use "
            "target-model errors, prediction/reference errors, held-out labels, "
            "target confidence derived from the evaluated model, or post-hoc "
            "failure correlations to define the axis or split membership. The "
            "Auditor checks target-error leakage, test-label leakage, tiny or "
            "degenerate splits, non-decreasing similarity, unstable baselines, "
            "confounding, metric direction, and post-hoc axis selection. The "
            "final answer must classify the SPC as valid, weak, invalid, or "
            "exploratory and state the claim boundary."
        )

    mcp.tool(name="list_procedures")(available_procedures)
    mcp.tool(name="get_procedure")(get_procedure_document)
    mcp.tool(name="get_procedure_examples")(get_procedure_examples)
    mcp.tool(name="start_spectra_audit_session")(start_spectra_audit_session)
    mcp.tool(name="prepare_spectra_audit_session")(prepare_spectra_audit_session)
    mcp.tool(name="run_spectra_audit_session")(run_spectra_audit_session)
    mcp.tool(name="start_generalizability_analysis")(start_generalizability_analysis)
    mcp.tool(name="plan_spectral_performance_curve")(plan_spectral_performance_curve)
    mcp.tool(name="start_spectra_investigator")(start_spectra_investigator)
    mcp.tool(name="select_spectra_execution_mode")(select_spectra_execution_mode)
    mcp.tool(name="decide_next_spectra_experiment")(decide_next_spectra_experiment)
    mcp.tool(name="recommend_spectral_property")(recommend_spectral_property)
    mcp.tool(name="plan_iterative_similarity_search")(plan_iterative_similarity_search)
    mcp.tool(name="plan_similarity_computation")(plan_similarity_computation)
    mcp.tool(name="score_similarity_hypothesis_curve")(score_similarity_hypothesis_curve)
    mcp.tool(name="update_hypothesis_ledger")(update_hypothesis_ledger)
    mcp.tool(name="choose_discriminating_experiment")(choose_discriminating_experiment)
    mcp.tool(name="plan_hypothesis_driven_dataset_acquisition")(plan_hypothesis_driven_dataset_acquisition)
    mcp.tool(name="distill_spectra_hypotheses")(distill_spectra_hypotheses)
    mcp.tool(name="synthesize_spectra_generalizability_finding")(synthesize_spectra_generalizability_finding)
    mcp.tool(name="review_investigator_checkpoint")(review_investigator_checkpoint)
    mcp.tool(name="reflect_on_replication_evidence")(reflect_on_replication_evidence)
    mcp.tool(name="translate_model_space_axis_to_domain_hypotheses")(translate_model_space_axis_to_domain_hypotheses)
    mcp.tool(name="assess_explanatory_depth")(assess_explanatory_depth)
    mcp.tool(name="enforce_mechanism_debt_gate")(enforce_mechanism_debt_gate)
    mcp.tool(name="plan_public_resource_acquisition")(plan_public_resource_acquisition)
    mcp.tool(name="plan_hypothesis_test_dataset_construction")(plan_hypothesis_test_dataset_construction)
    mcp.tool(name="prepare_dataset_scout_request")(prepare_dataset_scout_request)
    mcp.tool(name="distill_dataset_scout_output")(distill_dataset_scout_output)
    mcp.tool(name="prepare_dataset_constructor_request")(prepare_dataset_constructor_request)
    mcp.tool(name="distill_dataset_constructor_output")(distill_dataset_constructor_output)
    mcp.tool(name="validate_split_stats")(validate_split_stats)
    mcp.tool(name="review_generalizability_report")(review_generalizability_report)
    mcp.tool(name="list_benchmark_capsules")(list_benchmark_capsules)
    mcp.tool(name="get_benchmark_capsule")(get_benchmark_capsule)
    mcp.tool(name="get_benchmark_download_plan")(get_benchmark_download_plan)
    mcp.tool(name="create_audit_card_template")(create_audit_card_template)
    mcp.tool(name="compute_auspc")(compute_auspc)
    mcp.tool(name="validate_spectra_audit_card")(validate_spectra_audit_card)
    mcp.tool(name="score_spectra_agent_audit")(score_spectra_agent_audit)
    mcp.tool(name="render_spectra_audit_report")(render_spectra_audit_report)
    mcp.tool(name="list_similarity_definitions")(list_similarity_definitions)
    mcp.tool(name="get_similarity_definition")(get_similarity_definition)
    mcp.tool(name="suggest_similarity_definitions")(suggest_similarity_definitions)
    mcp.tool(name="get_similarity_example_script")(get_similarity_example_script)
    mcp.tool(name="render_similarity_definition")(render_similarity_definition)
    mcp.tool(name="validate_similarity_registry")(validate_similarity_registry)
    mcp.tool(name="list_similarity_computation_strategies")(list_similarity_computation_strategies)
    mcp.tool(name="get_similarity_computation_strategy")(get_similarity_computation_strategy)
    mcp.tool(name="suggest_similarity_computation_strategies")(suggest_similarity_computation_strategies)
    mcp.tool(name="get_similarity_computation_example_script")(get_similarity_computation_example_script)
    mcp.tool(name="render_similarity_computation_strategy")(render_similarity_computation_strategy)
    mcp.tool(name="validate_similarity_computation_registry")(validate_similarity_computation_registry)
    mcp.tool(name="list_operating_point_methods")(list_operating_point_methods)
    mcp.tool(name="get_operating_point_method")(get_operating_point_method)
    mcp.tool(name="suggest_operating_point_methods")(suggest_operating_point_methods)
    mcp.tool(name="render_operating_point_method")(render_operating_point_method)
    mcp.tool(name="validate_operating_point_registry")(validate_operating_point_registry)
    mcp.tool(name="list_spectra_memory_entries")(list_spectra_memory_entries)
    mcp.tool(name="get_spectra_memory_entry")(get_spectra_memory_entry)
    mcp.tool(name="search_spectra_memory_entries")(search_spectra_memory_entries)
    mcp.tool(name="suggest_reusable_spectra_memory")(suggest_reusable_spectra_memory)
    mcp.tool(name="render_spectra_memory_entry")(render_spectra_memory_entry)
    mcp.tool(name="validate_spectra_memory_registry")(validate_spectra_memory_registry)
    mcp.tool(name="list_dataset_catalog_entries")(list_dataset_catalog_entries)
    mcp.tool(name="get_dataset_catalog_entry")(get_dataset_catalog_entry)
    mcp.tool(name="search_dataset_catalog_entries")(search_dataset_catalog_entries)
    mcp.tool(name="suggest_dataset_catalog_entries")(suggest_dataset_catalog_entries)
    mcp.tool(name="render_dataset_catalog_entry")(render_dataset_catalog_entry)
    mcp.tool(name="validate_dataset_catalog")(validate_dataset_catalog)
    mcp.tool(name="run_spectra_audit")(run_spectra_audit)
    return mcp


try:
    mcp = create_mcp_server()
except ImportError:
    mcp = None


def main(argv: Optional[Sequence[str]] = None) -> None:
    raw_args = list(argv) if argv is not None else None
    if raw_args and raw_args[0] == "serve":
        raw_args = raw_args[1:]

    parser = argparse.ArgumentParser(description="Run the SPECTRA scientific skills MCP server.")
    parser.add_argument(
        "command",
        nargs="?",
        choices=("serve",),
        help="Optional subcommand. `serve` is accepted for readable MCP client configs.",
    )
    parser.add_argument(
        "--transport",
        default="stdio",
        choices=("stdio", "streamable-http", "sse"),
        help="MCP transport to use.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="HTTP host.")
    parser.add_argument("--port", default=8000, type=int, help="HTTP port.")
    args = parser.parse_args(raw_args)

    server = create_mcp_server(host=args.host, port=args.port)
    server.run(transport=args.transport)


if __name__ == "__main__":
    main()
