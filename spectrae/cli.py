"""Command-line interface for package-backed SPECTRA audits."""

import argparse
import json
import os
import pkgutil
import re
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

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


def _add_similarity_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "similarity-definitions",
        help="Search the literature-backed SPECTRA similarity definition registry.",
    )
    nested = parser.add_subparsers(dest="similarity_command", required=True)

    list_parser = nested.add_parser("list", help="List registry entries.")
    list_parser.add_argument("--data-type", default="")
    list_parser.add_argument("--scientific-unit", default="")
    list_parser.add_argument("--task", default="")
    list_parser.add_argument("--status", default="")
    list_parser.set_defaults(func=_list_similarity_definitions)

    get_parser = nested.add_parser("get", help="Return one registry entry.")
    get_parser.add_argument("definition_id")
    get_parser.set_defaults(func=_get_similarity_definition)

    suggest_parser = nested.add_parser("suggest", help="Suggest entries for a dataset/task.")
    suggest_parser.add_argument("--dataset-description", required=True)
    suggest_parser.add_argument("--task-description", default="")
    suggest_parser.add_argument("--data-type", default="")
    suggest_parser.add_argument("--required-input", action="append", default=[])
    suggest_parser.add_argument("--top-k", type=int, default=5)
    suggest_parser.set_defaults(func=_suggest_similarity_definitions)

    script_parser = nested.add_parser("script", help="Return the example Python script for an entry.")
    script_parser.add_argument("definition_id")
    script_parser.set_defaults(func=_get_similarity_script)

    validate_parser = nested.add_parser("validate", help="Validate every bundled registry entry.")
    validate_parser.set_defaults(func=_validate_similarity_registry)


def _add_similarity_computation_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "similarity-computation",
        help="Search the literature-backed SPECTRA similarity computation strategy registry.",
    )
    nested = parser.add_subparsers(dest="similarity_computation_command", required=True)

    list_parser = nested.add_parser("list", help="List computation strategies.")
    list_parser.add_argument("--strategy-family", default="")
    list_parser.add_argument("--exactness", default="")
    list_parser.add_argument("--data-shape", default="")
    list_parser.add_argument("--status", default="")
    list_parser.set_defaults(func=_list_similarity_computation_strategies)

    get_parser = nested.add_parser("get", help="Return one computation strategy.")
    get_parser.add_argument("strategy_id")
    get_parser.set_defaults(func=_get_similarity_computation_strategy)

    suggest_parser = nested.add_parser(
        "suggest",
        help="Suggest computation strategies after defining a similarity axis.",
    )
    suggest_parser.add_argument("--dataset-description", required=True)
    suggest_parser.add_argument("--similarity-definition", default="")
    suggest_parser.add_argument("--data-shape", default="")
    suggest_parser.add_argument("--data-size", default="")
    suggest_parser.add_argument("--required-input", action="append", default=[])
    suggest_parser.add_argument("--top-k", type=int, default=5)
    suggest_parser.set_defaults(func=_suggest_similarity_computation_strategies)

    script_parser = nested.add_parser("script", help="Return the example Python script for a strategy.")
    script_parser.add_argument("strategy_id")
    script_parser.set_defaults(func=_get_similarity_computation_script)

    validate_parser = nested.add_parser("validate", help="Validate every bundled computation strategy.")
    validate_parser.set_defaults(func=_validate_similarity_computation_registry)


def _add_operating_point_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "operating-points",
        help="Search the literature-backed SPECTRA targeted operating-point registry.",
    )
    nested = parser.add_subparsers(dest="operating_point_command", required=True)

    list_parser = nested.add_parser("list", help="List targeted operating-point methods.")
    list_parser.add_argument("--method-family", default="")
    list_parser.add_argument("--scientific-unit", default="")
    list_parser.add_argument("--data-type", default="")
    list_parser.add_argument("--curve-region", default="")
    list_parser.add_argument("--status", default="")
    list_parser.set_defaults(func=_list_operating_point_methods)

    get_parser = nested.add_parser("get", help="Return one operating-point method.")
    get_parser.add_argument("method_id")
    get_parser.set_defaults(func=_get_operating_point_method)

    suggest_parser = nested.add_parser(
        "suggest",
        help="Suggest targeted operating-point methods for a deployment question.",
    )
    suggest_parser.add_argument("--dataset-description", required=True)
    suggest_parser.add_argument("--deployment-question", default="")
    suggest_parser.add_argument("--data-type", default="")
    suggest_parser.add_argument("--novelty-axis", default="")
    suggest_parser.add_argument("--required-input", action="append", default=[])
    suggest_parser.add_argument("--top-k", type=int, default=5)
    suggest_parser.set_defaults(func=_suggest_operating_point_methods)

    validate_parser = nested.add_parser("validate", help="Validate every bundled operating-point method.")
    validate_parser.set_defaults(func=_validate_operating_point_registry)


def _add_memory_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "memory",
        help="Search reusable SPECTRA run memories from prior audits.",
    )
    nested = parser.add_subparsers(dest="memory_command", required=True)

    list_parser = nested.add_parser("list", help="List SPECTRA memory entries.")
    list_parser.add_argument("--domain", default="")
    list_parser.add_argument("--data-type", default="")
    list_parser.add_argument("--model-family", default="")
    list_parser.add_argument("--status", default="")
    list_parser.set_defaults(func=_list_memory_entries)

    get_parser = nested.add_parser("get", help="Return one memory entry.")
    get_parser.add_argument("entry_id")
    get_parser.set_defaults(func=_get_memory_entry)

    search_parser = nested.add_parser("search", help="Search prior run memories.")
    search_parser.add_argument("--query", default="")
    search_parser.add_argument("--domain", default="")
    search_parser.add_argument("--data-type", default="")
    search_parser.add_argument("--model-family", default="")
    search_parser.add_argument("--model-name", default="")
    search_parser.add_argument("--tag", action="append", default=[])
    search_parser.add_argument("--top-k", type=int, default=5)
    search_parser.set_defaults(func=_search_memory_entries)

    suggest_parser = nested.add_parser("suggest", help="Suggest reusable memories for a new audit.")
    suggest_parser.add_argument("--model-description", required=True)
    suggest_parser.add_argument("--dataset-description", default="")
    suggest_parser.add_argument("--domain", default="")
    suggest_parser.add_argument("--top-k", type=int, default=5)
    suggest_parser.set_defaults(func=_suggest_memory_entries)

    render_parser = nested.add_parser("render", help="Render one memory entry as Markdown.")
    render_parser.add_argument("entry_id")
    render_parser.set_defaults(func=_render_memory_entry)

    validate_parser = nested.add_parser("validate", help="Validate every bundled memory entry.")
    validate_parser.set_defaults(func=_validate_memory_registry)


def _add_dataset_catalog_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "dataset-catalog",
        help="Search portable dataset resources for SPECTRA audits.",
    )
    nested = parser.add_subparsers(dest="dataset_catalog_command", required=True)

    list_parser = nested.add_parser("list", help="List dataset catalog entries.")
    list_parser.add_argument("--domain", default="")
    list_parser.add_argument("--data-type", default="")
    list_parser.add_argument("--model-family", default="")
    list_parser.add_argument("--access", default="")
    list_parser.add_argument("--status", default="")
    list_parser.set_defaults(func=_list_dataset_entries)

    get_parser = nested.add_parser("get", help="Return one dataset catalog entry.")
    get_parser.add_argument("dataset_id")
    get_parser.set_defaults(func=_get_dataset_entry)

    search_parser = nested.add_parser("search", help="Search dataset catalog entries.")
    search_parser.add_argument("--query", default="")
    search_parser.add_argument("--domain", default="")
    search_parser.add_argument("--data-type", default="")
    search_parser.add_argument("--model-family", default="")
    search_parser.add_argument("--required-field", action="append", default=[])
    search_parser.add_argument("--top-k", type=int, default=5)
    search_parser.set_defaults(func=_search_dataset_entries)

    suggest_parser = nested.add_parser("suggest", help="Suggest datasets for an audit question.")
    suggest_parser.add_argument("--audit-question", required=True)
    suggest_parser.add_argument("--model-description", default="")
    suggest_parser.add_argument("--domain", default="")
    suggest_parser.add_argument("--current-dataset-limitation", default="")
    suggest_parser.add_argument("--top-k", type=int, default=5)
    suggest_parser.set_defaults(func=_suggest_dataset_entries)

    render_parser = nested.add_parser("render", help="Render one dataset entry as Markdown.")
    render_parser.add_argument("dataset_id")
    render_parser.set_defaults(func=_render_dataset_entry)

    validate_parser = nested.add_parser("validate", help="Validate every bundled dataset entry.")
    validate_parser.set_defaults(func=_validate_dataset_catalog)


def _add_install_codex_skill_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "install-codex-skill",
        help="Install the packaged SPECTRA Codex skill into ~/.codex/skills/spectra.",
    )
    parser.add_argument(
        "--target",
        default="",
        help="Target skill directory. Defaults to ~/.codex/skills/spectra.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite an existing SKILL.md.")
    parser.set_defaults(func=_install_codex_skill)


def _add_agent_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "agent",
        help="Prepare or run an autonomous /spectra multi-role audit session.",
    )
    nested = parser.add_subparsers(dest="agent_command", required=True)

    def add_common_arguments(common: argparse.ArgumentParser) -> None:
        common.add_argument("--question", required=True)
        common.add_argument("--paper", "--model-paper", dest="model_paper", required=True)
        common.add_argument("--model-description", required=True)
        common.add_argument("--dataset-description", default="")
        common.add_argument("--domain", default="unknown")
        common.add_argument("--constraints", default="")
        common.add_argument(
            "--audit-scope",
            default="auto",
            choices=("auto", "paper-claim-audit", "beyond-paper-discovery"),
            help=(
                "auto infers scope from the question; paper-claim-audit stays close "
                "to the model paper; beyond-paper-discovery treats the paper as context "
                "and searches for external/public tests."
            ),
        )
        common.add_argument("--out", "--output-root", dest="output_root", required=True)
        common.add_argument("--client-capability", action="append", default=[])
        common.add_argument(
            "--max-rounds",
            type=int,
            default=8,
            help="Maximum role executions before stopping. Use 0 with --execute for no fixed cap.",
        )
        common.add_argument(
            "--no-max-rounds",
            action="store_true",
            help="Run until the terminal gate, role failure, or user interruption instead of a fixed round cap.",
        )

    prepare_parser = nested.add_parser(
        "prepare",
        help="Create session state and role prompts without executing any agent.",
    )
    add_common_arguments(prepare_parser)
    prepare_parser.set_defaults(func=_prepare_agent_session)

    run_parser = nested.add_parser(
        "run",
        help="Run a /spectra session through a configurable host-agent command.",
    )
    add_common_arguments(run_parser)
    run_parser.add_argument(
        "--agent-command-template",
        default="",
        help=(
            "Shell command template used for each role. Available fields: "
            "{prompt_path}, {write_scope}, {role}, {round}."
        ),
    )
    run_parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually execute role commands. Without this flag, run prepares a dry-run session only.",
    )
    run_parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue an existing session_state.json under --out instead of preparing a fresh session.",
    )
    run_parser.set_defaults(func=_run_agent_session_cli)


def _add_ask_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "ask",
        help="Run a human-friendly autonomous /spectra audit from a question and obvious artifacts.",
    )
    parser.add_argument(
        "question_words",
        nargs="*",
        help="Audit question. Quote it as one string, or pass --question.",
    )
    parser.add_argument("--question", dest="question_opt", default="")
    parser.add_argument("--paper", "--model-paper", dest="model_paper", default="")
    parser.add_argument("--model", "--model-artifact", dest="model_artifact", default="")
    parser.add_argument("--model-description", default="")
    parser.add_argument("--dataset", "--data", dest="dataset", default="")
    parser.add_argument("--dataset-description", default="")
    parser.add_argument("--domain", default="auto")
    parser.add_argument(
        "--audit-scope",
        default="auto",
        choices=("auto", "paper-claim-audit", "beyond-paper-discovery"),
    )
    parser.add_argument(
        "--ask-mode",
        default="auto",
        choices=("auto", "audit", "splits"),
        help=(
            "auto routes split-construction requests to a focused SPECTRA split workflow "
            "and model-generalizability requests to the autonomous audit loop."
        ),
    )
    parser.add_argument("--constraints", default="")
    parser.add_argument(
        "--out",
        "--output-root",
        dest="output_root",
        default="",
        help=(
            "Output root. Defaults to SPECTRA_SCRATCH_ROOT when set, otherwise "
            "XDG_CACHE_HOME/spectra/runs or ~/.cache/spectra/runs."
        ),
    )
    parser.add_argument(
        "--scratch-root",
        default="",
        help="Root used when --out is omitted. Overrides the portable scratch-root default.",
    )
    parser.add_argument(
        "--agent-command-template",
        default="",
        help="Advanced override for the role runner command. Defaults to codex exec.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare prompts only. By default, ask executes the autonomous run.",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=0,
        help="Maximum role executions. Default 0 means no fixed cap.",
    )
    parser.add_argument(
        "--allow-prior-memory",
        action="store_true",
        help="Allow the agent to consult prior SPECTRA memories/catalog entries as priors.",
    )
    parser.set_defaults(func=_ask_agent_session_cli)


def _slugify(value: str, max_len: int = 72) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return (slug or "spectra_question")[:max_len].strip("_") or "spectra_question"


def _path_summary(path_text: str) -> str:
    if not path_text:
        return ""
    path = Path(path_text).expanduser()
    if not path.exists():
        return f"{path_text} (path not found at invocation time; agent may resolve it)"
    if path.is_dir():
        children = list(path.iterdir())
        preview = ", ".join(sorted(child.name for child in children[:8]))
        return f"{path} (directory, {len(children)} immediate entries: {preview})"
    size = path.stat().st_size
    return f"{path} (file, {size} bytes)"


def _infer_simple_domain(question: str, model_text: str, dataset_text: str) -> str:
    text = " ".join([question, model_text, dataset_text]).lower()
    if any(token in text for token in ["caduceus", "dna", "genomic", "genome", "ccre", "encode", "regulatory"]):
        return "regulatory-genomics"
    if any(token in text for token in ["molecule", "smiles", "drug", "chemical"]):
        return "molecular-ml"
    if any(token in text for token in ["single-cell", "single cell", "perturbation", "scrna"]):
        return "single-cell-perturbation"
    return "unknown"


def _default_scratch_root() -> Path:
    configured = os.environ.get("SPECTRA_SCRATCH_ROOT")
    if configured:
        return Path(configured).expanduser()
    cache_root = os.environ.get("XDG_CACHE_HOME")
    if cache_root:
        return Path(cache_root).expanduser() / "spectra" / "runs"
    return Path.home() / ".cache" / "spectra" / "runs"


def _default_output_root(question: str, scratch_root: str) -> str:
    root = Path(scratch_root).expanduser() if scratch_root else _default_scratch_root()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return str(root / f"{stamp}_{_slugify(question, max_len=48)}")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_agent_command_template(output_root: str, extra_paths: Sequence[str]) -> str:
    add_dirs = [str(_repo_root()), output_root]
    for raw in extra_paths:
        if not raw:
            continue
        path = Path(raw).expanduser()
        add_dirs.append(str(path if path.is_dir() else path.parent))
    unique_dirs = []
    seen = set()
    for directory in add_dirs:
        if directory and directory not in seen:
            unique_dirs.append(directory)
            seen.add(directory)
    add_dir_flags = " ".join(f"--add-dir {shlex.quote(directory)}" for directory in unique_dirs)
    return (
        "codex exec --skip-git-repo-check -C {write_scope} "
        f"{add_dir_flags} "
        "-s danger-full-access -c approval_policy=\"never\" - < {prompt_path}"
    )


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(content)
        if content and not content.endswith("\n"):
            handle.write("\n")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")


def _install_codex_skill(args: argparse.Namespace) -> int:
    target_dir = Path(args.target).expanduser() if args.target else Path.home() / ".codex" / "skills" / "spectra"
    target_path = target_dir / "SKILL.md"
    if target_path.exists() and not args.force:
        raise FileExistsError(
            f"{target_path} already exists. Re-run with --force to overwrite it."
        )
    data = pkgutil.get_data("spectrae", "agent_adapters/codex/SKILL.md")
    if data is None:
        raise FileNotFoundError("Packaged Codex skill template is missing.")
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path.write_text(data.decode("utf-8"), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "installed",
                "target": str(target_path),
                "next_step": "Start a fresh Codex session and ask `/spectra ...`.",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _infer_ask_mode(args: argparse.Namespace, question: str) -> str:
    if args.ask_mode != "auto":
        return args.ask_mode

    text = " ".join([question, args.constraints, args.dataset_description, args.domain]).lower()
    split_markers = [
        "spectra split",
        "spectra splits",
        "/spectra split",
        "/spectra splits",
        "construct splits",
        "generate splits",
        "create splits",
        "build splits",
        "split candidates",
        "split assignment",
        "split assignments",
        "spectral split",
        "spectral splits",
    ]
    split_action_markers = ["construct", "generate", "create", "build", "make", "produce", "running spectra"]
    model_audit_markers = [
        "audit model",
        "audit a model",
        "model generalization",
        "model generalizability",
        "generalize",
        "generalizes",
        "checkpoint",
        "model checkpoint",
        "paper claim",
        "paper claims",
        "foundation model",
    ]

    asks_for_splits = any(marker in text for marker in split_markers) or (
        "split" in text and "spectra" in text and any(marker in text for marker in split_action_markers)
    )
    asks_for_model_audit = any(marker in text for marker in model_audit_markers)
    has_model_artifact = bool(args.model_artifact or args.model_paper)

    if asks_for_splits and not (asks_for_model_audit and has_model_artifact):
        return "splits"
    return "audit"


def _build_split_construction_prompt(
    question: str,
    output_root: str,
    dataset_description: str,
    domain: str,
    constraints: str,
) -> str:
    return "\n".join(
        [
            "You are running focused SPECTRA split construction.",
            "",
            "This is not a broad model-generalizability audit. Do not run the Distiller/Investigator/Auditor loop, do not extract paper claims, and do not route to Dataset Scout or Dataset Fetcher.",
            "",
            "User question:",
            question,
            "",
            "Dataset:",
            dataset_description,
            "",
            "Domain:",
            domain,
            "",
            "Output root:",
            output_root,
            "",
            "Constraints:",
            constraints,
            "",
            "Canonical SPECTRA split-construction procedure:",
            "1. Inspect the dataset schema and identify row id, scientific unit/input column, label column if present, and task type.",
            "2. Choose a scientific similarity definition before computing similarities. Prefer the bundled registry first: run `spectra similarity-definitions suggest --dataset-description ... --task-description ...`, inspect the recommended entries, and record the selected definition id. If you choose a custom definition, document why the registry entry was insufficient.",
            "3. Choose a pairwise similarity computation strategy after fixing the definition. Prefer the bundled computation registry first: run `spectra similarity-computation suggest --dataset-description ... --similarity-definition ...`, inspect the recommended entries, and record the selected strategy id. If you choose a custom strategy, document why.",
            "4. Compute pairwise similarities according to the chosen definition and strategy. Write the pairwise similarities, nearest-neighbor table, or thresholded property/similarity graph used by SPECTRA.",
            "5. Construct SPECTRA split candidates from the pairwise similarity graph or spectral axis using prospective dataset features only; do not use target-model errors.",
            "6. Sweep at least three spectral parameters or thresholds when feasible, rather than accepting one graph setting.",
            "7. Verify that train-test similarity decreases across the generated split levels. If it does not decrease, mark the split candidate as invalid or diagnostic-only.",
            "8. If labels are available, train and test a simple fixed baseline across the generated splits and report whether performance decreases as train-test similarity decreases.",
            "9. Select a recommended split or state why no split is yet validated.",
            "",
            "Domain-specific guidance:",
            "- If the domain is molecular or SMILES-like, use RDKit when available: canonical SMILES, Morgan fingerprints, Tanimoto similarity, and Bemis-Murcko scaffolds.",
            "- The registry-backed choice is still required even when the eventual implementation is a domain-specific custom script.",
            "",
            "Required outputs inside the output root:",
            "- `split_construction_manifest.json`",
            "- `similarity_definition_selection.json`",
            "- `similarity_computation_selection.json`",
            "- `pairwise_similarity/` or `property_similarity_graph/`",
            "- `split_assignments/`",
            "- `diagnostics/train_test_similarity.csv`",
            "- `diagnostics/spectral_parameter_sweep.csv` when a parameter sweep is feasible",
            "- `baseline_validation/` when labels are available",
            "- `split_construction_report.md`",
            "",
            "Final response should summarize the generated split files, spectral parameters, similarity progression, baseline validation result if run, blockers, and artifact paths.",
        ]
    )


def _run_split_construction_cli(
    args: argparse.Namespace,
    question: str,
    output_root: str,
    dataset_description: str,
    domain: str,
    constraints: str,
) -> int:
    root = Path(output_root)
    write_scope = root / "split_construction"
    prompt_path = root / "split_construction_prompt.md"
    log_path = root / "split_construction.log"
    manifest_path = root / "split_construction_manifest.json"
    root.mkdir(parents=True, exist_ok=True)
    write_scope.mkdir(parents=True, exist_ok=True)

    prompt = _build_split_construction_prompt(
        question=question,
        output_root=str(root),
        dataset_description=dataset_description,
        domain=domain,
        constraints=constraints,
    )
    _write_text(prompt_path, prompt)
    _write_json(
        manifest_path,
        {
            "mode": "spectra_split_construction",
            "status": "prepared",
            "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "question": question,
            "dataset_description": dataset_description,
            "domain": domain,
            "constraints": constraints,
            "output_root": str(root),
            "write_scope": str(write_scope),
            "prompt_path": str(prompt_path),
            "log_path": str(log_path),
            "uses_general_audit_loop": False,
        },
    )

    result = {
        "human_friendly_invocation": True,
        "ask_mode": "splits",
        "status": "prepared_not_launched" if args.dry_run else "running",
        "output_root": str(root),
        "write_scope": str(write_scope),
        "prompt_path": str(prompt_path),
        "manifest": str(manifest_path),
        "log_path": str(log_path),
        "uses_general_audit_loop": False,
    }
    if args.dry_run:
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.dataset and not args.agent_command_template:
        try:
            from .focused_splits import UnsupportedFocusedSplitDataset, run_focused_split_construction

            split_summary = run_focused_split_construction(
                dataset_path=args.dataset,
                output_root=str(root),
                question=question,
                dataset_description=dataset_description,
                domain=domain,
                constraints=constraints,
            )
            result["status"] = "completed"
            result["returncode"] = 0
            result["split_constructor"] = split_summary.get("constructor", "native_focused_split_constructor")
            result["split_summary"] = split_summary
            _write_json(
                manifest_path,
                {
                    **json.loads(manifest_path.read_text(encoding="utf-8")),
                    "status": "completed",
                    "completed_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                    "returncode": 0,
                    "split_constructor": result["split_constructor"],
                    "native_split_summary": split_summary,
                },
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        except UnsupportedFocusedSplitDataset:
            pass

    command_template = args.agent_command_template or _default_agent_command_template(
        str(root),
        [args.dataset],
    )
    command = command_template.format(
        prompt_path=shlex.quote(str(prompt_path)),
        write_scope=shlex.quote(str(write_scope)),
        role="split_constructor",
        round=1,
    )
    with log_path.open("w", encoding="utf-8") as log_handle:
        completed = subprocess.run(
            command,
            shell=True,
            cwd=str(write_scope),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    result["status"] = "completed" if completed.returncode == 0 else "failed"
    result["returncode"] = completed.returncode
    result["command"] = command
    _write_json(
        manifest_path,
        {
            **json.loads(manifest_path.read_text(encoding="utf-8")),
            "status": result["status"],
            "completed_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "returncode": completed.returncode,
            "command": command,
        },
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return int(completed.returncode)


def _build_default_constraints(args: argparse.Namespace, output_root: str) -> str:
    parts = [
        "This is a human-friendly /spectra ask invocation. Infer the domain, audit scope, scientific unit, dataset schema, split strategy, and needed controls from the question and provided artifacts.",
        "Use the provided model/paper/dataset paths, this session root, and public/local resources acquired during this session when needed.",
        "Do not use /tmp; use the session root, SPECTRA_SCRATCH_ROOT, or another explicit scratch directory for caches, downloads, generated datasets, and model outputs.",
        "Do not use scripted applicability runners or prior generated reports/results as evidence.",
        "Use a cheap-first behavioral runtime policy: start with a bounded leakage-aware slice, simple controls, cached/chunked representations, and non-iterative probes or deterministic baselines before launching expensive optimized classifiers, fine-tuning, all-pairs graphs, or full-dataset runs.",
        "If a heavier solver, model fit, or full-scale computation is needed, declare the time/resource budget, fallback criterion, and cheaper fallback before launching; after one runtime failure or timeout, switch to the fallback rather than retrying the same slow method.",
        "Decide autonomously what to inspect and run, and continue until a bounded applicability recommendation is supported or a concrete blocker is documented.",
    ]
    if args.allow_prior_memory:
        parts.append(
            "Prior SPECTRA memories and dataset catalog entries may be used as priors, but all evidence for the final claim must be regenerated or validated inside this session root."
        )
    else:
        parts.append(
            "Do not read prior SPECTRA run artifacts or model-specific memory entries unless they are copied into this session root by the current run."
        )
    parts.append(f"Session root: {output_root}")
    if args.constraints:
        parts.append(f"Additional user constraints: {args.constraints}")
    return " ".join(parts)


def _build_split_default_constraints(args: argparse.Namespace, output_root: str) -> str:
    parts = [
        "This is a focused SPECTRA split-construction invocation, not a broad model-generalizability audit.",
        "Use the provided dataset path as the starting dataset and write generated split artifacts under the session root.",
        "Do not use /tmp; use the session root, SPECTRA_SCRATCH_ROOT, or another explicit scratch directory for caches, generated datasets, and temporary files.",
        "Do not read prior SPECTRA run artifacts as evidence unless the user explicitly supplies them as inputs.",
        "Begin by choosing and recording a similarity definition; prefer `spectra similarity-definitions suggest` before inventing a custom axis.",
        "Then choose and record a pairwise computation strategy; prefer `spectra similarity-computation suggest` before inventing a custom computation path.",
        "Compute the pairwise similarities or thresholded property graph before constructing splits.",
        "Generate prospective split candidates from dataset features only; do not use target-model errors or held-out labels to define split membership.",
        "When labels are available, use them only for balance diagnostics and baseline validation after split construction.",
        "Prefer deterministic, reproducible split construction and a spectral parameter sweep with at least three usable levels when feasible.",
        f"Session root: {output_root}",
    ]
    if args.constraints:
        parts.append(f"Additional user constraints: {args.constraints}")
    return " ".join(parts)


def _ask_agent_session_cli(args: argparse.Namespace) -> int:
    from .agent_orchestrator import SpectraAgentSessionConfig, run_agent_session

    question = (args.question_opt or " ".join(args.question_words)).strip()
    if not question:
        raise ValueError("spectra ask requires a question, either positionally or with --question")

    output_root = args.output_root or _default_output_root(question, args.scratch_root)
    model_description = args.model_description.strip()
    if not model_description:
        model_description = (
            "Frozen model/checkpoint artifact provided by the user: "
            f"{_path_summary(args.model_artifact) or 'no explicit model artifact provided'}."
        )
    dataset_description = args.dataset_description.strip()
    if not dataset_description:
        dataset_description = (
            "Raw user-provided dataset artifact for this audit question: "
            f"{_path_summary(args.dataset) or 'no explicit dataset artifact provided'}."
        )
    if args.dataset:
        dataset_description += " Treat this as the starting dataset; inspect schema/provenance and construct derived windows, splits, labels, or controls only as needed."

    domain = (
        _infer_simple_domain(question, model_description, dataset_description)
        if args.domain == "auto"
        else args.domain
    )
    ask_mode = _infer_ask_mode(args, question)
    model_paper = args.model_paper or "No explicit model paper path was provided; infer model-paper context from the supplied model artifact and public/local sources if needed."
    if ask_mode == "splits":
        constraints = _build_split_default_constraints(args, output_root)
        return _run_split_construction_cli(
            args=args,
            question=question,
            output_root=output_root,
            dataset_description=dataset_description,
            domain=domain,
            constraints=constraints,
        )
    constraints = _build_default_constraints(args, output_root)
    command_template = args.agent_command_template or _default_agent_command_template(
        output_root,
        [args.model_paper, args.model_artifact, args.dataset],
    )
    config = SpectraAgentSessionConfig(
        question=question,
        model_paper=model_paper,
        model_description=model_description,
        dataset_description=dataset_description,
        domain=domain,
        constraints=constraints,
        audit_scope=args.audit_scope,
        output_root=output_root,
        client_capabilities=["filesystem", "network", "autonomous_runner"],
        max_rounds=None if args.max_rounds == 0 else args.max_rounds,
        agent_command_template=command_template,
        dry_run=args.dry_run,
        resume=False,
    )
    result = run_agent_session(config)
    result["human_friendly_invocation"] = True
    result["ask_mode"] = ask_mode
    result["inferred_domain"] = domain
    result["output_root"] = output_root
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _agent_config_from_args(args: argparse.Namespace, dry_run: bool):
    from .agent_orchestrator import SpectraAgentSessionConfig

    return SpectraAgentSessionConfig(
        question=args.question,
        model_paper=args.model_paper,
        model_description=args.model_description,
        dataset_description=args.dataset_description,
        domain=args.domain,
        constraints=args.constraints,
        audit_scope=args.audit_scope,
        output_root=args.output_root,
        client_capabilities=args.client_capability,
        max_rounds=None if getattr(args, "no_max_rounds", False) or args.max_rounds == 0 else args.max_rounds,
        agent_command_template=getattr(args, "agent_command_template", ""),
        dry_run=dry_run,
        resume=getattr(args, "resume", False),
    )


def _prepare_agent_session(args: argparse.Namespace) -> int:
    from .agent_orchestrator import prepare_agent_session

    result = prepare_agent_session(_agent_config_from_args(args, dry_run=True))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _run_agent_session_cli(args: argparse.Namespace) -> int:
    from .agent_orchestrator import run_agent_session

    result = run_agent_session(_agent_config_from_args(args, dry_run=not args.execute))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _list_similarity_definitions(args: argparse.Namespace) -> int:
    from .similarity_registry import list_similarity_definitions

    result = list_similarity_definitions(
        data_type=args.data_type or None,
        scientific_unit=args.scientific_unit or None,
        task=args.task or None,
        status=args.status or None,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _get_similarity_definition(args: argparse.Namespace) -> int:
    from .similarity_registry import load_similarity_definition

    print(json.dumps(load_similarity_definition(args.definition_id), indent=2, sort_keys=True))
    return 0


def _suggest_similarity_definitions(args: argparse.Namespace) -> int:
    from .similarity_registry import suggest_similarity_definitions

    result = suggest_similarity_definitions(
        dataset_description=args.dataset_description,
        task_description=args.task_description,
        data_type=args.data_type,
        required_inputs=args.required_input,
        top_k=args.top_k,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _get_similarity_script(args: argparse.Namespace) -> int:
    from .similarity_registry import get_similarity_example_script

    print(json.dumps(get_similarity_example_script(args.definition_id), indent=2, sort_keys=True))
    return 0


def _validate_similarity_registry(args: argparse.Namespace) -> int:
    from .similarity_registry import validate_similarity_registry

    result = validate_similarity_registry()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] else 1


def _list_similarity_computation_strategies(args: argparse.Namespace) -> int:
    from .similarity_computation_registry import list_similarity_computation_strategies

    result = list_similarity_computation_strategies(
        strategy_family=args.strategy_family or None,
        exactness=args.exactness or None,
        data_shape=args.data_shape or None,
        status=args.status or None,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _get_similarity_computation_strategy(args: argparse.Namespace) -> int:
    from .similarity_computation_registry import load_similarity_computation_strategy

    print(json.dumps(load_similarity_computation_strategy(args.strategy_id), indent=2, sort_keys=True))
    return 0


def _suggest_similarity_computation_strategies(args: argparse.Namespace) -> int:
    from .similarity_computation_registry import suggest_similarity_computation_strategies

    result = suggest_similarity_computation_strategies(
        dataset_description=args.dataset_description,
        similarity_definition=args.similarity_definition,
        data_shape=args.data_shape,
        data_size=args.data_size,
        required_inputs=args.required_input,
        top_k=args.top_k,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _get_similarity_computation_script(args: argparse.Namespace) -> int:
    from .similarity_computation_registry import get_similarity_computation_example_script

    print(json.dumps(get_similarity_computation_example_script(args.strategy_id), indent=2, sort_keys=True))
    return 0


def _validate_similarity_computation_registry(args: argparse.Namespace) -> int:
    from .similarity_computation_registry import validate_similarity_computation_registry

    result = validate_similarity_computation_registry()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] else 1


def _list_operating_point_methods(args: argparse.Namespace) -> int:
    from .operating_point_registry import list_operating_point_methods

    result = list_operating_point_methods(
        method_family=args.method_family or None,
        scientific_unit=args.scientific_unit or None,
        data_type=args.data_type or None,
        curve_region=args.curve_region or None,
        status=args.status or None,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _get_operating_point_method(args: argparse.Namespace) -> int:
    from .operating_point_registry import load_operating_point_method

    print(json.dumps(load_operating_point_method(args.method_id), indent=2, sort_keys=True))
    return 0


def _suggest_operating_point_methods(args: argparse.Namespace) -> int:
    from .operating_point_registry import suggest_operating_point_methods

    result = suggest_operating_point_methods(
        dataset_description=args.dataset_description,
        deployment_question=args.deployment_question,
        data_type=args.data_type,
        novelty_axis=args.novelty_axis,
        required_inputs=args.required_input,
        top_k=args.top_k,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _validate_operating_point_registry(args: argparse.Namespace) -> int:
    from .operating_point_registry import validate_operating_point_registry

    result = validate_operating_point_registry()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] else 1


def _list_memory_entries(args: argparse.Namespace) -> int:
    from .memory_registry import list_memory_entries

    result = list_memory_entries(
        domain=args.domain or None,
        data_type=args.data_type or None,
        model_family=args.model_family or None,
        status=args.status or None,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _get_memory_entry(args: argparse.Namespace) -> int:
    from .memory_registry import load_memory_entry

    print(json.dumps(load_memory_entry(args.entry_id), indent=2, sort_keys=True))
    return 0


def _search_memory_entries(args: argparse.Namespace) -> int:
    from .memory_registry import search_memory_entries

    result = search_memory_entries(
        query=args.query,
        domain=args.domain,
        data_type=args.data_type,
        model_family=args.model_family,
        model_name=args.model_name,
        tags=args.tag,
        top_k=args.top_k,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _suggest_memory_entries(args: argparse.Namespace) -> int:
    from .memory_registry import suggest_reusable_memory

    result = suggest_reusable_memory(
        model_description=args.model_description,
        dataset_description=args.dataset_description,
        domain=args.domain,
        top_k=args.top_k,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _render_memory_entry(args: argparse.Namespace) -> int:
    from .memory_registry import render_memory_entry

    print(json.dumps(render_memory_entry(args.entry_id), indent=2, sort_keys=True))
    return 0


def _validate_memory_registry(args: argparse.Namespace) -> int:
    from .memory_registry import validate_memory_registry

    result = validate_memory_registry()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] else 1


def _list_dataset_entries(args: argparse.Namespace) -> int:
    from .dataset_catalog_registry import list_dataset_entries

    result = list_dataset_entries(
        domain=args.domain or None,
        data_type=args.data_type or None,
        model_family=args.model_family or None,
        access=args.access or None,
        status=args.status or None,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _get_dataset_entry(args: argparse.Namespace) -> int:
    from .dataset_catalog_registry import load_dataset_entry

    print(json.dumps(load_dataset_entry(args.dataset_id), indent=2, sort_keys=True))
    return 0


def _search_dataset_entries(args: argparse.Namespace) -> int:
    from .dataset_catalog_registry import search_dataset_entries

    result = search_dataset_entries(
        query=args.query,
        domain=args.domain,
        data_type=args.data_type,
        model_family=args.model_family,
        required_fields=args.required_field,
        top_k=args.top_k,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _suggest_dataset_entries(args: argparse.Namespace) -> int:
    from .dataset_catalog_registry import suggest_dataset_entries

    result = suggest_dataset_entries(
        audit_question=args.audit_question,
        model_description=args.model_description,
        domain=args.domain,
        current_dataset_limitation=args.current_dataset_limitation,
        top_k=args.top_k,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _render_dataset_entry(args: argparse.Namespace) -> int:
    from .dataset_catalog_registry import render_dataset_entry

    print(json.dumps(render_dataset_entry(args.dataset_id), indent=2, sort_keys=True))
    return 0


def _validate_dataset_catalog(args: argparse.Namespace) -> int:
    from .dataset_catalog_registry import validate_dataset_catalog

    result = validate_dataset_catalog()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] else 1


def _add_audit_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "audit",
        help="Run a deterministic SPECTRA audit from train/eval prediction files.",
    )
    parser.add_argument(
        "--domain",
        default="generic",
        help="Domain label or adapter. Use 'molecules' only when asking SPECTRA to compute Morgan similarity.",
    )
    parser.add_argument("--train", help="Training CSV path.")
    parser.add_argument(
        "--eval",
        required=True,
        help="Evaluation prediction CSV path containing raw labels and predictions.",
    )
    parser.add_argument("--out", required=True, help="Output directory for audit artifacts.")
    parser.add_argument(
        "--scientific-unit",
        default="sample",
        help="Scientific unit of generalization, e.g. sample, molecule, sequence, perturbation.",
    )
    parser.add_argument(
        "--mode",
        choices=("axis", "pairwise", "adapter"),
        default="axis",
        help="axis consumes a per-eval spectral axis; pairwise consumes a train-eval similarity graph; adapter computes a domain-specific axis.",
    )
    parser.add_argument("--unit-col", help="Optional evaluation-unit identifier column.")
    parser.add_argument("--axis-col", help="Per-evaluation-unit measured distance/overlap axis.")
    parser.add_argument(
        "--axis-type",
        default="similarity",
        choices=("similarity", "distance"),
        help="Whether larger axis values mean more train-like similarity or more novel distance.",
    )
    parser.add_argument("--axis-name", default="spectral_axis")
    parser.add_argument(
        "--pairwise-similarity",
        help="Long CSV of pairwise train-eval similarities produced by an agent or domain adapter.",
    )
    parser.add_argument("--eval-id-col", default="sample_id")
    parser.add_argument("--train-id-col", default="train_id")
    parser.add_argument("--similarity-eval-id-col", default="sample_id")
    parser.add_argument("--similarity-train-id-col", default="train_id")
    parser.add_argument("--similarity-col", default="similarity")
    parser.add_argument("--smiles-col", default="smiles", help="SMILES column in train and eval CSVs.")
    parser.add_argument(
        "--train-target-col",
        default="y",
        help="Training target column, used for target-support diagnostics when present.",
    )
    parser.add_argument(
        "--eval-target-col",
        "--target-col",
        default="y_true",
        help="Evaluation ground-truth target column.",
    )
    parser.add_argument("--pred-col", default="y_pred", help="Evaluation prediction column.")
    parser.add_argument(
        "--sample-id-col",
        help="Optional sample identifier column in the evaluation CSV.",
    )
    parser.add_argument(
        "--thresholds",
        help="Comma-separated thresholds for nested spectral subsets. Defaults to quantile thresholds.",
    )
    parser.add_argument(
        "--quantile-bins",
        type=int,
        default=5,
        help="Number of quantile-based spectral subsets when thresholds are not provided.",
    )
    parser.add_argument("--fingerprint-radius", type=int, default=2)
    parser.add_argument("--fingerprint-bits", type=int, default=1024)
    parser.add_argument(
        "--no-rdkit",
        action="store_true",
        help="Use transparent SMILES n-gram Jaccard instead of RDKit Morgan Tanimoto.",
    )
    parser.set_defaults(func=_run_audit)


def _run_audit(args: argparse.Namespace) -> int:
    domain = args.domain.strip().lower().replace("-", "_")
    mode = args.mode
    if mode == "adapter" or (domain in {"molecules", "molecule", "small_molecule"} and not args.axis_col and not args.pairwise_similarity):
        if domain not in {"molecules", "molecule", "small_molecule"}:
            raise ValueError("No adapter is implemented for domain: %s" % args.domain)
        if not args.train:
            raise ValueError("--train is required for the molecule adapter")
        result = run_molecule_audit(
            MoleculeAuditConfig(
                train_path=args.train,
                eval_path=args.eval,
                output_dir=args.out,
                smiles_col=args.smiles_col,
                train_target_col=args.train_target_col,
                eval_target_col=args.eval_target_col,
                pred_col=args.pred_col,
                sample_id_col=args.sample_id_col,
                thresholds=parse_threshold_string(args.thresholds or "1.0,0.8,0.7,0.6,0.5"),
                fingerprint_radius=args.fingerprint_radius,
                fingerprint_bits=args.fingerprint_bits,
                prefer_rdkit=not args.no_rdkit,
            )
        )
    elif mode == "pairwise" or args.pairwise_similarity:
        if not args.pairwise_similarity:
            raise ValueError("--pairwise-similarity is required for pairwise mode")
        result = run_pairwise_similarity_audit(
            PairwiseSimilarityAuditConfig(
                eval_path=args.eval,
                similarity_path=args.pairwise_similarity,
                output_dir=args.out,
                target_col=args.eval_target_col,
                pred_col=args.pred_col,
                eval_id_col=args.eval_id_col,
                similarity_eval_id_col=args.similarity_eval_id_col,
                similarity_train_id_col=args.similarity_train_id_col,
                similarity_col=args.similarity_col,
                domain=domain,
                scientific_unit=args.scientific_unit,
                train_path=args.train,
                train_id_col=args.train_id_col,
                train_target_col=args.train_target_col,
                axis_name=args.axis_name,
                thresholds=parse_optional_threshold_string(args.thresholds),
                quantile_bins=args.quantile_bins,
            )
        )
    else:
        if not args.axis_col:
            raise ValueError(
                "Generic SPECTRA audits require --axis-col or --pairwise-similarity. "
                "Use --mode adapter --domain molecules only for the molecule example adapter."
            )
        result = run_axis_audit(
            SpectralAxisAuditConfig(
                eval_path=args.eval,
                output_dir=args.out,
                target_col=args.eval_target_col,
                pred_col=args.pred_col,
                axis_col=args.axis_col,
                axis_type=args.axis_type,
                axis_name=args.axis_name,
                domain=domain,
                scientific_unit=args.scientific_unit,
                train_path=args.train,
                train_target_col=args.train_target_col,
                unit_col=args.unit_col or args.sample_id_col,
                thresholds=parse_optional_threshold_string(args.thresholds),
                quantile_bins=args.quantile_bins,
            )
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spectra",
        description="SPECTRA deterministic audit engine.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_ask_parser(subparsers)
    _add_audit_parser(subparsers)
    _add_agent_parser(subparsers)
    _add_similarity_parser(subparsers)
    _add_similarity_computation_parser(subparsers)
    _add_operating_point_parser(subparsers)
    _add_memory_parser(subparsers)
    _add_dataset_catalog_parser(subparsers)
    _add_install_codex_skill_parser(subparsers)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
