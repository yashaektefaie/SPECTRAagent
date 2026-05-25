"""Autonomous /spectra audit session orchestration.

This module provides the runtime shell around the MCP session contract. It does
not embed an LLM. Instead, it prepares role prompts and can call a configurable
host-agent command for each role. That makes the orchestration explicit and
testable while leaving Codex, Claude Code, or another agent runtime responsible
for actually executing the prompts.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .scientific_skill_mcp import start_spectra_audit_session


@dataclass
class SpectraAgentSessionConfig:
    question: str
    model_paper: str
    model_description: str
    output_root: str
    dataset_description: str = ""
    domain: str = "unknown"
    constraints: str = ""
    audit_scope: str = "auto"
    client_capabilities: Optional[List[str]] = None
    max_rounds: Optional[int] = 8
    agent_command_template: str = ""
    dry_run: bool = True
    resume: bool = False


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(content)
        if content and not content.endswith("\n"):
            handle.write("\n")


def _load_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object at %s" % path)
    return value


def _role_prompt(
    role: str,
    round_index: int,
    write_scope: Path,
    session_state_path: Path,
    session_contract: Dict[str, Any],
    handoff_path: Optional[Path] = None,
) -> str:
    role_graph = session_contract.get("role_graph", {})
    role_spec = role_graph.get(role, {})
    base_prompt = str(role_spec.get("prompt") or "")
    shared = session_contract.get("shared_context", {})
    runtime_policy = session_contract.get("runtime_probe_policy", {})
    isolation = [
        "Use only the session output root and user-provided model/data/paper paths unless a role handoff explicitly authorizes public/local resource acquisition.",
        "Do not read prior SPECTRA run artifacts unless they are inside this session root.",
        "Write only inside the assigned write scope.",
        "If a blocker is encountered, record the exact command, exception, recovery attempts, and next executable handoff.",
    ]
    if role == "investigator":
        expected = [
            "observations.md",
            "hypothesis_ledger.json",
            "why_this_next_experiment.md",
            "runtime_budget_and_fallbacks.md",
            "belief_update.md",
            "investigator_checkpoint.md",
            "distiller_handoff.json",
        ]
    elif role == "distiller":
        expected = [
            "distiller_report.md",
            "routing_decision.json",
            "investigator_handoff.json if routing back",
            "paper_ready_checkpoint.md or final synthesis if terminal",
            "overclaim_guardrails.md",
        ]
    elif role == "dataset_scout":
        expected = ["candidate_resources.csv", "scout_decision.json", "resource_risk_log.md"]
    elif role == "dataset_constructor":
        expected = [
            "constructed_dataset_manifest.json",
            "mapping_validation.json",
            "leakage_audit.json",
            "constructor_handoff.json",
        ]
    else:
        expected = role_spec.get("writes", [])

    parts = [
        "You are the SPECTRA %s for round %03d." % (role.replace("_", " ").title(), round_index),
        "",
        "Session state: %s" % session_state_path,
        "Write scope: %s" % write_scope,
        "",
        "User question: %s" % shared.get("question", ""),
        "Question mode: %s" % shared.get("question_mode", ""),
        "Model paper/reference: %s" % shared.get("model_paper_reference", ""),
        "Model description: %s" % shared.get("model_description", ""),
        "Dataset description: %s" % shared.get("dataset_description", ""),
        "Domain: %s" % shared.get("domain", ""),
        "Audit scope: %s" % shared.get("audit_scope", ""),
        "Audit depth: %s" % shared.get("audit_depth", ""),
        "User constraints: %s" % shared.get("constraints", ""),
        "",
        "User constraints override default role heuristics when they conflict.",
    ]
    if runtime_policy:
        parts.extend(
            [
                "",
                "Cheap-first behavioral runtime policy:",
                str(runtime_policy.get("purpose") or ""),
            ]
        )
        parts.extend("- %s" % item for item in runtime_policy.get("required_order", []))
    parts.extend(
        [
            "",
            "Role instructions:",
            base_prompt,
            "",
            "Isolation and quality rules:",
        ]
    )
    parts.extend("- %s" % item for item in isolation)
    if handoff_path is not None:
        parts.extend(["", "Read this handoff before starting: %s" % handoff_path])
    parts.extend(["", "Required outputs:"])
    parts.extend("- %s" % item for item in expected)
    parts.extend(
        [
            "",
            "Final response should summarize the route decision, key evidence, blockers, and artifact paths.",
        ]
    )
    return "\n".join(parts)


def _initial_manifest(config: SpectraAgentSessionConfig, session_contract: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "mode": "spectra_agent_session",
        "created_utc": _utc_now(),
        "status": "prepared",
        "output_root": config.output_root,
        "dry_run": config.dry_run,
        "max_rounds": config.max_rounds,
        "round_limit": "unbounded" if config.max_rounds is None else config.max_rounds,
        "agent_command_template_present": bool(config.agent_command_template),
        "session_contract_status": session_contract.get("status"),
        "orchestration_mode": session_contract.get("orchestration_mode"),
        "question_mode": session_contract.get("shared_context", {}).get("question_mode"),
        "audit_scope": session_contract.get("shared_context", {}).get("audit_scope"),
        "audit_depth": session_contract.get("shared_context", {}).get("audit_depth"),
        "current_role": "investigator",
        "current_round": 1,
    }


def prepare_agent_session(config: SpectraAgentSessionConfig) -> Dict[str, Any]:
    """Prepare an autonomous /spectra session without executing a role."""
    output_root = Path(config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    capabilities = config.client_capabilities or ["filesystem", "network"]
    if config.agent_command_template:
        capabilities = sorted(set(capabilities + ["autonomous_runner"]))

    session_contract = start_spectra_audit_session(
        question=config.question,
        model_paper=config.model_paper,
        model_description=config.model_description,
        dataset_description=config.dataset_description,
        domain=config.domain,
        client_capabilities=capabilities,
        output_root=str(output_root),
        constraints=config.constraints,
        max_cycles=config.max_rounds,
        audit_scope=config.audit_scope,
    )
    manifest = _initial_manifest(config, session_contract)
    session_state = {
        "manifest": manifest,
        "session_contract": session_contract,
        "role_counters": {"investigator": 1},
        "role_history": [],
        "routing_log": [],
    }
    session_state_path = output_root / "session_state.json"
    contract_path = output_root / "session_contract.json"
    manifest_path = output_root / "audit_session_manifest.json"
    _write_json(contract_path, session_contract)
    _write_json(manifest_path, manifest)
    _write_json(session_state_path, session_state)

    first_write_scope = output_root / "investigator_round_001"
    first_prompt_path = output_root / "role_prompts" / "investigator_round_001.md"
    first_work_order = {
        "role": "investigator",
        "round": 1,
        "write_scope": str(first_write_scope),
        "prompt_path": str(first_prompt_path),
        "status": "prepared",
        "created_utc": _utc_now(),
    }
    _write_text(
        first_prompt_path,
        _role_prompt("investigator", 1, first_write_scope, session_state_path, session_contract),
    )
    _write_json(output_root / "work_orders" / "investigator_round_001.json", first_work_order)
    _write_text(
        output_root / "README.md",
        "\n".join(
            [
                "# SPECTRA Agent Session",
                "",
                "This directory was prepared by the autonomous /spectra session runner.",
                "",
                "To execute roles automatically, call `spectra agent run` with an agent command template.",
                "Without an agent command, the prepared role prompts can be executed manually by a host coding agent.",
                "",
                "First role prompt:",
                str(first_prompt_path),
            ]
        ),
    )
    return {
        "status": "prepared",
        "output_root": str(output_root),
        "session_state": str(session_state_path),
        "session_contract": str(contract_path),
        "manifest": str(manifest_path),
        "first_work_order": first_work_order,
        "dry_run": config.dry_run,
        "agent_command_template_present": bool(config.agent_command_template),
    }


def _iteration_limit(max_rounds: Optional[int]) -> Optional[int]:
    if max_rounds is None:
        return None
    value = int(max_rounds)
    if value <= 0:
        return None
    return value


def _resume_start(output_root: Path, session_state: Dict[str, Any]) -> Dict[str, Any]:
    routing_log = session_state.get("routing_log") or []
    if not routing_log:
        return {"role": "investigator", "round": 1, "handoff": None}

    last_route = routing_log[-1]
    if last_route.get("terminal"):
        return {"terminal": True}

    role = str(last_route.get("role") or "investigator")
    handoff = last_route.get("handoff")
    counters = session_state.get("role_counters") or {}
    executed_rounds = [
        int(item.get("round", 0))
        for item in session_state.get("role_history", [])
        if item.get("role") == role and int(item.get("round", 0)) > 0
    ]
    max_executed = max(executed_rounds or [0])
    current_counter = int(counters.get(role, 0))
    if current_counter > max_executed:
        next_round = current_counter
    else:
        next_round = max_executed + 1
    return {
        "role": role,
        "round": next_round,
        "handoff": Path(handoff) if isinstance(handoff, str) and handoff else None,
    }


def _format_agent_command(template: str, prompt_path: Path, write_scope: Path, role: str, round_index: int) -> str:
    return template.format(
        prompt_path=str(prompt_path),
        write_scope=str(write_scope),
        role=role,
        round=round_index,
    )


def _run_external_role(
    command_template: str,
    prompt_path: Path,
    write_scope: Path,
    role: str,
    round_index: int,
    output_root: Path,
) -> Dict[str, Any]:
    write_scope.mkdir(parents=True, exist_ok=True)
    command = _format_agent_command(command_template, prompt_path, write_scope, role, round_index)
    log_path = output_root / "runner_logs" / ("%s_round_%03d.log" % (role, round_index))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.run(
            command,
            shell=True,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            cwd=str(output_root),
            env=os.environ.copy(),
            text=True,
        )
    return {
        "role": role,
        "round": round_index,
        "command": command,
        "returncode": process.returncode,
        "log_path": str(log_path),
        "completed_utc": _utc_now(),
    }


def _normalize_route_token(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _route_role_from_token(value: Any) -> str:
    token = _normalize_route_token(value)
    if token in {"terminal", "complete", "final"}:
        return "terminal"
    if token in {"final_synthesis", "synthesis_distiller", "route_to_final_synthesis"}:
        return "synthesis_distiller"
    if token in {"investigator", "return_to_investigator", "investigator_next"}:
        return "investigator"
    if token in {"dataset_scout", "scout"}:
        return "dataset_scout"
    if token in {"dataset_constructor", "constructor"}:
        return "dataset_constructor"
    if "synthesis_distiller" in token or "final_synthesis" in token:
        return "synthesis_distiller"
    if "dataset_constructor" in token or "constructor" in token:
        return "dataset_constructor"
    if "dataset_scout" in token or token.startswith("scout"):
        return "dataset_scout"
    if "investigator" in token:
        return "investigator"
    return ""


def _explicit_role_token(value: Any) -> str:
    token = _normalize_route_token(value)
    if token in {"investigator", "return_to_investigator", "investigator_next"}:
        return "investigator"
    if token in {"dataset_scout", "scout"}:
        return "dataset_scout"
    if token in {"dataset_constructor", "constructor"}:
        return "dataset_constructor"
    if token.startswith("investigator_for_"):
        return "investigator"
    if token.startswith("dataset_scout_for_") or token.startswith("scout_for_"):
        return "dataset_scout"
    if token.startswith("dataset_constructor_for_") or token.startswith("constructor_for_"):
        return "dataset_constructor"
    return ""


def _declares_bounded_terminal_synthesis(routing: Dict[str, Any]) -> bool:
    if not _truthy(routing.get("terminal")):
        return False
    route_token = _normalize_route_token(
        routing.get("route_decision")
        or routing.get("decision")
        or routing.get("route")
        or routing.get("next_role")
        or routing.get("recommended_next_role")
    )
    if not (
        route_token in {"terminal_bounded_synthesis", "bounded_terminal_synthesis"}
        or route_token.startswith("terminal_bounded_synthesis_")
        or route_token.startswith("bounded_terminal_synthesis_")
    ):
        return False
    verdict = routing.get("round017_verdict") or routing.get("verdict") or {}
    if isinstance(verdict, dict) and _truthy(verdict.get("terminal_for_broad_audit")):
        return True
    return _truthy(routing.get("terminal_for_broad_audit"))


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "allowed", "satisfied"}
    return bool(value)


def _routing_text(routing: Dict[str, Any]) -> str:
    return json.dumps(routing, sort_keys=True, default=str).lower()


def _mechanism_debt_blocks_final_synthesis(routing: Dict[str, Any]) -> bool:
    """Return true when a proxy-level finding is trying to become terminal.

    This is intentionally domain-agnostic. The session prompts ask the Distiller
    to emit explicit mechanism-debt fields; the text fallback catches older
    routing files that describe unresolved proxy debt only in prose.
    """
    if not routing:
        return False
    if _declares_bounded_terminal_synthesis(routing):
        return False
    if (
        _truthy(routing.get("mechanism_debt_satisfied"))
        or _truthy(routing.get("mechanism_supported_with_controls"))
        or _truthy(routing.get("mechanism_infeasibility_proof_provided"))
        or _truthy(routing.get("allow_proxy_terminal"))
    ):
        return False

    audit_depth = _normalize_route_token(
        routing.get("audit_depth")
        or routing.get("explanation_depth_policy")
        or routing.get("explanation_mode")
        or routing.get("mode")
    )
    if audit_depth in {"screening", "surface", "surface_audit", "curve_screen"}:
        return False

    if _truthy(routing.get("mechanism_debt_pending")) or _truthy(routing.get("proxy_debt_pending")):
        return True

    text = _routing_text(routing)
    proxy_tokens = {
        "surface_proxy",
        "domain_proxy",
        "model_space_pointer",
        "model_space",
        "embedding",
        "feature_space",
        "generic_similarity",
        "local_support",
        "oligonucleotide",
        "k_mer",
        "k-mer",
        "composition",
        "metadata_proxy",
        "batch_proxy",
        "site_proxy",
        "scaffold_proxy",
        "fingerprint_proxy",
        "curated_annotation_axis",
    }
    unresolved_tokens = {
        "mechanism_debt",
        "live_unresolved",
        "unresolved",
        "untested",
        "not_disambiguated",
        "without_controls",
        "needs_controls",
        "stronger_unresolved",
        "proxy_to_mechanism",
    }
    selected_axis = routing.get("selected_axis")
    selected_axis_depth = ""
    if isinstance(selected_axis, dict):
        selected_axis_depth = _normalize_route_token(
            selected_axis.get("explanatory_depth") or selected_axis.get("axis_depth_level")
        )
    has_proxy = any(token in text for token in proxy_tokens)
    has_unresolved = any(token in text for token in unresolved_tokens)
    has_resolved = (
        _normalize_route_token(routing.get("explanation_depth")) == "mechanism_supported_with_controls"
        or _normalize_route_token(routing.get("axis_depth_level")) == "mechanism_supported_with_controls"
        or selected_axis_depth == "mechanism_supported_with_controls"
    )
    return bool(has_proxy and has_unresolved and not has_resolved)


def _route_for_mechanism_debt(write_scope: Path, routing: Dict[str, Any]) -> Dict[str, Any]:
    requested = _explicit_role_token(
        routing.get("mechanism_debt_route")
        or routing.get("proxy_debt_route")
        or routing.get("required_next_role")
        or ""
    )
    nested_route = routing.get("next_route_if_synthesis_rejects_bounded_scope")
    if not requested and isinstance(nested_route, dict):
        requested = _route_role_from_token(nested_route.get("role") or nested_route.get("next_role"))
    if requested not in {"investigator", "dataset_scout", "dataset_constructor"}:
        requested = "investigator"

    explicit_handoff = (
        routing.get("mechanism_debt_handoff")
        or routing.get("proxy_debt_handoff")
        or routing.get("%s_handoff" % requested)
    )
    if isinstance(explicit_handoff, str) and explicit_handoff:
        handoff = Path(explicit_handoff)
    else:
        handoff_name = {
            "investigator": "investigator_handoff.json",
            "dataset_scout": "dataset_scout_handoff.json",
            "dataset_constructor": "dataset_constructor_handoff.json",
        }[requested]
        handoff = write_scope / handoff_name

    return {
        "role": requested,
        "terminal": False,
        "handoff": handoff,
        "routing": routing,
        "route_override_reason": "mechanism_debt_blocks_final_synthesis",
    }


def _next_role_from_routing(write_scope: Path, current_role: str) -> Dict[str, Any]:
    if current_role == "investigator":
        return {"role": "distiller", "terminal": False, "handoff": write_scope / "distiller_handoff.json"}
    if current_role == "dataset_scout":
        return {"role": "distiller", "terminal": False, "handoff": write_scope / "scout_decision.json"}
    if current_role == "dataset_constructor":
        return {"role": "distiller", "terminal": False, "handoff": write_scope / "constructor_handoff.json"}
    if current_role == "synthesis_distiller":
        return {"role": "", "terminal": True, "handoff": None}

    routing = _load_json_if_exists(write_scope / "routing_decision.json") or {}
    decision_text = json.dumps(routing, sort_keys=True).lower()
    routing_artifact = routing.get("routing_artifact")
    routing_handoff = Path(routing_artifact) if isinstance(routing_artifact, str) and routing_artifact else None
    nested_route = routing.get("next_route")
    nested_role = ""
    if isinstance(nested_route, dict):
        nested_role = nested_route.get("role") or nested_route.get("next_role") or ""
    explicit_route = (
        routing.get("route_to")
        or nested_role
        or routing.get("next_role")
        or routing.get("recommended_next_role")
        or routing.get("required_next_role")
        or routing.get("route")
        or routing.get("decision")
        or routing.get("return_to")
        or ""
    )
    route_role = _route_role_from_token(explicit_route)
    requested_final = (
        _truthy(routing.get("terminal"))
        or _truthy(routing.get("analysis_complete_for_workflow_demonstration"))
        or route_role in {"terminal", "synthesis_distiller"}
        or "final_synthesis" in decision_text
        or "synthesis_distiller" in decision_text
    )
    if requested_final and _mechanism_debt_blocks_final_synthesis(routing):
        return _route_for_mechanism_debt(write_scope, routing)
    if routing.get("terminal") or routing.get("analysis_complete_for_workflow_demonstration") or route_role == "terminal":
        return {"role": "", "terminal": True, "handoff": None, "routing": routing}
    if route_role == "synthesis_distiller":
        return {"role": "synthesis_distiller", "terminal": False, "handoff": write_scope / "routing_decision.json", "routing": routing}
    if route_role == "dataset_constructor":
        handoff = routing_handoff or write_scope / "dataset_constructor_handoff.json"
        return {"role": "dataset_constructor", "terminal": False, "handoff": handoff, "routing": routing}
    if route_role == "dataset_scout":
        handoff = routing_handoff or write_scope / "dataset_scout_handoff.json"
        return {"role": "dataset_scout", "terminal": False, "handoff": handoff, "routing": routing}
    if route_role == "investigator":
        handoff = routing_handoff or write_scope / "investigator_handoff.json"
        return {"role": "investigator", "terminal": False, "handoff": handoff, "routing": routing}

    # Compatibility fallback for older handoffs that only encode the route in prose.
    if "final_synthesis" in decision_text or "synthesis_distiller" in decision_text:
        return {"role": "synthesis_distiller", "terminal": False, "handoff": write_scope / "routing_decision.json", "routing": routing}
    if "dataset_constructor" in decision_text:
        handoff = routing_handoff or write_scope / "dataset_constructor_handoff.json"
        return {"role": "dataset_constructor", "terminal": False, "handoff": handoff, "routing": routing}
    if "dataset_scout" in decision_text:
        handoff = routing_handoff or write_scope / "dataset_scout_handoff.json"
        return {"role": "dataset_scout", "terminal": False, "handoff": handoff, "routing": routing}
    return {"role": "investigator", "terminal": False, "handoff": write_scope / "investigator_handoff.json", "routing": routing}


def run_agent_session(config: SpectraAgentSessionConfig) -> Dict[str, Any]:
    """Run or prepare an autonomous /spectra session.

    If dry_run is true or no agent command template is supplied, this only
    prepares the session and role prompts. If dry_run is false and a command
    template is supplied, each role is executed through that template.
    """
    output_root = Path(config.output_root)
    session_state_path = output_root / "session_state.json"
    if config.resume and session_state_path.exists():
        prepared = {
            "status": "resumed",
            "output_root": str(output_root),
            "session_state": str(session_state_path),
        }
    else:
        prepared = prepare_agent_session(config)
    if config.dry_run or not config.agent_command_template:
        prepared["status"] = "prepared_not_launched"
        prepared["launch_blocker"] = (
            "No roles executed because dry_run is true or agent_command_template is empty."
        )
        return prepared

    with session_state_path.open("r", encoding="utf-8") as handle:
        session_state = json.load(handle)
    session_contract = session_state["session_contract"]

    if config.resume:
        resume_start = _resume_start(output_root, session_state)
        if resume_start.get("terminal"):
            return {
                "status": "already_completed",
                "session_state": str(session_state_path),
                "role_history": session_state.get("role_history", []),
            }
        role = str(resume_start["role"])
        round_index = int(resume_start["round"])
        handoff_path = resume_start.get("handoff")
    else:
        role = "investigator"
        round_index = 1
        handoff_path = None
    role_history: List[Dict[str, Any]] = list(session_state.get("role_history") or [])
    role_counters: Dict[str, int] = dict(session_state.get("role_counters") or {"investigator": 1})
    iterations = 0
    round_limit = _iteration_limit(config.max_rounds)
    while round_limit is None or iterations < round_limit:
        iterations += 1
        role_counters[role] = max(int(role_counters.get(role, 0)), int(round_index))
        session_state["role_counters"] = role_counters
        _write_json(session_state_path, session_state)
        write_scope = output_root / ("%s_round_%03d" % (role, round_index))
        prompt_path = output_root / "role_prompts" / ("%s_round_%03d.md" % (role, round_index))
        _write_text(
            prompt_path,
            _role_prompt(role, round_index, write_scope, session_state_path, session_contract, handoff_path),
        )
        work_result = _run_external_role(
            config.agent_command_template,
            prompt_path,
            write_scope,
            role,
            round_index,
            output_root,
        )
        role_history.append(work_result)
        session_state["role_history"] = role_history
        _write_json(session_state_path, session_state)
        if work_result["returncode"] != 0:
            return {
                "status": "role_failed",
                "failed_role": role,
                "failed_round": round_index,
                "returncode": work_result["returncode"],
                "log_path": work_result["log_path"],
                "session_state": str(session_state_path),
            }

        next_route = _next_role_from_routing(write_scope, role)
        session_state.setdefault("routing_log", []).append(next_route)
        _write_json(session_state_path, session_state)
        if next_route.get("terminal"):
            return {
                "status": "completed",
                "terminal_role": role,
                "terminal_round": round_index,
                "session_state": str(session_state_path),
                "role_history": role_history,
            }

        role = str(next_route["role"])
        handoff = next_route.get("handoff")
        handoff_path = handoff if isinstance(handoff, Path) else None
        role_counters[role] = int(role_counters.get(role, 0)) + 1
        session_state["role_counters"] = role_counters
        round_index = role_counters[role]

    return {
        "status": "max_rounds_reached",
        "max_rounds": config.max_rounds,
        "round_limit": round_limit,
        "session_state": str(session_state_path),
        "role_history": role_history,
    }


def prepare_agent_session_from_kwargs(**kwargs: Any) -> Dict[str, Any]:
    return prepare_agent_session(SpectraAgentSessionConfig(**kwargs))


def run_agent_session_from_kwargs(**kwargs: Any) -> Dict[str, Any]:
    return run_agent_session(SpectraAgentSessionConfig(**kwargs))
