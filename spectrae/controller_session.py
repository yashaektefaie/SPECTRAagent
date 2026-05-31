"""Single-controller /spectra audit session launcher.

This module intentionally does not route roles, inspect controller artifacts, or
infer whether the scientific audit should continue. It prepares one SPECTRA
Controller prompt and can launch one host-agent process. The controller agent is
responsible for thinking, iterating, writing helper scripts, running commands,
and deciding when the audit is actually finished.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .scientific_skill_mcp import start_spectra_audit_session


@dataclass
class SpectraControllerSessionConfig:
    question: str
    model_paper: str
    model_description: str
    output_root: str
    dataset_description: str = ""
    domain: str = "unknown"
    constraints: str = ""
    audit_scope: str = "auto"
    client_capabilities: Optional[List[str]] = None
    agent_command_template: str = ""
    dry_run: bool = True


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


def _controller_prompt(session_contract: Dict[str, Any], output_root: Path) -> str:
    shared = session_contract.get("shared_context", {})
    runtime_policy = session_contract.get("runtime_probe_policy", {})
    terminal_gate = session_contract.get("terminal_gate", {})
    role_graph = session_contract.get("phase_graph") or session_contract.get("role_graph", {})
    controller_spec = role_graph.get("controller", {})
    controller_instructions = str(controller_spec.get("prompt") or "")
    expected_outputs = controller_spec.get("writes", [])

    parts = [
        "You are the only SPECTRA agent for this run.",
        "",
        "There is no external SPECTRA role router for this session. Do not wait for a Distiller, Investigator, Dataset Scout, Dataset Fetcher, Auditor, or synthesis agent to be launched. Treat those names only as internal phases in your own reasoning loop.",
        "",
        "Use Codex directly: inspect files, write small helper scripts when useful, run commands, parse outputs, update state, and keep iterating until the terminal gate is genuinely satisfied. You do not need prebuilt deterministic SPECTRA modules for ordinary analysis work; create the concrete analysis code you need inside this session root.",
        "",
        "Session root:",
        str(output_root),
        "",
        "User question:",
        str(shared.get("question", "")),
        "",
        "Question mode:",
        str(shared.get("question_mode", "")),
        "",
        "Model paper/reference:",
        str(shared.get("model_paper_reference", "")),
        "",
        "Model description:",
        str(shared.get("model_description", "")),
        "",
        "Dataset description:",
        str(shared.get("dataset_description", "")),
        "",
        "Domain:",
        str(shared.get("domain", "")),
        "",
        "Audit scope:",
        str(shared.get("audit_scope", "")),
        "",
        "Audit depth:",
        str(shared.get("audit_depth", "")),
        "",
        "User constraints:",
        str(shared.get("constraints", "")),
        "",
        "Controller instructions:",
        controller_instructions,
    ]

    if runtime_policy:
        parts.extend(
            [
                "",
                "Runtime policy:",
                str(runtime_policy.get("purpose") or ""),
            ]
        )
        parts.extend("- %s" % item for item in runtime_policy.get("required_order", []))

    if terminal_gate:
        parts.extend(
            [
                "",
                "Terminal gate:",
                json.dumps(terminal_gate, indent=2, sort_keys=True, default=str),
            ]
        )

    parts.extend(
        [
            "",
            "Expected compact outputs:",
        ]
    )
    parts.extend("- %s" % item for item in expected_outputs)
    parts.extend(
        [
            "",
            "Final response from the host-agent process should summarize the current claim boundary, key evidence, blockers if any, and artifact paths. Do not produce final_report.md unless the terminal gate passes or a hard external blocker prevents further execution.",
        ]
    )
    return "\n".join(parts)


def prepare_controller_session(config: SpectraControllerSessionConfig) -> Dict[str, Any]:
    """Prepare a single-controller SPECTRA session without launching an agent."""
    output_root = Path(config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    capabilities = sorted(
        set(
            (config.client_capabilities or ["filesystem", "network"])
            + ["codex", "single_agent_loop", "persistent_controller", "controller_loop"]
        )
    )
    session_contract = start_spectra_audit_session(
        question=config.question,
        model_paper=config.model_paper,
        model_description=config.model_description,
        dataset_description=config.dataset_description,
        domain=config.domain,
        client_capabilities=capabilities,
        output_root=str(output_root),
        constraints=config.constraints,
        max_cycles=None,
        audit_scope=config.audit_scope,
    )

    prompt_path = output_root / "controller_prompt.md"
    manifest_path = output_root / "session_manifest.json"
    contract_path = output_root / "session_contract.json"
    log_path = output_root / "controller_session.log"
    manifest = {
        "mode": "single_codex_spectra_controller_session",
        "created_utc": _utc_now(),
        "status": "prepared",
        "output_root": str(output_root),
        "write_scope": str(output_root),
        "prompt_path": str(prompt_path),
        "session_contract": str(contract_path),
        "log_path": str(log_path),
        "dry_run": config.dry_run,
        "agent_command_template_present": bool(config.agent_command_template),
        "orchestration_policy": "no_python_role_router",
        "routing_policy": "controller_agent_owns_iteration_and_terminality",
    }
    _write_json(contract_path, session_contract)
    _write_text(prompt_path, _controller_prompt(session_contract, output_root))
    _write_json(manifest_path, manifest)
    _write_text(
        output_root / "README.md",
        "\n".join(
            [
                "# SPECTRA Controller Session",
                "",
                "This directory contains a single-controller SPECTRA prompt.",
                "",
                "There is no Python role router in this session. The controller agent owns the audit loop and terminal decision.",
                "",
                "Prompt:",
                str(prompt_path),
            ]
        ),
    )

    return {
        "status": "prepared",
        "output_root": str(output_root),
        "write_scope": str(output_root),
        "prompt_path": str(prompt_path),
        "manifest": str(manifest_path),
        "session_contract": str(contract_path),
        "log_path": str(log_path),
        "dry_run": config.dry_run,
        "agent_command_template_present": bool(config.agent_command_template),
        "orchestration_policy": "no_python_role_router",
    }


def run_controller_session(config: SpectraControllerSessionConfig) -> Dict[str, Any]:
    """Prepare and optionally launch one persistent controller host-agent process."""
    prepared = prepare_controller_session(config)
    if config.dry_run or not config.agent_command_template:
        prepared["status"] = "prepared_not_launched"
        prepared["launch_blocker"] = (
            "No controller launched because dry_run is true or agent_command_template is empty."
        )
        return prepared

    prompt_path = Path(prepared["prompt_path"])
    write_scope = Path(prepared["write_scope"])
    log_path = Path(prepared["log_path"])
    command = config.agent_command_template.format(
        prompt_path=str(prompt_path),
        write_scope=str(write_scope),
        role="controller",
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

    status = "completed" if completed.returncode == 0 else "failed"
    manifest_path = Path(prepared["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "status": status,
            "completed_utc": _utc_now(),
            "returncode": completed.returncode,
            "command": command,
        }
    )
    _write_json(manifest_path, manifest)
    prepared.update(
        {
            "status": status,
            "returncode": completed.returncode,
            "command": command,
        }
    )
    return prepared
