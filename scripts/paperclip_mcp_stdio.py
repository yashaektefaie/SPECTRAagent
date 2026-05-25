#!/usr/bin/env python3
"""Minimal stdio MCP wrapper for the Paperclip CLI.

This exposes one MCP tool, ``paperclip``, that forwards a command string to the
authenticated Paperclip CLI. It exists because the official Paperclip Codex
skill endpoint currently returns 404, while the CLI itself is installed and
authenticated on this machine.
"""

import json
import os
import shlex
import subprocess
import sys
from typing import Any, Dict, Optional


PAPERCLIP_BIN = os.environ.get(
    "PAPERCLIP_BIN",
    "/ewsc/yektefai/envs/envs/boltz/bin/paperclip",
)
PROTOCOL_VERSION = "2025-03-26"


def _send(payload: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _result(message_id: Any, result: Dict[str, Any]) -> None:
    _send({"jsonrpc": "2.0", "id": message_id, "result": result})


def _error(message_id: Any, code: int, message: str) -> None:
    _send({"jsonrpc": "2.0", "id": message_id, "error": {"code": code, "message": message}})


def _read_message(line: str) -> Optional[Dict[str, Any]]:
    line = line.strip()
    if not line:
        return None
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _run_paperclip(command: str) -> Dict[str, Any]:
    if not command.strip():
        return {"ok": False, "text": "Missing Paperclip command."}
    if not os.path.exists(PAPERCLIP_BIN):
        return {"ok": False, "text": "Paperclip binary not found: %s" % PAPERCLIP_BIN}

    argv = [PAPERCLIP_BIN] + shlex.split(command)
    try:
        process = subprocess.run(
            argv,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            timeout=300,
        )
    except Exception as exc:  # pragma: no cover - defensive MCP boundary.
        return {"ok": False, "text": "Paperclip execution failed: %s" % exc}

    output = process.stdout or ""
    if len(output) > 120000:
        output = output[:120000] + "\n\n[truncated at 120000 characters]\n"
    return {"ok": process.returncode == 0, "text": output, "returncode": process.returncode}


def _tool_schema() -> Dict[str, Any]:
    return {
        "name": "paperclip",
        "description": (
            "Run a Paperclip CLI command against the indexed scientific-paper "
            "filesystem. Examples: search \"CRISPR\"; results --list; "
            "cat /papers/<id>/meta.json; grep -i \"dataset\" /papers/<id>/content.lines; "
            "map --from s_<id> \"extraction question\"."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Command arguments to pass to paperclip, without the leading 'paperclip'.",
                }
            },
            "required": ["command"],
        },
    }


def _handle(payload: Dict[str, Any]) -> None:
    method = payload.get("method")
    message_id = payload.get("id")
    params = payload.get("params") or {}

    if method == "initialize":
        _result(
            message_id,
            {
                "protocolVersion": PROTOCOL_VERSION,
                "serverInfo": {"name": "paperclip-cli-wrapper", "version": "0.1.0"},
                "capabilities": {"tools": {"listChanged": False}},
                "instructions": (
                    "Use tools/call with name='paperclip' and a command string. "
                    "The wrapper forwards to the locally authenticated Paperclip CLI."
                ),
            },
        )
        return

    if method in {"notifications/initialized", "initialized"}:
        return

    if method == "ping":
        _result(message_id, {})
        return

    if method == "tools/list":
        _result(message_id, {"tools": [_tool_schema()]})
        return

    if method == "tools/call":
        if params.get("name") != "paperclip":
            _error(message_id, -32602, "Unknown tool: %s" % params.get("name"))
            return
        arguments = params.get("arguments") or {}
        command = str(arguments.get("command") or "")
        result = _run_paperclip(command)
        _result(
            message_id,
            {
                "content": [{"type": "text", "text": result["text"]}],
                "isError": not result.get("ok", False),
            },
        )
        return

    if method in {"resources/list", "prompts/list"}:
        key = "resources" if method == "resources/list" else "prompts"
        _result(message_id, {key: []})
        return

    if message_id is not None:
        _error(message_id, -32601, "Method not found: %s" % method)


def main() -> int:
    for line in sys.stdin:
        payload = _read_message(line)
        if payload is None:
            continue
        try:
            _handle(payload)
        except Exception as exc:  # pragma: no cover - keep MCP server alive.
            if payload.get("id") is not None:
                _error(payload.get("id"), -32603, "Internal error: %s" % exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
