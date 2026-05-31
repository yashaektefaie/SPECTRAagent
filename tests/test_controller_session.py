import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from spectrae.cli import main
from spectrae.controller_session import (
    SpectraControllerSessionConfig,
    prepare_controller_session,
    run_controller_session,
)
from spectrae.scientific_skill_mcp import (
    prepare_spectra_audit_session,
    run_spectra_audit_session,
    start_spectra_audit_session,
)


class ControllerSessionTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory(prefix="spectra_controller_session_tests_")
        self.root = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def _config(self, name="session"):
        return SpectraControllerSessionConfig(
            question="Use /spectra to assess the generalizability of this model.",
            model_paper="paper.pdf",
            model_description="target model",
            dataset_description="dataset path",
            domain="regulatory DNA",
            output_root=str(self.root / name),
            client_capabilities=["filesystem", "network"],
            dry_run=True,
        )

    def test_prepare_controller_session_writes_single_prompt(self):
        result = prepare_controller_session(self._config())

        self.assertEqual(result["status"], "prepared")
        self.assertEqual(result["orchestration_policy"], "no_python_role_router")
        self.assertTrue(Path(result["prompt_path"]).exists())
        self.assertTrue(Path(result["session_contract"]).exists())
        self.assertTrue(Path(result["manifest"]).exists())
        self.assertFalse((Path(result["output_root"]) / "role_prompts").exists())
        self.assertFalse((Path(result["output_root"]) / "work_orders").exists())

        prompt = Path(result["prompt_path"]).read_text(encoding="utf-8")
        self.assertIn("You are the only SPECTRA agent", prompt)
        self.assertIn("There is no external SPECTRA role router", prompt)
        self.assertIn("broad_model_generalizability_audit", prompt)
        self.assertIn("Terminal gate", prompt)

    def test_run_controller_session_dry_run_does_not_execute_command(self):
        marker = self.root / "should_not_exist.txt"
        config = self._config("dry_run")
        config.agent_command_template = "touch %s" % marker
        config.dry_run = True

        result = run_controller_session(config)

        self.assertEqual(result["status"], "prepared_not_launched")
        self.assertFalse(marker.exists())
        self.assertTrue(Path(result["prompt_path"]).exists())

    def test_run_controller_session_executes_one_host_agent_command(self):
        out = self.root / "execute_once"
        config = self._config("execute_once")
        config.dry_run = False
        config.agent_command_template = "sh -c 'echo controller > {write_scope}/ran.txt'"

        result = run_controller_session(config)

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["returncode"], 0)
        self.assertEqual((out / "ran.txt").read_text(encoding="utf-8").strip(), "controller")
        self.assertFalse((out / "routing_log.json").exists())
        self.assertFalse((out / "session_state.json").exists())

    def test_mcp_prepare_and_run_wrappers_are_safe_by_default(self):
        prepared = prepare_spectra_audit_session(
            question="Use /spectra to assess the generalizability of this model.",
            model_paper="paper.pdf",
            model_description="target model",
            dataset_description="dataset path",
            output_root=str(self.root / "mcp_prepare"),
        )
        self.assertEqual(prepared["status"], "prepared")
        self.assertEqual(prepared["orchestration_policy"], "no_python_role_router")

        marker = self.root / "mcp_should_not_exist.txt"
        dry_run = run_spectra_audit_session(
            question="Use /spectra to assess the generalizability of this model.",
            model_paper="paper.pdf",
            model_description="target model",
            dataset_description="dataset path",
            output_root=str(self.root / "mcp_run"),
            agent_command_template="touch %s" % marker,
            execute_controller=False,
        )
        self.assertEqual(dry_run["status"], "prepared_not_launched")
        self.assertFalse(marker.exists())

    def test_start_session_returns_single_controller_contract(self):
        contract = start_spectra_audit_session(
            question="Use /spectra to assess the generalizability of this model.",
            model_paper="paper.pdf",
            model_description="target model",
            dataset_description="dataset path",
            output_root=str(self.root / "contract"),
            client_capabilities=["filesystem"],
        )

        self.assertEqual(contract["orchestration_mode"], "single_codex_controller_session")
        self.assertTrue(contract["supports"]["single_controller_loop"])
        self.assertFalse(contract["supports"]["subagent_delegation"])
        self.assertEqual([item["role"] for item in contract["spawn_plan"]], ["controller"])
        self.assertNotIn("orchestrator", contract["role_graph"])
        self.assertIn("internal controller phases", contract["client_orchestration_contract"])

    def test_cli_agent_prepare_uses_controller_session(self):
        out = self.root / "cli_prepare"
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            status = main(
                [
                    "agent",
                    "prepare",
                    "--question",
                    "Use /spectra to assess the generalizability of this model.",
                    "--paper",
                    "paper.pdf",
                    "--model-description",
                    "target model",
                    "--out",
                    str(out),
                ]
            )

        self.assertEqual(status, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["status"], "prepared")
        self.assertEqual(payload["orchestration_policy"], "no_python_role_router")
        self.assertTrue((out / "controller_prompt.md").exists())
        self.assertFalse((out / "role_prompts").exists())


if __name__ == "__main__":
    unittest.main()
