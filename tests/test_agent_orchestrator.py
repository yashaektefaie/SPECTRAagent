import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from spectrae.agent_orchestrator import (
    SpectraAgentSessionConfig,
    _next_role_from_routing,
    prepare_agent_session,
    run_agent_session,
)
from spectrae.scientific_skill_mcp import (
    prepare_spectra_audit_session,
    run_spectra_audit_session,
)


class AgentOrchestratorTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory(prefix="spectra_agent_orchestrator_tests_")
        self.root = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def _config(self, name="session"):
        return SpectraAgentSessionConfig(
            question="Use /spectra to assess the generalizability of this model.",
            model_paper="paper.pdf",
            model_description="target model",
            dataset_description="dataset path",
            domain="regulatory DNA",
            output_root=str(self.root / name),
            client_capabilities=["filesystem", "network"],
            max_rounds=4,
            dry_run=True,
        )

    def test_prepare_agent_session_writes_initial_work_order(self):
        result = prepare_agent_session(self._config())

        self.assertEqual(result["status"], "prepared")
        self.assertTrue(Path(result["session_state"]).exists())
        self.assertTrue(Path(result["session_contract"]).exists())
        prompt_path = Path(result["first_work_order"]["prompt_path"])
        self.assertTrue(prompt_path.exists())
        prompt = prompt_path.read_text(encoding="utf-8")
        self.assertIn("SPECTRA Investigator", prompt)
        self.assertIn("broad_model_generalizability_audit", prompt)

        state = json.loads(Path(result["session_state"]).read_text(encoding="utf-8"))
        self.assertEqual(state["manifest"]["current_role"], "investigator")
        self.assertEqual(
            state["session_contract"]["shared_context"]["question_mode"],
            "broad_model_generalizability_audit",
        )
        self.assertEqual(
            state["session_contract"]["shared_context"]["audit_scope"],
            "paper_claim_audit",
        )
        self.assertEqual(
            state["session_contract"]["shared_context"]["audit_depth"],
            "investigation",
        )

    def test_prepare_agent_session_includes_cheap_first_runtime_policy(self):
        result = prepare_agent_session(self._config("runtime_policy"))

        prompt = Path(result["first_work_order"]["prompt_path"]).read_text(encoding="utf-8")
        state = json.loads(Path(result["session_state"]).read_text(encoding="utf-8"))
        policy = state["session_contract"]["runtime_probe_policy"]

        self.assertEqual(policy["name"], "cheap_first_behavioral_runtime_policy")
        self.assertIn("Cheap-first behavioral runtime policy", prompt)
        self.assertIn("deterministic non-iterative probes", prompt)
        self.assertIn("After one timeout or runtime failure", prompt)
        self.assertIn("runtime_budget_and_fallbacks.md", prompt)

    def test_prepare_agent_session_can_use_beyond_paper_discovery_scope(self):
        config = self._config("beyond")
        config.question = (
            "Use /spectra to assess generalizability beyond the paper's reported evaluations."
        )
        config.audit_scope = "beyond-paper-discovery"

        result = prepare_agent_session(config)

        prompt = Path(result["first_work_order"]["prompt_path"]).read_text(encoding="utf-8")
        self.assertIn("broad_beyond_paper_generalizability_discovery", prompt)
        self.assertIn("Audit scope: beyond_paper_discovery", prompt)
        self.assertIn("paper as context, not as the boundary", prompt)

    def test_run_agent_session_dry_run_does_not_execute_agent_command(self):
        marker = self.root / "should_not_exist.txt"
        config = self._config("dry_run")
        config.agent_command_template = "touch %s" % marker
        config.dry_run = True

        result = run_agent_session(config)

        self.assertEqual(result["status"], "prepared_not_launched")
        self.assertFalse(marker.exists())
        self.assertTrue(Path(result["first_work_order"]["prompt_path"]).exists())

    def test_mcp_prepare_and_run_wrappers_are_safe_by_default(self):
        prepared = prepare_spectra_audit_session(
            question="Use /spectra to assess the generalizability of this model.",
            model_paper="paper.pdf",
            model_description="target model",
            dataset_description="dataset path",
            output_root=str(self.root / "mcp_prepare"),
        )
        self.assertEqual(prepared["status"], "prepared")

        marker = self.root / "mcp_should_not_exist.txt"
        dry_run = run_spectra_audit_session(
            question="Use /spectra to assess the generalizability of this model.",
            model_paper="paper.pdf",
            model_description="target model",
            dataset_description="dataset path",
            output_root=str(self.root / "mcp_run"),
            agent_command_template="touch %s" % marker,
            execute_roles=False,
        )
        self.assertEqual(dry_run["status"], "prepared_not_launched")
        self.assertFalse(marker.exists())

    def test_runner_routes_dataset_roles_back_to_distiller(self):
        scout_scope = self.root / "dataset_scout_round_001"
        constructor_scope = self.root / "dataset_constructor_round_001"

        scout_route = _next_role_from_routing(scout_scope, "dataset_scout")
        constructor_route = _next_role_from_routing(constructor_scope, "dataset_constructor")

        self.assertEqual(scout_route["role"], "distiller")
        self.assertEqual(str(scout_route["handoff"]), str(scout_scope / "scout_decision.json"))
        self.assertEqual(constructor_route["role"], "distiller")
        self.assertEqual(
            str(constructor_route["handoff"]),
            str(constructor_scope / "constructor_handoff.json"),
        )

    def test_runner_routes_distiller_final_synthesis_decision(self):
        distiller_scope = self.root / "distiller_round_001"
        distiller_scope.mkdir(parents=True, exist_ok=True)
        (distiller_scope / "routing_decision.json").write_text(
            json.dumps({"route": "final_synthesis", "reason": "paper-ready finding"}),
            encoding="utf-8",
        )

        route = _next_role_from_routing(distiller_scope, "distiller")

        self.assertEqual(route["role"], "synthesis_distiller")
        self.assertFalse(route["terminal"])
        self.assertEqual(
            str(route["handoff"]),
            str(distiller_scope / "routing_decision.json"),
        )

    def test_runner_blocks_final_synthesis_when_mechanism_debt_is_pending(self):
        distiller_scope = self.root / "distiller_round_001"
        distiller_scope.mkdir(parents=True, exist_ok=True)
        handoff = distiller_scope / "investigator_handoff.json"
        (distiller_scope / "routing_decision.json").write_text(
            json.dumps(
                {
                    "next_role": "synthesis_distiller",
                    "audit_depth": "investigation",
                    "explanation_depth": "surface_proxy",
                    "mechanism_debt_pending": True,
                    "mechanism_debt_satisfied": False,
                    "mechanism_debt_route": "investigator",
                    "mechanism_debt_handoff": str(handoff),
                }
            ),
            encoding="utf-8",
        )

        route = _next_role_from_routing(distiller_scope, "distiller")

        self.assertEqual(route["role"], "investigator")
        self.assertFalse(route["terminal"])
        self.assertEqual(str(route["handoff"]), str(handoff))
        self.assertEqual(route["route_override_reason"], "mechanism_debt_blocks_final_synthesis")

    def test_runner_routes_proxy_mechanism_debt_to_dataset_constructor(self):
        distiller_scope = self.root / "distiller_round_001"
        distiller_scope.mkdir(parents=True, exist_ok=True)
        handoff = distiller_scope / "dataset_constructor_handoff.json"
        (distiller_scope / "routing_decision.json").write_text(
            json.dumps(
                {
                    "decision": "route_to_final_synthesis",
                    "audit_depth": "investigation",
                    "selected_axis": {
                        "name": "generic embedding distance",
                        "explanatory_depth": "model_space_pointer",
                    },
                    "hypothesis_status": [
                        {"hypothesis": "deployment mechanism", "status": "live_unresolved"}
                    ],
                    "mechanism_debt_route": "dataset_constructor",
                    "mechanism_debt_handoff": str(handoff),
                }
            ),
            encoding="utf-8",
        )

        route = _next_role_from_routing(distiller_scope, "distiller")

        self.assertEqual(route["role"], "dataset_constructor")
        self.assertFalse(route["terminal"])
        self.assertEqual(str(route["handoff"]), str(handoff))

    def test_runner_honors_terminal_bounded_synthesis_with_open_mechanism_debt(self):
        distiller_scope = self.root / "distiller_round_001"
        distiller_scope.mkdir(parents=True, exist_ok=True)
        (distiller_scope / "routing_decision.json").write_text(
            json.dumps(
                {
                    "terminal": True,
                    "route_decision": "terminal_bounded_synthesis_reaffirmed_after_recovery",
                    "recommended_next_role": "final_synthesis",
                    "mechanism_debt_pending": True,
                    "mechanism_debt_satisfied": False,
                    "mechanism_debt_route": (
                        "terminal_bounded_synthesis_with_future_dataset_scout_or_constructor"
                    ),
                    "round017_verdict": {"terminal_for_broad_audit": True},
                }
            ),
            encoding="utf-8",
        )

        route = _next_role_from_routing(distiller_scope, "distiller")

        self.assertTrue(route["terminal"])
        self.assertEqual(route["role"], "")

    def test_mechanism_debt_route_ignores_future_constructor_phrase(self):
        distiller_scope = self.root / "distiller_round_001"
        distiller_scope.mkdir(parents=True, exist_ok=True)
        (distiller_scope / "routing_decision.json").write_text(
            json.dumps(
                {
                    "next_role": "final_synthesis",
                    "audit_depth": "investigation",
                    "mechanism_debt_pending": True,
                    "mechanism_debt_satisfied": False,
                    "mechanism_debt_route": (
                        "terminal_bounded_synthesis_with_future_dataset_scout_or_constructor"
                    ),
                }
            ),
            encoding="utf-8",
        )

        route = _next_role_from_routing(distiller_scope, "distiller")

        self.assertEqual(route["role"], "investigator")
        self.assertFalse(route["terminal"])
        self.assertEqual(route["route_override_reason"], "mechanism_debt_blocks_final_synthesis")

    def test_runner_allows_final_synthesis_when_mechanism_debt_is_satisfied(self):
        distiller_scope = self.root / "distiller_round_001"
        distiller_scope.mkdir(parents=True, exist_ok=True)
        (distiller_scope / "routing_decision.json").write_text(
            json.dumps(
                {
                    "next_role": "synthesis_distiller",
                    "audit_depth": "investigation",
                    "explanation_depth": "mechanism_supported_with_controls",
                    "mechanism_debt_pending": False,
                    "mechanism_debt_satisfied": True,
                }
            ),
            encoding="utf-8",
        )

        route = _next_role_from_routing(distiller_scope, "distiller")

        self.assertEqual(route["role"], "synthesis_distiller")
        self.assertFalse(route["terminal"])
        self.assertEqual(
            str(route["handoff"]),
            str(distiller_scope / "routing_decision.json"),
        )

    def test_runner_treats_synthesis_distiller_as_terminal(self):
        route = _next_role_from_routing(self.root / "synthesis_distiller_round_001", "synthesis_distiller")

        self.assertTrue(route["terminal"])
        self.assertEqual(route["role"], "")

    def test_runner_uses_distiller_routing_artifact_for_constructor_handoff(self):
        distiller_scope = self.root / "distiller_round_001"
        distiller_scope.mkdir(parents=True, exist_ok=True)
        handoff = distiller_scope / "dataset_constructor_handoff.json"
        (distiller_scope / "routing_decision.json").write_text(
            json.dumps(
                {
                    "next_role": "dataset_constructor",
                    "routing_artifact": str(handoff),
                    "terminal": False,
                }
            ),
            encoding="utf-8",
        )

        route = _next_role_from_routing(distiller_scope, "distiller")

        self.assertEqual(route["role"], "dataset_constructor")
        self.assertEqual(str(route["handoff"]), str(handoff))

    def test_runner_prefers_explicit_investigator_route_over_artifact_paths(self):
        distiller_scope = self.root / "distiller_round_001"
        distiller_scope.mkdir(parents=True, exist_ok=True)
        handoff = self.root / "dataset_constructor_round_001" / "investigator_handoff.json"
        (distiller_scope / "routing_decision.json").write_text(
            json.dumps(
                {
                    "next_role": "investigator",
                    "routing_artifact": str(handoff),
                    "accepted_artifacts": [
                        str(self.root / "dataset_constructor_round_001" / "sequence_table.parquet")
                    ],
                    "terminal": False,
                }
            ),
            encoding="utf-8",
        )

        route = _next_role_from_routing(distiller_scope, "distiller")

        self.assertEqual(route["role"], "investigator")
        self.assertEqual(str(route["handoff"]), str(handoff))

    def test_runner_uses_recommended_next_role_before_fallback_text(self):
        distiller_scope = self.root / "distiller_round_001"
        distiller_scope.mkdir(parents=True, exist_ok=True)
        handoff = distiller_scope / "investigator_handoff.json"
        (distiller_scope / "routing_decision.json").write_text(
            json.dumps(
                {
                    "recommended_next_role": "investigator",
                    "recommended_next_round": 2,
                    "routing_artifact": str(handoff),
                    "terminal": False,
                    "input_artifacts": {
                        "constructor_handoff_round_002": str(
                            self.root / "dataset_constructor_round_002" / "constructor_handoff.json"
                        ),
                        "constructed_dataset_manifest_round_002": str(
                            self.root
                            / "dataset_constructor_round_002"
                            / "constructed_dataset_manifest.json"
                        ),
                    },
                    "mechanism_debt_pending": True,
                    "mechanism_debt_route": "investigator",
                    "route_reason": (
                        "The dataset_constructor package is ready; the next executable "
                        "work is investigator behavior on the validated package."
                    ),
                }
            ),
            encoding="utf-8",
        )

        route = _next_role_from_routing(distiller_scope, "distiller")

        self.assertEqual(route["role"], "investigator")
        self.assertEqual(str(route["handoff"]), str(handoff))

    def test_runner_prefers_route_to_investigator_over_fallback_constructor_text(self):
        distiller_scope = self.root / "distiller_round_001"
        distiller_scope.mkdir(parents=True, exist_ok=True)
        handoff = distiller_scope / "investigator_handoff.json"
        (distiller_scope / "routing_decision.json").write_text(
            json.dumps(
                {
                    "route_to": "investigator",
                    "target_round": 2,
                    "routing_artifact": str(handoff),
                    "terminal": False,
                    "mechanism_debt_pending": True,
                    "mechanism_debt_satisfied": False,
                    "mechanism_debt_route": (
                        "investigator_round_002_lrb_eqtl_context_length_ladder"
                    ),
                    "next_route": {
                        "role": "investigator",
                        "task": "Run the controlled context ladder.",
                    },
                    "secondary_route_if_primary_fails_or_remains_proxy": {
                        "route_to": "dataset_constructor",
                        "task": "Construct a matched fallback dataset only if needed.",
                    },
                }
            ),
            encoding="utf-8",
        )

        route = _next_role_from_routing(distiller_scope, "distiller")

        self.assertEqual(route["role"], "investigator")
        self.assertEqual(str(route["handoff"]), str(handoff))

    def test_runner_uses_nested_next_route_before_fallback_constructor_text(self):
        distiller_scope = self.root / "distiller_round_001"
        distiller_scope.mkdir(parents=True, exist_ok=True)
        handoff = distiller_scope / "investigator_handoff.json"
        (distiller_scope / "routing_decision.json").write_text(
            json.dumps(
                {
                    "routing_artifact": str(handoff),
                    "terminal": False,
                    "next_route": {
                        "role": "investigator",
                        "task": "Run the controlled context ladder.",
                    },
                    "fallback_route": (
                        "dataset_constructor_for_matched_cCRE_or_dataset_scout"
                    ),
                }
            ),
            encoding="utf-8",
        )

        route = _next_role_from_routing(distiller_scope, "distiller")

        self.assertEqual(route["role"], "investigator")
        self.assertEqual(str(route["handoff"]), str(handoff))

    def test_runner_maps_descriptive_mechanism_debt_constructor_route(self):
        distiller_scope = self.root / "distiller_round_001"
        distiller_scope.mkdir(parents=True, exist_ok=True)
        handoff = distiller_scope / "dataset_constructor_handoff.json"
        (distiller_scope / "routing_decision.json").write_text(
            json.dumps(
                {
                    "decision": "route_to_final_synthesis",
                    "audit_depth": "investigation",
                    "explanation_depth": "surface_proxy",
                    "mechanism_debt_pending": True,
                    "mechanism_debt_satisfied": False,
                    "mechanism_debt_route": "dataset_constructor_for_matched_cCRE",
                    "mechanism_debt_handoff": str(handoff),
                }
            ),
            encoding="utf-8",
        )

        route = _next_role_from_routing(distiller_scope, "distiller")

        self.assertEqual(route["role"], "dataset_constructor")
        self.assertEqual(str(route["handoff"]), str(handoff))

    def test_run_agent_session_uses_per_role_round_counters(self):
        script = self.root / "fake_agent.py"
        script.write_text(
            "\n".join(
                [
                    "import json, sys",
                    "from pathlib import Path",
                    "role, round_index, write_scope = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])",
                    "write_scope.mkdir(parents=True, exist_ok=True)",
                    "if role == 'investigator':",
                    "    (write_scope / 'distiller_handoff.json').write_text('{}', encoding='utf-8')",
                    "elif role == 'distiller':",
                    "    handoff = write_scope / 'investigator_handoff.json'",
                    "    handoff.write_text('{}', encoding='utf-8')",
                    "    (write_scope / 'routing_decision.json').write_text(json.dumps({'next_role': 'investigator', 'routing_artifact': str(handoff)}), encoding='utf-8')",
                ]
            ),
            encoding="utf-8",
        )
        config = self._config("role_counters")
        config.dry_run = False
        config.max_rounds = 3
        config.agent_command_template = (
            "python %s {role} {round} {write_scope}" % script
        )

        result = run_agent_session(config)

        self.assertEqual(result["status"], "max_rounds_reached")
        self.assertTrue((self.root / "role_counters" / "investigator_round_001").exists())
        self.assertTrue((self.root / "role_counters" / "distiller_round_001").exists())
        self.assertTrue((self.root / "role_counters" / "investigator_round_002").exists())

    def test_run_agent_session_can_run_without_fixed_round_cap_until_terminal(self):
        script = self.root / "fake_unbounded_agent.py"
        script.write_text(
            "\n".join(
                [
                    "import json, sys",
                    "from pathlib import Path",
                    "role, round_index, write_scope = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])",
                    "write_scope.mkdir(parents=True, exist_ok=True)",
                    "if role == 'investigator':",
                    "    (write_scope / 'distiller_handoff.json').write_text('{}', encoding='utf-8')",
                    "elif role == 'distiller':",
                    "    (write_scope / 'routing_decision.json').write_text(json.dumps({'terminal': True, 'mechanism_debt_satisfied': True}), encoding='utf-8')",
                ]
            ),
            encoding="utf-8",
        )
        config = self._config("unbounded_terminal")
        config.dry_run = False
        config.max_rounds = None
        config.agent_command_template = (
            "python %s {role} {round} {write_scope}" % script
        )

        result = run_agent_session(config)

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["terminal_role"], "distiller")
        self.assertEqual(len(result["role_history"]), 2)

    def test_run_agent_session_resume_continues_from_last_routing_decision(self):
        script = self.root / "fake_resume_agent.py"
        script.write_text(
            "\n".join(
                [
                    "import json, sys",
                    "from pathlib import Path",
                    "role, round_index, write_scope = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])",
                    "write_scope.mkdir(parents=True, exist_ok=True)",
                    "(write_scope / 'executed_role.json').write_text(json.dumps({'role': role, 'round': round_index}), encoding='utf-8')",
                    "if role == 'investigator':",
                    "    (write_scope / 'distiller_handoff.json').write_text('{}', encoding='utf-8')",
                    "elif role == 'distiller':",
                    "    (write_scope / 'routing_decision.json').write_text(json.dumps({'terminal': True, 'mechanism_debt_satisfied': True}), encoding='utf-8')",
                ]
            ),
            encoding="utf-8",
        )
        config = self._config("resume_existing")
        config.dry_run = False
        config.max_rounds = 1
        config.agent_command_template = (
            "python %s {role} {round} {write_scope}" % script
        )

        capped = run_agent_session(config)
        self.assertEqual(capped["status"], "max_rounds_reached")

        config.resume = True
        config.max_rounds = None
        resumed = run_agent_session(config)

        self.assertEqual(resumed["status"], "completed")
        self.assertEqual(resumed["terminal_role"], "distiller")
        self.assertTrue((self.root / "resume_existing" / "distiller_round_001").exists())
        self.assertFalse((self.root / "resume_existing" / "distiller_round_002").exists())

    def test_run_agent_session_resume_updates_current_role_counter(self):
        script = self.root / "fake_resume_counter_agent.py"
        script.write_text(
            "\n".join(
                [
                    "import json, sys",
                    "from pathlib import Path",
                    "role, round_index, write_scope = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])",
                    "write_scope.mkdir(parents=True, exist_ok=True)",
                    "(write_scope / 'executed_role.json').write_text(json.dumps({'role': role, 'round': round_index}), encoding='utf-8')",
                    "if role == 'investigator':",
                    "    (write_scope / 'distiller_handoff.json').write_text('{}', encoding='utf-8')",
                    "elif role == 'distiller' and round_index == 3:",
                    "    handoff = write_scope / 'investigator_handoff.json'",
                    "    handoff.write_text('{}', encoding='utf-8')",
                    "    (write_scope / 'routing_decision.json').write_text(json.dumps({'next_role': 'investigator', 'routing_artifact': str(handoff)}), encoding='utf-8')",
                    "elif role == 'distiller':",
                    "    (write_scope / 'routing_decision.json').write_text(json.dumps({'terminal': True, 'mechanism_debt_satisfied': True}), encoding='utf-8')",
                ]
            ),
            encoding="utf-8",
        )
        config = self._config("resume_counter")
        prepared = prepare_agent_session(config)
        session_state_path = Path(prepared["session_state"])
        state = json.loads(session_state_path.read_text(encoding="utf-8"))
        state["role_history"] = [
            {"role": "investigator", "round": 1, "returncode": 0},
            {"role": "distiller", "round": 1, "returncode": 0},
            {"role": "investigator", "round": 2, "returncode": 0},
            {"role": "distiller", "round": 2, "returncode": 0},
        ]
        state["routing_log"] = [
            {
                "role": "distiller",
                "terminal": False,
                "handoff": str(self.root / "resume_counter" / "distiller_handoff.json"),
            }
        ]
        state["role_counters"] = {"investigator": 2, "distiller": 2}
        session_state_path.write_text(json.dumps(state), encoding="utf-8")

        config.resume = True
        config.dry_run = False
        config.max_rounds = None
        config.agent_command_template = "python %s {role} {round} {write_scope}" % script

        result = run_agent_session(config)

        self.assertEqual(result["status"], "completed")
        self.assertTrue((self.root / "resume_counter" / "distiller_round_003").exists())
        self.assertTrue((self.root / "resume_counter" / "investigator_round_003").exists())
        self.assertTrue((self.root / "resume_counter" / "distiller_round_004").exists())
        executed = json.loads(
            (self.root / "resume_counter" / "distiller_round_004" / "executed_role.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(executed, {"role": "distiller", "round": 4})


if __name__ == "__main__":
    unittest.main()
