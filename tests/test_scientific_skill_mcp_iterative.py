import unittest

from spectrae.scientific_skill_mcp import (
    assess_explanatory_depth,
    choose_discriminating_experiment,
    decide_next_spectra_experiment,
    distill_spectra_hypotheses,
    distill_dataset_constructor_output,
    enforce_mechanism_debt_gate,
    plan_hypothesis_driven_dataset_acquisition,
    prepare_dataset_scout_request,
    prepare_dataset_constructor_request,
    distill_dataset_scout_output,
    synthesize_spectra_generalizability_finding,
    plan_iterative_similarity_search,
    plan_hypothesis_test_dataset_construction,
    plan_public_resource_acquisition,
    reflect_on_replication_evidence,
    review_investigator_checkpoint,
    score_similarity_hypothesis_curve,
    select_spectra_execution_mode,
    start_spectra_audit_session,
    start_spectra_investigator,
    translate_model_space_axis_to_domain_hypotheses,
    update_hypothesis_ledger,
)


class ScientificSkillMcpIterativeTests(unittest.TestCase):
    def test_start_spectra_audit_session_builds_single_controller_plan(self):
        result = start_spectra_audit_session(
            question="Does Caduceus generalize to promoter-like cCRE negatives?",
            model_paper="Caduceus model paper text or PDF path",
            model_description="Caduceus frozen embeddings",
            dataset_description="ENCODE cCRE 1024 bp windows",
            domain="regulatory DNA",
            client_capabilities=["subagents", "filesystem", "network"],
            output_root="/work/spectra_session",
        )

        self.assertEqual(result["mode"], "spectra_audit_session")
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["orchestration_mode"], "single_codex_controller_session")
        self.assertFalse(result["supports"]["subagent_delegation"])
        self.assertTrue(result["spawn_plan"])
        self.assertEqual([item["role"] for item in result["spawn_plan"]], ["controller"])
        self.assertIn("controller", result["phase_graph"])
        self.assertIn("investigator", result["role_graph"])
        self.assertIn("distiller", result["role_graph"])
        self.assertIn("dataset_fetcher", result["role_graph"])
        self.assertIn("synthesis_distiller", result["role_graph"])
        self.assertIn("synthesize_spectra_generalizability_finding", result["terminal_gate"]["required_tool"])
        self.assertTrue(any("internal controller phases" in step for step in result["routing_policy"]))
        self.assertIn("paper_ready_spectra_finding.md", result["role_graph"]["synthesis_distiller"]["writes"])
        self.assertFalse(result["broad_question_behavior"]["enabled"])

    def test_start_spectra_audit_session_uses_controller_for_single_agent_clients(self):
        result = start_spectra_audit_session(
            question="Use /spectra to assess the generalizability of this model.",
            model_paper="paper.pdf",
            model_description="target model",
            client_capabilities=["filesystem"],
        )

        self.assertEqual(result["orchestration_mode"], "single_codex_controller_session")
        self.assertEqual([item["role"] for item in result["spawn_plan"]], ["controller"])
        self.assertTrue(result["sequential_fallback_plan"])
        self.assertFalse(result["supports"]["subagent_delegation"])
        self.assertTrue(result["supports"]["single_controller_loop"])
        self.assertTrue(result["broad_question_behavior"]["enabled"])
        self.assertEqual(result["shared_context"]["question_mode"], "broad_model_generalizability_audit")
        self.assertEqual(result["shared_context"]["audit_scope"], "paper_claim_audit")
        self.assertTrue(
            any(
                "Extract the model paper" in step
                for step in result["broad_question_behavior"]["paper_first_steps"]
            )
        )

    def test_start_spectra_audit_session_supports_beyond_paper_discovery(self):
        result = start_spectra_audit_session(
            question=(
                "Use /spectra to assess generalizability beyond the paper's reported "
                "evaluations; construct or acquire public regulatory datasets if needed."
            ),
            model_paper="paper.pdf",
            model_description="Caduceus representations",
            dataset_description="Do not restrict to the paper benchmarks.",
            domain="regulatory DNA",
            audit_scope="auto",
        )

        self.assertEqual(
            result["shared_context"]["question_mode"],
            "broad_beyond_paper_generalizability_discovery",
        )
        self.assertEqual(result["shared_context"]["audit_scope"], "beyond_paper_discovery")
        self.assertTrue(result["broad_question_behavior"]["beyond_paper_steps"])
        self.assertIn("do not restrict", result["broad_question_behavior"]["paper_context_policy"])
        self.assertTrue(result["broad_question_behavior"]["beyond_paper_steps"])

    def test_start_spectra_audit_session_broad_detector_allows_named_model(self):
        result = start_spectra_audit_session(
            question="Use /spectra to assess Caduceus generalizability beyond the paper.",
            model_paper="paper.pdf",
            model_description="Caduceus representations",
            audit_scope="auto",
        )

        self.assertEqual(
            result["shared_context"]["question_mode"],
            "broad_beyond_paper_generalizability_discovery",
        )

    def test_start_spectra_audit_session_blocks_without_paper_or_model(self):
        result = start_spectra_audit_session(
            question="Does this model generalize?",
            model_paper="",
        )

        self.assertEqual(result["status"], "blocked")
        self.assertIn("model_paper", result["missing_inputs"])
        self.assertIn("model_description", result["missing_inputs"])

    def test_score_similarity_hypothesis_curve_monotonic(self):
        curve = [
            {"subset": "all_eval", "mean_novelty": 0.1, "rmse": 1.0, "test_size": 100},
            {"subset": "mid", "mean_novelty": 0.5, "rmse": 1.2, "test_size": 80},
            {"subset": "hard", "mean_novelty": 0.9, "rmse": 1.5, "test_size": 60},
        ]

        result = score_similarity_hypothesis_curve(curve, leakage_risk="none")

        self.assertEqual(result["status"], "monotonic_supported")
        self.assertTrue(result["prospective_valid"])
        self.assertGreater(result["lowest_overlap_delta"], 0)

    def test_score_similarity_hypothesis_curve_leakage_policy(self):
        curve = [
            {"subset": "all_eval", "mean_novelty": 0.1, "rmse": 1.0, "test_size": 100},
            {"subset": "hard", "mean_novelty": 0.9, "rmse": 1.5, "test_size": 60},
        ]

        result = score_similarity_hypothesis_curve(
            curve,
            leakage_risk="post_hoc_uses_eval_labels",
        )

        self.assertEqual(result["status"], "monotonic_supported")
        self.assertFalse(result["prospective_valid"])
        self.assertTrue(result["warnings"])

    def test_plan_iterative_similarity_search_instructs_loop(self):
        result = plan_iterative_similarity_search(
            dataset_description="NABench mutated nucleotide fitness assay",
            task_description="generalization to unseen mutational regions",
            data_type="nucleotide",
            top_k=3,
        )

        self.assertEqual(result["mode"], "iterative_similarity_hypothesis_search")
        self.assertIn("selection_rule", result)
        self.assertIn("candidate_similarity_definitions", result)
        self.assertTrue(any("score_similarity_hypothesis_curve" in step for step in result["loop"]))
        self.assertTrue(any("fresh model" in step for step in result["loop"]))
        self.assertTrue(any("decide_next_spectra_experiment" in step for step in result["loop"]))
        self.assertIn("minimum_axis_diversity", result)
        self.assertIn("task_coverage_policy", result)
        self.assertIn("axis_search_budget", result["required_outputs"])
        self.assertIn("task_coverage_plan", result["required_outputs"])
        self.assertIn("task_screen_ranking", result["required_outputs"])
        self.assertTrue(any("untested class" in step for step in result["loop"]))
        self.assertTrue(any("task-coverage plan" in step for step in result["loop"]))
        self.assertTrue(any("untested feasible tasks remain" in step for step in result["loop"]))
        self.assertTrue(any("reflect_on_replication_evidence" in step for step in result["loop"]))
        self.assertTrue(any("translate_model_space_axis_to_domain_hypotheses" in step for step in result["loop"]))
        self.assertTrue(any("assess_explanatory_depth" in step for step in result["loop"]))
        self.assertTrue(any("enforce_mechanism_debt_gate" in step for step in result["loop"]))
        self.assertIn("mechanism_debt_register", result["required_outputs"])
        self.assertIn("mechanism_execution_manifest", result["required_outputs"])
        self.assertTrue(any("plan_public_resource_acquisition" in step for step in result["loop"]))
        self.assertTrue(any("plan_hypothesis_test_dataset_construction" in step for step in result["loop"]))
        self.assertIn("explanatory_depth_assessment", result["required_outputs"])
        self.assertIn("public_resource_acquisition_plan", result["required_outputs"])
        self.assertIn("hypothesis_test_dataset_plan", result["required_outputs"])
        self.assertIn("representative", result["task_coverage_policy"]["forbidden"])
        self.assertIn("queued-only", result["task_coverage_policy"]["forbidden"])
        self.assertIn("continuation_launch_manifest", result["required_outputs"])
        self.assertIn("continuation_status_log", result["required_outputs"])
        self.assertNotIn("launch or queue", str(result))
        self.assertIn("investigator_mode", result)
        self.assertIn("hypothesis_ledger", result["required_outputs"])
        self.assertTrue(any("hypothesis ledger" in step for step in result["loop"]))
        self.assertTrue(any("plan_hypothesis_driven_dataset_acquisition" in step for step in result["loop"]))
        self.assertIn("hypothesis_driven_acquisition_plan", result["required_outputs"])

    def test_start_spectra_investigator_centers_hypotheses(self):
        result = start_spectra_investigator(
            dataset_description="Caduceus visible downstream sequence tasks",
            model_description="Caduceus-PS frozen representations with fresh probes",
            domain="regulatory DNA",
        )

        self.assertEqual(result["mode"], "spectra_investigator")
        self.assertIn("hypothesis state", result["mandate"])
        self.assertIn("hypothesis_ledger.json", result["required_artifacts"])
        self.assertTrue(any("Do not run another similarity axis" in result["experiment_rule"] for _ in [0]))
        self.assertIn("axis checklist with no interpretation", result["forbidden_patterns"])

    def test_update_hypothesis_ledger_generates_competing_explanations(self):
        observations = [
            {"axis": "6mer_surface_cosine_support", "task": "enhancers", "curve_score": "weak_supported"},
            {"axis": "6mer_surface_cosine_support", "task": "promoter_tata", "curve_score": "not_explanatory"},
            {"axis": "caduceus_frozen_representation_cosine_support", "task": "enhancers", "curve_score": "monotonic_supported"},
            {"axis": "composition_gc_cpg_length_entropy_support", "task": "H3", "curve_score": "not_explanatory"},
            {"axis": "local_motif_word_and_splice_grammar_support", "task": "splice_sites_all", "curve_score": "weak_supported"},
        ]

        result = update_hypothesis_ledger(observations, domain="regulatory DNA")

        self.assertEqual(result["mode"], "spectra_investigator_hypothesis_update")
        self.assertFalse(result["analysis_complete"])
        self.assertIn("6mer_surface_cosine_support", result["mixed_axes"])
        hypothesis_ids = {hypothesis["hypothesis_id"] for hypothesis in result["hypotheses"]}
        self.assertIn("surface_sequence_support", hypothesis_ids)
        self.assertIn("model_representation_support", hypothesis_ids)
        self.assertIn("regulatory_grammar_support", hypothesis_ids)
        self.assertIn("hypothesis_ledger.json", result["required_outputs"])
        self.assertIn("distinguish", result["required_next_action"])

    def test_choose_discriminating_experiment_prioritizes_residual_tests(self):
        ledger = update_hypothesis_ledger(
            [
                {"axis": "6mer_surface_cosine_support", "task": "enhancers", "curve_score": "weak_supported"},
                {"axis": "caduceus_frozen_representation_cosine_support", "task": "enhancers", "curve_score": "weak_supported"},
                {"axis": "caduceus_frozen_representation_cosine_support", "task": "promoter_tata", "curve_score": "not_explanatory"},
            ],
            domain="regulatory DNA",
        )

        result = choose_discriminating_experiment(ledger, domain="regulatory DNA")

        self.assertEqual(result["decision"], "run_discriminating_experiment")
        self.assertFalse(result["analysis_complete"])
        self.assertEqual(
            result["selected_experiment"]["experiment_id"],
            "residual_representation_after_surface_match",
        )
        self.assertIn("why_this_next_experiment.md", result["required_outputs"])
        self.assertIn("not chosen because it is the next registry", result["why_not_checklist"])

    def test_review_investigator_checkpoint_rejects_checklist_report(self):
        result = review_investigator_checkpoint(
            "Tested axes. Mixed results. Mechanism debt remains. Continuation completed.",
            has_observations=True,
            has_hypothesis_ledger=False,
            has_competing_explanations=False,
            has_discriminating_experiment=False,
            has_belief_update=False,
            launched_or_completed_next_experiment=True,
        )

        self.assertEqual(result["decision"], "investigator_checkpoint_failed")
        self.assertIn("hypothesis_ledger", result["missing_investigator_artifacts"])
        self.assertTrue(result["checklist_smells_without_interpretation"])

    def test_hypothesis_driven_dataset_acquisition_targets_live_hypothesis(self):
        ledger = {
            "hypotheses": [
                {
                    "hypothesis_id": "regulatory_grammar_support",
                    "statement": "Caduceus fails when held-out examples require motif grammar or regulatory context not supported in training.",
                    "current_status": "live",
                    "predicts": "motif-family novelty should degrade after k-mer and GC matching",
                },
                {
                    "hypothesis_id": "model_representation_support",
                    "statement": "Caduceus embedding support has residual explanatory power beyond input proxies.",
                    "current_status": "live",
                    "predicts": "embedding support should map to biological annotations",
                },
            ]
        }

        result = plan_hypothesis_driven_dataset_acquisition(
            ledger,
            current_data_limitations=[
                "processed rows lack genomic coordinates",
                "curated JASPAR motif annotations are unavailable",
                "no counterfactual variant-effect labels",
            ],
            domain="regulatory DNA",
            allowed_network=True,
        )

        self.assertEqual(result["decision"], "acquire_or_construct_hypothesis_test_dataset")
        self.assertEqual(result["mode"], "hypothesis_driven_dataset_acquisition")
        self.assertFalse(result["analysis_complete"])
        self.assertTrue(result["acquisition_triggers"]["needs_coordinates_or_ids"])
        self.assertTrue(result["acquisition_triggers"]["needs_curated_annotations"])
        self.assertTrue(result["acquisition_triggers"]["needs_counterfactual_or_variant_effect_data"])
        self.assertIn("hypothesis_driven_acquisition_plan.json", result["required_outputs"])
        self.assertTrue(
            any(
                spec["hypothesis_id"] == "regulatory_grammar_support"
                and "falsification_test" in spec
                for spec in result["hypothesis_acquisition_specs"]
            )
        )
        self.assertTrue(
            any(
                "download a dataset without naming the hypothesis" in pattern
                for pattern in result["forbidden_patterns"]
            )
        )

    def test_distiller_prioritizes_narrow_mechanism_signal(self):
        curve_scores = [
            {
                "axis": "jaspar_regulatory_motif_support",
                "task_name": "enhancers_types",
                "curve_score": "monotonic",
            },
            {
                "axis": "jaspar_regulatory_motif_support",
                "task_name": "enhancers",
                "curve_score": "non-explanatory",
            },
            {
                "axis": "jaspar_regulatory_motif_support",
                "task_name": "promoter_tata",
                "curve_score": "non-explanatory",
            },
            {"axis": "kmer_support", "task_name": "enhancers_types", "curve_score": "localized"},
            {"axis": "kmer_support", "task_name": "promoter_tata", "curve_score": "localized"},
            {"axis": "composition_support", "task_name": "enhancers_types", "curve_score": "monotonic"},
            {"axis": "composition_support", "task_name": "promoter_tata", "curve_score": "non-explanatory"},
        ]
        ledger = {
            "live_hypotheses": [
                {
                    "id": "H3_regulatory_motif_mechanism",
                    "claim": "TF motif grammar may govern enhancer generalization.",
                    "status": "live",
                }
            ]
        }

        result = distill_spectra_hypotheses(
            curve_scores,
            hypothesis_ledger=ledger,
            belief_update="Coordinate/provenance context remains live.",
            domain="regulatory DNA",
        )

        self.assertEqual(result["mode"], "spectra_distiller")
        self.assertFalse(result["analysis_complete"])
        self.assertEqual(
            result["distilled_hypotheses"][0]["hypothesis_id"],
            "enhancer_subtype_regulatory_grammar",
        )
        self.assertIn("task_signal_summary", result)
        self.assertTrue(
            any(
                "class-conditional SPECTRA" in experiment
                for experiment in result["recommended_handoff_to_investigator"]["experiments_to_run"]
            )
        )
        self.assertTrue(
            any("global motif mechanism" in warning for warning in result["what_not_to_do"])
        )
        self.assertIn("investigator_handoff.json", result["required_outputs"])

    def test_distiller_sharpens_handoff_to_label_artifact_when_supported(self):
        curve_scores = [
            {
                "axis": "motif_family_support",
                "task_labeling": "binary_enhancers",
                "curve_score": "localized",
            },
            {
                "axis": "motif_family_support",
                "task_labeling": "enhancers_types",
                "curve_score": "monotonic",
            },
            {
                "axis": "motif_family_residual_support",
                "task_labeling": "enhancers_types",
                "curve_score": "localized",
            },
        ]
        ledger = {
            "hypotheses": [
                {
                    "id": "source_or_label_construction_artifact",
                    "posterior_status": "strengthened",
                    "evidence": [
                        "enhancers_types is exactly the same sequence set as enhancers",
                        "source_name suffix exactly matches labels",
                    ],
                }
            ]
        }

        result = distill_spectra_hypotheses(
            curve_scores,
            hypothesis_ledger=ledger,
            belief_update="source_name is label-coded and no subtype provenance is visible",
            domain="regulatory DNA",
        )

        self.assertEqual(
            result["distilled_hypotheses"][0]["hypothesis_id"],
            "label_construction_before_biology",
        )
        self.assertIn("binary_enhancers", result["task_signal_summary"])
        self.assertIn("enhancers_types", result["task_signal_summary"])
        self.assertTrue(
            any("enhancer subtype biology" in warning for warning in result["what_not_to_do"])
        )

    def test_final_synthesis_writes_paper_ready_bounded_finding(self):
        findings = {
            "source_inputs": {
                "matched_probe": "/work/matched_curve_summary.json",
            },
            "hypothesis_assessment": {
                "hypothesis": "Within-class local cCRE support is a regulatory-context novelty axis.",
                "status_after_matched_probe": "weakened_after_matching",
                "precise_insight": (
                    "For ENCODE cCRE V4 windows, the apparent local-support degradation in "
                    "frozen Caduceus probes is not a strong independent regulatory-context "
                    "generalization axis after composition and overlap controls."
                ),
                "evidence_boundary": [
                    "Frozen Caduceus representation plus logistic probe only.",
                    "No full Caduceus fine-tuning evidence.",
                ],
            },
            "paper_results_log_paragraph": (
                "After matching cCRE class, label balance, GC/CpG composition, "
                "interval length, promoter-distance proxy, and overlap, the local-support "
                "ROC-AUC delta matched the same-pool hash-control delta."
            ),
        }

        result = synthesize_spectra_generalizability_finding(
            target_model="Caduceus",
            model_paper_context="Caduceus is a bidirectional equivariant long-range DNA sequence model.",
            spectra_findings=findings,
            vanilla_agent_summary="Vanilla found split leakage but did not run target-model controls.",
            domain="regulatory DNA",
        )

        self.assertEqual(result["mode"], "spectra_final_finding_synthesis")
        self.assertTrue(result["analysis_complete"])
        self.assertEqual(result["claim_type"], "controlled attenuation / negative-axis finding")
        self.assertIn("one_sentence_finding", result)
        self.assertIn("paper_ready_sections", result)
        self.assertIn("results_paragraph", result["paper_ready_sections"])
        self.assertIn("composition", " ".join(result["overclaim_guardrails"]).lower())
        self.assertIn("paper_ready_spectra_finding.md", result["required_outputs"])
        self.assertTrue(result["vanilla_comparison_available"])

    def test_final_synthesis_reopens_stronger_unresolved_composition_axis(self):
        findings = {
            "whole_loop_summary": {
                "composition_control_challenge": {
                    "composition_low_high_absolute_roc_auc_gap": 0.1621,
                },
                "matched_caduceus_probe_result": {
                    "matched_regulatory_context_support_split": {
                        "validation_middle_to_test_low_roc_auc_delta": 0.0167,
                    },
                    "matched_same_pool_hash_control_split": {
                        "validation_to_test_roc_auc_delta": 0.0171,
                    },
                },
            },
            "hypothesis_assessment": {
                "status_after_matched_probe": "weakened_after_matching",
                "precise_insight": "Local cCRE support attenuated after matching.",
            },
        }

        result = synthesize_spectra_generalizability_finding(
            target_model="Caduceus",
            model_paper_context="Caduceus DNA sequence model.",
            spectra_findings=findings,
            domain="regulatory DNA",
        )

        self.assertFalse(result["analysis_complete"])
        self.assertFalse(result["terminal_report_allowed"])
        self.assertEqual(result["route"], "return_to_investigator_new_primary_axis")
        self.assertEqual(
            result["dominant_unresolved_axis"]["axis_id"],
            "gc_cpg_length_composition_regime",
        )
        self.assertGreater(
            result["dominant_unresolved_axis"]["gap_to_matched_local_support_ratio"],
            5.0,
        )
        self.assertIn(
            "composition_regime_generalization",
            result["recommended_handoff_to_investigator"]["hypothesis_id"],
        )

    def test_select_spectra_execution_mode_prefers_retraining(self):
        result = select_spectra_execution_mode(
            has_raw_labeled_data=True,
            has_trainable_model_or_baseline=True,
            can_retrain=True,
            has_fixed_predictions=True,
        )

        self.assertEqual(result["mode"], "benchmark")
        self.assertTrue(result["benchmark_mode_required"])
        self.assertFalse(result["fixed_prediction_binning_allowed_as_primary_result"])

    def test_select_spectra_execution_mode_allows_diagnostic_fallback(self):
        result = select_spectra_execution_mode(
            has_raw_labeled_data=False,
            has_trainable_model_or_baseline=False,
            can_retrain=False,
            has_fixed_predictions=True,
            retraining_unavailable_reason="Only fixed leaderboard predictions are available.",
        )

        self.assertEqual(result["mode"], "audit_fallback")
        self.assertTrue(result["audit_fallback_allowed"])
        self.assertIn("diagnostic", result["claim_strength"])

    def test_suspicious_data_screen_requires_behavioral_followup(self):
        result = decide_next_spectra_experiment(
            axis_name="test_15mer_train_presence",
            axis_result_status="monotonic_supported",
            behavior_tested=False,
            benchmark_mode_feasible=True,
            has_suspicious_data_overlap=True,
        )

        self.assertEqual(result["decision"], "run_behavioral_followup")
        self.assertFalse(result["analysis_complete_for_axis"])
        self.assertTrue(result["warnings"])

    def test_concrete_blocker_creates_continuation_checkpoint(self):
        result = decide_next_spectra_experiment(
            axis_name="variant_sequence_windows",
            axis_result_status="not_evaluable",
            behavior_tested=False,
            benchmark_mode_feasible=True,
            has_suspicious_data_overlap=True,
            concrete_blocker="Genome windows are not available in the visible bundle.",
            blocker_recovery_attempted=True,
        )

        self.assertEqual(result["decision"], "continuation_checkpoint_with_blocker")
        self.assertFalse(result["analysis_complete_for_axis"])
        self.assertFalse(result["terminal_stop_allowed"])
        self.assertIn("Genome windows", result["concrete_blocker"])
        self.assertIn("launch", result["required_next_action"])
        self.assertIn("not sufficient", result["required_next_action"])

    def test_non_explanatory_axis_requires_next_axis_while_scope_remains(self):
        result = decide_next_spectra_experiment(
            axis_name="global_6mer_jaccard",
            axis_result_status="not_explanatory",
            behavior_tested=True,
            benchmark_mode_feasible=True,
            tested_axis_count=2,
            min_axis_tests=5,
            axis_search_exhausted=False,
        )

        self.assertEqual(result["decision"], "try_next_axis")
        self.assertFalse(result["analysis_complete_for_axis"])
        self.assertIn("untested axis class", result["required_next_action"])
        self.assertTrue(result["warnings"])

    def test_exhausted_search_escalates_scope_without_generalization_claim(self):
        result = decide_next_spectra_experiment(
            axis_name="cell_type_embedding_distance",
            axis_result_status="not_explanatory",
            behavior_tested=True,
            benchmark_mode_feasible=True,
            tested_axis_count=8,
            min_axis_tests=5,
            axis_search_exhausted=True,
        )

        self.assertEqual(result["decision"], "expand_search_scope")
        self.assertFalse(result["analysis_complete_for_axis"])
        self.assertIn("Do not conclude", result["required_next_action"])
        self.assertIn("larger scope", result["required_next_action"])

    def test_supported_axis_must_be_replicated_before_selection(self):
        unreplicated = decide_next_spectra_experiment(
            axis_name="target_model_embedding_distance",
            axis_result_status="monotonic_supported",
            behavior_tested=True,
            benchmark_mode_feasible=True,
            tested_axis_count=5,
            supported_axis_replicated=False,
        )
        replicated = decide_next_spectra_experiment(
            axis_name="target_model_embedding_distance",
            axis_result_status="monotonic_supported",
            behavior_tested=True,
            benchmark_mode_feasible=True,
            tested_axis_count=6,
            supported_axis_replicated=True,
            replication_reflection_complete=True,
            axis_depth_level="mechanism_supported_with_controls",
        )

        self.assertEqual(unreplicated["decision"], "replicate_candidate_axis")
        self.assertFalse(unreplicated["analysis_complete_for_axis"])
        self.assertIn("Immediately test", unreplicated["required_next_action"])
        self.assertEqual(replicated["decision"], "continue_after_supported_axis")
        self.assertFalse(replicated["analysis_complete_for_axis"])
        self.assertFalse(replicated["terminal_stop_allowed"])

    def test_supported_axis_requires_depth_classification_before_selection(self):
        result = decide_next_spectra_experiment(
            axis_name="target_model_embedding_distance",
            axis_result_status="monotonic_supported",
            behavior_tested=True,
            benchmark_mode_feasible=True,
            tested_axis_count=6,
            supported_axis_replicated=True,
            replication_reflection_complete=True,
        )

        self.assertEqual(result["decision"], "classify_explanatory_depth")
        self.assertTrue(result["axis_depth_unknown"])
        self.assertFalse(result["analysis_complete_for_axis"])
        self.assertIn("assess_explanatory_depth", result["required_next_action"])

    def test_supported_proxy_axis_cannot_be_selected_without_mechanism_debt(self):
        result = decide_next_spectra_experiment(
            axis_name="surface_kmer5",
            axis_result_status="monotonic_supported",
            behavior_tested=True,
            benchmark_mode_feasible=True,
            tested_axis_count=6,
            supported_axis_replicated=True,
            replication_reflection_complete=True,
            axis_depth_level="surface_proxy",
        )

        self.assertEqual(result["decision"], "enforce_mechanism_debt_gate")
        self.assertTrue(result["mechanism_debt_pending"])
        self.assertFalse(result["analysis_complete_for_axis"])
        self.assertIn("mechanism", result["required_next_action"])

    def test_blocker_cannot_bypass_mechanism_debt_for_supported_axis(self):
        result = decide_next_spectra_experiment(
            axis_name="surface_kmer5",
            axis_result_status="monotonic_supported",
            behavior_tested=True,
            benchmark_mode_feasible=True,
            concrete_blocker="No genomic coordinates in processed task bundle.",
            blocker_recovery_attempted=True,
            tested_axis_count=6,
            supported_axis_replicated=True,
            replication_reflection_complete=True,
            axis_depth_level="surface_proxy",
        )

        self.assertEqual(result["decision"], "enforce_mechanism_debt_gate")
        self.assertFalse(result["analysis_complete_for_axis"])
        self.assertTrue(result["mechanism_debt_pending"])

    def test_replicated_axis_requires_reflection_before_selection(self):
        result = decide_next_spectra_experiment(
            axis_name="target_model_embedding_distance",
            axis_result_status="monotonic_supported",
            behavior_tested=True,
            benchmark_mode_feasible=True,
            tested_axis_count=6,
            supported_axis_replicated=True,
            replication_reflection_complete=False,
        )

        self.assertEqual(result["decision"], "reflect_on_replication_pattern")
        self.assertFalse(result["analysis_complete_for_axis"])
        self.assertIn("Compare supported and non-supported", result["required_next_action"])

    def test_reflect_on_replication_evidence_generates_residual_axes(self):
        evidence = [
            {
                "axis_id": "surface_kmer5",
                "task": "enhancers",
                "status": "monotonic_supported",
                "supported": True,
            },
            {
                "axis_id": "surface_kmer5",
                "task": "splice_sites_acceptors",
                "status": "not_explanatory",
                "supported": False,
            },
            {
                "axis_id": "caduceus_representation",
                "task": "enhancers",
                "status": "monotonic_supported",
                "supported": True,
            },
            {
                "axis_id": "caduceus_representation",
                "task": "promoter_all",
                "status": "not_explanatory",
                "supported": False,
            },
        ]

        result = reflect_on_replication_evidence(evidence)

        self.assertEqual(result["decision"], "derive_and_test_residual_axes")
        self.assertFalse(result["analysis_complete"])
        self.assertIn("surface_kmer5", result["mixed_axes"])
        self.assertIn("caduceus_representation", result["mixed_axes"])
        self.assertTrue(
            any(
                candidate["axis_id"] == "residual_model_embedding_after_sequence_similarity"
                for candidate in result["residual_axis_candidates"]
            )
        )

    def test_model_space_axis_translation_generates_biological_hypotheses(self):
        result = translate_model_space_axis_to_domain_hypotheses(
            model_space_axis_name="caduceus_representation_residual_after_kmer_match",
            domain="regulatory DNA",
            model_description="Caduceus-PS frozen embeddings",
            supported_tasks=["enhancers", "H3K14ac"],
            non_supported_tasks=["promoter_all", "splice_sites_acceptors"],
            available_annotations=["sequence", "task_family", "source_name"],
        )

        self.assertEqual(result["decision"], "test_domain_translation_axes")
        self.assertIn("behavioral pointer", result["model_space_interpretation"])
        self.assertIn("model_space_biological_translation.json", result["required_outputs"])
        self.assertTrue(
            any(
                hypothesis["hypothesis_id"] == "motif_family_support"
                for hypothesis in result["domain_hypotheses"]
            )
        )
        self.assertTrue(
            any(
                "Do not report model-embedding distance" in behavior
                for behavior in result["required_agent_behavior"]
            )
        )

    def test_proxy_axis_requires_mechanism_depth(self):
        result = assess_explanatory_depth(
            candidate_axis_name="gc_cpg_low_complexity_support_after_kmer_match",
            candidate_axis_description="GC, CpG, entropy, and low-complexity support in regulatory DNA",
            domain="regulatory DNA",
            evidence_status="monotonic_supported",
            available_annotations=["sequence", "task_family"],
        )

        self.assertEqual(result["decision"], "continue_to_mechanism")
        self.assertEqual(result["explanatory_depth_level"], "surface_proxy")
        self.assertFalse(result["analysis_complete"])
        self.assertIn("proxy_to_mechanism_plan.json", result["required_outputs"])
        self.assertTrue(
            any(
                hypothesis["hypothesis_id"] == "curated_motif_family_support"
                for hypothesis in result["mechanism_hypotheses"]
            )
        )

    def test_mechanism_axis_with_controls_can_be_supported(self):
        result = assess_explanatory_depth(
            candidate_axis_name="motif_grammar_residual_after_kmer_gc_match",
            candidate_axis_description="motif spacing and orientation after matching k-mer, GC, and length",
            domain="regulatory DNA",
            evidence_status="monotonic_supported",
            tested_controls=["kmer5", "gc_cpg", "length"],
        )

        self.assertEqual(result["decision"], "mechanism_depth_supported")
        self.assertEqual(result["explanatory_depth_level"], "mechanism_supported_with_controls")
        self.assertFalse(result["analysis_complete"])
        self.assertTrue(result["continuation_required"])
        self.assertFalse(result["terminal_stop_allowed"])

    def test_mechanism_debt_gate_requires_local_mechanism_test(self):
        result = enforce_mechanism_debt_gate(
            supported_axis_name="gc_cpg_entropy_profile",
            axis_depth_level="surface_proxy",
            domain="regulatory DNA",
            evidence_status="monotonic_supported",
            available_local_inputs=["sequence", "labels", "Caduceus embeddings"],
        )

        self.assertEqual(result["decision"], "execute_local_mechanism_or_mediation_test")
        self.assertTrue(result["mechanism_debt_active"])
        self.assertFalse(result["analysis_complete"])
        self.assertFalse(result["stop_allowed"])
        self.assertIn("mechanism_execution_manifest.json", result["required_outputs"])
        self.assertTrue(any("motif/PWM" in item for item in result["sequence_domain_local_tests"]))
        self.assertTrue(any("Missing coordinates" in item for item in result["forbidden_outs"]))

    def test_mechanism_debt_gate_does_not_allow_proxy_only_future_work(self):
        result = enforce_mechanism_debt_gate(
            supported_axis_name="caduceus_embedding_cosine",
            axis_depth_level="model_space_pointer",
            domain="regulatory DNA",
            evidence_status="localized_supported",
            available_local_inputs=[],
            source_provenance_recovery_attempted=True,
            public_resource_acquisition_attempted=True,
            constructed_dataset_attempted=False,
        )

        self.assertEqual(result["decision"], "construct_hypothesis_test_dataset_for_mechanism")
        self.assertFalse(result["stop_allowed"])
        self.assertIn("Local benchmark-mode proxy audit", " ".join(result["forbidden_outs"]))

    def test_mechanism_debt_gate_continues_after_executed_proxy_test(self):
        result = enforce_mechanism_debt_gate(
            supported_axis_name="motif_grammar_support",
            axis_depth_level="curated_annotation_axis",
            domain="regulatory DNA",
            evidence_status="weak_supported",
            mechanism_tests_executed=["JASPAR motif-family split with k-mer/GC/length matched residual curve"],
            available_local_inputs=["sequence"],
        )

        self.assertEqual(result["decision"], "local_tests_executed_continue_to_public_mechanism")
        self.assertFalse(result["analysis_complete"])
        self.assertFalse(result["stop_allowed"])
        self.assertFalse(result["terminal_stop_allowed"])

    def test_public_resource_acquisition_plans_genomic_downloads(self):
        result = plan_public_resource_acquisition(
            domain="regulatory DNA",
            missing_resources=["curated TFBS", "conservation", "hg38 FASTA"],
            dataset_description="DNA intervals with chrom, start, end, sequence, and task labels",
            scientific_question="Do motif families and conservation explain residual Caduceus failure?",
            local_identifiers_available=["chrom", "start", "end", "sequence"],
            allowed_network=True,
            allow_large_downloads=False,
        )

        self.assertEqual(result["decision"], "attempt_public_resource_acquisition")
        self.assertTrue(result["has_mapping_identifiers"])
        self.assertIn("public_resource_search_log.json", result["required_outputs"])
        self.assertTrue(
            any(
                resource["resource_class"] == "curated_tf_motifs"
                for resource in result["candidate_public_resources"]
            )
        )

    def test_public_resource_acquisition_requires_mapping_ids(self):
        result = plan_public_resource_acquisition(
            domain="regulatory DNA",
            missing_resources=["TFBS annotations"],
            dataset_description="anonymous rows with labels only",
            local_identifiers_available=["row_id"],
            allowed_network=True,
        )

        self.assertEqual(result["decision"], "recover_source_provenance_before_download")
        self.assertFalse(result["has_mapping_identifiers"])
        self.assertIn("mapping identifiers", result["required_next_action"])
        self.assertIn("source_provenance_recovery_log.json", result["required_outputs"])
        self.assertTrue(
            any(
                "upstream dataset provenance" in behavior
                for behavior in result["required_agent_behavior"]
            )
        )

    def test_public_resource_acquisition_uses_dataset_sources_as_mapping_evidence(self):
        result = plan_public_resource_acquisition(
            domain="regulatory DNA",
            missing_resources=["cCRE annotations"],
            dataset_description="processed sequence strings",
            local_identifiers_available=["sequence"],
            dataset_sources=["original benchmark repository has BED intervals"],
            allowed_network=True,
        )

        self.assertEqual(result["decision"], "attempt_public_resource_acquisition")
        self.assertTrue(result["has_mapping_identifiers"])
        self.assertIn("original benchmark repository", " ".join(result["dataset_sources"]))

    def test_public_resource_acquisition_plans_molecular_resources(self):
        result = plan_public_resource_acquisition(
            domain="small molecule drug discovery",
            missing_resources=["target family", "assay mechanism"],
            dataset_description="compound rows with SMILES, assay_id, and target_id",
            local_identifiers_available=["SMILES", "assay_id", "target_id"],
            allowed_network=True,
        )

        self.assertEqual(result["decision"], "attempt_public_resource_acquisition")
        self.assertTrue(result["has_mapping_identifiers"])
        self.assertTrue(
            any(
                resource["resource_class"] == "chemical_bioactivity_and_targets"
                for resource in result["candidate_public_resources"]
            )
        )
        self.assertIn("SMILES/InChI/InChIKey", result["source_provenance_targets"])

    def test_public_resource_acquisition_plans_perturbation_provenance(self):
        result = plan_public_resource_acquisition(
            domain="single-cell perturbation",
            missing_resources=["pathway ontology"],
            dataset_description="anonymous cells with labels only",
            local_identifiers_available=["row_id"],
            allowed_network=True,
        )

        self.assertEqual(result["decision"], "recover_source_provenance_before_download")
        self.assertIn("gene symbols or Ensembl IDs", result["source_provenance_targets"])
        self.assertTrue(
            any(
                resource["resource_class"] == "gene_pathway_ontology"
                for resource in result["candidate_public_resources"]
            )
        )

    def test_hypothesis_test_dataset_construction_plans_genomic_extension(self):
        result = plan_hypothesis_test_dataset_construction(
            domain="regulatory DNA",
            mechanism_hypothesis="Caduceus probe performance degrades when motif-family and cCRE context support are low after matching k-mer and GC/CpG proxies.",
            current_dataset_limitation="The visible benchmark rows are sequence strings without stable coordinates for cCRE joins.",
            available_local_resources=["trained Caduceus checkpoint", "probe code"],
            acquired_public_resources=["hg38 FASTA", "ENCODE cCRE BED", "JASPAR motifs"],
            target_model_description="Caduceus frozen representation with logistic probe",
            desired_labels=["cCRE class", "motif family support"],
            candidate_units=["variant-centered sequence windows"],
        )

        self.assertEqual(result["decision"], "construct_hypothesis_test_dataset")
        self.assertTrue(result["has_seed_resources"])
        self.assertIn("hypothesis-test dataset", result["agent_instruction"])
        self.assertIn("hypothesis_test_dataset_plan.json", result["required_outputs"])
        self.assertIn("constructed_dataset_leakage_audit.json", result["required_outputs"])
        self.assertTrue(
            any(
                candidate["dataset_type"] == "sequence_window_dataset"
                for candidate in result["dataset_candidates"]
            )
        )
        self.assertTrue(
            any(
                "public/local resources" in behavior
                for behavior in result["required_agent_behavior"]
            )
        )

    def test_hypothesis_test_dataset_construction_plans_molecular_extension(self):
        result = plan_hypothesis_test_dataset_construction(
            domain="small molecule drug discovery",
            mechanism_hypothesis="The model fails for compounds whose target family and scaffold are unsupported by training examples.",
            current_dataset_limitation="The benchmark includes SMILES but lacks target-family annotations.",
            available_local_resources=["SMILES table", "activity labels"],
            acquired_public_resources=["ChEMBL", "UniProt"],
            desired_labels=["bioactivity", "target family"],
            candidate_units=["compound-target assay rows"],
        )

        self.assertEqual(result["decision"], "construct_hypothesis_test_dataset")
        self.assertTrue(
            any(
                candidate["dataset_type"] == "compound_or_compound_target_dataset"
                for candidate in result["dataset_candidates"]
            )
        )
        self.assertTrue(
            any(
                "ChEMBL" in resource
                for resource in result["dataset_candidates"][0]["public_or_local_resources"]
            )
        )
        self.assertIn("constructed_dataset_spectra_results.json", result["required_outputs"])

    def test_prepare_dataset_constructor_request_for_enhancer_case(self):
        handoff = {
            "hypothesis_id": "label_construction_before_biology",
            "question": "The old enhancers_types task is label-construction limited before it is biological.",
        }
        investigator_summary = {
            "hypotheses": [
                {
                    "id": "label_construction_before_biology",
                    "evidence": [
                        "enhancers_types masks source_name as label-coded",
                        "coordinates are missing",
                        "same_sequence_set as binary enhancers",
                    ],
                }
            ]
        }

        result = prepare_dataset_constructor_request(
            handoff,
            investigator_summary=investigator_summary,
            domain="regulatory DNA",
            target_model_description="Caduceus frozen probes",
        )

        self.assertEqual(result["mode"], "spectra_dataset_constructor_request")
        self.assertEqual(result["hypothesis_id"], "label_construction_before_biology")
        self.assertIn("coordinate-backed enhancer subtype", result["construction_goal"])
        self.assertIn("label_name", result["dataset_spec"]["required_row_fields"])
        self.assertIn("split_candidates/", result["required_outputs"])
        self.assertIn("dataset_card.md", result["handoff_back_to_investigator"]["must_include"])
        self.assertTrue(
            any("reverse-complement" in gate for gate in result["quality_gates"])
        )

    def test_prepare_dataset_scout_request_preserves_enhancer_inconsistency(self):
        handoff = {
            "hypothesis_id": "label_construction_before_biology",
            "question": "Old enhancers_types looks label-construction-limited.",
        }
        investigator_summary = {
            "results": [
                "Old motif-family support looked explanatory.",
                "Revised coordinate-backed motif-family support was non-explanatory.",
            ]
        }

        result = prepare_dataset_scout_request(
            handoff,
            investigator_summary=investigator_summary,
            domain="regulatory DNA",
            target_model_description="Caduceus frozen probes",
            min_candidates=5,
        )

        self.assertEqual(result["mode"], "spectra_dataset_scout_request")
        self.assertEqual(result["decision"], "scout_before_construction")
        self.assertIn("Old enhancer-type motif support", result["inconsistency_to_explain"])
        self.assertGreaterEqual(len(result["candidate_resource_classes"]), 5)
        self.assertIn("dataset_candidate_table.csv", result["required_outputs"])
        self.assertTrue(
            any("does not construct" in gate for gate in result["quality_gates"])
        )

    def test_distill_dataset_scout_output_promotes_ranked_candidate(self):
        candidates = [
            {
                "candidate_id": "old_nt_enhancers_types",
                "dataset_or_resource": "old NT enhancer types",
                "labels_available": "named class labels available",
                "coordinates_or_stable_ids": "stable row ids only",
                "provenance_quality": "source provenance but row provenance masked",
                "construction_feasibility": "local bounded feasible",
                "what_inconsistency_it_tests": "old versus revised benchmark construction",
                "known_blockers": "coordinates unavailable",
                "recommendation": "Comparator only; do not promote as top construction target",
            },
            {
                "candidate_id": "revised_nt_enhancers_types",
                "dataset_or_resource": "NT revised enhancer types",
                "labels_available": "named class labels available",
                "coordinates_or_stable_ids": "chrom:start-end coordinates",
                "provenance_quality": "good source provenance",
                "construction_feasibility": "local bounded feasible",
                "what_inconsistency_it_tests": "old versus revised benchmark construction",
                "known_blockers": "label 1/2 semantics inferred",
                "recommendation": "Required comparator; do not simply repackage",
            },
            {
                "candidate_id": "encode_screen_ccre",
                "dataset_or_resource": "ENCODE SCREEN cCRE",
                "labels_available": "cCRE class labels",
                "coordinates_or_stable_ids": "BED coordinates",
                "provenance_quality": "high versioned source",
                "construction_feasibility": "bounded feasible",
                "what_inconsistency_it_tests": "coordinate regulatory context",
                "known_blockers": "",
                "recommendation": "Top construction candidate",
            },
            {
                "candidate_id": "fantom5",
                "dataset_or_resource": "FANTOM5 enhancer atlas",
                "labels_available": "documented enhancer activity labels",
                "coordinates_or_stable_ids": "coordinate-backed stable enhancer ids",
                "provenance_quality": "high versioned source provenance",
                "construction_feasibility": "high bounded feasible",
                "what_inconsistency_it_tests": "independent enhancer activity and tissue specificity",
                "known_blockers": "",
                "recommendation": "Backup construction candidate",
            },
        ]

        result = distill_dataset_scout_output(
            candidates,
            inconsistency_ledger=[{"inconsistency": "old motif support versus revised non-explanatory"}],
            scout_report="Compared revised, ENCODE, and VISTA candidates.",
            domain="regulatory DNA",
            min_candidates=3,
        )

        self.assertEqual(result["mode"], "spectra_dataset_scout_distiller")
        self.assertEqual(result["decision"], "promote_to_dataset_constructor")
        self.assertTrue(result["constructor_handoff"]["enabled"])
        self.assertFalse(result["blocking_gaps"])
        self.assertEqual(
            result["constructor_handoff"]["top_candidate"]["candidate_id"],
            "encode_screen_ccre",
        )

    def test_distill_dataset_scout_output_requires_more_scouting(self):
        result = distill_dataset_scout_output(
            [
                {
                    "candidate_id": "only_candidate",
                    "dataset_or_resource": "one easy dataset",
                    "labels_available": "labels",
                    "coordinates_or_stable_ids": "coordinates",
                    "provenance_quality": "good",
                    "construction_feasibility": "feasible",
                    "what_inconsistency_it_tests": "",
                }
            ],
            scout_report="Only checked revised dataset.",
            domain="regulatory DNA",
            min_candidates=3,
        )

        self.assertEqual(result["decision"], "continue_scouting")
        self.assertTrue(result["blocking_gaps"])
        self.assertTrue(result["continue_scouting_request"]["enabled"])

    def test_distill_dataset_constructor_output_handoffs_ready_package(self):
        manifest = {
            "output_directory": "/work/dataset_package",
            "artifacts": {
                "dataset_card": "/work/dataset_card.md",
                "construction_manifest": "/work/construction_manifest.json",
                "label_semantics": "/work/label_semantics.json",
                "provenance_table": "/work/provenance_table.csv",
                "sequence_table": "/work/sequence_table.parquet",
                "spectra_ready_schema": "/work/spectra_ready_schema.json",
                "mapping_validation": "/work/mapping_validation.json",
                "leakage_audit": "/work/leakage_audit.json",
                "confounder_audit": "/work/confounder_audit.json",
                "recommended_spectra_run": "/work/recommended_spectra_run.json",
            },
            "primary_dataset": {
                "name": "revised coordinate-backed enhancers_types",
                "rows": 33000,
                "coordinate_parse_fraction": 1.0,
                "assembly_status": "unknown_human_assembly_not_loader_proven",
            },
            "motif_resources": {"available": True},
        }
        mapping = {
            "primary_dataset_validated": True,
            "coordinate_parse_fraction": 1.0,
            "assembly_status": "unknown_human_assembly_not_loader_proven",
            "required_fields_present": {
                "row_id": True,
                "sequence": True,
                "chrom": True,
                "start": True,
                "end": True,
                "label_id": True,
                "label_name": True,
            },
        }
        leakage = {
            "cross_source_split_exact_duplicate_rows": 0,
            "cross_source_split_reverse_complement_duplicate_rows": 0,
            "cross_source_split_coordinate_duplicate_rows": 0,
            "coordinate_overlap": {"cross_source_split_overlapping_interval_pairs": 0},
            "exact_duplicate_rows": 10,
        }
        labels = {
            "loader_proven_class_dictionary_available": False,
            "labels": {
                "0": {"label_name": "none/non-enhancer", "semantic_confidence": "high"},
                "1": {"label_name": "tissue-specific enhancer", "semantic_confidence": "medium_inferred_not_loader_proven"},
                "2": {"label_name": "tissue-invariant enhancer", "semantic_confidence": "medium_inferred_not_loader_proven"},
            },
        }
        recommended = {
            "recommended_first_axis": "motif_family_support_levels_chr20_chr21",
            "recommended_first_split_file": "/work/split_candidates/motif.csv",
            "test_levels": ["low", "medium", "high"],
            "secondary_axis": {
                "name": "residual_motif_support_after_matching",
                "split_file": "/work/split_candidates/residual.csv",
            },
        }

        result = distill_dataset_constructor_output(
            manifest,
            mapping_validation=mapping,
            leakage_audit=leakage,
            label_semantics=labels,
            recommended_spectra_run=recommended,
            domain="regulatory DNA",
        )

        self.assertEqual(result["mode"], "spectra_dataset_package_distiller")
        self.assertEqual(result["decision"], "handoff_to_investigator")
        self.assertTrue(result["handoff_to_investigator"]["enabled"])
        self.assertIn("Genome assembly", " ".join(result["nonblocking_caveats"]))
        self.assertTrue(
            any("duplicate-collapsed sensitivity" in step for step in result["handoff_to_investigator"]["required_investigator_steps"])
        )

    def test_distill_dataset_constructor_output_accepts_main_artifacts_schema(self):
        manifest = {
            "ready_for_investigator": True,
            "main_artifacts": {
                "report": "/work/constructor_report.md",
                "labels": "/work/label_semantics.json",
                "provenance": "/work/provenance.json",
                "sequence_or_coordinate_table": "/work/sequence_table.csv.gz",
                "schema": "/work/schema.json",
                "mapping_validation": "/work/mapping_validation.json",
                "leakage_audit": "/work/leakage_audit.json",
                "confounder_audit": "/work/confounder_audit.json",
                "recommended_spectra_run": "/work/recommended_spectra_run.json",
            },
            "records": {
                "bounded_sequence_records": 100000,
                "bounded_sequence_records_with_sequence": 100000,
                "full_registry_coordinate_records": 2348854,
            },
            "split_candidates": {
                "recommended_regulatory_context_support_split": {
                    "recommended": True,
                    "path": "/work/splits/context.csv",
                    "split_counts": {
                        "train_high_context_support": 60000,
                        "validation_middle_context_support": 20000,
                        "test_low_context_support": 20000,
                    },
                }
            },
        }
        mapping = {
            "sequence_extraction": {
                "sample_records_with_sequence": 100000,
            }
        }
        leakage = {
            "duplicate_ccre_ids_in_full_registry": 0,
            "duplicate_coordinates_in_full_registry": 0,
        }
        labels = {
            "classes": {
                "dELS": "distal enhancer-like signature",
                "pELS": "proximal enhancer-like signature",
                "PLS": "promoter-like signature",
            }
        }
        recommended = {
            "first_run": {
                "primary_spectra_axis": "within-class local cCRE support / regulatory-context novelty",
                "primary_split_file": "/work/splits/context.csv",
            }
        }

        result = distill_dataset_constructor_output(
            manifest,
            mapping_validation=mapping,
            leakage_audit=leakage,
            label_semantics=labels,
            recommended_spectra_run=recommended,
            domain="regulatory DNA",
        )

        self.assertEqual(result["decision"], "handoff_to_investigator")
        self.assertTrue(result["handoff_to_investigator"]["enabled"])
        self.assertEqual(result["handoff_to_investigator"]["rows"], 100000)

    def test_distill_dataset_constructor_output_returns_blocked_package(self):
        result = distill_dataset_constructor_output(
            {"artifacts": {}, "primary_dataset": {"rows": 0}},
            mapping_validation={"primary_dataset_validated": False},
            leakage_audit={"cross_source_split_exact_duplicate_rows": 5},
            label_semantics={"labels": {"1": {}}},
            recommended_spectra_run={},
            domain="regulatory DNA",
        )

        self.assertEqual(result["decision"], "return_to_dataset_constructor")
        self.assertTrue(result["blocking_gaps"])
        self.assertTrue(result["return_to_constructor"]["enabled"])

    def test_hypothesis_test_dataset_construction_requires_resources_or_network(self):
        result = plan_hypothesis_test_dataset_construction(
            domain="clinical",
            mechanism_hypothesis="A site/time context axis drives deployment failure.",
            current_dataset_limitation="Only anonymized rows are available.",
            allowed_network=False,
        )

        self.assertEqual(result["decision"], "blocked_without_public_or_local_resources")
        self.assertFalse(result["has_seed_resources"])
        self.assertIn("resources needed", result["required_next_action"])

    def test_environment_blocker_requires_recovery_attempt(self):
        result = decide_next_spectra_experiment(
            axis_name="target_model_probe",
            axis_result_status="not_evaluable",
            behavior_tested=False,
            benchmark_mode_feasible=True,
            has_suspicious_data_overlap=True,
            concrete_blocker="Missing torch and transformers.",
        )

        self.assertEqual(result["decision"], "attempt_blocker_recovery")
        self.assertFalse(result["analysis_complete_for_axis"])
        self.assertIn("environment", result["required_next_action"])


if __name__ == "__main__":
    unittest.main()
