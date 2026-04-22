from __future__ import annotations

import unittest

from backend_lab.config import BackendConfig
from backend_lab.dataset import build_cell_context
from backend_lab.es_experiment import build_es_query_from_lookup_payload, rerank_es_hits
from backend_lab.preprocess import build_seed_schema, lookup_payload_from_preprocessing, lookup_payload_variants_from_preprocessing
from backend_lab.semantic import build_semantic_candidates, merge_shepherd_decision, should_run_cria_shepherd
from backend_lab.table_profile import build_table_profile_seed


class BackendLabTableAwareTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = BackendConfig.from_env()

    def test_g04_table_profile_detects_biography(self) -> None:
        profile = build_table_profile_seed(self.config.dataset_root, "2T_2022", "G04AXW0O").to_dict()
        self.assertEqual(profile["table_semantic_family"], "BIOGRAPHY")
        self.assertGreaterEqual(profile["confidence"], 0.8)
        self.assertEqual(profile["column_roles"]["0"][0]["role"], "PERSON_NAME_OR_ALIAS")
        self.assertEqual(profile["column_roles"]["1"][0]["role"], "BIRTH_DATE")
        self.assertEqual(profile["column_roles"]["4"][0]["role"], "COUNTRY")

    def test_ohgi_table_profile_detects_geography(self) -> None:
        profile = build_table_profile_seed(self.config.dataset_root, "2T_2022", "OHGI1JNY").to_dict()
        self.assertEqual(profile["table_semantic_family"], "GEOGRAPHY")
        self.assertIn(
            profile["column_roles"]["1"][0]["role"],
            {"BODY_OF_WATER_NAME", "GEOGRAPHIC_FEATURE_NAME"},
        )

    def test_seed_schema_uses_table_profile_for_weak_person_alias(self) -> None:
        context = build_cell_context(
            self.config.dataset_root,
            "2T_2022",
            "G04AXW0O",
            row_id=2,
            col_id=0,
        )
        profile = build_table_profile_seed(self.config.dataset_root, "2T_2022", "G04AXW0O").to_dict()
        schema = build_seed_schema(context, table_profile=profile).to_dict()
        cell_hypothesis = schema["cell_hypothesis"]
        retrieval_plan = schema["retrieval_plan"]

        self.assertEqual(cell_hypothesis["mention_strength"], "weak")
        self.assertEqual(cell_hypothesis["entity_hypotheses"][0]["coarse_type"], "PERSON")
        self.assertEqual(cell_hypothesis["entity_hypotheses"][0]["fine_type"], "HUMAN")
        self.assertEqual(retrieval_plan["hard_filters"]["coarse_type"], ["PERSON"])
        self.assertEqual(retrieval_plan["hard_filters"]["fine_type"], ["HUMAN"])

    def test_lookup_payload_variants_include_backoff_stages(self) -> None:
        context = build_cell_context(
            self.config.dataset_root,
            "2T_2022",
            "G04AXW0O",
            row_id=3,
            col_id=0,
        )
        profile = build_table_profile_seed(self.config.dataset_root, "2T_2022", "G04AXW0O").to_dict()
        schema = build_seed_schema(context, table_profile=profile).to_dict()
        variants = lookup_payload_variants_from_preprocessing(schema, top_k=10)
        stage_names = [item["stage"] for item in variants]

        self.assertIn("primary", stage_names)
        self.assertIn("hypothesis_1_relax_fine", stage_names)
        self.assertIn("no_type_context_only", stage_names)

    def test_body_of_water_column_overrides_country_interpretation(self) -> None:
        context = build_cell_context(
            self.config.dataset_root,
            "2T_2022",
            "OHGI1JNY",
            row_id=20,
            col_id=1,
        )
        profile = build_table_profile_seed(self.config.dataset_root, "2T_2022", "OHGI1JNY").to_dict()
        schema = build_seed_schema(context, table_profile=profile).to_dict()
        payload = lookup_payload_from_preprocessing(schema, top_k=10)
        es_query = build_es_query_from_lookup_payload(payload, preprocessing_schema=schema)

        self.assertEqual(schema["retrieval_plan"]["hard_filters"]["coarse_type"], ["LOCATION"])
        self.assertEqual(schema["retrieval_plan"]["hard_filters"]["fine_type"], ["LANDMARK"])
        self.assertEqual(payload["coarse_hints"], ["LOCATION"])
        self.assertEqual(payload["fine_hints"], ["LANDMARK"])
        self.assertIn("Lake Malawi", str(es_query))

    def test_unsupported_qualifier_penalizes_related_entity(self) -> None:
        context = build_cell_context(
            self.config.dataset_root,
            "2T_2022",
            "OHGI1JNY",
            row_id=13,
            col_id=1,
        )
        profile = build_table_profile_seed(self.config.dataset_root, "2T_2022", "OHGI1JNY").to_dict()
        schema = build_seed_schema(context, table_profile=profile).to_dict()
        payload = lookup_payload_from_preprocessing(schema, top_k=100)
        reranked = rerank_es_hits(
            es_result={
                "hits": {
                    "hits": [
                        {
                            "_score": 20.0,
                            "_source": {
                                "qid": "Q-related",
                                "label": "Crystal Lake (Michigan)",
                                "aliases": [],
                                "coarse_type": "LOCATION",
                                "fine_type": "LANDMARK",
                                "item_category": "ENTITY",
                                "prior": 0.2,
                                "wikipedia_url": "Crystal_Lake_(Michigan)",
                                "context_string": "small lake in Michigan",
                                "description": "lake in Michigan",
                            },
                        },
                        {
                            "_score": 18.0,
                            "_source": {
                                "qid": "Q-primary",
                                "label": "Lake Michigan",
                                "aliases": ["Michigan"],
                                "coarse_type": "LOCATION",
                                "fine_type": "LANDMARK",
                                "item_category": "ENTITY",
                                "prior": 0.4,
                                "wikipedia_url": "Lake_Michigan",
                                "context_string": "Great Lakes United States freshwater lake",
                                "description": "one of the Great Lakes of North America",
                            },
                        },
                    ]
                }
            },
            lookup_payload=payload,
            preprocessing_schema=schema,
        )

        self.assertEqual(reranked[0]["qid"], "Q-primary")
        related_features = next(item["features"] for item in reranked if item["qid"] == "Q-related")
        self.assertIn("crystal", related_features["unsupported_qualifier_tokens"])

    def test_cache_mode_override(self) -> None:
        self.assertFalse(self.config.with_cache_mode("off").llm_cache_enabled)
        self.assertTrue(self.config.with_cache_mode("on").llm_cache_enabled)
        self.assertEqual(self.config.with_cache_mode("env").llm_cache_enabled, self.config.llm_cache_enabled)

    def test_cria_shepherd_triggers_on_same_label_authority_conflict(self) -> None:
        decision = {
            "selected_qid": "Q-low",
            "selected_label": "Lake Victoria",
            "confidence": 0.88,
            "margin": 2.2,
            "abstain": False,
        }
        candidates = [
            {
                "qid": "Q-low",
                "label": "Lake Victoria",
                "features": {
                    "final_score": 36.0,
                    "prior": 0.1,
                    "weighted_context_overlap": 0.8,
                },
                "source": {
                    "item_category": "ENTITY",
                    "coarse_type": "LOCATION",
                    "fine_type": "LANDMARK",
                    "description": "lake in Washington",
                    "context_string": "Washington state",
                },
            },
            {
                "qid": "Q-high",
                "label": "Lake Victoria",
                "features": {
                    "final_score": 35.0,
                    "prior": 0.56,
                    "weighted_context_overlap": 1.7,
                },
                "source": {
                    "item_category": "ENTITY",
                    "coarse_type": "LOCATION",
                    "fine_type": "LANDMARK",
                    "description": "lake in east-central Africa",
                    "context_string": "Uganda Kenya Tanzania African Great Lakes",
                },
            },
        ]

        trigger = should_run_cria_shepherd(decision=decision, reranked_candidates=candidates)

        self.assertTrue(trigger.should_run)
        self.assertIn("same_label_cluster", trigger.reason_codes)
        self.assertIn("authority_conflict", trigger.reason_codes)

    def test_cria_shepherd_can_override_hard_ambiguity(self) -> None:
        decision = {
            "selected_qid": "Q-low",
            "selected_label": "Lake Victoria",
            "confidence": 0.88,
            "margin": 2.2,
            "abstain": False,
            "reason_codes": ["strong_lexical_match"],
        }
        candidates = [
            {"qid": "Q-low", "label": "Lake Victoria", "features": {}, "source": {}},
            {"qid": "Q-high", "label": "Lake Victoria", "features": {}, "source": {}},
        ]
        trigger = should_run_cria_shepherd(
            decision=decision,
            reranked_candidates=[
                {
                    "qid": "Q-low",
                    "label": "Lake Victoria",
                    "features": {"final_score": 36.0, "prior": 0.1},
                    "source": {"item_category": "ENTITY", "coarse_type": "LOCATION", "fine_type": "LANDMARK"},
                },
                {
                    "qid": "Q-high",
                    "label": "Lake Victoria",
                    "features": {"final_score": 35.0, "prior": 0.56},
                    "source": {"item_category": "ENTITY", "coarse_type": "LOCATION", "fine_type": "LANDMARK"},
                },
            ],
        )
        shepherd_result = {
            "selected_qid": "Q-high",
            "confidence": 0.86,
            "abstain": False,
            "reason": "The row context supports the primary African lake.",
        }

        merged = merge_shepherd_decision(
            deterministic_decision=decision,
            shepherd_result=shepherd_result,
            reranked_candidates=candidates,
            trigger=trigger,
        )

        self.assertEqual(merged["selected_qid"], "Q-high")
        self.assertEqual(merged["resolution_mode"], "shepherd_override_ambiguity")
        self.assertFalse(merged["abstain"])

    def test_cria_shepherd_unabstains_when_it_confirms_top_candidate(self) -> None:
        decision = {
            "selected_qid": "Q-person",
            "selected_label": "Nikki Randall",
            "confidence": 0.51,
            "margin": 0.8,
            "abstain": True,
            "reason_codes": ["same_type_cluster"],
        }
        candidates = [{"qid": "Q-person", "label": "Nikki Randall", "features": {}, "source": {}}]
        trigger = should_run_cria_shepherd(decision=decision, reranked_candidates=candidates)
        shepherd_result = {
            "selected_qid": "Q-person",
            "confidence": 0.84,
            "abstain": False,
            "reason": "The selected candidate matches the row context.",
        }

        merged = merge_shepherd_decision(
            deterministic_decision=decision,
            shepherd_result=shepherd_result,
            reranked_candidates=candidates,
            trigger=trigger,
        )

        self.assertEqual(merged["resolution_mode"], "shepherd_confirmed")
        self.assertFalse(merged["abstain"])

    def test_cria_shepherd_does_not_trigger_on_same_type_cluster_alone(self) -> None:
        decision = {
            "selected_qid": "Q-top",
            "selected_label": "Example Person",
            "confidence": 0.86,
            "margin": 3.2,
            "abstain": False,
        }
        candidates = [
            {
                "qid": f"Q-{index}",
                "label": f"Example Person {index}",
                "features": {
                    "final_score": 40.0 - index,
                    "prior": 0.4,
                    "weighted_context_overlap": 1.2,
                },
                "source": {
                    "item_category": "ENTITY",
                    "coarse_type": "PERSON",
                    "fine_type": "HUMAN",
                    "description": "person",
                    "context_string": "biography",
                },
            }
            for index in range(5)
        ]

        trigger = should_run_cria_shepherd(decision=decision, reranked_candidates=candidates)

        self.assertFalse(trigger.should_run)

    def test_cria_shepherd_triggers_on_generic_cleaner_label_competitor(self) -> None:
        decision = {
            "selected_qid": "Q-compound",
            "selected_label": "Alpha Beta Gamma",
            "confidence": 0.86,
            "margin": 2.6,
            "abstain": False,
        }
        candidates = [
            {
                "qid": "Q-compound",
                "label": "Alpha Beta Gamma",
                "features": {
                    "final_score": 28.7,
                    "prior": 0.32,
                    "mention_token_coverage": 1.0,
                    "extra_label_token_count": 2,
                    "expected_descriptor_overlap": 1,
                    "label_context_support_score": 0.62,
                },
                "source": {"item_category": "ENTITY", "coarse_type": "LOCATION", "fine_type": "LANDMARK"},
            },
            {
                "qid": "Q-clean",
                "label": "Alpha Gamma",
                "features": {
                    "final_score": 25.9,
                    "prior": 0.55,
                    "mention_token_coverage": 1.0,
                    "extra_label_token_count": 1,
                    "expected_descriptor_overlap": 1,
                    "label_context_support_score": 0.75,
                },
                "source": {"item_category": "ENTITY", "coarse_type": "LOCATION", "fine_type": "LANDMARK"},
            },
        ]

        trigger = should_run_cria_shepherd(decision=decision, reranked_candidates=candidates)

        self.assertTrue(trigger.should_run)
        self.assertIn("cleaner_label_competitor", trigger.reason_codes)

    def test_cria_shepherd_triggers_on_generic_underspecified_label_competitor(self) -> None:
        decision = {
            "selected_qid": "Q-bare",
            "selected_label": "Delta",
            "confidence": 0.86,
            "margin": 2.7,
            "abstain": False,
        }
        candidates = [
            {
                "qid": "Q-bare",
                "label": "Delta",
                "features": {
                    "final_score": 28.9,
                    "prior": 0.16,
                    "mention_token_coverage": 1.0,
                    "extra_label_token_count": 0,
                    "expected_descriptor_overlap": 0,
                    "weighted_context_overlap": 0.5,
                    "label_context_support_score": 0.45,
                },
                "source": {"item_category": "ENTITY", "coarse_type": "LOCATION", "fine_type": "LANDMARK"},
            },
            {
                "qid": "Q-expanded",
                "label": "Context Delta",
                "features": {
                    "final_score": 26.0,
                    "prior": 0.5,
                    "mention_token_coverage": 1.0,
                    "extra_label_token_count": 1,
                    "expected_descriptor_overlap": 2,
                    "weighted_context_overlap": 1.1,
                    "label_context_support_score": 0.75,
                },
                "source": {"item_category": "ENTITY", "coarse_type": "LOCATION", "fine_type": "LANDMARK"},
            },
        ]

        trigger = should_run_cria_shepherd(decision=decision, reranked_candidates=candidates)

        self.assertTrue(trigger.should_run)
        self.assertIn("underspecified_label_competitor", trigger.reason_codes)

    def test_shepherd_shortlist_keeps_label_context_competitor(self) -> None:
        reranked = []
        for index in range(12):
            reranked.append(
                {
                    "qid": f"Q-same-{index}",
                    "label": "Victoria Lake",
                    "reranked_rank": index + 1,
                    "raw_rank": index + 1,
                    "features": {
                        "candidate_family": "PRIMARY",
                        "final_score": 30.0 - index,
                        "heuristic_score": 30.0 - index,
                        "prior": 0.1 + (index / 100.0),
                        "has_wikipedia": True,
                        "label_context_support_score": 0.5,
                        "mention_token_coverage": 1.0,
                    },
                    "source": {
                        "item_category": "ENTITY",
                        "coarse_type": "LOCATION",
                        "fine_type": "LANDMARK",
                        "description": "same-label cluster member",
                        "context_string": "lake",
                    },
                }
            )
        reranked.append(
            {
                "qid": "Q-context",
                "label": "Lake Victoria",
                "reranked_rank": 39,
                "raw_rank": 7,
                "features": {
                    "candidate_family": "PRIMARY",
                    "final_score": 24.0,
                    "heuristic_score": 24.0,
                    "prior": 0.56,
                    "has_wikipedia": True,
                    "weighted_context_overlap": 0.85,
                    "label_context_support_score": 0.68,
                    "mention_token_coverage": 1.0,
                },
                "source": {
                    "item_category": "ENTITY",
                    "coarse_type": "LOCATION",
                    "fine_type": "LANDMARK",
                    "description": "context-supported competitor",
                    "context_string": "Africa Kenya Uganda lake",
                },
            }
        )

        shortlist = build_semantic_candidates(
            reranked,
            max_candidates=12,
            anchor_label="Victoria Lake",
        )

        self.assertIn("Q-context", {item.qid for item in shortlist})


if __name__ == "__main__":
    unittest.main()
