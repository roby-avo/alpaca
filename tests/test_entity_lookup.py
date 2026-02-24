from __future__ import annotations

import unittest

from src.entity_lookup import build_cache_key, normalize_context_inputs, rerank_candidates


class EntityLookupTests(unittest.TestCase):
    def test_normalize_context_inputs_dedupes_tokens(self) -> None:
        terms = normalize_context_inputs(["Cupertino, CA", "technology cupertino"])
        self.assertEqual(terms, ["cupertino", "ca", "technology"])

    def test_cache_key_changes_when_shape_changes(self) -> None:
        key_a = build_cache_key(
            mention_norm="apple",
            context_terms=["cupertino"],
            coarse_hints=["ORGANIZATION"],
            fine_hints=["COMPANY"],
            limit=20,
            include_top_k=False,
        )
        key_b = build_cache_key(
            mention_norm="apple",
            context_terms=["cupertino"],
            coarse_hints=["ORGANIZATION"],
            fine_hints=["COMPANY"],
            limit=20,
            include_top_k=True,
        )
        self.assertNotEqual(key_a, key_b)

    def test_rerank_candidates_uses_context_and_type(self) -> None:
        candidates = [
            {
                "qid": "Q1",
                "label": "Apple",
                "aliases": ["Apple Inc."],
                "context_string": "technology company cupertino california",
                "coarse_type": "ORGANIZATION",
                "fine_type": "COMPANY",
                "popularity": 1000.0,
                "prior": 0.9,
                "score": 1.0,
            },
            {
                "qid": "Q2",
                "label": "Apple",
                "aliases": ["apple fruit"],
                "context_string": "fruit tree food agriculture",
                "coarse_type": "PRODUCT",
                "fine_type": "FOOD",
                "popularity": 50.0,
                "prior": 0.2,
                "score": 1.1,
            },
        ]

        ranked = rerank_candidates(
            candidates,
            mention_norm="apple",
            context_terms=["technology", "cupertino"],
            coarse_hints=["ORGANIZATION"],
            fine_hints=["COMPANY"],
            exact_mode=True,
            limit=2,
        )
        self.assertEqual(ranked[0]["qid"], "Q1")
        self.assertGreater(ranked[0]["context_score"], ranked[1]["context_score"])
        self.assertGreater(ranked[0]["type_score"], ranked[1]["type_score"])


if __name__ == "__main__":
    unittest.main()
