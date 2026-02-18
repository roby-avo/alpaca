from __future__ import annotations

import unittest

from src.search_logic import build_quickwit_query, normalize_type_labels, rerank_hits_by_context


class SearchLogicTests(unittest.TestCase):
    def test_build_quickwit_query_with_coarse_and_fine_type_filters(self) -> None:
        query = build_quickwit_query(
            "Leonardo da Vinci",
            ["PERSON"],
            ["ARTIST", "PAINTER"],
        )
        self.assertEqual(
            query,
            "(leonardo da vinci) AND coarse_type:PERSON AND "
            "(fine_type:ARTIST OR fine_type:PAINTER)",
        )

    def test_build_quickwit_query_without_terms_fails(self) -> None:
        with self.assertRaises(ValueError):
            build_quickwit_query("!!!", None, None)

    def test_normalize_type_labels_rejects_invalid_values(self) -> None:
        with self.assertRaises(ValueError):
            normalize_type_labels(["PERSON", "ORG)"], field_name="coarse_type")

    def test_context_rerank_promotes_overlap(self) -> None:
        raw_hits = [
            {
                "id": "Q2",
                "name_text": "apple",
                "bow": "fruit company california",
                "coarse_type": "ORG",
                "fine_type": "FOOD_BRAND",
                "_score": 1.2,
            },
            {
                "id": "Q1",
                "name_text": "apple inc",
                "bow": "technology company cupertino california",
                "coarse_type": "ORG",
                "fine_type": "COMPANY",
                "_score": 1.0,
            },
        ]

        ranked = rerank_hits_by_context(raw_hits, "cupertino technology", limit=2)
        self.assertEqual(ranked[0]["id"], "Q1")
        self.assertGreater(ranked[0]["context_overlap"], ranked[1]["context_overlap"])


if __name__ == "__main__":
    unittest.main()
