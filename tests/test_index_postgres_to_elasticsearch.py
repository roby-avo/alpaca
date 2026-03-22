from __future__ import annotations

import unittest
from datetime import datetime, timezone

from src.index_postgres_to_elasticsearch import (
    _build_index_payload,
    _extract_explain_plan_rows,
    _prefer_estimated_row_total,
    _row_to_document,
)


class IndexPostgresToElasticsearchTests(unittest.TestCase):
    def test_row_to_document_trims_duplicate_primary_names_and_caps_secondary_names(self) -> None:
        row = (
            "Q220",
            "Rome",
            ["Rome", "Roma", "Roma", "Roma capitale"],
            ["Rome", "Rome city", "Roma", "capital of Lazio"],
            "capital of Italy",
            ["Q515"],
            "capital city; Italy",
            "LOCATION",
            "CITY",
            "ENTITY",
            42.0,
            0.9,
            "it.wikipedia.org|Roma",
            "it.dbpedia.org|Roma",
            datetime(2026, 3, 10, 8, 30, tzinfo=timezone.utc),
        )

        doc = _row_to_document(
            row,
            max_indexed_labels=1,
            max_indexed_aliases=1,
        )

        assert doc is not None
        self.assertEqual(doc["qid"], "Q220")
        self.assertEqual(doc["label"], "Rome")
        self.assertEqual(doc["labels"], ["Roma"])
        self.assertEqual(doc["aliases"], ["Rome city"])
        self.assertEqual(doc["description"], "capital of Italy")
        self.assertEqual(doc["types"], ["Q515"])
        self.assertEqual(doc["wikipedia_url"], "it.wikipedia.org|Roma")
        self.assertEqual(doc["dbpedia_url"], "it.dbpedia.org|Roma")
        self.assertEqual(doc["updated_at"], "2026-03-10T08:30:00+00:00")

    def test_build_index_payload_keeps_description_and_types_stored_only_but_indexes_ids_and_cross_refs(self) -> None:
        properties = _build_index_payload()["mappings"]["properties"]

        self.assertEqual(
            properties["qid"],
            {"type": "keyword", "normalizer": "alpaca_keyword_lower", "ignore_above": 64},
        )
        self.assertEqual(properties["description"], {"type": "text", "index": False})
        self.assertEqual(
            properties["types"],
            {"type": "keyword", "index": False, "doc_values": False},
        )
        self.assertEqual(
            properties["wikipedia_url"],
            {"type": "keyword", "normalizer": "alpaca_keyword_lower", "ignore_above": 2048},
        )
        self.assertEqual(
            properties["dbpedia_url"],
            {"type": "keyword", "normalizer": "alpaca_keyword_lower", "ignore_above": 2048},
        )
        self.assertNotIn("search_text", properties)

    def test_prefer_estimated_row_total_uses_largest_positive_candidate(self) -> None:
        self.assertEqual(_prefer_estimated_row_total(-1, 0, 123.4, 120), 123)
        self.assertIsNone(_prefer_estimated_row_total(None, 0, -5))

    def test_extract_explain_plan_rows_parses_json_plan(self) -> None:
        self.assertEqual(
            _extract_explain_plan_rows([{"Plan": {"Node Type": "Seq Scan", "Plan Rows": 98765}}]),
            98765,
        )
        self.assertIsNone(_extract_explain_plan_rows([{"Plan": {}}]))


if __name__ == "__main__":
    unittest.main()
