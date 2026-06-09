from __future__ import annotations

import unittest
from datetime import datetime, timezone

from src.index_postgres_to_elasticsearch import _build_index_payload, _row_to_document


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
        self.assertEqual(doc["coarse_type"], "LOCATION")
        self.assertEqual(doc["fine_type"], "CITY")
        self.assertEqual(doc["wikipedia_url"], "it.wikipedia.org|Roma")
        self.assertEqual(doc["dbpedia_url"], "it.dbpedia.org|Roma")
        self.assertEqual(doc["updated_at"], "2026-03-10T08:30:00+00:00")

    def test_row_to_document_recomputes_ner_typing_from_postgres_text(self) -> None:
        row = (
            "Q999",
            "Acme",
            ["Acme"],
            ["Acme Corporation"],
            "software company",
            [],
            "LOCATION",
            "CITY",
            "ENTITY",
            42.0,
            0.9,
            "",
            "",
            datetime(2026, 3, 10, 8, 30, tzinfo=timezone.utc),
        )

        doc = _row_to_document(row)

        assert doc is not None
        self.assertEqual(doc["coarse_type"], "ORGANIZATION")
        self.assertEqual(doc["fine_type"], "COMPANY")

    def test_build_index_payload_indexes_requested_identifier_fields(self) -> None:
        properties = _build_index_payload()["mappings"]["properties"]

        self.assertEqual(properties["qid"], {"type": "keyword", "doc_values": False})
        self.assertEqual(
            properties["description"],
            {"type": "text", "analyzer": "alpaca_text", "norms": False},
        )
        self.assertEqual(
            properties["types"],
            {"type": "keyword", "index": False, "doc_values": False},
        )
        self.assertEqual(
            properties["wikipedia_url"],
            {"type": "keyword", "doc_values": False},
        )
        self.assertEqual(
            properties["dbpedia_url"],
            {"type": "keyword", "doc_values": False},
        )
        self.assertNotIn("search_text", properties)


if __name__ == "__main__":
    unittest.main()
