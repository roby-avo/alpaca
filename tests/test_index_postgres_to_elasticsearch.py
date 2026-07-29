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

    def test_row_to_document_keeps_stored_type_without_entity_ner_join(self) -> None:
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
        self.assertEqual(doc["coarse_type"], "LOCATION")
        self.assertEqual(doc["fine_type"], "CITY")

    def test_row_to_document_prefers_wikidata_ner_classifier_fields(self) -> None:
        row = (
            "Q3441181",
            "Rome Against Rome",
            [],
            [],
            "1964 sword-and-sandal film",
            ["Q11424"],
            "",
            "",
            "ENTITY",
            15.0,
            0.5,
            "",
            "",
            datetime(2026, 7, 27, tzinfo=timezone.utc),
            "CREATIVE_WORK",
            "FILM",
            None,
            "SWORD_AND_SANDAL_FILM",
            ["SWORD_AND_SANDAL_FILM"],
            [],
            {"genre": ["SWORD_AND_SANDAL"]},
            4,
            ["CREATIVE_WORK", "FILM", "SWORD_AND_SANDAL_FILM"],
            "CREATIVE_WORK/FILM/SWORD_AND_SANDAL_FILM",
            ["CREATIVE_WORK", "FILM", "GENRE:SWORD_AND_SANDAL"],
            0.77,
            0.78,
            "wikidata-ner-classifier",
            "0.5.1",
            "1.0.0/0.3.1",
        )

        doc = _row_to_document(row)

        assert doc is not None
        self.assertEqual(doc["coarse_type"], "CREATIVE_WORK")
        self.assertEqual(doc["fine_type"], "FILM")
        self.assertEqual(doc["ner_specific_type"], "SWORD_AND_SANDAL_FILM")
        self.assertEqual(doc["ner_facets"], {"genre": ["SWORD_AND_SANDAL"]})
        self.assertEqual(doc["ner_specificity_level"], 4)
        self.assertEqual(
            doc["ner_retrieval_path"],
            ["CREATIVE_WORK", "FILM", "SWORD_AND_SANDAL_FILM"],
        )
        self.assertEqual(doc["ner_classifier_name"], "wikidata-ner-classifier")
        self.assertEqual(doc["ner_classifier_version"], "0.5.1")
        self.assertEqual(doc["ner_taxonomy_version"], "1.0.0/0.3.1")

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
