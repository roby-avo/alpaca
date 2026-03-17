from __future__ import annotations

import unittest
from datetime import datetime, timezone

from src.index_postgres_to_elasticsearch import _row_to_document


class IndexPostgresToElasticsearchTests(unittest.TestCase):
    def test_row_to_document_reads_labels_and_aliases_arrays(self) -> None:
        row = (
            "Q220",
            "Rome",
            ["Rome", "Roma"],
            ["Rome city"],
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

        doc = _row_to_document(row)

        assert doc is not None
        self.assertEqual(doc["qid"], "Q220")
        self.assertEqual(doc["label"], "Rome")
        self.assertEqual(doc["labels"], ["Rome", "Roma"])
        self.assertEqual(doc["aliases"], ["Rome city"])
        self.assertEqual(doc["description"], "capital of Italy")
        self.assertEqual(doc["types"], ["Q515"])
        self.assertEqual(doc["wikipedia_url"], "it.wikipedia.org|Roma")
        self.assertEqual(doc["dbpedia_url"], "it.dbpedia.org|Roma")
        self.assertEqual(doc["updated_at"], "2026-03-10T08:30:00+00:00")


if __name__ == "__main__":
    unittest.main()
