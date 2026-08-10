from __future__ import annotations

import json
import unittest

from src.classify_postgres_entities import REQUIRED_CLASSIFIER_VERSION
from src.update_elasticsearch_ner import _bulk_update_payload, _row_to_update


class UpdateElasticsearchNERTests(unittest.TestCase):
    def test_row_to_update_contains_only_current_ner_fields(self) -> None:
        qid, doc = _row_to_update(
            (
                "Q7259",
                "PERSON",
                "HUMAN",
                None,
                "ENGINEER",
                ["ENGINEER", "SCIENTIST", "WRITER"],
                [],
                {"occupation": ["ENGINEER", "SCIENTIST", "WRITER"]},
                4,
                ["PERSON", "HUMAN", "ENGINEER"],
                "PERSON/HUMAN/ENGINEER",
                ["PERSON", "HUMAN", "ENGINEER"],
                0.9678,
                0.99,
                "wikidata-ner-classifier",
                REQUIRED_CLASSIFIER_VERSION,
                "1.0.0/0.4.0",
            )
        )

        self.assertEqual(qid, "Q7259")
        self.assertEqual(doc["coarse_type"], "PERSON")
        self.assertEqual(doc["fine_type"], "HUMAN")
        self.assertEqual(doc["ner_specific_type"], "ENGINEER")
        self.assertEqual(
            doc["ner_specific_types"],
            ["ENGINEER", "SCIENTIST", "WRITER"],
        )
        self.assertNotIn("label", doc)
        self.assertNotIn("aliases", doc)
        self.assertNotIn("description", doc)

    def test_bulk_payload_uses_update_instead_of_reindex(self) -> None:
        payload = _bulk_update_payload(
            "alpaca-wikidata",
            [("Q7259", {"ner_specific_type": "ENGINEER"})],
        ).decode("utf-8")
        lines = payload.strip().splitlines()

        action = json.loads(lines[0])
        body = json.loads(lines[1])
        self.assertEqual(action["update"]["_id"], "Q7259")
        self.assertEqual(body, {"doc": {"ner_specific_type": "ENGINEER"}})

    def test_refuses_stale_classifier_rows(self) -> None:
        stale = (
            "Q7259",
            "PERSON",
            "HUMAN",
            None,
            "HUMAN",
            ["HUMAN"],
            [],
            {},
            2,
            ["PERSON", "HUMAN"],
            "PERSON/HUMAN",
            ["PERSON", "HUMAN"],
            0.9,
            0.9,
            "wikidata-ner-classifier",
            "0.5.1",
            "1.0.0/0.3.1",
        )

        with self.assertRaises(RuntimeError):
            _row_to_update(stale)


if __name__ == "__main__":
    unittest.main()
