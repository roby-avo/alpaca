from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

from src.test_qid_bow import build_cached_qid_bow_record, main


def _statement_for_item_qid(numeric_id: int) -> dict[str, object]:
    return {
        "mainsnak": {
            "snaktype": "value",
            "datavalue": {
                "value": {
                    "entity-type": "item",
                    "numeric-id": int(numeric_id),
                }
            },
        }
    }


def _sample_entity() -> dict[str, object]:
    return {
        "id": "Q42",
        "type": "item",
        "labels": {
            "en": {"language": "en", "value": "Douglas Adams"},
        },
        "aliases": {
            "en": [
                {"language": "en", "value": "Douglas Adams"},
                {"language": "en", "value": "Doug Adams"},
            ],
        },
        "descriptions": {
            "en": {"language": "en", "value": "English writer and humorist"},
        },
        "claims": {
            "P31": [_statement_for_item_qid(5)],
            "P27": [_statement_for_item_qid(145)],
        },
    }


class CachedQidBowCliTests(unittest.TestCase):
    def test_build_cached_qid_bow_record_keeps_aliases_disjoint_from_labels(self) -> None:
        record = build_cached_qid_bow_record(
            _sample_entity(),
            qid="Q42",
            context_label_map={"Q5": "human", "Q145": "United Kingdom"},
            language_allowlist=("en",),
            max_aliases_per_language=8,
            max_bow_tokens=128,
            max_context_object_ids=32,
            max_context_chars=640,
        )

        self.assertEqual(record["aliases"]["en"], ["Doug Adams"])
        self.assertIn("human", record["context"].casefold())
        self.assertIn("united kingdom", record["context"].casefold())

    @patch("src.test_qid_bow.resolve_postgres_dsn", return_value="postgresql://example")
    @patch("src.test_qid_bow.PostgresStore")
    def test_main_prints_jsonl_in_requested_order(self, mock_store_cls: object, _mock_dsn: object) -> None:
        mock_store = mock_store_cls.return_value
        mock_store.get_sample_entities.return_value = {
            "Q42": _sample_entity(),
            "Q1": {
                "id": "Q1",
                "type": "item",
                "labels": {"en": {"language": "en", "value": "Universe"}},
                "descriptions": {"en": {"language": "en", "value": "totality of space and time"}},
                "claims": {},
            },
        }
        mock_store.resolve_sample_cache_labels.side_effect = [
            {"Q5": "human", "Q145": "United Kingdom"},
            {},
        ]

        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = main(["--ids", "Q42,Q1"])

        self.assertEqual(exit_code, 0)
        self.assertEqual(stderr.getvalue(), "")
        lines = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]
        self.assertEqual([line["id"] for line in lines], ["Q42", "Q1"])

    @patch("src.test_qid_bow.resolve_postgres_dsn", return_value="postgresql://example")
    @patch("src.test_qid_bow.PostgresStore")
    def test_main_fails_when_qid_missing_from_cache(self, mock_store_cls: object, _mock_dsn: object) -> None:
        mock_store = mock_store_cls.return_value
        mock_store.get_sample_entities.return_value = {"Q42": _sample_entity()}

        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = main(["--ids", "Q42,Q1"])

        self.assertEqual(exit_code, 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("Missing 1 requested QIDs", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
