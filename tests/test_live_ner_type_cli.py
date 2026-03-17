from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

from src.test_live_ner_type import build_live_ner_report, classify_live_qid, main
from src.wikidata_sample_postgres import FetchResult


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


def _sample_company_entity() -> dict[str, object]:
    return {
        "id": "Q312",
        "type": "item",
        "labels": {
            "en": {"language": "en", "value": "Apple"},
            "it": {"language": "it", "value": "Apple"},
        },
        "aliases": {
            "en": [
                {"language": "en", "value": "Apple Inc."},
                {"language": "en", "value": "Apple Computer"},
            ],
        },
        "descriptions": {
            "en": {"language": "en", "value": "American technology company"},
            "it": {"language": "it", "value": "azienda tecnologica statunitense"},
        },
        "claims": {
            "P31": [_statement_for_item_qid(4830453)],
        },
    }


class LiveNerTypeCliTests(unittest.TestCase):
    def test_build_live_ner_report_matches_pipeline_style_fields(self) -> None:
        report = build_live_ner_report(
            _sample_company_entity(),
            qid="Q312",
            source_url="https://www.wikidata.org/wiki/Special:EntityData/Q312.json",
            language_allowlist=("en",),
            max_aliases_per_language=8,
        )

        self.assertEqual(report["qid"], "Q312")
        self.assertEqual(report["item_category"], "ENTITY")
        self.assertEqual(report["coarse_type"], "ORGANIZATION")
        self.assertEqual(report["fine_type"], "COMPANY")
        self.assertIn("ORGANIZATION", report["coarse_types"])
        self.assertIn("COMPANY", report["fine_types"])
        self.assertEqual(report["labels"]["en"], "Apple")
        self.assertIn("Apple Inc.", report["aliases"]["en"])
        self.assertEqual(report["ner_type_source"], "lexical_v1")

    @patch("src.test_live_ner_type.fetch_entity_payload")
    def test_classify_live_qid_raises_clear_error_when_fetch_fails(self, mock_fetch: object) -> None:
        mock_fetch.return_value = FetchResult(
            qid="Q42",
            status="error",
            source_url="https://www.wikidata.org/wiki/Special:EntityData/Q42.json",
            error="HTTP 404: not found",
            payload=None,
            http_status=404,
        )

        with self.assertRaisesRegex(RuntimeError, "Could not fetch live entity Q42"):
            classify_live_qid(
                "Q42",
                base_url="https://www.wikidata.org/wiki/Special:EntityData",
                language_allowlist=("en",),
                max_aliases_per_language=8,
                timeout_seconds=20.0,
                sleep_seconds=0.0,
                http_max_retries=2,
                http_retry_backoff_seconds=1.0,
                http_retry_max_sleep_seconds=2.0,
            )

    @patch("src.test_live_ner_type.fetch_entity_payload")
    def test_main_prints_json_report(self, mock_fetch: object) -> None:
        mock_fetch.return_value = FetchResult(
            qid="Q312",
            status="fetched",
            source_url="https://www.wikidata.org/wiki/Special:EntityData/Q312.json",
            payload=_sample_company_entity(),
        )

        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = main(["--qid", "Q312", "--pretty"])

        self.assertEqual(exit_code, 0)
        self.assertEqual(stderr.getvalue(), "")
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["qid"], "Q312")
        self.assertEqual(payload["coarse_type"], "ORGANIZATION")
        self.assertEqual(payload["fine_type"], "COMPANY")

    def test_main_rejects_multiple_qids(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = main(["--qid", "Q42,Q90"])

        self.assertEqual(exit_code, 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("--qid accepts exactly one QID.", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
