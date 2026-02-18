from __future__ import annotations

import unittest
from unittest import mock

from src.build_quickwit_index import _extract_index_ids, _ingest_chunk_with_retry, wait_for_index
from src.quickwit_client import QuickwitClientError


class BuildQuickwitIndexTests(unittest.TestCase):
    def test_extract_index_ids_accepts_index_id_and_id(self) -> None:
        payload = [
            {"index_id": "wikidata_entities"},
            {"id": "legacy_id"},
            {"index_config": {"index_id": "nested_index"}},
            {"index_uid": "uid_index:01ABCDEFG"},
            {"index_id": ""},
            {"other": "ignored"},
        ]
        self.assertEqual(
            _extract_index_ids(payload),
            {"wikidata_entities", "legacy_id", "nested_index", "uid_index"},
        )

    def test_wait_for_index_succeeds_after_retry(self) -> None:
        client = mock.Mock()
        client.list_indexes = mock.Mock(
            side_effect=[
                [],
                [{"index_id": "wikidata_entities"}],
            ]
        )

        with mock.patch("src.build_quickwit_index.time.sleep", return_value=None):
            wait_for_index(
                client=client,
                index_id="wikidata_entities",
                timeout_seconds=5.0,
                poll_interval_seconds=0.0,
            )

        self.assertEqual(client.list_indexes.call_count, 2)

    def test_wait_for_index_times_out(self) -> None:
        client = mock.Mock()
        client.list_indexes = mock.Mock(return_value=[])

        with self.assertRaises(QuickwitClientError):
            wait_for_index(
                client=client,
                index_id="missing_index",
                timeout_seconds=0.0,
                poll_interval_seconds=0.0,
            )

    def test_ingest_chunk_retries_transient_404(self) -> None:
        client = mock.Mock()
        client.ingest_ndjson = mock.Mock(
            side_effect=[
                QuickwitClientError("not found", status_code=404),
                {"ok": True},
            ]
        )

        with mock.patch("src.build_quickwit_index.time.sleep", return_value=None):
            _ingest_chunk_with_retry(
                client=client,
                index_id="wikidata_entities",
                ndjson_payload='{"id":"Q1"}\n',
                commit_mode="auto",
                retry_timeout_seconds=5.0,
                poll_interval_seconds=0.0,
            )

        self.assertEqual(client.ingest_ndjson.call_count, 2)

    def test_ingest_chunk_raises_after_retry_timeout(self) -> None:
        client = mock.Mock()
        client.ingest_ndjson = mock.Mock(
            side_effect=QuickwitClientError("not found", status_code=404)
        )

        with self.assertRaises(QuickwitClientError):
            _ingest_chunk_with_retry(
                client=client,
                index_id="wikidata_entities",
                ndjson_payload='{"id":"Q1"}\n',
                commit_mode="auto",
                retry_timeout_seconds=0.0,
                poll_interval_seconds=0.0,
            )


if __name__ == "__main__":
    unittest.main()
