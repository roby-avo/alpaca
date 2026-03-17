from __future__ import annotations

import io
import unittest
from unittest.mock import patch
from urllib.error import URLError

from src.wikidata_stats import (
    WIKIDATA_STATS_PAGE_URL,
    WikidataStatsSnapshot,
    fetch_wikidata_stats,
    resolve_expected_entity_total,
)


class WikidataStatsTests(unittest.TestCase):
    def test_fetch_wikidata_stats_parses_rendered_statistics_html(self) -> None:
        payload = (
            '{"parse":{"text":{"*":"<div>'
            'Wikidata currently contains 120,830,824 items. '
            '2,472,288,826 edits have been made since the project launch.'
            '</div>"}}}'
        ).encode("utf-8")

        with patch("src.wikidata_stats._fetch_url", return_value=payload):
            snapshot = fetch_wikidata_stats(timeout_seconds=1.0, user_agent="alpaca-tests")

        self.assertEqual(snapshot.items, 120_830_824)
        self.assertEqual(snapshot.edits, 2_472_288_826)
        self.assertEqual(snapshot.source_page_url, WIKIDATA_STATS_PAGE_URL)
        self.assertIn("action=parse", snapshot.source_api_url)

    def test_resolve_expected_entity_total_fetches_live_total_and_logs_it(self) -> None:
        snapshot = WikidataStatsSnapshot(
            items=120_830_824,
            edits=2_472_288_826,
            fetched_at_utc="2026-03-17T10:29:56+00:00",
            source_page_url=WIKIDATA_STATS_PAGE_URL,
            source_api_url="https://www.wikidata.org/w/api.php?action=parse",
        )
        stream = io.StringIO()

        with patch("src.wikidata_stats.fetch_wikidata_stats", return_value=snapshot):
            total = resolve_expected_entity_total(
                manual_total=None,
                fetch_live=True,
                timeout_seconds=5.0,
                log_stream=stream,
            )

        self.assertEqual(total, 120_830_824)
        self.assertIn("120,830,824", stream.getvalue())
        self.assertIn(WIKIDATA_STATS_PAGE_URL, stream.getvalue())

    def test_resolve_expected_entity_total_rejects_manual_and_live_combination(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "Use either --expected-entity-total or --fetch-live-entity-total, not both.",
        ):
            resolve_expected_entity_total(
                manual_total=120_830_824,
                fetch_live=True,
            )

    def test_resolve_expected_entity_total_wraps_network_errors(self) -> None:
        with patch(
            "src.wikidata_stats.fetch_wikidata_stats",
            side_effect=URLError("offline"),
        ):
            with self.assertRaisesRegex(
                ValueError,
                "Could not fetch live Wikidata item total:.*offline",
            ):
                resolve_expected_entity_total(
                    manual_total=None,
                    fetch_live=True,
                )


if __name__ == "__main__":
    unittest.main()
