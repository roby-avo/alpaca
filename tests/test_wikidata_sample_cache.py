from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path

from src.wikidata_sample_cache import (
    compute_age_hours,
    default_entity_ids,
    fetch_all_entities,
    parse_id_list,
    parse_positive_int,
    should_refresh_cache,
    write_run_report,
    FetchResult,
)


class WikidataSampleCacheTests(unittest.TestCase):
    def test_default_entity_ids_count_and_paris_included(self) -> None:
        ids = default_entity_ids(100)
        self.assertEqual(len(ids), 100)
        self.assertEqual(ids[0], "Q1")
        self.assertEqual(ids[89], "Q90")

    def test_parse_id_list_deduplicates(self) -> None:
        ids = parse_id_list("Q42, Q90, Q42")
        self.assertEqual(ids, ["Q42", "Q90"])

    def test_parse_positive_int(self) -> None:
        self.assertEqual(parse_positive_int("3"), 3)
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_positive_int("0")

    def test_compute_age_hours(self) -> None:
        now = datetime(2026, 2, 9, 12, 0, tzinfo=timezone.utc)
        age = compute_age_hours("2026-02-09T10:30:00Z", now)
        self.assertAlmostEqual(age, 1.5)

    def test_should_refresh_cache_based_on_age(self) -> None:
        now = datetime(2026, 2, 9, 12, 0, tzinfo=timezone.utc)
        fresh = {"retrieved_at_utc": "2026-02-09T11:30:00Z"}
        stale = {"retrieved_at_utc": "2026-02-08T11:30:00Z"}
        self.assertFalse(should_refresh_cache(fresh, max_age_hours=2.0, now_utc=now))
        self.assertTrue(should_refresh_cache(stale, max_age_hours=2.0, now_utc=now))

    def test_write_run_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_dir = Path(tmp_dir)
            results = [
                FetchResult(
                    entity_id="Q42",
                    status="fetched",
                    retrieved_at_utc="2026-02-09T10:00:00Z",
                    age_hours=0.0,
                    source_url="https://example/Q42.json",
                    cache_json_path=cache_dir / "Q42.json",
                    cache_meta_path=cache_dir / "Q42.meta.json",
                    http_status=200,
                ),
                FetchResult(
                    entity_id="Q90",
                    status="cache_hit",
                    retrieved_at_utc="2026-02-09T09:00:00Z",
                    age_hours=1.0,
                    source_url="https://example/Q90.json",
                    cache_json_path=cache_dir / "Q90.json",
                    cache_meta_path=cache_dir / "Q90.meta.json",
                    http_status=200,
                ),
            ]

            report_path = write_run_report(cache_dir, results)
            self.assertTrue(report_path.is_file())

            payload = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["total"], 2)
            self.assertEqual(payload["fetched"], 1)
            self.assertEqual(payload["cache_hits"], 1)
            self.assertEqual(payload["errors"], 0)

    def test_fetch_all_entities_parallel_preserves_input_order(self) -> None:
        def fake_fetch(
            entity_id: str,
            *,
            base_url: str,
            cache_dir: Path,
            timeout_seconds: float,
            max_age_hours: float,
            force_refresh: bool,
            now_utc: datetime,
            sleep_seconds: float,
        ) -> FetchResult:
            _ = (
                base_url,
                cache_dir,
                timeout_seconds,
                max_age_hours,
                force_refresh,
                now_utc,
                sleep_seconds,
            )
            if entity_id == "Q1":
                time.sleep(0.03)
            elif entity_id == "Q2":
                time.sleep(0.01)
            return FetchResult(
                entity_id=entity_id,
                status="cache_hit",
                retrieved_at_utc="2026-02-09T12:00:00Z",
                age_hours=0.0,
                source_url=f"https://example/{entity_id}.json",
                cache_json_path=Path(f"/tmp/{entity_id}.json"),
                cache_meta_path=Path(f"/tmp/{entity_id}.meta.json"),
                http_status=200,
            )

        results = asyncio.run(
            fetch_all_entities(
                entity_ids=["Q1", "Q2", "Q3"],
                base_url="https://example",
                cache_dir=Path("/tmp"),
                timeout_seconds=10.0,
                max_age_hours=24.0,
                force_refresh=False,
                now_utc=datetime(2026, 2, 9, 12, 0, tzinfo=timezone.utc),
                sleep_seconds=0.0,
                concurrency=3,
                fetch_fn=fake_fetch,
                show_progress=False,
            )
        )

        self.assertEqual([item.entity_id for item in results], ["Q1", "Q2", "Q3"])


if __name__ == "__main__":
    unittest.main()
