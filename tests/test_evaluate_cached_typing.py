from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.evaluate_cached_typing import (
    collect_ids_to_fetch,
    compute_age_hours,
    extract_entity_payload,
    resolve_ids,
)


class EvaluateCachedTypingTests(unittest.TestCase):
    def test_extract_entity_payload(self) -> None:
        raw = {
            "entities": {
                "Q42": {
                    "id": "Q42",
                    "labels": {"en": {"value": "Douglas Adams"}},
                }
            }
        }
        payload = extract_entity_payload(raw, "Q42")
        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertEqual(payload.get("id"), "Q42")

    def test_compute_age_hours_handles_missing(self) -> None:
        self.assertIsNone(compute_age_hours(None))

    def test_resolve_ids_from_argument(self) -> None:
        ids = resolve_ids(Path("."), "Q42,Q90")
        self.assertEqual(ids, ["Q42", "Q90"])

    def test_resolve_ids_ignores_meta_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_dir = Path(tmp_dir)
            (cache_dir / "Q42.json").write_text("{}", encoding="utf-8")
            (cache_dir / "Q42.meta.json").write_text("{}", encoding="utf-8")
            ids = resolve_ids(cache_dir, None)
            self.assertEqual(ids, ["Q42"])

    def test_collect_ids_to_fetch_missing_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_dir = Path(tmp_dir)
            (cache_dir / "Q42.json").write_text("{}", encoding="utf-8")
            ids = collect_ids_to_fetch(["Q42", "Q90"], cache_dir, force_refresh=False)
            self.assertEqual(ids, ["Q90"])

    def test_collect_ids_to_fetch_force_refresh(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_dir = Path(tmp_dir)
            (cache_dir / "Q42.json").write_text("{}", encoding="utf-8")
            ids = collect_ids_to_fetch(["Q42", "Q90"], cache_dir, force_refresh=True)
            self.assertEqual(ids, ["Q42", "Q90"])


if __name__ == "__main__":
    unittest.main()
