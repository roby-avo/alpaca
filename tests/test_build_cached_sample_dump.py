from __future__ import annotations

import bz2
import json
import tempfile
import unittest
from pathlib import Path

from src.build_cached_sample_dump import build_dump, extract_entity_payload, resolve_ids


class BuildCachedSampleDumpTests(unittest.TestCase):
    def test_extract_entity_payload_direct_key(self) -> None:
        raw = {"entities": {"Q42": {"id": "Q42", "labels": {"en": {"value": "Douglas Adams"}}}}}
        payload = extract_entity_payload(raw, "Q42")
        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertEqual(payload.get("id"), "Q42")

    def test_extract_entity_payload_fallback_on_canonical_id(self) -> None:
        raw = {"entities": {"Qx": {"id": "Q42", "labels": {"en": {"value": "Douglas Adams"}}}}}
        payload = extract_entity_payload(raw, "Q42")
        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertEqual(payload.get("id"), "Q42")

    def test_build_dump_writes_only_cached_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            cache_dir = root / "cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            output_path = root / "sample.json.bz2"

            raw_q42 = {"entities": {"Q42": {"id": "Q42", "labels": {"en": {"value": "Douglas Adams"}}}}}
            (cache_dir / "Q42.json").write_text(json.dumps(raw_q42), encoding="utf-8")

            written, missing = build_dump(cache_dir, output_path, ["Q42", "Q90"])
            self.assertEqual(written, 1)
            self.assertEqual(missing, 1)

            with bz2.open(output_path, mode="rt", encoding="utf-8") as handle:
                payload = json.loads(handle.read())

            self.assertEqual(len(payload), 1)
            self.assertEqual(payload[0]["id"], "Q42")

    def test_resolve_ids_prefers_explicit_ids(self) -> None:
        ids = resolve_ids(10, "Q42,Q90")
        self.assertEqual(ids, ["Q42", "Q90"])

    def test_resolve_ids_uses_default_range(self) -> None:
        ids = resolve_ids(3, None)
        self.assertEqual(ids, ["Q1", "Q2", "Q3"])


if __name__ == "__main__":
    unittest.main()
