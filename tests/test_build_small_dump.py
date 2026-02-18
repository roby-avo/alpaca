from __future__ import annotations

import bz2
import json
import tempfile
import unittest
from pathlib import Path

from src.build_small_dump import _build_from_count, _build_from_ids, parse_entity_id_list


def write_source_dump(path: Path) -> None:
    entities = [
        {"id": "L1", "labels": {"en": {"value": "lexeme"}}},
        {"id": "Q1", "labels": {"en": {"value": "Universe"}}},
        {"id": "P31", "labels": {"en": {"value": "instance of"}}},
        {"id": "Q42", "labels": {"en": {"value": "Douglas Adams"}}},
    ]

    with bz2.open(path, mode="wt", encoding="utf-8") as handle:
        handle.write("[\n")
        for index, entity in enumerate(entities):
            suffix = "," if index < len(entities) - 1 else ""
            handle.write(f"{json.dumps(entity, ensure_ascii=False)}{suffix}\n")
        handle.write("]\n")


class BuildSmallDumpTests(unittest.TestCase):
    def test_parse_entity_id_list_accepts_q_and_p(self) -> None:
        self.assertEqual(parse_entity_id_list("Q42,P31,Q42"), ["Q42", "P31"])

    def test_parse_entity_id_list_rejects_invalid(self) -> None:
        with self.assertRaises(ValueError):
            parse_entity_id_list("Q42,L1")

    def test_build_from_count_streams_supported_entities(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = root / "source.json.bz2"
            output = root / "small.json.bz2"
            write_source_dump(source)

            written, scanned = _build_from_count(source, output, count=2)
            self.assertEqual(written, 2)
            self.assertGreaterEqual(scanned, 3)

            with bz2.open(output, mode="rt", encoding="utf-8") as handle:
                payload = json.loads(handle.read())

            self.assertEqual([item["id"] for item in payload], ["Q1", "P31"])

    def test_build_from_ids_preserves_requested_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = root / "source.json.bz2"
            output = root / "small_ids.json.bz2"
            write_source_dump(source)

            written, missing, scanned = _build_from_ids(source, output, ["Q42", "P31", "Q999999"])
            self.assertEqual(written, 2)
            self.assertEqual(missing, 1)
            self.assertGreater(scanned, 0)

            with bz2.open(output, mode="rt", encoding="utf-8") as handle:
                payload = json.loads(handle.read())

            self.assertEqual([item["id"] for item in payload], ["Q42", "P31"])


if __name__ == "__main__":
    unittest.main()
