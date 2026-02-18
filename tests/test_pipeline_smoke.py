from __future__ import annotations

import bz2
import gzip
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from src.build_bow_docs import run as run_bow_docs
from src.build_labels_db import run as run_labels_db


def write_tiny_dump(path: Path) -> None:
    entities = [
        {
            "id": "Q1",
            "labels": {
                "en": {"language": "en", "value": "Universe"},
                "it": {"language": "it", "value": "Universo"},
            },
            "aliases": {
                "en": [{"language": "en", "value": "Cosmos"}],
                "it": [{"language": "it", "value": "Cosmo"}],
            },
            "descriptions": {
                "en": {"language": "en", "value": "totality of space and time"},
                "it": {"language": "it", "value": "totalita di spazio e tempo"},
            },
        },
        {
            "id": "P31",
            "labels": {
                "en": {"language": "en", "value": "instance of"},
            },
            "aliases": {
                "en": [{"language": "en", "value": "is a"}],
            },
            "descriptions": {
                "en": {
                    "language": "en",
                    "value": "that class of which this subject is a particular example and member",
                },
            },
        },
        {
            "id": "L99",
            "labels": {
                "en": {"language": "en", "value": "non-target lexeme"},
            },
            "descriptions": {
                "en": {"language": "en", "value": "should be ignored"},
            },
        },
    ]

    with bz2.open(path, mode="wt", encoding="utf-8") as handle:
        handle.write("[\n")
        for index, entity in enumerate(entities):
            line = json.dumps(entity, ensure_ascii=False)
            suffix = "," if index < len(entities) - 1 else ""
            handle.write(f"{line}{suffix}\n")
        handle.write("]\n")


def write_tiny_ner_types(path: Path) -> None:
    records = [
        {
            "id": "Q1",
            "coarse_types": ["THING"],
            "fine_types": ["COSMOLOGICAL_ENTITY"],
        },
        {
            "id": "P31",
            "coarse_types": ["RELATION"],
            "fine_types": ["PROPERTY"],
        },
    ]
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


class PipelineSmokeTests(unittest.TestCase):
    def test_build_labels_db_from_tiny_dump(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dump_path = root / "tiny.json.bz2"
            db_path = root / "labels.sqlite"
            write_tiny_dump(dump_path)

            exit_code = run_labels_db(
                dump_path=dump_path,
                db_path=db_path,
                batch_size=2,
                limit=0,
                disable_ner_classifier=False,
            )
            self.assertEqual(exit_code, 0)

            conn = sqlite3.connect(db_path)
            try:
                row_count = conn.execute("SELECT COUNT(*) FROM labels").fetchone()
                assert row_count is not None
                self.assertEqual(row_count[0], 2)

                ids = conn.execute("SELECT id FROM labels ORDER BY id").fetchall()
                self.assertEqual([row[0] for row in ids], ["P31", "Q1"])

                p31_payload_raw = conn.execute(
                    "SELECT labels_json FROM labels WHERE id = ?",
                    ("P31",),
                ).fetchone()
                assert p31_payload_raw is not None
                p31_payload = json.loads(p31_payload_raw[0])
                self.assertEqual(p31_payload["coarse_type"], "RELATION")
                self.assertEqual(p31_payload["fine_type"], "PROPERTY")
                self.assertNotIn("ner_coarse_types", p31_payload)
                self.assertNotIn("ner_fine_types", p31_payload)

                q1_payload_raw = conn.execute(
                    "SELECT labels_json FROM labels WHERE id = ?",
                    ("Q1",),
                ).fetchone()
                assert q1_payload_raw is not None
                q1_payload = json.loads(q1_payload_raw[0])
                self.assertIn("coarse_type", q1_payload)
                self.assertIn("fine_type", q1_payload)
                self.assertNotIn("ner_coarse_types", q1_payload)
                self.assertNotIn("ner_fine_types", q1_payload)
                self.assertTrue(len(q1_payload["coarse_type"]) >= 1)
                self.assertTrue(len(q1_payload["fine_type"]) >= 1)
            finally:
                conn.close()

    def test_build_bow_docs_from_tiny_dump_with_ner_types(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dump_path = root / "tiny.json.bz2"
            db_path = root / "labels.sqlite"
            output_path = root / "bow_docs.jsonl.gz"
            ner_types_path = root / "ner_types.jsonl"
            write_tiny_dump(dump_path)
            write_tiny_ner_types(ner_types_path)

            labels_exit = run_labels_db(
                dump_path=dump_path,
                db_path=db_path,
                batch_size=2,
                limit=0,
                disable_ner_classifier=False,
            )
            self.assertEqual(labels_exit, 0)

            bow_exit = run_bow_docs(
                dump_path=dump_path,
                labels_db_path=db_path,
                output_path=output_path,
                batch_size=2,
                limit=0,
                ner_types_path=ner_types_path,
            )
            self.assertEqual(bow_exit, 0)

            with gzip.open(output_path, mode="rt", encoding="utf-8") as handle:
                docs = [json.loads(line) for line in handle if line.strip()]

            self.assertEqual(len(docs), 2)
            docs_by_id = {doc["id"]: doc for doc in docs}
            self.assertEqual(set(docs_by_id), {"Q1", "P31"})

            self.assertIn("Universe", docs_by_id["Q1"]["name_text"])
            self.assertIn("Cosmos", docs_by_id["Q1"]["name_text"])
            self.assertIn("space", docs_by_id["Q1"]["bow"])
            self.assertIn("time", docs_by_id["Q1"]["bow"])
            self.assertEqual(docs_by_id["Q1"]["labels"]["en"], "Universe")
            self.assertIn("Cosmos", docs_by_id["Q1"]["aliases"]["en"])
            self.assertEqual(docs_by_id["Q1"]["coarse_type"], "THING")
            self.assertEqual(docs_by_id["Q1"]["fine_type"], "COSMOLOGICAL_ENTITY")
            self.assertEqual(docs_by_id["P31"]["coarse_type"], "RELATION")
            self.assertEqual(docs_by_id["P31"]["fine_type"], "PROPERTY")
            self.assertNotIn("ner_coarse_types", docs_by_id["Q1"])
            self.assertNotIn("ner_fine_types", docs_by_id["Q1"])
            self.assertNotIn("ner_coarse_types", docs_by_id["P31"])
            self.assertNotIn("ner_fine_types", docs_by_id["P31"])

    def test_limit_enables_fast_smoke_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dump_path = root / "tiny.json.bz2"
            db_path = root / "labels.sqlite"
            write_tiny_dump(dump_path)

            exit_code = run_labels_db(
                dump_path=dump_path,
                db_path=db_path,
                batch_size=2,
                limit=1,
                disable_ner_classifier=False,
            )
            self.assertEqual(exit_code, 0)

            conn = sqlite3.connect(db_path)
            try:
                row_count = conn.execute("SELECT COUNT(*) FROM labels").fetchone()
                assert row_count is not None
                self.assertEqual(row_count[0], 1)
            finally:
                conn.close()


if __name__ == "__main__":
    unittest.main()
