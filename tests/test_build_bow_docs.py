from __future__ import annotations

import bz2
import gzip
import json
import tempfile
import unittest
from pathlib import Path

from src.build_bow_docs import run as run_bow_docs
from src.build_labels_db import run as run_labels_db


def _entity_claim(target_id: str, property_id: str) -> dict[str, object]:
    numeric_id = int(target_id[1:])
    return {
        "mainsnak": {
            "snaktype": "value",
            "property": property_id,
            "datavalue": {
                "type": "wikibase-entityid",
                "value": {
                    "entity-type": "item",
                    "numeric-id": numeric_id,
                    "id": target_id,
                },
            },
        },
    }


def write_context_dump(path: Path) -> None:
    entities = [
        {
            "id": "Q1",
            "labels": {"en": {"language": "en", "value": "Douglas Adams"}},
            "aliases": {"en": [{"language": "en", "value": "DNA"}]},
            "descriptions": {"en": {"language": "en", "value": "English writer"}},
            "claims": {
                "P31": [_entity_claim("Q5", "P31")],
                "P27": [_entity_claim("Q30", "P27")],
            },
        },
        {
            "id": "Q5",
            "labels": {"en": {"language": "en", "value": "human"}},
            "descriptions": {"en": {"language": "en", "value": "common name for Homo sapiens"}},
        },
        {
            "id": "Q30",
            "labels": {"en": {"language": "en", "value": "United States"}},
            "descriptions": {"en": {"language": "en", "value": "country in North America"}},
        },
    ]

    with bz2.open(path, mode="wt", encoding="utf-8") as handle:
        handle.write("[\n")
        for index, entity in enumerate(entities):
            suffix = "," if index < len(entities) - 1 else ""
            handle.write(json.dumps(entity, ensure_ascii=False) + suffix + "\n")
        handle.write("]\n")


class BuildBowDocsContextTests(unittest.TestCase):
    def test_context_uses_object_labels_not_predicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dump_path = root / "context.json.bz2"
            labels_db_path = root / "labels.sqlite"
            output_path = root / "bow_docs.jsonl.gz"
            write_context_dump(dump_path)

            labels_exit = run_labels_db(
                dump_path=dump_path,
                db_path=labels_db_path,
                batch_size=5,
                limit=0,
                disable_ner_classifier=False,
            )
            self.assertEqual(labels_exit, 0)

            bow_exit = run_bow_docs(
                dump_path=dump_path,
                labels_db_path=labels_db_path,
                output_path=output_path,
                batch_size=5,
                limit=0,
                ner_types_path=None,
            )
            self.assertEqual(bow_exit, 0)

            with gzip.open(output_path, mode="rt", encoding="utf-8") as handle:
                docs = [json.loads(line) for line in handle if line.strip()]

            docs_by_id = {doc["id"]: doc for doc in docs}
            q1 = docs_by_id["Q1"]
            context = q1["context"].casefold()
            bow = q1["bow"].casefold()

            self.assertIn("human", context)
            self.assertIn("united states", context)
            self.assertNotIn("p31", context)
            self.assertNotIn("p27", context)
            self.assertNotIn("p31", bow)
            self.assertNotIn("p27", bow)


if __name__ == "__main__":
    unittest.main()
