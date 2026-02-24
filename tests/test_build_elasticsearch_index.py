from __future__ import annotations

import json
import unittest

from src.build_elasticsearch_index import iter_bulk_payloads_from_store, normalize_exact_text


class _FakeStore:
    def __init__(self) -> None:
        self._entities = [
            {
                "qid": "Q1",
                "label": "Città di Roma",
                "labels": {"it": "Città di Roma"},
                "aliases": {"en": ["Rome"]},
                "context_string": "Italy; Capital",
                "coarse_type": "LOCATION",
                "fine_type": "CITY",
                "popularity": 100.0,
                "cross_refs": {},
            },
            {
                "qid": "Q2",
                "label": "Apple Inc.",
                "labels": {"en": "Apple Inc."},
                "aliases": {"en": ["Apple"]},
                "context_string": "Technology; Cupertino",
                "coarse_type": "ORGANIZATION",
                "fine_type": "COMPANY",
                "popularity": 200.0,
                "cross_refs": {},
            },
        ]

    def iter_entities_for_indexing(self, *, batch_size: int):  # type: ignore[override]
        if batch_size <= 0:
            raise AssertionError("batch_size must be > 0")
        yield self._entities[:1]
        yield self._entities[1:]


class BuildElasticsearchIndexTests(unittest.TestCase):
    def test_normalize_exact_text_ascii_folds_and_strips_punctuation(self) -> None:
        self.assertEqual(normalize_exact_text(" Città-di Roma! "), "citta di roma")

    def test_iter_bulk_payloads_from_store_chunks_ndjson(self) -> None:
        store = _FakeStore()
        chunks = list(
            iter_bulk_payloads_from_store(
                store,  # type: ignore[arg-type]
                fetch_batch_size=1,
                max_chunk_bytes=500,
                index_name="wikidata_entities",
            )
        )
        self.assertGreaterEqual(len(chunks), 2)

        total_docs = 0
        for payload, doc_count in chunks:
            total_docs += doc_count
            lines = [line for line in payload.splitlines() if line]
            self.assertEqual(len(lines), doc_count * 2)
            for idx in range(0, len(lines), 2):
                meta = json.loads(lines[idx])
                doc = json.loads(lines[idx + 1])
                self.assertIn("index", meta)
                self.assertIn("qid", doc)
                self.assertIsInstance(doc.get("labels"), list)
                self.assertIsInstance(doc.get("aliases"), list)
                self.assertNotIn("aliases_by_lang", doc)
        self.assertEqual(total_docs, 2)


if __name__ == "__main__":
    unittest.main()
