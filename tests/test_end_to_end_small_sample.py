from __future__ import annotations

import bz2
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from src.common import tokenize
from src.search_logic import build_quickwit_query, rerank_hits_by_context
import src.run_pipeline as run_pipeline

try:
    from src.api import SearchRequest, search_entities
except ModuleNotFoundError:
    SearchRequest = None
    search_entities = None


def write_small_dump(path: Path) -> None:
    entities = [
        {
            "id": "Q312",
            "labels": {
                "en": {"language": "en", "value": "Apple Inc."},
            },
            "aliases": {
                "en": [{"language": "en", "value": "Apple"}],
            },
            "descriptions": {
                "en": {
                    "language": "en",
                    "value": "American technology company based in Cupertino.",
                },
            },
        },
        {
            "id": "Q89",
            "labels": {
                "en": {"language": "en", "value": "apple"},
            },
            "aliases": {
                "en": [{"language": "en", "value": "fruit"}],
            },
            "descriptions": {
                "en": {
                    "language": "en",
                    "value": "Edible fruit produced by an apple tree.",
                },
            },
        },
        {
            "id": "L1",
            "labels": {
                "en": {"language": "en", "value": "ignored lexeme"},
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


def write_small_ner_types(path: Path) -> None:
    records = [
        {
            "id": "Q312",
            "coarse_types": ["ORGANIZATION"],
            "fine_types": ["COMPANY"],
        },
        {
            "id": "Q89",
            "coarse_types": ["CONCEPT"],
            "fine_types": ["FOOD"],
        },
    ]
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


class FakeQuickwitClient:
    _indexes: dict[str, list[dict[str, Any]]] = {}

    def __init__(self, base_url: str, timeout_seconds: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    @classmethod
    def reset(cls) -> None:
        cls._indexes = {}

    def list_indexes(self) -> list[dict[str, str]]:
        return [{"index_id": index_id} for index_id in sorted(self._indexes)]

    def is_healthy(self) -> bool:
        return True

    def ensure_index(self, index_id: str, index_config: dict[str, Any]) -> dict[str, Any]:
        _ = index_config
        self._indexes.setdefault(index_id, [])
        return {"created": True}

    def ingest_ndjson(self, index_id: str, ndjson_payload: str, commit: str = "auto") -> dict[str, Any]:
        _ = commit
        bucket = self._indexes.setdefault(index_id, [])
        for line in ndjson_payload.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                bucket.append(parsed)
        return {"num_docs_for_processing": len(bucket)}

    def search(self, index_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        docs = self._indexes.get(index_id, [])
        query = payload.get("query")
        query_text = query if isinstance(query, str) else ""
        max_hits_raw = payload.get("max_hits")
        max_hits = max_hits_raw if isinstance(max_hits_raw, int) and max_hits_raw > 0 else 20

        terms, coarse_filters, fine_filters = self._parse_query(query_text)

        hits: list[dict[str, Any]] = []
        for doc in docs:
            if not isinstance(doc, dict):
                continue

            entity_id = doc.get("id")
            if not isinstance(entity_id, str) or not entity_id:
                continue

            name_text = doc.get("name_text") if isinstance(doc.get("name_text"), str) else ""
            bow = doc.get("bow") if isinstance(doc.get("bow"), str) else ""
            doc_terms = set(tokenize(f"{name_text} {bow}"))

            if terms and not set(terms).issubset(doc_terms):
                continue

            doc_coarse: set[str] = set()
            coarse_type = doc.get("coarse_type")
            if isinstance(coarse_type, str) and coarse_type:
                doc_coarse.add(coarse_type)
            if coarse_filters and doc_coarse.isdisjoint(coarse_filters):
                continue

            doc_fine: set[str] = set()
            fine_type = doc.get("fine_type")
            if isinstance(fine_type, str) and fine_type:
                doc_fine.add(fine_type)
            if fine_filters and doc_fine.isdisjoint(fine_filters):
                continue

            score = float(len(set(terms) & doc_terms)) if terms else 1.0
            hits.append({"_source": doc, "_score": score})

        hits.sort(
            key=lambda item: (
                float(item.get("_score", 0.0)),
                str(item.get("_source", {}).get("id", "")),
            ),
            reverse=True,
        )
        return {"num_hits": len(hits), "hits": hits[:max_hits]}

    @staticmethod
    def _parse_query(query: str) -> tuple[list[str], set[str], set[str]]:
        coarse_filters = set(re.findall(r"coarse_type:([A-Za-z0-9_.:/-]+)", query))
        fine_filters = set(re.findall(r"fine_type:([A-Za-z0-9_.:/-]+)", query))

        stripped = re.sub(r"coarse_type:[A-Za-z0-9_.:/-]+", " ", query)
        stripped = re.sub(r"fine_type:[A-Za-z0-9_.:/-]+", " ", stripped)
        terms = [token for token in tokenize(stripped) if token not in {"and", "or"}]
        return terms, coarse_filters, fine_filters


class EndToEndSmallSampleTests(unittest.TestCase):
    def test_pipeline_to_api_on_small_sample(self) -> None:
        FakeQuickwitClient.reset()
        workspace_root = Path(__file__).resolve().parents[1]
        index_config_path = workspace_root / "quickwit" / "wikidata-entities-index-config.json"

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dump_path = root / "tiny.json.bz2"
            labels_db_path = root / "labels.sqlite"
            bow_output_path = root / "bow_docs.jsonl.gz"
            ner_types_path = root / "ner_types.jsonl"
            write_small_dump(dump_path)
            write_small_ner_types(ner_types_path)

            argv = [
                "run_pipeline.py",
                "--dump-path",
                str(dump_path),
                "--labels-db-path",
                str(labels_db_path),
                "--bow-output-path",
                str(bow_output_path),
                "--ner-types-path",
                str(ner_types_path),
                "--index-config-path",
                str(index_config_path),
                "--quickwit-url",
                "http://fake-quickwit:7280",
                "--index-id",
                "wikidata_entities",
                "--labels-batch-size",
                "2",
                "--bow-batch-size",
                "2",
                "--quickwit-chunk-bytes",
                "2048",
                "--wait-timeout-seconds",
                "1",
                "--poll-interval-seconds",
                "0.1",
            ]

            with mock.patch.object(sys, "argv", argv), mock.patch(
                "src.build_quickwit_index.QuickwitClient",
                FakeQuickwitClient,
            ):
                exit_code = run_pipeline.main()

            self.assertEqual(exit_code, 0)
            indexed_docs = FakeQuickwitClient._indexes.get("wikidata_entities", [])
            self.assertEqual({doc.get("id") for doc in indexed_docs}, {"Q312", "Q89"})

            search_client = FakeQuickwitClient("http://fake-quickwit:7280")

            unfiltered_query = build_quickwit_query("apple")
            unfiltered_response = search_client.search(
                index_id="wikidata_entities",
                payload={"query": unfiltered_query, "max_hits": 20},
            )
            unfiltered_hits = rerank_hits_by_context(
                unfiltered_response["hits"],
                "cupertino technology",
                limit=2,
            )
            self.assertEqual(len(unfiltered_hits), 2)
            self.assertEqual(unfiltered_hits[0]["id"], "Q312")
            self.assertGreater(
                unfiltered_hits[0]["context_overlap"],
                unfiltered_hits[1]["context_overlap"],
            )

            filtered_query = build_quickwit_query(
                "apple",
                ["ORGANIZATION"],
                ["COMPANY"],
            )
            filtered_response = search_client.search(
                index_id="wikidata_entities",
                payload={"query": filtered_query, "max_hits": 20},
            )
            filtered_hits = rerank_hits_by_context(
                filtered_response["hits"],
                "cupertino technology",
                limit=2,
            )
            self.assertEqual(len(filtered_hits), 1)
            self.assertEqual(filtered_hits[0]["id"], "Q312")
            self.assertIn("coarse_type:ORGANIZATION", filtered_query)
            self.assertIn("fine_type:COMPANY", filtered_query)

            if SearchRequest is not None and search_entities is not None:
                with mock.patch(
                    "src.api.get_quickwit_client",
                    return_value=(
                        FakeQuickwitClient("http://fake-quickwit:7280"),
                        "wikidata_entities",
                    ),
                ):
                    api_response = search_entities(
                        SearchRequest(
                            query="apple",
                            context="cupertino technology",
                            ner_coarse_types=["ORGANIZATION"],
                            ner_fine_types=["COMPANY"],
                            limit=2,
                        )
                    )
                self.assertEqual(api_response.returned, 1)
                self.assertEqual(api_response.hits[0].id, "Q312")


if __name__ == "__main__":
    unittest.main()
