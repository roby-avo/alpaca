from __future__ import annotations

import unittest
from unittest.mock import patch

from src.postgres_store import (
    PostgresStore,
    _entity_triples_index_create_statements,
    _entity_triples_index_drop_statements,
    _entity_search_columns,
    _expand_dbpedia_ref,
    _expand_wikipedia_ref,
    compact_crosslink_hint,
)


class FakeCursor:
    def __init__(self, rows: list[tuple[str, str]]) -> None:
        self.rows = rows
        self.executed_sql = ""
        self.executed_params = ()

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def execute(self, sql: str, params: object = ()) -> None:
        self.executed_sql = sql
        self.executed_params = params

    def fetchall(self) -> list[tuple[str, str]]:
        return self.rows


class FakeConnection:
    def __init__(self, cursor: FakeCursor) -> None:
        self.cursor_obj = cursor

    def __enter__(self) -> "FakeConnection":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def cursor(self) -> FakeCursor:
        return self.cursor_obj


class PostgresStoreHelpersTests(unittest.TestCase):
    def test_entity_search_columns_keep_multilingual_labels_and_dedupe_aliases(self) -> None:
        columns = _entity_search_columns(
            label="Rome",
            labels={"en": "Rome", "it": "Roma", "mul": "Rome"},
            aliases={"en": ["Rome city"], "it": ["Roma", "Rome"]},
            cross_refs={
                "wikipedia": "https://it.wikipedia.org/wiki/Roma",
                "dbpedia": "https://it.dbpedia.org/resource/Roma",
            },
            popularity=10.0,
        )

        self.assertEqual(columns["labels"], ["Rome", "Roma"])
        self.assertEqual(columns["aliases"], ["Rome city"])
        self.assertEqual(columns["wikipedia_url"], "it.wikipedia.org|Roma")
        self.assertEqual(columns["dbpedia_url"], "it.dbpedia.org|Roma")

    def test_compact_and_expand_cross_refs_support_non_english_hosts(self) -> None:
        wikipedia = compact_crosslink_hint("https://it.wikipedia.org/wiki/Roma")
        dbpedia = compact_crosslink_hint("https://it.dbpedia.org/resource/Roma")

        self.assertEqual(wikipedia, "it.wikipedia.org|Roma")
        self.assertEqual(dbpedia, "it.dbpedia.org|Roma")
        self.assertEqual(_expand_wikipedia_ref(wikipedia), "https://it.wikipedia.org/wiki/Roma")
        self.assertEqual(_expand_dbpedia_ref(dbpedia), "https://it.dbpedia.org/resource/Roma")

    def test_entity_triples_indexes_add_incoming_edge_covering_index(self) -> None:
        create_statements = _entity_triples_index_create_statements()
        self.assertEqual(len(create_statements), 1)
        self.assertIn(
            "idx_entity_triples_object_qid_predicate_pid_subject_qid",
            create_statements[0],
        )
        self.assertIn(
            "ON entity_triples (object_qid, predicate_pid, subject_qid)",
            create_statements[0],
        )

    def test_entity_triples_index_drop_statements_only_remove_legacy_indexes(self) -> None:
        drop_statements = _entity_triples_index_drop_statements()
        self.assertEqual(
            drop_statements,
            [
                'DROP INDEX IF EXISTS "idx_entity_triples_subject_qid";',
                'DROP INDEX IF EXISTS "idx_entity_triples_object_qid";',
                'DROP INDEX IF EXISTS "idx_entity_triples_predicate_pid";',
            ],
        )

    def test_resolve_type_labels_batches_qids_and_filters_to_type_items(self) -> None:
        cursor = FakeCursor([("Q5", "human"), ("Q515", "city")])
        store = PostgresStore("postgresql://example/db")

        with patch.object(store, "_connect", return_value=FakeConnection(cursor)):
            labels = store.resolve_type_labels(["Q5", "Q515", "Q5", " "])

        self.assertEqual(labels, {"Q5": "human", "Q515": "city"})
        self.assertIn("UPPER(item_category) = 'TYPE'", cursor.executed_sql)
        self.assertEqual(cursor.executed_params, (["Q5", "Q515"],))


if __name__ == "__main__":
    unittest.main()
