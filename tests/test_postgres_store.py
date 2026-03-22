from __future__ import annotations

import unittest

from src.postgres_store import (
    _needs_legacy_entity_name_migration,
    PostgresStore,
    _entity_triples_index_create_statements,
    _entity_triples_index_drop_statements,
    _entity_search_columns,
    _expand_dbpedia_ref,
    _expand_wikipedia_ref,
    compact_crosslink_hint,
)


class PostgresStoreHelpersTests(unittest.TestCase):
    def test_legacy_name_migration_only_runs_when_legacy_sources_exist(self) -> None:
        self.assertFalse(
            _needs_legacy_entity_name_migration(
                entity_columns={"qid", "label", "labels", "aliases"},
                payload_type="",
            )
        )
        self.assertTrue(
            _needs_legacy_entity_name_migration(
                entity_columns={"qid", "label", "labels", "aliases", "name_variants"},
                payload_type="",
            )
        )
        self.assertTrue(
            _needs_legacy_entity_name_migration(
                entity_columns={"qid", "label", "labels", "aliases"},
                payload_type="bytea",
            )
        )

    def test_label_cache_eviction_stays_bounded(self) -> None:
        store = PostgresStore("postgresql://postgres@localhost:5432/alpaca", label_cache_size=2)

        store._cache_label("Q1", "Alpha")
        store._cache_label("Q2", "Beta")
        store._cache_label("Q3", "Gamma")

        self.assertEqual(list(store._label_cache.keys()), ["Q2", "Q3"])

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


if __name__ == "__main__":
    unittest.main()
