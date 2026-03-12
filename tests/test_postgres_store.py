from __future__ import annotations

import unittest

from src.postgres_store import (
    _entity_search_columns,
    _expand_dbpedia_ref,
    _expand_wikipedia_ref,
    compact_crosslink_hint,
)


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


if __name__ == "__main__":
    unittest.main()
