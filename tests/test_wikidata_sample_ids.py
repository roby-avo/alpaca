from __future__ import annotations

import unittest

from src.wikidata_sample_ids import default_demo_qids, resolve_qids


class WikidataSampleIdsTests(unittest.TestCase):
    def test_default_demo_qids_is_deterministic(self) -> None:
        self.assertEqual(default_demo_qids(3), ["Q1", "Q2", "Q3"])

    def test_resolve_qids_supports_count(self) -> None:
        self.assertEqual(resolve_qids(None, None, 2), ["Q1", "Q2"])

    def test_resolve_qids_rejects_multiple_selectors(self) -> None:
        with self.assertRaises(ValueError):
            resolve_qids("Q42", None, 2)


if __name__ == "__main__":
    unittest.main()
