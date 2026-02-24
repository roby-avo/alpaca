from __future__ import annotations

import unittest

from src.search_logic import normalize_type_labels


class SearchLogicTests(unittest.TestCase):
    def test_normalize_type_labels_dedupes_and_preserves_order(self) -> None:
        result = normalize_type_labels(
            ["PERSON", "ORG", "PERSON", " "],
            field_name="coarse_type",
        )
        self.assertEqual(result, ["PERSON", "ORG"])

    def test_normalize_type_labels_rejects_invalid_values(self) -> None:
        with self.assertRaises(ValueError):
            normalize_type_labels(["PERSON", "ORG)"], field_name="coarse_type")

    def test_normalize_type_labels_none_returns_empty(self) -> None:
        self.assertEqual(normalize_type_labels(None, field_name="fine_type"), [])


if __name__ == "__main__":
    unittest.main()
