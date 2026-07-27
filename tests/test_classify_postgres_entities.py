from __future__ import annotations

import unittest

from src.classify_postgres_entities import (
    REQUIRED_CLASSIFIER_VERSION,
    _classify_row,
)


class ClassifyPostgresEntitiesTests(unittest.TestCase):
    def test_classify_row_uses_resolved_type_labels_and_description_refinement(self) -> None:
        result = _classify_row(
            (
                "Q3441181",
                "Rome Against Rome",
                "1964 sword-and-sandal film",
                ["Q11424"],
                ["film"],
                [],
                [],
            )
        )

        assert result is not None
        self.assertEqual(result[1], "CREATIVE_WORK")
        self.assertEqual(result[2], "FILM")
        self.assertEqual(result[4], "SWORD_AND_SANDAL_FILM")
        self.assertEqual(result[8], 4)
        self.assertEqual(
            result[9],
            ["CREATIVE_WORK", "FILM", "SWORD_AND_SANDAL_FILM"],
        )
        self.assertEqual(result[15], REQUIRED_CLASSIFIER_VERSION)
        self.assertEqual(result[16], "1.0.0/0.3.1")

    def test_classify_row_abstains_without_type_anchor(self) -> None:
        result = _classify_row(("Q999999999", "Unknown", "untyped item", [], [], [], []))

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
