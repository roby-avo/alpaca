from __future__ import annotations

import unittest

from src.classify_postgres_entities import (
    REQUIRED_CLASSIFIER_VERSION,
    _classify_row,
    _predict_with_cached_branch,
    _predict_without_description,
)
from wikidata_ner import WikidataNERClassifier


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

    def test_cached_branch_refinement_exactly_matches_library_prediction(self) -> None:
        classifier = WikidataNERClassifier()
        cases = (
            ("Q1", ("film",), (), "1964 sword-and-sandal film"),
            ("Q2", ("film",), (), "American documentary film"),
            ("Q3", ("human",), (), "French painter and sculptor"),
            ("Q4", ("city",), (), "capital city in Europe"),
            ("Q5", ("novel",), (), "1995 science fiction novel"),
            ("Q6", ("business enterprise",), (), "American software company"),
            ("Q7", ("Q999999",), (), "unresolved item"),
            ("Q8", (), (), "untyped item"),
        )
        _predict_without_description.cache_clear()
        for qid, type_names, ancestor_names, description in cases:
            with self.subTest(qid=qid):
                expected = classifier.predict(
                    qid=qid,
                    types=type_names,
                    ancestor_types=ancestor_names,
                    description=description,
                )
                actual = _predict_with_cached_branch(
                    qid=qid,
                    type_names=type_names,
                    ancestor_names=ancestor_names,
                    description=description,
                )
                self.assertEqual(actual.to_dict(), expected.to_dict())

    def test_cached_branch_reuses_repeated_type_signatures_with_descriptions(self) -> None:
        _predict_without_description.cache_clear()
        for index in range(3):
            _predict_with_cached_branch(
                qid=f"Q{index}",
                type_names=("film",),
                ancestor_names=(),
                description=f"{1960 + index} documentary film",
            )

        cache = _predict_without_description.cache_info()
        self.assertEqual(cache.misses, 1)
        self.assertEqual(cache.hits, 2)


if __name__ == "__main__":
    unittest.main()
