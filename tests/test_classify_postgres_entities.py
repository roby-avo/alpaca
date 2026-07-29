from __future__ import annotations

import unittest

from src.classify_postgres_entities import (
    REQUIRED_CLASSIFIER_VERSION,
    _classify_row,
    _partition_rows_by_type_signature,
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

    def test_native_batch_exactly_matches_individual_library_prediction(self) -> None:
        classifier = WikidataNERClassifier()
        items = [
            {
                "qid": f"Q{index}",
                "types": [{"name": "film"}],
                "ancestor_types": [],
                "description": f"{1960 + index} documentary film",
            }
            for index in range(3)
        ]
        expected = [
            classifier.predict(
                qid=item["qid"],
                types=item["types"],
                ancestor_types=item["ancestor_types"],
                description=item["description"],
            )
            for item in items
        ]
        classifier.clear_branch_cache()
        actual = classifier.predict_batch(items, cache_size=100_000)

        self.assertEqual(
            [prediction.to_dict() for prediction in actual],
            [prediction.to_dict() for prediction in expected],
        )
        cache = classifier.branch_cache_info()
        self.assertEqual(cache.misses, 1)
        self.assertEqual(cache.hits, 2)

    def test_batches_are_balanced_without_splitting_normal_signature_groups(self) -> None:
        rows = [
            (
                f"Q{index}",
                f"Film {index}",
                f"{1960 + index} documentary film",
                ["Q11424"],
                ["film"],
                [],
                [],
            )
            for index in range(3)
        ]
        rows.extend(
            (
                f"Q{index + 3}",
                f"City {index}",
                "city in Europe",
                ["Q515"],
                ["city"],
                [],
                [],
            )
            for index in range(3)
        )

        partitions = _partition_rows_by_type_signature(rows, workers=2)

        self.assertEqual([len(partition) for partition in partitions], [3, 3])
        signatures_per_partition = [
            {tuple(row[4]) for row in partition}
            for partition in partitions
        ]
        self.assertTrue(all(len(signatures) == 1 for signatures in signatures_per_partition))


if __name__ == "__main__":
    unittest.main()
