from __future__ import annotations

import unittest

from src.classify_postgres_entities import (
    REQUIRED_CLASSIFIER_VERSION,
    _classify_row,
    _normalize_required_type_qid,
    _partition_rows_by_type_signature,
    _predict_human_item_fast,
    _stage_name,
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
        self.assertEqual(result[16], "1.0.0/0.4.0")

    def test_human_occupations_become_specific_types(self) -> None:
        result = _classify_row(
            (
                "Q7259",
                "Ada Lovelace",
                "English mathematician and writer",
                ["Q5", "Q170790", "Q36180", "Q81096"],
                ["human", "mathematician", "writer", "engineer"],
                [],
                [],
            )
        )

        assert result is not None
        self.assertEqual(result[1:3], ("PERSON", "HUMAN"))
        self.assertEqual(result[4], "ENGINEER")
        self.assertEqual(result[5], ["ENGINEER", "SCIENTIST", "WRITER"])
        self.assertEqual(
            result[7],
            {"occupation": ("ENGINEER", "SCIENTIST", "WRITER")},
        )

    def test_targeted_stage_name_is_separate_from_full_refresh(self) -> None:
        self.assertEqual(_normalize_required_type_qid("q5"), "Q5")
        self.assertEqual(
            _stage_name(source_table="entities", required_type_qid="Q5"),
            f"ner:entities:{REQUIRED_CLASSIFIER_VERSION}:type:Q5",
        )

    def test_human_fast_path_matches_library_hierarchy(self) -> None:
        item = {
            "qid": "Q7259",
            "types": [
                {"id": "Q5", "name": "human"},
                {"id": "Q170790", "name": "mathematician"},
                {"id": "Q36180", "name": "writer"},
                {"id": "Q81096", "name": "engineer"},
            ],
            "description": "English mathematician and writer",
        }
        expected = WikidataNERClassifier().predict_batch([item])[0]
        actual = _predict_human_item_fast(item)

        for field in (
            "coarse_type",
            "fine_type",
            "subtype",
            "specific_type",
            "specific_types",
            "facets",
            "specificity_level",
            "retrieval_path",
            "retrieval_key",
            "retrieval_tags",
            "abstained",
        ):
            self.assertEqual(getattr(actual, field), getattr(expected, field), field)

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

    def test_partitioning_balances_unique_signature_scan_cost(self) -> None:
        repeated = [
            (
                f"Q{index}",
                f"Film {index}",
                "documentary film",
                ["Q11424"],
                ["film"],
                [],
                [],
            )
            for index in range(4)
        ]
        unique = [
            (
                f"Q{index + 4}",
                f"Unique {index}",
                "item",
                [f"Q{1000 + index}"],
                [f"unique type {index}"],
                [],
                [],
            )
            for index in range(4)
        ]

        partitions = _partition_rows_by_type_signature(
            repeated + unique,
            workers=2,
            signature_cost_weight=64,
        )
        estimated_costs = [
            len(partition)
            + 64 * len({tuple(row[4]) for row in partition})
            for partition in partitions
        ]

        self.assertLessEqual(abs(estimated_costs[0] - estimated_costs[1]), 64)


if __name__ == "__main__":
    unittest.main()
