from __future__ import annotations

import unittest

from src.postgres_store import sampled_seed_row_number
from src.simulate_entities_size import project_entity_triple_stats


class SimulateEntitiesSizeTests(unittest.TestCase):
    def test_sampled_seed_row_number_is_deterministic_for_same_seed(self) -> None:
        first = [sampled_seed_row_number(sample_no=index, seed_count=5, random_seed=1337) for index in range(8)]
        second = [sampled_seed_row_number(sample_no=index, seed_count=5, random_seed=1337) for index in range(8)]

        self.assertEqual(first, second)

    def test_sampled_seed_row_number_changes_when_seed_changes(self) -> None:
        first = [sampled_seed_row_number(sample_no=index, seed_count=5, random_seed=1337) for index in range(8)]
        second = [sampled_seed_row_number(sample_no=index, seed_count=5, random_seed=1338) for index in range(8)]

        self.assertNotEqual(first, second)

    def test_project_entity_triple_stats_scales_counts_and_bytes_linearly(self) -> None:
        projection = project_entity_triple_stats(
            sample_entities=500,
            triple_stats={
                "rows": 5_000,
                "table_bytes": 50_000,
                "toast_bytes": 0,
                "index_bytes": 20_000,
                "total_bytes": 70_000,
            },
            project_rows=100_000_000,
        )

        self.assertEqual(projection["sample_triples"], 5_000)
        self.assertEqual(projection["projected_triples"], 1_000_000_000)
        self.assertEqual(projection["table_bytes"], 10_000_000_000)
        self.assertEqual(projection["index_bytes"], 4_000_000_000)
        self.assertEqual(projection["total_bytes"], 14_000_000_000)


if __name__ == "__main__":
    unittest.main()
