from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from src import run_pipeline


class _StubStore:
    def __init__(self, dsn: str) -> None:
        self.dsn = dsn

    def ensure_schema(self) -> None:
        return None

    def ensure_search_indexes(self) -> None:
        return None

    def compact_table_for_lookup(
        self,
        table_name: str,
        *,
        drop_cross_refs_trgm_index: bool,
        drop_context_inputs_table: bool,
        vacuum_full: bool,
        analyze: bool,
    ) -> None:
        self.table_name = table_name
        self.drop_cross_refs_trgm_index = drop_cross_refs_trgm_index
        self.drop_context_inputs_table = drop_context_inputs_table
        self.vacuum_full = vacuum_full
        self.analyze = analyze


class RunPipelineCliTests(unittest.TestCase):
    def test_main_passes_live_expected_total_to_pass1(self) -> None:
        argv = [
            "run_pipeline",
            "--dump-path",
            "/tmp/latest-all.json.bz2",
            "--postgres-dsn",
            "postgresql://postgres@localhost:5432/alpaca",
            "--fetch-live-entity-total",
        ]

        with (
            patch.object(sys, "argv", argv),
            patch("src.run_pipeline.resolve_expected_entity_total", return_value=120_830_824) as resolve_mock,
            patch("src.run_pipeline.run_postgres_pass1", return_value=0) as pass1_mock,
            patch("src.run_pipeline.PostgresStore", _StubStore),
        ):
            exit_code = run_pipeline.main()

        self.assertEqual(exit_code, 0)
        resolve_mock.assert_called_once()
        self.assertEqual(
            pass1_mock.call_args.kwargs["expected_entity_total"],
            120_830_824,
        )
        self.assertEqual(
            pass1_mock.call_args.kwargs["dump_path"],
            Path("/tmp/latest-all.json.bz2"),
        )


if __name__ == "__main__":
    unittest.main()
