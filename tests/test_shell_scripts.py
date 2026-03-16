from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ShellScriptRegressionTests(unittest.TestCase):
    def test_live_sample_pipeline_uses_sample_cache_directly(self) -> None:
        script = (ROOT / "scripts" / "run_live_sample_pipeline_docker.sh").read_text(encoding="utf-8")

        self.assertIn("--sample-cache-count", script)
        self.assertIn("--sample-cache-ids", script)
        self.assertNotIn("build_postgres_sample_dump", script)

    def test_build_small_dump_script_no_longer_exposes_live_mode(self) -> None:
        script = (ROOT / "scripts" / "build_small_dump.sh").read_text(encoding="utf-8")

        self.assertNotIn("--live", script)
        self.assertNotIn("wikidata_sample_postgres", script)


if __name__ == "__main__":
    unittest.main()
