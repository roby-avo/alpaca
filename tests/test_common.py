from __future__ import annotations

import unittest
from unittest.mock import patch

from src.common import default_postgres_dsn, resolve_postgres_dsn
from src.index_postgres_to_elasticsearch import default_elasticsearch_url


class CommonConfigTests(unittest.TestCase):
    def test_default_postgres_dsn_uses_docker_service_name_inside_container(self) -> None:
        with patch("src.common.os.path.exists", return_value=True):
            self.assertEqual(default_postgres_dsn(), "postgresql://postgres@postgres:5432/alpaca")

    def test_default_postgres_dsn_uses_localhost_outside_container(self) -> None:
        with patch("src.common.os.path.exists", return_value=False):
            self.assertEqual(resolve_postgres_dsn(None), "postgresql://postgres@localhost:5432/alpaca")

    def test_default_elasticsearch_url_uses_docker_service_name_inside_container(self) -> None:
        with patch("src.common.os.path.exists", return_value=True):
            self.assertEqual(default_elasticsearch_url(), "http://elasticsearch:9200")


if __name__ == "__main__":
    unittest.main()
