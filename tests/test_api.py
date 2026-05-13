from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from src import api


TEST_API_TOKEN = "s" * 32


class ApiAuthTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(api.app)

    def test_lookup_requires_bearer_token(self) -> None:
        with patch.dict(
            os.environ,
            {"ALPACA_API_TOKEN": TEST_API_TOKEN, "ALPACA_API_TOKEN_HASHES": ""},
            clear=False,
        ):
            response = self.client.post("/lookup", json={"mention": "Rome"})

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.headers.get("www-authenticate"), 'Bearer realm="alpaca"')

    def test_lookup_accepts_valid_bearer_token(self) -> None:
        lookup_payload = {
            "mention": "Rome",
            "mention_norm": "rome",
            "mention_context_terms": [],
            "coarse_hints": [],
            "fine_hints": [],
            "strategy": "test",
            "returned": 0,
            "cache_hit": False,
            "top1": None,
        }
        with (
            patch.dict(
                os.environ,
                {"ALPACA_API_TOKEN": TEST_API_TOKEN, "ALPACA_API_TOKEN_HASHES": ""},
                clear=False,
            ),
            patch("src.api._run_lookup", return_value=lookup_payload),
        ):
            response = self.client.post(
                "/lookup",
                headers={"Authorization": f"Bearer {TEST_API_TOKEN}"},
                json={"mention": "Rome"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["mention"], "Rome")

    def test_lookup_accepts_configured_token_hash(self) -> None:
        lookup_payload = {
            "mention": "Rome",
            "mention_norm": "rome",
            "mention_context_terms": [],
            "coarse_hints": [],
            "fine_hints": [],
            "strategy": "test",
            "returned": 0,
            "cache_hit": False,
            "top1": None,
        }
        with (
            patch.dict(
                os.environ,
                {
                    "ALPACA_API_TOKEN": "",
                    "ALPACA_API_TOKEN_HASHES": api._hash_token(TEST_API_TOKEN),
                },
                clear=False,
            ),
            patch("src.api._run_lookup", return_value=lookup_payload),
        ):
            response = self.client.post(
                "/lookup",
                headers={"Authorization": f"Bearer {TEST_API_TOKEN}"},
                json={"mention": "Rome"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["mention"], "Rome")

    def test_short_configured_token_fails_closed(self) -> None:
        with patch.dict(
            os.environ,
            {"ALPACA_API_TOKEN": "short-token", "ALPACA_API_TOKEN_HASHES": ""},
            clear=False,
        ):
            response = self.client.post(
                "/lookup",
                headers={"Authorization": "Bearer short-token"},
                json={"mention": "Rome"},
            )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"], "API authentication is misconfigured.")

    def test_openapi_documents_bearer_auth_and_avatar(self) -> None:
        response = self.client.get("/openapi.json")

        self.assertEqual(response.status_code, 200)
        schema = response.json()
        self.assertEqual(schema["info"]["x-logo"]["url"], "/assets/alpaca-avatar.png")
        self.assertIn("BearerAuth", schema["components"]["securitySchemes"])
        self.assertNotIn("ALPACA_API_TOKEN", json.dumps(schema["components"]["securitySchemes"]))
        self.assertEqual(
            schema["paths"]["/lookup"]["post"]["security"],
            [{"BearerAuth": []}],
        )

    def test_docs_content_length_matches_customized_html(self) -> None:
        response = self.client.get("/docs")

        self.assertEqual(response.status_code, 200)
        self.assertIn('image.src = "/assets/alpaca-avatar.png";', response.text)
        self.assertIn('description.insertAdjacentElement("afterend", wrapper);', response.text)
        self.assertEqual(int(response.headers["content-length"]), len(response.content))

    def test_healthz_uses_lightweight_postgres_ping(self) -> None:
        store = unittest.mock.Mock()
        with (
            patch.dict(
                os.environ,
                {"ALPACA_API_TOKEN": TEST_API_TOKEN, "ALPACA_API_TOKEN_HASHES": ""},
                clear=False,
            ),
            patch("src.api.PostgresStore", return_value=store),
        ):
            response = self.client.get(
                "/healthz",
                headers={"Authorization": f"Bearer {TEST_API_TOKEN}"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")
        store.ping.assert_called_once_with()
        store.count_entities.assert_not_called()
        store.ensure_schema.assert_not_called()
        store.ensure_search_indexes.assert_not_called()

    def test_openapi_does_not_expose_database_maintenance_endpoint(self) -> None:
        response = self.client.get("/openapi.json")

        self.assertEqual(response.status_code, 200)
        self.assertNotIn("/admin/reindex", response.json()["paths"])


class ApiElasticsearchDebugTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(api.app)

    def test_debug_elasticsearch_search_rejects_unsafe_index_name(self) -> None:
        with patch.dict(
            os.environ,
            {"ALPACA_API_TOKEN": TEST_API_TOKEN, "ALPACA_API_TOKEN_HASHES": ""},
            clear=False,
        ):
            response = self.client.post(
                "/debug/elasticsearch/BadIndex/_search",
                headers={"Authorization": f"Bearer {TEST_API_TOKEN}"},
                json={"query": {"match_all": {}}},
            )

        self.assertEqual(response.status_code, 422)

    def test_debug_elasticsearch_search_forwards_es_query_body(self) -> None:
        request_body = {"size": 1, "query": {"match_all": {}}}
        with (
            patch.dict(
                os.environ,
                {
                    "ALPACA_API_TOKEN": TEST_API_TOKEN,
                    "ALPACA_API_TOKEN_HASHES": "",
                    "ALPACA_ELASTICSEARCH_URL": "http://elasticsearch:9200",
                },
                clear=False,
            ),
            patch("src.api._debug_elasticsearch_search", return_value={"hits": {"hits": []}}) as search,
        ):
            response = self.client.post(
                "/debug/elasticsearch/alpaca-entities/_search",
                headers={"Authorization": f"Bearer {TEST_API_TOKEN}"},
                json=request_body,
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"hits": {"hits": []}})
        search.assert_called_once_with("alpaca-entities", request_body)


if __name__ == "__main__":
    unittest.main()
