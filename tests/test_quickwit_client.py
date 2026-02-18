from __future__ import annotations

import unittest
from unittest import mock

from src.quickwit_client import QuickwitClient, QuickwitClientError


class QuickwitClientTests(unittest.TestCase):
    def test_ensure_index_uses_create_endpoint(self) -> None:
        client = QuickwitClient("http://localhost:7280")
        config = {"index_id": "wikidata_entities"}

        request_mock = mock.Mock(return_value={"created": True})
        client._request_json = request_mock  # type: ignore[method-assign]

        result = client.ensure_index("wikidata_entities", config)

        self.assertEqual(result, {"created": True})
        request_mock.assert_called_once_with("POST", "/api/v1/indexes", payload=config)

    def test_ensure_index_returns_empty_on_create_conflict(self) -> None:
        client = QuickwitClient("http://localhost:7280")
        config = {"index_id": "wikidata_entities"}

        request_mock = mock.Mock(
            side_effect=QuickwitClientError("already exists", status_code=409)
        )
        client._request_json = request_mock  # type: ignore[method-assign]

        result = client.ensure_index("wikidata_entities", config)

        self.assertEqual(result, {})
        request_mock.assert_called_once_with("POST", "/api/v1/indexes", payload=config)

    def test_ensure_index_falls_back_to_legacy_endpoint(self) -> None:
        client = QuickwitClient("http://localhost:7280")
        config = {"index_id": "wikidata_entities"}

        request_mock = mock.Mock(
            side_effect=[
                QuickwitClientError("not found", status_code=404),
                {"created": True},
            ]
        )
        client._request_json = request_mock  # type: ignore[method-assign]

        result = client.ensure_index("wikidata_entities", config)

        self.assertEqual(result, {"created": True})
        self.assertEqual(
            request_mock.call_args_list,
            [
                mock.call("POST", "/api/v1/indexes", payload=config),
                mock.call(
                    "PUT",
                    "/api/v1/indexes/wikidata_entities?create=true",
                    payload=config,
                ),
            ],
        )

    def test_ensure_index_raises_on_server_error(self) -> None:
        client = QuickwitClient("http://localhost:7280")
        config = {"index_id": "wikidata_entities"}

        request_mock = mock.Mock(
            side_effect=QuickwitClientError("upstream error", status_code=502)
        )
        client._request_json = request_mock  # type: ignore[method-assign]

        with self.assertRaises(QuickwitClientError):
            client.ensure_index("wikidata_entities", config)


if __name__ == "__main__":
    unittest.main()
