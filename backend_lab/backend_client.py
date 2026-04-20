from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib import error, parse, request


class BackendRequestError(RuntimeError):
    pass


@dataclass(slots=True)
class BackendHttpClient:
    api_url: str
    es_url: str
    timeout_seconds: float = 30.0

    def _request_json(
        self,
        *,
        url: str,
        method: str = "GET",
        payload: dict[str, Any] | None = None,
    ) -> Any:
        body: bytes | None = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = request.Request(url=url, method=method, data=body, headers=headers)
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise BackendRequestError(f"HTTP {exc.code} for {url}: {detail}") from exc
        except TimeoutError as exc:
            raise BackendRequestError(f"Request timed out for {url}") from exc
        except error.URLError as exc:
            raise BackendRequestError(f"Request failed for {url}: {exc}") from exc

        if not raw.strip():
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise BackendRequestError(f"Response was not valid JSON for {url}") from exc

    def api_health(self) -> Any:
        return self._request_json(url=f"{self.api_url}/healthz")

    def api_lookup(self, payload: dict[str, Any], *, debug: bool = False) -> Any:
        path = "/debug/lookup" if debug else "/lookup"
        return self._request_json(url=f"{self.api_url}{path}", method="POST", payload=payload)

    def api_openapi(self) -> Any:
        return self._request_json(url=f"{self.api_url}/openapi.json")

    def es_root(self) -> Any:
        return self._request_json(url=self.es_url)

    def es_indices(self) -> Any:
        url = f"{self.es_url}/_cat/indices?format=json&v=true"
        return self._request_json(url=url)

    def es_mapping(self, *, index_name: str = "alpaca-entities") -> Any:
        url = f"{self.es_url}/{parse.quote(index_name)}/_mapping"
        return self._request_json(url=url)

    def es_get_qid(self, qid: str, *, index_name: str = "alpaca-entities") -> Any:
        payload = {"size": 1, "query": {"term": {"qid": qid.casefold()}}}
        url = f"{self.es_url}/{parse.quote(index_name)}/_search"
        return self._request_json(url=url, method="POST", payload=payload)

    def es_custom_search(
        self,
        *,
        payload: dict[str, Any],
        index_name: str = "alpaca-entities",
    ) -> Any:
        url = f"{self.es_url}/{parse.quote(index_name)}/_search"
        return self._request_json(url=url, method="POST", payload=payload)

    def es_search(
        self,
        *,
        query_text: str,
        size: int = 5,
        coarse_type: str = "",
        fine_type: str = "",
        index_name: str = "alpaca-entities",
    ) -> Any:
        must: list[dict[str, Any]] = [
            {
                "multi_match": {
                    "query": query_text,
                    "fields": ["label^5", "labels^3", "aliases^3", "context_string"],
                }
            }
        ]
        filters: list[dict[str, Any]] = []
        if coarse_type:
            filters.append({"term": {"coarse_type": coarse_type.casefold()}})
        if fine_type:
            filters.append({"term": {"fine_type": fine_type.casefold()}})
        payload = {"size": int(size), "query": {"bool": {"must": must, "filter": filters}}}
        url = f"{self.es_url}/{parse.quote(index_name)}/_search"
        return self._request_json(url=url, method="POST", payload=payload)
