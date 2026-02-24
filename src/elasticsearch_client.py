from __future__ import annotations

import json
import socket
from dataclasses import dataclass
from typing import Any
from urllib import error, parse, request


@dataclass(slots=True)
class ElasticsearchClientError(RuntimeError):
    message: str
    status_code: int | None = None

    def __str__(self) -> str:
        if self.status_code is None:
            return self.message
        return f"HTTP {self.status_code}: {self.message}"


class ElasticsearchClient:
    def __init__(self, base_url: str, timeout_seconds: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def is_healthy(self) -> bool:
        try:
            response = self._request_json("GET", "/_cluster/health")
        except ElasticsearchClientError:
            return False
        return isinstance(response, dict)

    def index_exists(self, index_name: str) -> bool:
        safe = parse.quote(index_name, safe="")
        try:
            self._request_json("HEAD", f"/{safe}")
            return True
        except ElasticsearchClientError as exc:
            if exc.status_code == 404:
                return False
            raise

    def delete_index(self, index_name: str) -> None:
        safe = parse.quote(index_name, safe="")
        try:
            self._request_json("DELETE", f"/{safe}")
        except ElasticsearchClientError as exc:
            if exc.status_code == 404:
                return
            raise

    def create_index(self, index_name: str, config: dict[str, Any]) -> None:
        safe = parse.quote(index_name, safe="")
        self._request_json("PUT", f"/{safe}", payload=config)

    def ensure_index(self, index_name: str, config: dict[str, Any], *, replace: bool = False) -> None:
        if replace:
            self.delete_index(index_name)
            self.create_index(index_name, config)
            return
        try:
            self.create_index(index_name, config)
        except ElasticsearchClientError as exc:
            if exc.status_code == 400 and "resource_already_exists_exception" in exc.message:
                return
            raise

    def bulk(self, index_name: str, ndjson_payload: str, *, refresh: bool = False) -> dict[str, Any]:
        safe = parse.quote(index_name, safe="")
        refresh_value = "true" if refresh else "false"
        response = self._request_json(
            "POST",
            f"/{safe}/_bulk?refresh={refresh_value}",
            payload=ndjson_payload.encode("utf-8"),
            content_type="application/x-ndjson",
        )
        if not isinstance(response, dict):
            raise ElasticsearchClientError("Elasticsearch bulk response was not a JSON object.")
        if response.get("errors") is True:
            items = response.get("items")
            detail = "Bulk indexing returned item errors."
            if isinstance(items, list):
                failed = 0
                sample_reasons: list[str] = []
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    index_result = item.get("index") or item.get("create") or item.get("update")
                    if not isinstance(index_result, dict):
                        continue
                    status = index_result.get("status")
                    if isinstance(status, int) and status < 300:
                        continue
                    failed += 1
                    error_payload = index_result.get("error")
                    if isinstance(error_payload, dict):
                        reason = error_payload.get("reason")
                        err_type = error_payload.get("type")
                        if isinstance(reason, str):
                            sample_reasons.append(
                                f"{err_type}: {reason}" if isinstance(err_type, str) else reason
                            )
                    if len(sample_reasons) >= 3:
                        break
                if failed:
                    detail = f"Bulk indexing had {failed} failed items. {' | '.join(sample_reasons)}"
            raise ElasticsearchClientError(detail)
        return response

    def refresh_index(self, index_name: str) -> None:
        safe = parse.quote(index_name, safe="")
        self._request_json("POST", f"/{safe}/_refresh")

    def search(self, index_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        safe = parse.quote(index_name, safe="")
        response = self._request_json("POST", f"/{safe}/_search", payload=payload)
        if not isinstance(response, dict):
            raise ElasticsearchClientError("Elasticsearch search response was not a JSON object.")
        return response

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | bytes | None = None,
        content_type: str = "application/json",
    ) -> Any:
        body: bytes | None
        if payload is None:
            body = None
        elif isinstance(payload, bytes):
            body = payload
        else:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        endpoint = f"{self.base_url}{path}"
        req = request.Request(url=endpoint, method=method, data=body)
        req.add_header("Accept", "application/json")
        if body is not None:
            req.add_header("Content-Type", content_type)

        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                raw = response.read()
                status = response.status
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ElasticsearchClientError(detail or exc.reason, status_code=exc.code) from exc
        except (TimeoutError, socket.timeout) as exc:
            raise ElasticsearchClientError(
                f"Request to Elasticsearch at {endpoint} timed out after {self.timeout_seconds:.1f}s"
            ) from exc
        except error.URLError as exc:
            reason = exc.reason if isinstance(exc.reason, str) else repr(exc.reason)
            raise ElasticsearchClientError(
                f"Could not reach Elasticsearch at {self.base_url}: {reason}"
            ) from exc

        if method == "HEAD":
            return {"status": status}
        if not raw:
            return {}

        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            text = raw.decode("utf-8", errors="replace")
            raise ElasticsearchClientError(
                f"Received non-JSON response from Elasticsearch endpoint {endpoint}: {text[:400]}"
            ) from exc
