from __future__ import annotations

import json
import socket
from dataclasses import dataclass
from typing import Any
from urllib import error, parse, request


@dataclass(slots=True)
class QuickwitClientError(RuntimeError):
    message: str
    status_code: int | None = None

    def __str__(self) -> str:
        if self.status_code is None:
            return self.message
        return f"HTTP {self.status_code}: {self.message}"


class QuickwitClient:
    def __init__(self, base_url: str, timeout_seconds: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def list_indexes(self) -> Any:
        return self._request_json("GET", "/api/v1/indexes")

    def is_healthy(self) -> bool:
        try:
            self.list_indexes()
            return True
        except QuickwitClientError:
            return False

    def ensure_index(self, index_id: str, index_config: dict[str, Any]) -> Any:
        # Quickwit index creation endpoints differ across releases.
        # Try the modern create endpoint first, then fall back.
        try:
            return self._request_json("POST", "/api/v1/indexes", payload=index_config)
        except QuickwitClientError as exc:
            if exc.status_code == 409:
                # Index already exists.
                return {}
            if exc.status_code not in {400, 404, 405}:
                raise

        safe_index_id = parse.quote(index_id, safe="")
        legacy_path = f"/api/v1/indexes/{safe_index_id}?create=true"
        try:
            return self._request_json("PUT", legacy_path, payload=index_config)
        except QuickwitClientError as exc:
            if exc.status_code == 409:
                return {}
            raise

    def ingest_ndjson(self, index_id: str, ndjson_payload: str, commit: str = "auto") -> Any:
        safe_index_id = parse.quote(index_id, safe="")
        safe_commit = parse.quote(commit, safe="")
        path = f"/api/v1/{safe_index_id}/ingest?commit={safe_commit}"
        return self._request_json(
            "POST",
            path,
            payload=ndjson_payload.encode("utf-8"),
            content_type="application/x-ndjson",
        )

    def search(self, index_id: str, payload: dict[str, Any]) -> Any:
        safe_index_id = parse.quote(index_id, safe="")
        path = f"/api/v1/{safe_index_id}/search"
        return self._request_json("POST", path, payload=payload)

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
        if body is not None:
            req.add_header("Content-Type", content_type)
        req.add_header("Accept", "application/json")

        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                raw = response.read()
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise QuickwitClientError(detail or exc.reason, status_code=exc.code) from exc
        except (TimeoutError, socket.timeout) as exc:
            raise QuickwitClientError(
                f"Request to Quickwit at {endpoint} timed out after {self.timeout_seconds:.1f}s"
            ) from exc
        except error.URLError as exc:
            reason = exc.reason if isinstance(exc.reason, str) else repr(exc.reason)
            raise QuickwitClientError(f"Could not reach Quickwit at {self.base_url}: {reason}") from exc

        if not raw:
            return {}

        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            text = raw.decode("utf-8", errors="replace")
            raise QuickwitClientError(
                f"Received non-JSON response from Quickwit endpoint {endpoint}: {text[:400]}"
            ) from exc
