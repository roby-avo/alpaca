from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from fastapi import APIRouter, Body, Depends, FastAPI, HTTPException, Security
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .common import resolve_configured_str, resolve_postgres_dsn
from .entity_lookup import EntityLookupService
from .index_postgres_to_elasticsearch import (
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    ELASTICSEARCH_URL_ENV,
    _normalize_es_url,
    default_elasticsearch_url,
)
from .postgres_store import PostgresStore, PostgresStoreError


API_TOKEN_ENV = "ALPACA_API_TOKEN"
API_TOKEN_HASHES_ENV = "ALPACA_API_TOKEN_HASHES"
ES_DEBUG_TIMEOUT_ENV = "ALPACA_ELASTICSEARCH_DEBUG_TIMEOUT_SECONDS"
ES_DEBUG_MAX_BODY_BYTES_ENV = "ALPACA_ELASTICSEARCH_DEBUG_MAX_BODY_BYTES"
ES_DEBUG_MAX_RESPONSE_BYTES_ENV = "ALPACA_ELASTICSEARCH_DEBUG_MAX_RESPONSE_BYTES"
MIN_API_TOKEN_LENGTH = 32
DEFAULT_ES_DEBUG_MAX_BODY_BYTES = 1_048_576
DEFAULT_ES_DEBUG_MAX_RESPONSE_BYTES = 5_242_880
_ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"
_ES_INDEX_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,254}$")
_SHA256_HEX_RE = re.compile(r"^[a-fA-F0-9]{64}$")
_WWW_AUTHENTICATE = {"WWW-Authenticate": 'Bearer realm="alpaca"'}
_bearer_scheme = HTTPBearer(
    scheme_name="BearerAuth",
    description="Enter your Alpaca API access token. Do not include the Bearer prefix.",
    auto_error=False,
)


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _parse_token_hashes(raw_hashes: str) -> tuple[str, ...]:
    token_hashes: list[str] = []
    for raw_hash in raw_hashes.replace("\n", ",").split(","):
        cleaned = raw_hash.strip().lower()
        if not cleaned:
            continue
        if not _SHA256_HEX_RE.match(cleaned):
            raise ValueError(f"{API_TOKEN_HASHES_ENV} must contain SHA-256 hex digests.")
        token_hashes.append(cleaned)
    return tuple(token_hashes)


def _configured_api_token_hashes() -> tuple[str, ...]:
    configured_hashes = os.getenv(API_TOKEN_HASHES_ENV, "").strip()
    if configured_hashes:
        return _parse_token_hashes(configured_hashes)

    raw_token = os.getenv(API_TOKEN_ENV, "").strip()
    if not raw_token:
        return ()
    if len(raw_token) < MIN_API_TOKEN_LENGTH:
        raise ValueError(f"{API_TOKEN_ENV} must be at least {MIN_API_TOKEN_LENGTH} characters.")
    return (_hash_token(raw_token),)


def require_api_token(credentials: HTTPAuthorizationCredentials | None = Security(_bearer_scheme)) -> None:
    try:
        expected_hashes = _configured_api_token_hashes()
    except ValueError as exc:
        raise HTTPException(
            status_code=503,
            detail="API authentication is misconfigured.",
        ) from exc

    if not expected_hashes:
        raise HTTPException(
            status_code=503,
            detail="API authentication is not configured.",
        )
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=401,
            detail="Unauthorized.",
            headers=_WWW_AUTHENTICATE,
        )
    token = credentials.credentials.strip()
    if not token:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized.",
            headers=_WWW_AUTHENTICATE,
        )

    token_hash = _hash_token(token)
    authenticated = False
    for expected_hash in expected_hashes:
        authenticated |= hmac.compare_digest(token_hash, expected_hash)
    if not authenticated:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized.",
            headers=_WWW_AUTHENTICATE,
        )


api_router = APIRouter(dependencies=[Depends(require_api_token)])


class LookupRequest(BaseModel):
    mention: str = Field(..., min_length=1, max_length=512)
    mention_context: str | list[str] | None = Field(default=None)
    crosslink_hints: str | list[str] | None = Field(default=None)
    coarse_hints: list[str] = Field(default_factory=list)
    fine_hints: list[str] = Field(default_factory=list)
    top_k: int = Field(default=20, ge=1, le=100)
    use_cache: bool = True


class LookupCandidate(BaseModel):
    qid: str
    label: str = ""
    labels: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)
    description: str | None = None
    types: list[str] = Field(default_factory=list)
    context_string: str = ""
    coarse_type: str = ""
    fine_type: str = ""
    item_category: str = ""
    popularity: float = 0.0
    score: float = 0.0
    name_score: float = 0.0
    context_score: float = 0.0
    type_score: float = 0.0
    prior_score: float = 0.0
    final_score: float = 0.0


class LookupResponse(BaseModel):
    mention: str
    mention_norm: str
    mention_context_terms: list[str]
    coarse_hints: list[str]
    fine_hints: list[str]
    strategy: str
    returned: int
    cache_hit: bool
    top1: LookupCandidate | None = None


class DebugLookupResponse(LookupResponse):
    top_k: list[LookupCandidate] = Field(default_factory=list)


app = FastAPI(
    title="Alpaca Retrieval API",
    version="0.1.0",
    docs_url=None,
    redoc_url=None,
    description=(
        "Deterministic context-aware entity retrieval API over PostgreSQL "
        "with Postgres-backed entities and query cache."
    ),
)
app.mount("/assets", StaticFiles(directory=_ASSETS_DIR), name="assets")


_SWAGGER_AVATAR_CSS = """
<style>
  .swagger-ui .info {
    min-height: 150px;
    padding-right: 170px;
    position: relative;
  }
  .swagger-ui .info .alpaca-description-avatar {
    position: absolute;
    right: 8px;
    top: 18px;
  }
  .swagger-ui .info .alpaca-description-avatar img {
    background: #fff;
    border: 1px solid #d8dde6;
    border-radius: 50%;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
    display: block;
    height: 124px;
    object-fit: cover;
    width: 124px;
  }
  @media (max-width: 640px) {
    .swagger-ui .info {
      min-height: 0;
      padding-right: 0;
    }
    .swagger-ui .info .alpaca-description-avatar {
      margin-top: 16px;
      position: static;
    }
    .swagger-ui .info .alpaca-description-avatar img {
      height: 96px;
      width: 96px;
    }
  }
</style>
"""

_SWAGGER_AVATAR_SCRIPT = """
<script>
  (function () {
    function mountAlpacaAvatar() {
      var description = document.querySelector(".swagger-ui .info .description");
      if (!description || document.querySelector(".alpaca-description-avatar")) {
        return;
      }
      var wrapper = document.createElement("div");
      wrapper.className = "alpaca-description-avatar";
      var image = document.createElement("img");
      image.src = "/assets/alpaca-avatar.png";
      image.alt = "Alpaca avatar";
      wrapper.appendChild(image);
      description.insertAdjacentElement("afterend", wrapper);
    }

    window.addEventListener("load", mountAlpacaAvatar);
    new MutationObserver(mountAlpacaAvatar).observe(document.body, {
      childList: true,
      subtree: true
    });
  })();
</script>
"""


def _custom_openapi() -> dict[str, Any]:
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    schema.setdefault("info", {})["x-logo"] = {"url": "/assets/alpaca-avatar.png"}
    app.openapi_schema = schema
    return app.openapi_schema


app.openapi = _custom_openapi  # type: ignore[method-assign]


@app.get("/docs", include_in_schema=False)
def swagger_ui_html() -> HTMLResponse:
    html = get_swagger_ui_html(
        openapi_url=app.openapi_url or "/openapi.json",
        title=f"{app.title} - Swagger UI",
        swagger_favicon_url="/assets/alpaca-avatar.png",
        swagger_ui_parameters={
            "displayRequestDuration": True,
            "persistAuthorization": False,
        },
    )
    body = (
        html.body.decode("utf-8")
        .replace("</head>", f"{_SWAGGER_AVATAR_CSS}</head>")
        .replace("</body>", f"{_SWAGGER_AVATAR_SCRIPT}</body>")
    )
    return HTMLResponse(body, status_code=html.status_code)


def get_lookup_service() -> EntityLookupService:
    return EntityLookupService(postgres_dsn=resolve_postgres_dsn(None))


@api_router.get("/healthz")
def healthz() -> dict[str, Any]:
    postgres_healthy = False
    postgres_dsn = resolve_postgres_dsn(None)
    try:
        store = PostgresStore(postgres_dsn)
        store.ping()
        postgres_healthy = True
    except Exception:
        postgres_healthy = False

    status = "ok" if postgres_healthy else "degraded"
    return {
        "status": status,
        "search_backend": "postgres",
        "postgres_healthy": postgres_healthy,
    }


def _coerce_lookup_candidate(raw: Mapping[str, Any]) -> LookupCandidate:
    raw_labels = raw.get("labels")
    labels: list[str] = []
    if isinstance(raw_labels, list):
        labels = [value for value in raw_labels if isinstance(value, str)]
    raw_aliases = raw.get("aliases")
    aliases: list[str] = []
    if isinstance(raw_aliases, list):
        aliases = [value for value in raw_aliases if isinstance(value, str)]
    elif isinstance(raw_aliases, Mapping):
        for values in raw_aliases.values():
            if isinstance(values, list):
                aliases.extend([value for value in values if isinstance(value, str)])
    else:
        # Backward-compatible decode path for cached results produced before/around the rename.
        raw_name_variants = raw.get("name_variants")
        if isinstance(raw_name_variants, list):
            aliases.extend([value for value in raw_name_variants if isinstance(value, str)])
        raw_labels_fallback = raw.get("labels")
        if isinstance(raw_labels_fallback, list):
            aliases.extend([value for value in raw_labels_fallback if isinstance(value, str)])
    raw_types = raw.get("types")
    types: list[str] = []
    if isinstance(raw_types, list):
        types = [value for value in raw_types if isinstance(value, str)]
    return LookupCandidate(
        qid=raw.get("qid") if isinstance(raw.get("qid"), str) else "",
        label=raw.get("label") if isinstance(raw.get("label"), str) else "",
        labels=labels,
        aliases=aliases,
        description=raw.get("description") if isinstance(raw.get("description"), str) else None,
        types=types,
        context_string=raw.get("context_string") if isinstance(raw.get("context_string"), str) else "",
        coarse_type=raw.get("coarse_type") if isinstance(raw.get("coarse_type"), str) else "",
        fine_type=raw.get("fine_type") if isinstance(raw.get("fine_type"), str) else "",
        item_category=raw.get("item_category") if isinstance(raw.get("item_category"), str) else "",
        popularity=float(raw.get("popularity", 0.0))
        if isinstance(raw.get("popularity"), (int, float))
        else 0.0,
        score=float(raw.get("score", 0.0)) if isinstance(raw.get("score"), (int, float)) else 0.0,
        name_score=float(raw.get("name_score", 0.0))
        if isinstance(raw.get("name_score"), (int, float))
        else 0.0,
        context_score=float(raw.get("context_score", 0.0))
        if isinstance(raw.get("context_score"), (int, float))
        else 0.0,
        type_score=float(raw.get("type_score", 0.0))
        if isinstance(raw.get("type_score"), (int, float))
        else 0.0,
        prior_score=float(raw.get("prior_score", 0.0))
        if isinstance(raw.get("prior_score"), (int, float))
        else 0.0,
        final_score=float(raw.get("final_score", 0.0))
        if isinstance(raw.get("final_score"), (int, float))
        else 0.0,
    )


def _run_lookup(request: LookupRequest, *, include_top_k: bool) -> dict[str, Any]:
    service = get_lookup_service()
    try:
        return service.lookup(
            mention=request.mention,
            mention_context=request.mention_context,
            crosslink_hints=request.crosslink_hints,
            coarse_hints=request.coarse_hints,
            fine_hints=request.fine_hints,
            top_k=request.top_k,
            include_top_k=include_top_k,
            use_cache=request.use_cache,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except PostgresStoreError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


def _resolve_float_env(env_var: str, default_value: float) -> float:
    raw = os.getenv(env_var, "").strip()
    if not raw:
        return default_value
    try:
        value = float(raw)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=f"{env_var} must be a number.") from exc
    if value <= 0:
        raise HTTPException(status_code=503, detail=f"{env_var} must be greater than zero.")
    return value


def _resolve_int_env(env_var: str, default_value: int) -> int:
    raw = os.getenv(env_var, "").strip()
    if not raw:
        return default_value
    try:
        value = int(raw)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=f"{env_var} must be an integer.") from exc
    if value <= 0:
        raise HTTPException(status_code=503, detail=f"{env_var} must be greater than zero.")
    return value


def _validate_es_index_name(index_name: str) -> str:
    cleaned = index_name.strip()
    if not _ES_INDEX_RE.match(cleaned):
        raise HTTPException(
            status_code=422,
            detail=(
                "Elasticsearch index name must be lowercase and contain only "
                "letters, digits, dot, underscore, or hyphen."
            ),
        )
    if ".." in cleaned:
        raise HTTPException(status_code=422, detail="Elasticsearch index name cannot contain '..'.")
    return cleaned


def _debug_elasticsearch_search(index_name: str, body: Mapping[str, Any]) -> dict[str, Any]:
    base_url = _normalize_es_url(
        resolve_configured_str(None, ELASTICSEARCH_URL_ENV, default_elasticsearch_url())
    )
    payload = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    max_body_bytes = _resolve_int_env(
        ES_DEBUG_MAX_BODY_BYTES_ENV,
        DEFAULT_ES_DEBUG_MAX_BODY_BYTES,
    )
    if len(payload) > max_body_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Elasticsearch debug query body exceeds {max_body_bytes} bytes.",
        )

    timeout_seconds = _resolve_float_env(
        ES_DEBUG_TIMEOUT_ENV,
        float(DEFAULT_REQUEST_TIMEOUT_SECONDS),
    )
    request = Request(
        f"{base_url}/{index_name}/_search",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    max_response_bytes = _resolve_int_env(
        ES_DEBUG_MAX_RESPONSE_BYTES_ENV,
        DEFAULT_ES_DEBUG_MAX_RESPONSE_BYTES,
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            response_body = response.read(max_response_bytes + 1)
    except HTTPError as exc:
        detail_body = exc.read(max_response_bytes + 1).decode("utf-8", errors="replace")
        status_code = int(exc.code) if 400 <= int(exc.code) < 500 else 502
        raise HTTPException(status_code=status_code, detail=detail_body[:max_response_bytes]) from exc
    except URLError as exc:
        raise HTTPException(status_code=502, detail=f"Elasticsearch request failed: {exc.reason}") from exc

    if len(response_body) > max_response_bytes:
        raise HTTPException(
            status_code=502,
            detail=f"Elasticsearch response exceeds {max_response_bytes} bytes.",
        )
    if not response_body:
        return {}
    try:
        parsed = json.loads(response_body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=502, detail="Elasticsearch returned invalid JSON.") from exc
    if not isinstance(parsed, dict):
        raise HTTPException(status_code=502, detail="Elasticsearch returned a non-object JSON response.")
    return parsed


@api_router.post("/lookup", response_model=LookupResponse)
def lookup_entity(request: LookupRequest) -> LookupResponse:
    raw = _run_lookup(request, include_top_k=False)
    top1_raw = raw.get("top1")
    top1 = _coerce_lookup_candidate(top1_raw) if isinstance(top1_raw, Mapping) else None
    return LookupResponse(
        mention=raw.get("mention") if isinstance(raw.get("mention"), str) else request.mention,
        mention_norm=raw.get("mention_norm") if isinstance(raw.get("mention_norm"), str) else "",
        mention_context_terms=raw.get("mention_context_terms")
        if isinstance(raw.get("mention_context_terms"), list)
        else [],
        coarse_hints=raw.get("coarse_hints") if isinstance(raw.get("coarse_hints"), list) else [],
        fine_hints=raw.get("fine_hints") if isinstance(raw.get("fine_hints"), list) else [],
        strategy=raw.get("strategy") if isinstance(raw.get("strategy"), str) else "unknown",
        returned=int(raw.get("returned")) if isinstance(raw.get("returned"), int) else 0,
        cache_hit=bool(raw.get("cache_hit")),
        top1=top1,
    )


@api_router.post("/debug/lookup", response_model=DebugLookupResponse)
def debug_lookup_entity(request: LookupRequest) -> DebugLookupResponse:
    raw = _run_lookup(request, include_top_k=True)
    top1_raw = raw.get("top1")
    top1 = _coerce_lookup_candidate(top1_raw) if isinstance(top1_raw, Mapping) else None
    top_k_items = raw.get("top_k")
    top_k: list[LookupCandidate] = []
    if isinstance(top_k_items, list):
        for item in top_k_items:
            if isinstance(item, Mapping):
                top_k.append(_coerce_lookup_candidate(item))
    return DebugLookupResponse(
        mention=raw.get("mention") if isinstance(raw.get("mention"), str) else request.mention,
        mention_norm=raw.get("mention_norm") if isinstance(raw.get("mention_norm"), str) else "",
        mention_context_terms=raw.get("mention_context_terms")
        if isinstance(raw.get("mention_context_terms"), list)
        else [],
        coarse_hints=raw.get("coarse_hints") if isinstance(raw.get("coarse_hints"), list) else [],
        fine_hints=raw.get("fine_hints") if isinstance(raw.get("fine_hints"), list) else [],
        strategy=raw.get("strategy") if isinstance(raw.get("strategy"), str) else "unknown",
        returned=int(raw.get("returned")) if isinstance(raw.get("returned"), int) else 0,
        cache_hit=bool(raw.get("cache_hit")),
        top1=top1,
        top_k=top_k,
    )


@api_router.post("/debug/elasticsearch/{index_name}/_search")
def debug_elasticsearch_search(
    index_name: str,
    body: dict[str, Any] = Body(
        ...,
        examples=[
            {
                "query": {
                    "multi_match": {
                        "query": "Rome",
                        "fields": ["label^4", "labels^2", "aliases", "context_string"],
                    }
                },
                "size": 5,
            }
        ],
    ),
) -> dict[str, Any]:
    cleaned_index_name = _validate_es_index_name(index_name)
    return _debug_elasticsearch_search(cleaned_index_name, body)


app.include_router(api_router)
