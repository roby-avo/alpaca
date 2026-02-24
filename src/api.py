from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .build_elasticsearch_index import run as run_elasticsearch_reindex
from .common import (
    resolve_elasticsearch_index,
    resolve_elasticsearch_url,
    resolve_postgres_dsn,
)
from .elasticsearch_client import ElasticsearchClient, ElasticsearchClientError
from .entity_lookup import EntityLookupService
from .postgres_store import PostgresStoreError


class LookupRequest(BaseModel):
    mention: str = Field(..., min_length=1, max_length=512)
    mention_context: str | list[str] | None = Field(default=None)
    coarse_hints: list[str] = Field(default_factory=list)
    fine_hints: list[str] = Field(default_factory=list)
    top_k: int = Field(default=20, ge=1, le=100)
    use_cache: bool = True


class LookupCandidate(BaseModel):
    qid: str
    label: str = ""
    labels: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)
    context_string: str = ""
    coarse_type: str = ""
    fine_type: str = ""
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


class ReindexRequest(BaseModel):
    replace_index: bool = True
    fetch_batch_size: int = Field(default=2000, ge=1, le=20000)
    chunk_bytes: int = Field(default=8_000_000, ge=1024)
    workers: int = Field(default=4, ge=1, le=64)
    wait_timeout_seconds: float = Field(default=120.0, ge=0.0)
    poll_interval_seconds: float = Field(default=2.0, ge=0.0)
    http_timeout_seconds: float = Field(default=120.0, gt=0.0)


app = FastAPI(
    title="alpaca retrieval api",
    version="0.1.0",
    description=(
        "Deterministic context-aware entity retrieval API over Elasticsearch "
        "with Postgres-backed entities and query cache."
    ),
)


def get_elasticsearch_client() -> tuple[ElasticsearchClient, str]:
    es_url = resolve_elasticsearch_url(None)
    index_name = resolve_elasticsearch_index(None)
    return ElasticsearchClient(es_url), index_name


def get_lookup_service() -> EntityLookupService:
    return EntityLookupService(
        postgres_dsn=resolve_postgres_dsn(None),
        elasticsearch_url=resolve_elasticsearch_url(None),
        index_name=resolve_elasticsearch_index(None),
    )


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    elasticsearch_healthy = False
    es_url = ""
    es_index = ""
    try:
        es_client, es_index = get_elasticsearch_client()
        es_url = es_client.base_url
        elasticsearch_healthy = es_client.is_healthy()
    except Exception:
        elasticsearch_healthy = False

    status = "ok" if elasticsearch_healthy else "degraded"
    return {
        "status": status,
        "search_backend": "elasticsearch",
        "elasticsearch_url": es_url,
        "elasticsearch_index": es_index,
        "elasticsearch_healthy": elasticsearch_healthy,
    }


def _coerce_lookup_candidate(raw: Mapping[str, Any]) -> LookupCandidate:
    return LookupCandidate(
        qid=raw.get("qid") if isinstance(raw.get("qid"), str) else "",
        label=raw.get("label") if isinstance(raw.get("label"), str) else "",
        labels=[value for value in raw.get("labels", []) if isinstance(value, str)]
        if isinstance(raw.get("labels"), list)
        else [],
        aliases=[value for value in raw.get("aliases", []) if isinstance(value, str)]
        if isinstance(raw.get("aliases"), list)
        else [],
        context_string=raw.get("context_string") if isinstance(raw.get("context_string"), str) else "",
        coarse_type=raw.get("coarse_type") if isinstance(raw.get("coarse_type"), str) else "",
        fine_type=raw.get("fine_type") if isinstance(raw.get("fine_type"), str) else "",
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
            coarse_hints=request.coarse_hints,
            fine_hints=request.fine_hints,
            top_k=request.top_k,
            include_top_k=include_top_k,
            use_cache=request.use_cache,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (ElasticsearchClientError, PostgresStoreError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/lookup", response_model=LookupResponse)
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


@app.post("/debug/lookup", response_model=DebugLookupResponse)
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


@app.post("/admin/reindex")
def admin_reindex(request: ReindexRequest) -> dict[str, Any]:
    try:
        status = run_elasticsearch_reindex(
            postgres_dsn=resolve_postgres_dsn(None),
            elasticsearch_url=resolve_elasticsearch_url(None),
            index_name=resolve_elasticsearch_index(None),
            replace_index=request.replace_index,
            fetch_batch_size=request.fetch_batch_size,
            chunk_bytes=request.chunk_bytes,
            worker_count=request.workers,
            wait_timeout_seconds=request.wait_timeout_seconds,
            poll_interval_seconds=request.poll_interval_seconds,
            http_timeout_seconds=request.http_timeout_seconds,
        )
    except (ElasticsearchClientError, PostgresStoreError, ValueError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return {
        "status": "ok" if status == 0 else "error",
        "exit_code": status,
        "backend": "elasticsearch",
        "index_name": resolve_elasticsearch_index(None),
    }
