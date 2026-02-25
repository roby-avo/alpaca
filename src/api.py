from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .common import resolve_postgres_dsn
from .entity_lookup import EntityLookupService
from .postgres_store import PostgresStore, PostgresStoreError


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
    aliases: list[str] = Field(default_factory=list)
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


class ReindexRequest(BaseModel):
    ensure_search_indexes: bool = True


app = FastAPI(
    title="alpaca retrieval api",
    version="0.1.0",
    description=(
        "Deterministic context-aware entity retrieval API over PostgreSQL "
        "with Postgres-backed entities and query cache."
    ),
)


@app.on_event("startup")
def startup_init() -> None:
    store = PostgresStore(resolve_postgres_dsn(None))
    store.ensure_schema()
    store.ensure_search_indexes()


def get_lookup_service() -> EntityLookupService:
    return EntityLookupService(postgres_dsn=resolve_postgres_dsn(None))


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    postgres_healthy = False
    postgres_dsn = resolve_postgres_dsn(None)
    entity_count: int | None = None
    try:
        store = PostgresStore(postgres_dsn)
        store.ensure_schema()
        postgres_healthy = True
        entity_count = store.count_entities()
    except Exception:
        postgres_healthy = False

    status = "ok" if postgres_healthy else "degraded"
    return {
        "status": status,
        "search_backend": "postgres",
        "postgres_dsn": postgres_dsn,
        "postgres_healthy": postgres_healthy,
        "entities": entity_count,
    }


def _coerce_lookup_candidate(raw: Mapping[str, Any]) -> LookupCandidate:
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
        labels = raw.get("labels")
        if isinstance(labels, list):
            aliases.extend([value for value in labels if isinstance(value, str)])
    return LookupCandidate(
        qid=raw.get("qid") if isinstance(raw.get("qid"), str) else "",
        label=raw.get("label") if isinstance(raw.get("label"), str) else "",
        aliases=aliases,
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
        store = PostgresStore(resolve_postgres_dsn(None))
        store.ensure_schema()
        if request.ensure_search_indexes:
            store.ensure_search_indexes()
        status = 0
    except (PostgresStoreError, ValueError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return {
        "status": "ok" if status == 0 else "error",
        "exit_code": status,
        "backend": "postgres",
        "search_indexes_ensured": bool(request.ensure_search_indexes),
    }
