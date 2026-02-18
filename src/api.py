from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .common import resolve_quickwit_index_id, resolve_quickwit_url
from .quickwit_client import QuickwitClient, QuickwitClientError
from .search_logic import build_quickwit_query, rerank_hits_by_context


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=512)
    context: str | None = Field(default=None, max_length=2048)
    ner_coarse_types: list[str] = Field(default_factory=list)
    ner_fine_types: list[str] = Field(default_factory=list)
    limit: int = Field(default=20, ge=1, le=100)


class EntityHit(BaseModel):
    id: str
    labels: dict[str, str] = Field(default_factory=dict)
    aliases: dict[str, list[str]] = Field(default_factory=dict)
    name_text: str
    context: str = ""
    bow: str
    coarse_type: str = ""
    fine_type: str = ""
    score: float
    context_overlap: float


class SearchResponse(BaseModel):
    query: str
    context: str | None
    ner_coarse_types: list[str]
    ner_fine_types: list[str]
    quickwit_query: str
    returned: int
    num_hits: int
    hits: list[EntityHit]


app = FastAPI(
    title="alpaca retrieval api",
    version="0.1.0",
    description=(
        "Context-aware and type-aware entity retrieval API over Quickwit "
        "using Wikidata-derived name/description features and NER coarse/fine types."
    ),
)


def get_quickwit_client() -> tuple[QuickwitClient, str]:
    quickwit_url = resolve_quickwit_url(None)
    index_id = resolve_quickwit_index_id(None)
    return QuickwitClient(quickwit_url), index_id


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    client, index_id = get_quickwit_client()
    healthy = client.is_healthy()
    return {
        "status": "ok" if healthy else "degraded",
        "quickwit_url": client.base_url,
        "index_id": index_id,
        "quickwit_healthy": healthy,
    }


@app.post("/v1/entities/search", response_model=SearchResponse)
def search_entities(request: SearchRequest) -> SearchResponse:
    try:
        quickwit_query = build_quickwit_query(
            request.query,
            request.ner_coarse_types,
            request.ner_fine_types,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    client, index_id = get_quickwit_client()
    fetch_size = min(500, max(request.limit * 5, request.limit))
    payload = {
        "query": quickwit_query,
        "max_hits": fetch_size,
    }

    try:
        quickwit_response = client.search(index_id=index_id, payload=payload)
    except QuickwitClientError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    raw_hits: list[Mapping[str, Any]] = []
    if isinstance(quickwit_response, Mapping):
        response_hits = quickwit_response.get("hits")
        if isinstance(response_hits, list):
            raw_hits = [item for item in response_hits if isinstance(item, Mapping)]

    ranked = rerank_hits_by_context(raw_hits, request.context, limit=request.limit)
    hits = [
        EntityHit(
            id=item["id"],
            labels=item["labels"],
            aliases=item["aliases"],
            name_text=item["name_text"],
            context=item["context"],
            bow=item["bow"],
            coarse_type=item["coarse_type"],
            fine_type=item["fine_type"],
            score=item["score"],
            context_overlap=item["context_overlap"],
        )
        for item in ranked
    ]

    num_hits = len(raw_hits)
    if isinstance(quickwit_response, Mapping):
        reported_num_hits = quickwit_response.get("num_hits")
        if isinstance(reported_num_hits, int):
            num_hits = reported_num_hits

    return SearchResponse(
        query=request.query,
        context=request.context,
        ner_coarse_types=request.ner_coarse_types,
        ner_fine_types=request.ner_fine_types,
        quickwit_query=quickwit_query,
        returned=len(hits),
        num_hits=num_hits,
        hits=hits,
    )
