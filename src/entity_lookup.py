from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .build_elasticsearch_index import normalize_exact_text, popularity_to_prior
from .common import tokenize
from .elasticsearch_client import ElasticsearchClient, ElasticsearchClientError
from .postgres_store import PostgresStore
from .search_logic import normalize_type_labels


DEFAULT_FUZZY_TOPK = 20


@dataclass(frozen=True, slots=True)
class LookupWeights:
    name: float = 0.62
    context: float = 0.23
    type: float = 0.10
    prior: float = 0.05


def normalize_context_inputs(mention_context: str | Sequence[str] | None) -> list[str]:
    if mention_context is None:
        return []
    if isinstance(mention_context, str):
        values = [mention_context]
    else:
        values = [item for item in mention_context if isinstance(item, str)]
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        stripped = value.strip()
        if not stripped:
            continue
        for token in tokenize(stripped):
            if token not in seen:
                seen.add(token)
                normalized.append(token)
    return normalized


def build_cache_key(
    *,
    mention_norm: str,
    context_terms: Sequence[str],
    coarse_hints: Sequence[str],
    fine_hints: Sequence[str],
    limit: int,
    include_top_k: bool,
) -> str:
    payload = {
        "mention_norm": mention_norm,
        "context_terms": list(context_terms),
        "coarse_hints": list(coarse_hints),
        "fine_hints": list(fine_hints),
        "limit": int(limit),
        "include_top_k": bool(include_top_k),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _safe_mapping(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        return {}
    return {str(k): v for k, v in raw.items() if isinstance(k, str)}


def _safe_str_list(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, str)]
    if isinstance(raw, tuple):
        return [item for item in raw if isinstance(item, str)]
    return []


def _extract_es_hits(response: Mapping[str, Any]) -> list[dict[str, Any]]:
    hits_root = response.get("hits")
    if not isinstance(hits_root, Mapping):
        return []
    hits = hits_root.get("hits")
    if not isinstance(hits, list):
        return []
    normalized: list[dict[str, Any]] = []
    for raw_hit in hits:
        if not isinstance(raw_hit, Mapping):
            continue
        source = raw_hit.get("_source")
        if not isinstance(source, Mapping):
            continue
        qid = source.get("qid")
        if not isinstance(qid, str) or not qid:
            continue
        score = raw_hit.get("_score", 0.0)
        try:
            score_value = float(score)
        except (TypeError, ValueError):
            score_value = 0.0
        normalized.append(
            {
                "qid": qid,
                "label": source.get("label") if isinstance(source.get("label"), str) else "",
                "labels": _safe_str_list(source.get("labels")),
                "aliases": _safe_str_list(source.get("aliases")),
                "context_string": source.get("context_string")
                if isinstance(source.get("context_string"), str)
                else "",
                "coarse_type": source.get("coarse_type")
                if isinstance(source.get("coarse_type"), str)
                else "",
                "fine_type": source.get("fine_type") if isinstance(source.get("fine_type"), str) else "",
                "popularity": float(source.get("popularity", 0.0))
                if isinstance(source.get("popularity"), (int, float))
                else 0.0,
                "prior": float(source.get("prior", 0.0))
                if isinstance(source.get("prior"), (int, float))
                else 0.0,
                "score": score_value,
            }
        )
    return normalized


def _normalize_score_range(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    max_v = max(values)
    min_v = min(values)
    if max_v <= min_v:
        return [1.0 if max_v > 0 else 0.0 for _ in values]
    return [(value - min_v) / (max_v - min_v) for value in values]


def _context_score(context_string: str, context_terms: set[str]) -> float:
    if not context_terms:
        return 0.0
    doc_terms = set(tokenize(context_string))
    if not doc_terms:
        return 0.0
    overlap = len(doc_terms & context_terms)
    return overlap / max(1, len(context_terms))


def _type_score(
    coarse_type: str,
    fine_type: str,
    *,
    coarse_hints: set[str],
    fine_hints: set[str],
) -> float:
    if fine_hints and fine_type in fine_hints:
        return 1.0
    if coarse_hints and coarse_type in coarse_hints:
        return 0.5
    return 0.0


def _dedupe_candidates(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        qid = candidate.get("qid")
        if not isinstance(qid, str) or qid in seen:
            continue
        seen.add(qid)
        deduped.append(dict(candidate))
    return deduped


def rerank_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    mention_norm: str,
    context_terms: Sequence[str],
    coarse_hints: Sequence[str],
    fine_hints: Sequence[str],
    weights: LookupWeights | None = None,
    exact_mode: bool,
    limit: int,
) -> list[dict[str, Any]]:
    active_weights = weights or LookupWeights()
    deduped = _dedupe_candidates(candidates)
    if not deduped:
        return []

    raw_scores = [
        float(candidate.get("score", 0.0))
        if isinstance(candidate.get("score"), (int, float))
        else 0.0
        for candidate in deduped
    ]
    normalized_name_scores = _normalize_score_range(raw_scores)
    context_term_set = set(context_terms)
    coarse_hint_set = set(coarse_hints)
    fine_hint_set = set(fine_hints)

    scored: list[dict[str, Any]] = []
    for candidate, name_score in zip(deduped, normalized_name_scores, strict=False):
        context_score = _context_score(
            candidate.get("context_string", "") if isinstance(candidate.get("context_string"), str) else "",
            context_term_set,
        )
        candidate_coarse = (
            candidate.get("coarse_type") if isinstance(candidate.get("coarse_type"), str) else ""
        )
        candidate_fine = candidate.get("fine_type") if isinstance(candidate.get("fine_type"), str) else ""
        type_score = _type_score(
            candidate_coarse,
            candidate_fine,
            coarse_hints=coarse_hint_set,
            fine_hints=fine_hint_set,
        )
        prior_value = (
            float(candidate.get("prior"))
            if isinstance(candidate.get("prior"), (int, float))
            else popularity_to_prior(
                float(candidate.get("popularity", 0.0))
                if isinstance(candidate.get("popularity"), (int, float))
                else 0.0
            )
        )

        label = candidate.get("label") if isinstance(candidate.get("label"), str) else ""
        aliases = candidate.get("aliases")
        alias_values: list[str] = []
        if isinstance(aliases, list):
            alias_values.extend([item for item in aliases if isinstance(item, str)])
        elif isinstance(aliases, Mapping):
            # Backward-compatible with older cached candidate shapes before aliases were flattened.
            for values in aliases.values():
                if isinstance(values, list):
                    alias_values.extend([item for item in values if isinstance(item, str)])
        exact_name_match = normalize_exact_text(label) == mention_norm or mention_norm in {
            normalize_exact_text(value) for value in alias_values
        }
        if exact_mode and exact_name_match:
            # For exact candidate sets, lexical equality is already satisfied; use a flat
            # name score so context/type/prior drive deterministic disambiguation.
            name_score = 1.0

        final_score = (
            active_weights.name * name_score
            + active_weights.context * context_score
            + active_weights.type * type_score
            + active_weights.prior * prior_value
        )
        if exact_mode and exact_name_match:
            final_score += 0.05

        scored.append(
            {
                **candidate,
                "name_score": name_score,
                "context_score": context_score,
                "type_score": type_score,
                "prior_score": prior_value,
                "exact_name_match": exact_name_match,
                "final_score": final_score,
            }
        )

    scored.sort(
        key=lambda item: (
            float(item.get("final_score", 0.0)),
            1 if item.get("exact_name_match") else 0,
            float(item.get("prior_score", 0.0)),
            item.get("qid") if isinstance(item.get("qid"), str) else "",
        ),
        reverse=True,
    )
    # Fix deterministic qid tie-break (ascending) after reverse sort.
    scored.sort(
        key=lambda item: (
            -float(item.get("final_score", 0.0)),
            -(1 if item.get("exact_name_match") else 0),
            -float(item.get("prior_score", 0.0)),
            item.get("qid") if isinstance(item.get("qid"), str) else "",
        )
    )
    return scored[:limit]


class EntityLookupService:
    def __init__(
        self,
        *,
        postgres_dsn: str,
        elasticsearch_url: str,
        index_name: str,
        http_timeout_seconds: float = 10.0,
    ) -> None:
        self.store = PostgresStore(postgres_dsn)
        self.es = ElasticsearchClient(elasticsearch_url, timeout_seconds=http_timeout_seconds)
        self.index_name = index_name

    def _type_filters(
        self,
        coarse_hints: Sequence[str],
        fine_hints: Sequence[str],
    ) -> list[dict[str, Any]]:
        filters: list[dict[str, Any]] = []
        if coarse_hints:
            filters.append({"terms": {"coarse_type": list(coarse_hints)}})
        if fine_hints:
            filters.append({"terms": {"fine_type": list(fine_hints)}})
        return filters

    def _exact_search(
        self,
        mention_norm: str,
        *,
        coarse_hints: Sequence[str],
        fine_hints: Sequence[str],
        size: int,
    ) -> list[dict[str, Any]]:
        if not mention_norm:
            return []
        query: dict[str, Any] = {
            "size": size,
            "track_scores": True,
            "sort": ["_score", {"prior": {"order": "desc"}}, {"qid": {"order": "asc"}}],
            "_source": True,
            "query": {
                "bool": {
                    "should": [
                        {"term": {"label.exact": {"value": mention_norm, "boost": 6.0}}},
                        {"term": {"aliases.exact": {"value": mention_norm, "boost": 4.0}}},
                    ],
                    "minimum_should_match": 1,
                    "filter": self._type_filters(coarse_hints, fine_hints),
                }
            },
        }
        response = self.es.search(self.index_name, query)
        return _extract_es_hits(response)

    def _fuzzy_search(
        self,
        mention: str,
        *,
        context_terms: Sequence[str],
        coarse_hints: Sequence[str],
        fine_hints: Sequence[str],
        size: int,
    ) -> list[dict[str, Any]]:
        should: list[dict[str, Any]] = [
            {
                "multi_match": {
                    "query": mention,
                    "fields": ["label^5", "aliases^3"],
                    "type": "best_fields",
                    "fuzziness": "AUTO",
                    "prefix_length": 1,
                    "operator": "and",
                }
            },
            {"match": {"label": {"query": mention, "boost": 4.0}}},
            {"match": {"aliases": {"query": mention, "boost": 2.0}}},
        ]
        if context_terms:
            should.append(
                {
                    "match": {
                        "context_string": {
                            "query": " ".join(context_terms),
                            "boost": 1.5,
                        }
                    }
                }
            )

        query: dict[str, Any] = {
            "size": size,
            "track_scores": True,
            "sort": ["_score", {"prior": {"order": "desc"}}, {"qid": {"order": "asc"}}],
            "_source": True,
            "query": {
                "bool": {
                    "should": should,
                    "minimum_should_match": 1,
                    "filter": self._type_filters(coarse_hints, fine_hints),
                }
            },
        }
        response = self.es.search(self.index_name, query)
        return _extract_es_hits(response)

    def lookup(
        self,
        *,
        mention: str,
        mention_context: str | Sequence[str] | None = None,
        coarse_hints: Sequence[str] | None = None,
        fine_hints: Sequence[str] | None = None,
        top_k: int = DEFAULT_FUZZY_TOPK,
        include_top_k: bool = False,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        mention_value = mention.strip()
        if not mention_value:
            raise ValueError("mention must be non-empty")
        mention_norm = normalize_exact_text(mention_value)
        if not mention_norm:
            raise ValueError("mention must contain at least one alphanumeric character")

        normalized_coarse = normalize_type_labels(coarse_hints, field_name="coarse_hints")
        normalized_fine = normalize_type_labels(fine_hints, field_name="fine_hints")
        context_terms = normalize_context_inputs(mention_context)
        limit = max(1, min(100, int(top_k)))

        cache_key = build_cache_key(
            mention_norm=mention_norm,
            context_terms=context_terms,
            coarse_hints=normalized_coarse,
            fine_hints=normalized_fine,
            limit=limit,
            include_top_k=include_top_k,
        )

        if use_cache:
            cached = self.store.get_query_cache(cache_key)
            if isinstance(cached, Mapping):
                response = dict(cached)
                response["cache_hit"] = True
                return response

        exact_hits = self._exact_search(
            mention_norm,
            coarse_hints=normalized_coarse,
            fine_hints=normalized_fine,
            size=max(10, limit),
        )
        strategy = "exact"
        if len(exact_hits) == 1:
            ranked = rerank_candidates(
                exact_hits,
                mention_norm=mention_norm,
                context_terms=context_terms,
                coarse_hints=normalized_coarse,
                fine_hints=normalized_fine,
                exact_mode=True,
                limit=1,
            )
        elif len(exact_hits) > 1:
            strategy = "exact_disambiguated"
            ranked = rerank_candidates(
                exact_hits,
                mention_norm=mention_norm,
                context_terms=context_terms,
                coarse_hints=normalized_coarse,
                fine_hints=normalized_fine,
                exact_mode=True,
                limit=limit,
            )
        else:
            strategy = "fuzzy"
            fuzzy_hits = self._fuzzy_search(
                mention_value,
                context_terms=context_terms,
                coarse_hints=normalized_coarse,
                fine_hints=normalized_fine,
                size=max(limit, DEFAULT_FUZZY_TOPK),
            )
            ranked = rerank_candidates(
                fuzzy_hits,
                mention_norm=mention_norm,
                context_terms=context_terms,
                coarse_hints=normalized_coarse,
                fine_hints=normalized_fine,
                exact_mode=False,
                limit=limit,
            )

        top1 = ranked[0] if ranked else None
        response: dict[str, Any] = {
            "mention": mention_value,
            "mention_norm": mention_norm,
            "mention_context_terms": context_terms,
            "coarse_hints": normalized_coarse,
            "fine_hints": normalized_fine,
            "strategy": strategy,
            "returned": len(ranked),
            "top1": top1,
            "cache_hit": False,
        }
        if include_top_k:
            response["top_k"] = ranked

        if use_cache:
            self.store.put_query_cache(cache_key, response)
        return response
