from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .common import tokenize
from .postgres_store import PostgresStore, compact_crosslink_hint
from .search_logic import normalize_type_labels


DEFAULT_FUZZY_TOPK = 20


@dataclass(frozen=True, slots=True)
class LookupWeights:
    name: float = 0.62
    context: float = 0.23
    type: float = 0.10
    prior: float = 0.05


def normalize_exact_text(text: str) -> str:
    value = unicodedata.normalize("NFC", text).casefold().strip()
    ascii_folded = unicodedata.normalize("NFKD", value)
    ascii_folded = "".join(ch for ch in ascii_folded if not unicodedata.combining(ch))
    compact: list[str] = []
    last_space = False
    for ch in ascii_folded:
        if ch.isalnum():
            compact.append(ch)
            last_space = False
            continue
        if not last_space:
            compact.append(" ")
            last_space = True
    return "".join(compact).strip()


def popularity_to_prior(popularity: float) -> float:
    value = max(0.0, float(popularity))
    return 1.0 - math.exp(-math.log1p(value) / 6.0)


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
    crosslink_terms: Sequence[str] = (),
    limit: int,
    include_top_k: bool,
) -> str:
    payload = {
        "mention_norm": mention_norm,
        "context_terms": list(context_terms),
        "crosslink_terms": list(crosslink_terms),
        "coarse_hints": list(coarse_hints),
        "fine_hints": list(fine_hints),
        "limit": int(limit),
        "include_top_k": bool(include_top_k),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


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
        label_values: list[str] = [label] if label else []
        variant_values: list[str] = []
        aliases = candidate.get("aliases")
        if isinstance(aliases, list):
            variant_values.extend([item for item in aliases if isinstance(item, str)])
        elif isinstance(aliases, Mapping):
            # Backward-compatible with older cached candidate shapes before aliases were flattened.
            for values in aliases.values():
                if isinstance(values, list):
                    variant_values.extend([item for item in values if isinstance(item, str)])
        else:
            # Backward-compatible with intermediate cached candidate shapes using name_variants.
            name_variants = candidate.get("name_variants")
            if isinstance(name_variants, list):
                variant_values.extend([item for item in name_variants if isinstance(item, str)])
            labels = candidate.get("labels")
            if isinstance(labels, list):
                variant_values.extend([item for item in labels if isinstance(item, str)])
        exact_name_match = mention_norm in {
            normalize_exact_text(value) for value in [*label_values, *variant_values] if value
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
    ) -> None:
        self.store = PostgresStore(postgres_dsn)

    def _fuzzy_search(
        self,
        mention: str,
        *,
        context_terms: Sequence[str],
        crosslink_terms: Sequence[str],
        coarse_hints: Sequence[str],
        fine_hints: Sequence[str],
        size: int,
    ) -> list[dict[str, Any]]:
        return self.store.search_candidates_fuzzy(
            mention_query=mention,
            context_query=" ".join(context_terms),
            crosslink_query=" ".join(crosslink_terms),
            coarse_hints=coarse_hints,
            fine_hints=fine_hints,
            size=size,
        )

    def lookup(
        self,
        *,
        mention: str,
        mention_context: str | Sequence[str] | None = None,
        crosslink_hints: str | Sequence[str] | None = None,
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
        raw_crosslink_values: list[str]
        if crosslink_hints is None:
            raw_crosslink_values = []
        elif isinstance(crosslink_hints, str):
            raw_crosslink_values = [crosslink_hints]
        else:
            raw_crosslink_values = [value for value in crosslink_hints if isinstance(value, str)]
        crosslink_values: list[str] = []
        crosslink_seen: set[str] = set()
        for raw_value in raw_crosslink_values:
            stripped = raw_value.strip()
            if not stripped or stripped in crosslink_seen:
                continue
            compacted = compact_crosslink_hint(stripped) or stripped
            if compacted in crosslink_seen:
                continue
            crosslink_seen.add(compacted)
            crosslink_values.append(compacted)
        crosslink_terms = normalize_context_inputs(crosslink_values)
        limit = max(1, min(100, int(top_k)))

        cache_key = build_cache_key(
            mention_norm=mention_norm,
            context_terms=context_terms,
            crosslink_terms=crosslink_terms,
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

        strategy = "fuzzy"
        fuzzy_hits = self._fuzzy_search(
            mention_value,
            context_terms=context_terms,
            crosslink_terms=crosslink_terms,
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
            "crosslink_hints": crosslink_values,
            "crosslink_terms": crosslink_terms,
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
