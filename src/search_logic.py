from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from .common import tokenize

_VALID_TYPE_LABEL_RE = re.compile(r"^[A-Za-z0-9_.:/-]+$")


def normalize_type_labels(
    type_labels: Sequence[str] | None,
    *,
    field_name: str,
) -> list[str]:
    if type_labels is None:
        return []

    normalized: list[str] = []
    seen: set[str] = set()
    for raw in type_labels:
        value = raw.strip()
        if not value:
            continue
        if not _VALID_TYPE_LABEL_RE.match(value):
            raise ValueError(
                f"Invalid value '{raw}' for {field_name}. Allowed characters: "
                "letters, digits, '_', '-', '.', ':', '/'."
            )
        if value not in seen:
            seen.add(value)
            normalized.append(value)

    return normalized


def _type_filter_clause(field: str, values: list[str]) -> str | None:
    if not values:
        return None

    if len(values) == 1:
        return f"{field}:{values[0]}"

    joined = " OR ".join(f"{field}:{value}" for value in values)
    return f"({joined})"


def build_quickwit_query(
    query_text: str,
    ner_coarse_types: Sequence[str] | None = None,
    ner_fine_types: Sequence[str] | None = None,
) -> str:
    query_terms = tokenize(query_text)
    if not query_terms:
        raise ValueError("Query must contain at least one alphanumeric term.")

    text_clause = " ".join(query_terms)

    coarse_values = normalize_type_labels(
        ner_coarse_types,
        field_name="coarse_type",
    )
    fine_values = normalize_type_labels(
        ner_fine_types,
        field_name="fine_type",
    )

    clauses = [f"({text_clause})"]

    coarse_clause = _type_filter_clause("coarse_type", coarse_values)
    if coarse_clause is not None:
        clauses.append(coarse_clause)

    fine_clause = _type_filter_clause("fine_type", fine_values)
    if fine_clause is not None:
        clauses.append(fine_clause)

    if len(clauses) == 1:
        return text_clause

    return " AND ".join(clauses)


def _extract_doc_and_score(hit: Mapping[str, Any]) -> tuple[dict[str, Any], float]:
    score = hit.get("_score", hit.get("score", 0.0))
    try:
        numeric_score = float(score)
    except (TypeError, ValueError):
        numeric_score = 0.0

    doc: dict[str, Any]
    if isinstance(hit.get("_source"), Mapping):
        doc = dict(hit["_source"])
    elif isinstance(hit.get("document"), Mapping):
        doc = dict(hit["document"])
    elif isinstance(hit.get("json"), Mapping):
        doc = dict(hit["json"])
    else:
        doc = {
            key: value
            for key, value in hit.items()
            if key not in {"_score", "score", "sort_value", "snippet"}
        }

    return doc, numeric_score


def _context_overlap_score(doc: Mapping[str, Any], context_terms: set[str]) -> float:
    if not context_terms:
        return 0.0

    name_text = _doc_name_text(doc)
    context = doc.get("context") if isinstance(doc.get("context"), str) else ""
    bow = doc.get("bow") if isinstance(doc.get("bow"), str) else ""
    doc_terms = set(tokenize(f"{name_text} {context} {bow}"))
    if not doc_terms:
        return 0.0

    overlap = len(context_terms & doc_terms)
    return overlap / len(context_terms)


def _safe_text_map(raw: Any) -> dict[str, str]:
    if not isinstance(raw, Mapping):
        return {}

    output: dict[str, str] = {}
    for key, value in raw.items():
        if isinstance(key, str) and isinstance(value, str):
            output[key] = value
    return output


def _safe_alias_map(raw: Any) -> dict[str, list[str]]:
    if not isinstance(raw, Mapping):
        return {}

    output: dict[str, list[str]] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not isinstance(value, list):
            continue
        aliases = [item for item in value if isinstance(item, str)]
        if aliases:
            output[key] = aliases
    return output


def _name_from_labels_aliases(
    labels: Mapping[str, str],
    aliases: Mapping[str, Sequence[str]],
) -> str:
    tokens: list[str] = []
    seen: set[str] = set()

    ordered_languages: list[str] = []
    if "en" in labels or "en" in aliases:
        ordered_languages.append("en")
    ordered_languages.extend(
        language for language in sorted(set(labels) | set(aliases)) if language != "en"
    )

    for language in ordered_languages:
        label = labels.get(language)
        if isinstance(label, str):
            candidate = label.strip()
            if candidate and candidate not in seen:
                seen.add(candidate)
                tokens.append(candidate)

        for alias in aliases.get(language, []):
            if not isinstance(alias, str):
                continue
            candidate = alias.strip()
            if candidate and candidate not in seen:
                seen.add(candidate)
                tokens.append(candidate)

    return " ".join(tokens)


def _doc_name_text(doc: Mapping[str, Any]) -> str:
    raw_name = doc.get("name_text")
    if isinstance(raw_name, str) and raw_name:
        return raw_name

    labels = _safe_text_map(doc.get("labels"))
    aliases = _safe_alias_map(doc.get("aliases"))
    return _name_from_labels_aliases(labels, aliases)


def rerank_hits_by_context(
    raw_hits: Sequence[Mapping[str, Any]],
    context: str | None,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    context_terms = set(tokenize(context or ""))

    scored: list[tuple[float, dict[str, Any]]] = []
    for raw_hit in raw_hits:
        doc, base_score = _extract_doc_and_score(raw_hit)

        entity_id = doc.get("id")
        if not isinstance(entity_id, str) or not entity_id:
            continue

        labels = _safe_text_map(doc.get("labels"))
        aliases = _safe_alias_map(doc.get("aliases"))

        name_text = _doc_name_text(doc)
        context = doc.get("context") if isinstance(doc.get("context"), str) else ""
        bow = doc.get("bow") if isinstance(doc.get("bow"), str) else ""
        coarse_type = doc.get("coarse_type") if isinstance(doc.get("coarse_type"), str) else ""
        fine_type = doc.get("fine_type") if isinstance(doc.get("fine_type"), str) else ""

        context_overlap = _context_overlap_score(doc, context_terms)
        final_score = base_score + context_overlap

        normalized_hit = {
            "id": entity_id,
            "labels": labels,
            "aliases": aliases,
            "name_text": name_text,
            "context": context,
            "bow": bow,
            "coarse_type": coarse_type,
            "fine_type": fine_type,
            "score": base_score,
            "context_overlap": context_overlap,
            "final_score": final_score,
        }
        scored.append((final_score, normalized_hit))

    scored.sort(key=lambda row: row[0], reverse=True)
    return [item for _, item in scored[:limit]]
