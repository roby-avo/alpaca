from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_LETTER_RE = re.compile(r"[A-Za-z]")
_DESCRIPTOR_TOKENS = frozenset(
    {
        "lake",
        "lakes",
        "sea",
        "ocean",
        "river",
        "basin",
        "bay",
        "gulf",
        "strait",
        "island",
        "mountain",
        "peak",
        "volcano",
        "city",
        "town",
        "village",
        "country",
        "state",
        "province",
        "region",
        "border",
        "harbor",
        "harbour",
        "port",
        "water",
        "freshwater",
        "salt",
        "inland",
        "geography",
        "great",
        "body",
        "landmark",
    }
)
_META_ENTITY_TOKENS = frozenset(
    {
        "category",
        "module",
        "template",
        "wikipedia",
        "wikimedia",
        "article",
        "paper",
        "journal",
        "encyclopedia",
        "history",
        "map",
        "tour",
        "level",
        "list",
    }
)
_OFF_DOMAIN_TOKENS = frozenset(
    {
        "ship",
        "station",
        "metro",
        "railway",
        "airport",
        "power",
        "plant",
        "route",
        "line",
        "bus",
        "motorway",
        "highway",
    }
)
_PREFIX_DESCRIPTOR_TOKENS = frozenset(
    {
        "lake",
        "sea",
        "river",
        "mountain",
        "mount",
        "island",
        "bay",
        "gulf",
        "strait",
        "country",
        "city",
        "region",
    }
)
_BODY_OF_WATER_TOKENS = frozenset(
    {
        "lake",
        "lakes",
        "sea",
        "ocean",
        "river",
        "bay",
        "gulf",
        "strait",
        "reservoir",
        "lagoon",
        "water",
        "freshwater",
    }
)
_CREATIVE_WORK_TOKENS = frozenset(
    {
        "painting",
        "film",
        "movie",
        "song",
        "album",
        "book",
        "novel",
        "poem",
        "sculpture",
        "photograph",
        "artwork",
        "drawing",
        "opera",
        "series",
    }
)
_SCHOLARLY_TOKENS = frozenset(
    {
        "article",
        "paper",
        "journal",
        "study",
        "proceedings",
        "review",
        "scholarly",
    }
)
_DERIVATIVE_TOPIC_TOKENS = frozenset(
    {
        "history",
        "map",
        "tour",
        "level",
        "encyclopedia",
        "atlas",
        "list",
        "chronology",
        "timeline",
    }
)
_QUALIFIER_STOP_TOKENS = frozenset(
    {
        "a",
        "an",
        "and",
        "at",
        "by",
        "for",
        "from",
        "in",
        "n",
        "of",
        "on",
        "the",
        "to",
    }
)
_LABEL_SUPPORT_STOP_TOKENS = _QUALIFIER_STOP_TOKENS | frozenset({"n", "note", "notes"})


def _normalize_text(text: str) -> str:
    return " ".join(_TOKEN_RE.findall(text.casefold()))


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.casefold())


def _is_searchworthy_term(text: str) -> bool:
    value = text.strip()
    if not value:
        return False
    if not _LETTER_RE.search(value):
        return False
    tokens = _tokenize(value)
    if not tokens:
        return False
    alpha_tokens = [token for token in tokens if any(char.isalpha() for char in token)]
    return bool(alpha_tokens)


def _extract_soft_context_weights(preprocessing_schema: dict[str, Any]) -> dict[str, float]:
    retrieval_plan = preprocessing_schema.get("retrieval_plan", {})
    soft_terms = retrieval_plan.get("soft_context_terms", []) if isinstance(retrieval_plan, dict) else []
    weights: dict[str, float] = {}
    if not isinstance(soft_terms, list):
        return weights
    for item in soft_terms:
        if not isinstance(item, dict):
            continue
        value = item.get("value")
        if not isinstance(value, str) or not value.strip():
            continue
        weight = item.get("weight")
        try:
            numeric_weight = float(weight)
        except (TypeError, ValueError):
            numeric_weight = 0.0
        normalized = _normalize_text(value)
        if normalized:
            weights[normalized] = max(weights.get(normalized, 0.0), numeric_weight)
    return weights


def _context_term_weights(
    lookup_payload: dict[str, Any],
    preprocessing_schema: dict[str, Any] | None,
) -> list[tuple[str, float]]:
    weight_by_text: dict[str, float] = {}

    if preprocessing_schema:
        for normalized, weight in _extract_soft_context_weights(preprocessing_schema).items():
            if normalized:
                weight_by_text[normalized] = max(weight_by_text.get(normalized, 0.0), float(weight))

    raw_context = lookup_payload.get("mention_context", [])
    if isinstance(raw_context, list):
        for item in raw_context:
            if not isinstance(item, str):
                continue
            value = item.strip()
            if not _is_searchworthy_term(value):
                continue
            weight_by_text[value] = max(weight_by_text.get(value, 0.0), 0.5)

    return sorted(weight_by_text.items(), key=lambda item: (item[1], item[0]), reverse=True)


def _descriptor_prefix_scores(
    preprocessing_schema: dict[str, Any] | None,
) -> dict[str, float]:
    if not preprocessing_schema:
        return {}

    scores: dict[str, float] = {}
    for normalized_value, weight in _extract_soft_context_weights(preprocessing_schema).items():
        for token in _tokenize(normalized_value):
            if token in _PREFIX_DESCRIPTOR_TOKENS:
                scores[token] = scores.get(token, 0.0) + float(weight)

    context = preprocessing_schema.get("context", {})
    if isinstance(context, dict):
        for key in ("row_values", "other_row_values", "sampled_column_values", "mention_context"):
            raw_values = context.get(key, [])
            if not isinstance(raw_values, list):
                continue
            for raw_value in raw_values:
                if not isinstance(raw_value, str):
                    continue
                for token in _tokenize(raw_value):
                    if token in _PREFIX_DESCRIPTOR_TOKENS:
                        scores[token] = scores.get(token, 0.0) + 0.15

    cell_hypothesis = preprocessing_schema.get("cell_hypothesis", {})
    if isinstance(cell_hypothesis, dict):
        raw_hints = cell_hypothesis.get("description_hints", [])
        if isinstance(raw_hints, list):
            for hint in raw_hints:
                if not isinstance(hint, str):
                    continue
                for token in _tokenize(hint):
                    if token in _PREFIX_DESCRIPTOR_TOKENS:
                        scores[token] = scores.get(token, 0.0) + 0.4

    return scores


def _extract_query_variants(
    lookup_payload: dict[str, Any],
    preprocessing_schema: dict[str, Any] | None,
) -> list[tuple[str, float]]:
    mention = lookup_payload.get("mention", "")
    if not isinstance(mention, str) or not mention.strip():
        return []

    mention_value = mention.strip()
    mention_tokens = set(_tokenize(mention_value))
    variants: dict[str, float] = {mention_value: 1.0}

    if preprocessing_schema:
        retrieval_plan = preprocessing_schema.get("retrieval_plan", {})
        raw_variants = retrieval_plan.get("query_variants", []) if isinstance(retrieval_plan, dict) else []
        if isinstance(raw_variants, list):
            for item in raw_variants:
                if not isinstance(item, dict):
                    continue
                text = item.get("text")
                if not isinstance(text, str) or not _is_searchworthy_term(text):
                    continue
                try:
                    weight = float(item.get("weight", 0.0))
                except (TypeError, ValueError):
                    weight = 0.0
                variants[text.strip()] = max(variants.get(text.strip(), 0.0), max(0.0, min(1.0, weight)))
        context = preprocessing_schema.get("context", {})
        col_id = context.get("col_id") if isinstance(context, dict) else None
        _, column_role = _table_profile_summary(preprocessing_schema, col_id if isinstance(col_id, int) else None)
        if column_role == "BODY_OF_WATER_NAME" and mention_tokens and not (mention_tokens & _BODY_OF_WATER_TOKENS):
            expanded = f"Lake {mention_value}"
            variants[expanded] = max(variants.get(expanded, 0.0), 0.96)

    for term, weight in _context_term_weights(lookup_payload, preprocessing_schema):
        text = term.strip()
        if not _is_searchworthy_term(text):
            continue
        token_set = set(_tokenize(text))
        if mention_tokens and (mention_tokens <= token_set or token_set <= mention_tokens):
            variants[text] = max(variants.get(text, 0.0), min(0.95, max(0.45, weight)))

    if mention_tokens and not (mention_tokens & _PREFIX_DESCRIPTOR_TOKENS):
        for token, score in sorted(
            _descriptor_prefix_scores(preprocessing_schema).items(),
            key=lambda item: (item[1], item[0]),
            reverse=True,
        )[:2]:
            expanded = f"{token.title()} {mention_value}"
            variants[expanded] = max(variants.get(expanded, 0.0), min(0.85, 0.35 + score))

    return sorted(variants.items(), key=lambda item: (item[1], len(item[0])), reverse=True)


def build_es_query_from_lookup_payload(
    lookup_payload: dict[str, Any],
    *,
    preprocessing_schema: dict[str, Any] | None = None,
    max_context_terms: int = 6,
    max_query_variants: int = 6,
) -> dict[str, Any]:
    mention = lookup_payload.get("mention", "")
    if not isinstance(mention, str) or not mention.strip():
        raise ValueError("lookup_payload.mention must be a non-empty string")

    mention_value = mention.strip()
    mention_context = lookup_payload.get("mention_context", [])
    coarse_hints = lookup_payload.get("coarse_hints", [])
    fine_hints = lookup_payload.get("fine_hints", [])
    item_category_filters = lookup_payload.get("item_category_filters", [])
    top_k = int(lookup_payload.get("top_k", 10))
    query_variants = _extract_query_variants(lookup_payload, preprocessing_schema)[: max(1, int(max_query_variants))]

    must: list[dict[str, Any]] = [
        {
            "bool": {
                "should": [],
                "minimum_should_match": 1,
            }
        }
    ]
    for text, weight in query_variants:
        boost_scale = 1.0 + (4.0 * max(0.0, min(1.0, weight)))
        must[0]["bool"]["should"].extend(
            [
                {"term": {"label.keyword": {"value": text.casefold(), "boost": 8.0 * boost_scale}}},
                {"match_phrase": {"label": {"query": text, "boost": 5.0 * boost_scale}}},
                {
                    "multi_match": {
                        "query": text,
                        "fields": ["label^8", "labels^4", "aliases^4"],
                        "type": "best_fields",
                        "boost": 2.0 * boost_scale,
                    }
                },
            ]
        )

    filters: list[dict[str, Any]] = []
    if isinstance(coarse_hints, list) and coarse_hints:
        filters.extend({"term": {"coarse_type": value.casefold()}} for value in coarse_hints if isinstance(value, str) and value)
    if isinstance(fine_hints, list) and fine_hints:
        filters.extend({"term": {"fine_type": value.casefold()}} for value in fine_hints if isinstance(value, str) and value)
    if isinstance(item_category_filters, list) and item_category_filters:
        filters.extend(
            {"term": {"item_category": value.casefold()}}
            for value in item_category_filters
            if isinstance(value, str) and value
        )

    should: list[dict[str, Any]] = []
    context_term_weights = _context_term_weights(lookup_payload, preprocessing_schema)
    if isinstance(mention_context, list):
        _ = mention_context
    for raw_value, weight in context_term_weights[: max(1, int(max_context_terms))]:
        value = raw_value.strip()
        if not _is_searchworthy_term(value):
            continue
        token_count = len(_tokenize(value))
        context_boost = 1.5 + (4.0 * max(0.0, min(1.0, weight)))
        should.append({"match": {"context_string": {"query": value, "boost": context_boost}}})
        if token_count <= 5:
            label_boost = 1.5 + (5.0 * max(0.0, min(1.0, weight)))
            should.append({"match_phrase": {"label": {"query": value, "boost": label_boost}}})
            should.append(
                {
                    "multi_match": {
                        "query": value,
                        "fields": ["label^6", "labels^3", "aliases^3"],
                        "type": "best_fields",
                        "boost": max(1.5, label_boost - 1.0),
                    }
                }
            )

    return {
        "size": max(1, min(100, top_k)),
        "query": {
            "bool": {
                "must": must,
                "filter": filters,
                "should": should,
                "minimum_should_match": 0,
            }
        },
    }


@dataclass(frozen=True, slots=True)
class CandidateFeatures:
    qid: str
    label: str
    raw_es_score: float
    heuristic_score: float
    final_score: float
    label_exact: bool
    alias_exact: bool
    label_prefix: bool
    label_contains: bool
    label_similarity: float
    alias_similarity: float
    best_name_similarity: float
    coarse_match: bool
    fine_match: bool
    item_category_match: bool
    context_overlap_count: int
    weighted_context_overlap: float
    prior: float
    has_wikipedia: bool
    expected_descriptor_overlap: int
    meta_token_overlap: int
    off_domain_token_overlap: int
    extra_label_descriptor_overlap: int
    extra_label_meta_overlap: int
    extra_label_off_domain_overlap: int
    candidate_family: str
    family_penalty: float
    family_group: str | None
    schema_family_match: bool
    column_role_match: bool
    row_template_alignment: float
    label_has_parenthetical_qualifier: bool
    unsupported_qualifier_count: int
    supported_qualifier_count: int
    unsupported_qualifier_tokens: list[str]
    mention_token_coverage: float
    extra_label_token_count: int
    context_supported_extra_label_token_count: int
    context_unsupported_extra_label_token_count: int
    context_unsupported_extra_label_tokens: list[str]
    label_context_support_score: float
    primary_entity_score: float
    derivative_penalty: float
    same_type_resolver_score: float
    same_type_resolver_adjustment: float
    same_type_resolver_applied: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _clamp(value: float, *, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _label_has_parenthetical_qualifier(label: str) -> bool:
    return "(" in label and ")" in label and label.find("(") < label.rfind(")")


def _content_token_set(text: str) -> set[str]:
    return {
        token
        for token in _tokenize(text)
        if token
        and token not in _LABEL_SUPPORT_STOP_TOKENS
        and not token.isdigit()
    }


def _qualifier_token_sets(
    *,
    label_tokens: set[str],
    mention_tokens: set[str],
    expected_descriptor_tokens: set[str],
    expected_context_tokens: set[str],
    column_role: str,
) -> tuple[set[str], set[str], set[str]]:
    extra_tokens = {
        token
        for token in (label_tokens - mention_tokens)
        if token and token not in _QUALIFIER_STOP_TOKENS and not token.isdigit()
    }
    supported_tokens = set(expected_descriptor_tokens) | set(expected_context_tokens)
    if column_role == "BODY_OF_WATER_NAME":
        supported_tokens |= _BODY_OF_WATER_TOKENS
    supported = extra_tokens & supported_tokens
    unsupported = extra_tokens - supported
    return extra_tokens, supported, unsupported


def _primary_entity_score(
    *,
    candidate_family: str,
    family_penalty: float,
    label_exact: bool,
    alias_exact: bool,
    has_wikipedia: bool,
    prior: float,
    schema_family_match: bool,
    column_role_match: bool,
    row_template_alignment: float,
    unsupported_qualifier_count: int,
    meta_token_overlap: int,
    off_domain_token_overlap: int,
) -> float:
    score = 0.0
    score += 0.24 if candidate_family == "PRIMARY" else 0.0
    score += 0.16 if label_exact or alias_exact else 0.0
    score += 0.12 if has_wikipedia else 0.0
    score += min(0.14, max(0.0, prior) * 0.18)
    score += 0.11 if schema_family_match else 0.0
    score += 0.13 if column_role_match else 0.0
    score += min(0.14, max(0.0, row_template_alignment) * 0.14)
    score -= min(0.22, unsupported_qualifier_count * 0.08)
    score -= min(0.16, meta_token_overlap * 0.05)
    score -= min(0.16, off_domain_token_overlap * 0.06)
    score -= min(0.2, family_penalty * 0.03)
    return round(_clamp(score), 4)
def _expected_descriptor_tokens(
    preprocessing_schema: dict[str, Any],
    lookup_payload: dict[str, Any],
) -> set[str]:
    tokens: set[str] = set()

    for value in lookup_payload.get("coarse_hints", []):
        if isinstance(value, str):
            tokens.update(token for token in _tokenize(value) if token in _DESCRIPTOR_TOKENS)
    for value in lookup_payload.get("fine_hints", []):
        if isinstance(value, str):
            tokens.update(token for token in _tokenize(value) if token in _DESCRIPTOR_TOKENS)

    context = preprocessing_schema.get("context", {})
    if isinstance(context, dict):
        for key in ("row_values", "other_row_values", "sampled_column_values", "mention_context"):
            raw_values = context.get(key, [])
            if not isinstance(raw_values, list):
                continue
            for value in raw_values:
                if not isinstance(value, str):
                    continue
                tokens.update(token for token in _tokenize(value) if token in _DESCRIPTOR_TOKENS)

    cell_hypothesis = preprocessing_schema.get("cell_hypothesis", {})
    if isinstance(cell_hypothesis, dict):
        for key in ("canonical_mention",):
            raw = cell_hypothesis.get(key)
            if isinstance(raw, str):
                tokens.update(token for token in _tokenize(raw) if token in _DESCRIPTOR_TOKENS)
        raw_hints = cell_hypothesis.get("description_hints", [])
        if isinstance(raw_hints, list):
            for hint in raw_hints:
                if not isinstance(hint, str):
                    continue
                tokens.update(token for token in _tokenize(hint) if token in _DESCRIPTOR_TOKENS)

    retrieval_plan = preprocessing_schema.get("retrieval_plan", {})
    if isinstance(retrieval_plan, dict):
        raw_terms = retrieval_plan.get("soft_context_terms", [])
        if isinstance(raw_terms, list):
            for item in raw_terms:
                if not isinstance(item, dict):
                    continue
                value = item.get("value")
                if not isinstance(value, str):
                    continue
                tokens.update(token for token in _tokenize(value) if token in _DESCRIPTOR_TOKENS)

    context = preprocessing_schema.get("context", {})
    col_id = context.get("col_id") if isinstance(context, dict) else None
    _, column_role = _table_profile_summary(preprocessing_schema, col_id if isinstance(col_id, int) else None)
    if column_role == "BODY_OF_WATER_NAME":
        tokens.update({"lake", "water", "freshwater"})

    return tokens


def _expected_context_tokens(
    preprocessing_schema: dict[str, Any],
    lookup_payload: dict[str, Any],
) -> set[str]:
    tokens: set[str] = set()

    mention = lookup_payload.get("mention", "")
    if isinstance(mention, str):
        tokens.update(_tokenize(mention))

    raw_context = lookup_payload.get("mention_context", [])
    if isinstance(raw_context, list):
        for value in raw_context:
            if isinstance(value, str):
                tokens.update(_tokenize(value))

    context = preprocessing_schema.get("context", {})
    if isinstance(context, dict):
        for key in ("row_values", "other_row_values", "sampled_column_values"):
            raw_values = context.get(key, [])
            if not isinstance(raw_values, list):
                continue
            for value in raw_values:
                if isinstance(value, str):
                    tokens.update(_tokenize(value))

    cell_hypothesis = preprocessing_schema.get("cell_hypothesis", {})
    if isinstance(cell_hypothesis, dict):
        for key in ("canonical_mention",):
            value = cell_hypothesis.get(key)
            if isinstance(value, str):
                tokens.update(_tokenize(value))
        raw_hints = cell_hypothesis.get("description_hints", [])
        if isinstance(raw_hints, list):
            for value in raw_hints:
                if isinstance(value, str):
                    tokens.update(_tokenize(value))

    retrieval_plan = preprocessing_schema.get("retrieval_plan", {})
    if isinstance(retrieval_plan, dict):
        raw_terms = retrieval_plan.get("soft_context_terms", [])
        if isinstance(raw_terms, list):
            for item in raw_terms:
                if not isinstance(item, dict):
                    continue
                value = item.get("value")
                if isinstance(value, str):
                    tokens.update(_tokenize(value))

    return tokens


def _family_group_key(label: str, candidate_family: str) -> str | None:
    if candidate_family == "PRIMARY":
        return None
    normalized = label.strip()
    if ":" in normalized:
        normalized = normalized.split(":", 1)[1]
    normalized = _normalize_text(normalized)
    return f"{candidate_family}:{normalized}" if normalized else candidate_family


def _table_profile_summary(preprocessing_schema: dict[str, Any], col_id: int | None) -> tuple[str, str]:
    table_profile = preprocessing_schema.get("table_profile", {})
    if not isinstance(table_profile, dict):
        return "", ""
    family = str(table_profile.get("table_semantic_family", "") or "")
    role = ""
    raw_roles = table_profile.get("column_roles", {})
    if isinstance(raw_roles, dict) and col_id is not None:
        values = raw_roles.get(str(col_id), [])
        if isinstance(values, list) and values and isinstance(values[0], dict):
            raw_role = values[0].get("role")
            if isinstance(raw_role, str):
                role = raw_role
    return family, role


def _classify_candidate_family(
    *,
    label: str,
    description: str,
    context_string: str,
    expected_context_tokens: set[str],
    coarse_type: str,
    item_category: str,
) -> tuple[str, float]:
    label_lower = label.casefold()
    label_description_tokens = set(_tokenize(" ".join(part for part in (label, description) if part)))

    if label_lower.startswith("category:") or {"wikimedia", "category"} <= label_description_tokens:
        return "META_CATEGORY", 4.5
    if label_lower.startswith("module:") or "module" in label_description_tokens or "template" in label_description_tokens:
        return "META_MODULE", 4.0

    # Protect common structured entity classes from being misclassified by related-topic context.
    if item_category == "ENTITY" and coarse_type in {"PERSON", "LOCATION", "ORGANIZATION"}:
        return "PRIMARY", 0.0

    creative_overlap = len(label_description_tokens & _CREATIVE_WORK_TOKENS)
    scholarly_overlap = len(label_description_tokens & _SCHOLARLY_TOKENS)
    derivative_overlap = len(label_description_tokens & _DERIVATIVE_TOPIC_TOKENS)
    expected_creative_overlap = len(expected_context_tokens & _CREATIVE_WORK_TOKENS)
    expected_scholarly_overlap = len(expected_context_tokens & _SCHOLARLY_TOKENS)
    expected_derivative_overlap = len(expected_context_tokens & _DERIVATIVE_TOPIC_TOKENS)

    if scholarly_overlap:
        penalty = 0.75 if expected_scholarly_overlap else 3.5
        return "SCHOLARLY_ARTICLE", penalty
    if creative_overlap:
        penalty = 0.75 if expected_creative_overlap else 3.25
        return "CREATIVE_WORK", penalty
    if derivative_overlap:
        penalty = 0.5 if expected_derivative_overlap else 2.5
        return "DERIVATIVE_TOPIC", penalty

    return "PRIMARY", 0.0


def _edit_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            substitution_cost = 0 if left_char == right_char else 1
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + substitution_cost,
                )
            )
        previous = current
    return previous[-1]


def _char_ngrams(text: str, n: int = 3) -> set[str]:
    compact = text.replace(" ", "")
    if not compact:
        return set()
    if len(compact) <= n:
        return {compact}
    return {compact[index : index + n] for index in range(len(compact) - n + 1)}


def _dice_similarity(left: str, right: str) -> float:
    left_ngrams = _char_ngrams(left)
    right_ngrams = _char_ngrams(right)
    if not left_ngrams or not right_ngrams:
        return 0.0
    overlap = len(left_ngrams & right_ngrams)
    return (2.0 * overlap) / (len(left_ngrams) + len(right_ngrams))


def _name_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    edit_similarity = 1.0 - (_edit_distance(left, right) / max(len(left), len(right), 1))
    dice_similarity = _dice_similarity(left, right)
    return max(0.0, min(1.0, max(edit_similarity, dice_similarity)))


def _candidate_type_key(item: dict[str, Any]) -> tuple[str, str, str]:
    source = item.get("source", {})
    if not isinstance(source, dict):
        source = {}
    return (
        str(source.get("item_category", "") or ""),
        str(source.get("coarse_type", "") or ""),
        str(source.get("fine_type", "") or ""),
    )


def _candidate_resolver_score(item: dict[str, Any]) -> float:
    features = item.get("features", {})
    if not isinstance(features, dict):
        return 0.0
    try:
        return float(features.get("same_type_resolver_score", features.get("final_score", 0.0)) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _candidate_final_score(item: dict[str, Any]) -> float:
    features = item.get("features", {})
    if not isinstance(features, dict):
        return 0.0
    try:
        return float(features.get("final_score", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _candidate_best_name_similarity(item: dict[str, Any]) -> float:
    features = item.get("features", {})
    if not isinstance(features, dict):
        return 0.0
    try:
        return float(features.get("best_name_similarity", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _apply_same_type_ambiguity_resolver(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(candidates) < 2:
        return candidates

    for item in candidates:
        features = item.get("features", {})
        if isinstance(features, dict):
            features["same_type_resolver_adjustment"] = 0.0
            features["same_type_resolver_applied"] = False

    top = candidates[0]
    top_key = _candidate_type_key(top)
    if not any(top_key):
        return candidates

    competitors = [item for item in candidates[:12] if _candidate_type_key(item) == top_key]
    if len(competitors) < 2:
        return candidates

    top_resolver_score = _candidate_resolver_score(top)
    top_final_score = _candidate_final_score(top)
    top_name_similarity = _candidate_best_name_similarity(top)
    best = max(competitors, key=_candidate_resolver_score)
    if best is top:
        return candidates

    best_resolver_score = _candidate_resolver_score(best)
    best_final_score = _candidate_final_score(best)
    best_name_similarity = _candidate_best_name_similarity(best)
    resolver_delta = best_resolver_score - top_resolver_score
    final_margin = top_final_score - best_final_score

    # Conservative gate: only intervene inside a same-type cluster when the
    # alternate candidate is still lexically plausible and identity evidence is
    # clearly better than the current top candidate.
    if resolver_delta < 1.0:
        return candidates
    if final_margin > 3.5:
        return candidates
    if best_name_similarity < max(0.74, top_name_similarity - 0.2):
        return candidates

    best_features = best.get("features", {})
    if isinstance(best_features, dict):
        adjustment = round(min(3.0, resolver_delta), 4)
        best_features["same_type_resolver_adjustment"] = adjustment
        best_features["same_type_resolver_applied"] = True
        best_features["final_score"] = round(best_final_score + adjustment, 4)

    candidates.sort(
        key=lambda item: (
            _candidate_final_score(item),
            _candidate_resolver_score(item),
            float(item.get("features", {}).get("raw_es_score", 0.0) or 0.0)
            if isinstance(item.get("features"), dict)
            else 0.0,
            item.get("qid") or "",
        ),
        reverse=True,
    )
    return candidates


def extract_candidate_features(
    *,
    hit: dict[str, Any],
    lookup_payload: dict[str, Any],
    preprocessing_schema: dict[str, Any],
) -> CandidateFeatures:
    source = hit.get("_source", {})
    if not isinstance(source, dict):
        source = {}

    mention = lookup_payload.get("mention", "")
    context = preprocessing_schema.get("context", {})
    col_id = context.get("col_id") if isinstance(context, dict) else None
    family, column_role = _table_profile_summary(preprocessing_schema, col_id if isinstance(col_id, int) else None)
    mention_norm = _normalize_text(mention) if isinstance(mention, str) else ""
    label = source.get("label") if isinstance(source.get("label"), str) else ""
    label_norm = _normalize_text(label)
    aliases = source.get("aliases") if isinstance(source.get("aliases"), list) else []
    alias_exact = any(_normalize_text(value) == mention_norm for value in aliases if isinstance(value, str))
    label_exact = label_norm == mention_norm and bool(label_norm)
    label_prefix = bool(label_norm and mention_norm and label_norm.startswith(mention_norm))
    label_contains = bool(label_norm and mention_norm and mention_norm in label_norm)
    label_similarity = _name_similarity(mention_norm, label_norm)
    alias_similarity = max(
        (_name_similarity(mention_norm, _normalize_text(value)) for value in aliases if isinstance(value, str)),
        default=0.0,
    )
    best_name_similarity = max(label_similarity, alias_similarity)

    coarse_hints = lookup_payload.get("coarse_hints", [])
    fine_hints = lookup_payload.get("fine_hints", [])
    item_category_filters = lookup_payload.get("item_category_filters", [])
    candidate_coarse = source.get("coarse_type") if isinstance(source.get("coarse_type"), str) else ""
    candidate_fine = source.get("fine_type") if isinstance(source.get("fine_type"), str) else ""
    candidate_item = source.get("item_category") if isinstance(source.get("item_category"), str) else ""
    coarse_match = candidate_coarse in coarse_hints if isinstance(coarse_hints, list) else False
    fine_match = candidate_fine in fine_hints if isinstance(fine_hints, list) else False
    item_category_match = candidate_item in item_category_filters if isinstance(item_category_filters, list) else False

    context_weights = _extract_soft_context_weights(preprocessing_schema)
    context_string = source.get("context_string") if isinstance(source.get("context_string"), str) else ""
    description = source.get("description") if isinstance(source.get("description"), str) else ""
    context_tokens = set(_tokenize(context_string))
    description_tokens = set(_tokenize(description))
    candidate_tokens = context_tokens | description_tokens | set(_tokenize(label))
    overlap_count = 0
    weighted_overlap = 0.0
    for normalized_value, weight in context_weights.items():
        tokens = [token for token in _tokenize(normalized_value) if token]
        if not tokens:
            continue
        if all(token in context_tokens for token in tokens):
            overlap_count += 1
            weighted_overlap += float(weight)

    prior = 0.0
    raw_prior = source.get("prior")
    if isinstance(raw_prior, (int, float)):
        prior = float(raw_prior)
    raw_score = hit.get("_score")
    es_score = float(raw_score) if isinstance(raw_score, (int, float)) else 0.0
    has_wikipedia = bool(source.get("wikipedia_url"))

    expected_descriptor_tokens = _expected_descriptor_tokens(preprocessing_schema, lookup_payload)
    expected_context_tokens = _expected_context_tokens(preprocessing_schema, lookup_payload)
    expected_descriptor_overlap = len(expected_descriptor_tokens & candidate_tokens)
    meta_token_overlap = len(_META_ENTITY_TOKENS & candidate_tokens)
    off_domain_token_overlap = len(_OFF_DOMAIN_TOKENS & candidate_tokens)
    mention_tokens = set(_tokenize(mention))
    label_tokens = set(_tokenize(label))
    mention_content_tokens = _content_token_set(mention) if isinstance(mention, str) else set()
    label_content_tokens = _content_token_set(label)
    mention_token_coverage = (
        len(mention_content_tokens & label_content_tokens) / len(mention_content_tokens)
        if mention_content_tokens
        else 0.0
    )
    extra_label_content_tokens = label_content_tokens - mention_content_tokens
    extra_label_tokens = label_tokens - mention_tokens
    extra_label_descriptor_overlap = len(expected_descriptor_tokens & extra_label_tokens)
    extra_label_meta_overlap = len(_META_ENTITY_TOKENS & extra_label_tokens)
    extra_label_off_domain_overlap = len(_OFF_DOMAIN_TOKENS & extra_label_tokens)
    candidate_family, family_penalty = _classify_candidate_family(
        label=label,
        description=description,
        context_string=context_string,
        expected_context_tokens=expected_context_tokens,
        coarse_type=candidate_coarse,
        item_category=candidate_item,
    )
    if (
        candidate_family in {"DERIVATIVE_TOPIC", "SCHOLARLY_ARTICLE", "CREATIVE_WORK"}
        and label_exact
        and expected_descriptor_overlap > 0
        and (prior >= 0.2 or has_wikipedia)
    ):
        candidate_family = "PRIMARY"
        family_penalty = 0.0
    family_group = _family_group_key(label, candidate_family)
    schema_family_match = False
    column_role_match = False
    row_template_alignment = 0.0
    if family == "BIOGRAPHY":
        schema_family_match = candidate_coarse == "PERSON"
        if column_role == "PERSON_NAME_OR_ALIAS":
            column_role_match = candidate_coarse == "PERSON" and candidate_fine == "HUMAN"
            row_template_alignment = min(1.0, weighted_overlap / 1.5)
    elif family == "GEOGRAPHY":
        schema_family_match = candidate_coarse == "LOCATION"
        if column_role in {"BODY_OF_WATER_NAME", "GEOGRAPHIC_FEATURE_NAME"}:
            column_role_match = candidate_coarse == "LOCATION" and candidate_fine == "LANDMARK"
            row_template_alignment = min(1.0, (expected_descriptor_overlap / 3.0) + (weighted_overlap / 3.0))

    body_of_water_role = column_role == "BODY_OF_WATER_NAME"
    body_of_water_token_overlap = len(candidate_tokens & _BODY_OF_WATER_TOKENS)
    label_body_of_water_token_overlap = len(label_tokens & _BODY_OF_WATER_TOKENS)
    _, supported_qualifier_tokens, unsupported_qualifier_tokens = _qualifier_token_sets(
        label_tokens=label_tokens,
        mention_tokens=mention_tokens,
        expected_descriptor_tokens=expected_descriptor_tokens,
        expected_context_tokens=expected_context_tokens,
        column_role=column_role,
    )
    label_parenthetical = _label_has_parenthetical_qualifier(label)
    unsupported_qualifier_count = len(unsupported_qualifier_tokens)
    supported_qualifier_count = len(supported_qualifier_tokens)
    label_context_supported_tokens = expected_descriptor_tokens | expected_context_tokens
    context_supported_extra_label_tokens = extra_label_content_tokens & label_context_supported_tokens
    context_unsupported_extra_label_tokens = extra_label_content_tokens - label_context_supported_tokens
    extra_label_token_count = len(extra_label_content_tokens)
    context_supported_extra_label_token_count = len(context_supported_extra_label_tokens)
    context_unsupported_extra_label_token_count = len(context_unsupported_extra_label_tokens)
    label_context_support_score = _clamp(
        (mention_token_coverage * 0.45)
        + (min(1.0, context_supported_extra_label_token_count / 2.0) * 0.25)
        + (min(1.0, expected_descriptor_overlap / 3.0) * 0.2)
        + (min(1.0, weighted_overlap / 2.5) * 0.1)
        - (min(1.0, context_unsupported_extra_label_token_count / 3.0) * 0.2)
    )
    derivative_penalty = 0.0
    derivative_penalty += family_penalty
    derivative_penalty += min(4.0, unsupported_qualifier_count * 1.2)
    if label_parenthetical and unsupported_qualifier_count:
        derivative_penalty += 1.25
    if candidate_family != "PRIMARY" and (meta_token_overlap or off_domain_token_overlap):
        derivative_penalty += 0.75
    primary_entity_score = _primary_entity_score(
        candidate_family=candidate_family,
        family_penalty=family_penalty,
        label_exact=label_exact,
        alias_exact=alias_exact,
        has_wikipedia=has_wikipedia,
        prior=prior,
        schema_family_match=schema_family_match,
        column_role_match=column_role_match,
        row_template_alignment=row_template_alignment,
        unsupported_qualifier_count=unsupported_qualifier_count,
        meta_token_overlap=meta_token_overlap,
        off_domain_token_overlap=off_domain_token_overlap,
    )

    heuristic_score = 0.0
    heuristic_score += 8.0 if label_exact else 0.0
    heuristic_score += 5.0 if alias_exact else 0.0
    heuristic_score += 2.0 if label_prefix else 0.0
    heuristic_score += 1.0 if label_contains else 0.0
    heuristic_score += max(0.0, (label_similarity - 0.62) * 8.0)
    heuristic_score += max(0.0, (alias_similarity - 0.72) * 6.0)
    heuristic_score += 1.5 if coarse_match else 0.0
    heuristic_score += 1.5 if fine_match else 0.0
    heuristic_score += 1.0 if item_category_match else 0.0
    heuristic_score += min(3.0, weighted_overlap * 2.0)
    heuristic_score += min(1.5, prior * 2.0)
    heuristic_score += 0.75 if has_wikipedia else 0.0
    heuristic_score += min(2.0, es_score / 100.0)
    heuristic_score += min(3.5, expected_descriptor_overlap * 1.25)
    heuristic_score += 1.2 if schema_family_match else 0.0
    heuristic_score += 1.8 if column_role_match else 0.0
    heuristic_score += min(1.5, row_template_alignment * 1.5)
    if body_of_water_role:
        heuristic_score += min(4.0, body_of_water_token_overlap * 1.4)
        heuristic_score += min(3.0, label_body_of_water_token_overlap * 2.0)
        if label_body_of_water_token_overlap and has_wikipedia:
            heuristic_score += 1.0
        if label_exact and body_of_water_token_overlap == 0:
            heuristic_score -= 6.0
        if candidate_fine == "COUNTRY":
            heuristic_score -= 7.0
    heuristic_score += min(2.5, extra_label_descriptor_overlap * 2.0)
    heuristic_score += min(1.25, supported_qualifier_count * 0.55)
    heuristic_score -= min(4.0, meta_token_overlap * 1.25)
    heuristic_score -= min(4.0, off_domain_token_overlap * 1.75)
    heuristic_score -= min(3.0, extra_label_meta_overlap * 1.5)
    heuristic_score -= min(3.5, extra_label_off_domain_overlap * 2.0)
    heuristic_score -= min(4.5, unsupported_qualifier_count * 1.1)
    if label_parenthetical and unsupported_qualifier_count:
        heuristic_score -= 1.2
    heuristic_score += primary_entity_score * 2.0

    if label_exact and expected_descriptor_overlap == 0 and off_domain_token_overlap > 0:
        heuristic_score -= 3.0
    if label_exact and expected_descriptor_overlap == 0 and meta_token_overlap > 0:
        heuristic_score -= 2.0
    if candidate_family != "PRIMARY" and label_exact:
        heuristic_score -= min(1.5, family_penalty * 0.5)

    final_score = heuristic_score - family_penalty
    same_type_resolver_score = (
        final_score
        + (primary_entity_score * 3.0)
        + min(2.0, row_template_alignment * 2.0)
        + min(2.0, weighted_overlap)
        + min(1.5, expected_descriptor_overlap * 0.45)
        - min(4.0, derivative_penalty * 0.75)
    )

    return CandidateFeatures(
        qid=source.get("qid") if isinstance(source.get("qid"), str) else "",
        label=label,
        raw_es_score=es_score,
        heuristic_score=heuristic_score,
        final_score=final_score,
        label_exact=label_exact,
        alias_exact=alias_exact,
        label_prefix=label_prefix,
        label_contains=label_contains,
        label_similarity=round(label_similarity, 4),
        alias_similarity=round(alias_similarity, 4),
        best_name_similarity=round(best_name_similarity, 4),
        coarse_match=coarse_match,
        fine_match=fine_match,
        item_category_match=item_category_match,
        context_overlap_count=overlap_count,
        weighted_context_overlap=weighted_overlap,
        prior=prior,
        has_wikipedia=has_wikipedia,
        expected_descriptor_overlap=expected_descriptor_overlap,
        meta_token_overlap=meta_token_overlap,
        off_domain_token_overlap=off_domain_token_overlap,
        extra_label_descriptor_overlap=extra_label_descriptor_overlap,
        extra_label_meta_overlap=extra_label_meta_overlap,
        extra_label_off_domain_overlap=extra_label_off_domain_overlap,
        candidate_family=candidate_family,
        family_penalty=family_penalty,
        family_group=family_group,
        schema_family_match=schema_family_match,
        column_role_match=column_role_match,
        row_template_alignment=round(row_template_alignment, 4),
        label_has_parenthetical_qualifier=label_parenthetical,
        unsupported_qualifier_count=unsupported_qualifier_count,
        supported_qualifier_count=supported_qualifier_count,
        unsupported_qualifier_tokens=sorted(unsupported_qualifier_tokens),
        mention_token_coverage=round(mention_token_coverage, 4),
        extra_label_token_count=extra_label_token_count,
        context_supported_extra_label_token_count=context_supported_extra_label_token_count,
        context_unsupported_extra_label_token_count=context_unsupported_extra_label_token_count,
        context_unsupported_extra_label_tokens=sorted(context_unsupported_extra_label_tokens),
        label_context_support_score=round(label_context_support_score, 4),
        primary_entity_score=primary_entity_score,
        derivative_penalty=round(derivative_penalty, 4),
        same_type_resolver_score=round(same_type_resolver_score, 4),
        same_type_resolver_adjustment=0.0,
        same_type_resolver_applied=False,
    )


def rerank_es_hits(
    *,
    es_result: dict[str, Any],
    lookup_payload: dict[str, Any],
    preprocessing_schema: dict[str, Any],
) -> list[dict[str, Any]]:
    hits = es_result.get("hits", {}).get("hits", [])
    if not isinstance(hits, list):
        return []
    candidates: list[dict[str, Any]] = []
    for rank, hit in enumerate(hits, start=1):
        if not isinstance(hit, dict):
            continue
        source = hit.get("_source", {})
        if not isinstance(source, dict):
            continue
        features = extract_candidate_features(
            hit=hit,
            lookup_payload=lookup_payload,
            preprocessing_schema=preprocessing_schema,
        )
        candidates.append(
            {
                "raw_rank": rank,
                "qid": source.get("qid"),
                "label": source.get("label"),
                "retrieved_by": hit.get("_retrieved_by") if isinstance(hit.get("_retrieved_by"), list) else [],
                "features": features.to_dict(),
                "source": {
                    "coarse_type": source.get("coarse_type"),
                    "fine_type": source.get("fine_type"),
                    "item_category": source.get("item_category"),
                    "prior": source.get("prior"),
                    "wikipedia_url": source.get("wikipedia_url"),
                    "dbpedia_url": source.get("dbpedia_url"),
                    "context_string": source.get("context_string"),
                    "description": source.get("description"),
                },
            }
        )
    candidates.sort(
        key=lambda item: (
            float(item["features"].get("final_score", 0.0)),
            float(item["features"].get("raw_es_score", 0.0)),
            float(item["features"].get("prior", 0.0)),
            item.get("qid") or "",
        ),
        reverse=True,
    )
    filtered_candidates: list[dict[str, Any]] = []
    seen_family_groups: set[str] = set()
    for item in candidates:
        family_group = item["features"].get("family_group")
        if isinstance(family_group, str) and family_group:
            if family_group in seen_family_groups:
                continue
            seen_family_groups.add(family_group)
        filtered_candidates.append(item)

    filtered_candidates = _apply_same_type_ambiguity_resolver(filtered_candidates)
    for reranked_index, item in enumerate(filtered_candidates, start=1):
        item["reranked_rank"] = reranked_index
    return filtered_candidates


def score_reranked_decision(candidates: list[dict[str, Any]]) -> dict[str, object]:
    if not candidates:
        return {
            "selected_qid": None,
            "selected_label": None,
            "confidence": 0.0,
            "margin": 0.0,
            "abstain": True,
            "reason_codes": ["no_candidates"],
        }

    top = candidates[0]
    top_features = top.get("features", {})
    second_features = candidates[1].get("features", {}) if len(candidates) > 1 else {}

    top_score = float(top_features.get("final_score", 0.0))
    second_score = float(second_features.get("final_score", 0.0)) if second_features else 0.0
    margin = top_score - second_score

    lexical_signal = 0.0
    if bool(top_features.get("label_exact")):
        lexical_signal = 1.0
    elif bool(top_features.get("alias_exact")):
        lexical_signal = 0.9
    elif bool(top_features.get("label_prefix")):
        lexical_signal = 0.7
    elif bool(top_features.get("label_contains")):
        lexical_signal = 0.45
    else:
        try:
            best_name_similarity = float(top_features.get("best_name_similarity", 0.0))
        except (TypeError, ValueError):
            best_name_similarity = 0.0
        if best_name_similarity >= 0.95:
            lexical_signal = 0.92
        elif best_name_similarity >= 0.9:
            lexical_signal = 0.84
        elif best_name_similarity >= 0.84:
            lexical_signal = 0.74
        elif best_name_similarity >= 0.76:
            lexical_signal = 0.58

    type_signal = 0.0
    type_signal += 0.4 if bool(top_features.get("coarse_match")) else 0.0
    type_signal += 0.4 if bool(top_features.get("fine_match")) else 0.0
    type_signal += 0.2 if bool(top_features.get("item_category_match")) else 0.0
    type_signal = min(1.0, type_signal)

    context_signal = min(
        1.0,
        (float(top_features.get("weighted_context_overlap", 0.0)) / 2.5)
        + (float(top_features.get("expected_descriptor_overlap", 0.0)) / 4.0),
    )
    schema_signal = min(
        1.0,
        (0.5 if bool(top_features.get("schema_family_match")) else 0.0)
        + (0.35 if bool(top_features.get("column_role_match")) else 0.0)
        + (float(top_features.get("row_template_alignment", 0.0)) * 0.3),
    )
    authority_signal = min(
        1.0,
        float(top_features.get("prior", 0.0)) * 1.4 + (0.2 if bool(top_features.get("has_wikipedia")) else 0.0),
    )
    family_signal = 1.0 if top_features.get("candidate_family") == "PRIMARY" else 0.45
    try:
        primary_entity_signal = float(top_features.get("primary_entity_score", 0.0) or 0.0)
    except (TypeError, ValueError):
        primary_entity_signal = 0.0
    try:
        derivative_penalty = float(top_features.get("derivative_penalty", 0.0) or 0.0)
    except (TypeError, ValueError):
        derivative_penalty = 0.0
    identity_signal = max(0.0, min(1.0, primary_entity_signal - min(0.45, derivative_penalty * 0.05)))
    score_signal = min(1.0, max(0.0, (top_score - 6.0) / 10.0))
    margin_signal = min(1.0, max(0.0, margin / 4.0))

    confidence = (
        0.24 * lexical_signal
        + 0.17 * type_signal
        + 0.12 * context_signal
        + 0.08 * schema_signal
        + 0.13 * authority_signal
        + 0.11 * family_signal
        + 0.06 * identity_signal
        + 0.08 * score_signal
        + 0.07 * margin_signal
    )
    confidence = max(0.0, min(1.0, confidence))

    reason_codes: list[str] = []
    if lexical_signal >= 0.9:
        reason_codes.append("strong_lexical_match")
    elif lexical_signal >= 0.58:
        reason_codes.append("fuzzy_lexical_match")
    if type_signal >= 0.8:
        reason_codes.append("type_alignment")
    if context_signal >= 0.5:
        reason_codes.append("context_alignment")
    if schema_signal >= 0.45:
        reason_codes.append("schema_alignment")
    if authority_signal >= 0.45:
        reason_codes.append("authority_signal")
    if family_signal >= 0.9:
        reason_codes.append("primary_family")
    if identity_signal >= 0.55:
        reason_codes.append("primary_identity_signal")
    if bool(top_features.get("same_type_resolver_applied")):
        reason_codes.append("same_type_resolver")
    if margin_signal >= 0.35:
        reason_codes.append("clear_margin")
    if not reason_codes:
        reason_codes.append("weak_evidence")

    same_label_competition = False
    if len(candidates) > 1:
        top_label = top.get("label")
        runner_up_label = candidates[1].get("label")
        if isinstance(top_label, str) and isinstance(runner_up_label, str):
            same_label_competition = _normalize_text(top_label) == _normalize_text(runner_up_label)

    abstain = bool(
        confidence < 0.58
        or top_score < 7.5
        or top_features.get("candidate_family") not in {"PRIMARY", "DERIVATIVE_TOPIC"}
        or (margin < 1.0 and (lexical_signal < 0.72 or same_label_competition))
    )

    return {
        "selected_qid": top.get("qid"),
        "selected_label": top.get("label"),
        "confidence": round(confidence, 4),
        "margin": round(margin, 4),
        "abstain": abstain,
        "candidate_family": top_features.get("candidate_family"),
        "top_final_score": round(top_score, 4),
        "runner_up_qid": candidates[1].get("qid") if len(candidates) > 1 else None,
        "runner_up_label": candidates[1].get("label") if len(candidates) > 1 else None,
        "reason_codes": reason_codes,
    }
