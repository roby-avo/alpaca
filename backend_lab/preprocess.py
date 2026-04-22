from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from .dataset import CellContext
from .table_profile import normalize_table_profile


_GENERIC_HEADER_RE = re.compile(r"^col\d+$", re.IGNORECASE)
_BRACKET_RE = re.compile(r"\[[^\]]+\]")
_WHITESPACE_RE = re.compile(r"\s+")
_DATE_HINT_RE = re.compile(r"\b\d{4}(?:[-/]\d{1,2}(?:[-/]\d{1,2})?)?\b")
_NUMERIC_HINT_RE = re.compile(r"\d")

BACKEND_ENTITY_CATEGORIES = frozenset(
    {
        "ENTITY",
        "TYPE",
        "PREDICATE",
        "DISAMBIGUATION",
        "LEXEME",
        "FORM",
        "SENSE",
        "MEDIAINFO",
        "OTHER",
    }
)

BACKEND_COARSE_TYPES = frozenset(
    {
        "PERSON",
        "ORGANIZATION",
        "LOCATION",
        "WORK",
        "PRODUCT",
        "EVENT",
        "BIOLOGICAL_TAXON",
        "THING",
        "RELATION",
        "MISC",
    }
)

BACKEND_FINE_TYPES = frozenset(
    {
        "PERSON",
        "HUMAN",
        "FICTIONAL_CHARACTER",
        "COMPANY",
        "NONPROFIT_ORG",
        "GOVERNMENT_ORG",
        "EDUCATIONAL_ORG",
        "SPORTS_TEAM",
        "COUNTRY",
        "CITY",
        "REGION",
        "LANDMARK",
        "CELESTIAL_BODY",
        "CONFLICT",
        "SPORT_EVENT",
        "EVENT_GENERIC",
        "FILM",
        "BOOK",
        "MUSIC_WORK",
        "SOFTWARE",
        "INTERNET_MEME",
        "DEVICE",
        "MEDICATION",
        "FOOD_BEVERAGE",
        "PRODUCT_GENERIC",
        "BIOLOGICAL_TAXON",
        "PROPERTY",
        "COSMOLOGICAL_ENTITY",
        "MISC",
    }
)

ENTITY_CATEGORY_SYNONYMS = {
    "GEOGRAPHICFEATURE": "ENTITY",
    "GEOGRAPHICALFEATURE": "ENTITY",
    "PLACE": "ENTITY",
    "LOCATION": "ENTITY",
}

COARSE_TYPE_SYNONYMS = {
    "GEOGRAPHICFEATURE": "LOCATION",
    "GEOGRAPHICALFEATURE": "LOCATION",
    "PLACE": "LOCATION",
    "WATERBODY": "LOCATION",
    "BODY_OF_WATER": "LOCATION",
    "SEA": "LOCATION",
    "LAKE": "LOCATION",
    "RIVER": "LOCATION",
    "MOUNTAIN": "LOCATION",
}

FINE_TYPE_SYNONYMS = {
    "WATERBODY": "LANDMARK",
    "BODY_OF_WATER": "LANDMARK",
    "SEA": "LANDMARK",
    "LAKE": "LANDMARK",
    "RIVER": "LANDMARK",
    "OCEAN": "LANDMARK",
    "GULF": "LANDMARK",
    "BAY": "LANDMARK",
    "STRAIT": "LANDMARK",
    "MOUNTAIN": "LANDMARK",
    "ISLAND": "LANDMARK",
    "AIRPORT": "LANDMARK",
    "BRIDGE": "LANDMARK",
    "BUILDING": "LANDMARK",
    "MONUMENT": "LANDMARK",
    "PERSON": "HUMAN",
}


def _clamp_unit_interval(raw: Any, default: float) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, value))


def _dedupe_strings(values: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = raw.strip()
        if not value or value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def normalize_mention_surface(text: str) -> str:
    value = text.strip()
    value = _BRACKET_RE.sub("", value)
    value = value.replace("*", " ")
    value = value.strip(" .,:;!?-_")
    value = _WHITESPACE_RE.sub(" ", value).strip()
    return value


def _is_generic_header(value: str) -> bool:
    return bool(_GENERIC_HEADER_RE.match(value.strip()))


@dataclass(frozen=True, slots=True)
class SignalValue:
    value: str
    confidence: float
    weight: float
    source: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RelatedCell:
    column_name: str
    cell_value: str
    relation: str
    confidence: float
    source: str = "row"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class QueryVariant:
    text: str
    confidence: float
    weight: float
    source: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CellHypothesis:
    canonical_mention: str
    mention_variants: list[str] = field(default_factory=list)
    entity_category: SignalValue | None = None
    coarse_type: SignalValue | None = None
    fine_type: SignalValue | None = None
    domain: SignalValue | None = None
    description_hints: list[str] = field(default_factory=list)
    entity_hypotheses: list[dict[str, object]] = field(default_factory=list)
    mention_strength: str = "unknown"
    weakness_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "canonical_mention": self.canonical_mention,
            "mention_variants": list(self.mention_variants),
            "entity_category": self.entity_category.to_dict() if self.entity_category else None,
            "coarse_type": self.coarse_type.to_dict() if self.coarse_type else None,
            "fine_type": self.fine_type.to_dict() if self.fine_type else None,
            "domain": self.domain.to_dict() if self.domain else None,
            "description_hints": list(self.description_hints),
            "entity_hypotheses": list(self.entity_hypotheses),
            "mention_strength": self.mention_strength,
            "weakness_reasons": list(self.weakness_reasons),
        }


@dataclass(frozen=True, slots=True)
class ColumnProfile:
    semantic_role: str
    confidence: float
    coarse_type_distribution: list[SignalValue] = field(default_factory=list)
    fine_type_distribution: list[SignalValue] = field(default_factory=list)
    sampled_values: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "semantic_role": self.semantic_role,
            "confidence": self.confidence,
            "coarse_type_distribution": [item.to_dict() for item in self.coarse_type_distribution],
            "fine_type_distribution": [item.to_dict() for item in self.fine_type_distribution],
            "sampled_values": list(self.sampled_values),
        }


@dataclass(frozen=True, slots=True)
class RowConstraints:
    related_cells: list[RelatedCell] = field(default_factory=list)
    context_terms: list[SignalValue] = field(default_factory=list)
    numeric_hints: list[str] = field(default_factory=list)
    temporal_hints: list[str] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "related_cells": [item.to_dict() for item in self.related_cells],
            "context_terms": [item.to_dict() for item in self.context_terms],
            "numeric_hints": list(self.numeric_hints),
            "temporal_hints": list(self.temporal_hints),
            "confidence": self.confidence,
        }


@dataclass(frozen=True, slots=True)
class RetrievalPlan:
    hard_filters: dict[str, list[str]] = field(default_factory=dict)
    soft_context_terms: list[SignalValue] = field(default_factory=list)
    query_variants: list[QueryVariant] = field(default_factory=list)
    hypothesis_plan: list[dict[str, object]] = field(default_factory=list)
    backoff_plan: list[dict[str, object]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "hard_filters": {key: list(values) for key, values in self.hard_filters.items()},
            "soft_context_terms": [item.to_dict() for item in self.soft_context_terms],
            "query_variants": [item.to_dict() for item in self.query_variants],
            "hypothesis_plan": list(self.hypothesis_plan),
            "backoff_plan": list(self.backoff_plan),
            "notes": list(self.notes),
        }


@dataclass(frozen=True, slots=True)
class PreprocessingSchema:
    metadata: dict[str, object]
    context: dict[str, object]
    table_profile: dict[str, object]
    cell_hypothesis: CellHypothesis
    column_profile: ColumnProfile
    row_constraints: RowConstraints
    retrieval_plan: RetrievalPlan

    def to_dict(self) -> dict[str, object]:
        return {
            "metadata": dict(self.metadata),
            "context": dict(self.context),
            "table_profile": dict(self.table_profile),
            "cell_hypothesis": self.cell_hypothesis.to_dict(),
            "column_profile": self.column_profile.to_dict(),
            "row_constraints": self.row_constraints.to_dict(),
            "retrieval_plan": self.retrieval_plan.to_dict(),
        }


def _seed_query_variants(context: CellContext) -> list[QueryVariant]:
    normalized = normalize_mention_surface(context.mention)
    ordered_variants = [normalized, context.mention, normalized.replace(".", "")]
    variants = _dedupe_strings(ordered_variants)
    output: list[QueryVariant] = []
    for index, text in enumerate(variants):
        if index == 0:
            confidence = 1.0
            weight = 1.0
        elif text == context.mention.strip():
            confidence = 0.8
            weight = 0.7
        else:
            confidence = 0.7
            weight = max(0.45, 0.7 - (index * 0.1))
        output.append(
            QueryVariant(
                text=text,
                confidence=confidence,
                weight=weight,
                source="seed",
            )
        )
    return output


def _seed_context_terms(context: CellContext) -> list[SignalValue]:
    signals: list[SignalValue] = []
    for value in context.other_row_values:
        source = "row"
        weight = 0.65 if not _NUMERIC_HINT_RE.search(value) else 0.25
        confidence = 0.7 if not _NUMERIC_HINT_RE.search(value) else 0.45
        signals.append(SignalValue(value=value, confidence=confidence, weight=weight, source=source))
    for value in context.sampled_column_values:
        signals.append(SignalValue(value=value, confidence=0.6, weight=0.5, source="column"))
    if context.column_name and not _is_generic_header(context.column_name):
        signals.insert(0, SignalValue(value=context.column_name, confidence=0.9, weight=0.75, source="column"))

    deduped: list[SignalValue] = []
    seen: set[str] = set()
    for signal in signals:
        key = signal.value.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(signal)
    return deduped


def _seed_related_cells(context: CellContext) -> list[RelatedCell]:
    output: list[RelatedCell] = []
    for index, value in enumerate(context.row_values):
        if index == context.col_id or not value.strip():
            continue
        header_value = context.header[index].strip() if index < len(context.header) else f"col{index}"
        output.append(
            RelatedCell(
                column_name=header_value,
                cell_value=value.strip(),
                relation="same_row_context",
                confidence=0.6,
            )
        )
    return output


def _weakness_reasons(mention: str) -> list[str]:
    reasons: list[str] = []
    normalized = normalize_mention_surface(mention)
    tokens = [token for token in normalized.split() if token]
    if len(tokens) <= 1:
        reasons.append("single_token")
    if len(normalized) <= 10:
        reasons.append("short_surface")
    if normalized.isalpha() and normalized[:1].isupper() and len(tokens) == 1:
        reasons.append("capitalized_common_word_shape")
    if normalized.casefold() in {"gauge", "savannah", "obsession", "houston", "victoria", "viper", "gaga"}:
        reasons.append("known_polysemous_surface")
    return reasons


def _mention_strength(mention: str) -> tuple[str, list[str]]:
    reasons = _weakness_reasons(mention)
    normalized = normalize_mention_surface(mention)
    tokens = [token for token in normalized.split() if token]
    if not reasons:
        return "strong", []
    if len(tokens) >= 2 and "known_polysemous_surface" not in reasons:
        return "medium", reasons
    return "weak", reasons


def _signal_dict(value: str, confidence: float, weight: float, source: str) -> dict[str, object]:
    return {
        "value": value,
        "confidence": round(max(0.0, min(1.0, confidence)), 4),
        "weight": round(max(0.0, min(1.0, weight)), 4),
        "source": source,
    }


def _role_for_col(table_profile: dict[str, object], col_id: int) -> str:
    raw_roles = table_profile.get("column_roles", {})
    if not isinstance(raw_roles, dict):
        return "UNKNOWN"
    values = raw_roles.get(str(col_id), [])
    if not isinstance(values, list) or not values:
        return "UNKNOWN"
    first = values[0]
    if not isinstance(first, dict):
        return "UNKNOWN"
    role = first.get("role")
    return role if isinstance(role, str) and role.strip() else "UNKNOWN"


def _role_hypotheses_for_col(table_profile: dict[str, object], col_id: int) -> list[dict[str, object]]:
    raw_roles = table_profile.get("column_roles", {})
    if not isinstance(raw_roles, dict):
        return []
    values = raw_roles.get(str(col_id), [])
    if not isinstance(values, list):
        return []
    return [item for item in values if isinstance(item, dict)]


def _role_confidence_for_col(table_profile: dict[str, object], col_id: int) -> float:
    values = _role_hypotheses_for_col(table_profile, col_id)
    if not values:
        return 0.0
    return _clamp_unit_interval(values[0].get("confidence"), 0.0)


def _schema_uncertainty(
    table_profile: dict[str, object],
    context: CellContext | dict[str, object],
    mention_strength: str,
) -> dict[str, object]:
    col_id_raw = context.col_id if isinstance(context, CellContext) else context.get("col_id")
    col_id = col_id_raw if isinstance(col_id_raw, int) else -1
    table_confidence = _clamp_unit_interval(table_profile.get("confidence"), 0.0)
    role_values = _role_hypotheses_for_col(table_profile, col_id)
    role = _role_for_col(table_profile, col_id) if col_id >= 0 else "UNKNOWN"
    role_confidence = _role_confidence_for_col(table_profile, col_id) if col_id >= 0 else 0.0
    second_confidence = _clamp_unit_interval(role_values[1].get("confidence"), 0.0) if len(role_values) > 1 else 0.0
    reasons: list[str] = []
    if table_confidence < 0.72 and role_confidence < 0.86:
        reasons.append("low_table_confidence")
    if role in {"", "UNKNOWN"}:
        reasons.append("unknown_column_role")
    if role_confidence < 0.72:
        reasons.append("low_role_confidence")
    if second_confidence and (role_confidence - second_confidence) < 0.15:
        reasons.append("ambiguous_column_role")
    if mention_strength in {"medium", "strong"} and (table_confidence < 0.8 or role_confidence < 0.8):
        reasons.append("surface_form_resists_uncertain_schema")

    return {
        "is_uncertain": bool(reasons),
        "reasons": reasons,
        "table_confidence": round(table_confidence, 4),
        "column_role": role,
        "column_role_confidence": round(role_confidence, 4),
        "runner_up_role_confidence": round(second_confidence, 4),
    }


def _seed_entity_hypotheses(
    context: CellContext,
    table_profile: dict[str, object],
    mention_strength: str,
) -> list[dict[str, object]]:
    family = str(table_profile.get("table_semantic_family", "") or "GENERIC_ENTITY_TABLE")
    role = _role_for_col(table_profile, context.col_id)
    hypotheses: list[dict[str, object]] = []

    def add(item_category: str, coarse_type: str, fine_type: str, confidence: float, source: str) -> None:
        hypotheses.append(
            {
                "item_category": item_category,
                "coarse_type": coarse_type,
                "fine_type": fine_type,
                "confidence": round(confidence, 4),
                "source": source,
            }
        )

    if family == "BIOGRAPHY" and role == "PERSON_NAME_OR_ALIAS":
        add("ENTITY", "PERSON", "HUMAN", 0.9 if mention_strength != "weak" else 0.84, "table_profile")
        add("ENTITY", "WORK", "FILM", 0.28, "fallback")
        add("ENTITY", "LOCATION", "CITY", 0.24, "fallback")
    elif family == "GEOGRAPHY" and role in {"BODY_OF_WATER_NAME", "GEOGRAPHIC_FEATURE_NAME"}:
        add("ENTITY", "LOCATION", "LANDMARK", 0.9, "table_profile")
        add("ENTITY", "WORK", "MUSIC_WORK", 0.18, "fallback")
    else:
        normalized = normalize_mention_surface(context.mention)
        if len(normalized.split()) >= 2:
            add("ENTITY", "PERSON", "HUMAN", 0.62, "surface")
            add("ENTITY", "ORGANIZATION", "COMPANY", 0.22, "fallback")
        else:
            add("ENTITY", "MISC", "MISC", 0.42, "surface")
            add("ENTITY", "PERSON", "HUMAN", 0.38, "fallback")
            add("ENTITY", "LOCATION", "CITY", 0.32, "fallback")

    deduped: list[dict[str, object]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in hypotheses:
        key = (
            str(item.get("item_category", "")),
            str(item.get("coarse_type", "")),
            str(item.get("fine_type", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped[:3]


def _seed_hard_filters(
    table_profile: dict[str, object],
    hypotheses: list[dict[str, object]],
    mention_strength: str,
    column_role: str = "",
) -> dict[str, list[str]]:
    if not hypotheses:
        return {}
    table_confidence = float(table_profile.get("confidence", 0.0) or 0.0)
    top = hypotheses[0]
    top_confidence = float(top.get("confidence", 0.0) or 0.0)
    if (
        column_role == "BODY_OF_WATER_NAME"
        and top.get("coarse_type") == "LOCATION"
        and top.get("fine_type") == "LANDMARK"
    ):
        return {
            "item_category": ["ENTITY"],
            "coarse_type": ["LOCATION"],
            "fine_type": ["LANDMARK"],
        }
    if mention_strength == "weak" and table_confidence < 0.8:
        return {}
    if top_confidence < 0.82 and table_confidence < 0.78:
        return {}
    return {
        "item_category": [str(top.get("item_category", "ENTITY"))],
        "coarse_type": [str(top.get("coarse_type", ""))] if top.get("coarse_type") else [],
        "fine_type": [str(top.get("fine_type", ""))] if top.get("fine_type") else [],
    }


def _seed_hypothesis_plan(hypotheses: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for index, item in enumerate(hypotheses[:3], start=1):
        output.append(
            {
                "rank": index,
                "item_category": item.get("item_category", "ENTITY"),
                "coarse_type": item.get("coarse_type", ""),
                "fine_type": item.get("fine_type", ""),
                "confidence": item.get("confidence", 0.0),
                "retrieval_mode": "hard" if index == 1 else "soft",
                "source": item.get("source", "seed"),
            }
        )
    return output


def _seed_backoff_plan() -> list[dict[str, object]]:
    return [
        {"stage": "primary", "description": "Use primary hard filters when confidence is strong."},
        {"stage": "relax_fine_type", "description": "Drop fine_type if candidate pool is too thin."},
        {"stage": "alternate_hypothesis", "description": "Retry with alternate semantic hypothesis."},
        {"stage": "no_type_context_only", "description": "Retry with no type filter but same row/table context."},
        {"stage": "item_category_only", "description": "Fallback to item_category-only retrieval."},
    ]


def build_seed_schema(context: CellContext, table_profile: dict[str, object] | None = None) -> PreprocessingSchema:
    canonical_mention = normalize_mention_surface(context.mention) or context.mention.strip()
    mention_variants = _dedupe_strings([context.mention, canonical_mention])
    description_hints: list[str] = []
    if context.column_name and not _is_generic_header(context.column_name):
        description_hints.append(context.column_name)

    temporal_hints = _dedupe_strings(
        [value for value in context.other_row_values if _DATE_HINT_RE.search(value)]
    )
    numeric_hints = _dedupe_strings(
        [value for value in context.other_row_values if _NUMERIC_HINT_RE.search(value)]
    )
    soft_context_terms = _seed_context_terms(context)
    mention_strength, weakness_reasons = _mention_strength(context.mention)
    normalized_table_profile = (
        normalize_table_profile(table_profile or {}, dataset_id=context.dataset_id, table_id=context.table_id)
        if table_profile is not None
        else normalize_table_profile({}, dataset_id=context.dataset_id, table_id=context.table_id)
    )
    entity_hypotheses = _seed_entity_hypotheses(
        context,
        normalized_table_profile,
        mention_strength,
    )
    column_role = _role_for_col(normalized_table_profile, context.col_id)
    schema_uncertainty = _schema_uncertainty(normalized_table_profile, context, mention_strength)
    hard_filters = _seed_hard_filters(
        normalized_table_profile,
        entity_hypotheses,
        mention_strength,
        column_role=column_role,
    )
    if bool(schema_uncertainty.get("is_uncertain")) and hard_filters:
        hard_filters = {"item_category": hard_filters.get("item_category", ["ENTITY"])}

    schema = PreprocessingSchema(
        metadata={
            "dataset_id": context.dataset_id,
            "table_id": context.table_id,
            "row_id": context.row_id,
            "col_id": context.col_id,
            "schema_version": "0.2.0",
            "source": "deterministic_seed",
            "schema_uncertainty": schema_uncertainty,
        },
        context=context.to_dict(),
        table_profile=normalized_table_profile,
        cell_hypothesis=CellHypothesis(
            canonical_mention=canonical_mention,
            mention_variants=mention_variants,
            description_hints=description_hints,
            entity_hypotheses=entity_hypotheses,
            mention_strength=mention_strength,
            weakness_reasons=weakness_reasons,
        ),
        column_profile=ColumnProfile(
            semantic_role=column_role.casefold(),
            confidence=float(normalized_table_profile.get("confidence", 0.0) or 0.0),
            sampled_values=list(context.sampled_column_values),
        ),
        row_constraints=RowConstraints(
            related_cells=_seed_related_cells(context),
            context_terms=soft_context_terms,
            numeric_hints=numeric_hints,
            temporal_hints=temporal_hints,
            confidence=0.45 if soft_context_terms else 0.0,
        ),
        retrieval_plan=RetrievalPlan(
            hard_filters=hard_filters,
            soft_context_terms=soft_context_terms,
            query_variants=_seed_query_variants(context),
            hypothesis_plan=_seed_hypothesis_plan(entity_hypotheses),
            backoff_plan=_seed_backoff_plan(),
            notes=[
                "Seed schema generated without LLM inference.",
                "Table profile is attached to support table-aware preprocessing.",
                *(
                    [f"Schema uncertainty downgraded hard filters: {', '.join(schema_uncertainty.get('reasons', []))}"]
                    if bool(schema_uncertainty.get("is_uncertain"))
                    else []
                ),
            ],
        ),
    )
    return schema


def _extract_json_fragment(raw_text: str) -> str:
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("LLM response did not contain a JSON object.")
    return text[start : end + 1]


def parse_llm_json_response(raw_text: str) -> dict[str, Any]:
    fragment = _extract_json_fragment(raw_text)
    parsed = json.loads(fragment)
    if not isinstance(parsed, dict):
        raise ValueError("LLM response JSON must be an object.")
    return parsed


def merge_schema_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = merge_schema_dicts(current, value)
        else:
            merged[key] = value
    return merged


def _signal_from_mapping(
    raw: Any,
    *,
    default_source: str,
    default_confidence: float = 0.7,
    default_weight: float | None = None,
) -> SignalValue | None:
    if isinstance(raw, str):
        value = raw.strip()
        if not value:
            return None
        weight = default_confidence if default_weight is None else default_weight
        return SignalValue(value=value, confidence=default_confidence, weight=weight, source=default_source)
    if not isinstance(raw, dict):
        return None

    value = raw.get("value")
    if not isinstance(value, str) or not value.strip():
        alt_value = raw.get("type")
        if isinstance(alt_value, str) and alt_value.strip():
            value = alt_value
        else:
            return None

    confidence = _clamp_unit_interval(raw.get("confidence", raw.get("probability")), default_confidence)
    fallback_weight = confidence if default_weight is None else default_weight
    weight = _clamp_unit_interval(raw.get("weight", confidence), fallback_weight)
    source = raw.get("source")
    return SignalValue(
        value=value.strip(),
        confidence=confidence,
        weight=weight,
        source=source if isinstance(source, str) and source.strip() else default_source,
    )


def _normalize_label(raw: str, *, valid: frozenset[str], synonyms: dict[str, str]) -> str:
    value = raw.strip()
    if not value:
        return ""
    normalized = value.upper().replace(" ", "_")
    normalized = synonyms.get(normalized, normalized)
    return normalized if normalized in valid else ""


def _signal_list_from_raw(
    raw: Any,
    *,
    default_source: str,
    default_confidence: float = 0.7,
    default_weight: float | None = None,
) -> list[SignalValue]:
    if isinstance(raw, list):
        items = raw
    else:
        items = [raw]
    output: list[SignalValue] = []
    seen: set[str] = set()
    for item in items:
        signal = _signal_from_mapping(
            item,
            default_source=default_source,
            default_confidence=default_confidence,
            default_weight=default_weight,
        )
        if signal is None or signal.value in seen:
            continue
        seen.add(signal.value)
        output.append(signal)
    return output


def normalize_preprocessing_schema(schema: dict[str, Any]) -> dict[str, Any]:
    normalized = merge_schema_dicts({}, schema)
    normalized["table_profile"] = normalize_table_profile(
        normalized.get("table_profile", {}),
        dataset_id=str(normalized.get("metadata", {}).get("dataset_id", "") or ""),
        table_id=str(normalized.get("metadata", {}).get("table_id", "") or ""),
    )

    cell_hypothesis = normalized.get("cell_hypothesis")
    if isinstance(cell_hypothesis, dict):
        for key in ("entity_category", "coarse_type", "fine_type", "domain"):
            signal = _signal_from_mapping(cell_hypothesis.get(key), default_source="llm")
            if signal and key == "entity_category":
                value = _normalize_label(
                    signal.value,
                    valid=BACKEND_ENTITY_CATEGORIES,
                    synonyms=ENTITY_CATEGORY_SYNONYMS,
                )
                signal = SignalValue(value=value, confidence=signal.confidence, weight=signal.weight, source=signal.source) if value else None
            elif signal and key == "coarse_type":
                value = _normalize_label(
                    signal.value,
                    valid=BACKEND_COARSE_TYPES,
                    synonyms=COARSE_TYPE_SYNONYMS,
                )
                signal = SignalValue(value=value, confidence=signal.confidence, weight=signal.weight, source=signal.source) if value else None
            elif signal and key == "fine_type":
                value = _normalize_label(
                    signal.value,
                    valid=BACKEND_FINE_TYPES,
                    synonyms=FINE_TYPE_SYNONYMS,
                )
                signal = SignalValue(value=value, confidence=signal.confidence, weight=signal.weight, source=signal.source) if value else None
            cell_hypothesis[key] = signal.to_dict() if signal else None
        raw_entity_hypotheses = cell_hypothesis.get("entity_hypotheses", [])
        normalized_hypotheses: list[dict[str, object]] = []
        if isinstance(raw_entity_hypotheses, list):
            for item in raw_entity_hypotheses:
                if not isinstance(item, dict):
                    continue
                item_category = _normalize_label(
                    str(item.get("item_category", "")),
                    valid=BACKEND_ENTITY_CATEGORIES,
                    synonyms=ENTITY_CATEGORY_SYNONYMS,
                )
                coarse_type = _normalize_label(
                    str(item.get("coarse_type", "")),
                    valid=BACKEND_COARSE_TYPES,
                    synonyms=COARSE_TYPE_SYNONYMS,
                )
                fine_type = _normalize_label(
                    str(item.get("fine_type", "")),
                    valid=BACKEND_FINE_TYPES,
                    synonyms=FINE_TYPE_SYNONYMS,
                )
                if not item_category:
                    item_category = "ENTITY"
                if not coarse_type:
                    continue
                if not fine_type:
                    fine_type = "MISC"
                normalized_hypotheses.append(
                    {
                        "item_category": item_category,
                        "coarse_type": coarse_type,
                        "fine_type": fine_type,
                        "confidence": _clamp_unit_interval(item.get("confidence"), 0.6),
                        "source": item.get("source") if isinstance(item.get("source"), str) else "llm",
                    }
                )
        cell_hypothesis["entity_hypotheses"] = normalized_hypotheses
        mention_strength = str(cell_hypothesis.get("mention_strength", "unknown") or "unknown").strip().lower()
        cell_hypothesis["mention_strength"] = mention_strength if mention_strength in {"weak", "medium", "strong", "unknown"} else "unknown"
        raw_weakness = cell_hypothesis.get("weakness_reasons", [])
        if isinstance(raw_weakness, list):
            cell_hypothesis["weakness_reasons"] = [str(item) for item in raw_weakness if str(item).strip()]
        else:
            cell_hypothesis["weakness_reasons"] = []

    column_profile = normalized.get("column_profile")
    if isinstance(column_profile, dict):
        column_profile["coarse_type_distribution"] = [
            SignalValue(
                value=value,
                confidence=item.confidence,
                weight=item.weight,
                source=item.source,
            ).to_dict()
            for item in _signal_list_from_raw(column_profile.get("coarse_type_distribution", []), default_source="llm")
            if (value := _normalize_label(item.value, valid=BACKEND_COARSE_TYPES, synonyms=COARSE_TYPE_SYNONYMS))
        ]
        column_profile["fine_type_distribution"] = [
            SignalValue(
                value=value,
                confidence=item.confidence,
                weight=item.weight,
                source=item.source,
            ).to_dict()
            for item in _signal_list_from_raw(column_profile.get("fine_type_distribution", []), default_source="llm")
            if (value := _normalize_label(item.value, valid=BACKEND_FINE_TYPES, synonyms=FINE_TYPE_SYNONYMS))
        ]
        column_profile["confidence"] = _clamp_unit_interval(column_profile.get("confidence"), 0.0)

    row_constraints = normalized.get("row_constraints")
    if isinstance(row_constraints, dict):
        row_constraints["context_terms"] = [
            item.to_dict()
            for item in _signal_list_from_raw(row_constraints.get("context_terms", []), default_source="llm")
        ]
        row_constraints["confidence"] = _clamp_unit_interval(row_constraints.get("confidence"), 0.0)

    retrieval_plan = normalized.get("retrieval_plan")
    if isinstance(retrieval_plan, dict):
        raw_hard_filters = retrieval_plan.get("hard_filters", {})
        hard_filters: dict[str, list[str]] = {}
        if isinstance(raw_hard_filters, dict):
            for field_name in ("entity_category", "coarse_type", "fine_type", "crosslink_hints", "item_category"):
                signals = _signal_list_from_raw(
                    raw_hard_filters.get(field_name, []),
                    default_source="llm",
                    default_confidence=1.0,
                    default_weight=1.0,
                )
                values: list[str] = []
                for signal in signals:
                    if signal.confidence < 0.8:
                        continue
                    normalized_value = signal.value
                    if field_name == "entity_category" or field_name == "item_category":
                        normalized_value = _normalize_label(
                            signal.value,
                            valid=BACKEND_ENTITY_CATEGORIES,
                            synonyms=ENTITY_CATEGORY_SYNONYMS,
                        )
                    elif field_name == "coarse_type":
                        normalized_value = _normalize_label(
                            signal.value,
                            valid=BACKEND_COARSE_TYPES,
                            synonyms=COARSE_TYPE_SYNONYMS,
                        )
                    elif field_name == "fine_type":
                        normalized_value = _normalize_label(
                            signal.value,
                            valid=BACKEND_FINE_TYPES,
                            synonyms=FINE_TYPE_SYNONYMS,
                        )
                    if normalized_value:
                        values.append(normalized_value)
                if values:
                    hard_filters[field_name if field_name != "entity_category" else "item_category"] = _dedupe_strings(values)
        retrieval_plan["hard_filters"] = hard_filters
        retrieval_plan["soft_context_terms"] = [
            item.to_dict()
            for item in _signal_list_from_raw(retrieval_plan.get("soft_context_terms", []), default_source="llm")
        ]
        raw_query_variants = retrieval_plan.get("query_variants", [])
        if not isinstance(raw_query_variants, list):
            raw_query_variants = [raw_query_variants]
        normalized_query_variants: list[dict[str, object]] = []
        seen_queries: set[str] = set()
        for item in raw_query_variants:
            if isinstance(item, str):
                text = item.strip()
                if not text or text in seen_queries:
                    continue
                seen_queries.add(text)
                normalized_query_variants.append(
                    QueryVariant(text=text, confidence=0.7, weight=0.7, source="llm").to_dict()
                )
                continue
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if not isinstance(text, str) or not text.strip() or text.strip() in seen_queries:
                continue
            seen_queries.add(text.strip())
            normalized_query_variants.append(
                QueryVariant(
                    text=text.strip(),
                    confidence=_clamp_unit_interval(item.get("confidence"), 0.7),
                    weight=_clamp_unit_interval(item.get("weight"), 0.7),
                    source=item.get("source") if isinstance(item.get("source"), str) else "llm",
                ).to_dict()
            )
        retrieval_plan["query_variants"] = normalized_query_variants
        raw_hypothesis_plan = retrieval_plan.get("hypothesis_plan", [])
        if isinstance(raw_hypothesis_plan, list):
            normalized_hypothesis_plan: list[dict[str, object]] = []
            for item in raw_hypothesis_plan:
                if not isinstance(item, dict):
                    continue
                coarse_type = _normalize_label(
                    str(item.get("coarse_type", "")),
                    valid=BACKEND_COARSE_TYPES,
                    synonyms=COARSE_TYPE_SYNONYMS,
                )
                fine_type = _normalize_label(
                    str(item.get("fine_type", "")),
                    valid=BACKEND_FINE_TYPES,
                    synonyms=FINE_TYPE_SYNONYMS,
                )
                item_category = _normalize_label(
                    str(item.get("item_category", "")),
                    valid=BACKEND_ENTITY_CATEGORIES,
                    synonyms=ENTITY_CATEGORY_SYNONYMS,
                ) or "ENTITY"
                if not coarse_type:
                    continue
                normalized_hypothesis_plan.append(
                    {
                        "rank": int(item.get("rank", len(normalized_hypothesis_plan) + 1) or 0),
                        "item_category": item_category,
                        "coarse_type": coarse_type,
                        "fine_type": fine_type or "MISC",
                        "confidence": _clamp_unit_interval(item.get("confidence"), 0.6),
                        "retrieval_mode": str(item.get("retrieval_mode", "soft") or "soft"),
                        "source": item.get("source") if isinstance(item.get("source"), str) else "llm",
                    }
                )
            retrieval_plan["hypothesis_plan"] = normalized_hypothesis_plan
        else:
            retrieval_plan["hypothesis_plan"] = []
        raw_backoff_plan = retrieval_plan.get("backoff_plan", [])
        if isinstance(raw_backoff_plan, list):
            retrieval_plan["backoff_plan"] = [
                {
                    "stage": str(item.get("stage", "")).strip(),
                    "description": str(item.get("description", "")).strip(),
                }
                for item in raw_backoff_plan
                if isinstance(item, dict) and str(item.get("stage", "")).strip()
            ]
        else:
            retrieval_plan["backoff_plan"] = []
        notes = retrieval_plan.get("notes", [])
        if isinstance(notes, str):
            retrieval_plan["notes"] = [notes]
        elif isinstance(notes, list):
            retrieval_plan["notes"] = [value for value in notes if isinstance(value, str)]
        else:
            retrieval_plan["notes"] = []

    _apply_table_profile_policy(normalized)
    _apply_schema_uncertainty_policy(normalized)
    return normalized


def _primary_column_role(schema: dict[str, Any]) -> str:
    table_profile = schema.get("table_profile", {})
    context = schema.get("context", {})
    if not isinstance(table_profile, dict) or not isinstance(context, dict):
        return ""
    col_id = context.get("col_id")
    raw_roles = table_profile.get("column_roles", {})
    if not isinstance(col_id, int) or not isinstance(raw_roles, dict):
        return ""
    values = raw_roles.get(str(col_id), [])
    if not isinstance(values, list) or not values or not isinstance(values[0], dict):
        return ""
    role = values[0].get("role")
    return role if isinstance(role, str) else ""


def _apply_table_profile_policy(schema: dict[str, Any]) -> None:
    role = _primary_column_role(schema)
    if role != "BODY_OF_WATER_NAME":
        return

    cell_hypothesis = schema.get("cell_hypothesis", {})
    retrieval_plan = schema.get("retrieval_plan", {})
    if not isinstance(cell_hypothesis, dict) or not isinstance(retrieval_plan, dict):
        return

    water_hypothesis = {
        "item_category": "ENTITY",
        "coarse_type": "LOCATION",
        "fine_type": "LANDMARK",
        "confidence": 0.92,
        "source": "table_profile_policy",
    }
    raw_hypotheses = cell_hypothesis.get("entity_hypotheses", [])
    hypotheses = raw_hypotheses if isinstance(raw_hypotheses, list) else []
    filtered = [
        item
        for item in hypotheses
        if not (
            isinstance(item, dict)
            and item.get("coarse_type") == "LOCATION"
            and item.get("fine_type") == "LANDMARK"
        )
    ]
    cell_hypothesis["entity_hypotheses"] = [water_hypothesis, *filtered][:3]

    retrieval_plan["hard_filters"] = {
        "item_category": ["ENTITY"],
        "coarse_type": ["LOCATION"],
        "fine_type": ["LANDMARK"],
    }
    retrieval_plan["hypothesis_plan"] = [
        {
            "rank": 1,
            "item_category": "ENTITY",
            "coarse_type": "LOCATION",
            "fine_type": "LANDMARK",
            "confidence": 0.92,
            "retrieval_mode": "hard",
            "source": "table_profile_policy",
        }
    ]

    query_variants = retrieval_plan.get("query_variants", [])
    if not isinstance(query_variants, list):
        query_variants = []
    existing_queries = {
        item.get("text", "").strip().casefold()
        for item in query_variants
        if isinstance(item, dict) and isinstance(item.get("text"), str)
    }
    canonical = cell_hypothesis.get("canonical_mention")
    mention = canonical if isinstance(canonical, str) and canonical.strip() else ""
    mention_tokens = {token.casefold() for token in mention.split()}
    if mention and not (mention_tokens & {"lake", "sea", "river", "ocean", "bay", "gulf", "strait"}):
        lake_variant = f"Lake {mention}".strip()
        if lake_variant.casefold() not in existing_queries:
            query_variants.insert(
                0,
                QueryVariant(text=lake_variant, confidence=0.92, weight=0.96, source="table_profile_policy").to_dict(),
            )
    retrieval_plan["query_variants"] = query_variants

    soft_terms = retrieval_plan.get("soft_context_terms", [])
    if not isinstance(soft_terms, list):
        soft_terms = []
    existing_terms = {
        item.get("value", "").strip().casefold()
        for item in soft_terms
        if isinstance(item, dict) and isinstance(item.get("value"), str)
    }
    for value, weight in (("lake", 0.85), ("body of water", 0.75), ("freshwater lake", 0.7)):
        if value not in existing_terms:
            soft_terms.append(SignalValue(value=value, confidence=0.9, weight=weight, source="table_profile_policy").to_dict())
    retrieval_plan["soft_context_terms"] = soft_terms


def _apply_schema_uncertainty_policy(schema: dict[str, Any]) -> None:
    table_profile = schema.get("table_profile", {})
    context = schema.get("context", {})
    cell_hypothesis = schema.get("cell_hypothesis", {})
    retrieval_plan = schema.get("retrieval_plan", {})
    metadata = schema.get("metadata", {})
    if not isinstance(table_profile, dict) or not isinstance(context, dict) or not isinstance(retrieval_plan, dict):
        return
    if not isinstance(metadata, dict):
        metadata = {}
        schema["metadata"] = metadata
    mention_strength = (
        str(cell_hypothesis.get("mention_strength", "unknown") or "unknown")
        if isinstance(cell_hypothesis, dict)
        else "unknown"
    )
    uncertainty = _schema_uncertainty(table_profile, context, mention_strength)
    metadata["schema_uncertainty"] = uncertainty
    if not bool(uncertainty.get("is_uncertain")):
        return

    hard_filters = retrieval_plan.get("hard_filters", {})
    if isinstance(hard_filters, dict) and hard_filters:
        item_filters = hard_filters.get("item_category") or hard_filters.get("entity_category") or ["ENTITY"]
        retrieval_plan["hard_filters"] = {"item_category": item_filters if isinstance(item_filters, list) else ["ENTITY"]}

    hypothesis_plan = retrieval_plan.get("hypothesis_plan", [])
    if isinstance(hypothesis_plan, list):
        for item in hypothesis_plan:
            if isinstance(item, dict):
                item["retrieval_mode"] = "soft"

    notes = retrieval_plan.get("notes", [])
    if isinstance(notes, str):
        notes = [notes]
    if not isinstance(notes, list):
        notes = []
    reasons = ", ".join(str(item) for item in uncertainty.get("reasons", []) if str(item))
    notes.append(f"Schema uncertainty downgraded type filters to soft guidance: {reasons}")
    retrieval_plan["notes"] = notes


def lookup_payload_from_preprocessing(
    schema: dict[str, Any],
    *,
    top_k: int = 100,
    max_soft_terms: int = 12,
) -> dict[str, object]:
    context = schema.get("context", {})
    cell_hypothesis = schema.get("cell_hypothesis", {})
    retrieval_plan = schema.get("retrieval_plan", {})
    hard_filters = retrieval_plan.get("hard_filters", {})
    soft_terms = retrieval_plan.get("soft_context_terms", [])
    query_variants = retrieval_plan.get("query_variants", [])

    mention = ""
    if isinstance(cell_hypothesis, dict):
        canonical = cell_hypothesis.get("canonical_mention")
        if isinstance(canonical, str) and canonical.strip():
            mention = canonical.strip()
    if not mention and isinstance(context, dict):
        raw_mention = context.get("mention")
        if isinstance(raw_mention, str):
            mention = raw_mention.strip()

    best_variant = mention
    best_weight = -1.0
    if isinstance(query_variants, list):
        for variant in query_variants:
            if not isinstance(variant, dict):
                continue
            text = variant.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            weight = _clamp_unit_interval(variant.get("weight"), 0.0)
            if weight > best_weight:
                best_variant = text.strip()
                best_weight = weight
    if best_variant:
        mention = best_variant

    mention_context: list[str] = []
    if isinstance(soft_terms, list):
        sorted_terms = sorted(
            [item for item in soft_terms if isinstance(item, dict)],
            key=lambda item: _clamp_unit_interval(item.get("weight"), 0.0),
            reverse=True,
        )
        for item in sorted_terms[: max(1, int(max_soft_terms))]:
            weight = _clamp_unit_interval(item.get("weight"), 0.0)
            if weight < 0.3:
                continue
            term = item.get("value")
            if isinstance(term, str) and term.strip() and term.strip() not in mention_context:
                mention_context.append(term.strip())

    if not mention_context and isinstance(context, dict):
        raw_context = context.get("mention_context")
        if isinstance(raw_context, list):
            mention_context = [value for value in raw_context if isinstance(value, str)]

    coarse_hints = []
    fine_hints = []
    crosslink_hints = []
    item_category_filters = []
    if isinstance(hard_filters, dict):
        coarse_hints = [value for value in hard_filters.get("coarse_type", []) if isinstance(value, str)]
        fine_hints = [value for value in hard_filters.get("fine_type", []) if isinstance(value, str)]
        crosslink_hints = [value for value in hard_filters.get("crosslink_hints", []) if isinstance(value, str)]
        item_category_filters = [value for value in hard_filters.get("item_category", []) if isinstance(value, str)]

    payload: dict[str, object] = {
        "mention": mention,
        "mention_context": mention_context,
        "coarse_hints": coarse_hints,
        "fine_hints": fine_hints,
        "crosslink_hints": crosslink_hints,
        "top_k": int(top_k),
        "use_cache": False,
    }
    if item_category_filters:
        payload["item_category_filters"] = item_category_filters
    return payload


def lookup_payload_variants_from_preprocessing(
    schema: dict[str, Any],
    *,
    top_k: int = 100,
    max_variants: int = 5,
) -> list[dict[str, object]]:
    primary_payload = lookup_payload_from_preprocessing(schema, top_k=top_k)
    retrieval_plan = schema.get("retrieval_plan", {})
    cell_hypothesis = schema.get("cell_hypothesis", {})
    variants: list[dict[str, object]] = []
    seen: set[str] = set()

    def add(stage: str, payload: dict[str, object], *, hypothesis_rank: int | None = None, relaxed: bool = False) -> None:
        signature = json.dumps(
            {
                "mention": payload.get("mention"),
                "context": payload.get("mention_context"),
                "coarse": payload.get("coarse_hints"),
                "fine": payload.get("fine_hints"),
                "item": payload.get("item_category_filters"),
            },
            sort_keys=True,
        )
        if signature in seen:
            return
        seen.add(signature)
        variants.append(
            {
                "stage": stage,
                "hypothesis_rank": hypothesis_rank,
                "relaxed": relaxed,
                "payload": payload,
            }
        )

    add("primary", primary_payload, hypothesis_rank=1, relaxed=False)

    hypothesis_plan = retrieval_plan.get("hypothesis_plan", []) if isinstance(retrieval_plan, dict) else []
    if isinstance(hypothesis_plan, list):
        for item in hypothesis_plan[: max(1, int(max_variants))]:
            if not isinstance(item, dict):
                continue
            payload = {
                **primary_payload,
                "coarse_hints": [item.get("coarse_type")] if item.get("coarse_type") else [],
                "fine_hints": [item.get("fine_type")] if item.get("fine_type") and item.get("fine_type") != "MISC" else [],
                "item_category_filters": [item.get("item_category")] if item.get("item_category") else [],
            }
            add(
                f"hypothesis_{int(item.get('rank', 0) or 0)}",
                payload,
                hypothesis_rank=int(item.get("rank", 0) or 0),
                relaxed=str(item.get("retrieval_mode", "soft")) != "hard",
            )
            relaxed_payload = {
                **payload,
                "fine_hints": [],
            }
            add(
                f"hypothesis_{int(item.get('rank', 0) or 0)}_relax_fine",
                relaxed_payload,
                hypothesis_rank=int(item.get("rank", 0) or 0),
                relaxed=True,
            )

    table_profile = schema.get("table_profile", {})
    table_family = str(table_profile.get("table_semantic_family", "") or "")
    mention_strength = str(cell_hypothesis.get("mention_strength", "unknown") or "unknown")
    context_only_payload = {
        **primary_payload,
        "coarse_hints": [],
        "fine_hints": [],
    }
    if mention_strength == "weak" or table_family in {"BIOGRAPHY", "GENERIC_ENTITY_TABLE"}:
        add("no_type_context_only", context_only_payload, relaxed=True)
    if primary_payload.get("item_category_filters"):
        add(
            "item_category_only",
            {
                **context_only_payload,
                "coarse_hints": [],
                "fine_hints": [],
            },
            relaxed=True,
        )
    return variants
