from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from .dataset import load_table


_DATE_RE = re.compile(r"^\d{4}(?:[-/]\d{1,2}(?:[-/]\d{1,2})?)?$")
_NUMERIC_RE = re.compile(r"\d")
_WORD_RE = re.compile(r"[A-Za-z]+")
_COUNTRY_HINTS = {
    "united states",
    "united kingdom",
    "canada",
    "guyana",
    "australia",
    "new zealand",
    "iran",
    "kazakhstan",
    "russia",
    "turkmenistan",
    "azerbaijan",
    "belgium",
    "uganda",
    "kenya",
    "tanzania",
    "france",
    "germany",
    "italy",
    "spain",
    "brazil",
    "india",
    "china",
    "japan",
    "mexico",
}
_BODY_OF_WATER_HINTS = {
    "lake",
    "sea",
    "ocean",
    "river",
    "gulf",
    "bay",
    "strait",
    "basin",
}


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().casefold().split())


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.casefold())


def _looks_date(value: str) -> bool:
    return bool(_DATE_RE.match(value.strip()))


def _looks_numeric(value: str) -> bool:
    stripped = value.strip()
    if not stripped or not _NUMERIC_RE.search(stripped):
        return False
    if not (stripped[:1].isdigit() or stripped[:1] in {"+", "-", "."}):
        return False
    alpha_words = _WORD_RE.findall(stripped)
    return len(alpha_words) <= 4


def _looks_text_blob(value: str) -> bool:
    return len(value.strip()) >= 60 and value.count(" ") >= 8


def _looks_entity_like(value: str) -> bool:
    stripped = value.strip()
    if not stripped or _looks_date(stripped) or _looks_numeric(stripped):
        return False
    tokens = stripped.split()
    alpha_tokens = [token for token in tokens if any(char.isalpha() for char in token)]
    return bool(alpha_tokens)


def _looks_person_name(value: str) -> bool:
    stripped = value.strip()
    if not stripped:
        return False
    if _looks_date(stripped) or _looks_numeric(stripped):
        return False
    tokens = [token for token in stripped.split() if any(char.isalpha() for char in token)]
    if len(tokens) >= 2:
        capitalized = sum(1 for token in tokens if token[:1].isupper())
        return capitalized >= max(2, len(tokens) - 1)
    return stripped[:1].isupper() and stripped.isascii()


def _looks_location_like(value: str) -> bool:
    stripped = value.strip()
    if not stripped or _looks_date(stripped):
        return False
    norm = _normalize_text(stripped)
    if norm in _COUNTRY_HINTS:
        return True
    if "," in stripped or "(" in stripped:
        return True
    tokens = [token for token in stripped.split() if any(char.isalpha() for char in token)]
    return bool(tokens) and all(token[:1].isupper() for token in tokens[: min(3, len(tokens))])


def _contains_body_of_water_hint(value: str) -> bool:
    return bool(set(_tokenize(value)) & _BODY_OF_WATER_HINTS)


@dataclass(frozen=True, slots=True)
class RoleHypothesis:
    role: str
    confidence: float
    source: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ColumnStats:
    col_id: int
    column_name: str
    non_empty_ratio: float
    unique_ratio: float
    date_ratio: float
    numeric_ratio: float
    person_name_ratio: float
    location_like_ratio: float
    text_blob_ratio: float
    entity_like_ratio: float
    body_of_water_hint_ratio: float
    sample_values: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TableProfile:
    dataset_id: str
    table_id: str
    table_semantic_family: str
    confidence: float
    column_roles: dict[str, list[RoleHypothesis]]
    row_template: list[str]
    table_hypotheses: list[RoleHypothesis]
    evidence_notes: list[str]
    sampled_rows: list[list[str]]
    column_stats: list[ColumnStats]
    induction_mode: str = "hybrid"

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "table_id": self.table_id,
            "table_semantic_family": self.table_semantic_family,
            "confidence": self.confidence,
            "column_roles": {
                key: [item.to_dict() for item in values]
                for key, values in self.column_roles.items()
            },
            "row_template": list(self.row_template),
            "table_hypotheses": [item.to_dict() for item in self.table_hypotheses],
            "evidence_notes": list(self.evidence_notes),
            "sampled_rows": [list(row) for row in self.sampled_rows],
            "column_stats": [item.to_dict() for item in self.column_stats],
            "induction_mode": self.induction_mode,
        }


def _ratio(matches: int, total: int) -> float:
    return round((matches / total), 4) if total else 0.0


def _sample_values(rows: list[list[str]], col_id: int, *, limit: int = 6) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if col_id >= len(row):
            continue
        value = row[col_id].strip()
        if not value or value in seen:
            continue
        seen.add(value)
        values.append(value)
        if len(values) >= limit:
            break
    return values


def _build_column_stats(header: list[str], rows: list[list[str]]) -> list[ColumnStats]:
    stats: list[ColumnStats] = []
    row_count = len(rows)
    for col_id, column_name in enumerate(header):
        values = [row[col_id].strip() for row in rows if col_id < len(row) and row[col_id].strip()]
        non_empty = len(values)
        unique = len({_normalize_text(value) for value in values})
        stats.append(
            ColumnStats(
                col_id=col_id,
                column_name=column_name.strip() or f"col{col_id}",
                non_empty_ratio=_ratio(non_empty, row_count),
                unique_ratio=_ratio(unique, non_empty),
                date_ratio=_ratio(sum(1 for value in values if _looks_date(value)), non_empty),
                numeric_ratio=_ratio(sum(1 for value in values if _looks_numeric(value)), non_empty),
                person_name_ratio=_ratio(sum(1 for value in values if _looks_person_name(value)), non_empty),
                location_like_ratio=_ratio(sum(1 for value in values if _looks_location_like(value)), non_empty),
                text_blob_ratio=_ratio(sum(1 for value in values if _looks_text_blob(value)), non_empty),
                entity_like_ratio=_ratio(sum(1 for value in values if _looks_entity_like(value)), non_empty),
                body_of_water_hint_ratio=_ratio(sum(1 for value in values if _contains_body_of_water_hint(value)), non_empty),
                sample_values=_sample_values(rows, col_id),
            )
        )
    return stats


def _country_like_column(stats: ColumnStats) -> bool:
    normalized = {_normalize_text(value) for value in stats.sample_values}
    country_hits = sum(1 for value in normalized if value in _COUNTRY_HINTS)
    return country_hits >= max(1, min(3, len(normalized) // 2)) or (
        stats.location_like_ratio >= 0.85 and stats.unique_ratio <= 0.8 and stats.person_name_ratio < 0.3
    )


def _infer_family(column_stats: list[ColumnStats]) -> tuple[str, float, list[str]]:
    notes: list[str] = []
    date_columns = [item for item in column_stats if item.date_ratio >= 0.65]
    location_columns = [item for item in column_stats if item.location_like_ratio >= 0.65 and item.numeric_ratio < 0.45]
    measure_columns = [item for item in column_stats if item.numeric_ratio >= 0.7 and item.date_ratio < 0.4]
    text_columns = [item for item in column_stats if item.text_blob_ratio >= 0.4]
    personish_first = bool(column_stats and column_stats[0].entity_like_ratio >= 0.85 and column_stats[0].numeric_ratio < 0.2)
    first_person_name = bool(column_stats and column_stats[0].person_name_ratio >= 0.45)
    first_body_of_water = bool(column_stats and column_stats[0].body_of_water_hint_ratio >= 0.3)

    if personish_first and date_columns and len(location_columns) >= 2:
        notes.append("Detected biography-style row template: name-like first column, one date column, multiple location-like columns.")
        confidence = 0.78 + (0.08 if first_person_name else 0.0)
        return "BIOGRAPHY", min(0.92, confidence), notes

    if first_body_of_water and measure_columns and len(measure_columns) >= 2:
        notes.append("Detected geography/landmark table: feature-like first column with several numeric measure columns.")
        return "GEOGRAPHY", 0.84, notes

    if measure_columns and text_columns and location_columns:
        notes.append("Detected structured geography/scientific table from numeric and location columns.")
        return "GEOGRAPHY", 0.68, notes

    if first_person_name and date_columns:
        notes.append("Detected person-centric table with explicit name and date columns.")
        return "BIOGRAPHY", 0.64, notes

    notes.append("Falling back to generic entity table interpretation.")
    return "GENERIC_ENTITY_TABLE", 0.45, notes


def _build_biography_roles(column_stats: list[ColumnStats]) -> tuple[dict[str, list[RoleHypothesis]], list[str], list[RoleHypothesis], list[str]]:
    roles: dict[str, list[RoleHypothesis]] = {}
    row_template: list[str] = []
    hypotheses = [RoleHypothesis(role="PERSON_TABLE", confidence=0.86, source="deterministic")]
    notes: list[str] = ["Biography family implies first target column should prefer human/person candidates."]

    date_index = next((item.col_id for item in column_stats if item.date_ratio >= 0.65), None)
    country_indices = {item.col_id for item in column_stats if _country_like_column(item)}
    name_col_id = next(
        (
            item.col_id
            for item in column_stats
            if item.numeric_ratio < 0.2 and item.date_ratio < 0.2 and item.entity_like_ratio >= 0.75 and (date_index is None or item.col_id < date_index)
        ),
        0,
    )

    for item in column_stats:
        key = str(item.col_id)
        if item.col_id == name_col_id and item.entity_like_ratio >= 0.75:
            roles[key] = [
                RoleHypothesis(role="PERSON_NAME_OR_ALIAS", confidence=0.9 if item.person_name_ratio >= 0.5 else 0.82, source="deterministic"),
                RoleHypothesis(role="ENTITY_NAME", confidence=0.35, source="deterministic"),
            ]
            row_template.append("PERSON_NAME_OR_ALIAS")
            continue
        if date_index is not None and item.col_id == date_index:
            roles[key] = [RoleHypothesis(role="BIRTH_DATE", confidence=0.94, source="deterministic")]
            row_template.append("BIRTH_DATE")
            continue
        if item.col_id in country_indices:
            roles[key] = [RoleHypothesis(role="COUNTRY", confidence=0.86, source="deterministic")]
            row_template.append("COUNTRY")
            continue
        if item.location_like_ratio >= 0.7:
            location_role = "BIRTH_REGION" if any("," in value or "(" in value for value in item.sample_values) else "BIRTH_CITY"
            roles[key] = [RoleHypothesis(role=location_role, confidence=0.78, source="deterministic")]
            row_template.append(location_role)
            continue
        if item.numeric_ratio >= 0.6:
            roles[key] = [RoleHypothesis(role="NUMERIC_ATTRIBUTE", confidence=0.6, source="deterministic")]
            row_template.append("NUMERIC_ATTRIBUTE")
            continue
        roles[key] = [RoleHypothesis(role="TEXT_ATTRIBUTE", confidence=0.45, source="deterministic")]
        row_template.append("TEXT_ATTRIBUTE")
    return roles, row_template, hypotheses, notes


def _build_geography_roles(column_stats: list[ColumnStats]) -> tuple[dict[str, list[RoleHypothesis]], list[str], list[RoleHypothesis], list[str]]:
    roles: dict[str, list[RoleHypothesis]] = {}
    row_template: list[str] = []
    hypotheses = [RoleHypothesis(role="GEOGRAPHIC_ENTITY_TABLE", confidence=0.88, source="deterministic")]
    notes: list[str] = ["Geography family implies landmark/location candidates should dominate entity retrieval."]
    feature_candidates = [
        item
        for item in column_stats
        if item.numeric_ratio < 0.25 and not _country_like_column(item) and item.text_blob_ratio < 0.35
    ]
    if not feature_candidates:
        feature_candidates = [
            item
            for item in column_stats
            if item.numeric_ratio < 0.25 and not _country_like_column(item)
        ]
    feature_col_id = max(
        feature_candidates or column_stats,
        key=lambda item: (
            item.body_of_water_hint_ratio,
            item.entity_like_ratio - item.numeric_ratio - item.text_blob_ratio,
            -item.col_id,
        ),
    ).col_id if column_stats else 0

    measure_role_by_unit = {
        "km2": "AREA_MEASURE",
        "sq": "AREA_MEASURE",
        "km3": "VOLUME_MEASURE",
        "cu": "VOLUME_MEASURE",
        "ft": "DEPTH_MEASURE",
        "m": "DEPTH_MEASURE",
        "length": "LENGTH_MEASURE",
    }

    for item in column_stats:
        key = str(item.col_id)
        joined_samples = " ".join(item.sample_values).casefold()
        if item.col_id == feature_col_id and item.entity_like_ratio >= 0.75 and item.numeric_ratio < 0.25:
            role = "BODY_OF_WATER_NAME" if item.body_of_water_hint_ratio >= 0.2 or any(_contains_body_of_water_hint(value) for value in item.sample_values) else "GEOGRAPHIC_FEATURE_NAME"
            roles[key] = [RoleHypothesis(role=role, confidence=0.88, source="deterministic")]
            row_template.append(role)
            continue
        if _country_like_column(item):
            roles[key] = [RoleHypothesis(role="COUNTRY", confidence=0.82, source="deterministic")]
            row_template.append("COUNTRY")
            continue
        if item.numeric_ratio >= 0.7:
            role = "NUMERIC_MEASURE"
            for unit, mapped_role in measure_role_by_unit.items():
                if unit in joined_samples:
                    role = mapped_role
                    break
            roles[key] = [RoleHypothesis(role=role, confidence=0.74, source="deterministic")]
            row_template.append(role)
            continue
        if item.text_blob_ratio >= 0.4:
            roles[key] = [RoleHypothesis(role="DESCRIPTION_TEXT", confidence=0.84, source="deterministic")]
            row_template.append("DESCRIPTION_TEXT")
            continue
        roles[key] = [RoleHypothesis(role="TEXT_ATTRIBUTE", confidence=0.45, source="deterministic")]
        row_template.append("TEXT_ATTRIBUTE")
    return roles, row_template, hypotheses, notes


def build_table_profile_seed(
    dataset_root: Any,
    dataset_id: str,
    table_id: str,
    *,
    sample_rows: int = 10,
) -> TableProfile:
    rows = load_table(dataset_root, dataset_id, table_id)
    if not rows:
        raise ValueError(f"Table is empty: {table_id}")
    header = rows[0]
    data_rows = rows[1 : 1 + max(1, int(sample_rows))]
    column_stats = _build_column_stats(header, data_rows)
    family, confidence, notes = _infer_family(column_stats)

    if family == "BIOGRAPHY":
        column_roles, row_template, table_hypotheses, role_notes = _build_biography_roles(column_stats)
    elif family == "GEOGRAPHY":
        column_roles, row_template, table_hypotheses, role_notes = _build_geography_roles(column_stats)
    else:
        column_roles = {
            str(item.col_id): [RoleHypothesis(role="UNKNOWN", confidence=0.3, source="deterministic")]
            for item in column_stats
        }
        row_template = ["UNKNOWN" for _ in column_stats]
        table_hypotheses = [RoleHypothesis(role="GENERIC_ENTITY_TABLE", confidence=0.45, source="deterministic")]
        role_notes = ["No strong family-specific row template inferred."]

    return TableProfile(
        dataset_id=dataset_id,
        table_id=table_id,
        table_semantic_family=family,
        confidence=round(confidence, 4),
        column_roles=column_roles,
        row_template=row_template,
        table_hypotheses=table_hypotheses,
        evidence_notes=notes + role_notes,
        sampled_rows=[header, *data_rows],
        column_stats=column_stats,
        induction_mode="hybrid",
    )


def normalize_table_profile(profile: dict[str, Any], *, dataset_id: str, table_id: str) -> dict[str, Any]:
    normalized = {
        "dataset_id": dataset_id,
        "table_id": table_id,
        "table_semantic_family": str(profile.get("table_semantic_family", "GENERIC_ENTITY_TABLE") or "GENERIC_ENTITY_TABLE"),
        "confidence": max(0.0, min(1.0, float(profile.get("confidence", 0.0) or 0.0))),
        "column_roles": {},
        "row_template": [],
        "table_hypotheses": [],
        "evidence_notes": [],
        "sampled_rows": profile.get("sampled_rows", []),
        "column_stats": profile.get("column_stats", []),
        "induction_mode": str(profile.get("induction_mode", "hybrid") or "hybrid"),
    }

    raw_column_roles = profile.get("column_roles", {})
    if isinstance(raw_column_roles, dict):
        for key, raw_values in raw_column_roles.items():
            values = raw_values if isinstance(raw_values, list) else [raw_values]
            normalized_values: list[dict[str, object]] = []
            for item in values:
                if isinstance(item, str):
                    role = item.strip()
                    if not role:
                        continue
                    normalized_values.append(RoleHypothesis(role=role, confidence=0.6, source="llm").to_dict())
                    continue
                if not isinstance(item, dict):
                    continue
                role = item.get("role") or item.get("value")
                if not isinstance(role, str) or not role.strip():
                    continue
                confidence = item.get("confidence", item.get("probability", 0.6))
                try:
                    confidence_value = max(0.0, min(1.0, float(confidence)))
                except (TypeError, ValueError):
                    confidence_value = 0.6
                source = item.get("source") if isinstance(item.get("source"), str) else "llm"
                normalized_values.append(RoleHypothesis(role=role.strip(), confidence=confidence_value, source=source).to_dict())
            if normalized_values:
                normalized["column_roles"][str(key)] = normalized_values

    raw_row_template = profile.get("row_template", [])
    if isinstance(raw_row_template, str):
        normalized["row_template"] = [part.strip() for part in raw_row_template.split("|") if part.strip()]
    elif isinstance(raw_row_template, list):
        normalized["row_template"] = [str(item).strip() for item in raw_row_template if str(item).strip()]

    raw_hypotheses = profile.get("table_hypotheses", [])
    if isinstance(raw_hypotheses, list):
        for item in raw_hypotheses:
            if isinstance(item, str) and item.strip():
                normalized["table_hypotheses"].append(RoleHypothesis(role=item.strip(), confidence=0.6, source="llm").to_dict())
                continue
            if not isinstance(item, dict):
                continue
            role = item.get("role") or item.get("value")
            if not isinstance(role, str) or not role.strip():
                continue
            confidence = item.get("confidence", item.get("probability", 0.6))
            try:
                confidence_value = max(0.0, min(1.0, float(confidence)))
            except (TypeError, ValueError):
                confidence_value = 0.6
            source = item.get("source") if isinstance(item.get("source"), str) else "llm"
            normalized["table_hypotheses"].append(RoleHypothesis(role=role.strip(), confidence=confidence_value, source=source).to_dict())

    raw_notes = profile.get("evidence_notes", [])
    if isinstance(raw_notes, str):
        normalized["evidence_notes"] = [raw_notes]
    elif isinstance(raw_notes, list):
        normalized["evidence_notes"] = [str(item) for item in raw_notes if str(item).strip()]
    return normalized
