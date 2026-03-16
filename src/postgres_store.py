from __future__ import annotations

import json
import math
import re
import zlib
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from .common import normalize_text


try:  # pragma: no cover - exercised in integration environments
    import psycopg  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    psycopg = None  # type: ignore


class PostgresStoreError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class EntityRecord:
    qid: str
    label: str
    labels: Mapping[str, str]
    aliases: Mapping[str, Sequence[str]]
    description: str | None
    types: Sequence[str]
    item_category: str
    coarse_type: str
    fine_type: str
    popularity: float
    cross_refs: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class EntityTripleRecord:
    subject_qid: str
    predicate_pid: str
    object_qid: str


_SQL_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_WIKIPEDIA_PREFIX = "https://en.wikipedia.org/wiki/"
_DBPEDIA_PREFIX = "https://dbpedia.org/resource/"
_WIKIPEDIA_DEFAULT_HOST = "en.wikipedia.org"
_DBPEDIA_DEFAULT_HOST = "dbpedia.org"
_HOSTED_REF_SEPARATOR = "|"
_PRIMARY_LABEL_LANGUAGE_PREFERENCE = ("en", "mul")
_EMPTY_ENTITY_NAME_PAYLOAD = '{"aliases":[],"labels":[]}'
_EMPTY_ENTITY_NAME_PAYLOAD_BYTES = zlib.compress(_EMPTY_ENTITY_NAME_PAYLOAD.encode("utf-8"), level=9)
_EMPTY_ENTITY_NAME_PAYLOAD_HEX = _EMPTY_ENTITY_NAME_PAYLOAD_BYTES.hex()
MAX_STORED_LABELS = 64
MAX_STORED_ALIASES = 128
DEFAULT_INDEX_PROFILE = "lean"
VALID_INDEX_PROFILES = frozenset({"lean", "full"})
DEFAULT_MAX_CONTEXT_CHARS = 512
_SIM_SAMPLE_MULTIPLIER = 1_103_515_245
_SIM_SAMPLE_SEED_MULTIPLIER = 97_531
_SIM_SAMPLE_INCREMENT = 12_345
_SIM_SAMPLE_MODULUS = 2_147_483_647


def entity_name_payload_table_name(base_table_name: str) -> str:
    stripped = base_table_name.strip()
    if not stripped:
        raise ValueError("Base table name must be non-empty.")
    return f"{stripped}_name_payloads"


def entity_context_inputs_table_name(base_table_name: str) -> str:
    stripped = base_table_name.strip()
    if not stripped:
        raise ValueError("Base table name must be non-empty.")
    return f"{stripped}_context_inputs"


def _quote_identifier(name: str) -> str:
    if not isinstance(name, str):
        raise ValueError("SQL identifier must be a string.")
    stripped = name.strip()
    if not _SQL_IDENTIFIER_RE.match(stripped):
        raise ValueError(
            f"Invalid SQL identifier '{name}'. Use letters, digits, and underscores only."
        )
    return f'"{stripped}"'


def sampled_seed_row_number(*, sample_no: int, seed_count: int, random_seed: int) -> int:
    if sample_no < 0:
        raise ValueError("sample_no must be >= 0")
    if seed_count <= 0:
        raise ValueError("seed_count must be > 0")
    if random_seed < 0:
        raise ValueError("random_seed must be >= 0")
    mixed = (
        ((sample_no + 1) * _SIM_SAMPLE_MULTIPLIER)
        + ((random_seed + 1) * _SIM_SAMPLE_SEED_MULTIPLIER)
        + _SIM_SAMPLE_INCREMENT
    ) % _SIM_SAMPLE_MODULUS
    return (mixed % seed_count) + 1


def _json_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _compact_wikipedia_ref(value: str) -> str:
    raw = value.strip()
    if not raw:
        return ""
    if raw.startswith(_WIKIPEDIA_PREFIX):
        return raw[len(_WIKIPEDIA_PREFIX):]
    if raw.startswith("http://") or raw.startswith("https://"):
        parsed = urlsplit(raw)
        marker = "/wiki/"
        if parsed.netloc and parsed.path and marker in parsed.path:
            title = parsed.path.split(marker, 1)[1]
            host = parsed.netloc.strip().lower()
            if not host:
                return title
            if host == _WIKIPEDIA_DEFAULT_HOST:
                return title
            return f"{host}{_HOSTED_REF_SEPARATOR}{title}"
    return raw


def _compact_dbpedia_ref(value: str) -> str:
    raw = value.strip()
    if not raw:
        return ""
    if raw.startswith(_DBPEDIA_PREFIX):
        return raw[len(_DBPEDIA_PREFIX):]
    if raw.startswith("http://") or raw.startswith("https://"):
        parsed = urlsplit(raw)
        marker = "/resource/"
        if parsed.netloc and parsed.path and marker in parsed.path:
            title = parsed.path.split(marker, 1)[1]
            host = parsed.netloc.strip().lower()
            if not host:
                return title
            if host == _DBPEDIA_DEFAULT_HOST:
                return title
            return f"{host}{_HOSTED_REF_SEPARATOR}{title}"
    return raw


def compact_crosslink_hint(value: str) -> str:
    raw = value.strip()
    if not raw:
        return ""
    compacted = _compact_wikipedia_ref(raw)
    if compacted != raw:
        return compacted
    compacted = _compact_dbpedia_ref(raw)
    if compacted != raw:
        return compacted
    return raw


def _expand_wikipedia_ref(value: str) -> str:
    raw = value.strip()
    if not raw:
        return ""
    if raw.startswith("http://") or raw.startswith("https://"):
        return raw
    if _HOSTED_REF_SEPARATOR in raw:
        host, title = raw.split(_HOSTED_REF_SEPARATOR, 1)
        host = host.strip()
        title = title.strip()
        if host and title:
            return f"https://{host}/wiki/{title}"
    if raw.startswith("/wiki/"):
        return f"https://en.wikipedia.org{raw}"
    return f"{_WIKIPEDIA_PREFIX}{raw}"


def _expand_dbpedia_ref(value: str) -> str:
    raw = value.strip()
    if not raw:
        return ""
    if raw.startswith("http://") or raw.startswith("https://"):
        return raw
    if _HOSTED_REF_SEPARATOR in raw:
        host, title = raw.split(_HOSTED_REF_SEPARATOR, 1)
        host = host.strip()
        title = title.strip()
        if host and title:
            return f"https://{host}/resource/{title}"
    if raw.startswith("/resource/"):
        return f"https://dbpedia.org{raw}"
    return f"{_DBPEDIA_PREFIX}{raw}"


def _require_psycopg() -> Any:
    if psycopg is None:
        raise PostgresStoreError(
            "psycopg is not installed. Install requirements with PostgreSQL support "
            "to use the Postgres entity/cache backend."
        )
    return psycopg


def _ordered_language_keys(values: Mapping[str, Any]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for language in _PRIMARY_LABEL_LANGUAGE_PREFERENCE:
        if language in values and language not in seen:
            seen.add(language)
            ordered.append(language)
    for language in sorted(values):
        if language in seen:
            continue
        ordered.append(language)
    return ordered


def _as_text_map(raw: Any) -> dict[str, str]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return {}
    if not isinstance(raw, Mapping):
        return {}
    output: dict[str, str] = {}
    for key, value in raw.items():
        if isinstance(key, str) and isinstance(value, str):
            output[key] = value
    return output


def _as_alias_map(raw: Any) -> dict[str, list[str]]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return {}
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


def _as_str_list(raw: Any) -> list[str]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return []
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, str)]


def _as_json_object(raw: Any) -> dict[str, Any]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return {}
    if not isinstance(raw, Mapping):
        return {}
    return {str(key): value for key, value in raw.items() if isinstance(key, str)}


def _extract_sample_entity_label(raw_entity_json: Any) -> str | None:
    if isinstance(raw_entity_json, str):
        try:
            raw_entity_json = json.loads(raw_entity_json)
        except json.JSONDecodeError:
            return None
    if not isinstance(raw_entity_json, Mapping):
        return None

    raw_labels = raw_entity_json.get("labels")
    if not isinstance(raw_labels, Mapping):
        return None

    preferred: dict[str, str] = {}
    for language, payload in raw_labels.items():
        if not isinstance(language, str) or not isinstance(payload, Mapping):
            continue
        raw_value = payload.get("value")
        if not isinstance(raw_value, str):
            continue
        normalized = normalize_text(raw_value)
        if not normalized:
            continue
        preferred[language] = normalized

    for language in _ordered_language_keys(preferred):
        value = preferred.get(language)
        if value:
            return value
    return None


def _popularity_to_prior_local(popularity: float) -> float:
    value = max(0.0, float(popularity))
    return 1.0 - math.exp(-math.log1p(value) / 6.0)


def _flatten_labels_map(labels: Mapping[str, str]) -> list[str]:
    flattened: list[str] = []
    seen: set[str] = set()
    for language in _ordered_language_keys(labels):
        raw_value = labels.get(language)
        if not isinstance(raw_value, str):
            continue
        value = normalize_text(raw_value)
        if not value or value in seen:
            continue
        seen.add(value)
        flattened.append(value)
    return flattened


def _flatten_aliases_map(
    aliases: Mapping[str, Sequence[str]],
    *,
    excluded: set[str] | None = None,
) -> list[str]:
    flattened: list[str] = []
    seen: set[str] = set()
    blocked = excluded if excluded is not None else set()
    for language in _ordered_language_keys(aliases):
        values = aliases.get(language, [])
        for raw_alias in values:
            if not isinstance(raw_alias, str):
                continue
            alias = normalize_text(raw_alias)
            if not alias or alias in seen or alias in blocked:
                continue
            seen.add(alias)
            flattened.append(alias)
    return flattened


def _join_terms(values: Sequence[str]) -> str:
    return " ".join(value for value in values if isinstance(value, str) and value)


def _limit_terms(values: Sequence[str], *, max_terms: int) -> list[str]:
    if max_terms <= 0:
        return []
    return [value for index, value in enumerate(values) if index < max_terms]


def _normalize_name_terms(
    values: Sequence[str],
    *,
    excluded: set[str] | None = None,
) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    blocked = excluded if excluded is not None else set()
    for raw_value in values:
        if not isinstance(raw_value, str):
            continue
        value = normalize_text(raw_value)
        if not value or value in seen or value in blocked:
            continue
        seen.add(value)
        normalized.append(value)
    return normalized


def _build_entity_name_sets(
    *,
    labels: Mapping[str, str],
    aliases: Mapping[str, Sequence[str]],
) -> tuple[list[str], list[str]]:
    labels_flat = _flatten_labels_map(labels)
    aliases_flat = _flatten_aliases_map(aliases, excluded=set(labels_flat))
    return labels_flat, aliases_flat


def _build_search_texts_from_name_sets(
    *,
    label: str,
    labels: Sequence[str],
    aliases: Sequence[str],
) -> tuple[str, str]:
    primary_label_normalized = normalize_text(label) if isinstance(label, str) else ""
    secondary_labels = [
        value
        for value in _normalize_name_terms(labels)
        if value and (not primary_label_normalized or value != primary_label_normalized)
    ]
    aliases_flat = _normalize_name_terms(aliases, excluded=set(_normalize_name_terms(labels)))
    return _join_terms(secondary_labels), _join_terms(aliases_flat)


def _legacy_label_values(label: Any, raw_labels: Any) -> list[str]:
    labels = _normalize_name_terms(_as_str_list(raw_labels))
    primary_label = normalize_text(label) if isinstance(label, str) else ""
    if not primary_label:
        return labels
    if primary_label in labels:
        return [primary_label, *[value for value in labels if value != primary_label]]
    return [primary_label, *labels]


def _encode_entity_name_payload(
    *,
    labels: Sequence[str],
    aliases: Sequence[str],
) -> bytes:
    normalized_labels = _normalize_name_terms(labels)
    normalized_aliases = _normalize_name_terms(aliases, excluded=set(normalized_labels))
    if not normalized_labels and not normalized_aliases:
        return _EMPTY_ENTITY_NAME_PAYLOAD_BYTES
    payload = _json_compact(
        {
            "labels": normalized_labels,
            "aliases": normalized_aliases,
        }
    )
    return zlib.compress(payload.encode("utf-8"), level=9)


def _decode_entity_document_payload(raw: Any) -> dict[str, Any]:
    if isinstance(raw, memoryview):
        raw = raw.tobytes()
    if isinstance(raw, bytearray):
        raw = bytes(raw)
    if isinstance(raw, bytes):
        if not raw:
            return [], []
        try:
            raw = zlib.decompress(raw).decode("utf-8")
        except zlib.error:
            try:
                raw = raw.decode("utf-8")
            except UnicodeDecodeError:
                return {
                    "labels": [],
                    "aliases": [],
                    "description": None,
                    "types": [],
                    "coarse_type": "",
                    "fine_type": "",
                    "item_category": "",
                    "popularity": 0.0,
                    "prior": 0.0,
                    "wikipedia_url": "",
                    "dbpedia_url": "",
                }
    payload = _as_json_object(raw)
    labels = _normalize_name_terms(_as_str_list(payload.get("labels")))
    aliases = _normalize_name_terms(_as_str_list(payload.get("aliases")), excluded=set(labels))
    description = payload.get("description")
    raw_types = payload.get("types")
    types = [value for value in _as_str_list(raw_types) if value]
    coarse_type = payload.get("coarse_type") if isinstance(payload.get("coarse_type"), str) else ""
    fine_type = payload.get("fine_type") if isinstance(payload.get("fine_type"), str) else ""
    item_category = payload.get("item_category") if isinstance(payload.get("item_category"), str) else ""
    popularity = payload.get("popularity")
    prior = payload.get("prior")
    wikipedia_url = payload.get("wikipedia_url") if isinstance(payload.get("wikipedia_url"), str) else ""
    dbpedia_url = payload.get("dbpedia_url") if isinstance(payload.get("dbpedia_url"), str) else ""
    return {
        "labels": labels,
        "aliases": aliases,
        "description": description if isinstance(description, str) else None,
        "types": types,
        "coarse_type": coarse_type,
        "fine_type": fine_type,
        "item_category": item_category,
        "popularity": float(popularity) if isinstance(popularity, (int, float)) else 0.0,
        "prior": float(prior) if isinstance(prior, (int, float)) else 0.0,
        "wikipedia_url": wikipedia_url,
        "dbpedia_url": dbpedia_url,
    }


def _decode_entity_name_payload(raw: Any) -> tuple[list[str], list[str]]:
    payload = _decode_entity_document_payload(raw)
    labels = payload.get("labels")
    aliases = payload.get("aliases")
    return (
        [value for value in labels if isinstance(value, str)] if isinstance(labels, list) else [],
        [value for value in aliases if isinstance(value, str)] if isinstance(aliases, list) else [],
    )


def _encode_entity_document_payload(
    *,
    labels: Sequence[str],
    aliases: Sequence[str],
    description: str | None,
    types: Sequence[str],
    coarse_type: str,
    fine_type: str,
    item_category: str,
    popularity: float,
    prior: float,
    wikipedia_url: str,
    dbpedia_url: str,
) -> bytes:
    normalized_labels = _normalize_name_terms(labels)
    normalized_aliases = _normalize_name_terms(aliases, excluded=set(normalized_labels))
    payload: dict[str, Any] = {
        "labels": normalized_labels,
        "aliases": normalized_aliases,
        "types": [value for value in types if isinstance(value, str) and value],
        "coarse_type": coarse_type if isinstance(coarse_type, str) else "",
        "fine_type": fine_type if isinstance(fine_type, str) else "",
        "item_category": item_category if isinstance(item_category, str) else "",
        "popularity": float(popularity),
        "prior": float(prior),
        "wikipedia_url": wikipedia_url if isinstance(wikipedia_url, str) else "",
        "dbpedia_url": dbpedia_url if isinstance(dbpedia_url, str) else "",
    }
    if isinstance(description, str) and description:
        payload["description"] = description
    return zlib.compress(_json_compact(payload).encode("utf-8"), level=9)


def _entity_search_columns(
    *,
    label: str,
    labels: Mapping[str, str],
    aliases: Mapping[str, Sequence[str]],
    cross_refs: Mapping[str, Any],
    popularity: float,
) -> dict[str, Any]:
    labels_flat_all, aliases_flat_all = _build_entity_name_sets(labels=labels, aliases=aliases)
    labels_flat = _limit_terms(labels_flat_all, max_terms=MAX_STORED_LABELS)
    aliases_flat = _limit_terms(aliases_flat_all, max_terms=MAX_STORED_ALIASES)
    wikipedia_url = ""
    raw_wikipedia = cross_refs.get("wikipedia")
    if isinstance(raw_wikipedia, str):
        wikipedia_url = _compact_wikipedia_ref(raw_wikipedia)
    dbpedia_url = ""
    raw_dbpedia = cross_refs.get("dbpedia")
    if isinstance(raw_dbpedia, str):
        dbpedia_url = _compact_dbpedia_ref(raw_dbpedia)
    return {
        "prior": _popularity_to_prior_local(popularity),
        "labels": labels_flat,
        "aliases": aliases_flat,
        "wikipedia_url": wikipedia_url,
        "dbpedia_url": dbpedia_url,
    }


def build_entity_context_string(
    *,
    related_labels: Sequence[str],
    max_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
) -> str:
    if max_chars <= 0:
        return ""
    values: list[str] = []
    seen: set[str] = set()
    current_len = 0
    for raw_value in related_labels:
        if not isinstance(raw_value, str):
            continue
        value = raw_value.strip()
        if not value or value in seen:
            continue
        extra = len(value) if not values else len(value) + 2
        if values and current_len + extra > max_chars:
            break
        if not values and len(value) > max_chars:
            values.append(value[:max_chars])
            break
        seen.add(value)
        values.append(value)
        current_len += extra
    return "; ".join(values)


class PostgresStore:
    def __init__(self, dsn: str) -> None:
        self.dsn = dsn.strip()
        if not self.dsn:
            raise ValueError("Postgres DSN must be non-empty.")

    def _connect(self) -> Any:
        pg = _require_psycopg()
        try:
            return pg.connect(self.dsn)
        except Exception as exc:  # pragma: no cover - connection issues are environment-specific
            raise PostgresStoreError(f"Could not connect to Postgres: {exc}") from exc

    def _table_exists(self, conn: Any, table_name: str) -> bool:
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass(%s)", (table_name,))
            row = cur.fetchone()
        return bool(row and row[0])

    def _column_data_type(self, conn: Any, *, table_name: str, column_name: str) -> str:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT data_type
                FROM information_schema.columns
                WHERE table_schema = current_schema()
                  AND table_name = %s
                  AND column_name = %s
                """,
                (table_name, column_name),
            )
            row = cur.fetchone()
        return row[0] if row and isinstance(row[0], str) else ""

    def _update_entity_name_rows(
        self,
        conn: Any,
        rows: Sequence[tuple[str, list[str], list[str]]],
    ) -> None:
        if not rows:
            return
        sql = """
        UPDATE entities
        SET labels = %s::text[],
            aliases = %s::text[],
            updated_at = NOW()
        WHERE qid = %s
          AND COALESCE(array_length(labels, 1), 0) = 0
          AND COALESCE(array_length(aliases, 1), 0) = 0
        """
        with conn.cursor() as cur:
            cur.executemany(sql, [(labels, aliases, qid) for qid, labels, aliases in rows])

    def _migrate_legacy_entity_names_to_entities(
        self,
        conn: Any,
        *,
        entity_columns: set[str],
    ) -> None:
        aliases_column = "aliases" if "aliases" in entity_columns else ""
        if not aliases_column and "name_variants" in entity_columns:
            aliases_column = "name_variants"

        legacy_labels_column = "legacy_labels" if "legacy_labels" in entity_columns else ""
        if not legacy_labels_column and "labels_json" in entity_columns:
            legacy_labels_column = "labels_json"
        payload_type = self._column_data_type(conn, table_name="entities", column_name="name_payload")

        select_columns = [
            "qid",
            "label",
            "labels" if "labels" in entity_columns else "ARRAY[]::text[] AS labels",
            "aliases" if "aliases" in entity_columns else "ARRAY[]::text[] AS aliases",
            legacy_labels_column if legacy_labels_column else "NULL::text[] AS legacy_labels",
            aliases_column if aliases_column else "NULL::text[] AS legacy_aliases",
            "name_payload" if payload_type else "NULL AS name_payload",
        ]
        read_cur = conn.cursor(name="alpaca_entities_legacy_name_migration")
        read_cur.itersize = 10_000
        read_cur.execute(f"SELECT {', '.join(select_columns)} FROM entities ORDER BY qid")
        try:
            while True:
                rows = read_cur.fetchmany(10_000)
                if not rows:
                    break
                name_rows: list[tuple[str, list[str], list[str]]] = []
                for qid, label, stored_labels, stored_aliases, raw_labels, raw_aliases, raw_payload in rows:
                    if not isinstance(qid, str):
                        continue
                    normalized_stored_labels = _normalize_name_terms(_as_str_list(stored_labels))
                    normalized_stored_aliases = _normalize_name_terms(
                        _as_str_list(stored_aliases),
                        excluded=set(normalized_stored_labels),
                    )
                    if normalized_stored_labels or normalized_stored_aliases:
                        continue
                    payload_labels: list[str] = []
                    payload_aliases: list[str] = []
                    if payload_type:
                        payload_labels, payload_aliases = _decode_entity_name_payload(raw_payload)
                    labels = payload_labels or _legacy_label_values(label, raw_labels)
                    aliases = payload_aliases or _normalize_name_terms(
                        _as_str_list(raw_aliases),
                        excluded=set(labels),
                    )
                    name_rows.append(
                        (
                            qid,
                            _limit_terms(labels, max_terms=MAX_STORED_LABELS),
                            _limit_terms(aliases, max_terms=MAX_STORED_ALIASES),
                        )
                    )
                self._update_entity_name_rows(conn, name_rows)
        finally:
            read_cur.close()

    def ensure_schema(self) -> None:
        ddl = """
        CREATE TABLE IF NOT EXISTS entities (
            qid TEXT PRIMARY KEY,
            label TEXT NOT NULL,
            labels TEXT[] NOT NULL DEFAULT ARRAY[]::text[],
            aliases TEXT[] NOT NULL DEFAULT ARRAY[]::text[],
            description TEXT,
            types TEXT[] NOT NULL DEFAULT ARRAY[]::text[],
            coarse_type TEXT NOT NULL DEFAULT '',
            fine_type TEXT NOT NULL DEFAULT '',
            item_category TEXT NOT NULL DEFAULT '',
            popularity DOUBLE PRECISION NOT NULL DEFAULT 0,
            prior DOUBLE PRECISION NOT NULL DEFAULT 0,
            wikipedia_url TEXT NOT NULL DEFAULT '',
            dbpedia_url TEXT NOT NULL DEFAULT '',
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS entity_triples (
            subject_qid TEXT NOT NULL,
            predicate_pid TEXT NOT NULL,
            object_qid TEXT NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (subject_qid, predicate_pid, object_qid)
        );

        CREATE TABLE IF NOT EXISTS query_cache (
            cache_key TEXT PRIMARY KEY,
            result JSONB NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS idx_query_cache_created_at ON query_cache (created_at);

        CREATE TABLE IF NOT EXISTS sample_entity_cache (
            qid TEXT PRIMARY KEY,
            entity_json JSONB NOT NULL,
            source_url TEXT NOT NULL DEFAULT '',
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );

        ALTER TABLE entities ADD COLUMN IF NOT EXISTS description TEXT;
        ALTER TABLE entities ADD COLUMN IF NOT EXISTS labels TEXT[] NOT NULL DEFAULT ARRAY[]::text[];
        ALTER TABLE entities ADD COLUMN IF NOT EXISTS aliases TEXT[] NOT NULL DEFAULT ARRAY[]::text[];
        ALTER TABLE entities ADD COLUMN IF NOT EXISTS types TEXT[] NOT NULL DEFAULT ARRAY[]::text[];
        ALTER TABLE entities ADD COLUMN IF NOT EXISTS coarse_type TEXT NOT NULL DEFAULT '';
        ALTER TABLE entities ADD COLUMN IF NOT EXISTS fine_type TEXT NOT NULL DEFAULT '';
        ALTER TABLE entities ADD COLUMN IF NOT EXISTS item_category TEXT NOT NULL DEFAULT '';
        ALTER TABLE entities ADD COLUMN IF NOT EXISTS popularity DOUBLE PRECISION NOT NULL DEFAULT 0;
        ALTER TABLE entities ADD COLUMN IF NOT EXISTS prior DOUBLE PRECISION NOT NULL DEFAULT 0;
        ALTER TABLE entities ADD COLUMN IF NOT EXISTS wikipedia_url TEXT NOT NULL DEFAULT '';
        ALTER TABLE entities ADD COLUMN IF NOT EXISTS dbpedia_url TEXT NOT NULL DEFAULT '';
        ALTER TABLE entities DROP COLUMN IF EXISTS context_string;
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(ddl)
                cur.execute(
                    """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_schema = current_schema()
                      AND table_name = 'entities'
                    """
                )
                entity_columns = {
                    row[0]
                    for row in cur.fetchall()
                    if row and isinstance(row[0], str)
                }
            self._migrate_legacy_entity_names_to_entities(conn, entity_columns=entity_columns)
            with conn.cursor() as cur:
                for index_name in (
                    "idx_entities_search_vector",
                    "idx_entities_label_trgm",
                    "idx_entities_aliases_trgm",
                    "idx_entities_aliases_text_trgm",
                    "idx_entities_name_variants_trgm",
                    "idx_entities_label_exact",
                    "idx_entities_aliases_exact",
                    "idx_entities_cross_refs_exact",
                    "idx_entities_cross_refs_text_trgm",
                    "idx_entities_cross_refs_url_trgm",
                ):
                    cur.execute(f"DROP INDEX IF EXISTS {_quote_identifier(index_name)}")
                cur.execute("DROP FUNCTION IF EXISTS alpaca_join_text_array(TEXT[])")
                cur.execute("DROP FUNCTION IF EXISTS alpaca_filter_fts_text(TEXT)")
                cur.execute("DROP FUNCTION IF EXISTS alpaca_filter_fts_context(TEXT)")
                cur.execute("DROP EXTENSION IF EXISTS pg_trgm CASCADE")
                cur.execute("ALTER TABLE entities DROP COLUMN IF EXISTS name_variants")
                cur.execute("ALTER TABLE entities DROP COLUMN IF EXISTS name_payload")
                cur.execute("ALTER TABLE entities DROP COLUMN IF EXISTS relation_object_qids")
                cur.execute("ALTER TABLE entities DROP COLUMN IF EXISTS search_vector")
                cur.execute("DROP TABLE IF EXISTS entity_name_payloads")
                cur.execute("DROP TABLE IF EXISTS entity_context_inputs")
            conn.commit()

    def ensure_search_indexes(
        self,
        table_name: str = "entities",
        *,
        index_profile: str = DEFAULT_INDEX_PROFILE,
    ) -> None:
        table_ident = _quote_identifier(table_name)
        index_prefix = f"idx_{table_name}"
        normalized_profile = normalize_text(index_profile).lower()
        if normalized_profile not in VALID_INDEX_PROFILES:
            raise ValueError(
                f"Unsupported index profile '{index_profile}'. Expected one of: "
                f"{', '.join(sorted(VALID_INDEX_PROFILES))}."
            )
        drop_parts = [
            f"DROP INDEX IF EXISTS {_quote_identifier(f'{index_prefix}_item_category')};",
        ]
        if table_name == "entities":
            drop_parts.extend(
                [
                    "DROP INDEX IF EXISTS idx_entity_triples_subject_qid;",
                    "DROP INDEX IF EXISTS idx_entity_triples_object_qid;",
                    "DROP INDEX IF EXISTS idx_entity_triples_predicate_pid;",
                ]
            )

        ddl_parts = [f"""
        CREATE INDEX IF NOT EXISTS {index_prefix}_coarse_type ON {table_ident} (coarse_type);
        CREATE INDEX IF NOT EXISTS {index_prefix}_fine_type ON {table_ident} (fine_type);
        CREATE INDEX IF NOT EXISTS {index_prefix}_label_lower ON {table_ident} (LOWER(label));
        CREATE INDEX IF NOT EXISTS {index_prefix}_updated_at ON {table_ident} (updated_at);
        CREATE INDEX IF NOT EXISTS {index_prefix}_wikipedia_url ON {table_ident} (wikipedia_url)
        WHERE COALESCE(wikipedia_url, '') <> '';
        CREATE INDEX IF NOT EXISTS {index_prefix}_dbpedia_url ON {table_ident} (dbpedia_url)
        WHERE COALESCE(dbpedia_url, '') <> '';
        """]
        ddl = "\n".join([*drop_parts, *ddl_parts])
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(ddl)
            conn.commit()

    def upsert_entities(
        self,
        rows: Sequence[EntityRecord],
    ) -> int:
        if not rows:
            return 0
        sql = """
        INSERT INTO entities (
            qid, label, labels, aliases, description, types, coarse_type, fine_type,
            item_category, popularity, prior, wikipedia_url, dbpedia_url
        ) VALUES (
            %s, %s, %s::text[], %s::text[], %s, %s::text[], %s, %s,
            %s, %s, %s, %s, %s
        )
        ON CONFLICT (qid) DO UPDATE SET
            label = EXCLUDED.label,
            labels = EXCLUDED.labels,
            aliases = EXCLUDED.aliases,
            description = EXCLUDED.description,
            types = EXCLUDED.types,
            coarse_type = EXCLUDED.coarse_type,
            fine_type = EXCLUDED.fine_type,
            item_category = EXCLUDED.item_category,
            popularity = EXCLUDED.popularity,
            prior = EXCLUDED.prior,
            wikipedia_url = EXCLUDED.wikipedia_url,
            dbpedia_url = EXCLUDED.dbpedia_url,
            updated_at = NOW()
        """
        payload: list[tuple[Any, ...]] = []
        for row in rows:
            search_cols = _entity_search_columns(
                label=row.label,
                labels=row.labels,
                aliases=row.aliases,
                cross_refs=row.cross_refs,
                popularity=float(row.popularity),
            )
            payload.append(
                (
                    row.qid,
                    row.label,
                    list(search_cols["labels"]),
                    list(search_cols["aliases"]),
                    row.description,
                    list(row.types),
                    row.coarse_type,
                    row.fine_type,
                    row.item_category,
                    float(row.popularity),
                    float(search_cols["prior"]),
                    str(search_cols["wikipedia_url"]),
                    str(search_cols["dbpedia_url"]),
                )
            )
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.executemany(sql, payload)
            conn.commit()
        return len(payload)

    def upsert_entity_triples(
        self,
        rows: Sequence[EntityTripleRecord],
    ) -> int:
        if not rows:
            return 0
        sql = """
        INSERT INTO entity_triples (subject_qid, predicate_pid, object_qid, updated_at)
        VALUES (%s, %s, %s, NOW())
        ON CONFLICT (subject_qid, predicate_pid, object_qid) DO UPDATE SET
            updated_at = NOW()
        """
        payload = [
            (row.subject_qid, row.predicate_pid, row.object_qid)
            for row in rows
            if row.subject_qid and row.predicate_pid and row.object_qid
        ]
        if not payload:
            return 0
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.executemany(sql, payload)
            conn.commit()
        return len(payload)

    def replace_entity_triples(
        self,
        *,
        subject_qids: Sequence[str],
        rows: Sequence[EntityTripleRecord],
    ) -> int:
        normalized_subjects = [qid for qid in subject_qids if isinstance(qid, str) and qid]
        if not normalized_subjects:
            return 0
        payload = [
            (row.subject_qid, row.predicate_pid, row.object_qid)
            for row in rows
            if row.subject_qid and row.predicate_pid and row.object_qid
        ]
        insert_sql = """
        INSERT INTO entity_triples (subject_qid, predicate_pid, object_qid, updated_at)
        VALUES (%s, %s, %s, NOW())
        ON CONFLICT (subject_qid, predicate_pid, object_qid) DO UPDATE SET
            updated_at = NOW()
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM entity_triples WHERE subject_qid = ANY(%s)",
                    (list(normalized_subjects),),
                )
                if payload:
                    cur.executemany(insert_sql, payload)
            conn.commit()
        return len(payload)

    def iter_entity_ids(self, *, batch_size: int) -> Iterator[list[str]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        sql = "SELECT qid FROM entities ORDER BY qid"
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                while True:
                    rows = cur.fetchmany(batch_size)
                    if not rows:
                        return
                    batch = [row[0] for row in rows if row and isinstance(row[0], str)]
                    if batch:
                        yield batch

    def load_context_inputs(self, qids: Sequence[str]) -> list[tuple[str, list[str]]]:
        if not qids:
            return []
        sql = """
        SELECT
            subject_qid,
            array_agg(DISTINCT object_qid ORDER BY object_qid) AS object_qids
        FROM entity_triples
        WHERE subject_qid = ANY(%s)
        GROUP BY subject_qid
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (list(qids),))
                rows = cur.fetchall()
        out: list[tuple[str, list[str]]] = []
        for qid, related in rows:
            if not isinstance(qid, str):
                continue
            out.append((qid, _as_str_list(related)))
        out.sort(key=lambda item: item[0])
        return out

    def resolve_labels(self, qids: Sequence[str]) -> dict[str, str]:
        if not qids:
            return {}
        sql = """
        SELECT qid, label
        FROM entities
        WHERE qid = ANY(%s)
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (list(qids),))
                rows = cur.fetchall()
        resolved: dict[str, str] = {}
        for qid, label in rows:
            if not isinstance(qid, str):
                continue
            if isinstance(label, str) and label.strip():
                resolved[qid] = label.strip()
        missing = [qid for qid in qids if isinstance(qid, str) and qid not in resolved]
        if missing:
            for qid, label in self.resolve_sample_cache_labels(missing).items():
                if qid not in resolved and label:
                    resolved[qid] = label
        return resolved

    def build_context_strings(
        self,
        qids: Sequence[str],
        *,
        chunk_size: int = 1000,
        max_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    ) -> dict[str, str]:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        normalized_qids: list[str] = []
        seen: set[str] = set()
        for qid in qids:
            if not isinstance(qid, str):
                continue
            cleaned = qid.strip()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            normalized_qids.append(cleaned)
        if not normalized_qids:
            return {}

        context_map: dict[str, str] = {}
        for start in range(0, len(normalized_qids), chunk_size):
            chunk = normalized_qids[start : start + chunk_size]
            batch = self.load_context_inputs(chunk)

            related_ids: list[str] = []
            related_seen: set[str] = set()
            for _qid, object_qids in batch:
                for object_qid in object_qids:
                    if object_qid in related_seen:
                        continue
                    related_seen.add(object_qid)
                    related_ids.append(object_qid)

            label_map = self.resolve_labels(related_ids)
            for qid, object_qids in batch:
                related_labels = [
                    label_map[object_qid].strip()
                    for object_qid in object_qids
                    if object_qid in label_map and label_map[object_qid].strip()
                ]
                context_map[qid] = build_entity_context_string(
                    related_labels=related_labels,
                    max_chars=max_chars,
                )
        return context_map

    def attach_context_strings(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        chunk_size: int = 1000,
        max_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    ) -> list[dict[str, Any]]:
        if not rows:
            return []
        hydrated = [dict(row) for row in rows]
        context_map = self.build_context_strings(
            [
                row.get("qid")
                for row in hydrated
                if isinstance(row.get("qid"), str)
            ],
            chunk_size=chunk_size,
            max_chars=max_chars,
        )
        for row in hydrated:
            qid = row.get("qid")
            row["context_string"] = context_map.get(qid, "") if isinstance(qid, str) else ""
        return hydrated

    def resolve_sample_cache_labels(self, qids: Sequence[str]) -> dict[str, str]:
        if not qids:
            return {}
        sql = """
        SELECT qid, entity_json
        FROM sample_entity_cache
        WHERE qid = ANY(%s)
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (list(qids),))
                rows = cur.fetchall()
        resolved: dict[str, str] = {}
        for qid, entity_json in rows:
            if not isinstance(qid, str):
                continue
            label = _extract_sample_entity_label(entity_json)
            if label:
                resolved[qid] = label
        return resolved

    def load_entity_name_sets(self, qids: Sequence[str]) -> dict[str, tuple[list[str], list[str]]]:
        if not qids:
            return {}
        sql = """
        SELECT qid, labels, aliases
        FROM entities
        WHERE qid = ANY(%s)
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (list(qids),))
                rows = cur.fetchall()
        return {
            qid: (_as_str_list(labels), _as_str_list(aliases))
            for qid, labels, aliases in rows
            if isinstance(qid, str)
        }

    def count_entities(self) -> int:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM entities")
                row = cur.fetchone()
        if not row:
            return 0
        count = row[0]
        return int(count) if isinstance(count, int) else 0

    def iter_entities_for_indexing(self, *, batch_size: int) -> Iterator[list[dict[str, Any]]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        sql = """
        SELECT
            qid, label, labels, aliases, description, types,
            coarse_type, fine_type, item_category, popularity, prior, wikipedia_url, dbpedia_url
        FROM entities
        ORDER BY qid
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                while True:
                    rows = cur.fetchmany(batch_size)
                    if not rows:
                        return
                    out: list[dict[str, Any]] = []
                    for row in rows:
                        if len(row) < 13:
                            continue
                        qid = row[0]
                        label = row[1]
                        if not isinstance(qid, str) or not isinstance(label, str):
                            continue
                        labels = _as_str_list(row[2])
                        aliases = _as_str_list(row[3])
                        out.append(
                            {
                                "qid": qid,
                                "label": label,
                                "labels": labels,
                                "aliases": aliases,
                                "description": row[4] if isinstance(row[4], str) else None,
                                "types": _as_str_list(row[5]),
                                "coarse_type": row[6] if isinstance(row[6], str) else "",
                                "fine_type": row[7] if isinstance(row[7], str) else "",
                                "item_category": row[8] if isinstance(row[8], str) else "",
                                "popularity": float(row[9]) if isinstance(row[9], (int, float)) else 0.0,
                                "prior": float(row[10]) if isinstance(row[10], (int, float)) else 0.0,
                                "cross_refs": {
                                    key: value
                                    for key, value in (
                                        (
                                            "wikipedia",
                                            _expand_wikipedia_ref(row[11]) if isinstance(row[11], str) else "",
                                        ),
                                        (
                                            "dbpedia",
                                            _expand_dbpedia_ref(row[12]) if isinstance(row[12], str) else "",
                                        ),
                                    )
                                    if value
                                },
                            }
                        )
                    if out:
                        yield self.attach_context_strings(
                            out,
                            chunk_size=min(2000, max(1, batch_size)),
                        )

    def _lookup_rows_to_candidates(self, rows: Sequence[Sequence[Any]]) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        for row in rows:
            if len(row) < 14:
                continue
            qid = row[0]
            label = row[1]
            if not isinstance(qid, str) or not isinstance(label, str):
                continue
            labels = _as_str_list(row[2])
            aliases = _as_str_list(row[3])
            popularity = row[9] if isinstance(row[9], (int, float)) else 0.0
            prior = row[10] if isinstance(row[10], (int, float)) else _popularity_to_prior_local(float(popularity))
            score = row[13] if isinstance(row[13], (int, float)) else 0.0
            candidates.append(
                {
                    "qid": qid,
                    "label": label,
                    "labels": labels,
                    "aliases": aliases,
                    "description": row[4] if isinstance(row[4], str) else None,
                    "types": _as_str_list(row[5]),
                    "context_string": "",
                    "coarse_type": row[6] if isinstance(row[6], str) else "",
                    "fine_type": row[7] if isinstance(row[7], str) else "",
                    "item_category": row[8] if isinstance(row[8], str) else "",
                    "popularity": float(popularity),
                    "prior": float(prior),
                    "cross_refs": {
                        key: value
                        for key, value in (
                            (
                                "wikipedia",
                                _expand_wikipedia_ref(row[11]) if isinstance(row[11], str) else "",
                            ),
                            (
                                "dbpedia",
                                _expand_dbpedia_ref(row[12]) if isinstance(row[12], str) else "",
                            ),
                        )
                        if value
                    },
                    "score": float(score),
                }
            )
        return candidates

    def search_candidates_exact(
        self,
        *,
        mention_exact: str,
        coarse_hints: Sequence[str],
        fine_hints: Sequence[str],
        crosslink_exact: Sequence[str],
        size: int,
    ) -> list[dict[str, Any]]:
        # Exact helper columns were removed from the default lean schema. Exactness is now
        # handled in reranking over fuzzy candidates.
        return []

    def search_candidates_fuzzy(
        self,
        *,
        mention_query: str,
        crosslink_exact: Sequence[str],
        coarse_hints: Sequence[str],
        fine_hints: Sequence[str],
        size: int,
    ) -> list[dict[str, Any]]:
        if not mention_query or size <= 0:
            return []
        exact_crosslinks = [value for value in crosslink_exact if isinstance(value, str) and value]
        mention_like = f"%{mention_query}%"
        base_match_sql = (
            "("
            "LOWER(label) = LOWER(%s) OR "
            "label ILIKE %s OR "
            "EXISTS (SELECT 1 FROM unnest(labels) AS v WHERE LOWER(v) = LOWER(%s)) OR "
            "EXISTS (SELECT 1 FROM unnest(labels) AS v WHERE v ILIKE %s) OR "
            "EXISTS (SELECT 1 FROM unnest(aliases) AS v WHERE LOWER(v) = LOWER(%s)) OR "
            "EXISTS (SELECT 1 FROM unnest(aliases) AS v WHERE v ILIKE %s)"
        )
        params: list[Any] = [
            mention_query,
            mention_like,
            mention_query,
            mention_like,
            mention_query,
            mention_like,
        ]
        if exact_crosslinks:
            base_match_sql += " OR wikipedia_url = ANY(%s) OR dbpedia_url = ANY(%s)"
            params.extend([list(exact_crosslinks), list(exact_crosslinks)])
        base_match_sql += ")"
        where_parts = [base_match_sql]
        if coarse_hints:
            where_parts.append("coarse_type = ANY(%s)")
            params.append(list(coarse_hints))
        if fine_hints:
            where_parts.append("fine_type = ANY(%s)")
            params.append(list(fine_hints))

        sql = f"""
        SELECT
            qid, label, labels, aliases, description, types,
            coarse_type, fine_type, item_category, popularity, prior, wikipedia_url, dbpedia_url,
            (
                CASE WHEN LOWER(label) = LOWER(%s) THEN 4.0 ELSE 0.0 END +
                CASE WHEN EXISTS (SELECT 1 FROM unnest(labels) AS v WHERE LOWER(v) = LOWER(%s)) THEN 3.0 ELSE 0.0 END +
                CASE WHEN EXISTS (SELECT 1 FROM unnest(aliases) AS v WHERE LOWER(v) = LOWER(%s)) THEN 2.5 ELSE 0.0 END +
                CASE WHEN label ILIKE %s THEN 1.5 ELSE 0.0 END +
                CASE WHEN EXISTS (SELECT 1 FROM unnest(labels) AS v WHERE v ILIKE %s) THEN 1.25 ELSE 0.0 END +
                CASE WHEN EXISTS (SELECT 1 FROM unnest(aliases) AS v WHERE v ILIKE %s) THEN 1.0 ELSE 0.0 END +
                CASE
                    WHEN %s THEN
                        CASE
                            WHEN wikipedia_url = ANY(%s) OR dbpedia_url = ANY(%s) THEN 1.5
                            ELSE 0.0
                        END
                    ELSE 0.0
                END
            ) AS score
        FROM entities
        WHERE {' AND '.join(where_parts)}
        ORDER BY score DESC, prior DESC, qid ASC
        LIMIT %s
        """
        score_params = [
            mention_query,
            mention_query,
            mention_query,
            mention_like,
            mention_like,
            mention_like,
            bool(exact_crosslinks),
            list(exact_crosslinks),
            list(exact_crosslinks),
        ]
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (*score_params, *params, int(size)))
                rows = cur.fetchall()
        return self.attach_context_strings(self._lookup_rows_to_candidates(rows))

    def get_query_cache(self, cache_key: str) -> dict[str, Any] | None:
        sql = "SELECT result FROM query_cache WHERE cache_key = %s"
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (cache_key,))
                row = cur.fetchone()
        if not row:
            return None
        raw = row[0]
        if isinstance(raw, Mapping):
            return {str(k): v for k, v in raw.items() if isinstance(k, str)}
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                return None
            if isinstance(parsed, Mapping):
                return {str(k): v for k, v in parsed.items() if isinstance(k, str)}
        return None

    def put_query_cache(self, cache_key: str, result: Mapping[str, Any]) -> None:
        sql = """
        INSERT INTO query_cache (cache_key, result, created_at)
        VALUES (%s, %s::jsonb, NOW())
        ON CONFLICT (cache_key) DO UPDATE SET
            result = EXCLUDED.result,
            created_at = NOW()
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (cache_key, _json_compact(dict(result))))
            conn.commit()

    def get_sample_entities(self, qids: Sequence[str]) -> dict[str, dict[str, Any]]:
        if not qids:
            return {}
        sql = """
        SELECT qid, entity_json
        FROM sample_entity_cache
        WHERE qid = ANY(%s)
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (list(qids),))
                rows = cur.fetchall()
        out: dict[str, dict[str, Any]] = {}
        for qid, payload in rows:
            if not isinstance(qid, str):
                continue
            if isinstance(payload, Mapping):
                out[qid] = {str(k): v for k, v in payload.items() if isinstance(k, str)}
                continue
            if isinstance(payload, str):
                try:
                    parsed = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, Mapping):
                    out[qid] = {str(k): v for k, v in parsed.items() if isinstance(k, str)}
        return out

    def count_sample_entities(self) -> int:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM sample_entity_cache")
                row = cur.fetchone()
        if not row:
            return 0
        value = row[0]
        return int(value) if isinstance(value, int) else 0

    def list_sample_entity_ids(self, *, limit: int) -> list[str]:
        if limit <= 0:
            return []
        sql = """
        SELECT qid
        FROM sample_entity_cache
        WHERE qid ~ '^Q[0-9]+$'
        ORDER BY CAST(SUBSTRING(qid FROM 2) AS BIGINT), qid
        LIMIT %s
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (limit,))
                rows = cur.fetchall()
        return [row[0] for row in rows if row and isinstance(row[0], str)]

    def iter_sample_entities(self, qids: Sequence[str], *, batch_size: int) -> Iterator[dict[str, Any]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if not qids:
            return

        normalized_qids: list[str] = []
        seen: set[str] = set()
        for qid in qids:
            if not isinstance(qid, str):
                continue
            cleaned = qid.strip()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            normalized_qids.append(cleaned)
        if not normalized_qids:
            return

        for start in range(0, len(normalized_qids), batch_size):
            chunk = normalized_qids[start : start + batch_size]
            cached = self.get_sample_entities(chunk)
            missing = [qid for qid in chunk if qid not in cached]
            if missing:
                sample = ", ".join(missing[:5])
                raise ValueError(
                    f"Missing {len(missing)} requested QIDs in sample_entity_cache"
                    f"{': ' + sample if sample else ''}."
                )
            for qid in chunk:
                yield cached[qid]

    def recreate_entities_like_table(
        self,
        table_name: str,
        *,
        drop_existing: bool = True,
        unlogged: bool = False,
    ) -> None:
        self._recreate_table_like(
            source_table="entities",
            dest_table=table_name,
            drop_existing=drop_existing,
            unlogged=unlogged,
        )

    def _recreate_table_like(
        self,
        *,
        source_table: str,
        dest_table: str,
        drop_existing: bool = True,
        unlogged: bool = False,
    ) -> None:
        source_ident = _quote_identifier(source_table)
        dest_ident = _quote_identifier(dest_table)
        ddl_parts = []
        if drop_existing:
            ddl_parts.append(f"DROP TABLE IF EXISTS {dest_ident}")
        table_kind = "CREATE UNLOGGED TABLE" if unlogged else "CREATE TABLE"
        ddl_parts.append(
            f"{table_kind} "
            f"{dest_ident} "
            f"(LIKE {source_ident} INCLUDING DEFAULTS INCLUDING CONSTRAINTS INCLUDING STORAGE INCLUDING COMPRESSION)"
        )
        ddl = ";\n".join(ddl_parts) + ";"
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(ddl)
            conn.commit()

    def recreate_entity_storage_like_tables(
        self,
        base_table_name: str,
        *,
        drop_existing: bool = True,
        unlogged: bool = False,
    ) -> dict[str, str]:
        self._recreate_table_like(
            source_table="entities",
            dest_table=base_table_name,
            drop_existing=drop_existing,
            unlogged=unlogged,
        )
        return {
            "entities_table": base_table_name,
            "payload_table": base_table_name,
            "context_table": base_table_name,
        }

    def _qid_replication_sql(
        self,
        *,
        seed_alias: str = "seed",
        sample_no_expr: str = "gs.sample_no",
    ) -> str:
        return f"""
        CASE
            WHEN {seed_alias}.qid ~ '^[QP][0-9]+$' THEN
                SUBSTRING({seed_alias}.qid FROM 1 FOR 1) ||
                (
                    SUBSTRING({seed_alias}.qid FROM 2)::bigint +
                    (({sample_no_expr} + 1) * %s::bigint)
                )::text
            ELSE {seed_alias}.qid || '__r' || ({sample_no_expr} + 1)::text
        END
        """

    def replicate_entity_storage_for_size_estimation(
        self,
        *,
        dest_table: str,
        dest_payload_table: str | None = None,
        dest_context_table: str | None = None,
        target_rows: int,
        seed_rows: int = 0,
        batch_rows: int = 100_000,
        random_seed: int = 0,
        on_chunk: Callable[[dict[str, int]], None] | None = None,
        disable_synchronous_commit: bool = False,
    ) -> dict[str, int]:
        if target_rows <= 0:
            raise ValueError("target_rows must be > 0")
        if seed_rows < 0:
            raise ValueError("seed_rows must be >= 0")
        if batch_rows <= 0:
            raise ValueError("batch_rows must be > 0")
        if random_seed < 0:
            raise ValueError("random_seed must be >= 0")

        dest_ident = _quote_identifier(dest_table)
        temp_seed = "_alpaca_entities_sim_seed"
        temp_seed_ident = _quote_identifier(temp_seed)

        seed_limit_sql = "" if seed_rows == 0 else " LIMIT %s"
        seed_limit_params: tuple[Any, ...] = () if seed_rows == 0 else (int(seed_rows),)
        create_seed_sql = (
            f"CREATE TEMP TABLE {temp_seed_ident} AS "
            "SELECT "
            "    ROW_NUMBER() OVER (ORDER BY seed.qid) AS seed_row_no, "
            "    seed.* "
            f"FROM (SELECT * FROM entities ORDER BY qid{seed_limit_sql}) AS seed"
        )
        qid_expr = self._qid_replication_sql(seed_alias="seed", sample_no_expr="gs.sample_no")
        insert_sql = f"""
        INSERT INTO {dest_ident} (
            qid, label, labels, aliases, description, types, coarse_type, fine_type,
            item_category, popularity, prior, wikipedia_url, dbpedia_url, updated_at
        )
        SELECT
            {qid_expr} AS qid,
            seed.label,
            seed.labels,
            seed.aliases,
            seed.description,
            seed.types,
            seed.coarse_type,
            seed.fine_type,
            seed.item_category,
            seed.popularity,
            seed.prior,
            seed.wikipedia_url,
            seed.dbpedia_url,
            seed.updated_at
        FROM generate_series(%s::bigint, %s::bigint) AS gs(sample_no)
        JOIN {temp_seed_ident} AS seed
          ON seed.seed_row_no = (
              (
                  (
                      (((gs.sample_no + 1) * {_SIM_SAMPLE_MULTIPLIER}::bigint) +
                      ((%s::bigint + 1) * {_SIM_SAMPLE_SEED_MULTIPLIER}::bigint) +
                      {_SIM_SAMPLE_INCREMENT}::bigint
                  ) %% {_SIM_SAMPLE_MODULUS}::bigint
              ) %% %s::bigint
              ) + 1
          )
        """

        with self._connect() as conn:
            with conn.cursor() as cur:
                if disable_synchronous_commit:
                    cur.execute("SET SESSION synchronous_commit = OFF")
                cur.execute(f"DROP TABLE IF EXISTS {temp_seed_ident}")
                cur.execute(create_seed_sql, seed_limit_params)
                cur.execute(
                    f"""
                    SELECT
                        COUNT(*)::bigint AS seed_count,
                        COALESCE(
                            MAX(
                                CASE
                                    WHEN qid ~ '^[QP][0-9]+$' THEN SUBSTRING(qid FROM 2)::bigint
                                    ELSE NULL
                                END
                            ),
                            0
                        )::bigint AS max_numeric_qid
                    FROM {temp_seed_ident}
                    """
                )
                row = cur.fetchone()
                if not row:
                    raise ValueError("Could not read seed rows for simulation.")
                seed_count = int(row[0]) if isinstance(row[0], int) else 0
                max_numeric_qid = int(row[1]) if isinstance(row[1], int) else 0
                if seed_count <= 0:
                    raise ValueError(
                        "Source entities table is empty (or selected seed rows = 0). "
                        "Run the pipeline first."
                    )

                stride = max(1, max_numeric_qid + 1)
                remaining = int(target_rows)
                sample_cursor = 0
                inserted_total = 0
                chunk_count = 0

                while remaining > 0:
                    rows_this_chunk = min(remaining, int(batch_rows))
                    sample_end = sample_cursor + rows_this_chunk - 1
                    cur.execute(
                        insert_sql,
                        (
                            int(stride),
                            int(sample_cursor),
                            int(sample_end),
                            int(random_seed),
                            int(seed_count),
                        ),
                    )
                    inserted = cur.rowcount if isinstance(cur.rowcount, int) else 0
                    if inserted <= 0:
                        raise PostgresStoreError(
                            "Simulation insert wrote 0 rows unexpectedly. "
                            "Check destination table constraints."
                        )
                    inserted_total += inserted
                    remaining -= inserted
                    sample_cursor = sample_end + 1
                    chunk_count += 1
                    if on_chunk is not None:
                        try:
                            on_chunk(
                                {
                                    "chunk_index": int(chunk_count),
                                    "chunk_rows": int(inserted),
                                    "rows_inserted_total": int(inserted_total),
                                    "rows_remaining": int(remaining),
                                    "samples_emitted": int(sample_cursor),
                                }
                            )
                        except Exception:
                            pass

                conn.commit()

                return {
                    "seed_rows_used": seed_count,
                    "target_rows": int(target_rows),
                    "inserted_rows": inserted_total,
                    "qid_stride": int(stride),
                    "samples_emitted": int(sample_cursor),
                    "random_seed": int(random_seed),
                    "chunks": int(chunk_count),
                }

    def truncate_table(self, table_name: str) -> None:
        table_ident = _quote_identifier(table_name)
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(f"TRUNCATE TABLE {table_ident}")
            conn.commit()

    def replicate_entities_for_size_estimation(
        self,
        *,
        dest_table: str,
        target_rows: int,
        seed_rows: int = 0,
        batch_rows: int = 100_000,
        random_seed: int = 0,
        on_chunk: Callable[[dict[str, int]], None] | None = None,
        disable_synchronous_commit: bool = False,
    ) -> dict[str, int]:
        return self.replicate_entity_storage_for_size_estimation(
            dest_table=dest_table,
            target_rows=target_rows,
            seed_rows=seed_rows,
            batch_rows=batch_rows,
            random_seed=random_seed,
            on_chunk=on_chunk,
            disable_synchronous_commit=disable_synchronous_commit,
        )

    def analyze_table(self, table_name: str) -> None:
        table_ident = _quote_identifier(table_name)
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(f"ANALYZE {table_ident}")
            conn.commit()

    def table_storage_stats(self, table_name: str) -> dict[str, int]:
        _quote_identifier(table_name)  # validate
        sql = """
        SELECT
            (SELECT COUNT(*) FROM %s_placeholder) AS rows,
            pg_relation_size(%s::regclass) AS table_bytes,
            pg_indexes_size(%s::regclass) AS index_bytes,
            pg_total_relation_size(%s::regclass) AS total_bytes,
            GREATEST(
                pg_total_relation_size(%s::regclass) -
                pg_relation_size(%s::regclass) -
                pg_indexes_size(%s::regclass),
                0
            ) AS toast_bytes
        """
        sql = sql.replace("%s_placeholder", _quote_identifier(table_name))
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql,
                    (
                        table_name,
                        table_name,
                        table_name,
                        table_name,
                        table_name,
                        table_name,
                    ),
                )
                row = cur.fetchone()
        if not row or len(row) < 5:
            return {
                "rows": 0,
                "table_bytes": 0,
                "index_bytes": 0,
                "toast_bytes": 0,
                "total_bytes": 0,
            }
        return {
            "rows": int(row[0]) if isinstance(row[0], int) else 0,
            "table_bytes": int(row[1]) if isinstance(row[1], int) else 0,
            "index_bytes": int(row[2]) if isinstance(row[2], int) else 0,
            "total_bytes": int(row[3]) if isinstance(row[3], int) else 0,
            "toast_bytes": int(row[4]) if isinstance(row[4], int) else 0,
        }

    def compact_table_for_lookup(
        self,
        table_name: str = "entities",
        *,
        drop_cross_refs_trgm_index: bool = True,
        drop_context_inputs_table: bool = False,
        vacuum_full: bool = False,
        analyze: bool = True,
    ) -> dict[str, Any]:
        table_ident = _quote_identifier(table_name)
        dropped_columns = [
            "labels_flat",         # legacy secondary-label array column
            "aliases_flat",        # legacy aliases array column
            "label_exact",         # legacy exact helper
            "aliases_exact",       # legacy exact helper
            "cross_refs_exact",    # legacy exact helper
            "name_variants",       # temporary normalized alias column (renamed back to aliases)
            "labels_text",         # legacy text helper
            "aliases_text",        # legacy text helper
            "search_document",     # legacy duplicated search text
            "context_string",      # replaced by lazy triples-backed synthesis
            "cross_refs_text",     # legacy text helper replaced by expression indexes
            "cross_refs",          # raw JSONB; lookup uses explicit URL columns
        ]

        index_drops: list[str] = []
        if drop_cross_refs_trgm_index:
            index_drops.append(f"idx_{table_name}_cross_refs_text_trgm")
            index_drops.append(f"idx_{table_name}_cross_refs_url_trgm")
        index_drops.append(f"idx_{table_name}_aliases_trgm")
        index_drops.append(f"idx_{table_name}_aliases_text_trgm")
        index_drops.append(f"idx_{table_name}_label_trgm")
        index_drops.append(f"idx_{table_name}_name_variants_trgm")
        index_drops.append(f"idx_{table_name}_label_exact")
        index_drops.append(f"idx_{table_name}_aliases_exact")
        index_drops.append(f"idx_{table_name}_cross_refs_exact")
        index_drops.append(f"idx_{table_name}_item_category")
        if table_name == "entities":
            index_drops.append("idx_entity_triples_subject_qid")
            index_drops.append("idx_entity_triples_object_qid")
            index_drops.append("idx_entity_triples_predicate_pid")
        dropped_tables: list[str] = []

        with self._connect() as conn:
            with conn.cursor() as cur:
                for column_name in dropped_columns:
                    cur.execute(f"ALTER TABLE {table_ident} DROP COLUMN IF EXISTS {_quote_identifier(column_name)}")
                for index_name in index_drops:
                    cur.execute(f"DROP INDEX IF EXISTS {_quote_identifier(index_name)}")
                if drop_context_inputs_table and table_name == "entities":
                    cur.execute("DROP TABLE IF EXISTS entity_context_inputs")
                    cur.execute("DROP TABLE IF EXISTS entity_name_payloads")
                    cur.execute("DROP TABLE IF EXISTS entity_triples")
                    dropped_tables.append("entity_context_inputs")
                    dropped_tables.append("entity_name_payloads")
                    dropped_tables.append("entity_triples")
                if analyze and not vacuum_full:
                    cur.execute(f"ANALYZE {table_ident}")
            conn.commit()

        if vacuum_full:
            # VACUUM FULL must run outside a transaction; use a fresh autocommit connection.
            conn = self._connect()
            try:
                conn.autocommit = True
                with conn.cursor() as cur:
                    cur.execute(f"VACUUM FULL ANALYZE {table_ident}")
            finally:
                conn.close()

        return {
            "table": table_name,
            "dropped_columns": dropped_columns,
            "dropped_indexes": index_drops,
            "dropped_tables": dropped_tables,
            "drop_context_inputs_table": bool(drop_context_inputs_table),
            "vacuum_full": bool(vacuum_full),
            "analyze": bool(analyze),
        }

    def upsert_sample_entities(
        self,
        rows: Sequence[tuple[str, Mapping[str, Any], str]],
    ) -> int:
        if not rows:
            return 0
        sql = """
        INSERT INTO sample_entity_cache (qid, entity_json, source_url, updated_at)
        VALUES (%s, %s::jsonb, %s, NOW())
        ON CONFLICT (qid) DO UPDATE SET
            entity_json = EXCLUDED.entity_json,
            source_url = EXCLUDED.source_url,
            updated_at = NOW()
        """
        payload = [
            (qid, _json_compact(dict(entity_json)), source_url)
            for qid, entity_json, source_url in rows
        ]
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.executemany(sql, payload)
            conn.commit()
        return len(payload)

    def prune_query_cache_older_than_seconds(self, ttl_seconds: int) -> int:
        if ttl_seconds <= 0:
            return 0
        sql = """
        DELETE FROM query_cache
        WHERE created_at < (NOW() - (%s * INTERVAL '1 second'))
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (ttl_seconds,))
                deleted = cur.rowcount
            conn.commit()
        return int(deleted) if isinstance(deleted, int) else 0

    def clear_entities(self) -> None:
        with self._connect() as conn:
            truncate_name_payloads = self._table_exists(conn, "entity_name_payloads")
            truncate_context_inputs = self._table_exists(conn, "entity_context_inputs")
            with conn.cursor() as cur:
                cur.execute("TRUNCATE TABLE entities")
                if self._table_exists(conn, "entity_triples"):
                    cur.execute("TRUNCATE TABLE entity_triples")
                if truncate_name_payloads:
                    cur.execute("TRUNCATE TABLE entity_name_payloads")
                if truncate_context_inputs:
                    cur.execute("TRUNCATE TABLE entity_context_inputs")
            conn.commit()

    def replace_entities(self, rows: Iterable[EntityRecord], *, batch_size: int) -> int:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        total = 0
        buffer: list[EntityRecord] = []
        for row in rows:
            buffer.append(row)
            if len(buffer) >= batch_size:
                total += self.upsert_entities(buffer)
                buffer.clear()
        if buffer:
            total += self.upsert_entities(buffer)
        return total
