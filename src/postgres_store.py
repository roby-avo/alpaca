from __future__ import annotations

import json
import math
import re
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
    relation_object_qids: Sequence[str]
    item_category: str
    coarse_type: str
    fine_type: str
    popularity: float
    cross_refs: Mapping[str, Any]
    context_string: str = ""


_SQL_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_WIKIPEDIA_PREFIX = "https://en.wikipedia.org/wiki/"
_DBPEDIA_PREFIX = "https://dbpedia.org/resource/"


def _quote_identifier(name: str) -> str:
    if not isinstance(name, str):
        raise ValueError("SQL identifier must be a string.")
    stripped = name.strip()
    if not _SQL_IDENTIFIER_RE.match(stripped):
        raise ValueError(
            f"Invalid SQL identifier '{name}'. Use letters, digits, and underscores only."
        )
    return f'"{stripped}"'


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
        if parsed.path and marker in parsed.path:
            return parsed.path.split(marker, 1)[1]
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
        if parsed.path and marker in parsed.path:
            return parsed.path.split(marker, 1)[1]
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
    if raw.startswith("/wiki/"):
        return f"https://en.wikipedia.org{raw}"
    return f"{_WIKIPEDIA_PREFIX}{raw}"


def _expand_dbpedia_ref(value: str) -> str:
    raw = value.strip()
    if not raw:
        return ""
    if raw.startswith("http://") or raw.startswith("https://"):
        return raw
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

    preferred: list[tuple[str, str]] = []
    for language, payload in raw_labels.items():
        if not isinstance(language, str) or not isinstance(payload, Mapping):
            continue
        raw_value = payload.get("value")
        if not isinstance(raw_value, str):
            continue
        normalized = normalize_text(raw_value)
        if not normalized:
            continue
        if language == "en":
            return normalized
        preferred.append((language, normalized))

    if not preferred:
        return None
    preferred.sort(key=lambda item: item[0])
    return preferred[0][1]


def _popularity_to_prior_local(popularity: float) -> float:
    value = max(0.0, float(popularity))
    return 1.0 - math.exp(-math.log1p(value) / 6.0)


def _flatten_labels_map(labels: Mapping[str, str]) -> list[str]:
    flattened: list[str] = []
    seen: set[str] = set()
    if "en" in labels:
        ordered_langs = ["en", *sorted(lang for lang in labels if lang != "en")]
    else:
        ordered_langs = sorted(labels)
    for language in ordered_langs:
        raw_value = labels.get(language)
        if not isinstance(raw_value, str):
            continue
        value = normalize_text(raw_value)
        if not value or value in seen:
            continue
        seen.add(value)
        flattened.append(value)
    return flattened


def _flatten_aliases_map(aliases: Mapping[str, Sequence[str]]) -> list[str]:
    flattened: list[str] = []
    seen: set[str] = set()
    if "en" in aliases:
        ordered_langs = ["en", *sorted(lang for lang in aliases if lang != "en")]
    else:
        ordered_langs = sorted(aliases)
    for language in ordered_langs:
        values = aliases.get(language, [])
        for raw_alias in values:
            if not isinstance(raw_alias, str):
                continue
            alias = normalize_text(raw_alias)
            if not alias or alias in seen:
                continue
            seen.add(alias)
            flattened.append(alias)
    return flattened


def _join_terms(values: Sequence[str]) -> str:
    return " ".join(value for value in values if isinstance(value, str) and value)


def _entity_search_columns(
    *,
    label: str,
    labels: Mapping[str, str],
    aliases: Mapping[str, Sequence[str]],
    context_string: str,
    cross_refs: Mapping[str, Any],
    popularity: float,
) -> dict[str, Any]:
    labels_flat_all = _flatten_labels_map(labels)
    primary_label_normalized = normalize_text(label) if isinstance(label, str) else ""
    labels_flat = [
        value
        for value in labels_flat_all
        if value and (not primary_label_normalized or value != primary_label_normalized)
    ]
    aliases_flat = _flatten_aliases_map(aliases)
    # Keep secondary labels + aliases together for lookup matching/reranking without
    # duplicating the primary label. This is the normalized lookup-facing name payload.
    lookup_aliases = [*labels_flat, *aliases_flat]
    aliases_text = _join_terms(lookup_aliases)
    clean_context = normalize_text(context_string) if context_string else ""
    # Keep the indexed context compact to reduce tsvector noise/size.
    context_search_text = clean_context[:256]
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
        "aliases": lookup_aliases,
        "aliases_text": aliases_text,
        "context_search_text": context_search_text,
        "wikipedia_url": wikipedia_url,
        "dbpedia_url": dbpedia_url,
    }


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

    def ensure_schema(self) -> None:
        ddl = """
        CREATE OR REPLACE FUNCTION alpaca_join_text_array(arr TEXT[])
        RETURNS TEXT
        LANGUAGE SQL
        IMMUTABLE
        PARALLEL SAFE
        RETURNS NULL ON NULL INPUT
        AS $$
            SELECT array_to_string(arr, ' ');
        $$;

        CREATE OR REPLACE FUNCTION alpaca_filter_fts_text(txt TEXT)
        RETURNS TEXT
        LANGUAGE SQL
        IMMUTABLE
        PARALLEL SAFE
        RETURNS NULL ON NULL INPUT
        AS $$
            SELECT COALESCE(string_agg(tok, ' ' ORDER BY ord), '')
            FROM (
                SELECT m.match_arr[1] AS tok, m.ord
                FROM regexp_matches(lower(txt), '[[:alnum:]]+', 'g')
                     WITH ORDINALITY AS m(match_arr, ord)
            ) AS parts
            WHERE char_length(tok) >= 2
              AND tok <> ALL(ARRAY[
                    'a','an','the','and','or','but',
                    'of','in','on','at','to','for','from','by','with','without',
                    'into','onto','over','under','after','before','during',
                    'between','among','via','per',
                    'is','are','was','were','be','been','being',
                    'this','that','these','those'
                ]::text[]);
        $$;

        CREATE OR REPLACE FUNCTION alpaca_filter_fts_context(txt TEXT)
        RETURNS TEXT
        LANGUAGE SQL
        IMMUTABLE
        PARALLEL SAFE
        RETURNS NULL ON NULL INPUT
        AS $$
            SELECT COALESCE(string_agg(tok, ' ' ORDER BY first_ord), '')
            FROM (
                SELECT tok, MIN(ord) AS first_ord
                FROM (
                    SELECT m.match_arr[1] AS tok, m.ord
                    FROM regexp_matches(lower(txt), '[[:alnum:]]+', 'g')
                         WITH ORDINALITY AS m(match_arr, ord)
                ) AS parts
                WHERE char_length(tok) >= 3
                  AND tok !~ '^[0-9]+$'
                  AND tok <> ALL(ARRAY[
                        'a','an','the','and','or','but',
                        'of','in','on','at','to','for','from','by','with','without',
                        'into','onto','over','under','after','before','during',
                        'between','among','via','per',
                        'is','are','was','were','be','been','being',
                        'this','that','these','those',
                        'monday','tuesday','wednesday','thursday','friday','saturday','sunday',
                        'january','february','march','april','may','june',
                        'july','august','september','october','november','december'
                    ]::text[])
                GROUP BY tok
                ORDER BY MIN(ord)
                LIMIT 64
            ) AS filtered;
        $$;

        CREATE TABLE IF NOT EXISTS entities (
            qid TEXT PRIMARY KEY,
            label TEXT NOT NULL,
            context_string TEXT NOT NULL DEFAULT '',
            aliases TEXT[] NOT NULL DEFAULT ARRAY[]::text[],
            search_vector TSVECTOR NOT NULL DEFAULT ''::tsvector,
            coarse_type TEXT NOT NULL DEFAULT '',
            fine_type TEXT NOT NULL DEFAULT '',
            item_category TEXT NOT NULL DEFAULT '',
            popularity DOUBLE PRECISION NOT NULL DEFAULT 0,
            prior DOUBLE PRECISION NOT NULL DEFAULT 0,
            wikipedia_url TEXT NOT NULL DEFAULT '',
            dbpedia_url TEXT NOT NULL DEFAULT '',
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS idx_entities_coarse_type ON entities (coarse_type);
        CREATE INDEX IF NOT EXISTS idx_entities_fine_type ON entities (fine_type);

        CREATE TABLE IF NOT EXISTS entity_context_inputs (
            qid TEXT PRIMARY KEY,
            relation_object_qids JSONB NOT NULL DEFAULT '[]'::jsonb,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
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

        ALTER TABLE entities ADD COLUMN IF NOT EXISTS aliases TEXT[] NOT NULL DEFAULT ARRAY[]::text[];
        ALTER TABLE entities ADD COLUMN IF NOT EXISTS search_vector TSVECTOR NOT NULL DEFAULT ''::tsvector;
        ALTER TABLE entities ADD COLUMN IF NOT EXISTS item_category TEXT NOT NULL DEFAULT '';
        ALTER TABLE entities ADD COLUMN IF NOT EXISTS prior DOUBLE PRECISION NOT NULL DEFAULT 0;
        ALTER TABLE entities ADD COLUMN IF NOT EXISTS wikipedia_url TEXT NOT NULL DEFAULT '';
        ALTER TABLE entities ADD COLUMN IF NOT EXISTS dbpedia_url TEXT NOT NULL DEFAULT '';
        ALTER TABLE entity_context_inputs ADD COLUMN IF NOT EXISTS relation_object_qids JSONB NOT NULL DEFAULT '[]'::jsonb;
        CREATE INDEX IF NOT EXISTS idx_entities_item_category ON entities (item_category);
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
                # Normalize into a single lookup-facing aliases array.
                if "name_variants" in entity_columns:
                    cur.execute(
                        """
                        UPDATE entities
                        SET aliases = COALESCE(name_variants, ARRAY[]::text[])
                        WHERE COALESCE(array_length(name_variants, 1), 0) > 0
                          AND COALESCE(array_length(aliases, 1), 0) = 0
                        """
                    )
                elif "labels" in entity_columns:
                    cur.execute(
                        """
                        UPDATE entities
                        SET aliases = COALESCE(labels, ARRAY[]::text[]) || COALESCE(aliases, ARRAY[]::text[])
                        WHERE COALESCE(array_length(labels, 1), 0) > 0
                        """
                    )
                if "labels" in entity_columns:
                    cur.execute("ALTER TABLE entities DROP COLUMN IF EXISTS labels")
                if "name_variants" in entity_columns:
                    cur.execute("ALTER TABLE entities DROP COLUMN IF EXISTS name_variants")
            conn.commit()

    def ensure_search_indexes(self, table_name: str = "entities") -> None:
        table_ident = _quote_identifier(table_name)
        index_prefix = f"idx_{table_name}"
        aliases_expr = "COALESCE(alpaca_join_text_array(aliases), '')"
        cross_ref_url_expr = "(COALESCE(wikipedia_url, '') || ' ' || COALESCE(dbpedia_url, ''))"
        aliases_nonempty_pred = "COALESCE(array_length(aliases, 1), 0) > 0"
        cross_refs_nonempty_pred = (
            "(COALESCE(wikipedia_url, '') <> '' OR COALESCE(dbpedia_url, '') <> '')"
        )
        ddl = f"""
        CREATE EXTENSION IF NOT EXISTS pg_trgm;

        CREATE INDEX IF NOT EXISTS {index_prefix}_coarse_type ON {table_ident} (coarse_type);
        CREATE INDEX IF NOT EXISTS {index_prefix}_fine_type ON {table_ident} (fine_type);
        CREATE INDEX IF NOT EXISTS {index_prefix}_item_category ON {table_ident} (item_category);
        CREATE INDEX IF NOT EXISTS {index_prefix}_search_vector ON {table_ident} USING GIN (search_vector);
        CREATE INDEX IF NOT EXISTS {index_prefix}_label_trgm ON {table_ident} USING GIN (label gin_trgm_ops);
        CREATE INDEX IF NOT EXISTS {index_prefix}_aliases_trgm
        ON {table_ident} USING GIN ({aliases_expr} gin_trgm_ops)
        WHERE {aliases_nonempty_pred};
        CREATE INDEX IF NOT EXISTS {index_prefix}_cross_refs_url_trgm
        ON {table_ident} USING GIN ({cross_ref_url_expr} gin_trgm_ops)
        WHERE {cross_refs_nonempty_pred};
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(ddl)
            conn.commit()

    def upsert_entities(
        self,
        rows: Sequence[EntityRecord],
        *,
        build_search_vector: bool = True,
    ) -> int:
        if not rows:
            return 0
        sql_with_search_vector = """
        INSERT INTO entities (
            qid, label, context_string,
            aliases, search_vector,
            coarse_type, fine_type, item_category, popularity, prior,
            wikipedia_url, dbpedia_url
        ) VALUES (
            %s, %s, %s,
            %s::text[],
            (
                setweight(to_tsvector('simple', alpaca_filter_fts_text(%s)), 'A') ||
                setweight(to_tsvector('simple', alpaca_filter_fts_text(%s)), 'B') ||
                setweight(to_tsvector('simple', alpaca_filter_fts_context(%s)), 'D')
            ),
            %s, %s, %s, %s, %s,
            %s, %s
        )
        ON CONFLICT (qid) DO UPDATE SET
            label = EXCLUDED.label,
            context_string = EXCLUDED.context_string,
            aliases = EXCLUDED.aliases,
            search_vector = EXCLUDED.search_vector,
            coarse_type = EXCLUDED.coarse_type,
            fine_type = EXCLUDED.fine_type,
            item_category = EXCLUDED.item_category,
            popularity = EXCLUDED.popularity,
            prior = EXCLUDED.prior,
            wikipedia_url = EXCLUDED.wikipedia_url,
            dbpedia_url = EXCLUDED.dbpedia_url,
            updated_at = NOW()
        """
        sql_without_search_vector = """
        INSERT INTO entities (
            qid, label, context_string,
            aliases, search_vector,
            coarse_type, fine_type, item_category, popularity, prior,
            wikipedia_url, dbpedia_url
        ) VALUES (
            %s, %s, %s,
            %s::text[],
            ''::tsvector,
            %s, %s, %s, %s, %s,
            %s, %s
        )
        ON CONFLICT (qid) DO UPDATE SET
            label = EXCLUDED.label,
            context_string = EXCLUDED.context_string,
            aliases = EXCLUDED.aliases,
            search_vector = EXCLUDED.search_vector,
            coarse_type = EXCLUDED.coarse_type,
            fine_type = EXCLUDED.fine_type,
            item_category = EXCLUDED.item_category,
            popularity = EXCLUDED.popularity,
            prior = EXCLUDED.prior,
            wikipedia_url = EXCLUDED.wikipedia_url,
            dbpedia_url = EXCLUDED.dbpedia_url,
            updated_at = NOW()
        """
        context_sql = """
        INSERT INTO entity_context_inputs (qid, relation_object_qids, updated_at)
        VALUES (%s, %s::jsonb, NOW())
        ON CONFLICT (qid) DO UPDATE SET
            relation_object_qids = EXCLUDED.relation_object_qids,
            updated_at = NOW()
        """
        payload: list[tuple[Any, ...]] = []
        context_payload: list[tuple[Any, ...]] = []
        for row in rows:
            search_cols = _entity_search_columns(
                label=row.label,
                labels=row.labels,
                aliases=row.aliases,
                context_string=row.context_string,
                cross_refs=row.cross_refs,
                popularity=float(row.popularity),
            )
            payload.append(
                (
                    row.qid,
                    row.label,
                    row.context_string,
                    list(search_cols["aliases"]),
                    normalize_text(row.label),
                    str(search_cols["aliases_text"]),
                    str(search_cols["context_search_text"]),
                    row.coarse_type,
                    row.fine_type,
                    row.item_category,
                    float(row.popularity),
                    float(search_cols["prior"]),
                    str(search_cols["wikipedia_url"]),
                    str(search_cols["dbpedia_url"]),
                )
                if build_search_vector
                else (
                    row.qid,
                    row.label,
                    row.context_string,
                    list(search_cols["aliases"]),
                    row.coarse_type,
                    row.fine_type,
                    row.item_category,
                    float(row.popularity),
                    float(search_cols["prior"]),
                    str(search_cols["wikipedia_url"]),
                    str(search_cols["dbpedia_url"]),
                )
            )
            context_payload.append((row.qid, _json_compact(list(row.relation_object_qids))))
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    sql_with_search_vector if build_search_vector else sql_without_search_vector,
                    payload,
                )
                cur.executemany(context_sql, context_payload)
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
        SELECT qid, relation_object_qids
        FROM entity_context_inputs
        WHERE qid = ANY(%s)
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

    def update_context_strings(self, rows: Sequence[tuple[str, str]]) -> int:
        if not rows:
            return 0
        sql = """
        UPDATE entities
        SET context_string = %s,
            search_vector = (
                setweight(to_tsvector('simple', alpaca_filter_fts_text(COALESCE(label, ''))), 'A') ||
                setweight(
                    to_tsvector(
                        'simple',
                        alpaca_filter_fts_text(COALESCE(alpaca_join_text_array(aliases), ''))
                    ),
                    'B'
                ) ||
                setweight(
                    to_tsvector(
                        'simple',
                        alpaca_filter_fts_context(LEFT(COALESCE(%s::text, ''), 256))
                    ),
                    'D'
                )
            ),
            updated_at = NOW()
        WHERE qid = %s
        """
        payload = [(context, context, qid) for qid, context in rows]
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.executemany(sql, payload)
            conn.commit()
        return len(payload)

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
            qid, label, aliases, context_string,
            coarse_type, fine_type, item_category, popularity, wikipedia_url, dbpedia_url
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
                        if len(row) < 10:
                            continue
                        qid = row[0]
                        label = row[1]
                        if not isinstance(qid, str) or not isinstance(label, str):
                            continue
                        out.append(
                            {
                                "qid": qid,
                                "label": label,
                                "aliases": _as_str_list(row[2]),
                                "context_string": row[3] if isinstance(row[3], str) else "",
                                "coarse_type": row[4] if isinstance(row[4], str) else "",
                                "fine_type": row[5] if isinstance(row[5], str) else "",
                                "item_category": row[6] if isinstance(row[6], str) else "",
                                "popularity": float(row[7]) if isinstance(row[7], (int, float)) else 0.0,
                                "cross_refs": {
                                    key: value
                                    for key, value in (
                                        (
                                            "wikipedia",
                                            _expand_wikipedia_ref(row[8]) if isinstance(row[8], str) else "",
                                        ),
                                        (
                                            "dbpedia",
                                            _expand_dbpedia_ref(row[9]) if isinstance(row[9], str) else "",
                                        ),
                                    )
                                    if value
                                },
                            }
                        )
                    if out:
                        yield out

    def _lookup_rows_to_candidates(self, rows: Sequence[Sequence[Any]]) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        for row in rows:
            if len(row) < 12:
                continue
            qid = row[0]
            label = row[1]
            if not isinstance(qid, str) or not isinstance(label, str):
                continue
            popularity = row[7] if isinstance(row[7], (int, float)) else 0.0
            prior = row[8] if isinstance(row[8], (int, float)) else _popularity_to_prior_local(float(popularity))
            score = row[11] if isinstance(row[11], (int, float)) else 0.0
            candidates.append(
                {
                    "qid": qid,
                    "label": label,
                    "aliases": _as_str_list(row[2]),
                    "context_string": row[3] if isinstance(row[3], str) else "",
                    "coarse_type": row[4] if isinstance(row[4], str) else "",
                    "fine_type": row[5] if isinstance(row[5], str) else "",
                    "item_category": row[6] if isinstance(row[6], str) else "",
                    "popularity": float(popularity),
                    "prior": float(prior),
                    "cross_refs": {
                        key: value
                        for key, value in (
                            (
                                "wikipedia",
                                _expand_wikipedia_ref(row[9]) if isinstance(row[9], str) else "",
                            ),
                            (
                                "dbpedia",
                                _expand_dbpedia_ref(row[10]) if isinstance(row[10], str) else "",
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
        context_query: str,
        crosslink_query: str,
        coarse_hints: Sequence[str],
        fine_hints: Sequence[str],
        size: int,
    ) -> list[dict[str, Any]]:
        if not mention_query or size <= 0:
            return []
        aliases_expr = "COALESCE(alpaca_join_text_array(aliases), '')"
        cross_ref_url_expr = "(COALESCE(wikipedia_url, '') || ' ' || COALESCE(dbpedia_url, ''))"
        aliases_nonempty_pred = "COALESCE(array_length(aliases, 1), 0) > 0"
        cross_refs_nonempty_pred = (
            "(COALESCE(wikipedia_url, '') <> '' OR COALESCE(dbpedia_url, '') <> '')"
        )
        where_parts = [
            "("
            "search_vector @@ plainto_tsquery('simple', %s) OR "
            f"label % %s OR ({aliases_nonempty_pred} AND {aliases_expr} % %s) OR "
            f"(%s <> '' AND {cross_refs_nonempty_pred} AND {cross_ref_url_expr} % %s)"
            ")"
        ]
        params: list[Any] = [
            mention_query,
            mention_query,
            mention_query,
            crosslink_query,
            crosslink_query,
        ]
        if coarse_hints:
            where_parts.append("coarse_type = ANY(%s)")
            params.append(list(coarse_hints))
        if fine_hints:
            where_parts.append("fine_type = ANY(%s)")
            params.append(list(fine_hints))

        sql = f"""
        SELECT
            qid, label, aliases, context_string,
            coarse_type, fine_type, item_category, popularity, prior, wikipedia_url, dbpedia_url,
            (
                COALESCE(ts_rank_cd(search_vector, plainto_tsquery('simple', %s)), 0.0) * 5.0 +
                GREATEST(
                    COALESCE(similarity(label, %s), 0.0),
                    COALESCE(similarity({aliases_expr}, %s), 0.0)
                ) * 2.0 +
                CASE
                    WHEN %s <> '' THEN COALESCE(similarity({cross_ref_url_expr}, %s), 0.0) * 1.5
                    ELSE 0.0
                END +
                CASE
                    WHEN %s <> '' THEN COALESCE(
                        ts_rank_cd(
                            to_tsvector('simple', alpaca_filter_fts_context(LEFT(context_string, 256))),
                            plainto_tsquery('simple', %s)
                        ),
                        0.0
                    )
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
            crosslink_query,
            crosslink_query,
            context_query,
            context_query,
        ]
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (*score_params, *params, int(size)))
                rows = cur.fetchall()
        return self._lookup_rows_to_candidates(rows)

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

    def recreate_entities_like_table(
        self,
        table_name: str,
        *,
        drop_existing: bool = True,
        unlogged: bool = False,
    ) -> None:
        table_ident = _quote_identifier(table_name)
        ddl_parts = []
        if drop_existing:
            ddl_parts.append(f"DROP TABLE IF EXISTS {table_ident}")
        table_kind = "CREATE UNLOGGED TABLE" if unlogged else "CREATE TABLE"
        ddl_parts.append(
            f"{table_kind} "
            f"{table_ident} "
            "(LIKE entities INCLUDING DEFAULTS INCLUDING CONSTRAINTS INCLUDING STORAGE INCLUDING COMPRESSION)"
        )
        ddl = ";\n".join(ddl_parts) + ";"
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(ddl)
            conn.commit()

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
        on_chunk: Callable[[dict[str, int]], None] | None = None,
        disable_synchronous_commit: bool = False,
    ) -> dict[str, int]:
        if target_rows <= 0:
            raise ValueError("target_rows must be > 0")
        if seed_rows < 0:
            raise ValueError("seed_rows must be >= 0")
        if batch_rows <= 0:
            raise ValueError("batch_rows must be > 0")

        dest_ident = _quote_identifier(dest_table)
        temp_seed = "_alpaca_entities_sim_seed"
        temp_seed_ident = _quote_identifier(temp_seed)

        seed_limit_sql = "" if seed_rows == 0 else " LIMIT %s"
        seed_limit_params: tuple[Any, ...] = () if seed_rows == 0 else (int(seed_rows),)
        create_seed_sql = (
            f"CREATE TEMP TABLE {temp_seed_ident} AS "
            f"SELECT * FROM entities ORDER BY qid{seed_limit_sql}"
        )

        insert_sql = f"""
        INSERT INTO {dest_ident} (
            qid, label, context_string,
            aliases, search_vector,
            coarse_type, fine_type, item_category, popularity, prior,
            wikipedia_url, dbpedia_url, updated_at
        )
        SELECT
            CASE
                WHEN seed.qid ~ '^[QP][0-9]+$' THEN
                    SUBSTRING(seed.qid FROM 1 FOR 1) ||
                    (
                        SUBSTRING(seed.qid FROM 2)::bigint +
                        (gs.replica_no * %s::bigint)
                    )::text
                ELSE seed.qid || '__r' || gs.replica_no::text
            END AS qid,
            seed.label,
            seed.context_string,
            seed.aliases,
            seed.search_vector,
            seed.coarse_type,
            seed.fine_type,
            seed.item_category,
            seed.popularity,
            seed.prior,
            seed.wikipedia_url,
            seed.dbpedia_url,
            seed.updated_at
        FROM {temp_seed_ident} AS seed
        CROSS JOIN generate_series(%s::bigint, %s::bigint) AS gs(replica_no)
        LIMIT %s
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
                max_replicas_per_chunk = max(1, int(batch_rows) // seed_count)
                remaining = int(target_rows)
                replica_cursor = 0
                inserted_total = 0
                chunk_count = 0

                while remaining > 0:
                    replicas_in_chunk = max(
                        1,
                        min(
                            max_replicas_per_chunk,
                            (remaining + seed_count - 1) // seed_count,
                        ),
                    )
                    replica_end = replica_cursor + replicas_in_chunk - 1
                    rows_this_chunk = min(remaining, seed_count * replicas_in_chunk)
                    cur.execute(
                        insert_sql,
                        (
                            int(stride),
                            int(replica_cursor),
                            int(replica_end),
                            int(rows_this_chunk),
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
                    replica_cursor = replica_end + 1
                    chunk_count += 1
                    conn.commit()
                    if on_chunk is not None:
                        try:
                            on_chunk(
                                {
                                    "chunk_index": int(chunk_count),
                                    "chunk_rows": int(inserted),
                                    "rows_inserted_total": int(inserted_total),
                                    "rows_remaining": int(remaining),
                                    "replicas_emitted": int(replica_cursor),
                                }
                            )
                        except Exception:
                            # Progress callbacks are best-effort and must not affect data writes.
                            pass

                return {
                    "seed_rows_used": seed_count,
                    "target_rows": int(target_rows),
                    "inserted_rows": inserted_total,
                    "qid_stride": int(stride),
                    "replicas_emitted": int(replica_cursor),
                    "chunks": int(chunk_count),
                }

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
            pg_total_relation_size(%s::regclass) AS total_bytes
        """
        sql = sql.replace("%s_placeholder", _quote_identifier(table_name))
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (table_name, table_name, table_name))
                row = cur.fetchone()
        if not row or len(row) < 4:
            return {"rows": 0, "table_bytes": 0, "index_bytes": 0, "total_bytes": 0}
        return {
            "rows": int(row[0]) if isinstance(row[0], int) else 0,
            "table_bytes": int(row[1]) if isinstance(row[1], int) else 0,
            "index_bytes": int(row[2]) if isinstance(row[2], int) else 0,
            "total_bytes": int(row[3]) if isinstance(row[3], int) else 0,
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
            "labels",              # legacy split name variants (or older multilingual map)
            "name_variants",       # temporary normalized alias column (renamed back to aliases)
            "relation_object_qids",  # only needed to rebuild context_string
            "labels_text",         # legacy text helper
            "aliases_text",        # legacy text helper replaced by expression indexes
            "search_document",     # duplicated source text for search_vector
            "cross_refs_text",     # legacy text helper replaced by expression indexes
            "cross_refs",          # raw JSONB; lookup uses explicit URL columns
        ]

        index_drops: list[str] = []
        if drop_cross_refs_trgm_index:
            index_drops.append(f"idx_{table_name}_cross_refs_text_trgm")
            index_drops.append(f"idx_{table_name}_cross_refs_url_trgm")
        index_drops.append(f"idx_{table_name}_aliases_text_trgm")
        index_drops.append(f"idx_{table_name}_label_exact")
        index_drops.append(f"idx_{table_name}_aliases_exact")
        index_drops.append(f"idx_{table_name}_cross_refs_exact")
        dropped_tables: list[str] = []

        with self._connect() as conn:
            with conn.cursor() as cur:
                for column_name in dropped_columns:
                    cur.execute(f"ALTER TABLE {table_ident} DROP COLUMN IF EXISTS {_quote_identifier(column_name)}")
                for index_name in index_drops:
                    cur.execute(f"DROP INDEX IF EXISTS {_quote_identifier(index_name)}")
                if drop_context_inputs_table and table_name == "entities":
                    cur.execute("DROP TABLE IF EXISTS entity_context_inputs")
                    dropped_tables.append("entity_context_inputs")
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
            with conn.cursor() as cur:
                cur.execute("TRUNCATE TABLE entities")
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
