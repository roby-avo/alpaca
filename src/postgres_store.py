from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

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
    coarse_type: str
    fine_type: str
    popularity: float
    cross_refs: Mapping[str, Any]
    context_string: str = ""


def _json_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


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
        CREATE TABLE IF NOT EXISTS entities (
            qid TEXT PRIMARY KEY,
            label TEXT NOT NULL,
            labels JSONB NOT NULL DEFAULT '{}'::jsonb,
            aliases JSONB NOT NULL DEFAULT '{}'::jsonb,
            relation_object_qids JSONB NOT NULL DEFAULT '[]'::jsonb,
            context_string TEXT NOT NULL DEFAULT '',
            coarse_type TEXT NOT NULL DEFAULT '',
            fine_type TEXT NOT NULL DEFAULT '',
            popularity DOUBLE PRECISION NOT NULL DEFAULT 0,
            cross_refs JSONB NOT NULL DEFAULT '{}'::jsonb,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS idx_entities_coarse_type ON entities (coarse_type);
        CREATE INDEX IF NOT EXISTS idx_entities_fine_type ON entities (fine_type);

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
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(ddl)
            conn.commit()

    def upsert_entities(self, rows: Sequence[EntityRecord]) -> int:
        if not rows:
            return 0
        sql = """
        INSERT INTO entities (
            qid, label, labels, aliases, relation_object_qids, context_string,
            coarse_type, fine_type, popularity, cross_refs
        ) VALUES (
            %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s,
            %s, %s, %s, %s::jsonb
        )
        ON CONFLICT (qid) DO UPDATE SET
            label = EXCLUDED.label,
            labels = EXCLUDED.labels,
            aliases = EXCLUDED.aliases,
            relation_object_qids = EXCLUDED.relation_object_qids,
            context_string = EXCLUDED.context_string,
            coarse_type = EXCLUDED.coarse_type,
            fine_type = EXCLUDED.fine_type,
            popularity = EXCLUDED.popularity,
            cross_refs = EXCLUDED.cross_refs,
            updated_at = NOW()
        """
        payload = [
            (
                row.qid,
                row.label,
                _json_compact(row.labels),
                _json_compact(row.aliases),
                _json_compact(list(row.relation_object_qids)),
                row.context_string,
                row.coarse_type,
                row.fine_type,
                float(row.popularity),
                _json_compact(row.cross_refs),
            )
            for row in rows
        ]
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.executemany(sql, payload)
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
        FROM entities
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
        SELECT qid, label, labels
        FROM entities
        WHERE qid = ANY(%s)
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (list(qids),))
                rows = cur.fetchall()
        resolved: dict[str, str] = {}
        for qid, label, labels_json in rows:
            if not isinstance(qid, str):
                continue
            if isinstance(label, str) and label.strip():
                resolved[qid] = label.strip()
                continue
            labels = _as_text_map(labels_json)
            preferred = labels.get("en")
            if isinstance(preferred, str) and preferred:
                resolved[qid] = preferred
                continue
            if labels:
                first_language = sorted(labels)[0]
                resolved[qid] = labels[first_language]
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
            updated_at = NOW()
        WHERE qid = %s
        """
        payload = [(context, qid) for qid, context in rows]
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
            qid, label, labels, aliases, context_string,
            coarse_type, fine_type, popularity, cross_refs
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
                        if len(row) < 9:
                            continue
                        qid = row[0]
                        label = row[1]
                        if not isinstance(qid, str) or not isinstance(label, str):
                            continue
                        out.append(
                            {
                                "qid": qid,
                                "label": label,
                                "labels": _as_text_map(row[2]),
                                "aliases": _as_alias_map(row[3]),
                                "context_string": row[4] if isinstance(row[4], str) else "",
                                "coarse_type": row[5] if isinstance(row[5], str) else "",
                                "fine_type": row[6] if isinstance(row[6], str) else "",
                                "popularity": float(row[7]) if isinstance(row[7], (int, float)) else 0.0,
                                "cross_refs": _as_json_object(row[8]),
                            }
                        )
                    if out:
                        yield out

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
