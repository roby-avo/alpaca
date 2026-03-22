from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ALL_COMPLETED, FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .common import resolve_configured_str, resolve_postgres_dsn, running_in_container, tqdm
from .postgres_store import PostgresStore
try:  # pragma: no cover - import depends on runtime environment
    import psycopg  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    psycopg = None  # type: ignore


ELASTICSEARCH_URL_ENV = "ALPACA_ELASTICSEARCH_URL"
DEFAULT_ELASTICSEARCH_URL_LOCAL = "http://localhost:9200"
DEFAULT_ELASTICSEARCH_URL_DOCKER = "http://elasticsearch:9200"
DEFAULT_INDEX_NAME = "alpaca-entities"
DEFAULT_TABLE_NAME = "entities"
DEFAULT_FETCH_SIZE = 15_000
DEFAULT_BULK_ACTIONS = 2_500
DEFAULT_WORKERS = max(1, min(6, (os.cpu_count() or 1)))
DEFAULT_MAX_INFLIGHT_FACTOR = 2
DEFAULT_MAX_RETRIES = 4
DEFAULT_RETRY_BACKOFF_SECONDS = 1.0
DEFAULT_REQUEST_TIMEOUT_SECONDS = 90.0
DEFAULT_FINAL_REFRESH_INTERVAL = "1s"
DEFAULT_FINAL_REPLICAS = 0
DEFAULT_WAIT_INDEX_READY_TIMEOUT_SECONDS = 180.0
DEFAULT_MAX_INDEXED_LABELS = 12
DEFAULT_MAX_INDEXED_ALIASES = 24
DEFAULT_MAX_CONTEXT_CHARS = 256

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class ElasticsearchIndexingError(RuntimeError):
    pass


def default_elasticsearch_url() -> str:
    if running_in_container():
        return DEFAULT_ELASTICSEARCH_URL_DOCKER
    return DEFAULT_ELASTICSEARCH_URL_LOCAL


def parse_positive_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return value


def parse_non_negative_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if value < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return value


def parse_positive_float(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return value


def _require_psycopg() -> Any:
    if psycopg is None:
        raise ElasticsearchIndexingError(
            "psycopg is not installed. Install dependencies with PostgreSQL support first."
        )
    return psycopg


def _normalize_es_url(raw: str) -> str:
    value = raw.strip()
    if not value:
        raise ValueError("Elasticsearch URL must be non-empty.")
    if not (value.startswith("http://") or value.startswith("https://")):
        raise ValueError("Elasticsearch URL must start with http:// or https://")
    return value.rstrip("/")


def _quote_table_name(raw: str) -> str:
    cleaned = raw.strip()
    if not cleaned:
        raise ValueError("Table name must be non-empty.")
    parts = cleaned.split(".")
    if len(parts) > 2:
        raise ValueError(
            f"Invalid table name '{raw}'. Use '<table>' or '<schema>.<table>' with letters/digits/_ only."
        )
    quoted: list[str] = []
    for part in parts:
        if not _IDENTIFIER_RE.match(part):
            raise ValueError(
                f"Invalid table identifier '{part}' in '{raw}'. Use letters/digits/_ only."
            )
        quoted.append(f'"{part}"')
    return ".".join(quoted)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stream entities from PostgreSQL and bulk-index them into Elasticsearch. "
            "Whatever currently exists in Postgres gets indexed (live sample or full production)."
        )
    )
    parser.add_argument("--postgres-dsn", help="Postgres DSN (defaults to ALPACA_POSTGRES_DSN).")
    parser.add_argument(
        "--elasticsearch-url",
        help=(
            "Elasticsearch base URL (defaults to ALPACA_ELASTICSEARCH_URL or "
            f"{DEFAULT_ELASTICSEARCH_URL_DOCKER} inside containers / "
            f"{DEFAULT_ELASTICSEARCH_URL_LOCAL} outside containers)."
        ),
    )
    parser.add_argument(
        "--index-name",
        default=DEFAULT_INDEX_NAME,
        help=f"Target Elasticsearch index name (default: {DEFAULT_INDEX_NAME}).",
    )
    parser.add_argument(
        "--table",
        default=DEFAULT_TABLE_NAME,
        help=f"Postgres source table (default: {DEFAULT_TABLE_NAME}).",
    )
    parser.add_argument(
        "--batch-size",
        type=parse_positive_int,
        default=DEFAULT_FETCH_SIZE,
        help=f"Rows fetched per server-side Postgres cursor read (default: {DEFAULT_FETCH_SIZE}).",
    )
    parser.add_argument(
        "--bulk-actions",
        type=parse_positive_int,
        default=DEFAULT_BULK_ACTIONS,
        help=f"Documents per Elasticsearch _bulk request (default: {DEFAULT_BULK_ACTIONS}).",
    )
    parser.add_argument(
        "--workers",
        type=parse_positive_int,
        default=DEFAULT_WORKERS,
        help=f"Parallel Elasticsearch bulk worker threads (default: {DEFAULT_WORKERS}).",
    )
    parser.add_argument(
        "--max-inflight",
        type=parse_non_negative_int,
        default=0,
        help=(
            "Maximum submitted bulk requests not yet completed. "
            f"Default: workers * {DEFAULT_MAX_INFLIGHT_FACTOR}."
        ),
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=parse_positive_float,
        default=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        help=f"HTTP timeout for Elasticsearch requests (default: {DEFAULT_REQUEST_TIMEOUT_SECONDS}).",
    )
    parser.add_argument(
        "--max-retries",
        type=parse_non_negative_int,
        default=DEFAULT_MAX_RETRIES,
        help=f"Retries for retryable bulk failures (default: {DEFAULT_MAX_RETRIES}).",
    )
    parser.add_argument(
        "--retry-backoff-seconds",
        type=parse_positive_float,
        default=DEFAULT_RETRY_BACKOFF_SECONDS,
        help=(
            f"Base exponential backoff between retries in seconds "
            f"(default: {DEFAULT_RETRY_BACKOFF_SECONDS})."
        ),
    )
    parser.add_argument(
        "--updated-since",
        help=(
            "Optional lower bound for updated_at filtering. "
            "Accepted by Postgres as timestamptz, e.g. '2026-03-01T00:00:00Z'."
        ),
    )
    parser.add_argument(
        "--recreate-index",
        action="store_true",
        help="Delete and recreate the index before indexing.",
    )
    parser.add_argument(
        "--skip-index-setup",
        action="store_true",
        help="Do not create/delete/tune index settings before indexing.",
    )
    parser.add_argument(
        "--final-refresh-interval",
        default=DEFAULT_FINAL_REFRESH_INTERVAL,
        help=(
            "Refresh interval applied after indexing (default: "
            f"{DEFAULT_FINAL_REFRESH_INTERVAL})."
        ),
    )
    parser.add_argument(
        "--final-replicas",
        type=parse_non_negative_int,
        default=DEFAULT_FINAL_REPLICAS,
        help=f"number_of_replicas after indexing (default: {DEFAULT_FINAL_REPLICAS}).",
    )
    parser.add_argument(
        "--skip-finalize-settings",
        action="store_true",
        help="Do not restore refresh/replica settings after indexing.",
    )
    parser.add_argument(
        "--skip-count-total",
        action="store_true",
        help=(
            "Skip counting source rows before indexing. "
            "Progress bar still updates but without fixed total."
        ),
    )
    parser.add_argument(
        "--exact-count-total",
        action="store_true",
        help=(
            "Use exact COUNT(*) for the progress bar total instead of a fast estimate. "
            "This can be much slower on large tables."
        ),
    )
    parser.add_argument(
        "--wait-index-ready-timeout-seconds",
        type=parse_positive_float,
        default=DEFAULT_WAIT_INDEX_READY_TIMEOUT_SECONDS,
        help=(
            "How long to wait for the target index primary shard to become active before "
            f"starting bulk indexing (default: {DEFAULT_WAIT_INDEX_READY_TIMEOUT_SECONDS})."
        ),
    )
    parser.add_argument(
        "--max-indexed-labels",
        type=parse_non_negative_int,
        default=DEFAULT_MAX_INDEXED_LABELS,
        help=(
            "Maximum secondary labels mirrored into Elasticsearch per document. "
            f"0 disables them (default: {DEFAULT_MAX_INDEXED_LABELS})."
        ),
    )
    parser.add_argument(
        "--max-indexed-aliases",
        type=parse_non_negative_int,
        default=DEFAULT_MAX_INDEXED_ALIASES,
        help=(
            "Maximum aliases mirrored into Elasticsearch per document. "
            f"0 disables them (default: {DEFAULT_MAX_INDEXED_ALIASES})."
        ),
    )
    parser.add_argument(
        "--max-context-chars",
        type=parse_non_negative_int,
        default=DEFAULT_MAX_CONTEXT_CHARS,
        help=(
            "Maximum graph-derived context characters stored per document. "
            f"0 disables context_string (default: {DEFAULT_MAX_CONTEXT_CHARS})."
        ),
    )
    return parser.parse_args()


def _es_request_json(
    *,
    base_url: str,
    method: str,
    path: str,
    body: Mapping[str, Any] | None = None,
    timeout_seconds: float,
    expected_statuses: Sequence[int],
) -> dict[str, Any]:
    payload = None
    headers: dict[str, str] = {}
    if body is not None:
        payload = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(f"{base_url}{path}", data=payload, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            status = int(response.getcode())
            response_body = response.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise ElasticsearchIndexingError(
            f"Elasticsearch {method} {path} failed with status {exc.code}: {detail[:1500]}"
        ) from exc
    except URLError as exc:
        raise ElasticsearchIndexingError(
            f"Elasticsearch {method} {path} failed: {exc.reason}"
        ) from exc

    if status not in expected_statuses:
        text = response_body.decode("utf-8", errors="replace")
        raise ElasticsearchIndexingError(
            f"Elasticsearch {method} {path} returned unexpected status {status}: {text[:1500]}"
        )
    if not response_body:
        return {}
    try:
        parsed = json.loads(response_body.decode("utf-8"))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _es_index_exists(*, base_url: str, index_name: str, timeout_seconds: float) -> bool:
    request = Request(f"{base_url}/{index_name}", method="HEAD")
    try:
        with urlopen(request, timeout=timeout_seconds):
            return True
    except HTTPError as exc:
        if exc.code == 404:
            return False
        detail = exc.read().decode("utf-8", errors="replace")
        raise ElasticsearchIndexingError(
            f"Could not check index '{index_name}' (status {exc.code}): {detail[:1000]}"
        ) from exc
    except URLError as exc:
        raise ElasticsearchIndexingError(
            f"Could not check index '{index_name}': {exc.reason}"
        ) from exc


def _build_index_payload() -> dict[str, Any]:
    return {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "refresh_interval": "-1",
            "codec": "best_compression",
            "analysis": {
                "filter": {
                    "alpaca_edge_2_20": {
                        "type": "edge_ngram",
                        "min_gram": 2,
                        "max_gram": 20,
                    }
                },
                "normalizer": {
                    "alpaca_keyword_lower": {
                        "type": "custom",
                        "filter": ["lowercase", "asciifolding"],
                    }
                },
                "analyzer": {
                    "alpaca_text": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": ["lowercase", "asciifolding"],
                    },
                    "alpaca_prefix_index": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": ["lowercase", "asciifolding", "alpaca_edge_2_20"],
                    },
                    "alpaca_prefix_search": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": ["lowercase", "asciifolding"],
                    },
                },
            },
        },
        "mappings": {
            "dynamic": "strict",
            "properties": {
                "qid": {"type": "keyword", "index": False, "doc_values": False},
                "label": {
                    "type": "text",
                    "analyzer": "alpaca_text",
                    "fields": {
                        "keyword": {
                            "type": "keyword",
                            "normalizer": "alpaca_keyword_lower",
                            "ignore_above": 512,
                        },
                        "prefix": {
                            "type": "text",
                            "analyzer": "alpaca_prefix_index",
                            "search_analyzer": "alpaca_prefix_search",
                            "norms": False,
                        }
                    },
                },
                "labels": {
                    "type": "text",
                    "analyzer": "alpaca_text",
                    "fields": {
                        "prefix": {
                            "type": "text",
                            "analyzer": "alpaca_prefix_index",
                            "search_analyzer": "alpaca_prefix_search",
                            "norms": False,
                        }
                    },
                },
                "description": {"type": "text", "index": False},
                "aliases": {
                    "type": "text",
                    "analyzer": "alpaca_text",
                    "fields": {
                        "prefix": {
                            "type": "text",
                            "analyzer": "alpaca_prefix_index",
                            "search_analyzer": "alpaca_prefix_search",
                            "norms": False,
                        }
                    },
                },
                "context_string": {
                    "type": "text",
                    "analyzer": "alpaca_text",
                    "norms": False,
                },
                "coarse_type": {
                    "type": "keyword",
                    "normalizer": "alpaca_keyword_lower",
                },
                "fine_type": {
                    "type": "keyword",
                    "normalizer": "alpaca_keyword_lower",
                },
                "item_category": {
                    "type": "keyword",
                    "normalizer": "alpaca_keyword_lower",
                },
                "types": {
                    "type": "keyword",
                    "index": False,
                    "doc_values": False,
                },
                "popularity": {"type": "float"},
                "prior": {"type": "half_float"},
                # Compact refs from Postgres, without full URL prefixes.
                "wikipedia_url": {
                    "type": "keyword",
                    "normalizer": "alpaca_keyword_lower",
                    "ignore_above": 2048,
                },
                "dbpedia_url": {
                    "type": "keyword",
                    "normalizer": "alpaca_keyword_lower",
                    "ignore_above": 2048,
                },
                "updated_at": {"type": "date"},
            },
        },
    }


def _prepare_index(
    *,
    base_url: str,
    index_name: str,
    recreate_index: bool,
    timeout_seconds: float,
) -> None:
    exists = _es_index_exists(
        base_url=base_url,
        index_name=index_name,
        timeout_seconds=timeout_seconds,
    )
    if exists and recreate_index:
        print(f"Deleting existing index {index_name}...")
        _es_request_json(
            base_url=base_url,
            method="DELETE",
            path=f"/{index_name}",
            timeout_seconds=timeout_seconds,
            expected_statuses=(200,),
        )
        exists = False

    if not exists:
        print(f"Creating index {index_name} with Alpaca mapping/settings...")
        _es_request_json(
            base_url=base_url,
            method="PUT",
            path=f"/{index_name}",
            body=_build_index_payload(),
            timeout_seconds=timeout_seconds,
            expected_statuses=(200,),
        )
        return

    print(
        f"Index {index_name} already exists. Setting refresh_interval=-1 and number_of_replicas=0 "
        "for faster bulk ingestion."
    )
    _es_request_json(
        base_url=base_url,
        method="PUT",
        path=f"/{index_name}/_settings",
        body={"index": {"refresh_interval": "-1", "number_of_replicas": 0}},
        timeout_seconds=timeout_seconds,
        expected_statuses=(200,),
    )


def _wait_for_index_ready(
    *,
    base_url: str,
    index_name: str,
    wait_timeout_seconds: float,
    request_timeout_seconds: float,
) -> None:
    timeout_s = max(1, int(wait_timeout_seconds))
    result = _es_request_json(
        base_url=base_url,
        method="GET",
        path=(
            f"/_cluster/health/{index_name}"
            f"?wait_for_status=yellow&wait_for_active_shards=1&timeout={timeout_s}s"
        ),
        timeout_seconds=max(request_timeout_seconds, wait_timeout_seconds + 5.0),
        expected_statuses=(200,),
    )
    timed_out = bool(result.get("timed_out"))
    active_primary = int(result.get("active_primary_shards", 0))
    status = str(result.get("status", "unknown"))
    if timed_out or active_primary < 1 or status == "red":
        details = (
            f"status={status} timed_out={timed_out} "
            f"active_primary_shards={active_primary} "
            f"unassigned_shards={int(result.get('unassigned_shards', 0))} "
            f"number_of_nodes={int(result.get('number_of_nodes', 0))}"
        )
        raise ElasticsearchIndexingError(
            f"Index '{index_name}' is not ready for writes. {details}"
        )


def _finalize_index(
    *,
    base_url: str,
    index_name: str,
    final_refresh_interval: str,
    final_replicas: int,
    timeout_seconds: float,
) -> None:
    _es_request_json(
        base_url=base_url,
        method="PUT",
        path=f"/{index_name}/_settings",
        body={
            "index": {
                "refresh_interval": final_refresh_interval,
                "number_of_replicas": int(final_replicas),
            }
        },
        timeout_seconds=timeout_seconds,
        expected_statuses=(200,),
    )
    _es_request_json(
        base_url=base_url,
        method="POST",
        path=f"/{index_name}/_refresh",
        timeout_seconds=timeout_seconds,
        expected_statuses=(200,),
    )


def _clean_terms(
    raw: Any,
    *,
    max_terms: int | None = None,
    excluded: set[str] | None = None,
) -> list[str]:
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    blocked = excluded if excluded is not None else set()
    for value in raw:
        if not isinstance(value, str):
            continue
        cleaned = value.strip()
        if not cleaned or cleaned in seen or cleaned in blocked:
            continue
        seen.add(cleaned)
        out.append(cleaned)
        if max_terms is not None and len(out) >= max_terms:
            break
    return out


def _as_float(raw: Any) -> float:
    if isinstance(raw, (int, float)):
        return float(raw)
    return 0.0


def _as_iso_datetime(raw: Any) -> str:
    if isinstance(raw, datetime):
        return raw.isoformat()
    if isinstance(raw, str):
        return raw.strip()
    return ""


def _row_to_document(
    row: Sequence[Any],
    *,
    max_indexed_labels: int = DEFAULT_MAX_INDEXED_LABELS,
    max_indexed_aliases: int = DEFAULT_MAX_INDEXED_ALIASES,
) -> dict[str, Any] | None:
    if len(row) < 14:
        return None
    qid = row[0]
    label = row[1]
    if not isinstance(qid, str) or not isinstance(label, str):
        return None
    clean_label = label.strip()
    if not clean_label:
        return None
    labels = _clean_terms(
        row[2],
        max_terms=max_indexed_labels,
        excluded={clean_label},
    )
    aliases = _clean_terms(
        row[3],
        max_terms=max_indexed_aliases,
        excluded={clean_label, *labels},
    )
    has_context_column = len(row) >= 15
    shift = 1 if has_context_column else 0
    context_string = row[6] if has_context_column and isinstance(row[6], str) else ""

    doc: dict[str, Any] = {
        "qid": qid,
        "label": clean_label,
        "labels": labels,
        "aliases": aliases,
        "types": _clean_terms(row[5]),
        "context_string": context_string,
        "coarse_type": row[6 + shift] if isinstance(row[6 + shift], str) else "",
        "fine_type": row[7 + shift] if isinstance(row[7 + shift], str) else "",
        "item_category": row[8 + shift] if isinstance(row[8 + shift], str) else "",
        "popularity": _as_float(row[9 + shift]),
        "prior": _as_float(row[10 + shift]),
        # Keep compact refs from Postgres to save space.
        "wikipedia_url": row[11 + shift] if isinstance(row[11 + shift], str) else "",
        "dbpedia_url": row[12 + shift] if isinstance(row[12 + shift], str) else "",
    }
    description = row[4] if isinstance(row[4], str) else ""
    if description:
        doc["description"] = description
    updated_at = _as_iso_datetime(row[13 + shift])
    if updated_at:
        doc["updated_at"] = updated_at
    return doc


def _iter_documents_from_postgres(
    *,
    postgres_dsn: str,
    table_name: str,
    batch_size: int,
    updated_since: str | None,
    max_indexed_labels: int,
    max_indexed_aliases: int,
    max_context_chars: int,
    yield_chunk_size: int,
) -> Iterator[list[dict[str, Any]]]:
    pg = _require_psycopg()
    table_sql = _quote_table_name(table_name)
    context_store = PostgresStore(postgres_dsn)
    table_name_tail = table_name.strip().split(".")[-1]
    effective_yield_chunk_size = min(max(1, int(yield_chunk_size)), max(1, int(batch_size)))
    source_alias = "src"
    where_sql = ""
    params: list[Any] = []
    if updated_since:
        where_sql = f"WHERE {source_alias}.updated_at >= %s::timestamptz"
        params.append(updated_since)

    sql = f"""
    SELECT
        {source_alias}.qid,
        {source_alias}.label,
        {source_alias}.labels,
        {source_alias}.aliases,
        {source_alias}.description,
        {source_alias}.types,
        {source_alias}.coarse_type,
        {source_alias}.fine_type,
        {source_alias}.item_category,
        {source_alias}.popularity,
        {source_alias}.prior,
        {source_alias}.wikipedia_url,
        {source_alias}.dbpedia_url,
        {source_alias}.updated_at
    FROM {table_sql} AS {source_alias}
    {where_sql}
    ORDER BY {source_alias}.qid
    """
    with pg.connect(postgres_dsn) as conn:
        with context_store._connect() as context_conn:
            # Named cursor streams rows from server to avoid loading the whole table.
            with conn.cursor(name="alpaca_es_export_cursor") as cur:
                cur.itersize = int(batch_size)
                cur.execute(sql, tuple(params))
                while True:
                    rows = cur.fetchmany(int(batch_size))
                    if not rows:
                        return
                    docs: list[dict[str, Any]] = []
                    for row in rows:
                        doc = _row_to_document(
                            row,
                            max_indexed_labels=max_indexed_labels,
                            max_indexed_aliases=max_indexed_aliases,
                        )
                        if doc:
                            docs.append(doc)
                    if docs:
                        if table_name_tail == "entities":
                            for docs_chunk in _chunked(docs, effective_yield_chunk_size):
                                yield context_store.attach_context_strings(
                                    docs_chunk,
                                    chunk_size=effective_yield_chunk_size,
                                    max_chars=max_context_chars,
                                    conn=context_conn,
                                )
                        else:
                            for docs_chunk in _chunked(docs, effective_yield_chunk_size):
                                yield docs_chunk


def _prefer_estimated_row_total(*values: Any) -> int | None:
    candidates: list[int] = []
    for raw in values:
        if not isinstance(raw, (int, float)):
            continue
        value = int(raw)
        if value <= 0:
            continue
        candidates.append(value)
    if not candidates:
        return None
    return max(candidates)


def _extract_explain_plan_rows(raw: Any) -> int | None:
    parsed = raw
    if isinstance(parsed, str):
        try:
            parsed = json.loads(parsed)
        except json.JSONDecodeError:
            return None
    if not isinstance(parsed, list) or not parsed:
        return None
    first = parsed[0]
    if not isinstance(first, Mapping):
        return None
    plan = first.get("Plan")
    if not isinstance(plan, Mapping):
        return None
    return _prefer_estimated_row_total(plan.get("Plan Rows"))


def _estimate_source_rows(
    *,
    postgres_dsn: str,
    table_name: str,
    updated_since: str | None,
) -> int | None:
    pg = _require_psycopg()
    table_sql = _quote_table_name(table_name)
    with pg.connect(postgres_dsn) as conn:
        with conn.cursor() as cur:
            if updated_since is None:
                cur.execute(
                    """
                    SELECT c.reltuples, s.n_live_tup
                    FROM pg_class AS c
                    LEFT JOIN pg_stat_all_tables AS s ON s.relid = c.oid
                    WHERE c.oid = to_regclass(%s)
                    """,
                    (table_name,),
                )
                stats_row = cur.fetchone()
                if stats_row:
                    estimate = _prefer_estimated_row_total(stats_row[0], stats_row[1])
                    if estimate is not None:
                        return estimate

            where_sql = ""
            params: list[Any] = []
            if updated_since:
                where_sql = "WHERE updated_at >= %s::timestamptz"
                params.append(updated_since)
            cur.execute(f"EXPLAIN (FORMAT JSON) SELECT 1 FROM {table_sql} {where_sql}", tuple(params))
            row = cur.fetchone()
    if not row:
        return None
    return _extract_explain_plan_rows(row[0])


def _count_source_rows_exact(
    *,
    postgres_dsn: str,
    table_name: str,
    updated_since: str | None,
) -> int:
    pg = _require_psycopg()
    table_sql = _quote_table_name(table_name)
    where_sql = ""
    params: list[Any] = []
    if updated_since:
        where_sql = "WHERE updated_at >= %s::timestamptz"
        params.append(updated_since)

    sql = f"SELECT COUNT(*) FROM {table_sql} {where_sql}"
    with pg.connect(postgres_dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, tuple(params))
            row = cur.fetchone()
    if not row:
        return 0
    count = row[0]
    return int(count) if isinstance(count, int) else 0


def _chunked(values: Sequence[dict[str, Any]], size: int) -> Iterator[list[dict[str, Any]]]:
    if size <= 0:
        raise ValueError("chunk size must be > 0")
    for start in range(0, len(values), size):
        yield list(values[start : start + size])


def _bulk_payload(index_name: str, docs: Sequence[dict[str, Any]]) -> bytes:
    lines: list[str] = []
    for doc in docs:
        qid = doc.get("qid")
        if not isinstance(qid, str) or not qid:
            continue
        lines.append(json.dumps({"index": {"_index": index_name, "_id": qid}}, separators=(",", ":")))
        lines.append(json.dumps(doc, ensure_ascii=False, separators=(",", ":")))
    return ("\n".join(lines) + "\n").encode("utf-8")


def _post_bulk(
    *,
    base_url: str,
    payload: bytes,
    timeout_seconds: float,
) -> dict[str, Any]:
    request = Request(
        f"{base_url}/_bulk?refresh=false",
        data=payload,
        headers={"Content-Type": "application/x-ndjson"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            status = int(response.getcode())
            raw = response.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise ElasticsearchIndexingError(
            f"Elasticsearch _bulk request failed with status {exc.code}: {detail[:2000]}"
        ) from exc
    except URLError as exc:
        raise ElasticsearchIndexingError(
            f"Elasticsearch _bulk request failed: {exc.reason}"
        ) from exc

    if status != 200:
        detail = raw.decode("utf-8", errors="replace")
        raise ElasticsearchIndexingError(
            f"Elasticsearch _bulk request returned status {status}: {detail[:2000]}"
        )
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ElasticsearchIndexingError(f"Could not parse _bulk response JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ElasticsearchIndexingError("Elasticsearch _bulk response was not a JSON object.")
    return parsed


def _summarize_bulk_failures(response: dict[str, Any]) -> tuple[int, bool, str]:
    items = response.get("items")
    if not isinstance(items, list):
        return 0, False, ""

    failed = 0
    retryable = True
    examples: list[str] = []

    for item in items:
        if not isinstance(item, dict) or not item:
            continue
        op_payload = next(iter(item.values()))
        if not isinstance(op_payload, dict):
            continue
        status = int(op_payload.get("status", 0))
        if status < 300:
            continue
        failed += 1
        error_obj = op_payload.get("error")
        reason = ""
        if isinstance(error_obj, dict):
            reason = str(error_obj.get("reason", "")).strip()
        elif error_obj is not None:
            reason = str(error_obj).strip()
        if len(examples) < 3:
            examples.append(f"status={status} reason={reason or 'n/a'}")
        if not (status >= 500 or status in {409, 429}):
            retryable = False

    return failed, retryable, "; ".join(examples)


def _index_bulk_with_retries(
    *,
    base_url: str,
    payload: bytes,
    doc_count: int,
    timeout_seconds: float,
    max_retries: int,
    retry_backoff_seconds: float,
) -> int:
    attempt = 0
    while True:
        response = _post_bulk(
            base_url=base_url,
            payload=payload,
            timeout_seconds=timeout_seconds,
        )
        if not bool(response.get("errors")):
            return int(doc_count)

        failed_count, retryable, sample = _summarize_bulk_failures(response)
        message = (
            f"Bulk request had {failed_count} failed docs out of {doc_count}. "
            f"Sample errors: {sample or 'n/a'}"
        )
        if not retryable or attempt >= max_retries:
            raise ElasticsearchIndexingError(message)
        sleep_seconds = retry_backoff_seconds * (2**attempt)
        print(
            f"WARN: {message} Retrying in {sleep_seconds:.1f}s "
            f"(attempt {attempt + 1}/{max_retries})..."
        )
        time.sleep(sleep_seconds)
        attempt += 1


def _drain_futures(
    futures: set[Future[int]],
    *,
    return_when: Any,
) -> tuple[int, set[Future[int]]]:
    if not futures:
        return 0, set()
    done, pending = wait(futures, return_when=return_when)
    indexed = 0
    for future in done:
        indexed += int(future.result())
    return indexed, set(pending)


def _run(args: argparse.Namespace) -> int:
    postgres_dsn = resolve_postgres_dsn(args.postgres_dsn)
    elasticsearch_url = _normalize_es_url(
        resolve_configured_str(
            args.elasticsearch_url,
            ELASTICSEARCH_URL_ENV,
            default_elasticsearch_url(),
        )
    )
    index_name = args.index_name.strip()
    if not index_name:
        raise ValueError("Index name must be non-empty.")
    table_name = args.table.strip()
    if not table_name:
        raise ValueError("Table name must be non-empty.")

    _es_request_json(
        base_url=elasticsearch_url,
        method="GET",
        path="/",
        timeout_seconds=float(args.request_timeout_seconds),
        expected_statuses=(200,),
    )

    if not args.skip_index_setup:
        _prepare_index(
            base_url=elasticsearch_url,
            index_name=index_name,
            recreate_index=bool(args.recreate_index),
            timeout_seconds=float(args.request_timeout_seconds),
        )

    total_rows: int | None = None
    if not args.skip_count_total:
        if args.exact_count_total:
            print("Counting source rows exactly for progress bar...")
            total_rows = _count_source_rows_exact(
                postgres_dsn=postgres_dsn,
                table_name=table_name,
                updated_since=args.updated_since,
            )
            print(f"Source rows to index: {total_rows} (exact)")
        else:
            print("Estimating source rows for progress bar from PostgreSQL statistics...")
            total_rows = _estimate_source_rows(
                postgres_dsn=postgres_dsn,
                table_name=table_name,
                updated_since=args.updated_since,
            )
            if total_rows is None:
                print("Fast estimate unavailable. Falling back to exact COUNT(*)...")
                total_rows = _count_source_rows_exact(
                    postgres_dsn=postgres_dsn,
                    table_name=table_name,
                    updated_since=args.updated_since,
                )
                print(f"Source rows to index: {total_rows} (exact fallback)")
            else:
                print(f"Source rows to index: ~{total_rows} (estimated)")

    max_inflight = (
        int(args.max_inflight)
        if int(args.max_inflight) > 0
        else int(args.workers) * DEFAULT_MAX_INFLIGHT_FACTOR
    )
    docs_read = 0
    docs_indexed = 0
    docs_submitted = 0
    bulk_submitted = 0
    futures: set[Future[int]] = set()

    print(
        "Indexing Postgres -> Elasticsearch:",
        f"table={table_name}",
        f"index={index_name}",
        f"batch_size={args.batch_size}",
        f"bulk_actions={args.bulk_actions}",
        f"workers={args.workers}",
        f"max_inflight={max_inflight}",
        f"updated_since={args.updated_since or 'n/a'}",
        f"max_indexed_labels={args.max_indexed_labels}",
        f"max_indexed_aliases={args.max_indexed_aliases}",
        f"max_context_chars={args.max_context_chars}",
    )
    print(
        "Waiting for Elasticsearch index shard to become active:",
        f"index={index_name}",
        f"timeout_seconds={args.wait_index_ready_timeout_seconds}",
    )
    _wait_for_index_ready(
        base_url=elasticsearch_url,
        index_name=index_name,
        wait_timeout_seconds=float(args.wait_index_ready_timeout_seconds),
        request_timeout_seconds=float(args.request_timeout_seconds),
    )

    with ThreadPoolExecutor(max_workers=int(args.workers)) as pool:
        with tqdm(total=total_rows, desc="pg->es", unit="doc") as progress:
            for batch_docs in _iter_documents_from_postgres(
                postgres_dsn=postgres_dsn,
                table_name=table_name,
                batch_size=int(args.batch_size),
                updated_since=args.updated_since,
                max_indexed_labels=int(args.max_indexed_labels),
                max_indexed_aliases=int(args.max_indexed_aliases),
                max_context_chars=int(args.max_context_chars),
                yield_chunk_size=int(args.bulk_actions),
            ):
                docs_read += len(batch_docs)
                progress.set_postfix(
                    read=docs_read,
                    indexed=docs_indexed,
                    inflight=len(futures),
                    submitted=docs_submitted,
                    bulk_reqs=bulk_submitted,
                )

                for chunk in _chunked(batch_docs, int(args.bulk_actions)):
                    payload = _bulk_payload(index_name, chunk)
                    if not payload.strip():
                        continue
                    future = pool.submit(
                        _index_bulk_with_retries,
                        base_url=elasticsearch_url,
                        payload=payload,
                        doc_count=len(chunk),
                        timeout_seconds=float(args.request_timeout_seconds),
                        max_retries=int(args.max_retries),
                        retry_backoff_seconds=float(args.retry_backoff_seconds),
                    )
                    futures.add(future)
                    docs_submitted += len(chunk)
                    bulk_submitted += 1
                    progress.update(len(chunk))
                    progress.set_postfix(
                        read=docs_read,
                        indexed=docs_indexed,
                        submitted=docs_submitted,
                        inflight=len(futures),
                        bulk_reqs=bulk_submitted,
                    )

                    if len(futures) >= max_inflight:
                        indexed, futures = _drain_futures(
                            futures,
                            return_when=FIRST_COMPLETED,
                        )
                        docs_indexed += indexed
                        progress.set_postfix(
                            read=docs_read,
                            indexed=docs_indexed,
                            submitted=docs_submitted,
                            inflight=len(futures),
                            bulk_reqs=bulk_submitted,
                        )

            indexed, futures = _drain_futures(futures, return_when=ALL_COMPLETED)
            docs_indexed += indexed
            progress.set_postfix(
                read=docs_read,
                indexed=docs_indexed,
                submitted=docs_submitted,
                inflight=len(futures),
                bulk_reqs=bulk_submitted,
            )

    if not args.skip_finalize_settings:
        print(
            "Restoring index runtime settings:",
            f"refresh_interval={args.final_refresh_interval}",
            f"number_of_replicas={args.final_replicas}",
        )
        _finalize_index(
            base_url=elasticsearch_url,
            index_name=index_name,
            final_refresh_interval=str(args.final_refresh_interval),
            final_replicas=int(args.final_replicas),
            timeout_seconds=float(args.request_timeout_seconds),
        )

    print(
        "Done:",
        f"docs_read={docs_read}",
        f"docs_indexed={docs_indexed}",
        f"bulk_requests={bulk_submitted}",
        f"index={index_name}",
        f"elasticsearch={elasticsearch_url}",
    )
    return 0


def main() -> int:
    args = parse_args()
    try:
        return _run(args)
    except (ValueError, ElasticsearchIndexingError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
