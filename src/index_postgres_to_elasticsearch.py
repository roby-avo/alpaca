from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ALL_COMPLETED, FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .common import resolve_configured_str, resolve_postgres_dsn, tqdm

try:  # pragma: no cover - import depends on runtime environment
    import psycopg  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    psycopg = None  # type: ignore


ELASTICSEARCH_URL_ENV = "ALPACA_ELASTICSEARCH_URL"
DEFAULT_ELASTICSEARCH_URL = "http://localhost:9200"
DEFAULT_INDEX_NAME = "alpaca-entities"
DEFAULT_TABLE_NAME = "entities"
DEFAULT_FETCH_SIZE = 10_000
DEFAULT_BULK_ACTIONS = 2_000
DEFAULT_WORKERS = 4
DEFAULT_MAX_RETRIES = 4
DEFAULT_RETRY_BACKOFF_SECONDS = 1.0
DEFAULT_REQUEST_TIMEOUT_SECONDS = 90.0
DEFAULT_FINAL_REFRESH_INTERVAL = "1s"
DEFAULT_FINAL_REPLICAS = 0

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class ElasticsearchIndexingError(RuntimeError):
    pass


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
            f"{DEFAULT_ELASTICSEARCH_URL})."
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
            "Default: workers * 3."
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
            "analysis": {
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
                    }
                },
            },
        },
        "mappings": {
            "dynamic": "strict",
            "properties": {
                "qid": {"type": "keyword"},
                "label": {
                    "type": "text",
                    "analyzer": "alpaca_text",
                    "copy_to": ["search_text"],
                    "fields": {
                        "keyword": {
                            "type": "keyword",
                            "normalizer": "alpaca_keyword_lower",
                        }
                    },
                },
                "aliases": {
                    "type": "text",
                    "analyzer": "alpaca_text",
                    "copy_to": ["search_text"],
                },
                "context_string": {
                    "type": "text",
                    "analyzer": "alpaca_text",
                    "copy_to": ["search_text"],
                },
                "search_text": {
                    "type": "text",
                    "analyzer": "alpaca_text",
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
                "popularity": {"type": "float"},
                "prior": {"type": "float"},
                # Compact refs from Postgres, without full URL prefixes.
                "wikipedia_url": {
                    "type": "keyword",
                    "ignore_above": 2048,
                },
                "dbpedia_url": {
                    "type": "keyword",
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


def _clean_aliases(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for value in raw:
        if not isinstance(value, str):
            continue
        cleaned = value.strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        out.append(cleaned)
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


def _row_to_document(row: Sequence[Any]) -> dict[str, Any] | None:
    if len(row) < 12:
        return None
    qid = row[0]
    label = row[1]
    if not isinstance(qid, str) or not isinstance(label, str):
        return None

    doc: dict[str, Any] = {
        "qid": qid,
        "label": label,
        "context_string": row[2] if isinstance(row[2], str) else "",
        "aliases": _clean_aliases(row[3]),
        "coarse_type": row[4] if isinstance(row[4], str) else "",
        "fine_type": row[5] if isinstance(row[5], str) else "",
        "item_category": row[6] if isinstance(row[6], str) else "",
        "popularity": _as_float(row[7]),
        "prior": _as_float(row[8]),
        # Keep compact refs from Postgres to save space.
        "wikipedia_url": row[9] if isinstance(row[9], str) else "",
        "dbpedia_url": row[10] if isinstance(row[10], str) else "",
    }
    updated_at = _as_iso_datetime(row[11])
    if updated_at:
        doc["updated_at"] = updated_at
    return doc


def _iter_documents_from_postgres(
    *,
    postgres_dsn: str,
    table_name: str,
    batch_size: int,
    updated_since: str | None,
) -> Iterator[list[dict[str, Any]]]:
    pg = _require_psycopg()
    table_sql = _quote_table_name(table_name)
    where_sql = ""
    params: list[Any] = []
    if updated_since:
        where_sql = "WHERE updated_at >= %s::timestamptz"
        params.append(updated_since)

    sql = f"""
    SELECT
        qid, label, context_string, aliases,
        coarse_type, fine_type, item_category,
        popularity, prior, wikipedia_url, dbpedia_url, updated_at
    FROM {table_sql}
    {where_sql}
    ORDER BY qid
    """
    with pg.connect(postgres_dsn) as conn:
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
                    doc = _row_to_document(row)
                    if doc:
                        docs.append(doc)
                if docs:
                    yield docs


def _count_source_rows(
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
            DEFAULT_ELASTICSEARCH_URL,
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
        print("Counting source rows for progress bar...")
        total_rows = _count_source_rows(
            postgres_dsn=postgres_dsn,
            table_name=table_name,
            updated_since=args.updated_since,
        )
        print(f"Source rows to index: {total_rows}")

    max_inflight = int(args.max_inflight) if int(args.max_inflight) > 0 else int(args.workers) * 3
    docs_read = 0
    docs_indexed = 0
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
    )

    with ThreadPoolExecutor(max_workers=int(args.workers)) as pool:
        with tqdm(total=total_rows, desc="pg->es", unit="doc") as progress:
            for batch_docs in _iter_documents_from_postgres(
                postgres_dsn=postgres_dsn,
                table_name=table_name,
                batch_size=int(args.batch_size),
                updated_since=args.updated_since,
            ):
                docs_read += len(batch_docs)
                progress.update(len(batch_docs))
                progress.set_postfix(
                    indexed=docs_indexed,
                    inflight=len(futures),
                    submitted=bulk_submitted,
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
                    bulk_submitted += 1

                    if len(futures) >= max_inflight:
                        indexed, futures = _drain_futures(
                            futures,
                            return_when=FIRST_COMPLETED,
                        )
                        docs_indexed += indexed
                        progress.set_postfix(
                            indexed=docs_indexed,
                            inflight=len(futures),
                            submitted=bulk_submitted,
                        )

            indexed, futures = _drain_futures(futures, return_when=ALL_COMPLETED)
            docs_indexed += indexed
            progress.set_postfix(
                indexed=docs_indexed,
                inflight=len(futures),
                submitted=bulk_submitted,
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
