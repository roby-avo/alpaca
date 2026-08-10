from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Sequence

from .classify_postgres_entities import (
    CLASSIFIER_DISTRIBUTION,
    REQUIRED_CLASSIFIER_VERSION,
    _normalize_required_type_qid,
    _stage_name as _classification_stage_name,
)
from .common import resolve_configured_str, resolve_postgres_dsn, tqdm
from .index_postgres_to_elasticsearch import (
    DEFAULT_BULK_ACTIONS,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_RETRY_BACKOFF_SECONDS,
    DEFAULT_MAX_RETRIES,
    ElasticsearchIndexingError,
    _es_index_exists,
    _es_request_json,
    _finalize_index,
    _index_bulk_with_retries,
    _normalize_es_url,
    _wait_for_index_ready,
    default_elasticsearch_url,
    parse_non_negative_int,
    parse_positive_float,
    parse_positive_int,
)

try:  # pragma: no cover - runtime dependency
    import psycopg  # type: ignore
    from psycopg.types.json import Jsonb  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    psycopg = None  # type: ignore
    Jsonb = None  # type: ignore


DEFAULT_SOURCE_TABLE = "entities"
DEFAULT_NER_TABLE = "entity_ner"
DEFAULT_INDEX_NAME = "alpaca-wikidata"
DEFAULT_BATCH_SIZE = 20_000
DEFAULT_WORKERS = 4
DEFAULT_FINAL_REFRESH_INTERVAL = "1s"
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _quote_identifier(raw: str) -> str:
    value = raw.strip()
    if not _IDENTIFIER_RE.fullmatch(value):
        raise ValueError(
            f"Invalid SQL identifier '{raw}'. Use letters, digits, and underscores only."
        )
    return f'"{value}"'


def _update_stage_name(*, index_name: str, required_type_qid: str) -> str:
    return (
        f"es-ner:{index_name}:{REQUIRED_CLASSIFIER_VERSION}:"
        f"type:{required_type_qid}"
    )


def _load_state(conn: Any, *, stage: str) -> tuple[str, int, bool]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT last_qid, processed, metadata FROM alpaca_pipeline_state WHERE stage = %s",
            (stage,),
        )
        row = cur.fetchone()
    if not row:
        return "", 0, False
    metadata = row[2] if isinstance(row[2], dict) else {}
    return str(row[0] or ""), int(row[1] or 0), bool(metadata.get("complete"))


def _classification_total(
    conn: Any,
    *,
    source_table: str,
    required_type_qid: str,
) -> int:
    stage = _classification_stage_name(
        source_table=source_table,
        required_type_qid=required_type_qid,
    )
    _last_qid, processed, complete = _load_state(conn, stage=stage)
    if not complete:
        raise RuntimeError(
            f"Classification stage '{stage}' is not complete; refusing to publish a partial NER refresh."
        )
    return processed


def _fetch_batch(
    conn: Any,
    *,
    source_table: str,
    ner_table: str,
    required_type_qid: str,
    last_qid: str,
    batch_size: int,
) -> list[tuple[Any, ...]]:
    source_ident = _quote_identifier(source_table)
    ner_ident = _quote_identifier(ner_table)
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT
                e.qid,
                n.coarse_type,
                n.fine_type,
                n.subtype,
                n.specific_type,
                n.specific_types,
                n.secondary_types,
                n.facets,
                n.specificity_level,
                n.retrieval_path,
                n.retrieval_key,
                n.retrieval_tags,
                n.confidence,
                n.specific_type_confidence,
                n.classifier_name,
                n.classifier_version,
                n.taxonomy_version
            FROM {source_ident} AS e
            LEFT JOIN {ner_ident} AS n ON n.qid = e.qid
            WHERE e.qid > %s
              AND %s = ANY(e.types)
            ORDER BY e.qid
            LIMIT %s
            """,
            (last_qid, required_type_qid, int(batch_size)),
        )
        return list(cur.fetchall())


_NER_SOURCE_FIELDS = (
    "coarse_type",
    "fine_type",
    "ner_subtype",
    "ner_specific_type",
    "ner_specific_types",
    "ner_secondary_types",
    "ner_facets",
    "ner_specificity_level",
    "ner_retrieval_path",
    "ner_retrieval_key",
    "ner_retrieval_tags",
    "ner_confidence",
    "ner_specific_type_confidence",
    "ner_classifier_name",
    "ner_classifier_version",
    "ner_taxonomy_version",
)


def _row_to_update(row: Sequence[Any]) -> tuple[str, dict[str, Any] | None]:
    qid = str(row[0])
    classifier_version = row[15]
    if classifier_version is None:
        return qid, None
    if classifier_version != REQUIRED_CLASSIFIER_VERSION:
        raise RuntimeError(
            f"{qid} is not classified by {CLASSIFIER_DISTRIBUTION}=="
            f"{REQUIRED_CLASSIFIER_VERSION} (found {classifier_version!r})."
        )
    if not isinstance(row[1], str) or not isinstance(row[2], str):
        raise RuntimeError(f"{qid} has no successful NER classification.")
    return qid, {
        "coarse_type": row[1],
        "fine_type": row[2],
        "ner_subtype": row[3],
        "ner_specific_type": row[4],
        "ner_specific_types": list(row[5] or ()),
        "ner_secondary_types": list(row[6] or ()),
        "ner_facets": dict(row[7] or {}),
        "ner_specificity_level": int(row[8] or 0),
        "ner_retrieval_path": list(row[9] or ()),
        "ner_retrieval_key": row[10],
        "ner_retrieval_tags": list(row[11] or ()),
        "ner_confidence": float(row[12]),
        "ner_specific_type_confidence": (
            float(row[13]) if row[13] is not None else None
        ),
        "ner_classifier_name": row[14],
        "ner_classifier_version": classifier_version,
        "ner_taxonomy_version": row[16],
    }


def _bulk_update_payload(
    index_name: str,
    updates: Sequence[tuple[str, dict[str, Any] | None]],
) -> bytes:
    lines: list[str] = []
    for qid, doc in updates:
        lines.append(
            json.dumps(
                {
                    "update": {
                        "_index": index_name,
                        "_id": qid,
                        "retry_on_conflict": 3,
                    }
                },
                separators=(",", ":"),
            )
        )
        body: dict[str, Any]
        if doc is None:
            body = {
                "script": {
                    "lang": "painless",
                    "source": "for (field in params.fields) { ctx._source.remove(field); }",
                    "params": {"fields": list(_NER_SOURCE_FIELDS)},
                }
            }
        else:
            body = {"doc": doc}
        lines.append(json.dumps(body, ensure_ascii=False, separators=(",", ":")))
    return ("\n".join(lines) + "\n").encode("utf-8")


def _chunks(values: Sequence[Any], size: int) -> list[Sequence[Any]]:
    return [values[start : start + size] for start in range(0, len(values), size)]


def _write_state(
    conn: Any,
    *,
    stage: str,
    last_qid: str,
    processed: int,
    complete: bool,
    required_type_qid: str,
    index_name: str,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO alpaca_pipeline_state (stage, last_qid, processed, metadata)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (stage) DO UPDATE SET
                last_qid = EXCLUDED.last_qid,
                processed = EXCLUDED.processed,
                metadata = EXCLUDED.metadata,
                updated_at = NOW()
            """,
            (
                stage,
                last_qid,
                int(processed),
                Jsonb(
                    {
                        "classifier": CLASSIFIER_DISTRIBUTION,
                        "classifier_version": REQUIRED_CLASSIFIER_VERSION,
                        "required_type_qid": required_type_qid,
                        "index": index_name,
                        "complete": complete,
                    }
                ),
            ),
        )
    conn.commit()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Bulk-update only NER fields in an existing Elasticsearch index from "
            "a completed targeted PostgreSQL classification stage."
        )
    )
    parser.add_argument("--postgres-dsn")
    parser.add_argument("--elasticsearch-url")
    parser.add_argument("--source-table", default=DEFAULT_SOURCE_TABLE)
    parser.add_argument("--ner-table", default=DEFAULT_NER_TABLE)
    parser.add_argument("--index-name", default=DEFAULT_INDEX_NAME)
    parser.add_argument("--required-type-qid", default="Q5")
    parser.add_argument("--batch-size", type=parse_positive_int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--bulk-actions", type=parse_positive_int, default=DEFAULT_BULK_ACTIONS)
    parser.add_argument("--workers", type=parse_positive_int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--request-timeout-seconds",
        type=parse_positive_float,
        default=DEFAULT_REQUEST_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--max-retries",
        type=parse_non_negative_int,
        default=DEFAULT_MAX_RETRIES,
    )
    parser.add_argument(
        "--retry-backoff-seconds",
        type=parse_positive_float,
        default=DEFAULT_RETRY_BACKOFF_SECONDS,
    )
    parser.add_argument("--reset", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    if psycopg is None or Jsonb is None:
        raise RuntimeError("psycopg is required; install the repository requirements.")
    source_table = args.source_table.strip()
    ner_table = args.ner_table.strip()
    _quote_identifier(source_table)
    _quote_identifier(ner_table)
    index_name = args.index_name.strip()
    if not index_name:
        raise ValueError("Index name must be non-empty.")
    required_type_qid = _normalize_required_type_qid(args.required_type_qid)
    if not required_type_qid:
        raise ValueError("--required-type-qid must be provided.")
    postgres_dsn = resolve_postgres_dsn(args.postgres_dsn)
    elasticsearch_url = _normalize_es_url(
        resolve_configured_str(
            args.elasticsearch_url,
            "ALPACA_ELASTICSEARCH_URL",
            default_elasticsearch_url(),
        )
    )
    stage = _update_stage_name(
        index_name=index_name,
        required_type_qid=required_type_qid,
    )

    if not _es_index_exists(
        base_url=elasticsearch_url,
        index_name=index_name,
        timeout_seconds=float(args.request_timeout_seconds),
    ):
        raise ElasticsearchIndexingError(f"Elasticsearch index '{index_name}' does not exist.")

    with psycopg.connect(postgres_dsn) as conn:
        total = _classification_total(
            conn,
            source_table=source_table,
            required_type_qid=required_type_qid,
        )
        if args.reset:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM alpaca_pipeline_state WHERE stage = %s", (stage,))
            conn.commit()
        last_qid, processed, complete = _load_state(conn, stage=stage)
        if complete:
            print(
                "Elasticsearch NER refresh already complete:",
                f"stage={stage}",
                f"processed={processed}",
            )
            return 0

        _es_request_json(
            base_url=elasticsearch_url,
            method="PUT",
            path=f"/{index_name}/_settings",
            body={"index": {"refresh_interval": "-1"}},
            timeout_seconds=float(args.request_timeout_seconds),
            expected_statuses=(200,),
        )
        _wait_for_index_ready(
            base_url=elasticsearch_url,
            index_name=index_name,
            wait_timeout_seconds=180.0,
            request_timeout_seconds=float(args.request_timeout_seconds),
        )

        print(
            "Updating Elasticsearch NER fields:",
            f"index={index_name}",
            f"classifier={CLASSIFIER_DISTRIBUTION}=={REQUIRED_CLASSIFIER_VERSION}",
            f"required_type_qid={required_type_qid}",
            f"resume_last_qid={last_qid or 'n/a'}",
            f"resume_processed={processed}",
            f"total={total}",
        )
        try:
            with ThreadPoolExecutor(max_workers=int(args.workers)) as pool:
                with tqdm(
                    total=total,
                    initial=min(processed, total),
                    desc="ner->es",
                    unit="doc",
                ) as progress:
                    while True:
                        rows = _fetch_batch(
                            conn,
                            source_table=source_table,
                            ner_table=ner_table,
                            required_type_qid=required_type_qid,
                            last_qid=last_qid,
                            batch_size=int(args.batch_size),
                        )
                        if not rows:
                            _write_state(
                                conn,
                                stage=stage,
                                last_qid=last_qid,
                                processed=processed,
                                complete=True,
                                required_type_qid=required_type_qid,
                                index_name=index_name,
                            )
                            break
                        updates = [_row_to_update(row) for row in rows]
                        futures = [
                            pool.submit(
                                _index_bulk_with_retries,
                                base_url=elasticsearch_url,
                                payload=_bulk_update_payload(index_name, chunk),
                                doc_count=len(chunk),
                                timeout_seconds=float(args.request_timeout_seconds),
                                max_retries=int(args.max_retries),
                                retry_backoff_seconds=float(args.retry_backoff_seconds),
                            )
                            for chunk in _chunks(updates, int(args.bulk_actions))
                        ]
                        updated = sum(int(future.result()) for future in futures)
                        if updated != len(updates):
                            raise ElasticsearchIndexingError(
                                f"Updated {updated} of {len(updates)} documents in the batch."
                            )
                        last_qid = str(rows[-1][0])
                        processed += len(rows)
                        _write_state(
                            conn,
                            stage=stage,
                            last_qid=last_qid,
                            processed=processed,
                            complete=False,
                            required_type_qid=required_type_qid,
                            index_name=index_name,
                        )
                        progress.update(len(rows))
                        progress.set_postfix(last_qid=last_qid, updated=updated)
        finally:
            _finalize_index(
                base_url=elasticsearch_url,
                index_name=index_name,
                final_refresh_interval=DEFAULT_FINAL_REFRESH_INTERVAL,
                final_replicas=0,
                timeout_seconds=float(args.request_timeout_seconds),
            )

    print(
        "Completed Elasticsearch NER refresh:",
        f"processed={processed}",
        f"last_qid={last_qid or 'n/a'}",
        f"index={index_name}",
    )
    return 0


def main() -> int:
    try:
        return run(parse_args())
    except (RuntimeError, ValueError, ElasticsearchIndexingError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
