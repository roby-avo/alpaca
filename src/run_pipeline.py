from __future__ import annotations

import argparse
import os
import sys

from .build_elasticsearch_index import run as run_elasticsearch_index
from .build_postgres_entities import run_pass1 as run_postgres_pass1
from .build_postgres_entities import run_pass2 as run_postgres_pass2
from .common import (
    parse_language_allowlist,
    resolve_dump_path,
    resolve_elasticsearch_index,
    resolve_elasticsearch_url,
    resolve_postgres_dsn,
)
from .elasticsearch_client import ElasticsearchClientError
from .postgres_store import PostgresStoreError


def parse_non_negative_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if value < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return value


def parse_positive_int(raw: str) -> int:
    value = parse_non_negative_int(raw)
    if value == 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return value


def parse_non_negative_float(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if value < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the deterministic alpaca pipeline: dump -> Postgres entities (pass1) -> "
            "Postgres deterministic context build (pass2) -> Elasticsearch indexing."
        )
    )
    parser.add_argument("--dump-path", help="Input dump path (.json/.json.gz/.json.bz2).")
    parser.add_argument("--postgres-dsn", help="Postgres DSN.")
    parser.add_argument("--elasticsearch-url", help="Elasticsearch base URL.")
    parser.add_argument("--elasticsearch-index", help="Elasticsearch index name.")

    parser.add_argument(
        "--limit",
        type=parse_non_negative_int,
        default=0,
        help="Entity parse limit for smoke runs (0 = no limit).",
    )
    parser.add_argument(
        "--pass1-batch-size",
        type=parse_positive_int,
        default=5000,
        help="Rows per Postgres upsert batch in pass1 (default: 5000).",
    )
    parser.add_argument(
        "--context-batch-size",
        type=parse_positive_int,
        default=1000,
        help="Entities per Postgres context batch in pass2 (default: 1000).",
    )
    parser.add_argument(
        "--es-fetch-batch-size",
        type=parse_positive_int,
        default=2000,
        help="Rows fetched from Postgres per ES indexing read batch (default: 2000).",
    )
    parser.add_argument(
        "--es-chunk-bytes",
        type=parse_positive_int,
        default=8_000_000,
        help="Max bulk NDJSON payload bytes per Elasticsearch request.",
    )
    parser.add_argument(
        "--workers",
        type=parse_positive_int,
        default=max(1, min(8, (os.cpu_count() or 1))),
        help="Parallel worker count for pass1 transform, pass2 context build, and ES indexing.",
    )
    parser.add_argument(
        "--replace-elasticsearch-index",
        action="store_true",
        help="Delete and recreate the Elasticsearch index before indexing.",
    )
    parser.add_argument(
        "--wait-timeout-seconds",
        type=parse_non_negative_float,
        default=120.0,
        help="How long to wait for Elasticsearch startup.",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=parse_non_negative_float,
        default=2.0,
        help="Polling interval while waiting for Elasticsearch.",
    )
    parser.add_argument(
        "--http-timeout-seconds",
        type=parse_positive_int,
        default=120,
        help="HTTP timeout per Elasticsearch request.",
    )

    parser.add_argument("--skip-pass1", action="store_true", help="Skip Postgres pass1 ingestion.")
    parser.add_argument(
        "--skip-pass2",
        action="store_true",
        help="Skip Postgres pass2 deterministic context build.",
    )
    parser.add_argument(
        "--skip-elasticsearch",
        action="store_true",
        help="Skip Elasticsearch index/ingest step.",
    )
    parser.add_argument(
        "--disable-ner-classifier",
        action="store_true",
        help="Disable lexical NER typing during Postgres pass1.",
    )
    parser.add_argument(
        "--languages",
        default="en",
        help="Comma-separated language allowlist for labels/descriptions (default: en).",
    )
    parser.add_argument(
        "--max-aliases-per-language",
        type=parse_non_negative_int,
        default=8,
        help="Max aliases stored per language (default: 8, 0 disables aliases).",
    )
    parser.add_argument(
        "--max-context-object-ids",
        type=parse_non_negative_int,
        default=32,
        help="Max claim object IDs retained per entity (default: 32).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        dump_path = resolve_dump_path(args.dump_path)
        postgres_dsn = resolve_postgres_dsn(args.postgres_dsn)
        elasticsearch_url = resolve_elasticsearch_url(args.elasticsearch_url)
        elasticsearch_index = resolve_elasticsearch_index(args.elasticsearch_index)
        language_allowlist = parse_language_allowlist(args.languages, arg_name="--languages")

        if not args.skip_pass1:
            pass1_status = run_postgres_pass1(
                dump_path=dump_path,
                postgres_dsn=postgres_dsn,
                batch_size=args.pass1_batch_size,
                limit=args.limit,
                language_allowlist=language_allowlist,
                max_aliases_per_language=args.max_aliases_per_language,
                max_context_object_ids=args.max_context_object_ids,
                disable_ner_classifier=args.disable_ner_classifier,
                worker_count=args.workers,
            )
            if pass1_status != 0:
                return pass1_status

        if not args.skip_pass2:
            pass2_status = run_postgres_pass2(
                postgres_dsn=postgres_dsn,
                batch_size=args.context_batch_size,
                worker_count=args.workers,
            )
            if pass2_status != 0:
                return pass2_status

        if not args.skip_elasticsearch:
            es_status = run_elasticsearch_index(
                postgres_dsn=postgres_dsn,
                elasticsearch_url=elasticsearch_url,
                index_name=elasticsearch_index,
                replace_index=args.replace_elasticsearch_index,
                fetch_batch_size=args.es_fetch_batch_size,
                chunk_bytes=args.es_chunk_bytes,
                worker_count=args.workers,
                wait_timeout_seconds=args.wait_timeout_seconds,
                poll_interval_seconds=args.poll_interval_seconds,
                http_timeout_seconds=float(args.http_timeout_seconds),
            )
            if es_status != 0:
                return es_status

        print("Pipeline completed successfully.")
        return 0
    except (FileNotFoundError, ValueError, ElasticsearchClientError, PostgresStoreError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
