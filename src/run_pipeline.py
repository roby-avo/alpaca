from __future__ import annotations

import argparse
import os
import sys

from .build_postgres_entities import run_pass1 as run_postgres_pass1
from .build_postgres_entities import run_pass2 as run_postgres_pass2
from .common import (
    parse_language_allowlist,
    resolve_dump_path,
    resolve_postgres_dsn,
)
from .postgres_store import PostgresStore, PostgresStoreError


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the deterministic alpaca pipeline: dump -> Postgres entities (pass1) -> "
            "Postgres triples ingest -> Postgres support indexes (pass2 kept as a compatibility no-op)."
        )
    )
    parser.add_argument("--dump-path", help="Input dump path (.json/.json.gz/.json.bz2).")
    parser.add_argument("--postgres-dsn", help="Postgres DSN.")

    parser.add_argument(
        "--limit",
        type=parse_non_negative_int,
        default=0,
        help="Entity parse limit for smoke runs (0 = no limit).",
    )
    parser.add_argument(
        "--expected-entity-total",
        type=parse_non_negative_int,
        default=0,
        help=(
            "Optional manual progress total override for pass1 (e.g., from Wikidata:Statistics). "
            "0 = auto sample from the local dump."
        ),
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
        help="Deprecated compatibility option; pass2 no longer materializes context strings.",
    )
    parser.add_argument(
        "--workers",
        type=parse_positive_int,
        default=max(1, min(8, (os.cpu_count() or 1))),
        help="Parallel worker count for pass1 transform and any compatibility pass2 work.",
    )

    parser.add_argument("--skip-pass1", action="store_true", help="Skip Postgres pass1 ingestion.")
    parser.add_argument(
        "--skip-pass2",
        action="store_true",
        help="Skip the compatibility pass2 no-op.",
    )
    parser.add_argument(
        "--disable-ner-classifier",
        action="store_true",
        help="Disable lexical NER typing during Postgres pass1.",
    )
    parser.add_argument(
        "--languages",
        default="en",
        help="Comma-separated language allowlist used for lexical typing inputs (default: en).",
    )
    parser.add_argument(
        "--max-aliases-per-language",
        type=parse_non_negative_int,
        default=8,
        help="Max aliases per language used for lexical typing (default: 8).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        dump_path = resolve_dump_path(args.dump_path)
        postgres_dsn = resolve_postgres_dsn(args.postgres_dsn)
        language_allowlist = parse_language_allowlist(args.languages, arg_name="--languages")

        if not args.skip_pass1:
            pass1_status = run_postgres_pass1(
                dump_path=dump_path,
                postgres_dsn=postgres_dsn,
                batch_size=args.pass1_batch_size,
                limit=args.limit,
                language_allowlist=language_allowlist,
                max_aliases_per_language=args.max_aliases_per_language,
                disable_ner_classifier=args.disable_ner_classifier,
                worker_count=args.workers,
                expected_entity_total=(args.expected_entity_total or None),
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

        store = PostgresStore(postgres_dsn)
        store.ensure_schema()
        store.ensure_search_indexes()
        # Finalize the default lean lookup layout while keeping the single-table entities layout intact.
        store.compact_table_for_lookup(
            "entities",
            drop_cross_refs_trgm_index=True,
            drop_context_inputs_table=False,
            vacuum_full=False,
            analyze=True,
        )

        print("Pipeline completed successfully (Postgres entities + triples ready for Elasticsearch export).")
        return 0
    except (FileNotFoundError, ValueError, PostgresStoreError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
