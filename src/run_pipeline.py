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
            "Postgres deterministic context build (pass2) -> Postgres search indexes."
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
        "--workers",
        type=parse_positive_int,
        default=max(1, min(8, (os.cpu_count() or 1))),
        help="Parallel worker count for pass1 transform and pass2 context build.",
    )

    parser.add_argument("--skip-pass1", action="store_true", help="Skip Postgres pass1 ingestion.")
    parser.add_argument(
        "--skip-pass2",
        action="store_true",
        help="Skip Postgres pass2 deterministic context build.",
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
                # Standard pipeline runs pass2, which rebuilds search_vector after context enrichment.
                build_search_vector=bool(args.skip_pass2),
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
        # Finalize the default lean lookup layout (keeps entity_context_inputs for traceability).
        store.compact_table_for_lookup(
            "entities",
            drop_cross_refs_trgm_index=True,
            drop_context_inputs_table=False,
            vacuum_full=False,
            analyze=True,
        )

        print("Pipeline completed successfully (Postgres-only search backend).")
        return 0
    except (FileNotFoundError, ValueError, PostgresStoreError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
