from __future__ import annotations

import argparse
import sys
import time

from .common import resolve_postgres_dsn
from .postgres_store import DEFAULT_INDEX_PROFILE, PostgresStore, PostgresStoreError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run explicit PostgreSQL schema/index maintenance for Alpaca. "
            "This is intentionally separate from API startup."
        )
    )
    parser.add_argument("--postgres-dsn", help="Postgres DSN (defaults to ALPACA_POSTGRES_DSN).")
    parser.add_argument(
        "--skip-schema",
        action="store_true",
        help="Skip schema creation/migration.",
    )
    parser.add_argument(
        "--ensure-search-indexes",
        action="store_true",
        help="Create/refresh search support indexes. This can be long-running on large tables.",
    )
    parser.add_argument(
        "--table",
        default="entities",
        help="Table to index when --ensure-search-indexes is set (default: entities).",
    )
    parser.add_argument(
        "--index-profile",
        default=DEFAULT_INDEX_PROFILE,
        help=f"Index profile for --ensure-search-indexes (default: {DEFAULT_INDEX_PROFILE}).",
    )
    return parser.parse_args()


def _run(args: argparse.Namespace) -> int:
    store = PostgresStore(resolve_postgres_dsn(args.postgres_dsn))
    started = time.monotonic()

    if not args.skip_schema:
        print("Ensuring PostgreSQL schema...")
        store.ensure_schema()
        print("Schema ready.")

    if args.ensure_search_indexes:
        print(
            "Ensuring PostgreSQL search indexes "
            f"for table={args.table!r} profile={args.index_profile!r}..."
        )
        store.ensure_search_indexes(args.table, index_profile=args.index_profile)
        print("Search indexes ready.")

    elapsed = time.monotonic() - started
    print(f"PostgreSQL maintenance completed in {elapsed:.1f}s.")
    return 0


def main() -> int:
    args = parse_args()
    try:
        return _run(args)
    except (PostgresStoreError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
