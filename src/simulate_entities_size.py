from __future__ import annotations

import argparse
import sys

from .common import resolve_postgres_dsn, tqdm
from .postgres_store import PostgresStore, PostgresStoreError


DEFAULT_PROJECT_ROWS = 100_000_000


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


def _format_bytes(num_bytes: int) -> str:
    value = float(max(0, int(num_bytes)))
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if value < 1024.0 or unit == "PB":
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{int(num_bytes)} B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a Postgres size-estimation simulation table by replicating already-built "
            "rows from entities. This preserves realistic row shape (including context_string and "
            "search columns) while avoiding a full Wikidata ingest."
        )
    )
    parser.add_argument("--postgres-dsn", help="Postgres DSN (defaults to ALPACA_POSTGRES_DSN).")
    parser.add_argument(
        "--dest-table",
        default="entities_size_sim",
        help="Destination simulation table name (default: entities_size_sim).",
    )
    parser.add_argument(
        "--target-rows",
        type=parse_positive_int,
        required=True,
        help="Number of rows to generate in the simulation table.",
    )
    parser.add_argument(
        "--seed-rows",
        type=parse_non_negative_int,
        default=0,
        help=(
            "How many rows to use from entities as the seed sample before replication "
            "(0 = use all current rows in entities)."
        ),
    )
    parser.add_argument(
        "--batch-rows",
        type=parse_positive_int,
        default=100_000,
        help="Rows inserted per replication batch (default: 100000).",
    )
    parser.add_argument(
        "--project-rows",
        type=parse_positive_int,
        default=DEFAULT_PROJECT_ROWS,
        help=f"Projected row count for linear size extrapolation (default: {DEFAULT_PROJECT_ROWS}).",
    )
    parser.add_argument(
        "--skip-indexes",
        action="store_true",
        help="Do not build lookup indexes on the simulation table (faster, less realistic).",
    )
    parser.add_argument(
        "--skip-analyze",
        action="store_true",
        help="Skip ANALYZE on the simulation table after load.",
    )
    parser.add_argument(
        "--keep-dest",
        action="store_true",
        help="Reuse an existing destination table definition instead of dropping/recreating it (it will still be truncated).",
    )
    parser.add_argument(
        "--fast-load",
        action="store_true",
        help=(
            "Estimation-only speed mode: creates the destination as UNLOGGED and disables "
            "synchronous_commit for replication writes."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.dest_table == "entities":
            raise ValueError("Refusing to use --dest-table entities. Pick a separate simulation table.")

        store = PostgresStore(resolve_postgres_dsn(args.postgres_dsn))
        store.ensure_schema()

        if not args.keep_dest:
            print(f"Preparing destination table {args.dest_table} (drop/recreate from entities schema)...")
            store.recreate_entities_like_table(
                args.dest_table,
                drop_existing=True,
                unlogged=bool(args.fast_load),
            )
        else:
            print(f"Reusing destination table {args.dest_table} and truncating it...")
            store.truncate_table(args.dest_table)

        print(
            "Replicating entities rows for size simulation:",
            f"target_rows={args.target_rows}",
            f"seed_rows={args.seed_rows or 'all'}",
            f"batch_rows={args.batch_rows}",
        )
        inserted_progress = 0
        with tqdm(total=int(args.target_rows), desc="sim-replicate", unit="row") as progress:
            def _on_chunk(stats: dict[str, int]) -> None:
                nonlocal inserted_progress
                total_inserted = int(stats.get("rows_inserted_total", inserted_progress))
                delta = max(0, total_inserted - inserted_progress)
                if delta:
                    progress.update(delta)
                    inserted_progress = total_inserted
                progress.set_postfix(
                    chunk=int(stats.get("chunk_index", 0)),
                    chunk_rows=int(stats.get("chunk_rows", 0)),
                    inserted=total_inserted,
                    remaining=int(stats.get("rows_remaining", 0)),
                )

            replication = store.replicate_entities_for_size_estimation(
                dest_table=args.dest_table,
                target_rows=int(args.target_rows),
                seed_rows=int(args.seed_rows),
                batch_rows=int(args.batch_rows),
                on_chunk=_on_chunk,
                disable_synchronous_commit=bool(args.fast_load),
            )
            if inserted_progress < int(args.target_rows):
                progress.update(max(0, int(args.target_rows) - inserted_progress))
                inserted_progress = int(args.target_rows)

        if not args.skip_indexes:
            print(f"Building lookup indexes on {args.dest_table} (GIN/trgm/exact/type)...")
            store.ensure_search_indexes(args.dest_table)
        else:
            print("Skipping index creation (--skip-indexes).")

        if not args.skip_analyze:
            print(f"Running ANALYZE on {args.dest_table}...")
            store.analyze_table(args.dest_table)

        stats = store.table_storage_stats(args.dest_table)
        rows = max(1, int(stats["rows"]))
        project_rows = int(args.project_rows)
        projected_table = int((stats["table_bytes"] / rows) * project_rows)
        projected_index = int((stats["index_bytes"] / rows) * project_rows)
        projected_total = int((stats["total_bytes"] / rows) * project_rows)

        print("Simulation complete:")
        print(
            f"  dest_table={args.dest_table} rows={stats['rows']} "
            f"(seed_rows_used={replication['seed_rows_used']} stride={replication['qid_stride']} chunks={replication.get('chunks', 0)})"
        )
        if args.fast_load:
            print("  mode=fast_load (UNLOGGED table + synchronous_commit=off during replication)")
        print(
            "  current_size:",
            f"table={_format_bytes(stats['table_bytes'])}",
            f"indexes={_format_bytes(stats['index_bytes'])}",
            f"total={_format_bytes(stats['total_bytes'])}",
        )
        print(
            f"  projected_{project_rows}_rows:",
            f"table={_format_bytes(projected_table)}",
            f"indexes={_format_bytes(projected_index)}",
            f"total={_format_bytes(projected_total)}",
        )
        print(
            "  note=This is a demonstrative estimate based on replicated row shapes. "
            "Index sizes can be underestimated if the seed sample is too small or text diversity is low."
        )
        return 0
    except (ValueError, PostgresStoreError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
