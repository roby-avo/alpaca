from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .common import ensure_parent_dir, open_text_for_write, resolve_postgres_dsn, tqdm
from .postgres_store import PostgresStore
from .wikidata_sample_ids import resolve_qids


def parse_positive_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a compact .json/.json.gz/.json.bz2 Wikidata-style dump from Postgres sample_entity_cache by explicit QIDs."
        )
    )
    parser.add_argument("--postgres-dsn", help="Postgres DSN (defaults to ALPACA_POSTGRES_DSN).")
    parser.add_argument("--output-path", required=True, help="Output dump path (.json/.json.gz/.json.bz2).")
    parser.add_argument("--ids", help="Comma-separated QIDs (example: Q42,Q90,Q64).")
    parser.add_argument("--ids-file", help="Text file with one QID per line (comments with # allowed).")
    parser.add_argument(
        "--count",
        type=parse_positive_int,
        help="Convenience mode: exports the first N cached sample entities by numeric QID order.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Write available entities and continue if some requested QIDs are missing in cache.",
    )
    return parser.parse_args()


def _write_entity(handle: Any, entity: dict[str, Any], *, is_first: bool) -> None:
    prefix = "" if is_first else ","
    encoded = json.dumps(entity, ensure_ascii=False, separators=(",", ":"))
    handle.write(f"{prefix}\n{encoded}")


def main() -> int:
    args = parse_args()
    try:
        selector_count = sum(
            1 for value in (args.ids, args.ids_file) if value
        ) + (1 if args.count is not None else 0)
        if selector_count != 1:
            raise ValueError("Provide exactly one of --ids, --ids-file, or --count.")

        output_path = Path(args.output_path).expanduser()
        ensure_parent_dir(output_path)

        store = PostgresStore(resolve_postgres_dsn(args.postgres_dsn))
        store.ensure_schema()
        if args.count is not None:
            qids = store.list_sample_entity_ids(limit=int(args.count))
            if len(qids) < int(args.count):
                raise ValueError(
                    f"Requested --count {args.count}, but Postgres sample cache has only {len(qids)} entities. "
                    "Fetch more first with src.wikidata_sample_postgres --count ..."
                )
        else:
            qids = resolve_qids(args.ids, args.ids_file, None)
        cached = store.get_sample_entities(qids)

        missing = [qid for qid in qids if qid not in cached]
        if missing and not args.allow_missing:
            raise ValueError(
                f"Missing {len(missing)} requested QIDs in Postgres sample cache. "
                "Fetch them first with src.wikidata_sample_postgres or use --allow-missing."
            )

        written = 0
        with open_text_for_write(output_path) as handle:
            handle.write("[")
            with tqdm(total=len(qids), desc="sample-dump-pg", unit="entity") as progress:
                for qid in qids:
                    payload = cached.get(qid)
                    progress.update(1)
                    if payload is None:
                        progress.set_postfix(written=written, missing=len(missing))
                        continue
                    _write_entity(handle, payload, is_first=(written == 0))
                    written += 1
                    progress.set_postfix(written=written, missing=len(missing))
            handle.write("\n]\n")

        if written == 0:
            raise ValueError("No entities were written to output dump.")

        print(
            "Built sample dump from Postgres cache:",
            f"requested={len(qids)}",
            f"written={written}",
            f"missing={len(missing)}",
            f"output={output_path}",
        )
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
