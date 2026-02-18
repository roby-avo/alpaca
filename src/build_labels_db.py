from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections.abc import Sequence
from pathlib import Path

from .common import (
    DUMP_PATH_ENV,
    LABELS_DB_PATH_ENV,
    ensure_existing_file,
    ensure_parent_dir,
    estimate_wikidata_entity_total,
    finalize_tqdm_total,
    extract_multilingual_payload,
    is_supported_entity_id,
    iter_wikidata_entities,
    keep_tqdm_total_ahead,
    parse_language_allowlist,
    resolve_dump_path,
    resolve_labels_db_path,
    select_alias_map_languages,
    select_text_map_languages,
    tqdm,
)
from .ner_typing import infer_ner_types

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS labels (
    id TEXT PRIMARY KEY,
    labels_json TEXT NOT NULL
)
"""

UPSERT_SQL = """
INSERT INTO labels (id, labels_json)
VALUES (?, ?)
ON CONFLICT(id) DO UPDATE SET labels_json = excluded.labels_json
"""

PRAGMA_STATEMENTS = (
    "PRAGMA journal_mode=WAL",
    "PRAGMA synchronous=NORMAL",
    "PRAGMA temp_store=MEMORY",
    "PRAGMA foreign_keys=OFF",
    "PRAGMA cache_size=-200000",
)


def parse_non_negative_int(raw: str) -> int:
    try:
        parsed = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc

    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")

    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stream a Wikidata dump and build a SQLite labels store "
            "(Q* and P* entities, compact language subset)."
        )
    )
    parser.add_argument(
        "--dump-path",
        help=(
            f"Input dump path (.json/.json.gz/.json.bz2). "
            f"Overrides ${DUMP_PATH_ENV}."
        ),
    )
    parser.add_argument(
        "--db-path",
        help=(
            f"Output SQLite file path for labels DB. "
            f"Overrides ${LABELS_DB_PATH_ENV}."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=parse_non_negative_int,
        default=5000,
        help="Rows per SQLite batch insert (default: 5000).",
    )
    parser.add_argument(
        "--limit",
        type=parse_non_negative_int,
        default=0,
        help="Max entities to parse for smoke runs (0 = no limit).",
    )
    parser.add_argument(
        "--disable-ner-classifier",
        action="store_true",
        help=(
            "Disable lightweight lexical NER typing enrichment in labels_json. "
            "When disabled, coarse_type/fine_type are emitted empty."
        ),
    )
    parser.add_argument(
        "--languages",
        default="en",
        help=(
            "Comma-separated language allowlist for labels/descriptions "
            "(default: en). Falls back to one available language when missing."
        ),
    )
    parser.add_argument(
        "--max-aliases-per-language",
        type=parse_non_negative_int,
        default=8,
        help="Max aliases stored per language (default: 8, 0 disables aliases).",
    )
    return parser.parse_args()


def configure_db(conn: sqlite3.Connection) -> None:
    for pragma in PRAGMA_STATEMENTS:
        conn.execute(pragma)

    conn.execute(CREATE_TABLE_SQL)
    conn.commit()


def flush_rows(conn: sqlite3.Connection, rows: list[tuple[str, str]]) -> int:
    if not rows:
        return 0

    with conn:
        conn.executemany(UPSERT_SQL, rows)
    return len(rows)


def run(
    dump_path: Path,
    db_path: Path,
    batch_size: int,
    limit: int,
    disable_ner_classifier: bool,
    language_allowlist: Sequence[str] | None = None,
    max_aliases_per_language: int = 8,
) -> int:
    if batch_size == 0:
        raise ValueError("--batch-size must be >= 1")
    if max_aliases_per_language < 0:
        raise ValueError("--max-aliases-per-language must be >= 0")

    active_language_allowlist = tuple(language_allowlist) if language_allowlist else ("en",)

    ensure_existing_file(
        dump_path,
        "Wikidata dump",
        hint=(
            f"Provide --dump-path or set ${DUMP_PATH_ENV}. "
            "Expected a readable dump file."
        ),
    )
    ensure_parent_dir(db_path)

    parsed_entities = 0
    candidate_entities = 0
    stored_rows = 0
    typed_rows = 0
    pending_rows: list[tuple[str, str]] = []

    conn = sqlite3.connect(db_path)
    try:
        configure_db(conn)
        progress_total = estimate_wikidata_entity_total(
            dump_path,
            limit=None if limit == 0 else limit,
        )
        if progress_total is not None:
            print(f"Progress estimate: labels-db total~{progress_total} entities")

        with tqdm(total=progress_total, desc="labels-db", unit="entity") as progress:
            for entity in iter_wikidata_entities(
                dump_path,
                limit=None if limit == 0 else limit,
            ):
                parsed_entities += 1
                progress.update(1)
                keep_tqdm_total_ahead(progress)

                entity_id = entity.get("id")
                if not isinstance(entity_id, str) or not is_supported_entity_id(entity_id):
                    continue

                candidate_entities += 1
                payload = extract_multilingual_payload(entity)
                payload["labels"] = select_text_map_languages(
                    payload["labels"],
                    active_language_allowlist,
                    fallback_to_any=True,
                )
                payload["descriptions"] = select_text_map_languages(
                    payload["descriptions"],
                    active_language_allowlist,
                    fallback_to_any=True,
                )
                payload["aliases"] = select_alias_map_languages(
                    payload["aliases"],
                    active_language_allowlist,
                    max_aliases_per_language=max_aliases_per_language,
                    fallback_to_any=False,
                )
                if disable_ner_classifier:
                    payload["coarse_type"] = ""
                    payload["fine_type"] = ""
                    payload["ner_type_source"] = "disabled"
                else:
                    coarse_types, fine_types, source = infer_ner_types(
                        entity_id=entity_id,
                        labels=payload["labels"],
                        aliases=payload["aliases"],
                        descriptions=payload["descriptions"],
                    )
                    payload["coarse_type"] = coarse_types[0] if coarse_types else ""
                    payload["fine_type"] = fine_types[0] if fine_types else ""
                    payload["ner_type_source"] = source
                    if payload["coarse_type"] or payload["fine_type"]:
                        typed_rows += 1

                serialized_payload = json.dumps(
                    payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                pending_rows.append((entity_id, serialized_payload))

                if len(pending_rows) >= batch_size:
                    stored_rows += flush_rows(conn, pending_rows)
                    pending_rows.clear()
                    progress.set_postfix(stored=stored_rows)

            stored_rows += flush_rows(conn, pending_rows)
            pending_rows.clear()
            progress.set_postfix(stored=stored_rows)
            finalize_tqdm_total(progress)
    finally:
        conn.close()

    print(
        "Completed labels DB build:",
        f"parsed={parsed_entities}",
        f"qid_pid={candidate_entities}",
        f"stored={stored_rows}",
        f"typed={typed_rows}",
        f"languages={','.join(active_language_allowlist)}",
        f"max_aliases_per_language={max_aliases_per_language}",
        f"db={db_path}",
    )
    return 0


def main() -> int:
    args = parse_args()

    try:
        dump_path = resolve_dump_path(args.dump_path)
        db_path = resolve_labels_db_path(args.db_path)
        language_allowlist = parse_language_allowlist(args.languages, arg_name="--languages")
        return run(
            dump_path=dump_path,
            db_path=db_path,
            batch_size=args.batch_size,
            limit=args.limit,
            disable_ner_classifier=args.disable_ner_classifier,
            language_allowlist=language_allowlist,
            max_aliases_per_language=args.max_aliases_per_language,
        )
    except (FileNotFoundError, ValueError, sqlite3.Error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
