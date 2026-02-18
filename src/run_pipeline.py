from __future__ import annotations

import argparse
import sys

from .build_bow_docs import run as run_bow_docs
from .build_labels_db import run as run_labels_db
from .build_quickwit_index import run as run_quickwit_index
from .common import (
    parse_language_allowlist,
    resolve_bow_output_path,
    resolve_dump_path,
    resolve_labels_db_path,
    resolve_ner_types_path,
    resolve_quickwit_index_config_path,
    resolve_quickwit_index_id,
    resolve_quickwit_url,
)
from .quickwit_client import QuickwitClientError


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
            "Run the full alpaca preprocessing pipeline: labels DB -> BOW docs -> "
            "Quickwit index/ingest."
        )
    )
    parser.add_argument("--dump-path", help="Input dump path (.json/.json.gz/.json.bz2).")
    parser.add_argument("--labels-db-path", help="Output labels SQLite path.")
    parser.add_argument("--bow-output-path", help="Output BOW JSONL(.gz) path.")
    parser.add_argument(
        "--ner-types-path",
        help=(
            "Optional NER type map (.jsonl/.jsonl.gz) with records "
            "{\"id\":...,\"coarse_types\":[...],\"fine_types\":[...]}."
        ),
    )
    parser.add_argument("--index-config-path", help="Quickwit index config JSON path.")
    parser.add_argument("--quickwit-url", help="Quickwit base URL.")
    parser.add_argument("--index-id", help="Quickwit index ID.")

    parser.add_argument(
        "--limit",
        type=parse_non_negative_int,
        default=0,
        help="Entity parse limit for smoke runs (0 = no limit).",
    )
    parser.add_argument(
        "--labels-batch-size",
        type=parse_positive_int,
        default=5000,
        help="SQLite insert batch size for labels DB step (default: 5000).",
    )
    parser.add_argument(
        "--bow-batch-size",
        type=parse_positive_int,
        default=5000,
        help="JSONL write batch size for BOW step (default: 5000).",
    )
    parser.add_argument(
        "--quickwit-chunk-bytes",
        type=parse_positive_int,
        default=8_000_000,
        help="Max NDJSON payload bytes per Quickwit ingest call.",
    )
    parser.add_argument(
        "--quickwit-ingest-limit",
        type=parse_non_negative_int,
        default=0,
        help="Optional document ingest limit for Quickwit step (0 = no limit).",
    )
    parser.add_argument(
        "--quickwit-final-commit",
        choices=("auto", "wait_for", "force"),
        default="wait_for",
        help="Commit mode for final Quickwit ingest chunk.",
    )
    parser.add_argument(
        "--wait-timeout-seconds",
        type=parse_non_negative_float,
        default=120.0,
        help="How long to wait for Quickwit startup.",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=parse_non_negative_float,
        default=2.0,
        help="Polling interval while waiting for Quickwit.",
    )
    parser.add_argument(
        "--quickwit-http-timeout-seconds",
        type=parse_positive_int,
        default=120,
        help="HTTP timeout per Quickwit request.",
    )

    parser.add_argument(
        "--skip-labels",
        action="store_true",
        help="Skip labels DB build step.",
    )
    parser.add_argument(
        "--skip-bow",
        action="store_true",
        help="Skip BOW docs build step.",
    )
    parser.add_argument(
        "--skip-quickwit",
        action="store_true",
        help="Skip Quickwit index/ingest step.",
    )
    parser.add_argument(
        "--disable-ner-classifier",
        action="store_true",
        help="Disable lexical NER typing during labels DB build.",
    )
    parser.add_argument(
        "--languages",
        default="en",
        help=(
            "Comma-separated language allowlist for labels/descriptions in labels DB "
            "(default: en)."
        ),
    )
    parser.add_argument(
        "--max-aliases-per-language",
        type=parse_non_negative_int,
        default=8,
        help="Max aliases stored/emitted per language (default: 8, 0 disables aliases).",
    )
    parser.add_argument(
        "--max-bow-tokens",
        type=parse_positive_int,
        default=128,
        help="Max unique tokens in bow field per entity (default: 128).",
    )
    parser.add_argument(
        "--max-context-object-ids",
        type=parse_non_negative_int,
        default=32,
        help="Max claim object IDs used to build context per entity (default: 32).",
    )
    parser.add_argument(
        "--max-context-chars",
        type=parse_non_negative_int,
        default=640,
        help="Max chars stored in context field per entity (default: 640).",
    )
    parser.add_argument(
        "--max-doc-bytes",
        type=parse_positive_int,
        default=4096,
        help="Hard cap for serialized docs emitted to JSONL (default: 4096).",
    )
    parser.add_argument(
        "--context-label-cache-size",
        type=parse_positive_int,
        default=200_000,
        help="LRU cache size for linked-object label lookups (default: 200000).",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    dump_path = resolve_dump_path(args.dump_path)
    labels_db_path = resolve_labels_db_path(args.labels_db_path)
    bow_output_path = resolve_bow_output_path(args.bow_output_path)
    ner_types_path = resolve_ner_types_path(args.ner_types_path)
    quickwit_url = resolve_quickwit_url(args.quickwit_url)
    index_id = resolve_quickwit_index_id(args.index_id)
    index_config_path = resolve_quickwit_index_config_path(args.index_config_path)

    try:
        language_allowlist = parse_language_allowlist(args.languages, arg_name="--languages")
        if not args.skip_labels:
            labels_status = run_labels_db(
                dump_path=dump_path,
                db_path=labels_db_path,
                batch_size=args.labels_batch_size,
                limit=args.limit,
                disable_ner_classifier=args.disable_ner_classifier,
                language_allowlist=language_allowlist,
                max_aliases_per_language=args.max_aliases_per_language,
            )
            if labels_status != 0:
                return labels_status

        if not args.skip_bow:
            bow_status = run_bow_docs(
                dump_path=dump_path,
                labels_db_path=labels_db_path,
                output_path=bow_output_path,
                batch_size=args.bow_batch_size,
                limit=args.limit,
                ner_types_path=ner_types_path,
                max_aliases_per_language=args.max_aliases_per_language,
                max_bow_tokens=args.max_bow_tokens,
                max_context_object_ids=args.max_context_object_ids,
                max_context_chars=args.max_context_chars,
                max_doc_bytes=args.max_doc_bytes,
                context_label_cache_size=args.context_label_cache_size,
            )
            if bow_status != 0:
                return bow_status

        if not args.skip_quickwit:
            quickwit_status = run_quickwit_index(
                quickwit_url=quickwit_url,
                index_id=index_id,
                index_config_path=index_config_path,
                docs_path=bow_output_path,
                chunk_bytes=args.quickwit_chunk_bytes,
                ingest_limit=args.quickwit_ingest_limit,
                final_commit=args.quickwit_final_commit,
                wait_timeout_seconds=args.wait_timeout_seconds,
                poll_interval_seconds=args.poll_interval_seconds,
                http_timeout_seconds=float(args.quickwit_http_timeout_seconds),
            )
            if quickwit_status != 0:
                return quickwit_status

        print("Pipeline completed successfully.")
        return 0
    except (FileNotFoundError, ValueError, QuickwitClientError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
