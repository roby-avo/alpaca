from __future__ import annotations

__test__ = False

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from typing import Any

from .build_bow_docs import (
    DEFAULT_MAX_ALIASES_PER_LANGUAGE,
    DEFAULT_MAX_BOW_TOKENS,
    DEFAULT_MAX_CONTEXT_CHARS,
    DEFAULT_MAX_CONTEXT_OBJECT_IDS,
    build_context_text,
    build_entity_bow,
    build_english_name_text,
    extract_claim_object_ids,
    parse_non_negative_int,
    parse_positive_int,
)
from .common import (
    extract_multilingual_payload,
    parse_language_allowlist,
    resolve_postgres_dsn,
    select_alias_map_languages,
    select_text_map_languages,
)
from .ner_typing import infer_ner_types
from .postgres_store import PostgresStore
from .wikidata_sample_ids import resolve_qids

_PREFERRED_LANGUAGES = ("en",)


def build_cached_qid_bow_record(
    entity: Mapping[str, Any],
    *,
    qid: str,
    context_label_map: Mapping[str, str],
    language_allowlist: Sequence[str],
    max_aliases_per_language: int,
    max_bow_tokens: int,
    max_context_object_ids: int,
    max_context_chars: int,
) -> dict[str, Any]:
    payload = extract_multilingual_payload(entity)
    labels = select_text_map_languages(
        payload.get("labels", {}),
        language_allowlist,
        fallback_to_any=True,
    )
    aliases = select_alias_map_languages(
        payload.get("aliases", {}),
        language_allowlist,
        max_aliases_per_language=max_aliases_per_language,
        fallback_to_any=False,
    )
    descriptions = select_text_map_languages(
        payload.get("descriptions", {}),
        language_allowlist,
        fallback_to_any=True,
    )
    coarse_types, fine_types, _source = infer_ner_types(
        entity_id=qid,
        labels=labels,
        aliases=aliases,
        descriptions=descriptions,
        claims=entity.get("claims") if isinstance(entity.get("claims"), Mapping) else None,
    )
    coarse_type = coarse_types[0] if coarse_types else ""
    fine_type = fine_types[0] if fine_types else ""

    claim_object_ids = extract_claim_object_ids(entity, limit=max_context_object_ids)
    context_labels = [
        context_label_map[object_id]
        for object_id in claim_object_ids
        if object_id in context_label_map and isinstance(context_label_map[object_id], str)
    ]

    return {
        "id": qid,
        "labels": labels,
        "aliases": aliases,
        "name_text": build_english_name_text(labels, aliases),
        "context": build_context_text(context_labels, max_chars=max_context_chars),
        "bow": build_entity_bow(
            labels=labels,
            aliases=aliases,
            descriptions=descriptions,
            coarse_type=coarse_type,
            fine_type=fine_type,
            max_tokens=max_bow_tokens,
        ),
        "coarse_type": coarse_type,
        "fine_type": fine_type,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build BOW-style JSONL records for explicit QIDs already cached in Postgres sample_entity_cache."
    )
    parser.add_argument("--postgres-dsn", help="Postgres DSN (defaults to ALPACA_POSTGRES_DSN).")
    parser.add_argument("--ids", help="Comma-separated QIDs (example: Q42,Q90,Q64).")
    parser.add_argument("--ids-file", help="Text file with one QID per line (comments with # allowed).")
    parser.add_argument(
        "--languages",
        default="en",
        help="Comma-separated language allowlist for labels/descriptions (default: en).",
    )
    parser.add_argument(
        "--max-aliases-per-language",
        type=parse_non_negative_int,
        default=DEFAULT_MAX_ALIASES_PER_LANGUAGE,
        help=(
            "Max aliases kept per language in output docs "
            f"(default: {DEFAULT_MAX_ALIASES_PER_LANGUAGE}, 0 disables aliases)."
        ),
    )
    parser.add_argument(
        "--max-bow-tokens",
        type=parse_positive_int,
        default=DEFAULT_MAX_BOW_TOKENS,
        help=f"Max unique tokens in bow field (default: {DEFAULT_MAX_BOW_TOKENS}).",
    )
    parser.add_argument(
        "--max-context-object-ids",
        type=parse_non_negative_int,
        default=DEFAULT_MAX_CONTEXT_OBJECT_IDS,
        help=(
            "Max claim object IDs read per entity for context expansion "
            f"(default: {DEFAULT_MAX_CONTEXT_OBJECT_IDS})."
        ),
    )
    parser.add_argument(
        "--max-context-chars",
        type=parse_non_negative_int,
        default=DEFAULT_MAX_CONTEXT_CHARS,
        help=(
            "Max chars stored in context field per entity "
            f"(default: {DEFAULT_MAX_CONTEXT_CHARS}, 0 disables context text)."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        qids = resolve_qids(args.ids, args.ids_file, None)
        language_allowlist = parse_language_allowlist(args.languages, arg_name="--languages")

        store = PostgresStore(resolve_postgres_dsn(args.postgres_dsn))
        store.ensure_schema()
        cached = store.get_sample_entities(qids)
        missing = [qid for qid in qids if qid not in cached]
        if missing:
            sample = ", ".join(missing[:5])
            raise ValueError(
                f"Missing {len(missing)} requested QIDs in sample_entity_cache"
                f"{': ' + sample if sample else ''}."
            )

        for qid in qids:
            entity = cached[qid]
            claim_object_ids = extract_claim_object_ids(entity, limit=args.max_context_object_ids)
            context_label_map = store.resolve_sample_cache_labels(claim_object_ids)
            record = build_cached_qid_bow_record(
                entity,
                qid=qid,
                context_label_map=context_label_map,
                language_allowlist=language_allowlist,
                max_aliases_per_language=args.max_aliases_per_language,
                max_bow_tokens=args.max_bow_tokens,
                max_context_object_ids=args.max_context_object_ids,
                max_context_chars=args.max_context_chars,
            )
            json.dump(record, sys.stdout, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
            sys.stdout.write("\n")
        return 0
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
