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
    build_english_name_text,
    parse_non_negative_int,
    parse_positive_int,
)
from .common import (
    DEFAULT_STOPWORDS,
    extract_multilingual_payload,
    normalize_text,
    parse_language_allowlist,
    resolve_postgres_dsn,
    select_alias_map_languages,
    select_text_map_languages,
    tokenize,
)
from .ner_typing import infer_ner_types
from .postgres_store import PostgresStore
from .wikidata_sample_ids import resolve_qids

_PREFERRED_LANGUAGES = ("en",)


def _append_label_tokens(
    token_buffer: list[str],
    text: str,
    *,
    max_tokens: int,
) -> None:
    if len(token_buffer) >= max_tokens:
        return
    for token in tokenize(text):
        if len(token_buffer) >= max_tokens:
            return
        if len(token) <= 1 or token in DEFAULT_STOPWORDS:
            continue
        token_buffer.append(token)


def build_graph_label_bow(
    *,
    outgoing_triples: Sequence[tuple[str, str]],
    incoming_triples: Sequence[tuple[str, str]],
    label_map: Mapping[str, str],
    max_tokens: int,
) -> str:
    tokens: list[str] = []

    for predicate_pid, object_qid in outgoing_triples:
        predicate_label = label_map.get(predicate_pid, "")
        if isinstance(predicate_label, str) and predicate_label:
            _append_label_tokens(tokens, predicate_label, max_tokens=max_tokens)
        object_label = label_map.get(object_qid, "")
        if isinstance(object_label, str) and object_label:
            _append_label_tokens(tokens, object_label, max_tokens=max_tokens)
        if len(tokens) >= max_tokens:
            break

    if len(tokens) < max_tokens:
        for predicate_pid, subject_qid in incoming_triples:
            predicate_label = label_map.get(predicate_pid, "")
            if isinstance(predicate_label, str) and predicate_label:
                _append_label_tokens(tokens, predicate_label, max_tokens=max_tokens)
            subject_label = label_map.get(subject_qid, "")
            if isinstance(subject_label, str) and subject_label:
                _append_label_tokens(tokens, subject_label, max_tokens=max_tokens)
            if len(tokens) >= max_tokens:
                break

    return normalize_text(" ".join(tokens))


def build_cached_qid_bow_record(
    entity: Mapping[str, Any],
    *,
    qid: str,
    outgoing_triples: Sequence[tuple[str, str]],
    incoming_triples: Sequence[tuple[str, str]],
    triple_label_map: Mapping[str, str],
    language_allowlist: Sequence[str],
    max_aliases_per_language: int,
    max_bow_tokens: int,
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

    return {
        "id": qid,
        "labels": labels,
        "aliases": aliases,
        "name_text": build_english_name_text(labels, aliases),
        "bow": build_graph_label_bow(
            outgoing_triples=outgoing_triples,
            incoming_triples=incoming_triples,
            label_map=triple_label_map,
            max_tokens=max_bow_tokens,
        ),
        "coarse_type": coarse_type,
        "fine_type": fine_type,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build graph-label BOW JSONL records for explicit QIDs using cached entities "
            "plus Postgres entity_triples."
        )
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
        help=f"Max BOW tokens emitted from resolved triple labels (default: {DEFAULT_MAX_BOW_TOKENS}).",
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

        triple_neighbors = store.load_entity_triple_neighbors(qids)
        triple_label_ids: list[str] = []
        seen_label_ids: set[str] = set()
        for qid in qids:
            neighbors = triple_neighbors.get(qid, {})
            for predicate_pid, object_qid in neighbors.get("outgoing", []):
                for entity_id in (predicate_pid, object_qid):
                    if entity_id not in seen_label_ids:
                        seen_label_ids.add(entity_id)
                        triple_label_ids.append(entity_id)
            for predicate_pid, subject_qid in neighbors.get("incoming", []):
                for entity_id in (predicate_pid, subject_qid):
                    if entity_id not in seen_label_ids:
                        seen_label_ids.add(entity_id)
                        triple_label_ids.append(entity_id)
        triple_label_map = store.resolve_labels(triple_label_ids)

        for qid in qids:
            entity = cached[qid]
            neighbors = triple_neighbors.get(qid, {})
            outgoing_triples = neighbors.get("outgoing", [])
            incoming_triples = neighbors.get("incoming", [])
            record = build_cached_qid_bow_record(
                entity,
                qid=qid,
                outgoing_triples=outgoing_triples,
                incoming_triples=incoming_triples,
                triple_label_map=triple_label_map,
                language_allowlist=language_allowlist,
                max_aliases_per_language=args.max_aliases_per_language,
                max_bow_tokens=args.max_bow_tokens,
            )
            json.dump(record, sys.stdout, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
            sys.stdout.write("\n")
        return 0
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
