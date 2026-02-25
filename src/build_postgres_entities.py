from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any
from urllib.parse import quote

from .build_bow_docs import extract_claim_object_ids
from .common import (
    DUMP_PATH_ENV,
    ensure_existing_file,
    estimate_wikidata_entity_total,
    extract_multilingual_payload,
    finalize_tqdm_total,
    is_supported_entity_id,
    iter_wikidata_entities,
    keep_tqdm_total_ahead,
    parse_language_allowlist,
    resolve_dump_path,
    resolve_postgres_dsn,
    select_alias_map_languages,
    select_text_map_languages,
    tqdm,
)
from .ner_typing import infer_ner_types
from .postgres_store import EntityRecord, PostgresStore


DEFAULT_MAX_CONTEXT_OBJECT_IDS = 32
DISAMBIGUATION_INSTANCE_OF_QIDS = frozenset(
    {
        "Q4167410",   # Wikimedia disambiguation page
        "Q22808320",  # Wikimedia human name disambiguation page
    }
)
CLASSLIKE_INSTANCE_OF_QIDS = frozenset(
    {
        "Q16889133",  # class
        "Q24017414",  # first-order class
    }
)


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
    if value <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return value


def _pick_primary_label(labels: Mapping[str, str]) -> str:
    english = labels.get("en")
    if isinstance(english, str) and english.strip():
        return english.strip()
    for language in sorted(labels):
        value = labels[language]
        if isinstance(value, str):
            candidate = value.strip()
            if candidate:
                return candidate
    return ""


def _extract_cross_refs(entity: Mapping[str, Any]) -> dict[str, Any]:
    raw_sitelinks = entity.get("sitelinks")
    cross_refs: dict[str, Any] = {}
    if not isinstance(raw_sitelinks, Mapping):
        return cross_refs

    enwiki_payload = raw_sitelinks.get("enwiki")
    if not isinstance(enwiki_payload, Mapping):
        return cross_refs

    en_title = enwiki_payload.get("title")
    if not isinstance(en_title, str) or not en_title.strip():
        return cross_refs

    title = en_title.strip()
    wiki_path = quote(title.replace(" ", "_"), safe="()'!*,._-")
    cross_refs["wikipedia"] = f"https://en.wikipedia.org/wiki/{wiki_path}"
    cross_refs["dbpedia"] = f"https://dbpedia.org/resource/{quote(title.replace(' ', '_'))}"
    return cross_refs


def _extract_popularity(entity: Mapping[str, Any]) -> float:
    raw_sitelinks = entity.get("sitelinks")
    if isinstance(raw_sitelinks, Mapping):
        return float(len(raw_sitelinks))
    return 0.0


def _claim_object_ids_for_property(
    entity: Mapping[str, Any],
    property_id: str,
    *,
    limit: int = 32,
) -> list[str]:
    claims = entity.get("claims")
    if not isinstance(claims, Mapping):
        return []
    selected = claims.get(property_id)
    if selected is None:
        return []
    wrapper = {"claims": {property_id: selected}}
    return extract_claim_object_ids(wrapper, limit=limit)


def infer_item_category(entity: Mapping[str, Any]) -> str:
    entity_id = entity.get("id")
    if not isinstance(entity_id, str) or not entity_id:
        return "OTHER"

    entity_type = entity.get("type")
    if entity_id.startswith("P") or entity_type == "property":
        return "PREDICATE"
    if entity_type == "lexeme":
        return "LEXEME"
    if entity_type == "form":
        return "FORM"
    if entity_type == "sense":
        return "SENSE"
    if entity_type == "mediainfo":
        return "MEDIAINFO"

    if not entity_id.startswith("Q"):
        return "OTHER"

    claims = entity.get("claims")
    if not isinstance(claims, Mapping):
        return "ENTITY"

    p31_ids = set(_claim_object_ids_for_property(entity, "P31", limit=16))
    if p31_ids & DISAMBIGUATION_INSTANCE_OF_QIDS:
        return "DISAMBIGUATION"

    p279_claims = claims.get("P279")
    if isinstance(p279_claims, Sequence) and not isinstance(
        p279_claims, (str, bytes, bytearray)
    ):
        if any(isinstance(statement, Mapping) for statement in p279_claims):
            return "TYPE"

    if p31_ids & CLASSLIKE_INSTANCE_OF_QIDS:
        return "TYPE"

    return "ENTITY"


def transform_entity_to_record(
    entity: Mapping[str, Any],
    *,
    language_allowlist: Sequence[str],
    max_aliases_per_language: int,
    max_context_object_ids: int,
    disable_ner_classifier: bool,
) -> EntityRecord | None:
    entity_id = entity.get("id")
    if not isinstance(entity_id, str) or not is_supported_entity_id(entity_id):
        return None

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
    label = _pick_primary_label(labels)
    if not label:
        return None

    if disable_ner_classifier:
        coarse_type = ""
        fine_type = ""
    else:
        coarse_types, fine_types, _source = infer_ner_types(
            entity_id=entity_id,
            labels=labels,
            aliases=aliases,
            descriptions=descriptions,
        )
        coarse_type = coarse_types[0] if coarse_types else ""
        fine_type = fine_types[0] if fine_types else ""

    relation_object_qids = extract_claim_object_ids(entity, limit=max_context_object_ids)
    popularity = _extract_popularity(entity)
    cross_refs = _extract_cross_refs(entity)
    item_category = infer_item_category(entity)

    return EntityRecord(
        qid=entity_id,
        label=label,
        labels=labels,
        aliases=aliases,
        relation_object_qids=relation_object_qids,
        item_category=item_category,
        coarse_type=coarse_type,
        fine_type=fine_type,
        popularity=popularity,
        cross_refs=cross_refs,
        context_string="",
    )


def _flush_transform_batch(
    store: PostgresStore,
    raw_entities: list[dict[str, Any]],
    *,
    executor: ThreadPoolExecutor | None,
    language_allowlist: Sequence[str],
    max_aliases_per_language: int,
    max_context_object_ids: int,
    disable_ner_classifier: bool,
    build_search_vector: bool,
) -> tuple[int, int]:
    if not raw_entities:
        return 0, 0

    typed_rows = 0
    records: list[EntityRecord] = []

    def _transform(entity: Mapping[str, Any]) -> EntityRecord | None:
        return transform_entity_to_record(
            entity,
            language_allowlist=language_allowlist,
            max_aliases_per_language=max_aliases_per_language,
            max_context_object_ids=max_context_object_ids,
            disable_ner_classifier=disable_ner_classifier,
        )

    if executor is not None and len(raw_entities) > 1:
        for record in executor.map(_transform, raw_entities):
            if record is None:
                continue
            if record.coarse_type or record.fine_type:
                typed_rows += 1
            records.append(record)
    else:
        for entity in raw_entities:
            record = _transform(entity)
            if record is None:
                continue
            if record.coarse_type or record.fine_type:
                typed_rows += 1
            records.append(record)

    stored = store.upsert_entities(records, build_search_vector=build_search_vector)
    return stored, typed_rows


def run_pass1(
    *,
    dump_path: Path,
    postgres_dsn: str,
    batch_size: int,
    limit: int,
    language_allowlist: Sequence[str] | None = None,
    max_aliases_per_language: int = 8,
    max_context_object_ids: int = DEFAULT_MAX_CONTEXT_OBJECT_IDS,
    disable_ner_classifier: bool = False,
    worker_count: int | None = None,
    build_search_vector: bool = True,
    expected_entity_total: int | None = None,
) -> int:
    if batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if max_aliases_per_language < 0:
        raise ValueError("--max-aliases-per-language must be >= 0")
    if max_context_object_ids < 0:
        raise ValueError("--max-context-object-ids must be >= 0")
    if expected_entity_total is not None and expected_entity_total <= 0:
        raise ValueError("--expected-entity-total must be > 0 when provided")

    ensure_existing_file(
        dump_path,
        "Wikidata dump",
        hint=f"Provide --dump-path or set ${DUMP_PATH_ENV}.",
    )

    active_languages = tuple(language_allowlist) if language_allowlist else ("en",)
    workers = max(1, worker_count or min(8, (os.cpu_count() or 1)))
    store = PostgresStore(postgres_dsn)
    store.ensure_schema()

    parsed_entities = 0
    stored_rows = 0
    typed_rows = 0
    pending_entities: list[dict[str, Any]] = []

    if expected_entity_total is not None:
        progress_total = int(expected_entity_total)
        if limit > 0:
            progress_total = min(progress_total, limit)
        print(f"Progress estimate: pg-pass1 total~{progress_total} entities (manual override)")
    else:
        progress_total = estimate_wikidata_entity_total(
            dump_path,
            limit=None if limit == 0 else limit,
        )
        if progress_total is not None:
            print(f"Progress estimate: pg-pass1 total~{progress_total} entities (sampled)")

    executor: ThreadPoolExecutor | None = (
        ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
    )
    try:
        with tqdm(total=progress_total, desc="pg-pass1", unit="entity") as progress:
            for entity in iter_wikidata_entities(dump_path, limit=None if limit == 0 else limit):
                parsed_entities += 1
                progress.update(1)
                keep_tqdm_total_ahead(progress)
                pending_entities.append(entity)

                if len(pending_entities) >= batch_size:
                    stored, typed = _flush_transform_batch(
                        store,
                        pending_entities,
                        executor=executor,
                        language_allowlist=active_languages,
                        max_aliases_per_language=max_aliases_per_language,
                        max_context_object_ids=max_context_object_ids,
                        disable_ner_classifier=disable_ner_classifier,
                        build_search_vector=build_search_vector,
                    )
                    stored_rows += stored
                    typed_rows += typed
                    pending_entities.clear()
                    progress.set_postfix(stored=stored_rows)

            stored, typed = _flush_transform_batch(
                store,
                pending_entities,
                executor=executor,
                language_allowlist=active_languages,
                max_aliases_per_language=max_aliases_per_language,
                max_context_object_ids=max_context_object_ids,
                disable_ner_classifier=disable_ner_classifier,
                build_search_vector=build_search_vector,
            )
            stored_rows += stored
            typed_rows += typed
            pending_entities.clear()
            progress.set_postfix(stored=stored_rows)
            finalize_tqdm_total(progress)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    print(
        "Completed Postgres pass1:",
        f"parsed={parsed_entities}",
        f"stored={stored_rows}",
        f"typed={typed_rows}",
        f"languages={','.join(active_languages)}",
        f"workers={workers}",
        f"build_search_vector={build_search_vector}",
    )
    return 0


def _build_context_string_for_qids(postgres_dsn: str, qids: Sequence[str]) -> int:
    store = PostgresStore(postgres_dsn)
    batch = store.load_context_inputs(qids)

    related_ids: list[str] = []
    seen: set[str] = set()
    for _qid, relation_object_qids in batch:
        for object_qid in relation_object_qids:
            if object_qid in seen:
                continue
            seen.add(object_qid)
            related_ids.append(object_qid)

    label_map = store.resolve_labels(related_ids)

    updates: list[tuple[str, str]] = []
    for qid, relation_object_qids in batch:
        context_tokens = {
            label_map[object_qid].strip()
            for object_qid in relation_object_qids
            if object_qid in label_map and label_map[object_qid].strip()
        }
        context_string = "; ".join(sorted(context_tokens))
        updates.append((qid, context_string))

    return store.update_context_strings(updates)


def run_pass2(
    *,
    postgres_dsn: str,
    batch_size: int = 1000,
    worker_count: int | None = None,
) -> int:
    if batch_size <= 0:
        raise ValueError("--context-batch-size must be > 0")
    workers = max(1, worker_count or min(8, (os.cpu_count() or 1)))
    store = PostgresStore(postgres_dsn)
    store.ensure_schema()

    total_entities = store.count_entities()
    updated_total = 0
    in_flight: set[Future[int]] = set()
    submitted = 0

    def _drain_ready(*, block: bool) -> None:
        nonlocal updated_total
        if not in_flight:
            return
        if block:
            done, pending = wait(in_flight, return_when=FIRST_COMPLETED)
        else:
            done = {future for future in in_flight if future.done()}
            pending = in_flight - done
            if not done:
                return
        in_flight.clear()
        in_flight.update(pending)
        for future in done:
            updated_total += future.result()

    with tqdm(total=total_entities or None, desc="pg-pass2", unit="entity") as progress:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for qids in store.iter_entity_ids(batch_size=batch_size):
                submitted += len(qids)
                in_flight.add(executor.submit(_build_context_string_for_qids, postgres_dsn, qids))
                if len(in_flight) >= max(1, workers * 2):
                    before = updated_total
                    _drain_ready(block=True)
                    delta = max(0, updated_total - before)
                    if delta:
                        progress.update(delta)
                        keep_tqdm_total_ahead(progress)
                        progress.set_postfix(updated=updated_total, queued=len(in_flight))
                else:
                    _drain_ready(block=False)
                    if updated_total > progress.n:
                        delta = int(updated_total - progress.n)
                        progress.update(delta)
                        keep_tqdm_total_ahead(progress)

            while in_flight:
                before = updated_total
                _drain_ready(block=True)
                delta = max(0, updated_total - before)
                if delta:
                    progress.update(delta)
                    keep_tqdm_total_ahead(progress)
                    progress.set_postfix(updated=updated_total, queued=len(in_flight))
        finalize_tqdm_total(progress)

    print(
        "Completed Postgres pass2:",
        f"entities={total_entities}",
        f"updated={updated_total}",
        f"workers={workers}",
        f"batch_size={batch_size}",
        f"submitted={submitted}",
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Two-pass Postgres ingestion for deterministic entity lookup."
    )
    parser.add_argument("--dump-path", help=f"Input dump path. Overrides ${DUMP_PATH_ENV}.")
    parser.add_argument("--postgres-dsn", help="Postgres DSN.")
    parser.add_argument(
        "--mode",
        choices=("pass1", "pass2", "all"),
        default="all",
        help="Which ingestion phase(s) to run.",
    )
    parser.add_argument("--batch-size", type=parse_positive_int, default=5000)
    parser.add_argument("--context-batch-size", type=parse_positive_int, default=1000)
    parser.add_argument("--limit", type=parse_non_negative_int, default=0)
    parser.add_argument(
        "--expected-entity-total",
        type=parse_non_negative_int,
        default=0,
        help=(
            "Optional manual progress total override for pass1 (e.g., from Wikidata:Statistics). "
            "0 = auto sample from the local dump."
        ),
    )
    parser.add_argument("--workers", type=parse_positive_int, default=max(1, min(8, (os.cpu_count() or 1))))
    parser.add_argument(
        "--languages",
        default="en",
        help="Comma-separated language allowlist for labels/descriptions.",
    )
    parser.add_argument("--max-aliases-per-language", type=parse_non_negative_int, default=8)
    parser.add_argument(
        "--max-context-object-ids",
        type=parse_non_negative_int,
        default=DEFAULT_MAX_CONTEXT_OBJECT_IDS,
    )
    parser.add_argument("--disable-ner-classifier", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        postgres_dsn = resolve_postgres_dsn(args.postgres_dsn)
        if args.mode in {"pass1", "all"}:
            dump_path = resolve_dump_path(args.dump_path)
            language_allowlist = parse_language_allowlist(args.languages, arg_name="--languages")
            status = run_pass1(
                dump_path=dump_path,
                postgres_dsn=postgres_dsn,
                batch_size=args.batch_size,
                limit=args.limit,
                language_allowlist=language_allowlist,
                max_aliases_per_language=args.max_aliases_per_language,
                max_context_object_ids=args.max_context_object_ids,
                disable_ner_classifier=args.disable_ner_classifier,
                worker_count=args.workers,
                expected_entity_total=(args.expected_entity_total or None),
            )
            if status != 0:
                return status

        if args.mode in {"pass2", "all"}:
            status = run_pass2(
                postgres_dsn=postgres_dsn,
                batch_size=args.context_batch_size,
                worker_count=args.workers,
            )
            if status != 0:
                return status
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
