from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
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
    normalize_text,
    parse_language_allowlist,
    resolve_dump_path,
    resolve_postgres_dsn,
    select_alias_map_languages,
    select_text_map_languages,
    tqdm,
)
from .ner_typing import infer_ner_types
from .postgres_store import (
    EntityRecord,
    EntityTripleRecord,
    PostgresStore,
    build_entity_context_string,
)
PRIMARY_LABEL_LANGUAGE_PREFERENCE = ("en", "mul")
NON_ARTICLE_WIKIPEDIA_SITES = frozenset(
    {
        "commonswiki",
        "foundationwiki",
        "incubatorwiki",
        "mediawikiwiki",
        "metawiki",
        "outreachwiki",
        "specieswiki",
        "strategywiki",
        "testwiki",
        "wikidatawiki",
    }
)
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
HUMAN_QID = "Q5"
_build_entity_context_string = build_entity_context_string


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
    for language in PRIMARY_LABEL_LANGUAGE_PREFERENCE:
        preferred = labels.get(language)
        if isinstance(preferred, str) and preferred.strip():
            return preferred.strip()
    for language in sorted(labels):
        value = labels[language]
        if isinstance(value, str):
            candidate = value.strip()
            if candidate:
                return candidate
    return ""


def _pick_primary_label_language(labels: Mapping[str, str]) -> str:
    for language in PRIMARY_LABEL_LANGUAGE_PREFERENCE:
        value = labels.get(language)
        if isinstance(value, str) and value.strip():
            return language
    for language in sorted(labels):
        value = labels[language]
        if isinstance(value, str) and value.strip():
            return language
    return ""


def _is_wikipedia_article_site(site_key: str) -> bool:
    return site_key.endswith("wiki") and site_key not in NON_ARTICLE_WIKIPEDIA_SITES


def _extract_sitelink_title(payload: Any) -> str:
    if not isinstance(payload, Mapping):
        return ""
    title = payload.get("title")
    if not isinstance(title, str):
        return ""
    return title.strip()


def _preferred_wikipedia_sitelink(
    raw_sitelinks: Mapping[str, Any],
    *,
    preferred_language: str,
) -> tuple[str, str] | None:
    candidates: dict[str, str] = {}
    for site_key, payload in raw_sitelinks.items():
        if not isinstance(site_key, str) or not _is_wikipedia_article_site(site_key):
            continue
        title = _extract_sitelink_title(payload)
        if title:
            candidates[site_key] = title
    if not candidates:
        return None

    if "enwiki" in candidates:
        return "enwiki", candidates["enwiki"]

    if preferred_language and preferred_language != "mul":
        preferred_site = f"{preferred_language}wiki"
        if preferred_site in candidates:
            return preferred_site, candidates[preferred_site]

    first_site = sorted(candidates)[0]
    return first_site, candidates[first_site]


def _site_key_to_wikipedia_host(site_key: str) -> str:
    language_key = site_key[:-4].strip().replace("_", "-")
    return f"{language_key}.wikipedia.org" if language_key else ""


def _site_key_to_dbpedia_host(site_key: str) -> str:
    language_key = site_key[:-4].strip().replace("_", "-").lower()
    if not language_key:
        return ""
    if language_key == "en":
        return "dbpedia.org"
    return f"{language_key}.dbpedia.org"


def _build_wikipedia_url(site_key: str, title: str) -> str:
    host = _site_key_to_wikipedia_host(site_key)
    if not host:
        return ""
    wiki_path = quote(title.replace(" ", "_"), safe="()'!*,._-")
    return f"https://{host}/wiki/{wiki_path}"


def _build_dbpedia_url(site_key: str, title: str) -> str:
    host = _site_key_to_dbpedia_host(site_key)
    if not host:
        return ""
    return f"https://{host}/resource/{quote(title.replace(' ', '_'))}"


def _extract_cross_refs(
    entity: Mapping[str, Any],
    *,
    preferred_language: str,
) -> dict[str, Any]:
    raw_sitelinks = entity.get("sitelinks")
    cross_refs: dict[str, Any] = {}
    if not isinstance(raw_sitelinks, Mapping):
        return cross_refs

    preferred_sitelink = _preferred_wikipedia_sitelink(
        raw_sitelinks,
        preferred_language=preferred_language,
    )
    if preferred_sitelink is None:
        return cross_refs

    site_key, title = preferred_sitelink
    wikipedia_url = _build_wikipedia_url(site_key, title)
    if wikipedia_url:
        cross_refs["wikipedia"] = wikipedia_url
    dbpedia_url = _build_dbpedia_url(site_key, title)
    if dbpedia_url:
        cross_refs["dbpedia"] = dbpedia_url
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


def _extract_entity_id_from_statement(statement: Mapping[str, Any]) -> str | None:
    mainsnak = statement.get("mainsnak")
    if not isinstance(mainsnak, Mapping) or mainsnak.get("snaktype") != "value":
        return None
    datavalue = mainsnak.get("datavalue")
    if not isinstance(datavalue, Mapping):
        return None
    raw_value = datavalue.get("value")
    if not isinstance(raw_value, Mapping):
        return None

    raw_id = raw_value.get("id")
    if isinstance(raw_id, str) and is_supported_entity_id(raw_id.strip()):
        return raw_id.strip()

    numeric_id = raw_value.get("numeric-id")
    if not isinstance(numeric_id, int) or numeric_id <= 0:
        return None
    entity_type = raw_value.get("entity-type")
    if entity_type == "item":
        return f"Q{numeric_id}"
    if entity_type == "property":
        return f"P{numeric_id}"
    return None


def extract_entity_triples(entity: Mapping[str, Any]) -> list[EntityTripleRecord]:
    entity_id = entity.get("id")
    claims = entity.get("claims")
    if not isinstance(entity_id, str) or not is_supported_entity_id(entity_id):
        return []
    if not isinstance(claims, Mapping):
        return []

    triples: list[EntityTripleRecord] = []
    seen: set[tuple[str, str, str]] = set()
    for predicate_pid in sorted(claims):
        statements = claims.get(predicate_pid)
        if not isinstance(predicate_pid, str) or not predicate_pid.startswith("P"):
            continue
        if not isinstance(statements, Sequence) or isinstance(statements, (str, bytes, bytearray)):
            continue
        for statement in statements:
            if not isinstance(statement, Mapping):
                continue
            object_id = _extract_entity_id_from_statement(statement)
            if not object_id:
                continue
            triple_key = (entity_id, predicate_pid, object_id)
            if triple_key in seen:
                continue
            seen.add(triple_key)
            triples.append(
                EntityTripleRecord(
                    subject_qid=entity_id,
                    predicate_pid=predicate_pid,
                    object_qid=object_id,
                )
            )
    return triples


def _extract_entity_type_qids(entity: Mapping[str, Any]) -> list[str]:
    type_qids: list[str] = []
    seen: set[str] = set()

    for qid in _claim_object_ids_for_property(entity, "P31", limit=32):
        if qid in seen:
            continue
        seen.add(qid)
        type_qids.append(qid)

    if HUMAN_QID in seen:
        for qid in _claim_object_ids_for_property(entity, "P106", limit=32):
            if qid in seen:
                continue
            seen.add(qid)
            type_qids.append(qid)

    return type_qids


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
    disable_ner_classifier: bool,
) -> EntityRecord | None:
    entity_id = entity.get("id")
    if not isinstance(entity_id, str) or not is_supported_entity_id(entity_id):
        return None

    payload = extract_multilingual_payload(entity)
    all_labels = payload.get("labels", {})
    all_aliases = payload.get("aliases", {})
    all_descriptions = payload.get("descriptions", {})
    labels_for_typing = select_text_map_languages(
        all_labels,
        language_allowlist,
        fallback_to_any=True,
    )
    aliases_for_typing = select_alias_map_languages(
        all_aliases,
        language_allowlist,
        max_aliases_per_language=max_aliases_per_language,
        fallback_to_any=False,
    )
    descriptions = select_text_map_languages(
        all_descriptions,
        language_allowlist,
        fallback_to_any=True,
    )
    label = _pick_primary_label(all_labels)
    if not label:
        return None
    raw_description = all_descriptions.get("en")
    description = normalize_text(raw_description) if isinstance(raw_description, str) else ""
    description = description or None

    if disable_ner_classifier:
        coarse_type = ""
        fine_type = ""
    else:
        coarse_types, fine_types, _source = infer_ner_types(
            entity_id=entity_id,
            labels=labels_for_typing,
            aliases=aliases_for_typing,
            descriptions=descriptions,
            claims=entity.get("claims") if isinstance(entity.get("claims"), Mapping) else None,
        )
        coarse_type = coarse_types[0] if coarse_types else ""
        fine_type = fine_types[0] if fine_types else ""

    type_qids = _extract_entity_type_qids(entity)
    popularity = _extract_popularity(entity)
    cross_refs = _extract_cross_refs(
        entity,
        preferred_language=_pick_primary_label_language(all_labels),
    )
    item_category = infer_item_category(entity)

    return EntityRecord(
        qid=entity_id,
        label=label,
        labels=all_labels,
        aliases=all_aliases,
        description=description,
        types=type_qids,
        item_category=item_category,
        coarse_type=coarse_type,
        fine_type=fine_type,
        popularity=popularity,
        cross_refs=cross_refs,
    )


def _flush_transform_batch(
    store: PostgresStore,
    raw_entities: list[dict[str, Any]],
    *,
    executor: ThreadPoolExecutor | None,
    language_allowlist: Sequence[str],
    max_aliases_per_language: int,
    disable_ner_classifier: bool,
) -> tuple[int, int, int]:
    if not raw_entities:
        return 0, 0, 0

    typed_rows = 0
    records: list[EntityRecord] = []
    triples: list[EntityTripleRecord] = []

    def _transform(entity: Mapping[str, Any]) -> tuple[EntityRecord | None, list[EntityTripleRecord]]:
        return (
            transform_entity_to_record(
                entity,
                language_allowlist=language_allowlist,
                max_aliases_per_language=max_aliases_per_language,
                disable_ner_classifier=disable_ner_classifier,
            ),
            extract_entity_triples(entity),
        )

    if executor is not None and len(raw_entities) > 1:
        for record, entity_triples in executor.map(_transform, raw_entities):
            triples.extend(entity_triples)
            if record is None:
                continue
            if record.coarse_type or record.fine_type:
                typed_rows += 1
            records.append(record)
    else:
        for entity in raw_entities:
            record, entity_triples = _transform(entity)
            triples.extend(entity_triples)
            if record is None:
                continue
            if record.coarse_type or record.fine_type:
                typed_rows += 1
            records.append(record)

    stored = store.upsert_entities(records)
    subject_qids = [
        entity.get("id")
        for entity in raw_entities
        if isinstance(entity.get("id"), str) and is_supported_entity_id(entity.get("id"))
    ]
    stored_triples = store.replace_entity_triples(subject_qids=subject_qids, rows=triples)
    return stored, typed_rows, stored_triples


def run_pass1(
    *,
    dump_path: Path,
    postgres_dsn: str,
    batch_size: int,
    limit: int,
    language_allowlist: Sequence[str] | None = None,
    max_aliases_per_language: int = 8,
    disable_ner_classifier: bool = False,
    worker_count: int | None = None,
    expected_entity_total: int | None = None,
) -> int:
    if batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if max_aliases_per_language < 0:
        raise ValueError("--max-aliases-per-language must be >= 0")
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
    stored_triples = 0
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
                    stored, typed, triples = _flush_transform_batch(
                        store,
                        pending_entities,
                        executor=executor,
                        language_allowlist=active_languages,
                        max_aliases_per_language=max_aliases_per_language,
                        disable_ner_classifier=disable_ner_classifier,
                    )
                    stored_rows += stored
                    typed_rows += typed
                    stored_triples += triples
                    pending_entities.clear()
                    progress.set_postfix(stored=stored_rows, triples=stored_triples)

            stored, typed, triples = _flush_transform_batch(
                store,
                pending_entities,
                executor=executor,
                language_allowlist=active_languages,
                max_aliases_per_language=max_aliases_per_language,
                disable_ner_classifier=disable_ner_classifier,
            )
            stored_rows += stored
            typed_rows += typed
            stored_triples += triples
            pending_entities.clear()
            progress.set_postfix(stored=stored_rows, triples=stored_triples)
            finalize_tqdm_total(progress)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    print(
        "Completed Postgres pass1:",
        f"parsed={parsed_entities}",
        f"stored={stored_rows}",
        f"triples={stored_triples}",
        f"typed={typed_rows}",
        f"languages={','.join(active_languages)}",
        f"workers={workers}",
    )
    return 0


def run_pass2(
    *,
    postgres_dsn: str,
    batch_size: int = 1000,
    worker_count: int | None = None,
) -> int:
    if batch_size <= 0:
        raise ValueError("--context-batch-size must be > 0")
    store = PostgresStore(postgres_dsn)
    store.ensure_schema()
    total_entities = store.count_entities()
    print(
        "Skipped Postgres pass2:",
        f"entities={total_entities}",
        "reason=context_string is built lazily from entity_triples",
        f"batch_size={batch_size}",
        f"workers={max(1, worker_count or min(8, (os.cpu_count() or 1)))}",
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Postgres ingestion for deterministic entity lookup (pass2 is a compatibility no-op)."
    )
    parser.add_argument("--dump-path", help=f"Input dump path. Overrides ${DUMP_PATH_ENV}.")
    parser.add_argument("--postgres-dsn", help="Postgres DSN.")
    parser.add_argument(
        "--mode",
        choices=("pass1", "pass2", "all"),
        default="all",
        help="Which ingestion phase(s) to run (pass2 is kept only for compatibility).",
    )
    parser.add_argument("--batch-size", type=parse_positive_int, default=5000)
    parser.add_argument(
        "--context-batch-size",
        type=parse_positive_int,
        default=1000,
        help="Deprecated compatibility option; pass2 no longer materializes context strings.",
    )
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
        help="Comma-separated language allowlist used for lexical typing inputs.",
    )
    parser.add_argument(
        "--max-aliases-per-language",
        type=parse_non_negative_int,
        default=8,
        help="Max aliases per language considered for lexical typing (stored aliases remain multilingual).",
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
