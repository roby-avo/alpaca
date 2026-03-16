from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
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
from .wikidata_sample_ids import resolve_qids
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
DEFAULT_MAX_ENTITY_TRIPLES = 12
DEFAULT_MAX_ENTITY_TRIPLES_PER_PREDICATE = 2
DEFAULT_TRIPLE_PREDICATE_SCORE = 32
TRIPLE_PREDICATE_BASE_SCORES = {
    "P106": 95,  # occupation
    "P171": 94,  # parent taxon
    "P17": 92,   # country
    "P27": 92,   # country of citizenship
    "P131": 92,  # located in the administrative territorial entity
    "P159": 91,  # headquarters location
    "P495": 90,  # country of origin
    "P136": 90,  # genre
    "P452": 90,  # industry
    "P50": 89,   # author
    "P57": 89,   # director
    "P175": 88,  # performer
    "P176": 88,  # manufacturer
    "P178": 88,  # developer
    "P39": 88,   # position held
    "P105": 86,  # taxon rank
    "P112": 85,  # founded by
    "P127": 84,  # owned by
    "P279": 82,  # subclass of
    "P400": 82,  # platform
    "P30": 80,   # continent
    "P361": 80,  # part of
    "P69": 79,   # educated at
    "P108": 79,  # employer
    "P86": 78,   # composer
    "P123": 78,  # publisher
    "P264": 77,  # record label
    "P31": 76,   # instance of
    "P407": 76,  # language of work or name
    "P641": 75,  # sport
    "P749": 75,  # parent organization
    "P355": 74,  # subsidiary
    "P740": 74,  # location of formation
    "P276": 74,  # location
    "P102": 73,  # member of political party
    "P206": 72,  # located in or next to body of water
    "P170": 72,  # creator
    "P463": 70,  # member of
    "P710": 70,  # participant
    "P527": 68,  # has part(s)
    "P166": 54,  # award received
}
TRIPLE_SUBJECT_PROFILE_BONUSES = {
    "PERSON": {
        "P106": 18,
        "P39": 16,
        "P27": 14,
        "P69": 10,
        "P108": 10,
        "P102": 8,
        "P463": 6,
        "P31": -6,
    },
    "ORGANIZATION": {
        "P159": 16,
        "P452": 18,
        "P17": 10,
        "P131": 10,
        "P112": 12,
        "P127": 12,
        "P749": 12,
        "P355": 10,
        "P740": 8,
        "P463": 4,
        "P31": -4,
    },
    "COMPANY": {
        "P452": 8,
        "P159": 6,
        "P127": 4,
    },
    "LOCATION": {
        "P17": 20,
        "P131": 18,
        "P361": 14,
        "P30": 10,
        "P206": 8,
        "P31": -2,
    },
    "WORK": {
        "P136": 16,
        "P50": 14,
        "P57": 14,
        "P175": 14,
        "P170": 12,
        "P86": 12,
        "P123": 10,
        "P264": 10,
        "P400": 10,
        "P407": 8,
        "P495": 8,
        "P31": -4,
    },
    "FILM": {
        "P57": 8,
        "P175": 4,
    },
    "BOOK": {
        "P50": 8,
    },
    "MUSIC_WORK": {
        "P175": 8,
        "P86": 6,
        "P264": 4,
    },
    "SOFTWARE": {
        "P178": 18,
        "P400": 16,
        "P136": 8,
        "P31": -2,
    },
    "PRODUCT": {
        "P176": 18,
        "P178": 10,
        "P400": 10,
        "P495": 8,
    },
    "EVENT": {
        "P276": 16,
        "P17": 12,
        "P131": 12,
        "P710": 10,
        "P641": 10,
        "P361": 8,
        "P31": -4,
    },
    "BIOLOGICAL_TAXON": {
        "P171": 20,
        "P105": 16,
        "P31": 4,
    },
    "TYPE": {
        "P279": 12,
        "P31": 4,
    },
    "PREDICATE": {
        "P31": 8,
        "P279": 6,
    },
}
TRIPLE_RANK_BONUSES = {0: 8, 1: 0, 2: -12}
TRIPLE_PROPERTY_OBJECT_PENALTY = 28
TRIPLE_HUMAN_INSTANCE_PENALTY = 6
_build_entity_context_string = build_entity_context_string


@dataclass(frozen=True, slots=True)
class EntityParseContext:
    entity_id: str
    labels: Mapping[str, str]
    aliases: Mapping[str, Sequence[str]]
    descriptions: Mapping[str, str]
    label: str
    description: str | None
    types: Sequence[str]
    item_category: str
    coarse_type: str
    fine_type: str
    popularity: float
    cross_refs: Mapping[str, Any]


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


def _statement_rank_priority(statement: Mapping[str, Any]) -> int:
    rank = statement.get("rank")
    if rank == "preferred":
        return 0
    if rank == "deprecated":
        return 2
    return 1


def _build_triple_subject_profiles(
    entity: Mapping[str, Any],
    *,
    subject_types: Sequence[str],
    item_category: str,
    coarse_type: str,
    fine_type: str,
) -> frozenset[str]:
    profiles: set[str] = set()

    normalized_item_category = item_category.strip().upper() if isinstance(item_category, str) else ""
    normalized_coarse_type = coarse_type.strip().upper() if isinstance(coarse_type, str) else ""
    normalized_fine_type = fine_type.strip().upper() if isinstance(fine_type, str) else ""

    if normalized_item_category:
        profiles.add(normalized_item_category)
    if normalized_coarse_type:
        profiles.add(normalized_coarse_type)
    if normalized_fine_type:
        profiles.add(normalized_fine_type)

    type_qids = {
        qid.strip()
        for qid in subject_types
        if isinstance(qid, str) and qid.strip()
    }
    if not type_qids:
        type_qids.update(_claim_object_ids_for_property(entity, "P31", limit=16))
    if HUMAN_QID in type_qids:
        profiles.add("PERSON")

    if not normalized_item_category:
        inferred_item_category = infer_item_category(entity)
        if inferred_item_category:
            profiles.add(inferred_item_category.strip().upper())

    if normalized_fine_type in {"COMPANY", "NONPROFIT_ORG", "GOVERNMENT_ORG", "EDUCATIONAL_ORG", "SPORTS_TEAM"}:
        profiles.add("ORGANIZATION")
    if normalized_fine_type in {"COUNTRY", "CITY", "REGION", "LANDMARK", "CELESTIAL_BODY"}:
        profiles.add("LOCATION")
    if normalized_fine_type in {"FILM", "BOOK", "MUSIC_WORK", "SOFTWARE", "INTERNET_MEME"}:
        profiles.add("WORK")
    if normalized_fine_type in {"DEVICE", "MEDICATION", "FOOD_BEVERAGE", "PRODUCT_GENERIC"}:
        profiles.add("PRODUCT")
    if normalized_fine_type in {"CONFLICT", "SPORT_EVENT", "EVENT_GENERIC"}:
        profiles.add("EVENT")

    if normalized_fine_type == "BIOLOGICAL_TAXON" or _claim_object_ids_for_property(entity, "P171", limit=1):
        profiles.add("BIOLOGICAL_TAXON")

    return frozenset(profiles)


def _triple_candidate_score(
    *,
    subject_qid: str,
    predicate_pid: str,
    object_qid: str,
    statement_rank_priority: int,
    subject_profiles: frozenset[str],
) -> int:
    # Score edges by likely disambiguation value because context strings only use object labels.
    score = TRIPLE_PREDICATE_BASE_SCORES.get(predicate_pid, DEFAULT_TRIPLE_PREDICATE_SCORE)
    score += TRIPLE_RANK_BONUSES.get(statement_rank_priority, 0)

    for profile in subject_profiles:
        score += TRIPLE_SUBJECT_PROFILE_BONUSES.get(profile, {}).get(predicate_pid, 0)

    if object_qid == subject_qid:
        return -10_000
    if object_qid.startswith("P"):
        score -= TRIPLE_PROPERTY_OBJECT_PENALTY
    if predicate_pid == "P31" and object_qid == HUMAN_QID:
        score -= TRIPLE_HUMAN_INSTANCE_PENALTY
    return score


def _predicate_priority(
    predicate_pid: str,
    predicate_candidates: Sequence[tuple[int, int, str]],
) -> tuple[int, int, str]:
    if not predicate_candidates:
        return (0, 0, predicate_pid)
    best_score = max(candidate[0] for candidate in predicate_candidates)
    return (-best_score, -len(predicate_candidates), predicate_pid)


def extract_entity_triples(
    entity: Mapping[str, Any],
    *,
    max_triples: int = DEFAULT_MAX_ENTITY_TRIPLES,
    max_triples_per_predicate: int = DEFAULT_MAX_ENTITY_TRIPLES_PER_PREDICATE,
    parse_context: EntityParseContext | None = None,
) -> list[EntityTripleRecord]:
    entity_id = entity.get("id")
    claims = entity.get("claims")
    if not isinstance(entity_id, str) or not is_supported_entity_id(entity_id):
        return []
    if not isinstance(claims, Mapping):
        return []

    subject_profiles = _build_triple_subject_profiles(
        entity,
        subject_types=parse_context.types if parse_context is not None else (),
        item_category=parse_context.item_category if parse_context is not None else "",
        coarse_type=parse_context.coarse_type if parse_context is not None else "",
        fine_type=parse_context.fine_type if parse_context is not None else "",
    )
    best_by_triple: dict[tuple[str, str, str], tuple[int, int, str]] = {}
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
            if object_id == entity_id:
                continue
            statement_rank_priority = _statement_rank_priority(statement)
            triple_key = (entity_id, predicate_pid, object_id)
            candidate = (
                _triple_candidate_score(
                    subject_qid=entity_id,
                    predicate_pid=predicate_pid,
                    object_qid=object_id,
                    statement_rank_priority=statement_rank_priority,
                    subject_profiles=subject_profiles,
                ),
                statement_rank_priority,
                object_id,
            )
            previous = best_by_triple.get(triple_key)
            if previous is None or candidate[0] > previous[0] or (
                candidate[0] == previous[0] and candidate[1] < previous[1]
            ):
                best_by_triple[triple_key] = candidate

    candidates_by_predicate: dict[str, list[tuple[int, int, str]]] = {}
    for (_subject_qid, predicate_pid, _object_qid), candidate in best_by_triple.items():
        candidates_by_predicate.setdefault(predicate_pid, []).append(candidate)

    ordered_predicates = sorted(
        candidates_by_predicate,
        key=lambda predicate_pid: _predicate_priority(predicate_pid, candidates_by_predicate[predicate_pid]),
    )
    ordered_objects_by_predicate = {
        predicate_pid: [
            object_id
            for _score, _rank_priority, object_id in sorted(
                candidates_by_predicate[predicate_pid],
                key=lambda item: (-item[0], item[1], item[2]),
            )
        ]
        for predicate_pid in ordered_predicates
    }

    triples: list[EntityTripleRecord] = []
    next_index_by_predicate = {predicate_pid: 0 for predicate_pid in ordered_predicates}
    while ordered_predicates:
        added = False
        for predicate_pid in ordered_predicates:
            object_qids = ordered_objects_by_predicate[predicate_pid]
            per_predicate_limit = len(object_qids)
            if max_triples_per_predicate > 0:
                per_predicate_limit = min(per_predicate_limit, max_triples_per_predicate)
            next_index = next_index_by_predicate[predicate_pid]
            if next_index >= per_predicate_limit:
                continue
            triples.append(
                EntityTripleRecord(
                    subject_qid=entity_id,
                    predicate_pid=predicate_pid,
                    object_qid=object_qids[next_index],
                )
            )
            next_index_by_predicate[predicate_pid] = next_index + 1
            added = True
            if max_triples > 0 and len(triples) >= max_triples:
                return triples
        if not added:
            break
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


def _build_entity_parse_context(
    entity: Mapping[str, Any],
    *,
    language_allowlist: Sequence[str],
    max_aliases_per_language: int,
    disable_ner_classifier: bool,
) -> EntityParseContext | None:
    entity_id = entity.get("id")
    if not isinstance(entity_id, str) or not is_supported_entity_id(entity_id):
        return None

    payload = extract_multilingual_payload(entity)
    all_labels = payload.get("labels", {})
    all_aliases = payload.get("aliases", {})
    all_descriptions = payload.get("descriptions", {})

    label = _pick_primary_label(all_labels)
    if not label:
        return None

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
    descriptions_for_typing = select_text_map_languages(
        all_descriptions,
        language_allowlist,
        fallback_to_any=True,
    )

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
            descriptions=descriptions_for_typing,
            claims=entity.get("claims") if isinstance(entity.get("claims"), Mapping) else None,
        )
        coarse_type = coarse_types[0] if coarse_types else ""
        fine_type = fine_types[0] if fine_types else ""

    return EntityParseContext(
        entity_id=entity_id,
        labels=all_labels,
        aliases=all_aliases,
        descriptions=all_descriptions,
        label=label,
        description=description,
        types=_extract_entity_type_qids(entity),
        item_category=infer_item_category(entity),
        coarse_type=coarse_type,
        fine_type=fine_type,
        popularity=_extract_popularity(entity),
        cross_refs=_extract_cross_refs(
            entity,
            preferred_language=_pick_primary_label_language(all_labels),
        ),
    )


def _entity_record_from_parse_context(parse_context: EntityParseContext) -> EntityRecord:
    return EntityRecord(
        qid=parse_context.entity_id,
        label=parse_context.label,
        labels=parse_context.labels,
        aliases=parse_context.aliases,
        description=parse_context.description,
        types=parse_context.types,
        item_category=parse_context.item_category,
        coarse_type=parse_context.coarse_type,
        fine_type=parse_context.fine_type,
        popularity=parse_context.popularity,
        cross_refs=parse_context.cross_refs,
    )


def transform_entity_to_record(
    entity: Mapping[str, Any],
    *,
    language_allowlist: Sequence[str],
    max_aliases_per_language: int,
    disable_ner_classifier: bool,
) -> EntityRecord | None:
    parse_context = _build_entity_parse_context(
        entity,
        language_allowlist=language_allowlist,
        max_aliases_per_language=max_aliases_per_language,
        disable_ner_classifier=disable_ner_classifier,
    )
    if parse_context is None:
        return None
    return _entity_record_from_parse_context(parse_context)


def _flush_transform_batch(
    store: PostgresStore,
    raw_entities: list[dict[str, Any]],
    *,
    executor: ThreadPoolExecutor | None,
    language_allowlist: Sequence[str],
    max_aliases_per_language: int,
    disable_ner_classifier: bool,
    max_entity_triples: int,
    max_entity_triples_per_predicate: int,
) -> tuple[int, int, int]:
    if not raw_entities:
        return 0, 0, 0

    typed_rows = 0
    records: list[EntityRecord] = []
    triples: list[EntityTripleRecord] = []

    def _transform(entity: Mapping[str, Any]) -> tuple[EntityRecord | None, list[EntityTripleRecord]]:
        parse_context = _build_entity_parse_context(
            entity,
            language_allowlist=language_allowlist,
            max_aliases_per_language=max_aliases_per_language,
            disable_ner_classifier=disable_ner_classifier,
        )
        record = _entity_record_from_parse_context(parse_context) if parse_context is not None else None
        return (
            record,
            extract_entity_triples(
                entity,
                max_triples=max_entity_triples,
                max_triples_per_predicate=max_entity_triples_per_predicate,
                parse_context=parse_context,
            ),
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


def _resolve_sample_cache_qids(
    store: PostgresStore,
    *,
    sample_cache_ids: str | None,
    sample_cache_ids_file: str | None,
    sample_cache_count: int | None,
    limit: int,
) -> list[str]:
    selector_count = (
        sum(1 for value in (sample_cache_ids, sample_cache_ids_file) if value)
        + (1 if sample_cache_count is not None else 0)
    )
    if selector_count != 1:
        raise ValueError(
            "Provide exactly one of --sample-cache-ids, --sample-cache-ids-file, or --sample-cache-count."
        )

    if sample_cache_count is not None:
        qids = store.list_sample_entity_ids(limit=int(sample_cache_count))
        if len(qids) < int(sample_cache_count):
            raise ValueError(
                f"Requested --sample-cache-count {sample_cache_count}, but sample_entity_cache has only {len(qids)} QIDs."
            )
    else:
        qids = resolve_qids(sample_cache_ids, sample_cache_ids_file, None)

    if limit > 0:
        qids = qids[:limit]
    if not qids:
        raise ValueError("No sample-cache QIDs were selected.")
    return qids


def run_pass1(
    *,
    dump_path: Path | None,
    postgres_dsn: str,
    batch_size: int,
    limit: int,
    language_allowlist: Sequence[str] | None = None,
    max_aliases_per_language: int = 8,
    disable_ner_classifier: bool = False,
    max_entity_triples: int = DEFAULT_MAX_ENTITY_TRIPLES,
    max_entity_triples_per_predicate: int = DEFAULT_MAX_ENTITY_TRIPLES_PER_PREDICATE,
    sample_cache_ids: str | None = None,
    sample_cache_ids_file: str | None = None,
    sample_cache_count: int | None = None,
    worker_count: int | None = None,
    expected_entity_total: int | None = None,
) -> int:
    if batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if max_aliases_per_language < 0:
        raise ValueError("--max-aliases-per-language must be >= 0")
    if max_entity_triples < 0:
        raise ValueError("--max-entity-triples must be >= 0")
    if max_entity_triples_per_predicate < 0:
        raise ValueError("--max-entity-triples-per-predicate must be >= 0")
    if expected_entity_total is not None and expected_entity_total <= 0:
        raise ValueError("--expected-entity-total must be > 0 when provided")

    active_languages = tuple(language_allowlist) if language_allowlist else ("en",)
    workers = max(1, worker_count or min(8, (os.cpu_count() or 1)))
    store = PostgresStore(postgres_dsn)
    store.ensure_schema()

    sample_cache_selector_count = (
        sum(1 for value in (sample_cache_ids, sample_cache_ids_file) if value)
        + (1 if sample_cache_count is not None else 0)
    )
    use_sample_cache = sample_cache_selector_count > 0
    if use_sample_cache and dump_path is not None:
        raise ValueError("Use either --dump-path or --sample-cache-* selectors, not both.")
    if not use_sample_cache and dump_path is None:
        raise ValueError("Provide --dump-path or one of the --sample-cache-* selectors.")

    selected_sample_qids: list[str] = []
    if use_sample_cache:
        selected_sample_qids = _resolve_sample_cache_qids(
            store,
            sample_cache_ids=sample_cache_ids,
            sample_cache_ids_file=sample_cache_ids_file,
            sample_cache_count=sample_cache_count,
            limit=limit,
        )
    else:
        assert dump_path is not None
        ensure_existing_file(
            dump_path,
            "Wikidata dump",
            hint=f"Provide --dump-path or set ${DUMP_PATH_ENV}.",
        )

    parsed_entities = 0
    stored_rows = 0
    typed_rows = 0
    stored_triples = 0
    pending_entities: list[dict[str, Any]] = []

    if expected_entity_total is not None:
        progress_total = int(expected_entity_total)
        if use_sample_cache:
            progress_total = min(progress_total, len(selected_sample_qids))
        elif limit > 0:
            progress_total = min(progress_total, limit)
        print(
            f"Progress estimate: pg-pass1 total~{progress_total} entities "
            f"({'sample-cache manual override' if use_sample_cache else 'manual override'})"
        )
    elif use_sample_cache:
        progress_total = len(selected_sample_qids)
        print(f"Progress estimate: pg-pass1 total~{progress_total} entities (sample-cache exact)")
    else:
        assert dump_path is not None
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
            entity_iter = (
                store.iter_sample_entities(selected_sample_qids, batch_size=batch_size)
                if use_sample_cache
                else iter_wikidata_entities(
                    dump_path,
                    limit=None if limit == 0 else limit,
                )
            )
            for entity in entity_iter:
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
                        max_entity_triples=max_entity_triples,
                        max_entity_triples_per_predicate=max_entity_triples_per_predicate,
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
                max_entity_triples=max_entity_triples,
                max_entity_triples_per_predicate=max_entity_triples_per_predicate,
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
        f"max_entity_triples={max_entity_triples}",
        f"max_entity_triples_per_predicate={max_entity_triples_per_predicate}",
        f"source={'sample_cache' if use_sample_cache else 'dump'}",
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
        description=(
            "Postgres ingestion for deterministic entity lookup "
            "(pass2 is a compatibility no-op; input can be a dump or sample_entity_cache)."
        )
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
    parser.add_argument(
        "--max-entity-triples",
        type=parse_non_negative_int,
        default=DEFAULT_MAX_ENTITY_TRIPLES,
        help=(
            "Per-entity cap for stored entity_triples after heuristic pruning "
            f"(default: {DEFAULT_MAX_ENTITY_TRIPLES}, 0 keeps all deduped edges)."
        ),
    )
    parser.add_argument(
        "--max-entity-triples-per-predicate",
        type=parse_non_negative_int,
        default=DEFAULT_MAX_ENTITY_TRIPLES_PER_PREDICATE,
        help=(
            "Per-predicate cap applied before the overall entity_triples cap "
            f"(default: {DEFAULT_MAX_ENTITY_TRIPLES_PER_PREDICATE}, 0 keeps all edges per predicate)."
        ),
    )
    parser.add_argument(
        "--sample-cache-ids",
        help="Comma-separated QIDs from Postgres sample_entity_cache (example: Q42,Q90,Q64).",
    )
    parser.add_argument(
        "--sample-cache-ids-file",
        help="Text file with one QID per line for Postgres sample_entity_cache.",
    )
    parser.add_argument(
        "--sample-cache-count",
        type=parse_positive_int,
        help="Read the first N cached sample QIDs by numeric order from sample_entity_cache.",
    )
    parser.add_argument("--disable-ner-classifier", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        postgres_dsn = resolve_postgres_dsn(args.postgres_dsn)
        sample_cache_selector_count = (
            sum(1 for value in (args.sample_cache_ids, args.sample_cache_ids_file) if value)
            + (1 if args.sample_cache_count is not None else 0)
        )
        dump_path: Path | None = None
        if sample_cache_selector_count == 0:
            dump_path = resolve_dump_path(args.dump_path)
        elif args.dump_path:
            raise ValueError("Use either --dump-path or --sample-cache-* selectors, not both.")
        if args.mode in {"pass1", "all"}:
            language_allowlist = parse_language_allowlist(args.languages, arg_name="--languages")
            status = run_pass1(
                dump_path=dump_path,
                postgres_dsn=postgres_dsn,
                batch_size=args.batch_size,
                limit=args.limit,
                language_allowlist=language_allowlist,
                max_aliases_per_language=args.max_aliases_per_language,
                disable_ner_classifier=args.disable_ner_classifier,
                max_entity_triples=args.max_entity_triples,
                max_entity_triples_per_predicate=args.max_entity_triples_per_predicate,
                sample_cache_ids=args.sample_cache_ids,
                sample_cache_ids_file=args.sample_cache_ids_file,
                sample_cache_count=args.sample_cache_count,
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
