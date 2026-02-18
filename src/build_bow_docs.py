from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .common import (
    BOW_OUTPUT_PATH_ENV,
    DEFAULT_STOPWORDS,
    DUMP_PATH_ENV,
    LABELS_DB_PATH_ENV,
    NER_TYPES_PATH_ENV,
    build_name_text,
    ensure_existing_file,
    ensure_parent_dir,
    estimate_wikidata_entity_total,
    finalize_tqdm_total,
    is_supported_entity_id,
    iter_wikidata_entities,
    keep_tqdm_total_ahead,
    normalize_text,
    open_text_for_read,
    open_text_for_write,
    resolve_bow_output_path,
    resolve_dump_path,
    resolve_labels_db_path,
    resolve_ner_types_path,
    select_alias_map_languages,
    select_text_map_languages,
    tokenize,
    tqdm,
)

SELECT_LABELS_SQL = "SELECT labels_json FROM labels WHERE id = ?"
_VALID_NER_TYPE_RE = re.compile(r"^[A-Za-z0-9_.:/-]+$")
_MAX_SQL_IN_PARAMS = 900
_PREFERRED_LANGUAGES = ("en",)

DEFAULT_MAX_ALIASES_PER_LANGUAGE = 8
DEFAULT_MAX_BOW_TOKENS = 128
DEFAULT_MAX_CONTEXT_OBJECT_IDS = 32
DEFAULT_MAX_CONTEXT_CHARS = 640
DEFAULT_MAX_DOC_BYTES = 4096
DEFAULT_CONTEXT_LABEL_CACHE_SIZE = 200_000


def parse_non_negative_int(raw: str) -> int:
    try:
        parsed = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc

    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")

    return parsed


def parse_positive_int(raw: str) -> int:
    parsed = parse_non_negative_int(raw)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build JSONL docs by streaming Wikidata IDs and reading multilingual "
            "labels payloads from SQLite."
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
        "--labels-db-path",
        help=(
            f"Path to labels SQLite DB produced by build_labels_db.py. "
            f"Overrides ${LABELS_DB_PATH_ENV}."
        ),
    )
    parser.add_argument(
        "--output-path",
        help=(
            f"Output JSONL path (.jsonl/.jsonl.gz/.jsonl.bz2). "
            f"Overrides ${BOW_OUTPUT_PATH_ENV}."
        ),
    )
    parser.add_argument(
        "--ner-types-path",
        help=(
            "Optional NER type map file (.jsonl/.jsonl.gz) with records "
            "{\"id\":...,\"coarse_types\":[...],\"fine_types\":[...]}. "
            f"Overrides ${NER_TYPES_PATH_ENV}."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=parse_non_negative_int,
        default=5000,
        help="Lines per output write batch (default: 5000).",
    )
    parser.add_argument(
        "--limit",
        type=parse_non_negative_int,
        default=0,
        help="Max entities to parse for smoke runs (0 = no limit).",
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
    parser.add_argument(
        "--max-doc-bytes",
        type=parse_positive_int,
        default=DEFAULT_MAX_DOC_BYTES,
        help=f"Hard cap for serialized JSONL record bytes (default: {DEFAULT_MAX_DOC_BYTES}).",
    )
    parser.add_argument(
        "--context-label-cache-size",
        type=parse_positive_int,
        default=DEFAULT_CONTEXT_LABEL_CACHE_SIZE,
        help=(
            "LRU cache size for linked-object label lookups "
            f"(default: {DEFAULT_CONTEXT_LABEL_CACHE_SIZE})."
        ),
    )
    return parser.parse_args()


def decode_payload(raw_json: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, dict):
        return None

    return parsed


def to_text_map(raw: Any) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}

    output: dict[str, str] = {}
    for key, value in raw.items():
        if isinstance(key, str) and isinstance(value, str):
            output[key] = value

    return output


def to_alias_map(raw: Any) -> dict[str, list[str]]:
    if not isinstance(raw, dict):
        return {}

    output: dict[str, list[str]] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not isinstance(value, list):
            continue

        aliases = [entry for entry in value if isinstance(entry, str)]
        if aliases:
            output[key] = aliases

    return output


def pick_primary_type(values: Sequence[str]) -> str:
    return values[0] if values else ""


def build_english_name_text(
    labels: Mapping[str, str],
    aliases: Mapping[str, Sequence[str]],
) -> str:
    english_labels = {"en": labels["en"]} if "en" in labels else {}
    english_aliases = {"en": aliases["en"]} if "en" in aliases else {}

    if english_labels or english_aliases:
        return build_name_text(english_labels, english_aliases)

    return build_name_text(labels, aliases)


def _append_tokens_from_text(
    token_buffer: list[str],
    seen_tokens: set[str],
    text: str,
    *,
    stopwords: set[str],
    max_tokens: int,
) -> None:
    if len(token_buffer) >= max_tokens:
        return

    for token in tokenize(text):
        if len(token_buffer) >= max_tokens:
            return
        if len(token) <= 1 or token in stopwords:
            continue
        if token in seen_tokens:
            continue
        seen_tokens.add(token)
        token_buffer.append(token)


def _pick_preferred_label(labels: Mapping[str, str]) -> str:
    english = labels.get("en")
    if isinstance(english, str):
        candidate = normalize_text(english)
        if candidate:
            return candidate

    for language in sorted(labels):
        value = labels[language]
        candidate = normalize_text(value)
        if candidate:
            return candidate
    return ""


def _extract_entity_id_from_datavalue_value(raw_value: Any) -> str | None:
    if not isinstance(raw_value, Mapping):
        return None

    raw_id = raw_value.get("id")
    if isinstance(raw_id, str):
        candidate = raw_id.strip()
        if is_supported_entity_id(candidate):
            return candidate

    numeric_id = raw_value.get("numeric-id")
    if not isinstance(numeric_id, int) or numeric_id <= 0:
        return None

    entity_type = raw_value.get("entity-type")
    if entity_type == "item":
        return f"Q{numeric_id}"
    if entity_type == "property":
        return f"P{numeric_id}"
    return None


def extract_claim_object_ids(entity: Mapping[str, Any], *, limit: int) -> list[str]:
    if limit <= 0:
        return []

    claims = entity.get("claims")
    if not isinstance(claims, Mapping):
        return []

    object_ids: list[str] = []
    seen_ids: set[str] = set()

    for statements in claims.values():
        if not isinstance(statements, Sequence) or isinstance(statements, (str, bytes, bytearray)):
            continue

        for statement in statements:
            if len(object_ids) >= limit:
                return object_ids
            if not isinstance(statement, Mapping):
                continue

            mainsnak = statement.get("mainsnak")
            if not isinstance(mainsnak, Mapping):
                continue
            if mainsnak.get("snaktype") != "value":
                continue

            datavalue = mainsnak.get("datavalue")
            if not isinstance(datavalue, Mapping):
                continue

            object_id = _extract_entity_id_from_datavalue_value(datavalue.get("value"))
            if object_id is None or object_id in seen_ids:
                continue

            seen_ids.add(object_id)
            object_ids.append(object_id)

    return object_ids


class ObjectLabelResolver:
    def __init__(self, cursor: sqlite3.Cursor, *, cache_size: int) -> None:
        self._cursor = cursor
        self._cache_size = max(1, cache_size)
        self._cache: OrderedDict[str, str | None] = OrderedDict()

    def resolve(self, object_ids: Sequence[str]) -> list[str]:
        if not object_ids:
            return []

        labels: list[str] = []
        seen_labels: set[str] = set()
        missing_ids: list[str] = []

        for object_id in object_ids:
            if object_id in self._cache:
                cached = self._cache.pop(object_id)
                self._cache[object_id] = cached
                if isinstance(cached, str) and cached and cached not in seen_labels:
                    seen_labels.add(cached)
                    labels.append(cached)
                continue
            missing_ids.append(object_id)

        if missing_ids:
            fetched = self._fetch_labels(missing_ids)
            for object_id in missing_ids:
                label = fetched.get(object_id)
                self._cache[object_id] = label
                while len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)
                if isinstance(label, str) and label and label not in seen_labels:
                    seen_labels.add(label)
                    labels.append(label)

        return labels

    def _fetch_labels(self, object_ids: Sequence[str]) -> dict[str, str]:
        rows_by_id: dict[str, str] = {}
        start = 0
        total = len(object_ids)
        while start < total:
            batch = object_ids[start : start + _MAX_SQL_IN_PARAMS]
            start += len(batch)
            placeholders = ",".join("?" for _ in batch)
            sql = f"SELECT id, labels_json FROM labels WHERE id IN ({placeholders})"
            self._cursor.execute(sql, tuple(batch))
            for row in self._cursor.fetchall():
                entity_id = row[0] if len(row) > 0 else None
                raw_payload = row[1] if len(row) > 1 else None
                if not isinstance(entity_id, str) or not isinstance(raw_payload, str):
                    continue
                payload = decode_payload(raw_payload)
                if payload is None:
                    continue
                label = _pick_preferred_label(to_text_map(payload.get("labels")))
                if label:
                    rows_by_id[entity_id] = label

        return rows_by_id


def build_context_text(labels: Sequence[str], *, max_chars: int) -> str:
    if max_chars <= 0:
        return ""

    deduped: list[str] = []
    seen: set[str] = set()
    current_len = 0

    for raw_label in labels:
        label = normalize_text(raw_label)
        if not label or label in seen:
            continue

        extra = len(label) if not deduped else len(label) + 1
        if deduped and current_len + extra > max_chars:
            break
        if not deduped and len(label) > max_chars:
            deduped.append(label[:max_chars])
            break

        deduped.append(label)
        seen.add(label)
        current_len += extra

    return " ".join(deduped)


def build_entity_bow(
    labels: Mapping[str, str],
    aliases: Mapping[str, Sequence[str]],
    descriptions: Mapping[str, str],
    *,
    coarse_type: str,
    fine_type: str,
    max_tokens: int,
    stopwords: set[str] | None = None,
) -> str:
    active_stopwords = stopwords if stopwords is not None else DEFAULT_STOPWORDS
    tokens: list[str] = []
    seen_tokens: set[str] = set()

    english_label = labels.get("en")
    if isinstance(english_label, str) and english_label:
        _append_tokens_from_text(
            tokens,
            seen_tokens,
            english_label,
            stopwords=active_stopwords,
            max_tokens=max_tokens,
        )
    else:
        for language in sorted(labels):
            _append_tokens_from_text(
                tokens,
                seen_tokens,
                labels[language],
                stopwords=active_stopwords,
                max_tokens=max_tokens,
            )

    english_aliases = aliases.get("en")
    if isinstance(english_aliases, Sequence) and not isinstance(english_aliases, (str, bytes)):
        for alias in english_aliases:
            if isinstance(alias, str) and alias:
                _append_tokens_from_text(
                    tokens,
                    seen_tokens,
                    alias,
                    stopwords=active_stopwords,
                    max_tokens=max_tokens,
                )
    else:
        for language in sorted(aliases):
            for alias in aliases[language]:
                _append_tokens_from_text(
                    tokens,
                    seen_tokens,
                    alias,
                    stopwords=active_stopwords,
                    max_tokens=max_tokens,
                )

    english_description = descriptions.get("en")
    if isinstance(english_description, str) and english_description:
        _append_tokens_from_text(
            tokens,
            seen_tokens,
            english_description,
            stopwords=active_stopwords,
            max_tokens=max_tokens,
        )
    else:
        for language in sorted(descriptions):
            description = descriptions[language]
            _append_tokens_from_text(
                tokens,
                seen_tokens,
                description,
                stopwords=active_stopwords,
                max_tokens=max_tokens,
            )

    for type_value in (coarse_type, fine_type):
        normalized = normalize_text(type_value).strip()
        if not normalized:
            continue
        _append_tokens_from_text(
            tokens,
            seen_tokens,
            normalized,
            stopwords=active_stopwords,
            max_tokens=max_tokens,
        )

    return " ".join(tokens)


def normalize_ner_type_list(raw: Any, field_name: str, *, line_number: int) -> list[str]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError(f"NER types {field_name} at line {line_number} must be a JSON array.")

    normalized: list[str] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, str):
            raise ValueError(f"NER types {field_name} at line {line_number} must contain strings.")

        value = item.strip()
        if not value:
            continue

        if not _VALID_NER_TYPE_RE.match(value):
            raise ValueError(
                f"NER type '{item}' at line {line_number} has unsupported characters. "
                "Use only letters, digits, '_', '-', '.', ':', '/'."
            )

        if value not in seen:
            seen.add(value)
            normalized.append(value)

    return normalized


def load_ner_types_map(
    ner_types_path: Path,
) -> dict[str, tuple[list[str], list[str]]]:
    mapping: dict[str, tuple[list[str], list[str]]] = {}

    with open_text_for_read(ner_types_path) as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Could not parse NER types JSON at line {line_number} in '{ner_types_path}': "
                    f"{exc.msg}"
                ) from exc

            if not isinstance(parsed, dict):
                raise ValueError(
                    f"NER types record at line {line_number} in '{ner_types_path}' must be a JSON object."
                )

            entity_id = parsed.get("id")
            if not isinstance(entity_id, str) or not entity_id.strip():
                raise ValueError(
                    f"NER types record at line {line_number} in '{ner_types_path}' is missing valid string field 'id'."
                )
            entity_id = entity_id.strip()

            coarse_raw = parsed.get("coarse_types", parsed.get("ner_coarse_types"))
            fine_raw = parsed.get("fine_types", parsed.get("ner_fine_types"))

            coarse_types = normalize_ner_type_list(
                coarse_raw,
                "coarse_types",
                line_number=line_number,
            )
            fine_types = normalize_ner_type_list(
                fine_raw,
                "fine_types",
                line_number=line_number,
            )

            mapping[entity_id] = (coarse_types, fine_types)

    return mapping


def _serialize_record(record: Mapping[str, Any]) -> tuple[str, int]:
    serialized = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
    encoded_len = len(serialized.encode("utf-8"))
    return serialized, encoded_len


def _trim_text_field(value: str) -> str:
    tokens = value.split()
    if len(tokens) <= 1:
        return ""
    keep = max(1, int(len(tokens) * 0.75))
    return " ".join(tokens[:keep])


def _shrink_aliases(aliases: dict[str, list[str]]) -> bool:
    longest_language = ""
    longest_size = 0
    for language, values in aliases.items():
        if len(values) > longest_size:
            longest_size = len(values)
            longest_language = language

    if not longest_language or longest_size <= 0:
        return False

    aliases[longest_language].pop()
    if not aliases[longest_language]:
        del aliases[longest_language]
    return True


def enforce_max_record_bytes(
    record: Mapping[str, Any],
    *,
    max_doc_bytes: int,
) -> tuple[str, int, bool]:
    serialized, encoded_len = _serialize_record(record)
    if encoded_len <= max_doc_bytes:
        return serialized, encoded_len, False

    working: dict[str, Any] = dict(record)
    raw_aliases = working.get("aliases")
    aliases = {
        language: list(values)
        for language, values in raw_aliases.items()
        if isinstance(language, str) and isinstance(values, list)
    } if isinstance(raw_aliases, Mapping) else {}
    working["aliases"] = aliases

    changed = False
    for field_name in ("context", "bow"):
        while encoded_len > max_doc_bytes:
            raw_value = working.get(field_name)
            if not isinstance(raw_value, str) or not raw_value.strip():
                break
            trimmed = _trim_text_field(raw_value)
            if trimmed == raw_value:
                break
            working[field_name] = trimmed
            serialized, encoded_len = _serialize_record(working)
            changed = True
            if encoded_len <= max_doc_bytes:
                return serialized, encoded_len, changed

    while encoded_len > max_doc_bytes and _shrink_aliases(aliases):
        serialized, encoded_len = _serialize_record(working)
        changed = True
        if encoded_len <= max_doc_bytes:
            return serialized, encoded_len, changed

    if encoded_len > max_doc_bytes:
        entity_id = working.get("id")
        raise ValueError(
            f"Document for '{entity_id}' is {encoded_len} bytes, over --max-doc-bytes={max_doc_bytes}. "
            "Lower max aliases/context, increase max doc bytes, or further compact labels."
        )

    return serialized, encoded_len, changed


def run(
    dump_path: Path,
    labels_db_path: Path,
    output_path: Path,
    batch_size: int,
    limit: int,
    ner_types_path: Path | None,
    max_aliases_per_language: int = DEFAULT_MAX_ALIASES_PER_LANGUAGE,
    max_bow_tokens: int = DEFAULT_MAX_BOW_TOKENS,
    max_context_object_ids: int = DEFAULT_MAX_CONTEXT_OBJECT_IDS,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    max_doc_bytes: int = DEFAULT_MAX_DOC_BYTES,
    context_label_cache_size: int = DEFAULT_CONTEXT_LABEL_CACHE_SIZE,
) -> int:
    if batch_size == 0:
        raise ValueError("--batch-size must be >= 1")
    if max_bow_tokens <= 0:
        raise ValueError("--max-bow-tokens must be >= 1")
    if max_context_object_ids < 0:
        raise ValueError("--max-context-object-ids must be >= 0")
    if max_context_chars < 0:
        raise ValueError("--max-context-chars must be >= 0")
    if max_doc_bytes <= 0:
        raise ValueError("--max-doc-bytes must be >= 1")
    if context_label_cache_size <= 0:
        raise ValueError("--context-label-cache-size must be >= 1")

    ensure_existing_file(
        dump_path,
        "Wikidata dump",
        hint=f"Provide --dump-path or set ${DUMP_PATH_ENV}.",
    )
    ensure_existing_file(
        labels_db_path,
        "Labels SQLite DB",
        hint=(
            f"Run build_labels_db.py first or pass --labels-db-path / set ${LABELS_DB_PATH_ENV}."
        ),
    )

    ner_types_map: dict[str, tuple[list[str], list[str]]] = {}
    if ner_types_path is not None:
        ensure_existing_file(
            ner_types_path,
            "NER types map",
            hint=(
                "Provide --ner-types-path / set "
                f"${NER_TYPES_PATH_ENV} to a valid JSONL file."
            ),
        )
        ner_types_map = load_ner_types_map(ner_types_path)

    ensure_parent_dir(output_path)

    parsed_entities = 0
    skipped_non_qp = 0
    missing_in_db = 0
    written_docs = 0
    typed_docs = 0
    context_docs = 0
    trimmed_docs = 0
    total_doc_bytes = 0
    max_observed_doc_bytes = 0
    write_buffer: list[str] = []

    conn = sqlite3.connect(labels_db_path)
    try:
        cursor = conn.cursor()
        object_label_resolver = ObjectLabelResolver(
            cursor,
            cache_size=context_label_cache_size,
        )
        progress_total = estimate_wikidata_entity_total(
            dump_path,
            limit=None if limit == 0 else limit,
        )
        if progress_total is not None:
            print(f"Progress estimate: bow-docs total~{progress_total} entities")

        with open_text_for_write(output_path) as output_handle:
            with tqdm(total=progress_total, desc="bow-docs", unit="entity") as progress:
                for entity in iter_wikidata_entities(
                    dump_path,
                    limit=None if limit == 0 else limit,
                ):
                    parsed_entities += 1
                    progress.update(1)
                    keep_tqdm_total_ahead(progress)

                    entity_id = entity.get("id")
                    if not isinstance(entity_id, str) or not is_supported_entity_id(entity_id):
                        skipped_non_qp += 1
                        continue

                    cursor.execute(SELECT_LABELS_SQL, (entity_id,))
                    row = cursor.fetchone()
                    if row is None:
                        missing_in_db += 1
                        continue

                    raw_payload = row[0]
                    if not isinstance(raw_payload, str):
                        missing_in_db += 1
                        continue

                    payload = decode_payload(raw_payload)
                    if payload is None:
                        missing_in_db += 1
                        continue

                    labels = select_text_map_languages(
                        to_text_map(payload.get("labels")),
                        _PREFERRED_LANGUAGES,
                        fallback_to_any=True,
                    )
                    aliases = select_alias_map_languages(
                        to_alias_map(payload.get("aliases")),
                        _PREFERRED_LANGUAGES,
                        max_aliases_per_language=max_aliases_per_language,
                        fallback_to_any=False,
                    )
                    descriptions = select_text_map_languages(
                        to_text_map(payload.get("descriptions")),
                        _PREFERRED_LANGUAGES,
                        fallback_to_any=True,
                    )

                    payload_coarse_type = payload.get("coarse_type")
                    payload_fine_type = payload.get("fine_type")

                    coarse_type = (
                        normalize_text(payload_coarse_type)
                        if isinstance(payload_coarse_type, str)
                        else ""
                    )
                    fine_type = (
                        normalize_text(payload_fine_type)
                        if isinstance(payload_fine_type, str)
                        else ""
                    )

                    mapped_ner_types = ner_types_map.get(entity_id)
                    if mapped_ner_types is not None:
                        mapped_coarse_types, mapped_fine_types = mapped_ner_types
                        mapped_coarse_type = pick_primary_type(mapped_coarse_types)
                        mapped_fine_type = pick_primary_type(mapped_fine_types)
                        if mapped_coarse_type:
                            coarse_type = mapped_coarse_type
                        if mapped_fine_type:
                            fine_type = mapped_fine_type

                    if coarse_type or fine_type:
                        typed_docs += 1

                    name_text = build_english_name_text(labels, aliases)
                    claim_object_ids = extract_claim_object_ids(
                        entity,
                        limit=max_context_object_ids,
                    )
                    context_labels = object_label_resolver.resolve(claim_object_ids)
                    context_text = build_context_text(
                        context_labels,
                        max_chars=max_context_chars,
                    )
                    if context_text:
                        context_docs += 1

                    bow = build_entity_bow(
                        labels=labels,
                        aliases=aliases,
                        descriptions=descriptions,
                        coarse_type=coarse_type,
                        fine_type=fine_type,
                        max_tokens=max_bow_tokens,
                    )

                    record = {
                        "id": entity_id,
                        "labels": labels,
                        "aliases": aliases,
                        "name_text": name_text,
                        "context": context_text,
                        "bow": bow,
                        "coarse_type": coarse_type,
                        "fine_type": fine_type,
                    }
                    serialized_record, record_bytes, was_trimmed = enforce_max_record_bytes(
                        record,
                        max_doc_bytes=max_doc_bytes,
                    )
                    if was_trimmed:
                        trimmed_docs += 1
                    total_doc_bytes += record_bytes
                    max_observed_doc_bytes = max(max_observed_doc_bytes, record_bytes)

                    write_buffer.append(serialized_record + "\n")
                    written_docs += 1

                    if len(write_buffer) >= batch_size:
                        output_handle.writelines(write_buffer)
                        write_buffer.clear()
                        progress.set_postfix(written=written_docs, typed=typed_docs)

            if write_buffer:
                output_handle.writelines(write_buffer)
                write_buffer.clear()
            finalize_tqdm_total(progress)
    finally:
        conn.close()

    print(
        "Completed BOW docs build:",
        f"parsed={parsed_entities}",
        f"written={written_docs}",
        f"typed_docs={typed_docs}",
        f"context_docs={context_docs}",
        f"trimmed_docs={trimmed_docs}",
        f"avg_doc_bytes={(total_doc_bytes / written_docs):.1f}" if written_docs else "avg_doc_bytes=0.0",
        f"max_doc_bytes={max_observed_doc_bytes}",
        f"missing_labels={missing_in_db}",
        f"skipped_non_qp={skipped_non_qp}",
        f"max_bow_tokens={max_bow_tokens}",
        f"max_context_object_ids={max_context_object_ids}",
        f"max_context_chars={max_context_chars}",
        f"max_doc_bytes_cap={max_doc_bytes}",
        f"output={output_path}",
    )
    return 0


def main() -> int:
    args = parse_args()

    try:
        dump_path = resolve_dump_path(args.dump_path)
        labels_db_path = resolve_labels_db_path(args.labels_db_path)
        output_path = resolve_bow_output_path(args.output_path)
        ner_types_path = resolve_ner_types_path(args.ner_types_path)
        return run(
            dump_path=dump_path,
            labels_db_path=labels_db_path,
            output_path=output_path,
            batch_size=args.batch_size,
            limit=args.limit,
            ner_types_path=ner_types_path,
            max_aliases_per_language=args.max_aliases_per_language,
            max_bow_tokens=args.max_bow_tokens,
            max_context_object_ids=args.max_context_object_ids,
            max_context_chars=args.max_context_chars,
            max_doc_bytes=args.max_doc_bytes,
            context_label_cache_size=args.context_label_cache_size,
        )
    except (FileNotFoundError, ValueError, sqlite3.Error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
