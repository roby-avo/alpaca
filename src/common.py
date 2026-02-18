from __future__ import annotations

import bz2
import gzip
import io
import json
import os
import re
from contextlib import ExitStack
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, TextIO

DUMP_PATH_ENV = "ALPACA_DUMP_PATH"
LABELS_DB_PATH_ENV = "ALPACA_LABELS_DB_PATH"
BOW_OUTPUT_PATH_ENV = "ALPACA_BOW_OUTPUT_PATH"
NER_TYPES_PATH_ENV = "ALPACA_NER_TYPES_PATH"
QUICKWIT_URL_ENV = "ALPACA_QUICKWIT_URL"
QUICKWIT_INDEX_ID_ENV = "ALPACA_QUICKWIT_INDEX_ID"
QUICKWIT_INDEX_CONFIG_PATH_ENV = "ALPACA_QUICKWIT_INDEX_CONFIG_PATH"

DEFAULT_DUMP_PRIMARY = Path("/downloads/latest-all.json.bz2")
DEFAULT_DUMP_SECONDARY = Path("./data/input/latest-all.json.bz2")
DEFAULT_LABELS_DB_PATH = Path("./data/output/wikidata_labels.sqlite")
DEFAULT_BOW_OUTPUT_PATH = Path("./data/output/bow_docs.jsonl.gz")
DEFAULT_QUICKWIT_URL = "http://localhost:7280"
DEFAULT_QUICKWIT_INDEX_ID = "wikidata_entities"
DEFAULT_QUICKWIT_INDEX_CONFIG_PATH = Path("./quickwit/wikidata-entities-index-config.json")

_WHITESPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)
_LANG_CODE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9-]{0,15}$")
_ESTIMATE_SAMPLE_RECORDS = 20_000
_ESTIMATE_MAX_SAMPLE_TEXT_BYTES = 64_000_000

DEFAULT_STOPWORDS: set[str] = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "das",
    "de",
    "del",
    "der",
    "des",
    "di",
    "die",
    "du",
    "e",
    "el",
    "en",
    "ein",
    "eine",
    "et",
    "for",
    "from",
    "gli",
    "i",
    "il",
    "in",
    "is",
    "la",
    "las",
    "le",
    "les",
    "lo",
    "los",
    "of",
    "on",
    "or",
    "per",
    "the",
    "to",
    "un",
    "una",
    "und",
    "uno",
    "von",
    "with",
    "y",
    "zu",
}

try:
    from tqdm import tqdm  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    class _NoopTqdm:
        def __init__(self, *_: Any, **__: Any) -> None:
            self.n = 0

        def __enter__(self) -> "_NoopTqdm":
            return self

        def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            return None

        def update(self, n: int = 1) -> None:
            self.n += n

        def set_postfix(self, *_: Any, **__: Any) -> None:
            return None

        def close(self) -> None:
            return None

    def tqdm(*args: Any, **kwargs: Any) -> _NoopTqdm:  # type: ignore
        _ = args
        _ = kwargs
        return _NoopTqdm()


def default_dump_path() -> Path:
    if DEFAULT_DUMP_PRIMARY.is_file():
        return DEFAULT_DUMP_PRIMARY
    return DEFAULT_DUMP_SECONDARY


def resolve_dump_path(cli_value: str | None) -> Path:
    return resolve_configured_path(cli_value, DUMP_PATH_ENV, default_dump_path())


def resolve_labels_db_path(cli_value: str | None) -> Path:
    return resolve_configured_path(cli_value, LABELS_DB_PATH_ENV, DEFAULT_LABELS_DB_PATH)


def resolve_bow_output_path(cli_value: str | None) -> Path:
    return resolve_configured_path(cli_value, BOW_OUTPUT_PATH_ENV, DEFAULT_BOW_OUTPUT_PATH)


def resolve_ner_types_path(cli_value: str | None) -> Path | None:
    return resolve_optional_path(cli_value, NER_TYPES_PATH_ENV)


def resolve_quickwit_index_config_path(cli_value: str | None) -> Path:
    return resolve_configured_path(
        cli_value,
        QUICKWIT_INDEX_CONFIG_PATH_ENV,
        DEFAULT_QUICKWIT_INDEX_CONFIG_PATH,
    )


def resolve_quickwit_url(cli_value: str | None) -> str:
    return resolve_configured_str(cli_value, QUICKWIT_URL_ENV, DEFAULT_QUICKWIT_URL)


def resolve_quickwit_index_id(cli_value: str | None) -> str:
    return resolve_configured_str(cli_value, QUICKWIT_INDEX_ID_ENV, DEFAULT_QUICKWIT_INDEX_ID)


def resolve_configured_path(
    cli_value: str | None,
    env_var: str,
    default_path: Path,
) -> Path:
    if cli_value:
        return Path(cli_value).expanduser()

    env_value = os.getenv(env_var)
    if env_value:
        return Path(env_value).expanduser()

    return default_path


def resolve_optional_path(cli_value: str | None, env_var: str) -> Path | None:
    if cli_value is not None:
        stripped = cli_value.strip()
        return Path(stripped).expanduser() if stripped else None

    env_value = os.getenv(env_var)
    if env_value is not None:
        stripped = env_value.strip()
        return Path(stripped).expanduser() if stripped else None

    return None


def resolve_configured_str(cli_value: str | None, env_var: str, default_value: str) -> str:
    if cli_value:
        return cli_value.strip()

    env_value = os.getenv(env_var)
    if env_value:
        return env_value.strip()

    return default_value


def ensure_existing_file(path: Path, label: str, *, hint: str | None = None) -> None:
    if not path.exists():
        message = f"{label} was not found at '{path}'."
        if hint:
            message = f"{message} {hint}"
        raise FileNotFoundError(message)

    if not path.is_file():
        raise FileNotFoundError(f"{label} path '{path}' exists but is not a regular file.")


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def open_text_for_read(path: Path) -> TextIO:
    suffix = path.suffix.lower()
    if suffix == ".bz2":
        return bz2.open(path, mode="rt", encoding="utf-8")
    if suffix == ".gz":
        return gzip.open(path, mode="rt", encoding="utf-8")
    return path.open(mode="rt", encoding="utf-8")


def open_text_for_write(path: Path) -> TextIO:
    suffix = path.suffix.lower()
    if suffix == ".bz2":
        return bz2.open(path, mode="wt", encoding="utf-8")
    if suffix == ".gz":
        return gzip.open(path, mode="wt", encoding="utf-8")
    return path.open(mode="wt", encoding="utf-8")


def iter_wikidata_entities(dump_path: Path, limit: int | None = None) -> Iterator[dict[str, Any]]:
    emitted = 0
    with open_text_for_read(dump_path) as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            cleaned = _clean_dump_line(raw_line)
            if cleaned is None:
                continue

            try:
                parsed = json.loads(cleaned)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Could not parse JSON at line {line_number} in '{dump_path}': {exc.msg}"
                ) from exc

            if not isinstance(parsed, dict):
                continue

            yield parsed
            emitted += 1
            if limit is not None and emitted >= limit:
                return


def _clean_dump_line(raw_line: str) -> str | None:
    line = raw_line.strip()
    if not line or line == "[" or line == "]":
        return None
    if line.endswith(","):
        line = line[:-1]
    return line or None


def _clean_jsonl_line(raw_line: str) -> str | None:
    line = raw_line.strip()
    return line or None


def _estimate_total_records_from_sample(
    path: Path,
    *,
    normalize_line: Callable[[str], str | None],
    limit: int | None = None,
) -> int | None:
    if not path.is_file():
        return None

    if limit is not None and limit <= 0:
        return None

    file_size_bytes = path.stat().st_size
    if file_size_bytes <= 0:
        return None

    sampled_records = 0
    sampled_text_bytes = 0
    sampled_compressed_bytes = 0
    reached_eof = True

    with ExitStack() as stack:
        raw_handle = stack.enter_context(path.open(mode="rb"))
        suffix = path.suffix.lower()
        if suffix == ".bz2":
            binary_handle = stack.enter_context(bz2.BZ2File(raw_handle, mode="rb"))
        elif suffix == ".gz":
            binary_handle = stack.enter_context(gzip.GzipFile(fileobj=raw_handle, mode="rb"))
        else:
            binary_handle = raw_handle
        handle = stack.enter_context(io.TextIOWrapper(binary_handle, encoding="utf-8"))

        for raw_line in handle:
            sampled_text_bytes += len(raw_line.encode("utf-8"))
            sampled_compressed_bytes = raw_handle.tell()

            cleaned = normalize_line(raw_line)
            if cleaned is not None:
                sampled_records += 1

            if (
                sampled_records >= _ESTIMATE_SAMPLE_RECORDS
                or sampled_text_bytes >= _ESTIMATE_MAX_SAMPLE_TEXT_BYTES
            ):
                reached_eof = False
                break

    if sampled_records <= 0 or sampled_text_bytes <= 0:
        return None

    if reached_eof:
        estimated_records = sampled_records
    else:
        average_record_compressed_bytes = sampled_compressed_bytes / sampled_records
        if average_record_compressed_bytes <= 0:
            average_record_text_bytes = sampled_text_bytes / sampled_records
            if average_record_text_bytes <= 0:
                return None
            estimated_records = int(file_size_bytes / average_record_text_bytes)
        else:
            estimated_records = int(file_size_bytes / average_record_compressed_bytes)

        estimated_records = max(sampled_records, estimated_records, 1)

    if limit is not None:
        estimated_records = min(estimated_records, limit)

    return estimated_records


def estimate_wikidata_entity_total(dump_path: Path, *, limit: int | None = None) -> int | None:
    return _estimate_total_records_from_sample(
        dump_path,
        normalize_line=_clean_dump_line,
        limit=limit,
    )


def estimate_jsonl_record_total(path: Path, *, limit: int | None = None) -> int | None:
    return _estimate_total_records_from_sample(
        path,
        normalize_line=_clean_jsonl_line,
        limit=limit,
    )


def keep_tqdm_total_ahead(progress: Any, *, min_extra: int = 100) -> None:
    total = getattr(progress, "total", None)
    n = getattr(progress, "n", None)
    if not isinstance(total, (int, float)) or not isinstance(n, (int, float)):
        return

    if n <= total:
        return

    extra = max(min_extra, int(max(1, n * 0.05)))
    try:
        progress.total = int(n) + extra
    except Exception:
        return

    refresh = getattr(progress, "refresh", None)
    if callable(refresh):
        refresh()


def finalize_tqdm_total(progress: Any) -> None:
    total = getattr(progress, "total", None)
    n = getattr(progress, "n", None)
    if not isinstance(total, (int, float)) or not isinstance(n, (int, float)):
        return

    if int(total) == int(n):
        return

    try:
        progress.total = int(n)
    except Exception:
        return

    refresh = getattr(progress, "refresh", None)
    if callable(refresh):
        refresh()


def is_supported_entity_id(entity_id: str) -> bool:
    return entity_id.startswith("Q") or entity_id.startswith("P")


def parse_language_allowlist(raw: str, *, arg_name: str) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for token in raw.split(","):
        language = token.strip().casefold()
        if not language:
            continue
        if not _LANG_CODE_RE.match(language):
            raise ValueError(
                f"{arg_name} contains invalid language code '{token}'. "
                "Use comma-separated values like 'en,it,fr'."
            )
        if language in seen:
            continue
        seen.add(language)
        normalized.append(language)

    if not normalized:
        raise ValueError(f"{arg_name} must contain at least one language code.")

    return normalized


def _first_non_empty_text(value_map: Mapping[str, str]) -> tuple[str, str] | None:
    for language in sorted(value_map):
        value = normalize_text(value_map[language])
        if value:
            return language, value
    return None


def select_text_map_languages(
    value_map: Mapping[str, str],
    preferred_languages: Sequence[str],
    *,
    fallback_to_any: bool,
) -> dict[str, str]:
    if not value_map:
        return {}

    selected: dict[str, str] = {}
    for language in preferred_languages:
        value = value_map.get(language)
        if not isinstance(value, str):
            continue
        normalized = normalize_text(value)
        if normalized:
            selected[language] = normalized

    if selected:
        return selected

    if not fallback_to_any:
        return {}

    fallback = _first_non_empty_text(value_map)
    if fallback is None:
        return {}
    language, value = fallback
    return {language: value}


def select_alias_map_languages(
    alias_map: Mapping[str, Sequence[str]],
    preferred_languages: Sequence[str],
    *,
    max_aliases_per_language: int,
    fallback_to_any: bool,
) -> dict[str, list[str]]:
    if not alias_map or max_aliases_per_language <= 0:
        return {}

    selected: dict[str, list[str]] = {}
    for language in preferred_languages:
        aliases = alias_map.get(language)
        if not isinstance(aliases, Sequence) or isinstance(aliases, (str, bytes, bytearray)):
            continue

        compacted: list[str] = []
        seen: set[str] = set()
        for raw_alias in aliases:
            if not isinstance(raw_alias, str):
                continue
            candidate = normalize_text(raw_alias)
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            compacted.append(candidate)
            if len(compacted) >= max_aliases_per_language:
                break

        if compacted:
            selected[language] = compacted

    if selected:
        return selected

    if not fallback_to_any:
        return {}

    for language in sorted(alias_map):
        aliases = alias_map[language]
        if not isinstance(aliases, Sequence) or isinstance(aliases, (str, bytes, bytearray)):
            continue
        compacted: list[str] = []
        seen: set[str] = set()
        for raw_alias in aliases:
            if not isinstance(raw_alias, str):
                continue
            candidate = normalize_text(raw_alias)
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            compacted.append(candidate)
            if len(compacted) >= max_aliases_per_language:
                break
        if compacted:
            return {language: compacted}

    return {}


def extract_multilingual_payload(entity: Mapping[str, Any]) -> dict[str, Any]:
    labels = extract_value_map(entity.get("labels"))
    aliases = extract_alias_map(entity.get("aliases"))
    descriptions = extract_value_map(entity.get("descriptions"))

    return {
        "labels": labels,
        "aliases": aliases,
        "descriptions": descriptions,
    }


def extract_value_map(raw_value_map: Any) -> dict[str, str]:
    if not isinstance(raw_value_map, Mapping):
        return {}

    extracted: dict[str, str] = {}
    for language, payload in raw_value_map.items():
        if not isinstance(language, str) or not isinstance(payload, Mapping):
            continue

        value = payload.get("value")
        if isinstance(value, str):
            normalized = normalize_text(value)
            if normalized:
                extracted[language] = normalized

    return dict(sorted(extracted.items(), key=lambda pair: pair[0]))


def extract_alias_map(raw_alias_map: Any) -> dict[str, list[str]]:
    if not isinstance(raw_alias_map, Mapping):
        return {}

    extracted: dict[str, list[str]] = {}
    for language, aliases_payload in raw_alias_map.items():
        if not isinstance(language, str):
            continue

        if not isinstance(aliases_payload, Sequence) or isinstance(
            aliases_payload, (str, bytes, bytearray)
        ):
            continue

        deduped_aliases: list[str] = []
        seen: set[str] = set()
        for alias_payload in aliases_payload:
            if not isinstance(alias_payload, Mapping):
                continue

            alias_value = alias_payload.get("value")
            if not isinstance(alias_value, str):
                continue

            normalized_alias = normalize_text(alias_value)
            if not normalized_alias or normalized_alias in seen:
                continue

            seen.add(normalized_alias)
            deduped_aliases.append(normalized_alias)

        if deduped_aliases:
            extracted[language] = deduped_aliases

    return dict(sorted(extracted.items(), key=lambda pair: pair[0]))


def normalize_text(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    normalized = normalize_text(text).casefold()
    return [match.group(0) for match in _TOKEN_RE.finditer(normalized)]


def build_name_text(labels: Mapping[str, str], aliases: Mapping[str, Sequence[str]]) -> str:
    values: list[str] = []
    seen: set[str] = set()

    for language in sorted(labels):
        candidate = normalize_text(labels[language])
        if candidate and candidate not in seen:
            seen.add(candidate)
            values.append(candidate)

    for language in sorted(aliases):
        for alias in aliases[language]:
            candidate = normalize_text(alias)
            if candidate and candidate not in seen:
                seen.add(candidate)
                values.append(candidate)

    return " ".join(values)


def build_bow_text_from_descriptions(
    descriptions: Mapping[str, str],
    *,
    stopwords: set[str] | None = None,
) -> str:
    active_stopwords = stopwords if stopwords is not None else DEFAULT_STOPWORDS

    tokens: list[str] = []
    for language in sorted(descriptions):
        for token in tokenize(descriptions[language]):
            if len(token) <= 1 or token in active_stopwords:
                continue
            tokens.append(token)

    return " ".join(tokens)
