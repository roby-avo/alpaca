from __future__ import annotations

__test__ = False

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from typing import Any

from .build_postgres_entities import infer_item_category
from .common import (
    extract_multilingual_payload,
    parse_language_allowlist,
    select_alias_map_languages,
    select_text_map_languages,
)
from .ner_typing import infer_ner_types
from .wikidata_sample_ids import resolve_qids
from .wikidata_sample_postgres import (
    DEFAULT_BASE_URL,
    DEFAULT_HTTP_MAX_RETRIES,
    DEFAULT_HTTP_RETRY_BACKOFF_SECONDS,
    DEFAULT_HTTP_RETRY_MAX_SLEEP_SECONDS,
    fetch_entity_payload,
    parse_non_negative_float,
    parse_non_negative_int,
)


def build_live_ner_report(
    entity: Mapping[str, Any],
    *,
    qid: str,
    source_url: str,
    language_allowlist: Sequence[str],
    max_aliases_per_language: int,
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
    coarse_types, fine_types, source = infer_ner_types(
        entity_id=qid,
        labels=labels,
        aliases=aliases,
        descriptions=descriptions,
        claims=entity.get("claims") if isinstance(entity.get("claims"), Mapping) else None,
    )
    return {
        "qid": qid,
        "source_url": source_url,
        "languages": list(language_allowlist),
        "labels": labels,
        "aliases": aliases,
        "descriptions": descriptions,
        "item_category": infer_item_category(entity),
        "coarse_types": coarse_types,
        "fine_types": fine_types,
        "coarse_type": coarse_types[0] if coarse_types else "",
        "fine_type": fine_types[0] if fine_types else "",
        "ner_type_source": source,
    }


def classify_live_qid(
    qid: str,
    *,
    base_url: str,
    language_allowlist: Sequence[str],
    max_aliases_per_language: int,
    timeout_seconds: float,
    sleep_seconds: float,
    http_max_retries: int,
    http_retry_backoff_seconds: float,
    http_retry_max_sleep_seconds: float,
) -> dict[str, Any]:
    result = fetch_entity_payload(
        qid,
        base_url=base_url,
        timeout_seconds=timeout_seconds,
        sleep_seconds=sleep_seconds,
        max_retries=http_max_retries,
        retry_backoff_seconds=http_retry_backoff_seconds,
        retry_max_sleep_seconds=http_retry_max_sleep_seconds,
    )
    if result.payload is None:
        detail = result.error or "Entity payload missing in response."
        raise RuntimeError(f"Could not fetch live entity {qid} from {result.source_url}: {detail}")

    return build_live_ner_report(
        result.payload,
        qid=qid,
        source_url=result.source_url,
        language_allowlist=language_allowlist,
        max_aliases_per_language=max_aliases_per_language,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch one live Wikidata QID and print its inferred NER typing as JSON."
    )
    parser.add_argument(
        "--qid",
        required=True,
        help="Single QID to classify (example: Q42).",
    )
    parser.add_argument(
        "--languages",
        default="en",
        help="Comma-separated language allowlist for labels/descriptions (default: en).",
    )
    parser.add_argument(
        "--max-aliases-per-language",
        type=parse_non_negative_int,
        default=8,
        help="Max aliases considered per language (default: 8, 0 disables aliases).",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"Wikidata EntityData base URL (default: {DEFAULT_BASE_URL}).",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=parse_non_negative_float,
        default=20.0,
        help="HTTP timeout for the live fetch (default: 20.0).",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=parse_non_negative_float,
        default=0.0,
        help="Delay after a successful fetch (default: 0.0).",
    )
    parser.add_argument(
        "--http-max-retries",
        type=parse_non_negative_int,
        default=DEFAULT_HTTP_MAX_RETRIES,
        help=f"HTTP retries for 429/5xx responses (default: {DEFAULT_HTTP_MAX_RETRIES}).",
    )
    parser.add_argument(
        "--http-retry-backoff-seconds",
        type=parse_non_negative_float,
        default=DEFAULT_HTTP_RETRY_BACKOFF_SECONDS,
        help=(
            "Base exponential backoff between HTTP retries "
            f"(default: {DEFAULT_HTTP_RETRY_BACKOFF_SECONDS})."
        ),
    )
    parser.add_argument(
        "--http-retry-max-sleep-seconds",
        type=parse_non_negative_float,
        default=DEFAULT_HTTP_RETRY_MAX_SLEEP_SECONDS,
        help=(
            "Max sleep between HTTP retries "
            f"(default: {DEFAULT_HTTP_RETRY_MAX_SLEEP_SECONDS})."
        ),
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the JSON result.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        qids = resolve_qids(args.qid, None)
        if len(qids) != 1:
            raise ValueError("--qid accepts exactly one QID.")

        report = classify_live_qid(
            qids[0],
            base_url=args.base_url,
            language_allowlist=parse_language_allowlist(args.languages, arg_name="--languages"),
            max_aliases_per_language=args.max_aliases_per_language,
            timeout_seconds=args.timeout_seconds,
            sleep_seconds=args.sleep_seconds,
            http_max_retries=args.http_max_retries,
            http_retry_backoff_seconds=args.http_retry_backoff_seconds,
            http_retry_max_sleep_seconds=args.http_retry_max_sleep_seconds,
        )
    except (RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    json.dump(
        report,
        sys.stdout,
        ensure_ascii=False,
        indent=2 if args.pretty else None,
        sort_keys=True,
    )
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
