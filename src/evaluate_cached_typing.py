from __future__ import annotations

import asyncio
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .ner_typing import infer_ner_types
from .wikidata_sample_cache import (
    DEFAULT_BASE_URL,
    DEFAULT_CONCURRENCY,
    DEFAULT_MAX_AGE_HOURS,
    fetch_all_entities,
    parse_id_list,
    parse_iso_utc,
    parse_positive_int,
)


def extract_entity_payload(raw_json: dict[str, Any], expected_id: str) -> dict[str, Any] | None:
    entities = raw_json.get("entities")
    if not isinstance(entities, dict):
        return None

    payload = entities.get(expected_id)
    if isinstance(payload, dict):
        return payload

    # Fallback: some payloads might include a canonical ID different from filename.
    for value in entities.values():
        if isinstance(value, dict):
            candidate_id = value.get("id")
            if isinstance(candidate_id, str) and candidate_id == expected_id:
                return value

    return None


def extract_value_map(raw_map: Any) -> dict[str, str]:
    if not isinstance(raw_map, dict):
        return {}

    output: dict[str, str] = {}
    for language, payload in raw_map.items():
        if not isinstance(language, str) or not isinstance(payload, dict):
            continue
        value = payload.get("value")
        if isinstance(value, str) and value.strip():
            output[language] = value.strip()

    return output


def extract_alias_map(raw_map: Any) -> dict[str, list[str]]:
    if not isinstance(raw_map, dict):
        return {}

    output: dict[str, list[str]] = {}
    for language, aliases in raw_map.items():
        if not isinstance(language, str) or not isinstance(aliases, list):
            continue
        values: list[str] = []
        seen: set[str] = set()
        for alias_payload in aliases:
            if not isinstance(alias_payload, dict):
                continue
            value = alias_payload.get("value")
            if not isinstance(value, str):
                continue
            normalized = value.strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                values.append(normalized)
        if values:
            output[language] = values

    return output


def compute_age_hours(retrieved_at_utc: str | None) -> float | None:
    if not retrieved_at_utc:
        return None
    try:
        retrieved = parse_iso_utc(retrieved_at_utc)
    except ValueError:
        return None
    now_utc = datetime.now(timezone.utc)
    return max(0.0, (now_utc - retrieved).total_seconds() / 3600.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate lexical NER typing on cached Wikidata Special:EntityData JSON files "
            "and report inferred coarse/fine types with cache age."
        )
    )
    parser.add_argument(
        "--cache-dir",
        default="./data/cache/wikidata_entities",
        help="Cache directory containing Q<ID>.json and Q<ID>.meta.json files.",
    )
    parser.add_argument(
        "--ids",
        help="Optional comma-separated IDs to inspect (example: Q90,Q42).",
    )
    parser.add_argument(
        "--no-fetch-missing",
        action="store_true",
        help=(
            "Do not fetch missing IDs from live Wikidata. "
            "By default, missing IDs from --ids are fetched and cached."
        ),
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Refresh requested IDs even if cached JSON exists.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"EntityData base URL used for fetch-missing mode (default: {DEFAULT_BASE_URL}).",
    )
    parser.add_argument(
        "--max-age-hours",
        type=float,
        default=DEFAULT_MAX_AGE_HOURS,
        help=(
            "Maximum age for cache reuse when fetching missing IDs "
            f"(default: {DEFAULT_MAX_AGE_HOURS})."
        ),
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=20.0,
        help="HTTP timeout per entity fetch in fetch-missing mode (default: 20).",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Sleep after successful fetch tasks in fetch-missing mode (default: 0).",
    )
    parser.add_argument(
        "--concurrency",
        type=parse_positive_int,
        default=DEFAULT_CONCURRENCY,
        help=f"Maximum parallel fetch tasks in fetch-missing mode (default: {DEFAULT_CONCURRENCY}).",
    )
    parser.add_argument(
        "--contains",
        help="Optional case-insensitive substring filter over English label.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=100,
        help="Maximum rows to print (default: 100).",
    )
    return parser.parse_args()


def resolve_ids(cache_dir: Path, ids_arg: str | None) -> list[str]:
    if ids_arg:
        parsed = parse_id_list(ids_arg)
        if not parsed:
            raise ValueError("--ids provided but no valid IDs were parsed.")
        return parsed

    ids: list[str] = []
    for json_path in sorted(cache_dir.glob("Q*.json")):
        if json_path.name.endswith(".meta.json"):
            continue
        ids.append(json_path.stem)
    return ids


def collect_ids_to_fetch(
    ids: list[str],
    cache_dir: Path,
    *,
    force_refresh: bool,
) -> list[str]:
    if force_refresh:
        return ids

    return [
        entity_id
        for entity_id in ids
        if not (cache_dir / f"{entity_id}.json").is_file()
    ]


def main() -> int:
    args = parse_args()

    cache_dir = Path(args.cache_dir).expanduser()
    if not cache_dir.is_dir():
        print(f"ERROR: Cache directory not found: {cache_dir}", file=sys.stderr)
        return 1

    try:
        ids = resolve_ids(cache_dir, args.ids)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not ids:
        print("No cached entity JSON files found.")
        return 0

    if args.ids and not args.no_fetch_missing:
        ids_to_fetch = collect_ids_to_fetch(
            ids,
            cache_dir,
            force_refresh=bool(args.force_refresh),
        )
        if ids_to_fetch:
            print(
                f"Fetching {len(ids_to_fetch)} entity IDs into cache before evaluation...",
                file=sys.stderr,
            )
            fetch_results = asyncio.run(
                fetch_all_entities(
                    entity_ids=ids_to_fetch,
                    base_url=args.base_url,
                    cache_dir=cache_dir,
                    timeout_seconds=max(1.0, float(args.timeout_seconds)),
                    max_age_hours=max(0.0, float(args.max_age_hours)),
                    force_refresh=bool(args.force_refresh),
                    now_utc=datetime.now(timezone.utc),
                    sleep_seconds=max(0.0, float(args.sleep_seconds)),
                    concurrency=int(args.concurrency),
                    show_progress=True,
                )
            )
            for row in fetch_results:
                if row.status == "error":
                    print(
                        f"WARN: could not fetch {row.entity_id}: {row.error}",
                        file=sys.stderr,
                    )

    needle = args.contains.casefold() if args.contains else None
    rows_printed = 0
    missing_after_fetch: list[str] = []

    print("id\tage_h\tlabel_en\tcoarse\tfine")
    for entity_id in ids:
        if rows_printed >= max(1, args.max_rows):
            break

        entity_json_path = cache_dir / f"{entity_id}.json"
        entity_meta_path = cache_dir / f"{entity_id}.meta.json"

        if not entity_json_path.is_file():
            missing_after_fetch.append(entity_id)
            continue

        try:
            raw_json = json.loads(entity_json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue

        if not isinstance(raw_json, dict):
            continue

        entity_payload = extract_entity_payload(raw_json, entity_id)
        if entity_payload is None:
            continue

        labels = extract_value_map(entity_payload.get("labels"))
        aliases = extract_alias_map(entity_payload.get("aliases"))
        descriptions = extract_value_map(entity_payload.get("descriptions"))
        label_en = labels.get("en", "")

        if needle and needle not in label_en.casefold():
            continue

        coarse, fine, _ = infer_ner_types(entity_id, labels, aliases, descriptions)

        retrieved_at_utc: str | None = None
        if entity_meta_path.is_file():
            try:
                meta = json.loads(entity_meta_path.read_text(encoding="utf-8"))
                if isinstance(meta, dict):
                    candidate = meta.get("retrieved_at_utc")
                    if isinstance(candidate, str):
                        retrieved_at_utc = candidate
            except json.JSONDecodeError:
                retrieved_at_utc = None

        age_hours = compute_age_hours(retrieved_at_utc)
        age_text = f"{age_hours:.2f}" if age_hours is not None else "na"

        print(
            f"{entity_id}\t{age_text}\t{label_en or '-'}\t"
            f"{','.join(coarse) or '-'}\t{','.join(fine) or '-'}"
        )
        rows_printed += 1

    if rows_printed == 0:
        if missing_after_fetch:
            missing_preview = ",".join(missing_after_fetch[:10])
            print(
                "No rows matched filters. Missing cache entries for IDs: "
                f"{missing_preview}"
            )
        else:
            print("No rows matched filters.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
