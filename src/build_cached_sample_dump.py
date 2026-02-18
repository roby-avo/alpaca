from __future__ import annotations

import argparse
import bz2
import json
import sys
from pathlib import Path
from typing import Any

from .common import ensure_parent_dir, tqdm
from .wikidata_sample_cache import DEFAULT_CACHE_DIR, DEFAULT_SAMPLE_COUNT, default_entity_ids, parse_id_list


def parse_positive_int(raw: str) -> int:
    try:
        parsed = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc

    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def extract_entity_payload(raw_json: dict[str, Any], expected_id: str) -> dict[str, Any] | None:
    entities = raw_json.get("entities")
    if not isinstance(entities, dict):
        return None

    payload = entities.get(expected_id)
    if isinstance(payload, dict):
        return payload

    for value in entities.values():
        if not isinstance(value, dict):
            continue
        candidate_id = value.get("id")
        if isinstance(candidate_id, str) and candidate_id == expected_id:
            return value

    return None


def resolve_ids(count: int, ids_arg: str | None) -> list[str]:
    if ids_arg:
        ids = parse_id_list(ids_arg)
        if not ids:
            raise ValueError("No valid IDs were parsed from --ids.")
        return ids

    return default_entity_ids(count)


def build_dump(cache_dir: Path, output_path: Path, ids: list[str]) -> tuple[int, int]:
    payloads: list[dict[str, Any]] = []
    missing_count = 0

    with tqdm(total=len(ids), desc="cached-sample-dump", unit="entity") as progress:
        for entity_id in ids:
            entity_path = cache_dir / f"{entity_id}.json"
            if not entity_path.is_file():
                missing_count += 1
                progress.update(1)
                progress.set_postfix(written=len(payloads), missing=missing_count)
                continue

            try:
                raw_json = json.loads(entity_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                missing_count += 1
                progress.update(1)
                progress.set_postfix(written=len(payloads), missing=missing_count)
                continue

            if not isinstance(raw_json, dict):
                missing_count += 1
                progress.update(1)
                progress.set_postfix(written=len(payloads), missing=missing_count)
                continue

            payload = extract_entity_payload(raw_json, entity_id)
            if payload is None:
                missing_count += 1
                progress.update(1)
                progress.set_postfix(written=len(payloads), missing=missing_count)
                continue

            payloads.append(payload)
            progress.update(1)
            progress.set_postfix(written=len(payloads), missing=missing_count)

    ensure_parent_dir(output_path)
    with bz2.open(output_path, mode="wt", encoding="utf-8") as handle:
        handle.write("[\n")
        for index, payload in enumerate(payloads):
            line = json.dumps(payload, ensure_ascii=False)
            suffix = "," if index < len(payloads) - 1 else ""
            handle.write(f"{line}{suffix}\n")
        handle.write("]\n")

    return len(payloads), missing_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a compact .json.bz2 Wikidata-style dump from cached Special:EntityData files."
        )
    )
    parser.add_argument(
        "--cache-dir",
        default=str(DEFAULT_CACHE_DIR),
        help=f"Cache directory containing Q<ID>.json files (default: {DEFAULT_CACHE_DIR}).",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Output dump path (.json.bz2).",
    )
    parser.add_argument(
        "--count",
        type=parse_positive_int,
        default=DEFAULT_SAMPLE_COUNT,
        help=f"Number of default IDs (Q1..Qn) if --ids is not provided (default: {DEFAULT_SAMPLE_COUNT}).",
    )
    parser.add_argument(
        "--ids",
        help="Optional comma-separated IDs to include (example: Q42,Q90).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        cache_dir = Path(args.cache_dir).expanduser()
        if not cache_dir.is_dir():
            raise FileNotFoundError(f"Cache directory not found: {cache_dir}")

        output_path = Path(args.output_path).expanduser()
        ids = resolve_ids(args.count, args.ids)

        written, missing = build_dump(cache_dir, output_path, ids)
        if written == 0:
            raise ValueError(
                f"No entities were written to {output_path}. "
                "Verify cache contents or use --force-refresh when fetching."
            )

        print(
            "Built cached sample dump:",
            f"ids_requested={len(ids)}",
            f"written={written}",
            f"missing_or_invalid={missing}",
            f"output={output_path}",
        )
        return 0
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
