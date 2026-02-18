from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, TextIO

from .common import (
    DUMP_PATH_ENV,
    estimate_wikidata_entity_total,
    ensure_existing_file,
    ensure_parent_dir,
    finalize_tqdm_total,
    is_supported_entity_id,
    iter_wikidata_entities,
    keep_tqdm_total_ahead,
    open_text_for_write,
    tqdm,
)

_ENTITY_ID_RE = re.compile(r"^[QP][1-9][0-9]*$")


def parse_positive_int(raw: str) -> int:
    try:
        parsed = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc

    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a deterministic small Wikidata-style dump (.json/.json.gz/.json.bz2) "
            "from a larger source dump."
        )
    )
    parser.add_argument(
        "--source-dump-path",
        required=True,
        help=(
            "Path to source dump (.json/.json.gz/.json.bz2). "
            f"Use an explicit path instead of relying on ${DUMP_PATH_ENV}."
        ),
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Output path for the compact dump (.json/.json.gz/.json.bz2).",
    )
    parser.add_argument(
        "--count",
        type=parse_positive_int,
        default=1000,
        help=(
            "Number of Q*/P* entities to include when --ids is not provided "
            "(default: 1000)."
        ),
    )
    parser.add_argument(
        "--ids",
        help="Optional comma-separated explicit IDs (example: Q42,Q90,P31).",
    )
    return parser.parse_args()


def parse_entity_id_list(raw: str) -> list[str]:
    values = [token.strip() for token in raw.replace("\n", ",").split(",")]
    ids: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value:
            continue
        if not _ENTITY_ID_RE.match(value):
            raise ValueError(f"Invalid entity ID '{value}'. Expected IDs like Q42 or P31.")
        if value in seen:
            continue
        seen.add(value)
        ids.append(value)
    return ids


def _write_entity(handle: TextIO, entity: dict[str, Any], is_first: bool) -> None:
    prefix = "" if is_first else ","
    encoded = json.dumps(entity, ensure_ascii=False, separators=(",", ":"))
    handle.write(f"{prefix}\n{encoded}")


def _complete_progress_early(progress: Any, scanned: int) -> None:
    total = getattr(progress, "total", None)
    if not isinstance(total, (int, float)):
        return
    if scanned >= total:
        return
    try:
        progress.total = scanned
    except Exception:
        return
    refresh = getattr(progress, "refresh", None)
    if callable(refresh):
        refresh()


def _build_from_count(
    source_dump_path: Path,
    output_path: Path,
    count: int,
) -> tuple[int, int]:
    written = 0
    scanned = 0
    progress_total = estimate_wikidata_entity_total(source_dump_path)
    if progress_total is not None:
        print(f"Progress estimate: small-dump-count total~{progress_total} entities")

    with open_text_for_write(output_path) as handle:
        with tqdm(total=progress_total, desc="small-dump-count", unit="entity") as progress:
            handle.write("[")
            for entity in iter_wikidata_entities(source_dump_path):
                scanned += 1
                progress.update(1)
                keep_tqdm_total_ahead(progress)
                entity_id = entity.get("id")
                if not isinstance(entity_id, str) or not is_supported_entity_id(entity_id):
                    continue

                _write_entity(handle, entity, is_first=(written == 0))
                written += 1
                progress.set_postfix(written=written, requested=count)
                if written >= count:
                    _complete_progress_early(progress, scanned)
                    break

            finalize_tqdm_total(progress)
            handle.write("\n]\n")

    return written, scanned


def _build_from_ids(
    source_dump_path: Path,
    output_path: Path,
    ids: list[str],
) -> tuple[int, int, int]:
    requested_order = [entity_id for entity_id in ids if is_supported_entity_id(entity_id)]
    requested_set = set(requested_order)
    found: dict[str, dict[str, Any]] = {}
    scanned = 0
    progress_total = estimate_wikidata_entity_total(source_dump_path)
    if progress_total is not None:
        print(f"Progress estimate: small-dump-ids total~{progress_total} entities")

    with tqdm(total=progress_total, desc="small-dump-ids", unit="entity") as progress:
        for entity in iter_wikidata_entities(source_dump_path):
            scanned += 1
            progress.update(1)
            keep_tqdm_total_ahead(progress)
            entity_id = entity.get("id")
            if not isinstance(entity_id, str) or entity_id not in requested_set:
                continue

            if entity_id not in found:
                found[entity_id] = entity
                progress.set_postfix(found=len(found), requested=len(requested_set))
                if len(found) >= len(requested_set):
                    _complete_progress_early(progress, scanned)
                    break
        finalize_tqdm_total(progress)

    written = 0
    with open_text_for_write(output_path) as handle:
        handle.write("[")
        for entity_id in requested_order:
            payload = found.get(entity_id)
            if payload is None:
                continue
            _write_entity(handle, payload, is_first=(written == 0))
            written += 1
        handle.write("\n]\n")

    missing = len(requested_order) - written
    return written, missing, scanned


def main() -> int:
    args = parse_args()

    try:
        source_dump_path = Path(args.source_dump_path).expanduser()
        output_path = Path(args.output_path).expanduser()
        ensure_existing_file(
            source_dump_path,
            "Source dump",
            hint="Pass a valid --source-dump-path to an existing dump file.",
        )
        ensure_parent_dir(output_path)

        if args.ids:
            ids = parse_entity_id_list(args.ids)
            if not ids:
                raise ValueError("No valid IDs were parsed from --ids.")

            written, missing, scanned = _build_from_ids(
                source_dump_path=source_dump_path,
                output_path=output_path,
                ids=ids,
            )
            if written == 0:
                raise ValueError(
                    "No requested IDs were found in the source dump. "
                    "Verify --ids and source dump contents."
                )
            print(
                "Built small dump:",
                f"mode=ids",
                f"requested={len(ids)}",
                f"written={written}",
                f"missing={missing}",
                f"scanned={scanned}",
                f"output={output_path}",
            )
            return 0

        written, scanned = _build_from_count(
            source_dump_path=source_dump_path,
            output_path=output_path,
            count=args.count,
        )
        if written == 0:
            raise ValueError("No Q*/P* entities were written. Source dump may be invalid.")

        print(
            "Built small dump:",
            f"mode=count",
            f"requested={args.count}",
            f"written={written}",
            f"missing={max(0, args.count - written)}",
            f"scanned={scanned}",
            f"output={output_path}",
        )
        return 0
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
