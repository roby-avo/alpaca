from __future__ import annotations

import re
from pathlib import Path

_QID_RE = re.compile(r"^Q[1-9][0-9]*$")


def default_demo_qids(count: int) -> list[str]:
    if count <= 0:
        raise ValueError("--count must be > 0.")
    return [f"Q{index}" for index in range(1, count + 1)]


def parse_qid_list(raw: str) -> list[str]:
    values = [token.strip().upper() for token in raw.replace("\n", ",").split(",")]
    ids: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value:
            continue
        if not _QID_RE.match(value):
            raise ValueError(f"Invalid entity ID '{value}'. Expected QIDs like Q42.")
        if value in seen:
            continue
        seen.add(value)
        ids.append(value)
    return ids


def load_qids_from_file(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    tokens: list[str] = []
    for line in lines:
        cleaned = line.strip()
        if not cleaned or cleaned.startswith("#"):
            continue
        tokens.append(cleaned)
    return parse_qid_list(",".join(tokens))


def resolve_qids(ids: str | None, ids_file: str | None, count: int | None = None) -> list[str]:
    specified = sum(1 for value in (ids, ids_file) if value) + (1 if count is not None else 0)
    if specified > 1:
        raise ValueError("Use only one of --ids, --ids-file, or --count.")

    if ids:
        parsed = parse_qid_list(ids)
    elif ids_file:
        parsed = load_qids_from_file(Path(ids_file).expanduser())
    elif count is not None:
        parsed = default_demo_qids(count)
    else:
        raise ValueError("Provide --ids, --ids-file, or --count.")
    if not parsed:
        raise ValueError("No valid QIDs were provided.")
    return parsed
