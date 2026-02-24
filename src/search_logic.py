from __future__ import annotations

import re
from collections.abc import Sequence

_VALID_TYPE_LABEL_RE = re.compile(r"^[A-Za-z0-9_.:/-]+$")


def normalize_type_labels(
    type_labels: Sequence[str] | None,
    *,
    field_name: str,
) -> list[str]:
    if type_labels is None:
        return []

    normalized: list[str] = []
    seen: set[str] = set()
    for raw in type_labels:
        value = raw.strip()
        if not value:
            continue
        if not _VALID_TYPE_LABEL_RE.match(value):
            raise ValueError(
                f"Invalid value '{raw}' for {field_name}. Allowed characters: "
                "letters, digits, '_', '-', '.', ':', '/'."
            )
        if value not in seen:
            seen.add(value)
            normalized.append(value)

    return normalized
