from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class JsonDiskCache:
    def __init__(self, root: Path) -> None:
        self.root = root

    def _path_for_key(self, namespace: str, key_payload: Any) -> Path:
        digest = hashlib.sha256(_stable_json(key_payload).encode("utf-8")).hexdigest()
        return self.root / namespace / f"{digest}.json"

    def get(self, namespace: str, key_payload: Any) -> dict[str, Any] | None:
        path = self._path_for_key(namespace, key_payload)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else None

    def set(self, namespace: str, key_payload: Any, value: dict[str, Any]) -> dict[str, Any]:
        path = self._path_for_key(namespace, key_payload)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        return value
