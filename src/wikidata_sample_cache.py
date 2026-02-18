from __future__ import annotations

import asyncio
import argparse
import hashlib
import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, request

from .common import tqdm

DEFAULT_CACHE_DIR = Path("./data/cache/wikidata_entities")
DEFAULT_BASE_URL = "https://www.wikidata.org/wiki/Special:EntityData"
DEFAULT_SAMPLE_COUNT = 80
DEFAULT_MAX_AGE_HOURS = 72.0
DEFAULT_CONCURRENCY = 12


@dataclass(slots=True)
class FetchResult:
    entity_id: str
    status: str
    retrieved_at_utc: str | None
    age_hours: float | None
    source_url: str
    cache_json_path: Path
    cache_meta_path: Path
    http_status: int | None = None
    error: str | None = None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_iso_utc(raw_value: str) -> datetime:
    value = raw_value.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def compute_age_hours(retrieved_at_utc: str, now_utc: datetime) -> float:
    retrieved = parse_iso_utc(retrieved_at_utc)
    return max(0.0, (now_utc - retrieved).total_seconds() / 3600.0)


def default_entity_ids(count: int) -> list[str]:
    # Deterministic IDs that include Q90 (Paris) within the first 100.
    # We intentionally sample early Wikidata IDs to keep fetch logic simple.
    if count < 1:
        return []
    return [f"Q{index}" for index in range(1, count + 1)]


def parse_id_list(raw: str) -> list[str]:
    values = [token.strip() for token in raw.replace("\n", ",").split(",")]
    ids: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value:
            continue
        if not value.startswith("Q"):
            raise ValueError(f"Invalid entity ID '{value}'. Expected IDs like Q42.")
        if value not in seen:
            seen.add(value)
            ids.append(value)
    return ids


def parse_positive_int(raw: str) -> int:
    try:
        parsed = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc

    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def load_ids_from_file(path: Path) -> list[str]:
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    tokens: list[str] = []
    for line in raw_lines:
        cleaned = line.strip()
        if not cleaned or cleaned.startswith("#"):
            continue
        tokens.append(cleaned)

    return parse_id_list(",".join(tokens))


def load_cached_meta(meta_path: Path) -> dict[str, Any] | None:
    if not meta_path.is_file():
        return None
    try:
        parsed = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, dict):
        return None
    return parsed


def should_refresh_cache(meta: dict[str, Any] | None, max_age_hours: float, now_utc: datetime) -> bool:
    if meta is None:
        return True
    retrieved_at = meta.get("retrieved_at_utc")
    if not isinstance(retrieved_at, str) or not retrieved_at.strip():
        return True

    try:
        age_hours = compute_age_hours(retrieved_at, now_utc)
    except ValueError:
        return True

    return age_hours > max_age_hours


def fetch_entity_json(url: str, timeout_seconds: float) -> tuple[bytes, dict[str, str], int]:
    req = request.Request(url=url, method="GET")
    req.add_header("Accept", "application/json")
    req.add_header("User-Agent", "alpaca-sample-cache/0.1")

    with request.urlopen(req, timeout=timeout_seconds) as resp:
        body = resp.read()
        headers = {key.lower(): value for key, value in resp.headers.items()}
        status = getattr(resp, "status", 200)

    return body, headers, status


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fetch_or_reuse(
    entity_id: str,
    *,
    base_url: str,
    cache_dir: Path,
    timeout_seconds: float,
    max_age_hours: float,
    force_refresh: bool,
    now_utc: datetime,
    sleep_seconds: float,
) -> FetchResult:
    entity_json_path = cache_dir / f"{entity_id}.json"
    entity_meta_path = cache_dir / f"{entity_id}.meta.json"
    source_url = f"{base_url.rstrip('/')}/{entity_id}.json"

    cached_meta = load_cached_meta(entity_meta_path)
    needs_refresh = force_refresh or should_refresh_cache(cached_meta, max_age_hours, now_utc)

    if not needs_refresh and entity_json_path.is_file() and cached_meta is not None:
        retrieved_at = cached_meta.get("retrieved_at_utc")
        age_hours: float | None = None
        if isinstance(retrieved_at, str):
            try:
                age_hours = compute_age_hours(retrieved_at, now_utc)
            except ValueError:
                age_hours = None

        return FetchResult(
            entity_id=entity_id,
            status="cache_hit",
            retrieved_at_utc=retrieved_at if isinstance(retrieved_at, str) else None,
            age_hours=age_hours,
            source_url=source_url,
            cache_json_path=entity_json_path,
            cache_meta_path=entity_meta_path,
            http_status=int(cached_meta.get("http_status", 0)) if cached_meta.get("http_status") else None,
        )

    try:
        body, headers, status_code = fetch_entity_json(source_url, timeout_seconds)
    except error.HTTPError as exc:
        message = exc.read().decode("utf-8", errors="replace")[:300]
        return FetchResult(
            entity_id=entity_id,
            status="error",
            retrieved_at_utc=None,
            age_hours=None,
            source_url=source_url,
            cache_json_path=entity_json_path,
            cache_meta_path=entity_meta_path,
            http_status=exc.code,
            error=f"HTTP {exc.code}: {message or exc.reason}",
        )
    except error.URLError as exc:
        reason = exc.reason if isinstance(exc.reason, str) else repr(exc.reason)
        return FetchResult(
            entity_id=entity_id,
            status="error",
            retrieved_at_utc=None,
            age_hours=None,
            source_url=source_url,
            cache_json_path=entity_json_path,
            cache_meta_path=entity_meta_path,
            error=f"Network error: {reason}",
        )

    retrieved_at_utc = utc_now_iso()
    sha256 = hashlib.sha256(body).hexdigest()

    entity_json_path.parent.mkdir(parents=True, exist_ok=True)
    entity_json_path.write_bytes(body)

    payload_label_en: str | None = None
    try:
        parsed = json.loads(body.decode("utf-8"))
        if isinstance(parsed, dict):
            entities = parsed.get("entities")
            if isinstance(entities, dict):
                entity = entities.get(entity_id)
                if isinstance(entity, dict):
                    labels = entity.get("labels")
                    if isinstance(labels, dict):
                        en_payload = labels.get("en")
                        if isinstance(en_payload, dict):
                            en_value = en_payload.get("value")
                            if isinstance(en_value, str):
                                payload_label_en = en_value
    except (UnicodeDecodeError, json.JSONDecodeError):
        payload_label_en = None

    meta = {
        "id": entity_id,
        "source_url": source_url,
        "retrieved_at_utc": retrieved_at_utc,
        "http_status": status_code,
        "etag": headers.get("etag"),
        "last_modified": headers.get("last-modified"),
        "content_type": headers.get("content-type"),
        "bytes": len(body),
        "sha256": sha256,
        "label_en": payload_label_en,
    }
    write_json(entity_meta_path, meta)

    if sleep_seconds > 0:
        time.sleep(sleep_seconds)

    return FetchResult(
        entity_id=entity_id,
        status="fetched",
        retrieved_at_utc=retrieved_at_utc,
        age_hours=0.0,
        source_url=source_url,
        cache_json_path=entity_json_path,
        cache_meta_path=entity_meta_path,
        http_status=status_code,
    )


async def fetch_all_entities(
    entity_ids: list[str],
    *,
    base_url: str,
    cache_dir: Path,
    timeout_seconds: float,
    max_age_hours: float,
    force_refresh: bool,
    now_utc: datetime,
    sleep_seconds: float,
    concurrency: int,
    fetch_fn: Callable[..., FetchResult] = fetch_or_reuse,
    show_progress: bool = True,
) -> list[FetchResult]:
    semaphore = asyncio.Semaphore(max(1, concurrency))
    ordered_results: list[FetchResult | None] = [None] * len(entity_ids)
    counters = {"fetched": 0, "cache_hit": 0, "error": 0}

    progress = tqdm(
        total=len(entity_ids),
        desc="wikidata-cache",
        unit="entity",
    ) if show_progress else None

    async def worker(index: int, entity_id: str) -> None:
        async with semaphore:
            result = await asyncio.to_thread(
                fetch_fn,
                entity_id,
                base_url=base_url,
                cache_dir=cache_dir,
                timeout_seconds=timeout_seconds,
                max_age_hours=max_age_hours,
                force_refresh=force_refresh,
                now_utc=now_utc,
                sleep_seconds=sleep_seconds,
            )
            ordered_results[index] = result
            counters[result.status] = counters.get(result.status, 0) + 1
            if progress is not None:
                progress.update(1)
                progress.set_postfix(
                    fetched=counters["fetched"],
                    cache_hits=counters["cache_hit"],
                    errors=counters["error"],
                )

    try:
        tasks = [
            asyncio.create_task(worker(index, entity_id))
            for index, entity_id in enumerate(entity_ids)
        ]
        await asyncio.gather(*tasks)
        return [result for result in ordered_results if result is not None]
    finally:
        if progress is not None:
            progress.close()


def write_run_report(cache_dir: Path, results: list[FetchResult]) -> Path:
    report = {
        "generated_at_utc": utc_now_iso(),
        "total": len(results),
        "fetched": sum(1 for row in results if row.status == "fetched"),
        "cache_hits": sum(1 for row in results if row.status == "cache_hit"),
        "errors": sum(1 for row in results if row.status == "error"),
        "items": [
            {
                "id": row.entity_id,
                "status": row.status,
                "retrieved_at_utc": row.retrieved_at_utc,
                "age_hours": row.age_hours,
                "source_url": row.source_url,
                "cache_json_path": str(row.cache_json_path),
                "cache_meta_path": str(row.cache_meta_path),
                "http_status": row.http_status,
                "error": row.error,
            }
            for row in results
        ],
    }

    report_path = cache_dir / "last_run_report.json"
    write_json(report_path, report)
    return report_path


def print_summary(results: list[FetchResult], report_path: Path) -> None:
    fetched = sum(1 for row in results if row.status == "fetched")
    cache_hits = sum(1 for row in results if row.status == "cache_hit")
    errors = [row for row in results if row.status == "error"]

    age_values = [row.age_hours for row in results if row.age_hours is not None]
    oldest = max(age_values) if age_values else None
    newest = min(age_values) if age_values else None

    print(
        "Completed Wikidata sample cache run:",
        f"total={len(results)}",
        f"fetched={fetched}",
        f"cache_hits={cache_hits}",
        f"errors={len(errors)}",
        f"report={report_path}",
    )

    if oldest is not None and newest is not None:
        print(
            "Cache age range (hours):",
            f"newest={newest:.2f}",
            f"oldest={oldest:.2f}",
        )

    for row in errors[:10]:
        print(f"ERROR {row.entity_id}: {row.error}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch and cache live Wikidata entities from Special:EntityData. "
            "Stores raw JSON plus metadata with retrieval timestamps for cache age tracking."
        )
    )
    parser.add_argument(
        "--cache-dir",
        default=str(DEFAULT_CACHE_DIR),
        help=f"Cache directory (default: {DEFAULT_CACHE_DIR}).",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"EntityData base URL (default: {DEFAULT_BASE_URL}).",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_SAMPLE_COUNT,
        help=f"Number of entity IDs to fetch when no explicit IDs are provided (default: {DEFAULT_SAMPLE_COUNT}).",
    )
    parser.add_argument(
        "--max-age-hours",
        type=float,
        default=DEFAULT_MAX_AGE_HOURS,
        help=(
            "Maximum age for cache reuse in hours. Older items are refreshed "
            f"(default: {DEFAULT_MAX_AGE_HOURS})."
        ),
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=20.0,
        help="HTTP timeout per entity fetch (default: 20).",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.1,
        help=(
            "Sleep after each successful fetch task to avoid hammering endpoints "
            "(default: 0.1)."
        ),
    )
    parser.add_argument(
        "--concurrency",
        type=parse_positive_int,
        default=DEFAULT_CONCURRENCY,
        help=f"Maximum parallel fetch tasks (default: {DEFAULT_CONCURRENCY}).",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Refresh all requested entities even if cache is fresh.",
    )
    parser.add_argument(
        "--ids",
        help="Comma-separated entity IDs (example: Q42,Q90,Q30).",
    )
    parser.add_argument(
        "--ids-file",
        help="Path to text file with one QID per line (comments with # are ignored).",
    )
    return parser.parse_args()


def resolve_entity_ids(args: argparse.Namespace) -> list[str]:
    if args.ids and args.ids_file:
        raise ValueError("Use either --ids or --ids-file, not both.")

    if args.ids:
        ids = parse_id_list(args.ids)
    elif args.ids_file:
        ids_file_path = Path(args.ids_file).expanduser()
        if not ids_file_path.is_file():
            raise FileNotFoundError(f"IDs file not found: {ids_file_path}")
        ids = load_ids_from_file(ids_file_path)
    else:
        if args.count < 1:
            raise ValueError("--count must be >= 1")
        ids = default_entity_ids(args.count)

    if not ids:
        raise ValueError("No entity IDs were resolved.")

    return ids


def main() -> int:
    args = parse_args()

    try:
        entity_ids = resolve_entity_ids(args)
        cache_dir = Path(args.cache_dir).expanduser()
        cache_dir.mkdir(parents=True, exist_ok=True)

        now_utc = datetime.now(timezone.utc)
        results = asyncio.run(
            fetch_all_entities(
                entity_ids=entity_ids,
                base_url=args.base_url,
                cache_dir=cache_dir,
                timeout_seconds=max(1.0, float(args.timeout_seconds)),
                max_age_hours=max(0.0, float(args.max_age_hours)),
                force_refresh=bool(args.force_refresh),
                now_utc=now_utc,
                sleep_seconds=max(0.0, float(args.sleep_seconds)),
                concurrency=int(args.concurrency),
            )
        )

        report_path = write_run_report(cache_dir, results)
        print_summary(results, report_path)
        return 0
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
