from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib import error, request

from .build_bow_docs import extract_claim_object_ids
from .common import resolve_postgres_dsn, tqdm
from .postgres_store import PostgresStore
from .wikidata_sample_ids import resolve_qids


DEFAULT_BASE_URL = "https://www.wikidata.org/wiki/Special:EntityData"
DEFAULT_MAX_CONTEXT_OBJECT_IDS = 32
DEFAULT_SUPPORT_FETCH_CONCURRENCY_CAP = 4
DEFAULT_SUPPORT_FETCH_MIN_SLEEP_SECONDS = 0.25
DEFAULT_HTTP_MAX_RETRIES = 5
DEFAULT_HTTP_RETRY_BACKOFF_SECONDS = 1.0
DEFAULT_HTTP_RETRY_MAX_SLEEP_SECONDS = 10.0
DEFAULT_SUPPORT_PREFETCH_MAX_IN_FLIGHT_FACTOR = 2
DEFAULT_SUPPORT_RATE_LIMIT_ABORT_THRESHOLD = 25
DEFAULT_MAX_CONTEXT_SUPPORT_PREFETCH = 256


@dataclass(frozen=True, slots=True)
class FetchResult:
    qid: str
    status: str
    source_url: str
    error: str | None = None
    payload: dict[str, Any] | None = None
    http_status: int | None = None


def parse_positive_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return value


def parse_non_negative_float(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if value < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return value


def parse_non_negative_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if value < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return value


def _extract_entity_payload(raw_json: Mapping[str, Any], qid: str) -> dict[str, Any] | None:
    entities = raw_json.get("entities")
    if not isinstance(entities, Mapping):
        return None
    payload = entities.get(qid)
    if isinstance(payload, Mapping):
        return {str(k): v for k, v in payload.items() if isinstance(k, str)}
    return None


def _collect_related_entity_ids(
    payloads: Sequence[Mapping[str, Any]],
    *,
    max_context_object_ids: int,
) -> list[str]:
    if max_context_object_ids <= 0:
        return []
    related_ids: list[str] = []
    seen: set[str] = set()
    for payload in payloads:
        for object_id in extract_claim_object_ids(payload, limit=max_context_object_ids):
            if object_id in seen:
                continue
            seen.add(object_id)
            related_ids.append(object_id)
    return related_ids


def _deterministic_sample(values: Sequence[str], *, limit: int) -> list[str]:
    if limit <= 0:
        return []
    if len(values) <= limit:
        return list(values)
    n = len(values)
    # Evenly samples across the original deterministic order, preserving order in output.
    indices = [int(i * n / limit) for i in range(limit)]
    out: list[str] = []
    seen: set[int] = set()
    for idx in indices:
        if idx in seen:
            continue
        seen.add(idx)
        out.append(values[idx])
    if len(out) < limit:
        for idx, value in enumerate(values):
            if idx in seen:
                continue
            out.append(value)
            if len(out) >= limit:
                break
    return out


def fetch_entity_payload(
    qid: str,
    *,
    base_url: str,
    timeout_seconds: float,
    sleep_seconds: float,
    max_retries: int,
    retry_backoff_seconds: float,
    retry_max_sleep_seconds: float,
) -> FetchResult:
    source_url = f"{base_url.rstrip('/')}/{qid}.json"
    attempts = max(1, int(max_retries) + 1)
    last_http_status: int | None = None
    for attempt in range(attempts):
        req = request.Request(url=source_url, method="GET")
        req.add_header("Accept", "application/json")
        req.add_header("User-Agent", "alpaca-sample-postgres/0.1")
        try:
            with request.urlopen(req, timeout=timeout_seconds) as resp:
                body = resp.read()
        except error.HTTPError as exc:
            last_http_status = exc.code
            if exc.code in (429, 502, 503, 504) and attempt + 1 < attempts:
                retry_after_header = exc.headers.get("Retry-After") if exc.headers else None
                retry_after_seconds: float | None = None
                if isinstance(retry_after_header, str):
                    try:
                        retry_after_seconds = float(retry_after_header.strip())
                    except ValueError:
                        retry_after_seconds = None
                backoff = max(0.0, float(retry_backoff_seconds)) * (2**attempt)
                wait_seconds = retry_after_seconds if retry_after_seconds is not None else backoff
                if retry_max_sleep_seconds > 0:
                    wait_seconds = min(wait_seconds, float(retry_max_sleep_seconds))
                if wait_seconds > 0:
                    time.sleep(wait_seconds)
                continue
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            return FetchResult(
                qid=qid,
                status="error",
                source_url=source_url,
                error=f"HTTP {exc.code}: {detail}",
                http_status=exc.code,
            )
        except error.URLError as exc:
            reason = exc.reason if isinstance(exc.reason, str) else repr(exc.reason)
            if attempt + 1 < attempts:
                backoff = max(0.0, float(retry_backoff_seconds)) * (2**attempt)
                if retry_max_sleep_seconds > 0:
                    backoff = min(backoff, float(retry_max_sleep_seconds))
                if backoff > 0:
                    time.sleep(backoff)
                continue
            return FetchResult(qid=qid, status="error", source_url=source_url, error=f"Network error: {reason}")

        try:
            decoded = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            return FetchResult(qid=qid, status="error", source_url=source_url, error=f"Invalid JSON: {exc}")

        if not isinstance(decoded, Mapping):
            return FetchResult(
                qid=qid,
                status="error",
                source_url=source_url,
                error="Top-level JSON is not an object.",
            )

        payload = _extract_entity_payload(decoded, qid)
        if payload is None:
            return FetchResult(
                qid=qid,
                status="error",
                source_url=source_url,
                error="Entity payload missing in response.",
            )

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
        return FetchResult(qid=qid, status="fetched", source_url=source_url, payload=payload)

    return FetchResult(
        qid=qid,
        status="error",
        source_url=source_url,
        error=f"HTTP {last_http_status}: retry budget exhausted" if last_http_status else "Retry budget exhausted",
        http_status=last_http_status,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch live Wikidata Special:EntityData payloads by explicit QIDs and cache them in Postgres."
        )
    )
    parser.add_argument("--postgres-dsn", help="Postgres DSN (defaults to ALPACA_POSTGRES_DSN).")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--ids", help="Comma-separated QIDs (example: Q42,Q90,Q64).")
    parser.add_argument("--ids-file", help="Text file with one QID per line (comments with # allowed).")
    parser.add_argument(
        "--count",
        type=parse_positive_int,
        help="Deterministic count mode: fetches the first N available QIDs by probing upward from Q1.",
    )
    parser.add_argument(
        "--max-probe-id",
        type=parse_positive_int,
        default=50000,
        help="Upper QID probe bound for --count mode (default: 50000).",
    )
    parser.add_argument("--force-refresh", action="store_true", help="Refetch IDs even if already cached in Postgres.")
    parser.add_argument(
        "--max-context-object-ids",
        type=parse_positive_int,
        default=DEFAULT_MAX_CONTEXT_OBJECT_IDS,
        help=(
            "Per-seed cap for one-hop related entity prefetch used to support context-string label "
            f"resolution (default: {DEFAULT_MAX_CONTEXT_OBJECT_IDS})."
        ),
    )
    parser.add_argument("--concurrency", type=parse_positive_int, default=12)
    parser.add_argument("--timeout-seconds", type=parse_non_negative_float, default=20.0)
    parser.add_argument("--sleep-seconds", type=parse_non_negative_float, default=0.1)
    parser.add_argument("--http-max-retries", type=parse_non_negative_int, default=DEFAULT_HTTP_MAX_RETRIES)
    parser.add_argument(
        "--http-retry-backoff-seconds",
        type=parse_non_negative_float,
        default=DEFAULT_HTTP_RETRY_BACKOFF_SECONDS,
    )
    parser.add_argument(
        "--http-retry-max-sleep-seconds",
        type=parse_non_negative_float,
        default=DEFAULT_HTTP_RETRY_MAX_SLEEP_SECONDS,
    )
    parser.add_argument(
        "--max-context-support-prefetch",
        type=parse_non_negative_int,
        default=DEFAULT_MAX_CONTEXT_SUPPORT_PREFETCH,
        help=(
            "Live-demo cap on one-hop support entities fetched for context labels "
            f"(default: {DEFAULT_MAX_CONTEXT_SUPPORT_PREFETCH}, 0 disables support prefetch)."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        selector_count = sum(
            1 for value in (args.ids, args.ids_file) if value
        ) + (1 if args.count is not None else 0)
        if selector_count != 1:
            raise ValueError("Provide exactly one of --ids, --ids-file, or --count.")

        store = PostgresStore(resolve_postgres_dsn(args.postgres_dsn))
        store.ensure_schema()
        count_mode = args.count is not None
        qids: list[str] = []
        if not count_mode:
            qids = resolve_qids(args.ids, args.ids_file, None)

        fetched_rows: list[tuple[str, Mapping[str, Any], str]] = []
        seed_errors: list[FetchResult] = []
        support_errors: list[FetchResult] = []
        nonfatal_not_found = 0
        seed_rate_limited = 0
        seed_transient_skipped = 0
        fetched_count = 0
        cache_hits = 0
        context_support_candidates = 0
        context_support_sampled = 0
        context_support_cache_hits = 0
        context_support_fetched = 0
        context_support_not_found = 0
        context_support_rate_limited = 0

        def _iter_fetch_many(
            missing_qids: list[str],
            *,
            concurrency: int,
            sleep_seconds: float,
        ):
            if not missing_qids:
                return

            max_workers = max(1, int(concurrency))
            max_in_flight = max(
                max_workers,
                max_workers * DEFAULT_SUPPORT_PREFETCH_MAX_IN_FLIGHT_FACTOR,
            )

            def _submit(executor: ThreadPoolExecutor, qid: str):
                return executor.submit(
                    fetch_entity_payload,
                    qid,
                    base_url=args.base_url,
                    timeout_seconds=float(args.timeout_seconds),
                    sleep_seconds=float(sleep_seconds),
                    max_retries=int(args.http_max_retries),
                    retry_backoff_seconds=float(args.http_retry_backoff_seconds),
                    retry_max_sleep_seconds=float(args.http_retry_max_sleep_seconds),
                )

            qid_iter = iter(missing_qids)
            in_flight: set[Any] = set()
            future_to_qid: dict[Any, str] = {}
            executor = ThreadPoolExecutor(max_workers=max_workers)
            try:
                while True:
                    while len(in_flight) < max_in_flight:
                        try:
                            next_qid = next(qid_iter)
                        except StopIteration:
                            break
                        future = _submit(executor, next_qid)
                        in_flight.add(future)
                        future_to_qid[future] = next_qid

                    if not in_flight:
                        break

                    done, pending = wait(in_flight, return_when=FIRST_COMPLETED)
                    in_flight = set(pending)
                    for future in done:
                        qid = future_to_qid.pop(future, "")
                        try:
                            yield future.result()
                        except Exception as exc:
                            # Defensive guard: convert unexpected worker exceptions into fetch errors.
                            yield FetchResult(
                                qid=qid or "UNKNOWN",
                                status="error",
                                source_url="",
                                error=f"Unhandled fetch exception: {exc!r}",
                            )
            finally:
                for future in list(in_flight):
                    future.cancel()
                executor.shutdown(wait=False, cancel_futures=True)

        if not count_mode:
            existing = store.get_sample_entities(qids) if not args.force_refresh else {}
            to_fetch = [qid for qid in qids if qid not in existing]
            cache_hits = len(existing)

            with tqdm(total=len(qids), desc="sample-postgres", unit="entity") as progress:
                if cache_hits:
                    progress.update(cache_hits)
                    progress.set_postfix(cache_hits=cache_hits, fetched=0, errors=0)

                for result in _iter_fetch_many(
                    to_fetch,
                    concurrency=int(args.concurrency),
                    sleep_seconds=float(args.sleep_seconds),
                ):
                    progress.update(1)
                    if result.status == "fetched" and result.payload is not None:
                        fetched_rows.append((result.qid, result.payload, result.source_url))
                        fetched_count += 1
                    else:
                        seed_errors.append(result)
                    progress.set_postfix(cache_hits=cache_hits, fetched=fetched_count, errors=len(seed_errors))

                if fetched_rows:
                    store.upsert_sample_entities(fetched_rows)

            requested = len(qids)
            seed_qids = list(qids)
        else:
            target_count = int(args.count)
            max_probe_id = max(target_count, int(args.max_probe_id))
            selected_qids: list[str] = []
            selected_seen: set[str] = set()
            probe_cursor = 1

            if not args.force_refresh:
                cached_prefill = store.list_sample_entity_ids(limit=target_count)
                if cached_prefill:
                    selected_qids.extend(cached_prefill)
                    selected_seen.update(cached_prefill)
                    cache_hits = len(cached_prefill)

            with tqdm(total=target_count, desc="sample-postgres", unit="entity") as progress:
                if cache_hits:
                    progress.update(cache_hits)
                    progress.set_postfix(
                        cache_hits=cache_hits,
                        fetched=fetched_count,
                        not_found=nonfatal_not_found,
                        rate_limited=seed_rate_limited,
                        skipped=seed_transient_skipped,
                        errors=len(seed_errors),
                    )
                while len(selected_qids) < target_count and probe_cursor <= max_probe_id:
                    chunk_size = max(16, args.concurrency * 4)
                    candidate_qids = [f"Q{value}" for value in range(probe_cursor, probe_cursor + chunk_size)]
                    probe_cursor += chunk_size
                    candidate_qids = [qid for qid in candidate_qids if int(qid[1:]) <= max_probe_id]
                    if not candidate_qids:
                        break

                    existing_batch = (
                        {} if args.force_refresh else store.get_sample_entities(candidate_qids)
                    )
                    for qid in candidate_qids:
                        if qid in existing_batch and qid not in selected_seen and len(selected_qids) < target_count:
                            selected_seen.add(qid)
                            selected_qids.append(qid)
                            cache_hits += 1
                            progress.update(1)
                    missing_batch = [qid for qid in candidate_qids if qid not in existing_batch]

                    if len(selected_qids) >= target_count:
                        progress.set_postfix(
                            cache_hits=cache_hits,
                            fetched=fetched_count,
                            not_found=nonfatal_not_found,
                            rate_limited=seed_rate_limited,
                            skipped=seed_transient_skipped,
                            errors=len(seed_errors),
                        )
                        break

                    batch_rows: list[tuple[str, Mapping[str, Any], str]] = []
                    for result in _iter_fetch_many(
                        missing_batch,
                        concurrency=int(args.concurrency),
                        sleep_seconds=float(args.sleep_seconds),
                    ):
                        if result.status == "fetched" and result.payload is not None:
                            batch_rows.append((result.qid, result.payload, result.source_url))
                            if result.qid not in selected_seen and len(selected_qids) < target_count:
                                selected_seen.add(result.qid)
                                selected_qids.append(result.qid)
                                fetched_count += 1
                                progress.update(1)
                        elif result.http_status == 404:
                            nonfatal_not_found += 1
                        elif result.http_status in (429, 502, 503, 504):
                            seed_rate_limited += 1
                            # Count mode is best-effort by design: skip transiently failing IDs and continue probing.
                            seed_transient_skipped += 1
                        else:
                            seed_errors.append(result)
                        progress.set_postfix(
                            cache_hits=cache_hits,
                            fetched=fetched_count,
                            not_found=nonfatal_not_found,
                            rate_limited=seed_rate_limited,
                            skipped=seed_transient_skipped,
                            errors=len(seed_errors),
                        )

                    if batch_rows:
                        store.upsert_sample_entities(batch_rows)

                if len(selected_qids) < target_count:
                    raise ValueError(
                        f"Could not collect {target_count} live entities by QID probe up to Q{max_probe_id}. "
                        f"Collected={len(selected_qids)}. Increase --max-probe-id."
                    )
            requested = target_count
            seed_qids = list(selected_qids)

        seed_payloads = store.get_sample_entities(seed_qids)
        if len(seed_payloads) < len(seed_qids):
            missing_seeds = [qid for qid in seed_qids if qid not in seed_payloads]
            raise ValueError(
                f"Internal error: {len(missing_seeds)} seed entities are missing from sample_entity_cache "
                "after fetch. Re-run with --force-refresh."
            )

        support_candidates = _collect_related_entity_ids(
            list(seed_payloads.values()),
            max_context_object_ids=int(args.max_context_object_ids),
        )
        seed_qid_set = set(seed_qids)
        support_qids = [qid for qid in support_candidates if qid not in seed_qid_set]
        context_support_candidates = len(support_qids)
        support_cap = int(args.max_context_support_prefetch)
        if support_cap == 0:
            support_qids = []
        elif support_cap > 0:
            support_qids = _deterministic_sample(support_qids, limit=support_cap)
        context_support_sampled = len(support_qids)

        if support_qids:
            existing_support = {} if args.force_refresh else store.get_sample_entities(support_qids)
            support_to_fetch = [qid for qid in support_qids if qid not in existing_support]
            context_support_cache_hits = len(existing_support)
            support_concurrency = max(1, min(int(args.concurrency), DEFAULT_SUPPORT_FETCH_CONCURRENCY_CAP))
            support_sleep_seconds = max(float(args.sleep_seconds), DEFAULT_SUPPORT_FETCH_MIN_SLEEP_SECONDS)

            with tqdm(total=len(support_qids), desc="sample-context", unit="entity") as progress:
                if context_support_cache_hits:
                    progress.update(context_support_cache_hits)
                    progress.set_postfix(
                        cache_hits=context_support_cache_hits,
                        fetched=0,
                        not_found=0,
                        rate_limited=0,
                        errors=0,
                    )

                batch_rows: list[tuple[str, Mapping[str, Any], str]] = []
                support_prefetch_aborted = False
                for result in _iter_fetch_many(
                    support_to_fetch,
                    concurrency=support_concurrency,
                    sleep_seconds=support_sleep_seconds,
                ):
                    progress.update(1)
                    if result.status == "fetched" and result.payload is not None:
                        batch_rows.append((result.qid, result.payload, result.source_url))
                        context_support_fetched += 1
                    elif result.http_status == 404:
                        context_support_not_found += 1
                    elif result.http_status == 429:
                        context_support_rate_limited += 1
                    else:
                        support_errors.append(result)
                    progress.set_postfix(
                        cache_hits=context_support_cache_hits,
                        fetched=context_support_fetched,
                        not_found=context_support_not_found,
                        rate_limited=context_support_rate_limited,
                        errors=len(support_errors),
                    )
                    if context_support_rate_limited >= DEFAULT_SUPPORT_RATE_LIMIT_ABORT_THRESHOLD:
                        support_prefetch_aborted = True
                        break
                if batch_rows:
                    store.upsert_sample_entities(batch_rows)
                if support_prefetch_aborted:
                    print(
                        "WARNING: Aborting remaining context-support prefetch after repeated HTTP 429 rate limits. "
                        "Continuing with partial support cache for live demo responsiveness.",
                        file=sys.stderr,
                    )

        all_errors = [*seed_errors, *support_errors]

        print(
            "Wikidata sample cached in Postgres:",
            f"requested={requested}",
            f"cache_hits={cache_hits}",
            f"fetched={fetched_count}",
            f"not_found={nonfatal_not_found}",
            f"seed_rate_limited={seed_rate_limited}",
            f"seed_transient_skipped={seed_transient_skipped}",
            f"context_support_candidates={context_support_candidates}",
            f"context_support_sampled={context_support_sampled}",
            f"context_support_cache_hits={context_support_cache_hits}",
            f"context_support_fetched={context_support_fetched}",
            f"context_support_not_found={context_support_not_found}",
            f"context_support_rate_limited={context_support_rate_limited}",
            f"seed_errors={len(seed_errors)}",
            f"context_support_errors={len(support_errors)}",
            f"errors_total={len(all_errors)}",
            f"timestamp={datetime.now(timezone.utc).isoformat()}",
        )
        for result in all_errors[:10]:
            print(f"ERROR {result.qid}: {result.error}", file=sys.stderr)
        if support_errors or context_support_rate_limited:
            print(
                "WARNING: Context-support prefetch was partial due to upstream limits/errors. "
                "Pipeline can continue, but some context strings may be less complete on this run.",
                file=sys.stderr,
            )
        if count_mode:
            if seed_errors:
                print(
                    "WARNING: Some seed probes failed with non-transient errors in count mode and were skipped.",
                    file=sys.stderr,
                )
            return 0
        return 0 if not seed_errors else 1
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
