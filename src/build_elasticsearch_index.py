from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import unicodedata
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import Any

from .common import (
    ELASTICSEARCH_INDEX_ENV,
    ELASTICSEARCH_URL_ENV,
    POSTGRES_DSN_ENV,
    finalize_tqdm_total,
    keep_tqdm_total_ahead,
    resolve_elasticsearch_index,
    resolve_elasticsearch_url,
    resolve_postgres_dsn,
    tqdm,
)
from .elasticsearch_client import ElasticsearchClient, ElasticsearchClientError
from .postgres_store import PostgresStore


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


def normalize_exact_text(text: str) -> str:
    value = unicodedata.normalize("NFC", text).casefold().strip()
    ascii_folded = unicodedata.normalize("NFKD", value)
    ascii_folded = "".join(ch for ch in ascii_folded if not unicodedata.combining(ch))
    compact = []
    last_space = False
    for ch in ascii_folded:
        if ch.isalnum():
            compact.append(ch)
            last_space = False
            continue
        if not last_space:
            compact.append(" ")
            last_space = True
    return "".join(compact).strip()


def _flatten_aliases(aliases_by_lang: Mapping[str, Sequence[str]]) -> list[str]:
    flattened: list[str] = []
    seen: set[str] = set()
    if "en" in aliases_by_lang:
        ordered_langs = ["en", *sorted(lang for lang in aliases_by_lang if lang != "en")]
    else:
        ordered_langs = sorted(aliases_by_lang)
    for language in ordered_langs:
        values = aliases_by_lang.get(language, [])
        for alias in values:
            if not isinstance(alias, str):
                continue
            candidate = alias.strip()
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            flattened.append(candidate)
    return flattened


def _flatten_labels(labels_by_lang: Mapping[str, str]) -> list[str]:
    flattened: list[str] = []
    seen: set[str] = set()
    if "en" in labels_by_lang:
        ordered_langs = ["en", *sorted(lang for lang in labels_by_lang if lang != "en")]
    else:
        ordered_langs = sorted(labels_by_lang)
    for language in ordered_langs:
        value = labels_by_lang.get(language)
        if not isinstance(value, str):
            continue
        candidate = value.strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        flattened.append(candidate)
    return flattened


def popularity_to_prior(popularity: float) -> float:
    value = max(0.0, float(popularity))
    # Keeps scores in [0,1) for large sitelink counts and is deterministic without corpus-wide stats.
    return 1.0 - math.exp(-math.log1p(value) / 6.0)


def build_index_document(entity: Mapping[str, Any]) -> dict[str, Any]:
    label = entity.get("label") if isinstance(entity.get("label"), str) else ""
    labels_by_lang = entity.get("labels") if isinstance(entity.get("labels"), Mapping) else {}
    labels_flat = _flatten_labels(labels_by_lang)
    aliases_by_lang = entity.get("aliases") if isinstance(entity.get("aliases"), Mapping) else {}
    aliases_flat = _flatten_aliases(aliases_by_lang)  # preserves deterministic language ordering
    popularity = float(entity.get("popularity", 0.0))
    return {
        "qid": entity.get("qid"),
        "label": label,
        "labels": labels_flat,
        "aliases": aliases_flat,
        "context_string": entity.get("context_string") if isinstance(entity.get("context_string"), str) else "",
        "coarse_type": entity.get("coarse_type") if isinstance(entity.get("coarse_type"), str) else "",
        "fine_type": entity.get("fine_type") if isinstance(entity.get("fine_type"), str) else "",
        "popularity": popularity,
        "prior": popularity_to_prior(popularity),
        "cross_refs": entity.get("cross_refs") if isinstance(entity.get("cross_refs"), Mapping) else {},
    }


def default_index_config() -> dict[str, Any]:
    return {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "refresh_interval": "30s",
            "analysis": {
                "analyzer": {
                    "alpaca_text": {
                        "tokenizer": "standard",
                        "filter": ["lowercase", "asciifolding"],
                    }
                },
                "normalizer": {
                    "alpaca_exact": {
                        "type": "custom",
                        "filter": ["lowercase", "asciifolding"],
                    }
                }
            },
        },
        "mappings": {
            "dynamic": "strict",
            "properties": {
                "qid": {"type": "keyword"},
                "label": {
                    "type": "text",
                    "analyzer": "alpaca_text",
                    "fields": {
                        "exact": {
                            "type": "keyword",
                            "normalizer": "alpaca_exact",
                        }
                    },
                },
                "aliases": {
                    "type": "text",
                    "analyzer": "alpaca_text",
                    "fields": {
                        "exact": {
                            "type": "keyword",
                            "normalizer": "alpaca_exact",
                        }
                    },
                },
                "labels": {"type": "keyword"},
                "context_string": {"type": "text", "analyzer": "alpaca_text"},
                "coarse_type": {"type": "keyword"},
                "fine_type": {"type": "keyword"},
                "popularity": {"type": "float"},
                "prior": {"type": "float"},
                "cross_refs": {"type": "object", "enabled": False},
            },
        },
    }


def wait_for_elasticsearch(
    client: ElasticsearchClient,
    *,
    timeout_seconds: float,
    poll_interval_seconds: float,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while True:
        if client.is_healthy():
            return
        if time.monotonic() >= deadline:
            raise ElasticsearchClientError(
                f"Elasticsearch did not become healthy at {client.base_url} "
                f"within {timeout_seconds:.1f}s"
            )
        time.sleep(max(0.2, poll_interval_seconds))


def iter_bulk_payloads_from_store(
    store: PostgresStore,
    *,
    fetch_batch_size: int,
    max_chunk_bytes: int,
    index_name: str,
) -> Iterator[tuple[str, int]]:
    if max_chunk_bytes <= 0:
        raise ValueError("max_chunk_bytes must be > 0")
    lines: list[str] = []
    current_bytes = 0
    doc_count = 0
    for entities in store.iter_entities_for_indexing(batch_size=fetch_batch_size):
        for entity in entities:
            source_doc = build_index_document(entity)
            qid = source_doc.get("qid")
            if not isinstance(qid, str) or not qid:
                continue
            meta_line = json.dumps({"index": {"_index": index_name, "_id": qid}}, separators=(",", ":"))
            doc_line = json.dumps(source_doc, ensure_ascii=False, separators=(",", ":"))
            pair = f"{meta_line}\n{doc_line}\n"
            pair_bytes = len(pair.encode("utf-8"))
            if pair_bytes > max_chunk_bytes:
                raise ValueError(
                    f"Single document bulk payload for '{qid}' exceeds --chunk-bytes={max_chunk_bytes}."
                )
            if lines and current_bytes + pair_bytes > max_chunk_bytes:
                yield "".join(lines), doc_count
                lines = []
                current_bytes = 0
                doc_count = 0
            lines.append(pair)
            current_bytes += pair_bytes
            doc_count += 1
    if lines:
        yield "".join(lines), doc_count


def _bulk_worker(
    elasticsearch_url: str,
    index_name: str,
    payload: str,
    *,
    timeout_seconds: float,
) -> int:
    client = ElasticsearchClient(elasticsearch_url, timeout_seconds=timeout_seconds)
    response = client.bulk(index_name, payload, refresh=False)
    items = response.get("items")
    if isinstance(items, list):
        return len(items)
    return 0


def ingest_all_chunks_parallel(
    *,
    store: PostgresStore,
    elasticsearch_url: str,
    index_name: str,
    fetch_batch_size: int,
    chunk_bytes: int,
    worker_count: int,
    http_timeout_seconds: float,
) -> int:
    workers = max(1, worker_count)
    total_docs = store.count_entities()
    ingested_total = 0
    in_flight: set[Future[int]] = set()

    def _drain_ready(*, block: bool) -> None:
        nonlocal ingested_total
        if not in_flight:
            return
        if block:
            done, pending = wait(in_flight, return_when=FIRST_COMPLETED)
        else:
            done = {future for future in in_flight if future.done()}
            pending = in_flight - done
            if not done:
                return
        in_flight.clear()
        in_flight.update(pending)
        for future in done:
            ingested_total += future.result()

    chunk_iter = iter_bulk_payloads_from_store(
        store,
        fetch_batch_size=fetch_batch_size,
        max_chunk_bytes=chunk_bytes,
        index_name=index_name,
    )
    with tqdm(total=total_docs or None, desc="es-index", unit="doc") as progress:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for payload, doc_count in chunk_iter:
                in_flight.add(
                    executor.submit(
                        _bulk_worker,
                        elasticsearch_url,
                        index_name,
                        payload,
                        timeout_seconds=http_timeout_seconds,
                    )
                )
                if len(in_flight) >= max(1, workers * 2):
                    before = ingested_total
                    _drain_ready(block=True)
                    delta = max(0, ingested_total - before)
                    if delta:
                        progress.update(delta)
                        keep_tqdm_total_ahead(progress)
                        progress.set_postfix(ingested=ingested_total, queued=len(in_flight))
                else:
                    _drain_ready(block=False)
                    if ingested_total > progress.n:
                        progress.update(int(ingested_total - progress.n))
                        keep_tqdm_total_ahead(progress)
                _ = doc_count  # keeps intent explicit; actual count confirmed by ES bulk response
            while in_flight:
                before = ingested_total
                _drain_ready(block=True)
                delta = max(0, ingested_total - before)
                if delta:
                    progress.update(delta)
                    keep_tqdm_total_ahead(progress)
                    progress.set_postfix(ingested=ingested_total, queued=len(in_flight))
        finalize_tqdm_total(progress)
    return ingested_total


def run(
    *,
    postgres_dsn: str,
    elasticsearch_url: str,
    index_name: str,
    replace_index: bool,
    fetch_batch_size: int,
    chunk_bytes: int,
    worker_count: int,
    wait_timeout_seconds: float,
    poll_interval_seconds: float,
    http_timeout_seconds: float,
) -> int:
    store = PostgresStore(postgres_dsn)
    store.ensure_schema()
    client = ElasticsearchClient(elasticsearch_url, timeout_seconds=http_timeout_seconds)

    wait_for_elasticsearch(
        client,
        timeout_seconds=wait_timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
    )
    client.ensure_index(index_name, default_index_config(), replace=replace_index)

    ingested = ingest_all_chunks_parallel(
        store=store,
        elasticsearch_url=elasticsearch_url,
        index_name=index_name,
        fetch_batch_size=fetch_batch_size,
        chunk_bytes=chunk_bytes,
        worker_count=worker_count,
        http_timeout_seconds=http_timeout_seconds,
    )
    client.refresh_index(index_name)
    print(
        "Completed Elasticsearch indexing:",
        f"index={index_name}",
        f"ingested={ingested}",
        f"workers={max(1, worker_count)}",
        f"chunk_bytes={chunk_bytes}",
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build/refresh Elasticsearch index from Postgres entities table."
    )
    parser.add_argument("--postgres-dsn", help=f"Postgres DSN. Overrides ${POSTGRES_DSN_ENV}.")
    parser.add_argument(
        "--elasticsearch-url",
        help=f"Elasticsearch base URL. Overrides ${ELASTICSEARCH_URL_ENV}.",
    )
    parser.add_argument(
        "--index-name",
        help=f"Elasticsearch index name. Overrides ${ELASTICSEARCH_INDEX_ENV}.",
    )
    parser.add_argument("--replace-index", action="store_true", help="Delete and recreate the index.")
    parser.add_argument("--fetch-batch-size", type=parse_positive_int, default=2000)
    parser.add_argument("--chunk-bytes", type=parse_positive_int, default=8_000_000)
    parser.add_argument(
        "--workers",
        type=parse_positive_int,
        default=max(1, min(8, (os.cpu_count() or 1))),
    )
    parser.add_argument("--wait-timeout-seconds", type=parse_non_negative_float, default=120.0)
    parser.add_argument("--poll-interval-seconds", type=parse_non_negative_float, default=2.0)
    parser.add_argument("--http-timeout-seconds", type=parse_positive_int, default=120)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        return run(
            postgres_dsn=resolve_postgres_dsn(args.postgres_dsn),
            elasticsearch_url=resolve_elasticsearch_url(args.elasticsearch_url),
            index_name=resolve_elasticsearch_index(args.index_name),
            replace_index=args.replace_index,
            fetch_batch_size=args.fetch_batch_size,
            chunk_bytes=args.chunk_bytes,
            worker_count=args.workers,
            wait_timeout_seconds=args.wait_timeout_seconds,
            poll_interval_seconds=args.poll_interval_seconds,
            http_timeout_seconds=float(args.http_timeout_seconds),
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
