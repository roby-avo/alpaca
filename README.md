# alpaca

![alpaca logo](assets/alpaca-logo.png)

Streaming Wikidata preprocessing + Quickwit indexing for entity retrieval.

This repo is documented around exactly **two high-level run options**.

## Requirements

- Python 3.12
- Docker + Docker Compose
- A running Quickwit + API stack:

```bash
docker compose build api
docker compose up -d quickwit api
```

## Option 1: Live Sample Dump (from live Wikidata entities)

Use this when you want a fast realistic run on `N` live entity items.

Command:

```bash
./scripts/run_live_sample_pipeline_docker.sh --count 1000 --force-refresh
```

What it does:

1. Fetches live entities from Wikidata `Special:EntityData` into cache.
2. Builds a compact sample dump from cached live JSON.
3. Runs full indexing pipeline (labels DB -> BOW docs -> Quickwit ingest).

Default sample dump path (fixed name regardless of count):

- `data/input/live_sample_dump.json.bz2`

Useful flags:

- `--ids Q42,Q90,Q30` instead of `--count`
- `--index-id wikidata_entities_live_custom`
- `--cache-dir ...`
- `--output-dump ...`

## Option 2: Full Dump Indexing (local downloaded full dump)

Use this for full-scale indexing from your downloaded dump.

Command:

```bash
./scripts/run_full_dump_pipeline_docker.sh \
  --dump-path /absolute/path/latest-all.json.bz2
```

What it does:

1. Builds labels DB (streaming).
2. Builds BOW docs (streaming).
3. Ingests docs into Quickwit.
4. Rebinds API to the new index.
5. Verifies indexed docs + API health.

Useful flags:

- `--index-id wikidata_entities_full_custom`
- `--output-dir ./data/output`
- `--labels-batch-size 2000`
- `--bow-batch-size 2000`
- `--quickwit-chunk-bytes 4000000`
- `ALPACA_LANGUAGES=en` (default; compact labels DB)
- `ALPACA_MAX_BOW_TOKENS=128`
- `ALPACA_MAX_CONTEXT_OBJECT_IDS=32`
- `ALPACA_MAX_CONTEXT_CHARS=640`
- `ALPACA_MAX_DOC_BYTES=4096`

## Progress Tracking (every heavy step)

Progress bars (`tqdm`) are shown for:

- live fetch: `wikidata-cache`
- sample dump build: `cached-sample-dump`
- labels DB: `labels-db`
- BOW docs: `bow-docs`
- Quickwit ingest: `quickwit-ingest`

You also get explicit `Step X/Y` logs in the wrapper scripts.

For streaming steps where the exact total is unknown upfront, Alpaca computes a fast estimate from a small front-of-file sample and file size (constant-time metadata + bounded sample read), then uses that estimate as `tqdm total`.
For compressed dumps (`.bz2` / `.gz`), estimate quality is improved by sampling consumed **compressed bytes** directly instead of relying on fixed ratio guesses.

## Memory Safety (30 GB RAM VM)

Both pipelines are streaming and do not load the full dump into memory.

The full-dump wrapper uses conservative defaults designed for large runs on ~30GB RAM:

- `--labels-batch-size 2000`
- `--bow-batch-size 2000`
- `--quickwit-chunk-bytes 4000000`
- `ALPACA_LANGUAGES=en`
- `ALPACA_MAX_ALIASES_PER_LANGUAGE=8`
- `ALPACA_MAX_BOW_TOKENS=128`
- `ALPACA_MAX_CONTEXT_OBJECT_IDS=32`
- `ALPACA_MAX_CONTEXT_CHARS=640`
- `ALPACA_MAX_DOC_BYTES=4096`
- no `--limit` for full runs (true full indexing)

Additional notes:

- SQLite is WAL-backed with bounded cache settings.
- Input dump is parsed line-by-line.
- Quickwit ingest is chunked NDJSON, not one giant payload.
- Output docs are size-capped and include compact `context` built from linked-object English labels (not predicate IDs).

## Troubleshooting

If a script says required Docker services are not running, start them first:

```bash
docker compose up -d quickwit api
```

The API container bind-mounts local `src/`, `scripts/`, and `quickwit/`, so Python code changes are picked up immediately without rebuild.

Rebuild only when dependencies or Dockerfile layers change:

```bash
docker compose build api
```

To remove all Quickwit indexes quickly:

```bash
./scripts/delete_quickwit_indexes.sh --yes
```

To preview before deleting:

```bash
./scripts/delete_quickwit_indexes.sh --dry-run --prefix wikidata_entities
```

## Cleanup

Remove generated artifacts while preserving `.gitkeep` files:

```bash
./scripts/clean_repo_artifacts.sh --yes
```

Also remove cached live entities:

```bash
./scripts/clean_repo_artifacts.sh --yes --include-cache
```
