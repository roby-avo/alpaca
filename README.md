# alpaca

![alpaca logo](assets/alpaca-logo.png)

Deterministic entity lookup over Wikidata-style data using:
- PostgreSQL (entity store, search indexes, query cache, live demo sample cache)
- FastAPI (lookup API)
- Adminer (optional UI for inspecting PostgreSQL data)

## Local Stack (No Auth)

Local Docker setup is intentionally passwordless / no-auth for dev:
- Postgres uses `trust`
- Adminer connects to Postgres with empty password (same local dev assumption)

## Version Pinning

Image versions are pinned explicitly in `.env.example`.

Create your local `.env` first:

```bash
cp .env.example .env
```

Then adjust versions in `.env` after checking Docker Hub.

## Start Services

```bash
docker compose pull
docker compose build api
docker compose up -d postgres adminer api
```

Useful URLs:
- API: [http://localhost:8000](http://localhost:8000)
- Adminer (Postgres UI): [http://localhost:8080](http://localhost:8080)

## Full Dump / Production Pipeline (Local Dump)

Run the PostgreSQL pipeline on a local Wikidata dump:

```bash
python3 -m src.run_pipeline --dump-path /absolute/path/latest-all.json.bz2 --workers 8
```

If running inside the API container (recommended with Docker Compose), put the dump under `./data/input` and use `/mnt/input/...`:

```bash
docker compose exec -T api python -m src.run_pipeline \
  --dump-path /mnt/input/latest-all.json.bz2 \
  --workers 8 \
  --pass1-batch-size 5000 \
  --context-batch-size 1000
```

What it does:
1. Ingest dump entities into Postgres (`entities`)
2. Build deterministic `context_string` from related entity labels (pass2)
3. Build/refresh Postgres search indexes (`GIN` FTS + `pg_trgm`)

## Live Demo Sample (Cached in Postgres)

There is no filesystem cache for live demo entities.
Live `Special:EntityData` payloads are cached in Postgres table `sample_entity_cache` by QID.

### One-command flow (recommended)

```bash
./scripts/run_live_sample_pipeline_docker.sh --ids Q42,Q90,Q64,Q60
```

Or use an IDs file:

```bash
./scripts/run_live_sample_pipeline_docker.sh --ids-file ./sample_qids.txt
```

Or use deterministic count mode:

```bash
./scripts/run_live_sample_pipeline_docker.sh --count 120
```

This does:
1. Start local services (`postgres`, `adminer`, `api`)
2. Fetch live Wikidata seed entities into Postgres `sample_entity_cache`
3. Optionally prefetch a capped one-hop support sample for context label resolution
4. Build a compact Wikidata-style dump from the Postgres cache (seed entities only)
5. Run the Postgres pipeline on the generated dump

Useful live demo tuning (to avoid upstream throttling):
- `--concurrency`
- `--sleep-seconds`
- `--http-max-retries`
- `--max-context-support-prefetch`

## PostgreSQL Search / Matching Logic

Candidate retrieval uses PostgreSQL only:
- Exact match over normalized `label`, `aliases`, and `cross_refs`
- Fuzzy match over `label`, `aliases`, `context_string`, and `cross_refs`
- Type filtering via `coarse_type` / `fine_type`
- Item category stored per row (`ENTITY`, `DISAMBIGUATION`, `TYPE`, `PREDICATE`, plus a few non-item fallbacks like `LEXEME`)
- Prior-aware reranking (`prior` + context/type/name scores)
- Optional LLM augmentation via `crosslink_hints` in lookup requests

Internally the `entities` table stores search helper fields (normalized exact text, flattened arrays, and `tsvector`) and builds indexes via `/admin/reindex` or pipeline startup.

## Explore PostgreSQL Data (UI + Stats)

Open [http://localhost:8080](http://localhost:8080) (Adminer) and connect with:
- System: `PostgreSQL`
- Server: `postgres` (from the Adminer container)
- Username: `postgres`
- Password: *(leave empty)*
- Database: `alpaca`

### Useful SQL Queries (Stats + Sanity Checks)

Entity / cache row counts:

```sql
SELECT 'entities' AS table_name, count(*) AS rows FROM entities
UNION ALL
SELECT 'query_cache', count(*) FROM query_cache
UNION ALL
SELECT 'sample_entity_cache', count(*) FROM sample_entity_cache;
```

Table sizes:

```sql
SELECT
  relname AS table_name,
  pg_size_pretty(pg_total_relation_size(relid)) AS total_size,
  pg_size_pretty(pg_relation_size(relid)) AS table_size,
  pg_size_pretty(pg_total_relation_size(relid) - pg_relation_size(relid)) AS index_size
FROM pg_catalog.pg_statio_user_tables
ORDER BY pg_total_relation_size(relid) DESC;
```

Table activity stats:

```sql
SELECT
  relname,
  n_live_tup,
  seq_scan,
  seq_tup_read,
  idx_scan,
  idx_tup_fetch,
  n_tup_ins,
  n_tup_upd
FROM pg_stat_user_tables
ORDER BY n_live_tup DESC;
```

Index usage stats:

```sql
SELECT
  s.relname AS table_name,
  s.indexrelname AS index_name,
  s.idx_scan,
  pg_size_pretty(pg_relation_size(s.indexrelid)) AS index_size
FROM pg_stat_user_indexes s
ORDER BY s.idx_scan DESC, pg_relation_size(s.indexrelid) DESC;
```

Check search indexes on `entities`:

```sql
SELECT indexname, indexdef
FROM pg_indexes
WHERE tablename = 'entities'
ORDER BY indexname;
```

## Storage Estimation (Sample + Replicate)

For demonstrative sizing (without ingesting the full dump), use a small live sample to build proper `context_string`, then replicate the fully materialized `entities` rows into a separate simulation table.

Suggested flow:
1. Build a small but real sample (`sample_entity_cache` + pass1/pass2):
```bash
./scripts/run_live_sample_pipeline_docker.sh --count 500
```
2. Create a simulation table with replicated rows (same row shape, same search columns):
```bash
docker compose exec -T api python -m src.simulate_entities_size \
  --target-rows 5000000 \
  --seed-rows 500 \
  --dest-table entities_size_sim
```
3. Read the output projection (it prints current size and a linear projection to 100M rows by default).

Notes:
- This preserves realistic row format after context has already been built.
- It is still an estimate: very small seeds can underestimate index size because text diversity is lower than full Wikidata.

## API Endpoints

- `GET /healthz`
- `POST /lookup`
- `POST /debug/lookup`
- `POST /admin/reindex`

Example debug lookup with context + crosslink hint:

```bash
curl -s http://localhost:8000/debug/lookup \
  -H 'Content-Type: application/json' \
  -d '{
    "mention":"boston",
    "mention_context":["massachusetts","new england","city","capital"],
    "coarse_hints":["LOCATION"],
    "fine_hints":["CITY"],
    "crosslink_hints":["https://dbpedia.org/resource/Boston"],
    "top_k":10
  }'
```

## Build a Deterministic Small Dump (from a Local Large Dump)

```bash
python3 -m src.build_small_dump \
  --source-dump-path /absolute/path/latest-all.json.bz2 \
  --output-path data/input/small_dump.json.bz2 \
  --ids Q42,Q90,P31
```

## Cleanup

Remove generated input/output artifacts:

```bash
./scripts/clean_repo_artifacts.sh --yes
```

Reset Postgres sample cache only (live demo entities):

```bash
docker compose exec postgres psql -U postgres -d alpaca -c "TRUNCATE TABLE sample_entity_cache;"
```

Reset indexed entities and query cache:

```bash
docker compose exec postgres psql -U postgres -d alpaca -c "TRUNCATE TABLE entities, query_cache;"
```
