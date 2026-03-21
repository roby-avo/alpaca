# alpaca

![alpaca logo](assets/alpaca-logo.png)

Deterministic entity lookup over Wikidata-style data using:
- PostgreSQL (entity store, triples store, query cache, live demo sample cache)
- Elasticsearch (optional external index for integration experiments)
- FastAPI (lookup API)
- Adminer (optional UI for inspecting PostgreSQL data)
- Elasticvue (UI for inspecting Elasticsearch data)

## Local Stack (No Auth)

Local Docker setup is intentionally passwordless / no-auth for dev:
- Postgres uses `trust`
- Elasticsearch security is disabled (`xpack.security.enabled=false`)
- Elasticvue connects directly to Elasticsearch over CORS
- Adminer connects to Postgres with empty password (same local dev assumption)
- Postgres defaults to `max_connections=200` (as requested for compatibility)
- Postgres keeps defaults for most internals; only `shm_size` and `max_connections`
  are explicitly configured in `.env.example`

Elasticvue is kept intentionally simple here:
- waits for Elasticsearch health before booting
- preloads a single local cluster at `http://localhost:9200`

## Version Pinning

Image versions are pinned explicitly in `.env.example`.

Create your local `.env` first:

```bash
cp .env.example .env
```

Then adjust versions in `.env` after checking Docker Hub / Elastic registry.

## VM Postgres Rollout

Before running one-off maintenance on a large `entities` table, deploy the Compose
Postgres config update first so the instance restarts with the current values from
`.env.example`.

Suggested order:
1. Update `.env` on the VM (`ALPACA_POSTGRES_SHM_SIZE`, `ALPACA_POSTGRES_MAX_CONNECTIONS`).
2. Restart the Postgres service: `docker compose up -d postgres`
3. Confirm the live setting: `SHOW max_connections;`
4. Run your one-off `entities` maintenance only after the new settings are live

## Start Services

```bash
docker compose pull
docker compose build api
docker compose up -d postgres adminer elasticsearch elasticvue api
```

Useful URLs:
- API: [http://localhost:8000](http://localhost:8000)
- Adminer (Postgres UI): [http://localhost:8080](http://localhost:8080)
- Elasticsearch API: [http://localhost:9200](http://localhost:9200)
- Elasticvue (Elasticsearch UI): [http://localhost:5601](http://localhost:5601)

Elasticvue is intentionally kept minimal here. Open the preloaded cluster and use
the REST query tools directly instead of carrying a full Kibana setup.

## Full Dump / Production Pipeline (Local Dump)

Run the PostgreSQL pipeline on a local Wikidata dump:

```bash
python3 -m src.run_pipeline \
  --dump-path /absolute/path/latest-all.json.bz2 \
  --fetch-live-entity-total \
  --workers 8
```

If running inside the API container (recommended with Docker Compose), put the dump under `./data/input` and use `/mnt/input/...`:

```bash
docker compose exec -T api python -m src.run_pipeline \
  --dump-path /mnt/input/latest-all.json.bz2 \
  --fetch-live-entity-total \
  --workers 8 \
  --pass1-batch-size 5000
```

Use `--fetch-live-entity-total` for a production/full-dump run when you want the
progress bar to track the current Wikidata item count from `Wikidata:Statistics`
instead of a local compressed-size estimate.

What it does:
1. Ingest dump entities into Postgres (`entities`) as the single intermediate storage table
2. Store a pruned, context-oriented subset of entity-to-entity Wikidata triples in Postgres (`entity_triples`)
3. Build `context_string` lazily from neighboring entity labels when lookup/export needs it
4. Build/refresh lean Postgres support indexes (`label` / cross-ref / type + incoming triples index on `entity_triples(object_qid, predicate_pid, subject_qid)`)

Default triple pruning keeps up to 12 edges per entity and up to 2 per predicate.
Selection is scored for context usefulness: statement rank, predicate priors, and subject kind
(for example people prefer occupation/citizenship over generic `instance of`, and locations
prefer country/admin-area edges).
Use `--max-entity-triples 0 --max-entity-triples-per-predicate 0` to keep all deduped edges.

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
4. Run the Postgres pipeline directly from `sample_entity_cache` (no temporary dump is written)

Useful live demo tuning (to avoid upstream throttling):
- `--concurrency`
- `--sleep-seconds`
- `--http-max-retries`
- `--max-context-support-prefetch`

Quick one-off NER typing check for a live entity:

```bash
python -m src.test_live_ner_type --qid Q42 --pretty
```

Quick one-off BOW check for cached QIDs:

```bash
./scripts/test_qid_bow.sh --ids Q42,Q90
```

This helper emits a single graph-derived `bow` built from `entity_triples` by resolving and tokenizing:
- outgoing predicate labels
- outgoing neighbor labels
- incoming predicate labels
- incoming neighbor labels

## Mirror PostgreSQL Entities to Elasticsearch

Elasticsearch indexing reads directly from PostgreSQL `entities`, so the same
single table used for parsing/intermediate storage is also the export source.

The production Elasticsearch mapping is intentionally lean:
- indexed text stays focused on `label`, secondary `labels`, `aliases`, and compact triples-backed `context_string`
- `description`, `types`, `wikipedia_url`, and `dbpedia_url` are still returned in `_source`, but are not indexed
- secondary names are clipped before export (defaults: `--max-indexed-labels 12`, `--max-indexed-aliases 24`)
- graph context is clipped before export (default: `--max-context-chars 256`)

Start required services first:

```bash
docker compose up -d postgres elasticsearch api
```

Recreate and fully mirror the index:

```bash
docker compose exec api python -m src.index_postgres_to_elasticsearch \
  --index-name alpaca-entities \
  --recreate-index \
  --workers 4 \
  --batch-size 10000 \
  --bulk-actions 2000 \
  --skip-count-total
```

Inside the `api` container, the default endpoints are the Docker Compose service
names:
- PostgreSQL: `postgresql://postgres@postgres:5432/alpaca`
- Elasticsearch: `http://elasticsearch:9200`

Incremental sync (only rows updated after a timestamp):

```bash
docker compose exec api python -m src.index_postgres_to_elasticsearch \
  --index-name alpaca-entities \
  --updated-since '2026-03-01T00:00:00Z'
```

Note: use `docker compose exec` (without `-T`) to keep TTY enabled so `tqdm`
renders the live progress bar.
For a full 100M+ row mirror, `--skip-count-total` is recommended so indexing
starts immediately instead of paying for an upfront `COUNT(*)`.

Local helper script (same module):

```bash
./scripts/run_postgres_to_elasticsearch.sh --index-name alpaca-entities --recreate-index --skip-count-total
```

The helper starts `postgres`, `elasticsearch`, and `api`, then runs the indexer
inside the `api` container on the Docker Compose network with explicit
Compose-network endpoints for Postgres and Elasticsearch.

## PostgreSQL Search / Matching Logic

Candidate retrieval uses PostgreSQL only:
- Candidate generation from direct `label` / `labels[]` / `aliases[]` matching plus exact cross-ref matches
- Reranking with lexical similarity over `label`, secondary names, and graph-neighborhood context
- Type filtering via `coarse_type` / `fine_type`
- Item category stored per row (`ENTITY`, `DISAMBIGUATION`, `TYPE`, `PREDICATE`, plus a few non-item fallbacks like `LEXEME`)
- Prior-aware reranking (`prior` + context/type/name scores)
- Optional LLM augmentation via `crosslink_hints` in lookup requests

Internally the `entities` table stores explicit multilingual `labels[]` and `aliases[]`, while `entity_triples` stores a pruned `(subject_qid, predicate_pid, object_qid)` edge set used to build compact neighbor-label context.

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
SELECT 'entity_triples', count(*) FROM entity_triples
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

Check support indexes on `entities` / `entity_triples`:

```sql
SELECT indexname, indexdef
FROM pg_indexes
WHERE tablename IN ('entities', 'entity_triples')
ORDER BY indexname;
```

## Explore Elasticsearch Data (UI + Quick Checks)

Open [http://localhost:5601](http://localhost:5601) (Elasticvue).

Elasticvue starts after Elasticsearch is healthy and preloads
`http://localhost:9200` as the local cluster. The current Elasticsearch CORS
settings are already enough for this no-auth local setup.

Quick index inspection commands:

```bash
curl -s http://localhost:9200/_cat/indices?v
```

```bash
curl -s http://localhost:9200/<index_name>/_search \
  -H 'Content-Type: application/json' \
  -d '{"size":5,"query":{"match_all":{}}}'
```

## Storage Estimation (Sample + Replicate)

For demonstrative sizing (without ingesting the full dump), use a small live sample to build a realistic `entities` + `entity_triples` footprint, then replicate the `entities` rows into a separate simulation table.

Suggested flow:
1. Build a small but real sample (`sample_entity_cache` + pass1):
```bash
./scripts/run_live_sample_pipeline_docker.sh --count 500
```
2. Create a simulation table with replicated rows (same row shape, same search columns):
```bash
docker compose exec -T api python -m src.simulate_entities_size \
  --target-rows 5000000 \
  --seed-rows 500 \
  --random-seed 1337 \
  --dest-table entities_size_sim
```
3. Read the output projection (it prints current entity-table size plus a linear `entity_triples` projection to 100M rows by default).

Notes:
- Entities are estimated from deterministic random sampling with replacement from the selected seed rows.
- `entity_triples` are projected linearly from the sampled entities/triples ratio and the sampled triples table bytes.
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
docker compose exec postgres psql -U postgres -d alpaca -c "TRUNCATE TABLE entity_triples, entities, query_cache;"
```
