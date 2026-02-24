# alpaca

![alpaca logo](assets/alpaca-logo.png)

Deterministic entity lookup over Wikidata-style data using:
- Postgres (entity store + query cache + live demo sample cache)
- Elasticsearch (search/index)
- FastAPI (lookup API)
- Elasticvue (UI for exploring Elasticsearch data)

## Local Stack (No Auth)

Local Docker setup is intentionally passwordless / no-auth for dev:
- Postgres uses `trust`
- Elasticsearch has security disabled

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
docker compose up -d postgres elasticsearch elasticvue api
```

Useful URLs:
- API: [http://localhost:8000](http://localhost:8000)
- Elasticvue (ES UI): [http://localhost:8080](http://localhost:8080)
- Elasticsearch: [http://localhost:9200](http://localhost:9200)

## Live Demo Sample (Cached in Postgres)

There is no filesystem cache for live demo entities anymore.
Live `Special:EntityData` payloads are cached in Postgres table `sample_entity_cache` by QID.

### One-command flow (recommended)

```bash
./scripts/run_live_sample_pipeline_docker.sh --ids Q42,Q90,Q64,Q60
```

Or use an IDs file:

```bash
./scripts/run_live_sample_pipeline_docker.sh --ids-file ./sample_qids.txt
```

Or use deterministic count mode (probes upward from `Q1`, skipping missing IDs):

```bash
./scripts/run_live_sample_pipeline_docker.sh --count 120
```

This does:
1. Fetch live Wikidata seed entities into Postgres `sample_entity_cache`
2. Prefetch one-hop related entity IDs from claim objects into `sample_entity_cache` (for context label resolution)
3. Build a compact Wikidata-style dump from that Postgres cache (seed entities only)
4. Run the Postgres -> deterministic context build -> Elasticsearch indexing pipeline
5. Bring the API online

### Step-by-step (manual)

Fetch live entities into Postgres sample cache (and prefetch one-hop related entities for context support):

```bash
python3 -m src.wikidata_sample_postgres --ids Q42,Q90,Q64,Q60 --force-refresh
```

Deterministic count mode:

```bash
python3 -m src.wikidata_sample_postgres --count 120 --force-refresh
```

Build sample dump from Postgres sample cache:

```bash
python3 -m src.build_postgres_sample_dump \
  --ids Q42,Q90,Q64,Q60 \
  --output-path data/input/live_sample_dump.json.bz2
```

Or with deterministic count:

```bash
python3 -m src.build_postgres_sample_dump \
  --count 120 \
  --output-path data/input/live_sample_dump.json.bz2
```

Run indexing pipeline:

```bash
python3 -m src.run_pipeline \
  --dump-path data/input/live_sample_dump.json.bz2 \
  --replace-elasticsearch-index \
  --workers 4
```

## Full Dump / Local Dump Subset

Run the Elasticsearch pipeline on a local Wikidata dump:

```bash
python3 -m src.run_pipeline --dump-path /absolute/path/latest-all.json.bz2 --replace-elasticsearch-index
```

Build a deterministic small dump from a local large dump (for smoke tests):

```bash
python3 -m src.build_small_dump \
  --source-dump-path /absolute/path/latest-all.json.bz2 \
  --output-path data/input/small_dump.json.bz2 \
  --ids Q42,Q90,P31
```

## Explore Elasticsearch Data (UI)

Open [http://localhost:8080](http://localhost:8080) and use the preconfigured cluster (`alpaca-local`).

Indexed source documents use:
- `label` (primary label)
- `labels` (flat array of label strings, not language-keyed)
- `aliases` (flat array of alias strings)
- `context_string`, `coarse_type`, `fine_type`, `prior`, `cross_refs`

Useful Elasticvue queries:

```json
{"query":{"match_all":{}},"size":20}
```

```json
{"query":{"match":{"label":"boston"}},"size":20}
```

```json
{"query":{"term":{"qid":"Q90"}}}
```

## API Endpoints

- `GET /healthz`
- `POST /lookup`
- `POST /debug/lookup`
- `POST /admin/reindex`

Example debug lookup with context:

```bash
curl -s http://localhost:8000/debug/lookup \
  -H 'Content-Type: application/json' \
  -d '{
    "mention":"boston",
    "mention_context":["massachusetts","new england","city","capital"],
    "coarse_hints":["LOCATION"],
    "fine_hints":["CITY"],
    "top_k":10
  }'
```

## Exact Matching (Why no `label_norm` / `alias_norms` fields anymore)

Exact matching now uses Elasticsearch normalized keyword subfields:
- `label.exact`
- `aliases.exact`

This keeps the source documents cleaner and avoids duplicating normalized values in separate top-level fields.

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
