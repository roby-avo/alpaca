# backend_lab

`backend_lab` is a read-only research interface for building and evaluating CRIA-style cell-level entity linking over the Alpaca backend.

It does not change the Alpaca backend, Elasticsearch index, PostgreSQL database, or Docker services. It only reads local benchmark files, sends read-only lookup/search requests, and optionally calls an OpenAI-compatible LLM provider for preprocessing or ambiguity resolution.

## What This Module Adds

- Dataset utilities for `Coverage_exp/Datasets`.
- Safe backend inspection commands for the Alpaca API and Elasticsearch.
- Table-aware schema induction before per-cell linking.
- LLM-assisted cell preprocessing that converts table evidence into a retrieval plan.
- Hypothesis-aware Elasticsearch retrieval with automatic backoff.
- Deterministic CRIA reranking with auditable features and confidence scores.
- Optional LLM adjudication for hard ambiguity cases.
- Evaluation commands for CEA accuracy, retrieval recovery, cache status, and JSON/CSV export.

## Algorithm Names

The current experimental variants use these names:

| Name | Meaning |
| --- | --- |
| `Baseline` | Direct or simple deterministic retrieval/ranking without the full CRIA table-aware pipeline. |
| `CRIA-deterministic-only` | Table-aware preprocessing, hypothesis-aware retrieval, deterministic CRIA reranking, and no LLM adjudicator after retrieval. |
| `CRIA-LLM-only` | Candidate set is retrieved first, then an LLM performs the full final ranking/selection over a compact candidate pack. |
| `CRIA-deterministic+LLM` | Deterministic CRIA performs the normal ranking; an LLM adjudicator is called only when the edge detector finds a hard ambiguity. |

The preferred scalable setting is usually `CRIA-deterministic+LLM`: deterministic and auditable by default, but with an LLM available for edge cases such as same-label geographic entities, underspecified labels, authority conflicts, and low-margin decisions.

## Pipeline Overview

1. Context building
   - Read the table cell, row values, column samples, headers, and target metadata.
   - Build a compact context object for one CEA target cell.

2. Table profile induction
   - Build deterministic statistics per column: value shape, numeric/date cues, entity-likeness, row regularities, and cross-column patterns.
   - Optionally run one LLM refinement pass per table over the deterministic summary.
   - Produce `table_profile` with `table_semantic_family`, ranked `column_roles`, `row_template`, table hypotheses, confidence, and evidence notes.

3. Cell preprocessing
   - Use cell text, row context, column context, and table profile to create ranked `entity_hypotheses`.
   - Detect weak mentions such as short strings, one-token common names, high-polysemy labels, or low-specificity mentions.
   - Decide whether type evidence should become a hard filter or remain a soft hint.

4. Hypothesis-aware retrieval
   - Query Elasticsearch using canonical mentions, query variants, top semantic hypotheses, and table-aware context terms.
   - Use up to `--top-k 100` candidates per mention in the main experiments.
   - Merge candidates across hypotheses and deduplicate by QID.
   - Run automatic backoff if the first query is too narrow.

5. Deterministic CRIA reranking
   - Score candidates using lexical fit, type/schema alignment, row/table context compatibility, authority, popularity, prior, primary-vs-derivative features, and unsupported qualifier penalties.
   - Produce a top candidate, confidence, margin, reason codes, and abstention signal.

6. Edge detection
   - Detect cases where the deterministic result is structurally risky.
   - Trigger families include low margin, low confidence, abstention, same-label cluster, same-type cluster, authority conflict, low-authority top candidate, derivative/list-like candidate, cleaner-label competitor, and underspecified-label competitor.

7. CRIA-deterministic+LLM adjudication
   - Build a diversified candidate pack from the reranked list.
   - Give the LLM only the table/cell context and candidate pack.
   - The LLM can select one supplied QID or abstain; it cannot invent entities.
   - Merge the adjudicator decision back through deterministic confidence rules.

## Hard Filters vs Soft Hints

Hard filters shrink the search space. They are useful only when evidence is strong and consistent.

Examples of hard-filter evidence:

- The mention is intrinsically specific, such as `Caspian Sea`.
- The table profile has high confidence.
- Cell, row, and table evidence agree on the type.
- A body-of-water column plus lake-area/depth/volume columns supports `LOCATION/LANDMARK`.

Soft hints preserve recall. They are safer when the mention is ambiguous.

Examples of soft-hint evidence:

- One-token mentions such as `Victoria`, `Malawi`, or `Georgia`.
- Weak or synthetic headers such as `col0`, `col1`.
- Conflicting evidence between label, row, and table profile.
- Common names that can refer to people, places, organizations, works, or administrative regions.

This is central to generalizability: weak mentions should not be overcommitted to one brittle type before candidate retrieval.

## Read-Only Safety

The commands in this folder are intended to be safe for shared backend inspection:

- No schema migrations.
- No writes to PostgreSQL.
- No writes to Elasticsearch.
- No writes to the Alpaca API.
- Generated experiment outputs go under `backend_lab/out/`, which is ignored by git.
- LLM caches go under `backend_lab/.cache/`, which is ignored by git.

## Setup

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r backend_lab/requirements.txt
cp .env.example .env
```

Edit `.env` with your backend endpoints and LLM provider credentials.

The CLI automatically loads a repo-level `.env` file via `python-dotenv`.

## Environment Variables

Important backend settings:

```bash
ALPACA_API_URL=http://roberto-vm.vpn.sintef:8004
ALPACA_ES_URL=http://roberto-vm.vpn.sintef:9209
ALPACA_DATASET_ROOT=/Users/abubakarialidu/alpaca/Coverage_exp/Datasets
```

Important LLM settings:

```bash
ALPACA_LLM_API_KEY=your_provider_key
ALPACA_LLM_BASE_URL=https://api.deepinfra.com/v1/openai
ALPACA_LLM_MODEL=openai/gpt-oss-20b
ALPACA_LLM_TEMPERATURE=0
```

For DeepInfra, `DEEPINFRA_TOKEN` is also accepted if `ALPACA_LLM_API_KEY` is not set:

```bash
DEEPINFRA_TOKEN=your_deepinfra_key
ALPACA_LLM_BASE_URL=https://api.deepinfra.com/v1/openai
ALPACA_LLM_MODEL=openai/gpt-oss-20b
```

Timeout and retry controls:

```bash
ALPACA_LLM_TIMEOUT_SECONDS=90
ALPACA_LLM_TIMEOUT_MULTIPLIER=1.8
ALPACA_LLM_TIMEOUT_MAX_SECONDS=300
ALPACA_LLM_MAX_RETRIES=2
ALPACA_LLM_RETRY_BACKOFF_SECONDS=2
ALPACA_LLM_RETRY_MAX_SLEEP_SECONDS=20
```

Cache controls:

```bash
ALPACA_LLM_CACHE_ENABLED=1
ALPACA_LLM_CACHE_DIR=./backend_lab/.cache
```

Evaluation commands default to `--cache-mode off` so paper experiments are not biased by previous runs. Interactive inspection commands default to `--cache-mode env`.

## Provider-Agnostic LLM Configuration

`backend_lab` uses the OpenAI SDK interface, so any provider with an OpenAI-compatible chat completions endpoint can be used by changing the base URL, model, and API key.

DeepInfra example:

```bash
ALPACA_LLM_API_KEY=your_deepinfra_key
ALPACA_LLM_BASE_URL=https://api.deepinfra.com/v1/openai
ALPACA_LLM_MODEL=openai/gpt-oss-20b
```

OpenAI example:

```bash
ALPACA_LLM_API_KEY=your_openai_key
ALPACA_LLM_BASE_URL=https://api.openai.com/v1
ALPACA_LLM_MODEL=gpt-4.1-mini
```

DeepSeek example:

```bash
ALPACA_LLM_API_KEY=your_deepseek_key
ALPACA_LLM_BASE_URL=https://api.deepseek.com/v1
ALPACA_LLM_MODEL=deepseek-chat
```

Claude/Anthropic note:

Anthropic's native API is not the same as the OpenAI chat completions API. To use Claude through this interface, route it through an OpenAI-compatible gateway or proxy, then set:

```bash
ALPACA_LLM_API_KEY=your_gateway_key
ALPACA_LLM_BASE_URL=https://your-openai-compatible-gateway/v1
ALPACA_LLM_MODEL=your-claude-model-name
```

The same pattern works for OpenRouter, LiteLLM, local OpenAI-compatible servers, and other gateways.

## CLI Commands

Show all commands:

```bash
python -m backend_lab --help
```

List datasets:

```bash
python -m backend_lab datasets
```

Preview a table:

```bash
python -m backend_lab table --dataset 2T_2022 --table OHGI1JNY --limit 8
```

Preview CEA targets:

```bash
python -m backend_lab targets --dataset 2T_2022 --task cea --limit 5
```

Inspect the table profile:

```bash
python -m backend_lab table-profile \
  --dataset 2T_2022 \
  --table G04AXW0O \
  --cache-mode env
```

Run LLM preprocessing for one cell:

```bash
python -m backend_lab llm-preprocess \
  --dataset 2T_2022 \
  --table OHGI1JNY \
  --row 1 \
  --col 1 \
  --cache-mode env
```

Run one full cell-level ES retrieval and deterministic reranking:

```bash
python -m backend_lab llm-es-candidates \
  --dataset 2T_2022 \
  --table OHGI1JNY \
  --row 1 \
  --col 1 \
  --top-k 100 \
  --cache-mode off
```

Run one cell with edge-triggered LLM adjudication:

```bash
python -m backend_lab llm-es-candidates \
  --dataset 2T_2022 \
  --table OHGI1JNY \
  --row 8 \
  --col 1 \
  --top-k 100 \
  --cria-deterministic-llm \
  --cache-mode off
```

Run one table evaluation with CRIA-deterministic+LLM:

```bash
python -m backend_lab cea-batch-eval \
  --dataset 2T_2022 \
  --table OHGI1JNY \
  --limit 20 \
  --top-k 100 \
  --cria-deterministic-llm \
  --cache-mode off
```

Run CRIA-LLM-only as an ablation:

```bash
python -m backend_lab cea-batch-eval \
  --dataset 2T_2022 \
  --table OHGI1JNY \
  --limit 20 \
  --top-k 100 \
  --cria-llm \
  --cache-mode off
```

Run a multi-table experiment and export JSON/CSV summaries:

```bash
python -m backend_lab multi-table-eval \
  --dataset 2T_2022 \
  --tables G04AXW0O OHGI1JNY B20WIQKU QNP7O8L5 \
  --limit-per-table 20 \
  --top-k 100 \
  --cria-deterministic-llm \
  --summary-only \
  --cache-mode off \
  --json-out backend_lab/out/multi_table_eval.json \
  --csv-out backend_lab/out/multi_table_eval.csv
```

Run a backend API health check:

```bash
python -m backend_lab api-health
```

Fetch one Elasticsearch document by QID:

```bash
python -m backend_lab qid --qid Q31
```

Run a direct Elasticsearch search:

```bash
python -m backend_lab es-search \
  --query Boston \
  --size 5 \
  --coarse-type LOCATION \
  --fine-type CITY
```

## Cache Mode

LLM-backed commands accept:

```bash
--cache-mode off|on|env
```

Recommended usage:

| Scenario | Cache mode |
| --- | --- |
| Paper experiment or fair benchmark | `--cache-mode off` |
| Local inspection while developing prompts | `--cache-mode env` or `--cache-mode on` |
| Production-style repeated calls | `--cache-mode on` |

Evaluation summaries include cache metadata:

- `cache_enabled`
- `cache_scope`
- `schema_profile_cache_enabled`

## Outputs

Experiment outputs should be written to `backend_lab/out/`:

```bash
backend_lab/out/*.json
backend_lab/out/*.csv
```

That directory is ignored by git because it contains generated results, not source code.

## Evaluation Fields

The evaluation commands report:

- `end_to_end_accuracy`
- `top_candidate_accuracy`
- `answered_accuracy`
- `coverage`
- `deterministic_only_accuracy`
- `hybrid_accuracy`
- `accuracy_after_retrieval_backoff`
- `semantic_invocations`
- `semantic_overrides`
- `weak_mention_count`
- `multi_hypothesis_count`
- `retrieval_backoff_count`
- `retrieval_recovered_by_backoff`
- `schema_profile_confidence`
- `schema_induction_mode`
- cache metadata

The error report separates failure and recovery families so it is easier to tell whether gains came from schema induction, retrieval backoff, deterministic reranking, or LLM adjudication.

## PostgreSQL Inspection

Manual PostgreSQL inspection helpers are stored in:

```bash
backend_lab/queries/postgres_inspection.sql
```

These are intended for read-only `psql` sessions.

## Development Checks

Run syntax checks:

```bash
python -m compileall backend_lab
```

Run the table-aware unit tests:

```bash
python -m unittest tests.test_backend_lab_table_aware
```

## Notes for Contributors

- Do not commit `.env`.
- Do not commit `.venv`.
- Do not commit `backend_lab/out/`.
- Do not commit `backend_lab/.cache/`.
- Keep backend interactions read-only unless a separate backend change is explicitly reviewed.
- Prefer adding new provider support through configuration rather than provider-specific code paths.
