from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from importlib.resources import files
from typing import Any

from .common import resolve_postgres_dsn, tqdm

try:  # pragma: no cover - depends on runtime environment
    import psycopg  # type: ignore
    from psycopg.types.json import Jsonb  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    psycopg = None  # type: ignore
    Jsonb = None  # type: ignore

try:  # pragma: no cover - depends on runtime environment
    from wikidata_ner import WikidataNERClassifier  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    WikidataNERClassifier = None  # type: ignore


CLASSIFIER_DISTRIBUTION = "wikidata-ner-classifier"
REQUIRED_CLASSIFIER_VERSION = "0.5.0"
DEFAULT_SOURCE_TABLE = "entities"
DEFAULT_NER_TABLE = "entity_ner"
DEFAULT_BATCH_SIZE = 20_000
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_WORKER_CLASSIFIER: Any = None


def _positive_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return value


def _non_negative_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if value < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return value


def _quote_identifier(raw: str) -> str:
    value = raw.strip()
    if not _IDENTIFIER_RE.fullmatch(value):
        raise ValueError(
            f"Invalid SQL identifier '{raw}'. Use letters, digits, and underscores only."
        )
    return f'"{value}"'


def _require_dependencies() -> Any:
    if psycopg is None or Jsonb is None:
        raise RuntimeError("psycopg is required; install the repository requirements.")
    if WikidataNERClassifier is None:
        raise RuntimeError(
            f"{CLASSIFIER_DISTRIBUTION}=={REQUIRED_CLASSIFIER_VERSION} is required."
        )
    try:
        installed = version(CLASSIFIER_DISTRIBUTION)
    except PackageNotFoundError as exc:
        raise RuntimeError(
            f"{CLASSIFIER_DISTRIBUTION}=={REQUIRED_CLASSIFIER_VERSION} is required."
        ) from exc
    if installed != REQUIRED_CLASSIFIER_VERSION:
        raise RuntimeError(
            f"Expected {CLASSIFIER_DISTRIBUTION}=={REQUIRED_CLASSIFIER_VERSION}, "
            f"found {installed}."
        )
    return psycopg


def _classifier() -> Any:
    global _WORKER_CLASSIFIER
    if _WORKER_CLASSIFIER is None:
        _WORKER_CLASSIFIER = WikidataNERClassifier()
    return _WORKER_CLASSIFIER


@lru_cache(maxsize=1)
def _taxonomy_version() -> str:
    fine_resource = files("wikidata_ner").joinpath("data/B_full_rule_spec.json")
    refinement_resource = files("wikidata_ner").joinpath("data/subtype_rules.json")
    with fine_resource.open("r", encoding="utf-8") as stream:
        fine_payload = json.load(stream)
    with refinement_resource.open("r", encoding="utf-8") as stream:
        refinement_payload = json.load(stream)
    return (
        f"{fine_payload.get('version', 'unknown')}/"
        f"{refinement_payload.get('version', 'unknown')}"
    )


@lru_cache(maxsize=100_000)
def _predict_without_description(
    type_names: tuple[str, ...],
    ancestor_names: tuple[str, ...],
) -> Any:
    return _classifier().predict(
        qid="CACHE",
        types=list(type_names),
        ancestor_types=list(ancestor_names),
    )


def _clean_strings(raw: Any) -> list[str]:
    if not isinstance(raw, (list, tuple)):
        return []
    return [value.strip() for value in raw if isinstance(value, str) and value.strip()]


def _classify_row(row: tuple[Any, ...]) -> tuple[Any, ...] | None:
    qid = row[0] if len(row) > 0 and isinstance(row[0], str) else ""
    if not qid:
        return None
    label = row[1] if len(row) > 1 and isinstance(row[1], str) else None
    description = row[2] if len(row) > 2 and isinstance(row[2], str) else None
    type_qids = _clean_strings(row[3] if len(row) > 3 else ())
    type_labels = _clean_strings(row[4] if len(row) > 4 else ())
    ancestor_qids = _clean_strings(row[5] if len(row) > 5 else ())
    ancestor_labels = _clean_strings(row[6] if len(row) > 6 else ())

    direct_types = [
        {"id": type_qid, "name": type_labels[index]}
        for index, type_qid in enumerate(type_qids)
        if index < len(type_labels)
    ]
    ancestor_types = [
        {"id": type_qid, "name": ancestor_labels[index]}
        for index, type_qid in enumerate(ancestor_qids)
        if index < len(ancestor_labels)
    ]
    type_names = tuple(value["name"] for value in direct_types)
    ancestor_names = tuple(value["name"] for value in ancestor_types)
    if description:
        prediction = _classifier().predict(
            qid=qid,
            types=direct_types,
            ancestor_types=ancestor_types,
            label=label,
            description=description,
        )
    else:
        prediction = _predict_without_description(type_names, ancestor_names)
    if prediction.abstained or not prediction.coarse_type or not prediction.fine_type:
        return None
    retrieval = prediction.to_retrieval_fields(prefix="ner")
    return (
        qid,
        prediction.coarse_type,
        prediction.fine_type,
        prediction.subtype,
        prediction.specific_type,
        list(prediction.specific_types),
        list(prediction.secondary_types),
        dict(prediction.facets),
        int(prediction.specificity_level),
        list(prediction.retrieval_path),
        retrieval.get("ner_retrieval_key"),
        list(retrieval.get("ner_retrieval_tags") or ()),
        float(prediction.confidence),
        (
            float(prediction.specific_type_confidence)
            if prediction.specific_type_confidence is not None
            else None
        ),
        CLASSIFIER_DISTRIBUTION,
        REQUIRED_CLASSIFIER_VERSION,
        _taxonomy_version(),
    )


def _ensure_schema(conn: Any, *, source_table: str, ner_table: str) -> None:
    source_ident = _quote_identifier(source_table)
    ner_ident = _quote_identifier(ner_table)
    stage = f"ner:{source_table}:{REQUIRED_CLASSIFIER_VERSION}"
    with conn.cursor() as cur:
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {ner_ident} (
                qid TEXT PRIMARY KEY,
                coarse_type TEXT NOT NULL,
                fine_type TEXT NOT NULL,
                subtype TEXT,
                specific_type TEXT,
                specific_types TEXT[] NOT NULL DEFAULT ARRAY[]::text[],
                secondary_types TEXT[] NOT NULL DEFAULT ARRAY[]::text[],
                facets JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                specificity_level SMALLINT NOT NULL DEFAULT 0,
                retrieval_path TEXT[] NOT NULL DEFAULT ARRAY[]::text[],
                retrieval_key TEXT,
                retrieval_tags TEXT[] NOT NULL DEFAULT ARRAY[]::text[],
                confidence REAL NOT NULL,
                specific_type_confidence REAL,
                classifier_name TEXT NOT NULL DEFAULT '{CLASSIFIER_DISTRIBUTION}',
                classifier_version TEXT NOT NULL,
                taxonomy_version TEXT NOT NULL DEFAULT '',
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );

            CREATE TABLE IF NOT EXISTS alpaca_pipeline_state (
                stage TEXT PRIMARY KEY,
                last_qid TEXT NOT NULL DEFAULT '',
                processed BIGINT NOT NULL DEFAULT 0,
                metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );

            ALTER TABLE {ner_ident}
                ADD COLUMN IF NOT EXISTS secondary_types
                    TEXT[] NOT NULL DEFAULT ARRAY[]::text[];
            ALTER TABLE {ner_ident}
                ADD COLUMN IF NOT EXISTS specificity_level
                    SMALLINT NOT NULL DEFAULT 0;
            ALTER TABLE {ner_ident}
                ADD COLUMN IF NOT EXISTS retrieval_path
                    TEXT[] NOT NULL DEFAULT ARRAY[]::text[];
            ALTER TABLE {ner_ident}
                ADD COLUMN IF NOT EXISTS classifier_name
                    TEXT NOT NULL DEFAULT '{CLASSIFIER_DISTRIBUTION}';
            ALTER TABLE {ner_ident}
                ADD COLUMN IF NOT EXISTS taxonomy_version
                    TEXT NOT NULL DEFAULT '';

            COMMENT ON TABLE {ner_ident} IS
                'Successful deterministic item classifications produced by '
                '{CLASSIFIER_DISTRIBUTION}=={REQUIRED_CLASSIFIER_VERSION}. '
                'Rows without a confident classification are represented as '
                'ABSTAINED by entities_with_ner after the classification pass completes.';
            """
        )
        cur.execute("SELECT to_regclass(%s)", (source_table,))
        source_exists = bool((cur.fetchone() or (None,))[0])
        if source_exists and source_table == DEFAULT_SOURCE_TABLE and ner_table == DEFAULT_NER_TABLE:
            cur.execute(
                f"""
                CREATE OR REPLACE VIEW entities_with_ner AS
                SELECT
                    e.qid,
                    e.label,
                    e.labels,
                    e.aliases,
                    e.description,
                    e.types,
                    e.ancestor_types,
                    CASE
                        WHEN e.qid NOT LIKE 'Q%' THEN 'NOT_APPLICABLE'
                        WHEN n.qid IS NOT NULL THEN 'CLASSIFIED'
                        WHEN COALESCE(
                            (state.metadata ->> 'complete')::boolean,
                            FALSE
                        ) THEN 'ABSTAINED'
                        ELSE 'PENDING'
                    END AS ner_status,
                    n.coarse_type AS ner_coarse_type,
                    n.fine_type AS ner_fine_type,
                    n.subtype AS ner_subtype,
                    n.specific_type AS ner_specific_type,
                    n.specific_types AS ner_specific_types,
                    n.secondary_types AS ner_secondary_types,
                    n.facets AS ner_facets,
                    n.specificity_level AS ner_specificity_level,
                    n.retrieval_path AS ner_retrieval_path,
                    n.retrieval_key AS ner_retrieval_key,
                    n.retrieval_tags AS ner_retrieval_tags,
                    n.confidence AS ner_confidence,
                    n.specific_type_confidence AS ner_specific_type_confidence,
                    n.classifier_name AS ner_classifier_name,
                    n.classifier_version AS ner_classifier_version,
                    n.taxonomy_version AS ner_taxonomy_version,
                    e.item_category,
                    e.popularity,
                    e.prior,
                    e.wikipedia_url,
                    e.dbpedia_url,
                    e.updated_at
                FROM {source_ident} AS e
                LEFT JOIN {ner_ident} AS n ON n.qid = e.qid
                LEFT JOIN alpaca_pipeline_state AS state
                    ON state.stage = '{stage}';

                COMMENT ON VIEW entities_with_ner IS
                    'Entities joined to the complete wikidata-ner-classifier '
                    'coarse -> fine -> subtype -> specific hierarchy.';
                """
            )
    conn.commit()


def _load_state(conn: Any, *, stage: str) -> tuple[str, int]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT last_qid, processed FROM alpaca_pipeline_state WHERE stage = %s",
            (stage,),
        )
        row = cur.fetchone()
    if not row:
        return "", 0
    return (
        row[0] if isinstance(row[0], str) else "",
        int(row[1]) if isinstance(row[1], int) else 0,
    )


def _reset_state(conn: Any, *, stage: str, ner_table: str) -> None:
    ner_ident = _quote_identifier(ner_table)
    with conn.cursor() as cur:
        cur.execute(f"TRUNCATE TABLE {ner_ident}")
        cur.execute("DELETE FROM alpaca_pipeline_state WHERE stage = %s", (stage,))
    conn.commit()


def _count_items(conn: Any, *, source_table: str, include_non_items: bool) -> int:
    source_ident = _quote_identifier(source_table)
    where_sql = "" if include_non_items else "WHERE qid LIKE 'Q%'"
    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {source_ident} {where_sql}")
        row = cur.fetchone()
    return int(row[0]) if row and isinstance(row[0], int) else 0


def _fetch_batch(
    conn: Any,
    *,
    source_table: str,
    last_qid: str,
    batch_size: int,
    include_non_items: bool,
) -> list[tuple[Any, ...]]:
    source_ident = _quote_identifier(source_table)
    item_filter = "" if include_non_items else "AND e.qid LIKE 'Q%%'"
    sql = f"""
    SELECT
        e.qid,
        e.label,
        e.description,
        e.types,
        ARRAY(
            SELECT COALESCE(type_entity.label, typed.qid)
            FROM unnest(e.types) WITH ORDINALITY AS typed(qid, ord)
            LEFT JOIN {source_ident} AS type_entity ON type_entity.qid = typed.qid
            ORDER BY typed.ord
        ) AS type_labels,
        e.ancestor_types,
        ARRAY(
            SELECT COALESCE(ancestor_entity.label, ancestor.qid)
            FROM unnest(e.ancestor_types) WITH ORDINALITY AS ancestor(qid, ord)
            LEFT JOIN {source_ident} AS ancestor_entity ON ancestor_entity.qid = ancestor.qid
            ORDER BY ancestor.ord
        ) AS ancestor_labels
    FROM {source_ident} AS e
    WHERE e.qid > %s
      {item_filter}
    ORDER BY e.qid
    LIMIT %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, (last_qid, int(batch_size)))
        return list(cur.fetchall())


def _write_batch(
    conn: Any,
    *,
    ner_table: str,
    stage: str,
    last_qid: str,
    processed: int,
    classified_rows: list[tuple[Any, ...]],
) -> None:
    ner_ident = _quote_identifier(ner_table)
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TEMP TABLE IF NOT EXISTS alpaca_ner_batch (
                qid TEXT,
                coarse_type TEXT,
                fine_type TEXT,
                subtype TEXT,
                specific_type TEXT,
                specific_types TEXT[],
                secondary_types TEXT[],
                facets JSONB,
                specificity_level SMALLINT,
                retrieval_path TEXT[],
                retrieval_key TEXT,
                retrieval_tags TEXT[],
                confidence REAL,
                specific_type_confidence REAL,
                classifier_name TEXT,
                classifier_version TEXT,
                taxonomy_version TEXT
            ) ON COMMIT PRESERVE ROWS
            """
        )
        cur.execute("TRUNCATE TABLE alpaca_ner_batch")
        if classified_rows:
            with cur.copy(
                """
                COPY alpaca_ner_batch (
                    qid, coarse_type, fine_type, subtype, specific_type,
                    specific_types, secondary_types, facets, specificity_level,
                    retrieval_path, retrieval_key, retrieval_tags, confidence,
                    specific_type_confidence, classifier_name,
                    classifier_version, taxonomy_version
                ) FROM STDIN
                """
            ) as copy:
                for row in classified_rows:
                    mutable = list(row)
                    mutable[7] = Jsonb(mutable[7])
                    copy.write_row(tuple(mutable))
            cur.execute(
                f"""
                INSERT INTO {ner_ident} (
                    qid, coarse_type, fine_type, subtype, specific_type,
                    specific_types, secondary_types, facets, specificity_level,
                    retrieval_path, retrieval_key, retrieval_tags, confidence,
                    specific_type_confidence, classifier_name,
                    classifier_version, taxonomy_version
                )
                SELECT
                    qid, coarse_type, fine_type, subtype, specific_type,
                    specific_types, secondary_types, facets, specificity_level,
                    retrieval_path, retrieval_key, retrieval_tags, confidence,
                    specific_type_confidence, classifier_name,
                    classifier_version, taxonomy_version
                FROM alpaca_ner_batch
                ON CONFLICT (qid) DO UPDATE SET
                    coarse_type = EXCLUDED.coarse_type,
                    fine_type = EXCLUDED.fine_type,
                    subtype = EXCLUDED.subtype,
                    specific_type = EXCLUDED.specific_type,
                    specific_types = EXCLUDED.specific_types,
                    secondary_types = EXCLUDED.secondary_types,
                    facets = EXCLUDED.facets,
                    specificity_level = EXCLUDED.specificity_level,
                    retrieval_path = EXCLUDED.retrieval_path,
                    retrieval_key = EXCLUDED.retrieval_key,
                    retrieval_tags = EXCLUDED.retrieval_tags,
                    confidence = EXCLUDED.confidence,
                    specific_type_confidence = EXCLUDED.specific_type_confidence,
                    classifier_name = EXCLUDED.classifier_name,
                    classifier_version = EXCLUDED.classifier_version,
                    taxonomy_version = EXCLUDED.taxonomy_version,
                    updated_at = NOW()
                """
            )
        cur.execute(
            """
            INSERT INTO alpaca_pipeline_state (stage, last_qid, processed, metadata)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (stage) DO UPDATE SET
                last_qid = EXCLUDED.last_qid,
                processed = EXCLUDED.processed,
                metadata = EXCLUDED.metadata,
                updated_at = NOW()
            """,
            (
                stage,
                last_qid,
                int(processed),
                Jsonb(
                    {
                        "classifier": CLASSIFIER_DISTRIBUTION,
                        "classifier_version": REQUIRED_CLASSIFIER_VERSION,
                        "taxonomy_version": _taxonomy_version(),
                        "complete": False,
                    }
                ),
            ),
        )
    conn.commit()


def _mark_stage_complete(
    conn: Any,
    *,
    stage: str,
    last_qid: str,
    processed: int,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO alpaca_pipeline_state (
                stage, last_qid, processed, metadata
            )
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (stage) DO UPDATE SET
                last_qid = EXCLUDED.last_qid,
                processed = EXCLUDED.processed,
                metadata = alpaca_pipeline_state.metadata || EXCLUDED.metadata,
                updated_at = NOW()
            """,
            (stage, last_qid, int(processed), Jsonb({"complete": True})),
        )
    conn.commit()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classify Wikidata items already stored in Postgres using "
            f"{CLASSIFIER_DISTRIBUTION}=={REQUIRED_CLASSIFIER_VERSION}."
        )
    )
    parser.add_argument("--postgres-dsn", help="Postgres DSN.")
    parser.add_argument("--source-table", default=DEFAULT_SOURCE_TABLE)
    parser.add_argument("--ner-table", default=DEFAULT_NER_TABLE)
    parser.add_argument("--batch-size", type=_positive_int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--workers",
        type=_positive_int,
        default=max(1, min(8, os.cpu_count() or 1)),
    )
    parser.add_argument(
        "--chunksize",
        type=_positive_int,
        default=256,
        help="Rows sent to each process-pool work unit (default: 256).",
    )
    parser.add_argument(
        "--limit",
        type=_non_negative_int,
        default=0,
        help="Optional number of rows to process in this invocation (0 = all remaining).",
    )
    parser.add_argument(
        "--include-non-items",
        action="store_true",
        help="Also run the item classifier on non-Q identifiers.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear generated NER results and the resume checkpoint before starting.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Create/migrate the NER table and joined inspection view, then exit.",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    pg = _require_dependencies()
    source_table = args.source_table.strip()
    ner_table = args.ner_table.strip()
    _quote_identifier(source_table)
    _quote_identifier(ner_table)
    stage = f"ner:{source_table}:{REQUIRED_CLASSIFIER_VERSION}"

    with pg.connect(resolve_postgres_dsn(args.postgres_dsn)) as conn:
        _ensure_schema(conn, source_table=source_table, ner_table=ner_table)
        if args.prepare_only:
            print(
                "NER schema ready:",
                f"table={ner_table}",
                f"classifier={CLASSIFIER_DISTRIBUTION}=={REQUIRED_CLASSIFIER_VERSION}",
                f"taxonomy={_taxonomy_version()}",
            )
            return 0
        if args.reset:
            _reset_state(conn, stage=stage, ner_table=ner_table)
        last_qid, processed = _load_state(conn, stage=stage)
        total = _count_items(
            conn,
            source_table=source_table,
            include_non_items=bool(args.include_non_items),
        )
        invocation_start = processed
        invocation_limit = int(args.limit)
        classified_total = 0
        abstained_total = 0
        exhausted = False

        print(
            "Classifying Postgres entities:",
            f"classifier={CLASSIFIER_DISTRIBUTION}=={REQUIRED_CLASSIFIER_VERSION}",
            f"source={source_table}",
            f"destination={ner_table}",
            f"workers={args.workers}",
            f"batch_size={args.batch_size}",
            f"resume_last_qid={last_qid or 'n/a'}",
            f"resume_processed={processed}",
            f"estimated_total={total}",
        )
        with ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
            with tqdm(
                total=total,
                initial=min(processed, total),
                desc="wikidata-ner",
                unit="item",
            ) as progress:
                while True:
                    remaining = (
                        invocation_limit - (processed - invocation_start)
                        if invocation_limit > 0
                        else int(args.batch_size)
                    )
                    if invocation_limit > 0 and remaining <= 0:
                        break
                    fetch_size = min(int(args.batch_size), remaining)
                    rows = _fetch_batch(
                        conn,
                        source_table=source_table,
                        last_qid=last_qid,
                        batch_size=fetch_size,
                        include_non_items=bool(args.include_non_items),
                    )
                    if not rows:
                        exhausted = True
                        break
                    predictions = list(
                        executor.map(
                            _classify_row,
                            rows,
                            chunksize=int(args.chunksize),
                        )
                    )
                    classified_rows = [row for row in predictions if row is not None]
                    batch_processed = len(rows)
                    batch_classified = len(classified_rows)
                    batch_abstained = batch_processed - batch_classified
                    last_qid = str(rows[-1][0])
                    processed += batch_processed
                    classified_total += batch_classified
                    abstained_total += batch_abstained
                    _write_batch(
                        conn,
                        ner_table=ner_table,
                        stage=stage,
                        last_qid=last_qid,
                        processed=processed,
                        classified_rows=classified_rows,
                    )
                    progress.update(batch_processed)
                    progress.set_postfix(
                        classified=classified_total,
                        abstained=abstained_total,
                        last_qid=last_qid,
                    )
        if exhausted:
            _mark_stage_complete(
                conn,
                stage=stage,
                last_qid=last_qid,
                processed=processed,
            )

    print(
        "Completed Wikidata NER classification:",
        f"processed_this_run={processed - invocation_start}",
        f"processed_checkpoint={processed}",
        f"classified_this_run={classified_total}",
        f"abstained_this_run={abstained_total}",
        f"last_qid={last_qid or 'n/a'}",
    )
    return 0


def main() -> int:
    try:
        return run(parse_args())
    except (RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
