from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .backend_client import BackendHttpClient, BackendRequestError
from .config import BackendConfig
from .dataset import (
    CellContext,
    build_cell_context,
    iter_cea_targets,
    iter_cta_targets,
    list_dataset_ids,
    load_table,
)
from .evaluation import load_cea_ground_truth
from .es_experiment import build_es_query_from_lookup_payload, rerank_es_hits, score_reranked_decision
from .llm_client import LlmConfigurationError, LlmRequestError, OpenAICompatibleLlmClient
from .preprocess import build_seed_schema, lookup_payload_from_preprocessing, lookup_payload_variants_from_preprocessing
from .semantic import (
    build_cria_llm_payload,
    build_shepherd_payload,
    build_semantic_payload,
    merge_cria_llm_decision,
    merge_shepherd_decision,
    merge_semantic_decision,
    should_run_cria_shepherd,
    should_run_semantic_fallback,
)
from .table_profile import build_table_profile_seed


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def _config_with_cache_mode(config: BackendConfig, cache_mode: str | None) -> BackendConfig:
    if not cache_mode:
        return config
    return config.with_cache_mode(cache_mode)


def _build_table_profile(
    *,
    dataset_root: Path,
    dataset_id: str,
    table_id: str,
    llm_client: OpenAICompatibleLlmClient | None,
) -> tuple[dict[str, object], dict[str, object] | None]:
    seed_profile = build_table_profile_seed(dataset_root, dataset_id, table_id).to_dict()
    if llm_client is None:
        return seed_profile, None
    llm_result = llm_client.induce_table_profile(seed_profile)
    profile = llm_result.get("table_profile", seed_profile)
    if not isinstance(profile, dict):
        profile = seed_profile
    return profile, llm_result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only analysis CLI for Coverage_exp against the live Alpaca backend."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("datasets", help="List dataset IDs under Coverage_exp/Datasets.")

    table_parser = subparsers.add_parser("table", help="Show a benchmark table.")
    table_parser.add_argument("--dataset", required=True)
    table_parser.add_argument("--table", required=True)
    table_parser.add_argument("--limit", type=int, default=8)

    target_parser = subparsers.add_parser("targets", help="Show CEA or CTA target rows.")
    target_parser.add_argument("--dataset", required=True)
    target_parser.add_argument("--task", choices=("cea", "cta"), required=True)
    target_parser.add_argument("--limit", type=int, default=10)

    cell_parser = subparsers.add_parser(
        "cell-query",
        help="Build a conservative lookup payload for one benchmark cell.",
    )
    cell_parser.add_argument("--dataset", required=True)
    cell_parser.add_argument("--table", required=True)
    cell_parser.add_argument("--row", type=int, required=True)
    cell_parser.add_argument("--col", type=int, required=True)
    cell_parser.add_argument("--top-k", type=int, default=100)

    preprocess_parser = subparsers.add_parser(
        "llm-preprocess",
        help="Run the provider-agnostic OpenAI-compatible LLM preprocessing step for one cell.",
    )
    preprocess_parser.add_argument("--dataset", required=True)
    preprocess_parser.add_argument("--table", required=True)
    preprocess_parser.add_argument("--row", type=int, required=True)
    preprocess_parser.add_argument("--col", type=int, required=True)
    preprocess_parser.add_argument("--top-k", type=int, default=100)
    preprocess_parser.add_argument("--model", default="")
    preprocess_parser.add_argument("--cache-mode", choices=("off", "on", "env"), default="env")

    llm_lookup_parser = subparsers.add_parser(
        "llm-lookup",
        help="Run LLM preprocessing for one cell and then call the live Alpaca lookup API.",
    )
    llm_lookup_parser.add_argument("--dataset", required=True)
    llm_lookup_parser.add_argument("--table", required=True)
    llm_lookup_parser.add_argument("--row", type=int, required=True)
    llm_lookup_parser.add_argument("--col", type=int, required=True)
    llm_lookup_parser.add_argument("--top-k", type=int, default=100)
    llm_lookup_parser.add_argument("--model", default="")
    llm_lookup_parser.add_argument("--debug", action="store_true")
    llm_lookup_parser.add_argument("--cache-mode", choices=("off", "on", "env"), default="env")

    llm_es_parser = subparsers.add_parser(
        "llm-es-candidates",
        help="Run LLM preprocessing, retrieve ES candidates, and emit deterministic reranking features.",
    )
    llm_es_parser.add_argument("--dataset", required=True)
    llm_es_parser.add_argument("--table", required=True)
    llm_es_parser.add_argument("--row", type=int, required=True)
    llm_es_parser.add_argument("--col", type=int, required=True)
    llm_es_parser.add_argument("--top-k", type=int, default=100)
    llm_es_parser.add_argument("--model", default="")
    llm_es_parser.add_argument("--semantic-fallback", action="store_true")
    llm_es_parser.add_argument("--cria-deterministic-llm", dest="cria_shepherd", action="store_true")
    llm_es_parser.add_argument("--cria-shepherd", dest="cria_shepherd", action="store_true", help=argparse.SUPPRESS)
    llm_es_parser.add_argument("--cria-llm", action="store_true")
    llm_es_parser.add_argument("--cria-llm-candidates", type=int, default=20)
    llm_es_parser.add_argument("--cache-mode", choices=("off", "on", "env"), default="off")

    batch_eval_parser = subparsers.add_parser(
        "cea-batch-eval",
        help="Run a small-batch CEA evaluation with LLM preprocessing, ES retrieval, and deterministic post-processing.",
    )
    batch_eval_parser.add_argument("--dataset", required=True)
    batch_eval_parser.add_argument("--table", default="")
    batch_eval_parser.add_argument("--limit", type=int, default=10)
    batch_eval_parser.add_argument("--offset", type=int, default=0)
    batch_eval_parser.add_argument("--top-k", type=int, default=100)
    batch_eval_parser.add_argument("--model", default="")
    batch_eval_parser.add_argument("--semantic-fallback", action="store_true")
    batch_eval_parser.add_argument("--cria-deterministic-llm", dest="cria_shepherd", action="store_true")
    batch_eval_parser.add_argument("--cria-shepherd", dest="cria_shepherd", action="store_true", help=argparse.SUPPRESS)
    batch_eval_parser.add_argument("--cria-llm", action="store_true")
    batch_eval_parser.add_argument("--cria-llm-candidates", type=int, default=20)
    batch_eval_parser.add_argument("--cache-mode", choices=("off", "on", "env"), default="off")

    multi_table_parser = subparsers.add_parser(
        "multi-table-eval",
        help="Run CEA evaluation across multiple tables and return an aggregate calibration summary.",
    )
    multi_table_parser.add_argument("--dataset", required=True)
    multi_table_parser.add_argument("--tables", nargs="+", required=True)
    multi_table_parser.add_argument("--limit-per-table", type=int, default=10)
    multi_table_parser.add_argument("--top-k", type=int, default=100)
    multi_table_parser.add_argument("--model", default="")
    multi_table_parser.add_argument("--semantic-fallback", action="store_true")
    multi_table_parser.add_argument("--cria-deterministic-llm", dest="cria_shepherd", action="store_true")
    multi_table_parser.add_argument("--cria-shepherd", dest="cria_shepherd", action="store_true", help=argparse.SUPPRESS)
    multi_table_parser.add_argument("--cria-llm", action="store_true")
    multi_table_parser.add_argument("--cria-llm-candidates", type=int, default=20)
    multi_table_parser.add_argument("--hide-results", action="store_true")
    multi_table_parser.add_argument("--summary-only", action="store_true")
    multi_table_parser.add_argument("--json-out", default="")
    multi_table_parser.add_argument("--csv-out", default="")
    multi_table_parser.add_argument("--cache-mode", choices=("off", "on", "env"), default="off")

    table_profile_parser = subparsers.add_parser(
        "table-profile",
        help="Build a deterministic-plus-LLM table profile for one benchmark table.",
    )
    table_profile_parser.add_argument("--dataset", required=True)
    table_profile_parser.add_argument("--table", required=True)
    table_profile_parser.add_argument("--model", default="")
    table_profile_parser.add_argument("--cache-mode", choices=("off", "on", "env"), default="env")

    lookup_parser = subparsers.add_parser("lookup", help="Run a live Alpaca lookup for one cell.")
    lookup_parser.add_argument("--dataset", required=True)
    lookup_parser.add_argument("--table", required=True)
    lookup_parser.add_argument("--row", type=int, required=True)
    lookup_parser.add_argument("--col", type=int, required=True)
    lookup_parser.add_argument("--top-k", type=int, default=100)
    lookup_parser.add_argument("--debug", action="store_true")

    es_parser = subparsers.add_parser("es-search", help="Run a direct Elasticsearch search.")
    es_parser.add_argument("--query", required=True)
    es_parser.add_argument("--size", type=int, default=5)
    es_parser.add_argument("--coarse-type", default="")
    es_parser.add_argument("--fine-type", default="")

    qid_parser = subparsers.add_parser("qid", help="Fetch one Elasticsearch document by QID.")
    qid_parser.add_argument("--qid", required=True)

    subparsers.add_parser("api-health", help="Call the live Alpaca health endpoint.")
    subparsers.add_parser("es-indices", help="List Elasticsearch indices.")

    return parser


def _context_and_payload(args: argparse.Namespace, config: BackendConfig) -> tuple[dict[str, object], dict[str, object]]:
    context = build_cell_context(
        config.dataset_root,
        args.dataset,
        args.table,
        row_id=int(args.row),
        col_id=int(args.col),
    )
    table_profile, _ = _build_table_profile(
        dataset_root=config.dataset_root,
        dataset_id=args.dataset,
        table_id=args.table,
        llm_client=None,
    )
    seed_schema = build_seed_schema(context, table_profile=table_profile).to_dict()
    payload = lookup_payload_from_preprocessing(seed_schema, top_k=int(args.top_k))
    return context.to_dict(), payload


def _build_context_seed_and_llm_result(
    args: argparse.Namespace,
    config: BackendConfig,
) -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, object]]:
    config = _config_with_cache_mode(config, getattr(args, "cache_mode", None))
    context = build_cell_context(
        config.dataset_root,
        args.dataset,
        args.table,
        row_id=int(args.row),
        col_id=int(args.col),
    )
    llm_client = OpenAICompatibleLlmClient.from_config(
        config,
        model=(args.model or config.llm_model),
    )
    table_profile, table_profile_llm_result = _build_table_profile(
        dataset_root=config.dataset_root,
        dataset_id=args.dataset,
        table_id=args.table,
        llm_client=llm_client,
    )
    seed_schema = build_seed_schema(context, table_profile=table_profile).to_dict()
    if table_profile_llm_result:
        seed_schema["metadata"]["table_profile_usage"] = table_profile_llm_result.get("usage")
    llm_result = llm_client.preprocess(seed_schema)
    payload = lookup_payload_from_preprocessing(
        llm_result["preprocessing_schema"],
        top_k=int(args.top_k),
    )
    return context.to_dict(), seed_schema, llm_result, payload


def _run_es_rerank(
    *,
    client: BackendHttpClient,
    payload: dict[str, object],
    preprocessing_schema: dict[str, object],
) -> tuple[dict[str, object], dict[str, object], list[dict[str, object]]]:
    es_query = build_es_query_from_lookup_payload(
        payload,
        preprocessing_schema=preprocessing_schema,
    )
    es_result = client.es_custom_search(payload=es_query)
    reranked = rerank_es_hits(
        es_result=es_result,
        lookup_payload=payload,
        preprocessing_schema=preprocessing_schema,
    )
    return es_query, es_result, reranked


def _merge_es_hits(results: list[tuple[str, dict[str, object]]]) -> dict[str, object]:
    merged_hits: list[dict[str, object]] = []
    by_qid: dict[str, dict[str, object]] = {}
    for stage, es_result in results:
        hits = es_result.get("hits", {}).get("hits", []) if isinstance(es_result, dict) else []
        if not isinstance(hits, list):
            continue
        for hit in hits:
            if not isinstance(hit, dict):
                continue
            source = hit.get("_source", {})
            if not isinstance(source, dict):
                continue
            qid = source.get("qid")
            if not isinstance(qid, str) or not qid:
                continue
            current = by_qid.get(qid)
            stage_list = [stage]
            score = float(hit.get("_score", 0.0) or 0.0)
            if current is None:
                copied = dict(hit)
                copied["_retrieved_by"] = stage_list
                by_qid[qid] = copied
                continue
            existing_score = float(current.get("_score", 0.0) or 0.0)
            existing_stages = current.get("_retrieved_by", [])
            if isinstance(existing_stages, list) and stage not in existing_stages:
                existing_stages.append(stage)
                current["_retrieved_by"] = existing_stages
            if score > existing_score:
                copied = dict(hit)
                copied["_retrieved_by"] = current.get("_retrieved_by", stage_list)
                by_qid[qid] = copied
    merged_hits = list(by_qid.values())
    merged_hits.sort(key=lambda item: float(item.get("_score", 0.0) or 0.0), reverse=True)
    return {
        "hits": {
            "hits": merged_hits,
            "total": {"value": len(merged_hits), "relation": "eq"},
        }
    }


def _run_es_rerank_with_backoff(
    *,
    client: BackendHttpClient,
    preprocessing_schema: dict[str, object],
    top_k: int,
) -> tuple[dict[str, object], dict[str, object], list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    payload_variants = lookup_payload_variants_from_preprocessing(preprocessing_schema, top_k=int(top_k))
    stage_runs: list[dict[str, object]] = []
    collected_results: list[tuple[str, dict[str, object]]] = []
    primary_payload: dict[str, object] | None = None
    primary_query: dict[str, object] | None = None
    primary_result: dict[str, object] | None = None

    for index, stage in enumerate(payload_variants):
        payload = stage.get("payload", {})
        if not isinstance(payload, dict):
            continue
        es_query = build_es_query_from_lookup_payload(payload, preprocessing_schema=preprocessing_schema)
        es_result = client.es_custom_search(payload=es_query)
        hits = es_result.get("hits", {}).get("hits", []) if isinstance(es_result, dict) else []
        hit_count = len(hits) if isinstance(hits, list) else 0
        stage_runs.append(
            {
                "stage": stage.get("stage"),
                "hypothesis_rank": stage.get("hypothesis_rank"),
                "relaxed": bool(stage.get("relaxed")),
                "payload": payload,
                "hit_count": hit_count,
            }
        )
        if index == 0:
            primary_payload = payload
            primary_query = es_query
            primary_result = es_result
        collected_results.append((str(stage.get("stage", f"stage_{index+1}")), es_result))
        if hit_count >= max(6, int(top_k)):
            break

    merged_result = _merge_es_hits(collected_results)
    reranked = rerank_es_hits(
        es_result=merged_result,
        lookup_payload=primary_payload or lookup_payload_from_preprocessing(preprocessing_schema, top_k=int(top_k)),
        preprocessing_schema=preprocessing_schema,
    )
    reranked = reranked[: max(1, min(100, int(top_k)))]
    backoff_summary = {
        "stages_attempted": len(stage_runs),
        "used_backoff": len(stage_runs) > 1,
        "recovered_candidates": bool(reranked) and len(stage_runs) > 1,
        "stage_runs": stage_runs,
    }
    return (
        primary_query or {},
        primary_result or {"hits": {"hits": []}},
        reranked,
        stage_runs,
        backoff_summary,
    )


def _should_expand_semantic_recall(trigger: dict[str, object], reranked_candidates: list[dict[str, object]]) -> bool:
    if not reranked_candidates:
        return False
    reason_codes = trigger.get("reason_codes", [])
    if not isinstance(reason_codes, (list, tuple)):
        return False
    reason_set = {str(code) for code in reason_codes}
    if "same_label_cluster" not in reason_set:
        return False
    top_label = reranked_candidates[0].get("label")
    if not isinstance(top_label, str) or not top_label.strip():
        return False
    same_label_count = sum(1 for item in reranked_candidates[:10] if item.get("label") == top_label)
    return same_label_count >= 3


def _expand_payload_for_semantic_recall(payload: dict[str, object], base_top_k: int) -> dict[str, object]:
    expanded_payload = dict(payload)
    expanded_payload["top_k"] = max(int(base_top_k), min(100, int(base_top_k) * 3))
    return expanded_payload


def _run_hybrid_resolution(
    *,
    dataset_root: Path,
    context: CellContext,
    top_k: int,
    client: BackendHttpClient,
    llm_client: OpenAICompatibleLlmClient,
    semantic_fallback: bool,
    cria_shepherd: bool = False,
    cria_llm: bool = False,
    cria_llm_candidates: int = 20,
    precomputed_table_profile: dict[str, object] | None = None,
    precomputed_table_profile_llm_result: dict[str, object] | None = None,
) -> dict[str, object]:
    if precomputed_table_profile is None:
        table_profile, table_profile_llm_result = _build_table_profile(
            dataset_root=dataset_root,
            dataset_id=context.dataset_id,
            table_id=context.table_id,
            llm_client=llm_client,
        )
    else:
        table_profile = precomputed_table_profile
        table_profile_llm_result = precomputed_table_profile_llm_result
    seed_schema = build_seed_schema(context, table_profile=table_profile).to_dict()
    llm_result = llm_client.preprocess(seed_schema)
    preprocessing_schema = llm_result["preprocessing_schema"]
    payload = lookup_payload_from_preprocessing(preprocessing_schema, top_k=int(top_k))
    es_query, es_result, reranked, retrieval_stage_runs, retrieval_backoff = _run_es_rerank_with_backoff(
        client=client,
        preprocessing_schema=preprocessing_schema,
        top_k=int(top_k),
    )
    deterministic_decision = score_reranked_decision(reranked)
    semantic_trigger = should_run_semantic_fallback(
        decision=deterministic_decision,
        reranked_candidates=reranked,
    )
    shepherd_trigger = should_run_cria_shepherd(
        decision=deterministic_decision,
        reranked_candidates=reranked,
    )
    semantic_payload: dict[str, object] | None = None
    semantic_result: dict[str, object] | None = None
    semantic_es_query: dict[str, object] | None = None
    semantic_es_result: dict[str, object] | None = None
    semantic_reranked = reranked
    cria_llm_payload: dict[str, object] | None = None
    cria_llm_result: dict[str, object] | None = None
    shepherd_payload: dict[str, object] | None = None
    shepherd_result: dict[str, object] | None = None
    shepherd_reranked = reranked
    final_decision = dict(deterministic_decision)
    final_decision["resolution_mode"] = "deterministic_only"
    if cria_llm:
        cria_llm_payload = build_cria_llm_payload(
            context=context,
            preprocessing_schema=preprocessing_schema,
            lookup_payload=payload,
            reranked_candidates=reranked,
            deterministic_decision=deterministic_decision,
            max_candidates=max(1, int(cria_llm_candidates)),
        )
        cria_llm_result = llm_client.cria_llm_rank(cria_llm_payload)
        final_decision = merge_cria_llm_decision(
            deterministic_decision=deterministic_decision,
            cria_llm_result=cria_llm_result,
            reranked_candidates=reranked,
        )
    elif cria_shepherd and shepherd_trigger.should_run:
        shepherd_payload = build_shepherd_payload(
            context=context,
            preprocessing_schema=preprocessing_schema,
            reranked_candidates=shepherd_reranked,
            decision=deterministic_decision,
            trigger=shepherd_trigger,
        )
        shepherd_result = llm_client.cria_shepherd_resolve(shepherd_payload)
        final_decision = merge_shepherd_decision(
            deterministic_decision=deterministic_decision,
            shepherd_result=shepherd_result,
            reranked_candidates=shepherd_reranked,
            trigger=shepherd_trigger,
        )
    elif semantic_fallback and semantic_trigger.should_run:
        semantic_payload_source = reranked
        if _should_expand_semantic_recall(semantic_trigger.to_dict(), reranked):
            expanded_payload = _expand_payload_for_semantic_recall(payload, int(top_k))
            semantic_es_query, semantic_es_result, semantic_reranked = _run_es_rerank(
                client=client,
                payload=expanded_payload,
                preprocessing_schema=preprocessing_schema,
            )
            semantic_payload_source = semantic_reranked
        semantic_payload = build_semantic_payload(
            context=context,
            preprocessing_schema=preprocessing_schema,
            reranked_candidates=semantic_payload_source,
            decision=deterministic_decision,
        )
        semantic_result = llm_client.semantic_resolve(semantic_payload)
        final_decision = merge_semantic_decision(
            deterministic_decision=deterministic_decision,
            semantic_result=semantic_result,
            reranked_candidates=semantic_payload_source,
        )
    return {
        "context": context.to_dict(),
        "preprocessing_schema_seed": seed_schema,
        "table_profile": table_profile,
        "table_profile_llm_result": table_profile_llm_result,
        "llm_result": llm_result,
        "lookup_payload": payload,
        "es_query": es_query,
        "raw_es_result": es_result,
        "reranked_candidates": reranked,
        "retrieval_stage_runs": retrieval_stage_runs,
        "retrieval_backoff": retrieval_backoff,
        "deterministic_decision": deterministic_decision,
        "semantic_trigger": semantic_trigger.to_dict(),
        "shepherd_trigger": shepherd_trigger.to_dict(),
        "semantic_es_query": semantic_es_query,
        "semantic_es_result": semantic_es_result,
        "semantic_reranked_candidates": semantic_reranked if semantic_reranked is not reranked else None,
        "semantic_payload": semantic_payload,
        "semantic_result": semantic_result,
        "cria_llm_payload": cria_llm_payload,
        "cria_llm_result": cria_llm_result,
        "shepherd_reranked_candidates": shepherd_reranked if shepherd_reranked is not reranked else None,
        "shepherd_payload": shepherd_payload,
        "shepherd_result": shepherd_result,
        "decision": final_decision,
    }


def _gold_rank_diagnostics(
    *,
    reranked_candidates: list[dict[str, object]],
    gold_qids: set[str],
    selected_qid: str,
) -> dict[str, object]:
    gold_candidates: list[dict[str, object]] = []
    selected_candidate: dict[str, object] | None = None
    for item in reranked_candidates:
        qid = item.get("qid")
        if isinstance(qid, str) and qid == selected_qid:
            selected_candidate = item
        if isinstance(qid, str) and qid in gold_qids:
            gold_candidates.append(item)

    def rank_value(item: dict[str, object], key: str) -> int | None:
        raw_value = item.get(key)
        return int(raw_value) if isinstance(raw_value, int) else None

    gold_after_ranks = [rank for item in gold_candidates if (rank := rank_value(item, "reranked_rank")) is not None]
    gold_raw_ranks = [rank for item in gold_candidates if (rank := rank_value(item, "raw_rank")) is not None]
    best_gold = min(gold_candidates, key=lambda item: int(item.get("reranked_rank", 10**9))) if gold_candidates else None
    best_gold_features = best_gold.get("features", {}) if isinstance(best_gold, dict) else {}
    selected_features = selected_candidate.get("features", {}) if isinstance(selected_candidate, dict) else {}
    best_gold_retrieved_by = best_gold.get("retrieved_by", []) if isinstance(best_gold, dict) else []

    return {
        "gold_in_candidates": bool(gold_candidates),
        "candidate_count": len(reranked_candidates),
        "gold_rank_after_rerank": min(gold_after_ranks) if gold_after_ranks else None,
        "gold_rank_raw": min(gold_raw_ranks) if gold_raw_ranks else None,
        "gold_recall_at_1": bool(gold_after_ranks and min(gold_after_ranks) <= 1),
        "gold_recall_at_10": bool(gold_after_ranks and min(gold_after_ranks) <= 10),
        "gold_recall_at_25": bool(gold_after_ranks and min(gold_after_ranks) <= 25),
        "gold_recall_at_50": bool(gold_after_ranks and min(gold_after_ranks) <= 50),
        "gold_recall_at_100": bool(gold_after_ranks and min(gold_after_ranks) <= 100),
        "gold_retrieved_by": best_gold_retrieved_by if isinstance(best_gold_retrieved_by, list) else [],
        "gold_candidate_qid": best_gold.get("qid") if isinstance(best_gold, dict) else None,
        "gold_candidate_label": best_gold.get("label") if isinstance(best_gold, dict) else None,
        "gold_candidate_score": best_gold_features.get("final_score") if isinstance(best_gold_features, dict) else None,
        "selected_candidate_rank_after_rerank": selected_candidate.get("reranked_rank") if isinstance(selected_candidate, dict) else None,
        "selected_candidate_score": selected_features.get("final_score") if isinstance(selected_features, dict) else None,
        "gold_candidate_features": best_gold_features if isinstance(best_gold_features, dict) else {},
        "selected_candidate_features": selected_features if isinstance(selected_features, dict) else {},
    }


def _evaluate_cea_rows(
    *,
    dataset_root: object,
    dataset_id: str,
    rows: list[object],
    top_k: int,
    client: BackendHttpClient,
    llm_client: OpenAICompatibleLlmClient,
    semantic_fallback: bool,
    cria_shepherd: bool = False,
    cria_llm: bool = False,
    cria_llm_candidates: int = 20,
) -> dict[str, object]:
    results: list[dict[str, object]] = []
    answered = 0
    abstained = 0
    correct = 0
    candidate_correct = 0
    semantic_invocations = 0
    semantic_overrides = 0
    shepherd_invocations = 0
    shepherd_overrides = 0
    cria_llm_invocations = 0
    cria_llm_overrides = 0
    schema_confidences: list[float] = []
    weak_mention_count = 0
    multi_hypothesis_count = 0
    retrieval_backoff_count = 0
    deterministic_correct = 0
    recovered_by_backoff = 0
    gold_in_candidates_count = 0
    gold_recall_at_1 = 0
    gold_recall_at_10 = 0
    gold_recall_at_25 = 0
    gold_recall_at_50 = 0
    gold_recall_at_100 = 0
    gold_rank_after_values: list[int] = []
    gold_rank_raw_values: list[int] = []
    table_profile_cache: dict[str, tuple[dict[str, object], dict[str, object] | None]] = {}

    for row in rows:
        context = build_cell_context(
            dataset_root,
            dataset_id,
            row.table_id,
            row_id=row.row_id,
            col_id=row.col_id,
        )
        cached_profile = table_profile_cache.get(row.table_id)
        if cached_profile is None:
            cached_profile = _build_table_profile(
                dataset_root=Path(dataset_root),
                dataset_id=dataset_id,
                table_id=row.table_id,
                llm_client=llm_client,
            )
            table_profile_cache[row.table_id] = cached_profile
        resolution = _run_hybrid_resolution(
            dataset_root=Path(dataset_root),
            context=context,
            top_k=int(top_k),
            client=client,
            llm_client=llm_client,
            semantic_fallback=bool(semantic_fallback),
            cria_shepherd=bool(cria_shepherd),
            cria_llm=bool(cria_llm),
            cria_llm_candidates=int(cria_llm_candidates),
            precomputed_table_profile=cached_profile[0],
            precomputed_table_profile_llm_result=cached_profile[1],
        )
        decision = resolution["decision"] if isinstance(resolution["decision"], dict) else {}
        deterministic_decision = (
            resolution["deterministic_decision"]
            if isinstance(resolution["deterministic_decision"], dict)
            else {}
        )
        semantic_trigger = resolution["semantic_trigger"] if isinstance(resolution["semantic_trigger"], dict) else {}
        semantic_result = resolution["semantic_result"] if isinstance(resolution["semantic_result"], dict) else None
        cria_llm_result = resolution["cria_llm_result"] if isinstance(resolution.get("cria_llm_result"), dict) else None
        shepherd_trigger = resolution["shepherd_trigger"] if isinstance(resolution.get("shepherd_trigger"), dict) else {}
        shepherd_result = resolution["shepherd_result"] if isinstance(resolution.get("shepherd_result"), dict) else None
        preprocessing_schema = resolution["llm_result"].get("preprocessing_schema", {}) if isinstance(resolution.get("llm_result"), dict) else {}
        table_profile = preprocessing_schema.get("table_profile", {}) if isinstance(preprocessing_schema, dict) else {}
        cell_hypothesis = preprocessing_schema.get("cell_hypothesis", {}) if isinstance(preprocessing_schema, dict) else {}
        retrieval_backoff = resolution.get("retrieval_backoff", {}) if isinstance(resolution.get("retrieval_backoff"), dict) else {}

        selected_qid = decision.get("selected_qid")
        selected_qid_str = selected_qid if isinstance(selected_qid, str) else ""
        gold_qids = set(row.gold_qids)
        reranked_candidates = (
            resolution.get("reranked_candidates", [])
            if isinstance(resolution.get("reranked_candidates"), list)
            else []
        )
        gold_diagnostics = _gold_rank_diagnostics(
            reranked_candidates=reranked_candidates,
            gold_qids=gold_qids,
            selected_qid=selected_qid_str,
        )
        if bool(gold_diagnostics.get("gold_in_candidates")):
            gold_in_candidates_count += 1
        if bool(gold_diagnostics.get("gold_recall_at_1")):
            gold_recall_at_1 += 1
        if bool(gold_diagnostics.get("gold_recall_at_10")):
            gold_recall_at_10 += 1
        if bool(gold_diagnostics.get("gold_recall_at_25")):
            gold_recall_at_25 += 1
        if bool(gold_diagnostics.get("gold_recall_at_50")):
            gold_recall_at_50 += 1
        if bool(gold_diagnostics.get("gold_recall_at_100")):
            gold_recall_at_100 += 1
        if isinstance(gold_diagnostics.get("gold_rank_after_rerank"), int):
            gold_rank_after_values.append(int(gold_diagnostics["gold_rank_after_rerank"]))
        if isinstance(gold_diagnostics.get("gold_rank_raw"), int):
            gold_rank_raw_values.append(int(gold_diagnostics["gold_rank_raw"]))
        decision_correct = bool(selected_qid_str in gold_qids and not bool(decision.get("abstain")))
        candidate_is_gold = bool(gold_diagnostics.get("gold_recall_at_1"))
        deterministic_qid = deterministic_decision.get("selected_qid")
        deterministic_correct_flag = isinstance(deterministic_qid, str) and deterministic_qid in gold_qids and not bool(deterministic_decision.get("abstain"))
        if deterministic_correct_flag:
            deterministic_correct += 1

        if bool(semantic_fallback) and bool(semantic_trigger.get("should_run")):
            semantic_invocations += 1
        if isinstance(decision.get("resolution_mode"), str) and str(decision.get("resolution_mode")).startswith("semantic_override"):
            semantic_overrides += 1
        if bool(cria_shepherd) and bool(shepherd_trigger.get("should_run")):
            shepherd_invocations += 1
        if isinstance(decision.get("resolution_mode"), str) and str(decision.get("resolution_mode")).startswith("shepherd_override"):
            shepherd_overrides += 1
        if bool(cria_llm):
            cria_llm_invocations += 1
        if isinstance(decision.get("resolution_mode"), str) and str(decision.get("resolution_mode")) == "cria_llm_override":
            cria_llm_overrides += 1

        if bool(decision.get("abstain")):
            abstained += 1
        else:
            answered += 1
        if decision_correct:
            correct += 1
        if candidate_is_gold:
            candidate_correct += 1
        if isinstance(table_profile, dict):
            try:
                schema_confidences.append(float(table_profile.get("confidence", 0.0) or 0.0))
            except (TypeError, ValueError):
                pass
        if isinstance(cell_hypothesis, dict):
            if str(cell_hypothesis.get("mention_strength", "unknown")) == "weak":
                weak_mention_count += 1
            entity_hypotheses = cell_hypothesis.get("entity_hypotheses", [])
            if isinstance(entity_hypotheses, list) and len(entity_hypotheses) > 1:
                multi_hypothesis_count += 1
        if bool(retrieval_backoff.get("used_backoff")):
            retrieval_backoff_count += 1
            if decision_correct:
                recovered_by_backoff += 1

        results.append(
            {
                "table_id": row.table_id,
                "row_id": row.row_id,
                "col_id": row.col_id,
                "mention": context.mention,
                "gold_qids": list(row.gold_qids),
                "selected_qid": selected_qid_str or None,
                "selected_label": decision.get("selected_label"),
                "correct": decision_correct,
                "top_reranked_is_gold": candidate_is_gold,
                "candidate_is_gold": candidate_is_gold,
                "gold_diagnostics": gold_diagnostics,
                "gold_in_candidates": gold_diagnostics.get("gold_in_candidates"),
                "gold_rank_after_rerank": gold_diagnostics.get("gold_rank_after_rerank"),
                "gold_rank_raw": gold_diagnostics.get("gold_rank_raw"),
                "gold_recall_at_10": gold_diagnostics.get("gold_recall_at_10"),
                "gold_recall_at_100": gold_diagnostics.get("gold_recall_at_100"),
                "abstain": bool(decision.get("abstain")),
                "confidence": decision.get("confidence"),
                "deterministic_confidence": deterministic_decision.get("confidence"),
                "resolution_mode": decision.get("resolution_mode"),
                "semantic_trigger": semantic_trigger,
                "semantic_result": semantic_result,
                "cria_llm_result": cria_llm_result,
                "shepherd_trigger": shepherd_trigger,
                "shepherd_result": shepherd_result,
                "reason_codes": decision.get("reason_codes"),
                "schema_profile_confidence": table_profile.get("confidence") if isinstance(table_profile, dict) else None,
                "mention_strength": cell_hypothesis.get("mention_strength") if isinstance(cell_hypothesis, dict) else None,
                "entity_hypothesis_count": len(cell_hypothesis.get("entity_hypotheses", []))
                if isinstance(cell_hypothesis, dict) and isinstance(cell_hypothesis.get("entity_hypotheses", []), list)
                else 0,
                "retrieval_backoff": retrieval_backoff,
                "retrieval_recovered_by_backoff": bool(retrieval_backoff.get("used_backoff")) and decision_correct,
                "hard_filters": preprocessing_schema.get("retrieval_plan", {}).get("hard_filters", {})
                if isinstance(preprocessing_schema, dict)
                else {},
            }
        )

    total = len(rows)
    return {
        "summary": {
            "evaluated": total,
            "answered": answered,
            "abstained": abstained,
            "end_to_end_accuracy": round((correct / total), 4) if total else 0.0,
            "top_candidate_accuracy": round((candidate_correct / total), 4) if total else 0.0,
            "answered_accuracy": round((correct / answered), 4) if answered else 0.0,
            "coverage": round((answered / total), 4) if total else 0.0,
            "semantic_fallback_enabled": bool(semantic_fallback),
            "semantic_invocations": semantic_invocations,
            "semantic_overrides": semantic_overrides,
            "cria_shepherd_enabled": bool(cria_shepherd),
            "cria_shepherd_invocations": shepherd_invocations,
            "cria_shepherd_overrides": shepherd_overrides,
            "cria_llm_enabled": bool(cria_llm),
            "cria_llm_invocations": cria_llm_invocations,
            "cria_llm_overrides": cria_llm_overrides,
            "cria_llm_candidate_limit": int(cria_llm_candidates),
            "deterministic_only_accuracy": round((deterministic_correct / total), 4) if total else 0.0,
            "hybrid_accuracy": round((correct / total), 4) if total else 0.0,
            "accuracy_after_retrieval_backoff": round((correct / total), 4) if total else 0.0,
            "schema_profile_confidence": round((sum(schema_confidences) / len(schema_confidences)), 4) if schema_confidences else 0.0,
            "weak_mention_count": weak_mention_count,
            "multi_hypothesis_count": multi_hypothesis_count,
            "retrieval_backoff_count": retrieval_backoff_count,
            "retrieval_recovered_by_backoff": recovered_by_backoff,
            "gold_in_candidates_count": gold_in_candidates_count,
            "gold_missing_count": total - gold_in_candidates_count,
            "recall_at_1": round((gold_recall_at_1 / total), 4) if total else 0.0,
            "recall_at_10": round((gold_recall_at_10 / total), 4) if total else 0.0,
            "recall_at_25": round((gold_recall_at_25 / total), 4) if total else 0.0,
            "recall_at_50": round((gold_recall_at_50 / total), 4) if total else 0.0,
            "recall_at_100": round((gold_recall_at_100 / total), 4) if total else 0.0,
            "mean_gold_rank_after_rerank": round((sum(gold_rank_after_values) / len(gold_rank_after_values)), 4)
            if gold_rank_after_values
            else None,
            "mean_gold_rank_raw": round((sum(gold_rank_raw_values) / len(gold_rank_raw_values)), 4)
            if gold_rank_raw_values
            else None,
        },
        "results": results,
    }


def _normalize_reason_codes(value: object) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return []


def _classify_error_family(result: dict[str, object]) -> str | None:
    if bool(result.get("correct")):
        return None

    abstain = bool(result.get("abstain"))
    resolution_mode = str(result.get("resolution_mode", "") or "")
    reason_codes = set(_normalize_reason_codes(result.get("reason_codes")))
    semantic_trigger = result.get("semantic_trigger")
    shepherd_trigger = result.get("shepherd_trigger")
    trigger_codes = set()
    shepherd_trigger_codes = set()
    hard_filters = result.get("hard_filters", {})
    mention_strength = str(result.get("mention_strength", "") or "")
    schema_confidence = result.get("schema_profile_confidence")
    try:
        schema_confidence_value = float(schema_confidence)
    except (TypeError, ValueError):
        schema_confidence_value = 0.0
    if isinstance(semantic_trigger, dict):
        trigger_codes = set(_normalize_reason_codes(semantic_trigger.get("reason_codes")))
    if isinstance(shepherd_trigger, dict):
        shepherd_trigger_codes = set(_normalize_reason_codes(shepherd_trigger.get("reason_codes")))
    all_trigger_codes = trigger_codes | shepherd_trigger_codes

    if mention_strength == "weak" and isinstance(hard_filters, dict) and any(hard_filters.values()):
        if "no_candidates" in reason_codes or abstain:
            return "overcommitted_hard_filter"
    if not bool(result.get("gold_in_candidates")):
        return "gold_missing_from_candidates"
    if isinstance(result.get("gold_rank_after_rerank"), int) and int(result.get("gold_rank_after_rerank") or 0) > 1:
        selected_features = {}
        gold_diagnostics = result.get("gold_diagnostics")
        if isinstance(gold_diagnostics, dict):
            selected_features = gold_diagnostics.get("selected_candidate_features", {})
        if isinstance(selected_features, dict) and bool(selected_features.get("same_type_resolver_applied")):
            return "same_type_resolver_misfire"
    if "no_candidates" in reason_codes or "no_candidates" in all_trigger_codes:
        return "retrieval_no_candidates"
    if mention_strength == "weak" and (abstain or resolution_mode.startswith("semantic")):
        return "weak_mention_type_error"
    if mention_strength == "weak" and resolution_mode.startswith("shepherd"):
        return "weak_mention_type_error"
    if schema_confidence_value >= 0.75 and (resolution_mode.startswith("semantic") or resolution_mode.startswith("shepherd")):
        return "wrong_schema_induction"
    if "same_label_cluster" in all_trigger_codes:
        return "same_label_ambiguity"
    if "same_type_cluster" in all_trigger_codes:
        return "same_type_ambiguity"
    if "authority_conflict" in all_trigger_codes or "low_authority_top_candidate" in all_trigger_codes:
        return "authority_duplicate_ambiguity"
    if "derivative_or_unsupported_candidate" in all_trigger_codes or "list_or_disambiguation_candidate" in all_trigger_codes:
        return "derivative_candidate_ambiguity"
    if abstain and resolution_mode.startswith("semantic"):
        return "semantic_abstention"
    if abstain and resolution_mode.startswith("shepherd"):
        return "shepherd_abstention"
    if abstain:
        return "deterministic_abstention"
    if resolution_mode.startswith("semantic"):
        return "semantic_misresolution"
    if resolution_mode.startswith("shepherd"):
        return "shepherd_misresolution"
    return "deterministic_misresolution"


def _build_error_report(results: list[dict[str, object]]) -> dict[str, object]:
    failures = [item for item in results if not bool(item.get("correct"))]
    recoveries = [item for item in results if bool(item.get("retrieval_recovered_by_backoff"))]
    family_counter: Counter[str] = Counter()
    resolution_counter: Counter[str] = Counter()
    trigger_counter: Counter[str] = Counter()
    samples: list[dict[str, object]] = []

    for item in failures:
        error_family = _classify_error_family(item)
        if error_family:
            family_counter[error_family] += 1
        resolution_mode = str(item.get("resolution_mode", "") or "unknown")
        resolution_counter[resolution_mode] += 1
        semantic_trigger = item.get("semantic_trigger")
        if isinstance(semantic_trigger, dict):
            for reason_code in _normalize_reason_codes(semantic_trigger.get("reason_codes")):
                trigger_counter[reason_code] += 1
        shepherd_trigger = item.get("shepherd_trigger")
        if isinstance(shepherd_trigger, dict):
            for reason_code in _normalize_reason_codes(shepherd_trigger.get("reason_codes")):
                trigger_counter[f"shepherd:{reason_code}"] += 1
        if len(samples) < 5:
            samples.append(
                {
                    "table_id": item.get("table_id"),
                    "row_id": item.get("row_id"),
                    "col_id": item.get("col_id"),
                    "mention": item.get("mention"),
                    "selected_qid": item.get("selected_qid"),
                    "selected_label": item.get("selected_label"),
                    "gold_rank_after_rerank": item.get("gold_rank_after_rerank"),
                    "gold_rank_raw": item.get("gold_rank_raw"),
                    "gold_in_candidates": item.get("gold_in_candidates"),
                    "abstain": item.get("abstain"),
                    "resolution_mode": item.get("resolution_mode"),
                    "error_family": error_family,
                }
            )

    return {
        "total_failures": len(failures),
        "incorrect_rows": len(failures),
        "abstained_rows": sum(1 for item in failures if bool(item.get("abstain"))),
        "error_families": dict(sorted(family_counter.items())),
        "recovery_families": {"retrieval_recovered_by_backoff": len(recoveries)} if recoveries else {},
        "resolution_modes": dict(sorted(resolution_counter.items())),
        "semantic_trigger_reasons": dict(sorted(trigger_counter.items())),
        "sample_failures": samples,
    }


def _write_json_output(path_text: str, payload: dict[str, object]) -> str:
    path = Path(path_text).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return str(path)


def _write_multi_table_csv(path_text: str, payload: dict[str, object]) -> str:
    path = Path(path_text).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    aggregate_summary = payload.get("aggregate_summary", {})
    error_report = payload.get("error_report", {})
    per_table = payload.get("per_table", [])
    rows: list[dict[str, object]] = []

    if isinstance(aggregate_summary, dict):
        rows.append(
            {
                "row_type": "aggregate",
                "table": "",
                **aggregate_summary,
                "error_families": json.dumps(error_report.get("error_families", {}), ensure_ascii=False)
                if isinstance(error_report, dict)
                else "{}",
                "semantic_trigger_reasons": json.dumps(
                    error_report.get("semantic_trigger_reasons", {}), ensure_ascii=False
                )
                if isinstance(error_report, dict)
                else "{}",
            }
        )

    if isinstance(per_table, list):
        for item in per_table:
            if not isinstance(item, dict):
                continue
            summary = item.get("summary", {})
            table_error_report = item.get("error_report", {})
            table_id = item.get("table")
            if not isinstance(summary, dict):
                continue
            rows.append(
                {
                    "row_type": "table",
                    "table": table_id,
                    **summary,
                    "error_families": json.dumps(table_error_report.get("error_families", {}), ensure_ascii=False)
                    if isinstance(table_error_report, dict)
                    else "{}",
                    "semantic_trigger_reasons": json.dumps(
                        table_error_report.get("semantic_trigger_reasons", {}), ensure_ascii=False
                    )
                    if isinstance(table_error_report, dict)
                    else "{}",
                }
            )

    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = BackendConfig.from_env()
    client = BackendHttpClient(api_url=config.api_url, es_url=config.es_url)
    try:
        if args.command == "datasets":
            _print_json(
                {
                    "dataset_root": str(config.dataset_root),
                    "datasets": list_dataset_ids(config.dataset_root),
                }
            )
            return 0

        if args.command == "table":
            rows = load_table(config.dataset_root, args.dataset, args.table)
            _print_json(
                {
                    "dataset": args.dataset,
                    "table": args.table,
                    "rows_total": len(rows),
                    "preview": rows[: max(1, int(args.limit))],
                }
            )
            return 0

        if args.command == "targets":
            if args.task == "cea":
                targets = iter_cea_targets(config.dataset_root, args.dataset)
                preview = [asdict(target) for target in targets[: max(1, int(args.limit))]]
            else:
                targets = iter_cta_targets(config.dataset_root, args.dataset)
                preview = [asdict(target) for target in targets[: max(1, int(args.limit))]]
            _print_json(
                {
                    "dataset": args.dataset,
                    "task": args.task,
                    "count": len(targets),
                    "preview": preview,
                }
            )
            return 0

        if args.command == "cell-query":
            context = build_cell_context(
                config.dataset_root,
                args.dataset,
                args.table,
                row_id=int(args.row),
                col_id=int(args.col),
            )
            table_profile, _ = _build_table_profile(
                dataset_root=config.dataset_root,
                dataset_id=args.dataset,
                table_id=args.table,
                llm_client=None,
            )
            seed_schema = build_seed_schema(context, table_profile=table_profile).to_dict()
            payload = lookup_payload_from_preprocessing(seed_schema, top_k=int(args.top_k))
            _print_json(
                {
                    "context": context.to_dict(),
                    "preprocessing_schema_seed": seed_schema,
                    "lookup_payload": payload,
                }
            )
            return 0

        if args.command == "llm-preprocess":
            context, seed_schema, llm_result, payload = _build_context_seed_and_llm_result(args, config)
            _print_json(
                {
                    "context": context,
                    "preprocessing_schema_seed": seed_schema,
                    "llm_result": llm_result,
                    "lookup_payload": payload,
                }
            )
            return 0

        if args.command == "table-profile":
            llm_config = _config_with_cache_mode(config, getattr(args, "cache_mode", None))
            llm_client = OpenAICompatibleLlmClient.from_config(
                llm_config,
                model=(args.model or llm_config.llm_model),
            )
            seed_profile = build_table_profile_seed(llm_config.dataset_root, args.dataset, args.table).to_dict()
            llm_result = llm_client.induce_table_profile(seed_profile)
            _print_json(
                {
                    "seed_profile": seed_profile,
                    "llm_result": llm_result,
                    "cache_enabled": llm_config.llm_cache_enabled,
                    "schema_induction_mode": "hybrid",
                }
            )
            return 0

        if args.command == "llm-lookup":
            context, seed_schema, llm_result, payload = _build_context_seed_and_llm_result(args, config)
            result = client.api_lookup(payload, debug=bool(args.debug))
            _print_json(
                {
                    "context": context,
                    "preprocessing_schema_seed": seed_schema,
                    "llm_result": llm_result,
                    "lookup_payload": payload,
                    "response": result,
                }
            )
            return 0

        if args.command == "llm-es-candidates":
            llm_config = _config_with_cache_mode(config, getattr(args, "cache_mode", None))
            llm_client = OpenAICompatibleLlmClient.from_config(
                llm_config,
                model=(args.model or llm_config.llm_model),
            )
            context = build_cell_context(
                llm_config.dataset_root,
                args.dataset,
                args.table,
                row_id=int(args.row),
                col_id=int(args.col),
            )
            result = _run_hybrid_resolution(
                dataset_root=llm_config.dataset_root,
                context=context,
                top_k=int(args.top_k),
                client=client,
                llm_client=llm_client,
                semantic_fallback=bool(args.semantic_fallback),
                cria_shepherd=bool(args.cria_shepherd),
                cria_llm=bool(args.cria_llm),
                cria_llm_candidates=int(args.cria_llm_candidates),
            )
            result["cache_enabled"] = llm_config.llm_cache_enabled
            result["cria_shepherd_enabled"] = bool(args.cria_shepherd)
            result["cria_llm_enabled"] = bool(args.cria_llm)
            _print_json(result)
            return 0

        if args.command == "cea-batch-eval":
            llm_config = _config_with_cache_mode(config, getattr(args, "cache_mode", None))
            llm_client = OpenAICompatibleLlmClient.from_config(
                llm_config,
                model=(args.model or llm_config.llm_model),
            )
            gt_rows = load_cea_ground_truth(llm_config.dataset_root, args.dataset)
            if args.table:
                gt_rows = [row for row in gt_rows if row.table_id == args.table]
            start = max(0, int(args.offset))
            end = start + max(1, int(args.limit))
            selected_rows = gt_rows[start:end]
            evaluation = _evaluate_cea_rows(
                dataset_root=llm_config.dataset_root,
                dataset_id=args.dataset,
                rows=selected_rows,
                top_k=int(args.top_k),
                client=client,
                llm_client=llm_client,
                semantic_fallback=bool(args.semantic_fallback),
                cria_shepherd=bool(args.cria_shepherd),
                cria_llm=bool(args.cria_llm),
                cria_llm_candidates=int(args.cria_llm_candidates),
            )
            summary = evaluation["summary"] if isinstance(evaluation["summary"], dict) else {}
            results = evaluation["results"] if isinstance(evaluation["results"], list) else []
            summary = {
                "dataset": args.dataset,
                "table": args.table or None,
                "offset": start,
                "limit": len(selected_rows),
                "cache_enabled": llm_config.llm_cache_enabled,
                "cache_scope": ["table_profile", "preprocess", "semantic_resolve", "cria_shepherd_resolve", "cria_llm_rank"] if llm_config.llm_cache_enabled else [],
                "schema_profile_cache_enabled": llm_config.llm_cache_enabled,
                "schema_induction_mode": "hybrid",
                **summary,
            }
            _print_json({"summary": summary, "results": results})
            return 0

        if args.command == "multi-table-eval":
            llm_config = _config_with_cache_mode(config, getattr(args, "cache_mode", None))
            llm_client = OpenAICompatibleLlmClient.from_config(
                llm_config,
                model=(args.model or llm_config.llm_model),
            )
            gt_rows = load_cea_ground_truth(llm_config.dataset_root, args.dataset)
            requested_tables = [table_id.strip() for table_id in args.tables if table_id.strip()]
            per_table: list[dict[str, object]] = []
            total_evaluated = 0
            total_answered = 0
            total_abstained = 0
            total_correct = 0
            total_candidate_correct = 0
            total_semantic_invocations = 0
            total_semantic_overrides = 0
            total_shepherd_invocations = 0
            total_shepherd_overrides = 0
            total_cria_llm_invocations = 0
            total_cria_llm_overrides = 0
            total_weak_mentions = 0
            total_multi_hypothesis = 0
            total_retrieval_backoff = 0
            total_backoff_recovered = 0
            total_gold_in_candidates = 0
            total_gold_recall_at_1 = 0.0
            total_gold_recall_at_10 = 0.0
            total_gold_recall_at_25 = 0.0
            total_gold_recall_at_50 = 0.0
            total_gold_recall_at_100 = 0.0
            weighted_gold_rank_after_sum = 0.0
            weighted_gold_rank_after_count = 0
            weighted_gold_rank_raw_sum = 0.0
            weighted_gold_rank_raw_count = 0
            schema_confidence_weighted_sum = 0.0

            for table_id in requested_tables:
                table_rows = [row for row in gt_rows if row.table_id == table_id][: max(1, int(args.limit_per_table))]
                evaluation = _evaluate_cea_rows(
                    dataset_root=llm_config.dataset_root,
                    dataset_id=args.dataset,
                    rows=table_rows,
                    top_k=int(args.top_k),
                    client=client,
                    llm_client=llm_client,
                    semantic_fallback=bool(args.semantic_fallback),
                    cria_shepherd=bool(args.cria_shepherd),
                    cria_llm=bool(args.cria_llm),
                    cria_llm_candidates=int(args.cria_llm_candidates),
                )
                summary = evaluation["summary"] if isinstance(evaluation["summary"], dict) else {}
                results = evaluation["results"] if isinstance(evaluation["results"], list) else []
                error_report = _build_error_report(results)
                per_table.append(
                    {
                        "table": table_id,
                        "summary": {
                            "dataset": args.dataset,
                            "table": table_id,
                            "limit": len(table_rows),
                            "cache_enabled": llm_config.llm_cache_enabled,
                            "cache_scope": ["table_profile", "preprocess", "semantic_resolve", "cria_shepherd_resolve", "cria_llm_rank"] if llm_config.llm_cache_enabled else [],
                            "schema_profile_cache_enabled": llm_config.llm_cache_enabled,
                            "schema_induction_mode": "hybrid",
                            **summary,
                        },
                        "error_report": error_report,
                        "results": results,
                    }
                )
                total_evaluated += int(summary.get("evaluated", 0) or 0)
                total_answered += int(summary.get("answered", 0) or 0)
                total_abstained += int(summary.get("abstained", 0) or 0)
                total_correct += sum(1 for item in results if bool(item.get("correct")))
                total_candidate_correct += sum(1 for item in results if bool(item.get("candidate_is_gold")))
                total_semantic_invocations += int(summary.get("semantic_invocations", 0) or 0)
                total_semantic_overrides += int(summary.get("semantic_overrides", 0) or 0)
                total_shepherd_invocations += int(summary.get("cria_shepherd_invocations", 0) or 0)
                total_shepherd_overrides += int(summary.get("cria_shepherd_overrides", 0) or 0)
                total_cria_llm_invocations += int(summary.get("cria_llm_invocations", 0) or 0)
                total_cria_llm_overrides += int(summary.get("cria_llm_overrides", 0) or 0)
                total_weak_mentions += int(summary.get("weak_mention_count", 0) or 0)
                total_multi_hypothesis += int(summary.get("multi_hypothesis_count", 0) or 0)
                total_retrieval_backoff += int(summary.get("retrieval_backoff_count", 0) or 0)
                total_backoff_recovered += int(summary.get("retrieval_recovered_by_backoff", 0) or 0)
                table_evaluated = int(summary.get("evaluated", 0) or 0)
                total_gold_in_candidates += int(summary.get("gold_in_candidates_count", 0) or 0)
                total_gold_recall_at_1 += float(summary.get("recall_at_1", 0.0) or 0.0) * table_evaluated
                total_gold_recall_at_10 += float(summary.get("recall_at_10", 0.0) or 0.0) * table_evaluated
                total_gold_recall_at_25 += float(summary.get("recall_at_25", 0.0) or 0.0) * table_evaluated
                total_gold_recall_at_50 += float(summary.get("recall_at_50", 0.0) or 0.0) * table_evaluated
                total_gold_recall_at_100 += float(summary.get("recall_at_100", 0.0) or 0.0) * table_evaluated
                if isinstance(summary.get("mean_gold_rank_after_rerank"), (int, float)):
                    count = int(summary.get("gold_in_candidates_count", 0) or 0)
                    weighted_gold_rank_after_sum += float(summary.get("mean_gold_rank_after_rerank", 0.0) or 0.0) * count
                    weighted_gold_rank_after_count += count
                if isinstance(summary.get("mean_gold_rank_raw"), (int, float)):
                    count = int(summary.get("gold_in_candidates_count", 0) or 0)
                    weighted_gold_rank_raw_sum += float(summary.get("mean_gold_rank_raw", 0.0) or 0.0) * count
                    weighted_gold_rank_raw_count += count
                schema_confidence_weighted_sum += float(summary.get("schema_profile_confidence", 0.0) or 0.0) * max(1, len(table_rows))

            aggregate_results = [
                item
                for table_payload in per_table
                for item in table_payload.get("results", [])
                if isinstance(item, dict)
            ]
            aggregate_summary = {
                "dataset": args.dataset,
                "tables": requested_tables,
                "limit_per_table": int(args.limit_per_table),
                "tables_evaluated": len(per_table),
                "evaluated": total_evaluated,
                "answered": total_answered,
                "abstained": total_abstained,
                "end_to_end_accuracy": round((total_correct / total_evaluated), 4) if total_evaluated else 0.0,
                "top_candidate_accuracy": round((total_candidate_correct / total_evaluated), 4)
                if total_evaluated
                else 0.0,
                "answered_accuracy": round((total_correct / total_answered), 4) if total_answered else 0.0,
                "coverage": round((total_answered / total_evaluated), 4) if total_evaluated else 0.0,
                "semantic_fallback_enabled": bool(args.semantic_fallback),
                "semantic_invocations": total_semantic_invocations,
                "semantic_overrides": total_semantic_overrides,
                "cria_shepherd_enabled": bool(args.cria_shepherd),
                "cria_shepherd_invocations": total_shepherd_invocations,
                "cria_shepherd_overrides": total_shepherd_overrides,
                "cria_llm_enabled": bool(args.cria_llm),
                "cria_llm_invocations": total_cria_llm_invocations,
                "cria_llm_overrides": total_cria_llm_overrides,
                "cria_llm_candidate_limit": int(args.cria_llm_candidates),
                "deterministic_only_accuracy": round(
                    (
                        sum(float(item.get("summary", {}).get("deterministic_only_accuracy", 0.0) or 0.0) * max(1, int(item.get("summary", {}).get("evaluated", 0) or 0)) for item in per_table)
                        / total_evaluated
                    ),
                    4,
                ) if total_evaluated else 0.0,
                "hybrid_accuracy": round((total_correct / total_evaluated), 4) if total_evaluated else 0.0,
                "accuracy_after_retrieval_backoff": round((total_correct / total_evaluated), 4) if total_evaluated else 0.0,
                "schema_profile_confidence": round((schema_confidence_weighted_sum / total_evaluated), 4) if total_evaluated else 0.0,
                "weak_mention_count": total_weak_mentions,
                "multi_hypothesis_count": total_multi_hypothesis,
                "retrieval_backoff_count": total_retrieval_backoff,
                "retrieval_recovered_by_backoff": total_backoff_recovered,
                "gold_in_candidates_count": total_gold_in_candidates,
                "gold_missing_count": total_evaluated - total_gold_in_candidates,
                "recall_at_1": round((total_gold_recall_at_1 / total_evaluated), 4) if total_evaluated else 0.0,
                "recall_at_10": round((total_gold_recall_at_10 / total_evaluated), 4) if total_evaluated else 0.0,
                "recall_at_25": round((total_gold_recall_at_25 / total_evaluated), 4) if total_evaluated else 0.0,
                "recall_at_50": round((total_gold_recall_at_50 / total_evaluated), 4) if total_evaluated else 0.0,
                "recall_at_100": round((total_gold_recall_at_100 / total_evaluated), 4) if total_evaluated else 0.0,
                "mean_gold_rank_after_rerank": round((weighted_gold_rank_after_sum / weighted_gold_rank_after_count), 4)
                if weighted_gold_rank_after_count
                else None,
                "mean_gold_rank_raw": round((weighted_gold_rank_raw_sum / weighted_gold_rank_raw_count), 4)
                if weighted_gold_rank_raw_count
                else None,
                "cache_enabled": llm_config.llm_cache_enabled,
                "cache_scope": ["table_profile", "preprocess", "semantic_resolve", "cria_shepherd_resolve", "cria_llm_rank"] if llm_config.llm_cache_enabled else [],
                "schema_profile_cache_enabled": llm_config.llm_cache_enabled,
                "schema_induction_mode": "hybrid",
            }
            output_payload: dict[str, object] = {
                "aggregate_summary": aggregate_summary,
                "error_report": _build_error_report(aggregate_results),
                "per_table": per_table,
            }
            json_out = str(args.json_out or "").strip()
            csv_out = str(args.csv_out or "").strip()
            export_paths: dict[str, str] = {}
            if json_out:
                export_paths["json_out"] = _write_json_output(json_out, output_payload)
            if csv_out:
                export_paths["csv_out"] = _write_multi_table_csv(csv_out, output_payload)

            rendered_payload = output_payload
            if bool(args.hide_results) or bool(args.summary_only):
                rendered_payload = {
                    "aggregate_summary": aggregate_summary,
                    "error_report": output_payload["error_report"],
                    "per_table": [
                        {
                            "table": item.get("table"),
                            "summary": item.get("summary"),
                            "error_report": item.get("error_report"),
                        }
                        for item in per_table
                    ],
                }
            if export_paths:
                rendered_payload = {
                    **rendered_payload,
                    "export_paths": export_paths,
                }
            _print_json(rendered_payload)
            return 0

        if args.command == "lookup":
            context, payload = _context_and_payload(args, config)
            result = client.api_lookup(payload, debug=bool(args.debug))
            _print_json({"context": context, "lookup_payload": payload, "response": result})
            return 0

        if args.command == "es-search":
            result = client.es_search(
                query_text=args.query,
                size=int(args.size),
                coarse_type=args.coarse_type,
                fine_type=args.fine_type,
            )
            _print_json(result)
            return 0

        if args.command == "qid":
            _print_json(client.es_get_qid(args.qid))
            return 0

        if args.command == "api-health":
            _print_json(client.api_health())
            return 0

        if args.command == "es-indices":
            _print_json(client.es_indices())
            return 0
    except (
        BackendRequestError,
        FileNotFoundError,
        IndexError,
        LlmConfigurationError,
        LlmRequestError,
        ValueError,
    ) as exc:
        _print_json({"error": str(exc)})
        return 1

    parser.error(f"Unknown command: {args.command}")
    return 2
