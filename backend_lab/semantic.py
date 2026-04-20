from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .dataset import CellContext


@dataclass(frozen=True, slots=True)
class SemanticTrigger:
    should_run: bool
    reason_codes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SemanticCandidate:
    qid: str
    label: str
    candidate_family: str
    final_score: float
    heuristic_score: float
    prior: float
    has_wikipedia: bool
    coarse_type: str
    fine_type: str
    item_category: str
    description: str
    context_string: str
    reranked_rank: int
    raw_rank: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _normalize_label(text: str) -> str:
    return " ".join(part for part in "".join(char.lower() if char.isalnum() else " " for char in text).split() if part)


def should_run_semantic_fallback(
    *,
    decision: dict[str, object],
    reranked_candidates: list[dict[str, object]],
    max_candidates: int = 5,
) -> SemanticTrigger:
    if not reranked_candidates:
        return SemanticTrigger(should_run=False, reason_codes=("no_candidates",))

    reason_codes: list[str] = []
    if bool(decision.get("abstain")):
        reason_codes.append("deterministic_abstain")

    margin = decision.get("margin")
    try:
        numeric_margin = float(margin)
    except (TypeError, ValueError):
        numeric_margin = 0.0
    if numeric_margin < 2.0:
        reason_codes.append("low_margin")

    confidence = decision.get("confidence")
    try:
        numeric_confidence = float(confidence)
    except (TypeError, ValueError):
        numeric_confidence = 0.0
    deterministic_abstain = bool(decision.get("abstain"))
    strong_deterministic = (not deterministic_abstain) and numeric_confidence >= 0.84 and numeric_margin >= 2.5
    if strong_deterministic:
        return SemanticTrigger(should_run=False, reason_codes=("strong_deterministic_decision",))

    if numeric_confidence < 0.78:
        reason_codes.append("low_confidence")

    top_candidate = reranked_candidates[0] if reranked_candidates else {}
    top_features = top_candidate.get("features", {}) if isinstance(top_candidate, dict) else {}
    top_label = top_candidate.get("label") if isinstance(top_candidate, dict) else ""
    second_label = ""
    if len(reranked_candidates) > 1 and isinstance(reranked_candidates[1], dict):
        second_label = reranked_candidates[1].get("label") if isinstance(reranked_candidates[1].get("label"), str) else ""

    try:
        best_name_similarity = float(top_features.get("best_name_similarity", 0.0))
    except (TypeError, ValueError):
        best_name_similarity = 0.0
    try:
        row_template_alignment = float(top_features.get("row_template_alignment", 0.0))
    except (TypeError, ValueError):
        row_template_alignment = 0.0
    top_family = str(top_features.get("candidate_family", "")).strip()
    same_label_top2 = bool(
        isinstance(top_label, str)
        and isinstance(second_label, str)
        and top_label.strip()
        and second_label.strip()
        and _normalize_label(top_label) == _normalize_label(second_label)
    )

    typo_robust_deterministic = (
        not deterministic_abstain
        and top_family == "PRIMARY"
        and not same_label_top2
        and (
            (best_name_similarity >= 0.84 and numeric_confidence >= 0.7)
            or (best_name_similarity >= 0.76 and numeric_confidence >= 0.66 and numeric_margin >= 1.5)
            or (
                bool(top_features.get("schema_family_match"))
                and bool(top_features.get("column_role_match"))
                and row_template_alignment >= 0.35
                and numeric_confidence >= 0.74
            )
        )
    )
    if typo_robust_deterministic:
        return SemanticTrigger(should_run=False, reason_codes=("strong_fuzzy_deterministic_decision",))

    confident_non_clustered_deterministic = (
        not deterministic_abstain
        and top_family == "PRIMARY"
        and not same_label_top2
        and best_name_similarity >= 0.9
        and numeric_confidence >= 0.8
    )
    if confident_non_clustered_deterministic:
        return SemanticTrigger(should_run=False, reason_codes=("confident_non_clustered_deterministic_decision",))

    top_slice = reranked_candidates[: max(2, min(max_candidates, len(reranked_candidates)))]
    top_labels = {
        str(item.get("label", "")).strip().casefold()
        for item in top_slice
        if isinstance(item.get("label"), str) and str(item.get("label")).strip()
    }
    top_families = {
        str(item.get("features", {}).get("candidate_family", "")).strip()
        for item in top_slice
        if isinstance(item.get("features"), dict)
    }
    top_types = {
        (
            str(item.get("source", {}).get("coarse_type", "")).strip(),
            str(item.get("source", {}).get("fine_type", "")).strip(),
        )
        for item in top_slice
        if isinstance(item.get("source"), dict)
    }

    uncertainty_gate = deterministic_abstain or numeric_margin < 1.5 or numeric_confidence < 0.78

    if uncertainty_gate and len(top_labels) == 1 and len(top_slice) > 1:
        reason_codes.append("same_label_cluster")
    if uncertainty_gate and len(top_types) == 1 and len(top_slice) > 1:
        reason_codes.append("same_type_cluster")
    if uncertainty_gate and top_families and top_families <= {"PRIMARY"} and len(top_slice) > 1:
        reason_codes.append("primary_family_competition")

    should_run = bool(reason_codes)
    return SemanticTrigger(should_run=should_run, reason_codes=tuple(dict.fromkeys(reason_codes)))


def _candidate_type_key(item: dict[str, object]) -> tuple[str, str, str]:
    source = item.get("source", {})
    if not isinstance(source, dict):
        source = {}
    return (
        str(source.get("item_category", "") or "").strip(),
        str(source.get("coarse_type", "") or "").strip(),
        str(source.get("fine_type", "") or "").strip(),
    )


def _float_feature(item: dict[str, object], key: str, default: float = 0.0) -> float:
    features = item.get("features", {})
    if not isinstance(features, dict):
        return default
    try:
        return float(features.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def _int_rank(item: dict[str, object], key: str, default: int = 10**9) -> int:
    try:
        return int(item.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def _source_text(item: dict[str, object]) -> str:
    source = item.get("source", {})
    if not isinstance(source, dict):
        return ""
    return " ".join(
        str(source.get(key, "") or "")
        for key in ("description", "context_string", "wikipedia_url", "dbpedia_url")
    ).casefold()


def _looks_like_list_or_stub(item: dict[str, object]) -> bool:
    text = _source_text(item)
    list_markers = (
        "wikimedia set index",
        "set index article",
        "list of",
        "disambiguation",
        "category:",
        "surname",
        "given name",
    )
    return any(marker in text for marker in list_markers)


def should_run_cria_shepherd(
    *,
    decision: dict[str, object],
    reranked_candidates: list[dict[str, object]],
    max_candidates: int = 12,
) -> SemanticTrigger:
    """Decide whether CRIA-Shepherd should adjudicate a hard ambiguity.

    The older semantic fallback is intentionally conservative. Shepherd is a
    stronger LLM adjudication stage for cases where deterministic ranking has
    likely retrieved the answer but may be choosing between close semantic
    duplicates, derivatives, or low-authority stubs.
    """
    if not reranked_candidates:
        return SemanticTrigger(should_run=False, reason_codes=("no_candidates",))

    try:
        margin = float(decision.get("margin", 0.0) or 0.0)
    except (TypeError, ValueError):
        margin = 0.0
    try:
        confidence = float(decision.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    deterministic_abstain = bool(decision.get("abstain"))
    low_margin = margin < 2.0
    very_low_confidence = confidence < 0.75
    low_confidence = confidence < 0.82

    top = reranked_candidates[0]
    top_label = top.get("label") if isinstance(top.get("label"), str) else ""
    normalized_top_label = _normalize_label(top_label) if isinstance(top_label, str) else ""
    top_type = _candidate_type_key(top)
    top_prior = _float_feature(top, "prior")
    top_score = _float_feature(top, "final_score")
    top_unsupported = _float_feature(top, "unsupported_qualifier_count")
    top_derivative_penalty = _float_feature(top, "derivative_penalty")
    top_mention_coverage = _float_feature(top, "mention_token_coverage")
    top_extra_label_tokens = _float_feature(top, "extra_label_token_count")
    top_expected_descriptor_overlap = _float_feature(top, "expected_descriptor_overlap")
    top_weighted_context_overlap = _float_feature(top, "weighted_context_overlap")
    top_label_context_support = _float_feature(top, "label_context_support_score")
    top_has_parenthetical = bool(top.get("features", {}).get("label_has_parenthetical_qualifier")) if isinstance(top.get("features"), dict) else False
    top_is_list_or_stub = _looks_like_list_or_stub(top)

    scan_limit = max(max_candidates, 25)
    scanned = reranked_candidates[:scan_limit]
    same_label_candidates = [
        item
        for item in scanned
        if isinstance(item.get("label"), str) and normalized_top_label and _normalize_label(str(item.get("label"))) == normalized_top_label
    ]
    same_type_candidates = [item for item in reranked_candidates[:max_candidates] if _candidate_type_key(item) == top_type]

    stronger_authority_same_label = [
        item
        for item in same_label_candidates[1:]
        if _float_feature(item, "prior") >= top_prior + 0.18
        or _float_feature(item, "weighted_context_overlap") > _float_feature(top, "weighted_context_overlap") + 0.5
    ]
    authority_conflict = bool(stronger_authority_same_label)

    if top_prior < 0.2:
        high_authority_competitors = [
            item
            for item in scanned[1:]
            if _float_feature(item, "prior") >= 0.35
            and (
                _candidate_type_key(item) == top_type
                or _float_feature(item, "best_name_similarity") >= 0.75
                or (
                    bool(normalized_top_label)
                    and _normalize_label(str(item.get("label", ""))).endswith(normalized_top_label)
                )
            )
        ]
        low_authority_top_candidate = bool(high_authority_competitors)
    else:
        low_authority_top_candidate = False

    derivative_or_unsupported = bool(
        top_unsupported > 0
        or top_derivative_penalty >= 1.0
        or top_has_parenthetical
        or top_is_list_or_stub
    )

    close_competitors = [
        item
        for item in reranked_candidates[1:max_candidates]
        if top_score - _float_feature(item, "final_score") < 2.5
    ]
    has_close_competitor = bool(close_competitors)
    same_label_cluster = len(same_label_candidates) > 1
    same_type_cluster = len(same_type_candidates) > 2
    label_competitors = [
        item
        for item in scanned[1:]
        if _candidate_type_key(item) == top_type
        and _float_feature(item, "mention_token_coverage") >= max(0.6, top_mention_coverage - 0.05)
        and top_score - _float_feature(item, "final_score") < 4.5
    ]
    cleaner_label_competitors = [
        item
        for item in label_competitors
        if top_extra_label_tokens >= 2
        and _float_feature(item, "extra_label_token_count") < top_extra_label_tokens
        and (
            _float_feature(item, "prior") >= top_prior + 0.12
            or _float_feature(item, "label_context_support_score") >= top_label_context_support + 0.08
            or _float_feature(item, "expected_descriptor_overlap") > top_expected_descriptor_overlap
        )
    ]
    underspecified_label_competitors = [
        item
        for item in label_competitors
        if top_extra_label_tokens == 0
        and top_mention_coverage >= 0.95
        and _float_feature(item, "extra_label_token_count") <= 3
        and _float_feature(item, "expected_descriptor_overlap") > top_expected_descriptor_overlap
        and (
            _float_feature(item, "prior") >= top_prior + 0.12
            or _float_feature(item, "weighted_context_overlap") >= top_weighted_context_overlap + 0.35
            or _float_feature(item, "label_context_support_score") >= top_label_context_support + 0.08
        )
    ]
    label_context_ambiguity = bool(cleaner_label_competitors or underspecified_label_competitors)
    same_type_hard_ambiguity = bool(
        same_type_cluster
        and has_close_competitor
        and (
            deterministic_abstain
            or very_low_confidence
            or derivative_or_unsupported
            or low_authority_top_candidate
            or label_context_ambiguity
        )
    )

    reason_codes: list[str] = []
    if deterministic_abstain:
        reason_codes.append("deterministic_abstain")
    if low_margin and (
        deterministic_abstain
        or same_label_cluster
        or derivative_or_unsupported
        or low_authority_top_candidate
        or confidence < 0.78
    ):
        reason_codes.append("low_margin")
    if very_low_confidence and (low_margin or deterministic_abstain or has_close_competitor):
        reason_codes.append("low_confidence")
    if same_label_cluster and (low_margin or authority_conflict or low_authority_top_candidate or deterministic_abstain):
        reason_codes.append("same_label_cluster")
    if same_type_hard_ambiguity:
        reason_codes.append("same_type_cluster")
    if cleaner_label_competitors:
        reason_codes.append("cleaner_label_competitor")
    if underspecified_label_competitors:
        reason_codes.append("underspecified_label_competitor")
    if authority_conflict:
        reason_codes.append("authority_conflict")
    if low_authority_top_candidate and (
        same_label_cluster
        or has_close_competitor
        or low_margin
        or derivative_or_unsupported
    ):
        reason_codes.append("low_authority_top_candidate")
    if derivative_or_unsupported:
        reason_codes.append("derivative_or_unsupported_candidate")
    if top_is_list_or_stub:
        reason_codes.append("list_or_disambiguation_candidate")
    if has_close_competitor and (
        same_label_cluster
        or low_authority_top_candidate
        or derivative_or_unsupported
        or deterministic_abstain
        or label_context_ambiguity
    ):
        reason_codes.append("close_competitor")

    if not reason_codes and confidence >= 0.9 and margin >= 3.0:
        return SemanticTrigger(should_run=False, reason_codes=("strong_unambiguous_deterministic_decision",))

    return SemanticTrigger(should_run=bool(reason_codes), reason_codes=tuple(dict.fromkeys(reason_codes)))


def build_semantic_candidates(
    reranked_candidates: list[dict[str, object]],
    *,
    max_candidates: int = 5,
    anchor_label: str = "",
) -> list[SemanticCandidate]:
    limit = max(1, int(max_candidates))
    scan_limit = max(12, limit * 6)
    if not reranked_candidates:
        return []

    chosen_items: list[dict[str, object]] = []
    seen_qids: set[str] = set()
    target_count = max(limit, 8)

    def _add_item(item: dict[str, object]) -> None:
        qid = item.get("qid")
        if not isinstance(qid, str) or not qid or qid in seen_qids:
            return
        seen_qids.add(qid)
        chosen_items.append(item)

    normalized_anchor_label = _normalize_label(anchor_label) if anchor_label.strip() else ""
    if not normalized_anchor_label:
        top_label = reranked_candidates[0].get("label")
        normalized_anchor_label = _normalize_label(top_label) if isinstance(top_label, str) else ""

    for item in reranked_candidates[: min(3, len(reranked_candidates))]:
        _add_item(item)

    if normalized_anchor_label:
        same_label_cluster = [
            item
            for item in reranked_candidates[: max(scan_limit, limit)]
            if isinstance(item.get("label"), str) and _normalize_label(str(item.get("label"))) == normalized_anchor_label
        ]
        same_label_cluster.sort(
            key=lambda item: (
                float(item.get("features", {}).get("prior", 0.0)) if isinstance(item.get("features"), dict) else 0.0,
                float(item.get("features", {}).get("weighted_context_overlap", 0.0)) if isinstance(item.get("features"), dict) else 0.0,
                -int(item.get("reranked_rank", 0) or 0),
            ),
            reverse=True,
        )
        for item in same_label_cluster:
            if len([item for item in chosen_items if isinstance(item.get("label"), str) and _normalize_label(str(item.get("label"))) == normalized_anchor_label]) >= min(5, target_count):
                break
            _add_item(item)

    label_context_candidates = sorted(
        reranked_candidates[: max(scan_limit, limit)],
        key=lambda item: (
            _float_feature(item, "mention_token_coverage"),
            _float_feature(item, "label_context_support_score"),
            _float_feature(item, "prior"),
            _float_feature(item, "weighted_context_overlap"),
            -_int_rank(item, "reranked_rank"),
        ),
        reverse=True,
    )
    for item in label_context_candidates:
        if len(chosen_items) >= target_count:
            break
        if _float_feature(item, "mention_token_coverage") < 0.6:
            continue
        if _float_feature(item, "label_context_support_score") < 0.45 and _float_feature(item, "prior") < 0.35:
            continue
        _add_item(item)

    for item in reranked_candidates[:limit]:
        if len(chosen_items) >= target_count:
            break
        _add_item(item)

    high_prior_candidates = sorted(
        reranked_candidates[: max(scan_limit, limit)],
        key=lambda item: (
            float(item.get("features", {}).get("prior", 0.0)) if isinstance(item.get("features"), dict) else 0.0,
            float(item.get("features", {}).get("weighted_context_overlap", 0.0)) if isinstance(item.get("features"), dict) else 0.0,
        ),
        reverse=True,
    )
    for item in high_prior_candidates:
        if len(chosen_items) >= target_count:
            break
        _add_item(item)

    output: list[SemanticCandidate] = []
    for item in chosen_items:
        features = item.get("features", {})
        source = item.get("source", {})
        if not isinstance(features, dict) or not isinstance(source, dict):
            continue
        qid = item.get("qid")
        label = item.get("label")
        if not isinstance(qid, str) or not qid or not isinstance(label, str) or not label:
            continue
        output.append(
            SemanticCandidate(
                qid=qid,
                label=label,
                candidate_family=str(features.get("candidate_family", "")),
                final_score=float(features.get("final_score", 0.0)),
                heuristic_score=float(features.get("heuristic_score", 0.0)),
                prior=float(features.get("prior", 0.0)),
                has_wikipedia=bool(features.get("has_wikipedia")),
                coarse_type=str(source.get("coarse_type", "")),
                fine_type=str(source.get("fine_type", "")),
                item_category=str(source.get("item_category", "")),
                description=str(source.get("description", "") or ""),
                context_string=str(source.get("context_string", "") or ""),
                reranked_rank=int(item.get("reranked_rank", 0) or 0),
                raw_rank=int(item.get("raw_rank", 0) or 0),
            )
        )
    return output


def build_semantic_payload(
    *,
    context: CellContext,
    preprocessing_schema: dict[str, Any],
    reranked_candidates: list[dict[str, object]],
    decision: dict[str, object],
    max_candidates: int = 5,
) -> dict[str, object]:
    cell_hypothesis = preprocessing_schema.get("cell_hypothesis", {})
    retrieval_plan = preprocessing_schema.get("retrieval_plan", {})
    table_profile = preprocessing_schema.get("table_profile", {})

    return {
        "context": {
            "dataset_id": context.dataset_id,
            "table_id": context.table_id,
            "row_id": context.row_id,
            "col_id": context.col_id,
            "mention": context.mention,
            "column_name": context.column_name,
            "row_values": context.row_values,
            "other_row_values": context.other_row_values,
            "sampled_column_values": context.sampled_column_values,
        },
        "preprocessing_summary": {
            "canonical_mention": cell_hypothesis.get("canonical_mention"),
            "entity_category": cell_hypothesis.get("entity_category"),
            "coarse_type": cell_hypothesis.get("coarse_type"),
            "fine_type": cell_hypothesis.get("fine_type"),
            "entity_hypotheses": cell_hypothesis.get("entity_hypotheses", []),
            "mention_strength": cell_hypothesis.get("mention_strength"),
            "description_hints": cell_hypothesis.get("description_hints", []),
            "hard_filters": retrieval_plan.get("hard_filters", {}),
            "soft_context_terms": retrieval_plan.get("soft_context_terms", []),
            "table_profile": {
                "table_semantic_family": table_profile.get("table_semantic_family"),
                "confidence": table_profile.get("confidence"),
                "column_role": (
                    table_profile.get("column_roles", {}).get(str(context.col_id), [{}])[0]
                    if isinstance(table_profile.get("column_roles"), dict)
                    else None
                ),
                "row_template": table_profile.get("row_template", []),
            },
        },
        "deterministic_decision": decision,
        "candidates": [
            candidate.to_dict()
            for candidate in build_semantic_candidates(
                reranked_candidates,
                max_candidates=max_candidates,
                anchor_label=str(decision.get("selected_label", "") or ""),
            )
        ],
    }


def build_shepherd_payload(
    *,
    context: CellContext,
    preprocessing_schema: dict[str, Any],
    reranked_candidates: list[dict[str, object]],
    decision: dict[str, object],
    trigger: SemanticTrigger,
    max_candidates: int = 12,
) -> dict[str, object]:
    payload = build_semantic_payload(
        context=context,
        preprocessing_schema=preprocessing_schema,
        reranked_candidates=reranked_candidates,
        decision=decision,
        max_candidates=max_candidates,
    )
    payload["adjudicator"] = {
        "name": "cria-shepherd",
        "purpose": "LLM adjudication over a deterministic CRIA shortlist.",
        "trigger_reason_codes": list(trigger.reason_codes),
        "candidate_policy": "Select only from provided candidates or abstain.",
        "normal_path": "ES top-k retrieval -> deterministic CRIA reranking -> Shepherd hard-ambiguity adjudication.",
    }
    return payload


def _candidate_payload_for_llm_ranker(item: dict[str, object]) -> dict[str, object] | None:
    features = item.get("features", {})
    source = item.get("source", {})
    if not isinstance(features, dict) or not isinstance(source, dict):
        return None
    qid = item.get("qid")
    label = item.get("label")
    if not isinstance(qid, str) or not qid or not isinstance(label, str) or not label:
        return None

    def _short_text(value: object, limit: int = 360) -> str:
        text = str(value or "")
        return text if len(text) <= limit else f"{text[:limit].rstrip()}..."

    feature_keys = (
        "final_score",
        "heuristic_score",
        "raw_es_score",
        "prior",
        "best_name_similarity",
        "weighted_context_overlap",
        "label_context_support_score",
        "row_template_alignment",
        "candidate_family",
        "primary_entity_score",
        "derivative_penalty",
        "unsupported_qualifier_count",
        "context_unsupported_extra_label_tokens",
    )
    return {
        "qid": qid,
        "label": label,
        "reranked_rank": item.get("reranked_rank"),
        "raw_rank": item.get("raw_rank"),
        "features": {key: features.get(key) for key in feature_keys if key in features},
        "source": {
            "item_category": source.get("item_category"),
            "coarse_type": source.get("coarse_type"),
            "fine_type": source.get("fine_type"),
            "description": _short_text(source.get("description"), 240),
            "context_string": _short_text(source.get("context_string"), 360),
            "wikipedia_url": source.get("wikipedia_url"),
            "dbpedia_url": source.get("dbpedia_url"),
        },
    }


def build_cria_llm_payload(
    *,
    context: CellContext,
    preprocessing_schema: dict[str, Any],
    lookup_payload: dict[str, Any],
    reranked_candidates: list[dict[str, object]],
    deterministic_decision: dict[str, object],
    max_candidates: int = 20,
) -> dict[str, object]:
    cell_hypothesis = preprocessing_schema.get("cell_hypothesis", {})
    retrieval_plan = preprocessing_schema.get("retrieval_plan", {})
    table_profile = preprocessing_schema.get("table_profile", {})
    compact_candidates = []
    candidate_by_qid = {
        str(item.get("qid")): item
        for item in reranked_candidates
        if isinstance(item.get("qid"), str)
    }
    candidate_order = [
        candidate.qid
        for candidate in build_semantic_candidates(
            reranked_candidates,
            max_candidates=max(1, int(max_candidates)),
            anchor_label=str(deterministic_decision.get("selected_label", "") or ""),
        )
    ]
    if not candidate_order:
        candidate_order = [
            str(item.get("qid"))
            for item in reranked_candidates[: max(1, int(max_candidates))]
            if isinstance(item.get("qid"), str)
        ]
    for qid in candidate_order[: max(1, int(max_candidates))]:
        item = candidate_by_qid.get(qid)
        if item is None:
            continue
        payload_item = _candidate_payload_for_llm_ranker(item)
        if payload_item is not None:
            compact_candidates.append(payload_item)

    return {
        "ranker": {
            "name": "cria-llm",
            "purpose": "Complete LLM ranking over a retrieved CRIA candidate pack.",
            "candidate_policy": "Rank every provided candidate exactly once. Select only from provided qids or abstain.",
            "normal_path": "table/cell preprocessing -> ES top-k retrieval -> deterministic features -> LLM ranking over candidate pack.",
            "ranking_instruction": "Rank independently by semantic fit to row/table evidence. Do not copy deterministic rank or final_score.",
            "candidate_count": len(compact_candidates),
        },
        "context": {
            "dataset_id": context.dataset_id,
            "table_id": context.table_id,
            "row_id": context.row_id,
            "col_id": context.col_id,
            "mention": context.mention,
            "column_name": context.column_name,
            "row_values": context.row_values,
            "other_row_values": context.other_row_values,
            "sampled_column_values": context.sampled_column_values,
            "mention_context": context.mention_context,
        },
        "preprocessing_summary": {
            "canonical_mention": cell_hypothesis.get("canonical_mention"),
            "mention_variants": cell_hypothesis.get("mention_variants", []),
            "entity_hypotheses": cell_hypothesis.get("entity_hypotheses", []),
            "mention_strength": cell_hypothesis.get("mention_strength"),
            "description_hints": cell_hypothesis.get("description_hints", []),
            "hard_filters": retrieval_plan.get("hard_filters", {}),
            "soft_context_terms": retrieval_plan.get("soft_context_terms", []),
            "table_profile": {
                "table_semantic_family": table_profile.get("table_semantic_family"),
                "confidence": table_profile.get("confidence"),
                "column_role": (
                    table_profile.get("column_roles", {}).get(str(context.col_id), [{}])[0]
                    if isinstance(table_profile.get("column_roles"), dict)
                    else None
                ),
                "row_template": table_profile.get("row_template", []),
                "evidence_notes": table_profile.get("evidence_notes", []),
            },
        },
        "lookup_payload": lookup_payload,
        "deterministic_baseline": deterministic_decision,
        "candidates": compact_candidates,
    }


def merge_semantic_decision(
    *,
    deterministic_decision: dict[str, object],
    semantic_result: dict[str, object] | None,
    reranked_candidates: list[dict[str, object]],
) -> dict[str, object]:
    if not semantic_result:
        merged = dict(deterministic_decision)
        merged["resolution_mode"] = "deterministic_only"
        return merged

    merged = dict(deterministic_decision)
    merged["semantic_result"] = semantic_result

    semantic_qid = semantic_result.get("selected_qid")
    semantic_abstain = bool(semantic_result.get("abstain"))
    if not isinstance(semantic_qid, str):
        semantic_qid = ""

    candidate_by_qid = {
        str(item.get("qid")): item
        for item in reranked_candidates
        if isinstance(item.get("qid"), str)
    }
    if semantic_abstain or not semantic_qid or semantic_qid not in candidate_by_qid:
        merged["resolution_mode"] = "deterministic_kept"
        return merged

    deterministic_qid = deterministic_decision.get("selected_qid")
    if not isinstance(deterministic_qid, str):
        deterministic_qid = ""

    try:
        deterministic_confidence = float(deterministic_decision.get("confidence", 0.0))
    except (TypeError, ValueError):
        deterministic_confidence = 0.0
    try:
        deterministic_margin = float(deterministic_decision.get("margin", 0.0))
    except (TypeError, ValueError):
        deterministic_margin = 0.0
    try:
        semantic_confidence = float(semantic_result.get("confidence", 0.0))
    except (TypeError, ValueError):
        semantic_confidence = 0.0

    if semantic_qid == deterministic_qid:
        merged["semantic_agreement"] = True
        merged["resolution_mode"] = "semantic_confirmed"
        merged["confidence"] = round(max(deterministic_confidence, semantic_confidence), 4)
        merged["abstain"] = False
        return merged

    if (
        bool(deterministic_decision.get("abstain"))
        and semantic_confidence >= 0.72
    ):
        chosen = candidate_by_qid[semantic_qid]
        merged.update(
            {
                "selected_qid": semantic_qid,
                "selected_label": chosen.get("label"),
                "confidence": round(semantic_confidence, 4),
                "abstain": False,
                "resolution_mode": "semantic_override_after_abstain",
                "reason_codes": ["semantic_override_after_abstain"],
            }
        )
        return merged

    if (
        deterministic_margin < 1.5
        and deterministic_confidence < 0.82
        and semantic_confidence >= 0.82
    ):
        chosen = candidate_by_qid[semantic_qid]
        merged.update(
            {
                "selected_qid": semantic_qid,
                "selected_label": chosen.get("label"),
                "confidence": round(semantic_confidence, 4),
                "abstain": False,
                "resolution_mode": "semantic_override_low_margin",
                "reason_codes": ["semantic_override_low_margin"],
            }
        )
        return merged

    merged["resolution_mode"] = "deterministic_kept"
    return merged


def merge_cria_llm_decision(
    *,
    deterministic_decision: dict[str, object],
    cria_llm_result: dict[str, object] | None,
    reranked_candidates: list[dict[str, object]],
) -> dict[str, object]:
    merged = dict(deterministic_decision)
    merged["resolution_mode"] = "cria_llm_unavailable"
    if not cria_llm_result:
        return merged

    merged["cria_llm_result"] = cria_llm_result
    candidate_by_qid = {
        str(item.get("qid")): item
        for item in reranked_candidates
        if isinstance(item.get("qid"), str)
    }
    selected_qid = cria_llm_result.get("selected_qid")
    if not isinstance(selected_qid, str):
        selected_qid = ""
    try:
        confidence = max(0.0, min(1.0, float(cria_llm_result.get("confidence", 0.0) or 0.0)))
    except (TypeError, ValueError):
        confidence = 0.0
    if bool(cria_llm_result.get("abstain")) or not selected_qid or selected_qid not in candidate_by_qid:
        merged.update(
            {
                "selected_qid": None,
                "selected_label": None,
                "confidence": confidence,
                "abstain": True,
                "resolution_mode": "cria_llm_abstained",
                "reason_codes": ["cria_llm_abstained"],
            }
        )
        return merged

    chosen = candidate_by_qid[selected_qid]
    deterministic_qid = deterministic_decision.get("selected_qid")
    mode = "cria_llm_confirmed" if selected_qid == deterministic_qid else "cria_llm_override"
    merged.update(
        {
            "selected_qid": selected_qid,
            "selected_label": chosen.get("label"),
            "confidence": round(confidence, 4),
            "abstain": False,
            "resolution_mode": mode,
            "reason_codes": [mode],
        }
    )
    return merged


def merge_shepherd_decision(
    *,
    deterministic_decision: dict[str, object],
    shepherd_result: dict[str, object] | None,
    reranked_candidates: list[dict[str, object]],
    trigger: SemanticTrigger,
) -> dict[str, object]:
    if not shepherd_result:
        merged = dict(deterministic_decision)
        merged["resolution_mode"] = "deterministic_only"
        return merged

    merged = dict(deterministic_decision)
    merged["shepherd_result"] = shepherd_result

    candidate_by_qid = {
        str(item.get("qid")): item
        for item in reranked_candidates
        if isinstance(item.get("qid"), str)
    }
    shepherd_qid = shepherd_result.get("selected_qid")
    if not isinstance(shepherd_qid, str):
        shepherd_qid = ""
    shepherd_abstain = bool(shepherd_result.get("abstain"))

    deterministic_qid = deterministic_decision.get("selected_qid")
    if not isinstance(deterministic_qid, str):
        deterministic_qid = ""

    try:
        deterministic_confidence = float(deterministic_decision.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        deterministic_confidence = 0.0
    try:
        deterministic_margin = float(deterministic_decision.get("margin", 0.0) or 0.0)
    except (TypeError, ValueError):
        deterministic_margin = 0.0
    try:
        shepherd_confidence = float(shepherd_result.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        shepherd_confidence = 0.0

    trigger_reasons = set(trigger.reason_codes)
    hard_ambiguity_reasons = {
        "deterministic_abstain",
        "low_margin",
        "same_label_cluster",
        "same_type_cluster",
        "authority_conflict",
        "low_authority_top_candidate",
        "derivative_or_unsupported_candidate",
        "list_or_disambiguation_candidate",
        "cleaner_label_competitor",
        "underspecified_label_competitor",
        "close_competitor",
    }
    hard_ambiguity = bool(trigger_reasons & hard_ambiguity_reasons)

    if shepherd_abstain or not shepherd_qid or shepherd_qid not in candidate_by_qid:
        if bool(deterministic_decision.get("abstain")) or (
            shepherd_abstain and shepherd_confidence >= 0.72 and deterministic_confidence < 0.78
        ):
            merged["abstain"] = True
            merged["resolution_mode"] = "shepherd_abstained"
        else:
            merged["resolution_mode"] = "shepherd_kept_deterministic"
        return merged

    if shepherd_qid == deterministic_qid:
        merged["shepherd_agreement"] = True
        merged["resolution_mode"] = "shepherd_confirmed"
        merged["confidence"] = round(max(deterministic_confidence, shepherd_confidence), 4)
        if (
            (not bool(deterministic_decision.get("abstain")) and shepherd_confidence >= 0.7 and deterministic_confidence >= 0.68)
            or (bool(deterministic_decision.get("abstain")) and shepherd_confidence >= 0.78)
        ):
            merged["abstain"] = False
        return merged

    if bool(deterministic_decision.get("abstain")) and shepherd_confidence >= 0.7:
        chosen = candidate_by_qid[shepherd_qid]
        merged.update(
            {
                "selected_qid": shepherd_qid,
                "selected_label": chosen.get("label"),
                "confidence": round(shepherd_confidence, 4),
                "abstain": False,
                "resolution_mode": "shepherd_override_after_abstain",
                "reason_codes": ["shepherd_override_after_abstain", *list(trigger.reason_codes)],
            }
        )
        return merged

    if hard_ambiguity and shepherd_confidence >= 0.78:
        chosen = candidate_by_qid[shepherd_qid]
        merged.update(
            {
                "selected_qid": shepherd_qid,
                "selected_label": chosen.get("label"),
                "confidence": round(max(shepherd_confidence, min(0.94, deterministic_confidence + 0.02)), 4),
                "abstain": False,
                "resolution_mode": "shepherd_override_ambiguity",
                "reason_codes": ["shepherd_override_ambiguity", *list(trigger.reason_codes)],
            }
        )
        return merged

    if deterministic_margin < 1.5 and shepherd_confidence >= 0.82:
        chosen = candidate_by_qid[shepherd_qid]
        merged.update(
            {
                "selected_qid": shepherd_qid,
                "selected_label": chosen.get("label"),
                "confidence": round(shepherd_confidence, 4),
                "abstain": False,
                "resolution_mode": "shepherd_override_low_margin",
                "reason_codes": ["shepherd_override_low_margin", *list(trigger.reason_codes)],
            }
        )
        return merged

    merged["resolution_mode"] = "shepherd_kept_deterministic"
    return merged
