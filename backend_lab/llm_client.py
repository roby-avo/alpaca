from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

from .cache import JsonDiskCache
from .config import BackendConfig
from .preprocess import merge_schema_dicts, normalize_preprocessing_schema, parse_llm_json_response
from .table_profile import normalize_table_profile


class LlmConfigurationError(RuntimeError):
    pass


class LlmRequestError(RuntimeError):
    pass


def _extract_message_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts).strip()
    return str(content)


def build_preprocessing_messages(seed_schema: dict[str, Any]) -> list[dict[str, str]]:
    backend_taxonomy = {
        "item_category": ["ENTITY", "TYPE", "PREDICATE", "DISAMBIGUATION", "LEXEME", "FORM", "SENSE", "MEDIAINFO", "OTHER"],
        "coarse_type": ["PERSON", "ORGANIZATION", "LOCATION", "WORK", "PRODUCT", "EVENT", "BIOLOGICAL_TAXON", "THING", "RELATION", "MISC"],
        "fine_type_examples": [
            "HUMAN",
            "COMPANY",
            "COUNTRY",
            "CITY",
            "REGION",
            "LANDMARK",
            "CELESTIAL_BODY",
            "FILM",
            "BOOK",
            "MUSIC_WORK",
            "SOFTWARE",
            "DEVICE",
            "FOOD_BEVERAGE",
            "SPORT_EVENT",
            "BIOLOGICAL_TAXON",
            "MISC",
        ],
    }
    system_prompt = (
        "You are a table entity-linking preprocessing model. "
        "Return JSON only. "
        "Refine the provided deterministic seed schema into a richer preprocessing schema. "
        "Reason at cell level, column level, and row level. "
        "Use confidence and weight values in the range [0,1]. "
        "Only promote signals to hard_filters when confidence is strong. "
        "Keep output compatible with the provided schema keys. "
        "Use the backend taxonomy, not your own ontology. "
        "For structural category use item_category values like ENTITY or TYPE, not semantic labels like GeographicFeature. "
        "For semantic type constraints use backend coarse_type/fine_type labels only. "
        "For example, bodies of water like the Caspian Sea should usually map to coarse_type LOCATION and fine_type LANDMARK for this backend. "
        f"Backend taxonomy reference: {backend_taxonomy}. "
        "For cell_hypothesis.entity_category, coarse_type, fine_type, and domain, always use objects of the form "
        "{\"value\": string, \"confidence\": number, \"weight\": number, \"source\": string}. "
        "For column_profile coarse_type_distribution and fine_type_distribution, always return lists of those same objects. "
        "For retrieval_plan.hard_filters, always return arrays of strings only, for example "
        "{\"item_category\": [\"ENTITY\"], \"coarse_type\": [\"LOCATION\"], \"fine_type\": [\"COUNTRY\"]}. "
        "Do not place confidence objects inside hard_filters. "
        "Put lower-confidence evidence into retrieval_plan.soft_context_terms instead. "
        "Preserve and use table_profile for table-aware reasoning. "
        "Populate cell_hypothesis.entity_hypotheses with 1-3 ranked hypotheses. "
        "For weak mentions in generic or no-header tables, include an alternate person/location/work hypothesis rather than overcommitting."
    )
    user_prompt = (
        "Refine this seed schema for cell-level entity linking.\n"
        "Do not add prose outside JSON.\n"
        "Fill or improve these sections when justified: cell_hypothesis, column_profile, "
        "row_constraints, retrieval_plan.\n"
        "Prefer a sparse retrieval_plan: only a few strong hard filters and a short list of high-value soft_context_terms.\n"
        "Good hard_filters are stable type/category constraints. "
        "Good soft_context_terms are place names, row entities, and short semantic cues.\n\n"
        f"{seed_schema}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_json_repair_messages(raw_text: str) -> list[dict[str, str]]:
    system_prompt = (
        "You repair malformed JSON. "
        "Return valid JSON only. "
        "Do not add commentary. "
        "Preserve the original structure and values as closely as possible."
    )
    user_prompt = (
        "Repair this malformed JSON so it becomes valid JSON.\n"
        "Return only the repaired JSON object.\n\n"
        f"{raw_text}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_table_profile_messages(seed_profile: dict[str, Any]) -> list[dict[str, str]]:
    system_prompt = (
        "You are a table schema induction model for entity linking. "
        "Return JSON only. "
        "You receive a deterministic table-profile seed built from sampled rows and per-column statistics. "
        "Refine it into a normalized table profile for downstream cell entity linking. "
        "Infer the table semantic family, ranked column roles, row template, table hypotheses, confidence, and concise evidence notes. "
        "Prefer generic reusable roles such as PERSON_NAME_OR_ALIAS, BIRTH_DATE, BIRTH_CITY, BIRTH_REGION, COUNTRY, "
        "GEOGRAPHIC_FEATURE_NAME, BODY_OF_WATER_NAME, DESCRIPTION_TEXT, AREA_MEASURE, DEPTH_MEASURE, VOLUME_MEASURE, LENGTH_MEASURE, TEXT_ATTRIBUTE, UNKNOWN. "
        "If the table looks like a biography table with stage names or aliases in the first column and birth attributes in later columns, say so explicitly. "
        "Output keys: table_semantic_family, confidence, column_roles, row_template, table_hypotheses, evidence_notes. "
        "column_roles must be a mapping from column id strings to ranked lists of {role, confidence, source}. "
        "Keep the response sparse and consistent with the seed."
    )
    user_prompt = (
        "Refine this deterministic table-profile seed for table-aware entity linking.\n"
        "Do not output prose outside JSON.\n\n"
        f"{seed_profile}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_semantic_adjudication_messages(payload: dict[str, Any]) -> list[dict[str, str]]:
    system_prompt = (
        "You are a semantic adjudication model for table cell entity linking. "
        "Return JSON only. "
        "Choose at most one candidate qid from the provided candidates or abstain. "
        "You must not invent qids. "
        "Use row context, column context, candidate descriptions, and candidate context strings to resolve ambiguity. "
        "Prefer broad, well-supported entities over incidental derivative entities when the table semantics indicate the main referent. "
        "If the evidence is not strong enough, abstain. "
        "Output this schema exactly: "
        "{\"selected_qid\": string|null, \"confidence\": number, \"abstain\": boolean, "
        "\"reason\": string, \"supporting_signals\": [string]}. "
        "confidence must be in [0,1]."
    )
    user_prompt = (
        "Resolve this ambiguous table cell by choosing the best candidate qid or abstaining.\n"
        "Do not output prose outside JSON.\n\n"
        f"{payload}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_cria_shepherd_messages(payload: dict[str, Any]) -> list[dict[str, str]]:
    system_prompt = (
        "You are CRIA-Shepherd, an LLM adjudicator for table cell entity linking. "
        "Return JSON only. "
        "You receive a mention, table context, row context, deterministic CRIA decision, "
        "and a shortlist of candidates already retrieved from the backend. "
        "Choose exactly one candidate qid from the provided candidates, or abstain if the evidence is genuinely insufficient. "
        "You must not invent qids, labels, facts, or candidates. "
        "Do not assume the deterministic rank-1 candidate is correct; use it as one signal among others. "
        "Resolve same-label duplicate entities by preferring the candidate whose description and context are best entailed by the row/table. "
        "Prefer canonical, broad, well-supported primary entities over incidental geographic stubs, set-index/list pages, "
        "disambiguation-like pages, or derivative entities when the table asks for the main referent. "
        "Use numeric fields such as prior, final_score, and reranked_rank only as supporting evidence, not as the sole reason. "
        "Output this schema exactly: "
        "{\"selected_qid\": string|null, \"confidence\": number, \"abstain\": boolean, "
        "\"reason\": string, \"supporting_signals\": [string], "
        "\"rejected_candidates\": [{\"qid\": string, \"reason\": string}]}. "
        "confidence must be in [0,1]."
    )
    user_prompt = (
        "Adjudicate this CRIA candidate shortlist for one table cell.\n"
        "Return only the JSON object.\n\n"
        f"{payload}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_cria_llm_rank_messages(payload: dict[str, Any]) -> list[dict[str, str]]:
    system_prompt = (
        "You are CRIA-LLM, a complete LLM candidate ranker for table cell entity linking. "
        "Return JSON only. "
        "You receive a mention, table context, row context, preprocessing/search metadata, "
        "a deterministic CRIA decision, and a candidate pack retrieved from the backend. "
        "Rank the provided candidates from best to worst for the target cell. "
        "You must not invent qids, labels, facts, or candidates. "
        "You may select exactly one candidate qid from the provided candidates or abstain if none is supported. "
        "Use the row context, table profile, search payload, candidate descriptions, candidate context strings, "
        "and numeric ranking features. "
        "Rank independently from the deterministic baseline; deterministic rank and final_score are diagnostic signals, not labels. "
        "A high deterministic score must not outrank explicit row/table incompatibility. "
        "If the row gives a country, region, date, role, measurement, or description, prefer candidates whose description/context supports that evidence. "
        "Demote candidates whose description/context conflicts with row evidence, even if their lexical match or final_score is high. "
        "Resolve same-label duplicates by preferring the candidate whose description/context is entailed by the row/table. "
        "Ranking scores should reflect semantic fit to the table cell, not copied numeric final_score values. "
        "Output this schema exactly: "
        "{\"selected_qid\": string|null, \"confidence\": number, \"abstain\": boolean, "
        "\"ranking\": [{\"qid\": string, \"rank\": number, \"score\": number, \"reason\": string}], "
        "\"reason\": string, \"supporting_signals\": [string]}. "
        "The ranking list should include every candidate qid supplied in the candidate pack exactly once. "
        "confidence and ranking scores must be in [0,1]."
    )
    user_prompt = (
        "Rank this CRIA candidate pack and select the best entity for the table cell.\n"
        "Return only the JSON object.\n\n"
        f"{payload}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


@dataclass(slots=True)
class OpenAICompatibleLlmClient:
    api_key: str
    base_url: str
    model: str
    timeout_seconds: float = 90.0
    timeout_multiplier: float = 1.8
    timeout_max_seconds: float = 300.0
    max_retries: int = 2
    retry_backoff_seconds: float = 2.0
    retry_max_sleep_seconds: float = 20.0
    temperature: float = 0.0
    cache_enabled: bool = True
    cache_dir: Path | None = None

    @classmethod
    def from_config(cls, config: BackendConfig, *, model: str | None = None) -> "OpenAICompatibleLlmClient":
        if not config.llm_api_key:
            raise LlmConfigurationError(
                "No LLM API key configured. Set ALPACA_LLM_API_KEY or DEEPINFRA_TOKEN."
            )
        return cls(
            api_key=config.llm_api_key,
            base_url=config.llm_base_url,
            model=model or config.llm_model,
            timeout_seconds=config.llm_timeout_seconds,
            timeout_multiplier=config.llm_timeout_multiplier,
            timeout_max_seconds=config.llm_timeout_max_seconds,
            max_retries=config.llm_max_retries,
            retry_backoff_seconds=config.llm_retry_backoff_seconds,
            retry_max_sleep_seconds=config.llm_retry_max_sleep_seconds,
            temperature=config.llm_temperature,
            cache_enabled=config.llm_cache_enabled,
            cache_dir=config.llm_cache_dir,
        )

    def preprocess(self, seed_schema: dict[str, Any]) -> dict[str, Any]:
        cache_key = {
            "kind": "preprocess",
            "model": self.model,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "seed_schema": seed_schema,
        }
        cached = self._cache_get("preprocess", cache_key)
        if cached is not None:
            return cached

        prompt_seed_schema = _compact_preprocessing_seed(seed_schema)
        response, raw_text = self._chat_json(build_preprocessing_messages(prompt_seed_schema))

        try:
            parsed = parse_llm_json_response(raw_text)
        except Exception as exc:
            try:
                _, repaired_text = self._chat_json(build_json_repair_messages(raw_text))
                parsed = parse_llm_json_response(repaired_text)
                raw_text = repaired_text
            except Exception as repair_exc:
                raise LlmRequestError(f"Could not parse LLM JSON output: {exc}") from repair_exc

        merged = normalize_preprocessing_schema(merge_schema_dicts(seed_schema, parsed))
        usage = getattr(response, "usage", None)
        usage_dict = {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }
        result = {
            "model": self.model,
            "base_url": self.base_url,
            "usage": usage_dict,
            "preprocessing_schema": merged,
            "raw_response_text": raw_text,
        }
        return self._cache_set("preprocess", cache_key, result)

    def induce_table_profile(self, seed_profile: dict[str, Any]) -> dict[str, Any]:
        cache_key = {
            "kind": "table_profile",
            "model": self.model,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "seed_profile": seed_profile,
        }
        cached = self._cache_get("table_profile", cache_key)
        if cached is not None:
            return cached

        response, raw_text = self._chat_json(build_table_profile_messages(seed_profile))
        try:
            parsed = parse_llm_json_response(raw_text)
        except Exception as exc:
            raise LlmRequestError(f"Could not parse LLM table-profile JSON output: {exc}") from exc

        merged = merge_schema_dicts(seed_profile, parsed)
        normalized = normalize_table_profile(
            merged,
            dataset_id=str(seed_profile.get("dataset_id", "") or ""),
            table_id=str(seed_profile.get("table_id", "") or ""),
        )
        usage = getattr(response, "usage", None)
        usage_dict = {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }
        result = {
            "model": self.model,
            "base_url": self.base_url,
            "usage": usage_dict,
            "table_profile": normalized,
            "raw_response_text": raw_text,
        }
        return self._cache_set("table_profile", cache_key, result)

    def semantic_resolve(self, payload: dict[str, Any]) -> dict[str, Any]:
        cache_key = {
            "kind": "semantic_resolve",
            "model": self.model,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "payload": payload,
        }
        cached = self._cache_get("semantic_resolve", cache_key)
        if cached is not None:
            return cached

        response, raw_text = self._chat_json(build_semantic_adjudication_messages(payload))

        try:
            parsed = parse_llm_json_response(raw_text)
        except Exception as exc:
            raise LlmRequestError(f"Could not parse semantic LLM JSON output: {exc}") from exc

        selected_qid = parsed.get("selected_qid")
        if selected_qid is not None and not isinstance(selected_qid, str):
            selected_qid = None
        abstain = bool(parsed.get("abstain"))
        confidence = parsed.get("confidence", 0.0)
        try:
            confidence_value = max(0.0, min(1.0, float(confidence)))
        except (TypeError, ValueError):
            confidence_value = 0.0
        supporting_signals = parsed.get("supporting_signals", [])
        if not isinstance(supporting_signals, list):
            supporting_signals = []
        usage = getattr(response, "usage", None)
        usage_dict = {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }
        result = {
            "model": self.model,
            "base_url": self.base_url,
            "usage": usage_dict,
            "selected_qid": selected_qid,
            "confidence": confidence_value,
            "abstain": abstain,
            "reason": parsed.get("reason", ""),
            "supporting_signals": [value for value in supporting_signals if isinstance(value, str)],
            "raw_response_text": raw_text,
        }
        return self._cache_set("semantic_resolve", cache_key, result)

    def cria_shepherd_resolve(self, payload: dict[str, Any]) -> dict[str, Any]:
        cache_key = {
            "kind": "cria_shepherd_resolve",
            "model": self.model,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "payload": payload,
        }
        cached = self._cache_get("cria_shepherd_resolve", cache_key)
        if cached is not None:
            return cached

        response, raw_text = self._chat_json(build_cria_shepherd_messages(payload))

        try:
            parsed = parse_llm_json_response(raw_text)
        except Exception as exc:
            try:
                _, repaired_text = self._chat_json(build_json_repair_messages(raw_text))
                parsed = parse_llm_json_response(repaired_text)
                raw_text = repaired_text
            except Exception as repair_exc:
                raise LlmRequestError(f"Could not parse CRIA-Shepherd JSON output: {exc}") from repair_exc

        selected_qid = parsed.get("selected_qid")
        if selected_qid is not None and not isinstance(selected_qid, str):
            selected_qid = None
        abstain = bool(parsed.get("abstain"))
        confidence = parsed.get("confidence", 0.0)
        try:
            confidence_value = max(0.0, min(1.0, float(confidence)))
        except (TypeError, ValueError):
            confidence_value = 0.0
        supporting_signals = parsed.get("supporting_signals", [])
        if not isinstance(supporting_signals, list):
            supporting_signals = []
        rejected_candidates = parsed.get("rejected_candidates", [])
        if not isinstance(rejected_candidates, list):
            rejected_candidates = []
        normalized_rejections: list[dict[str, str]] = []
        for item in rejected_candidates:
            if not isinstance(item, dict):
                continue
            qid = item.get("qid")
            reason = item.get("reason")
            if isinstance(qid, str):
                normalized_rejections.append(
                    {
                        "qid": qid,
                        "reason": reason if isinstance(reason, str) else "",
                    }
                )
        usage = getattr(response, "usage", None)
        usage_dict = {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }
        result = {
            "model": self.model,
            "base_url": self.base_url,
            "usage": usage_dict,
            "selected_qid": selected_qid,
            "confidence": confidence_value,
            "abstain": abstain,
            "reason": parsed.get("reason", ""),
            "supporting_signals": [value for value in supporting_signals if isinstance(value, str)],
            "rejected_candidates": normalized_rejections,
            "raw_response_text": raw_text,
        }
        return self._cache_set("cria_shepherd_resolve", cache_key, result)

    def cria_llm_rank(self, payload: dict[str, Any]) -> dict[str, Any]:
        cache_key = {
            "kind": "cria_llm_rank",
            "model": self.model,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "payload": payload,
        }
        cached = self._cache_get("cria_llm_rank", cache_key)
        if cached is not None:
            return cached

        response, raw_text = self._chat_json(build_cria_llm_rank_messages(payload))

        try:
            parsed = parse_llm_json_response(raw_text)
        except Exception as exc:
            try:
                _, repaired_text = self._chat_json(build_json_repair_messages(raw_text))
                parsed = parse_llm_json_response(repaired_text)
                raw_text = repaired_text
            except Exception as repair_exc:
                raise LlmRequestError(f"Could not parse CRIA-LLM ranking JSON output: {exc}") from repair_exc

        selected_qid = parsed.get("selected_qid")
        if selected_qid is not None and not isinstance(selected_qid, str):
            selected_qid = None
        abstain = bool(parsed.get("abstain"))
        try:
            confidence_value = max(0.0, min(1.0, float(parsed.get("confidence", 0.0) or 0.0)))
        except (TypeError, ValueError):
            confidence_value = 0.0

        provided_qids = {
            str(item.get("qid"))
            for item in payload.get("candidates", [])
            if isinstance(item, dict) and isinstance(item.get("qid"), str)
        }
        ranking = parsed.get("ranking", [])
        if not isinstance(ranking, list):
            ranking = []
        normalized_ranking: list[dict[str, object]] = []
        seen_qids: set[str] = set()
        for item in ranking:
            if not isinstance(item, dict):
                continue
            qid = item.get("qid")
            if not isinstance(qid, str) or qid not in provided_qids or qid in seen_qids:
                continue
            seen_qids.add(qid)
            try:
                rank_value = int(item.get("rank", len(normalized_ranking) + 1) or len(normalized_ranking) + 1)
            except (TypeError, ValueError):
                rank_value = len(normalized_ranking) + 1
            try:
                score_value = max(0.0, min(1.0, float(item.get("score", 0.0) or 0.0)))
            except (TypeError, ValueError):
                score_value = 0.0
            normalized_ranking.append(
                {
                    "qid": qid,
                    "rank": rank_value,
                    "score": score_value,
                    "reason": item.get("reason", "") if isinstance(item.get("reason"), str) else "",
                }
            )

        normalized_ranking.sort(key=lambda item: int(item.get("rank", 10**9) or 10**9))
        supporting_signals = parsed.get("supporting_signals", [])
        if not isinstance(supporting_signals, list):
            supporting_signals = []
        usage = getattr(response, "usage", None)
        usage_dict = {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }
        result = {
            "model": self.model,
            "base_url": self.base_url,
            "usage": usage_dict,
            "selected_qid": selected_qid,
            "confidence": confidence_value,
            "abstain": abstain,
            "ranking": normalized_ranking,
            "reason": parsed.get("reason", ""),
            "supporting_signals": [value for value in supporting_signals if isinstance(value, str)],
            "raw_response_text": raw_text,
        }
        return self._cache_set("cria_llm_rank", cache_key, result)

    def _chat_json(self, messages: list[dict[str, str]]) -> tuple[Any, str]:
        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:
            raise LlmConfigurationError(
                "The openai package is not installed. Install backend_lab/requirements.txt first."
            ) from exc

        current_timeout = max(1.0, float(self.timeout_seconds))
        max_attempts = max(1, int(self.max_retries) + 1)
        last_exc: Exception | None = None
        response: Any = None

        for attempt in range(max_attempts):
            client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=current_timeout,
            )
            create_kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "response_format": {"type": "json_object"},
            }
            if self.temperature is not None:
                create_kwargs["temperature"] = self.temperature

            try:
                response = client.chat.completions.create(**create_kwargs)
                break
            except Exception as exc:
                last_exc = exc
                if "response_format" in create_kwargs:
                    create_kwargs.pop("response_format", None)
                    try:
                        response = client.chat.completions.create(**create_kwargs)
                        break
                    except Exception as fallback_exc:
                        last_exc = fallback_exc
                if not _should_retry_request_error(last_exc) or attempt >= max_attempts - 1:
                    raise LlmRequestError(f"LLM request failed: {last_exc}") from last_exc
                sleep_seconds = min(
                    self.retry_max_sleep_seconds,
                    self.retry_backoff_seconds * (2 ** attempt),
                )
                time.sleep(max(0.0, sleep_seconds))
                current_timeout = min(
                    self.timeout_max_seconds,
                    max(current_timeout + 1.0, current_timeout * self.timeout_multiplier),
                )

        if response is None and last_exc is not None:
            raise LlmRequestError(f"LLM request failed: {last_exc}") from last_exc

        if not getattr(response, "choices", None):
            raise LlmRequestError("LLM response did not contain any choices.")
        raw_text = _extract_message_text(response.choices[0].message)
        if not raw_text.strip():
            raise LlmRequestError("LLM response content was empty.")
        return response, raw_text

    def _cache_get(self, namespace: str, key_payload: dict[str, Any]) -> dict[str, Any] | None:
        if not self.cache_enabled or self.cache_dir is None:
            return None
        return JsonDiskCache(self.cache_dir).get(namespace, key_payload)

    def _cache_set(self, namespace: str, key_payload: dict[str, Any], value: dict[str, Any]) -> dict[str, Any]:
        if not self.cache_enabled or self.cache_dir is None:
            return value
        return JsonDiskCache(self.cache_dir).set(namespace, key_payload, value)


def _compact_preprocessing_seed(seed_schema: dict[str, Any]) -> dict[str, Any]:
    compact = merge_schema_dicts({}, seed_schema)
    table_profile = compact.get("table_profile", {})
    if isinstance(table_profile, dict):
        compact["table_profile"] = {
            "dataset_id": table_profile.get("dataset_id"),
            "table_id": table_profile.get("table_id"),
            "table_semantic_family": table_profile.get("table_semantic_family"),
            "confidence": table_profile.get("confidence"),
            "column_roles": table_profile.get("column_roles", {}),
            "row_template": table_profile.get("row_template", []),
            "table_hypotheses": table_profile.get("table_hypotheses", []),
            "evidence_notes": table_profile.get("evidence_notes", [])[:4],
        }

    context = compact.get("context", {})
    if isinstance(context, dict):
        trimmed_context = dict(context)
        for key in ("row_values", "other_row_values", "sampled_column_values", "mention_context"):
            raw_values = trimmed_context.get(key, [])
            if isinstance(raw_values, list):
                trimmed_values: list[str] = []
                for item in raw_values[:8]:
                    if not isinstance(item, str):
                        continue
                    value = item.strip()
                    if len(value) > 220:
                        value = value[:220].rstrip() + "..."
                    trimmed_values.append(value)
                trimmed_context[key] = trimmed_values
        compact["context"] = trimmed_context

    retrieval_plan = compact.get("retrieval_plan", {})
    if isinstance(retrieval_plan, dict):
        trimmed_plan = dict(retrieval_plan)
        soft_terms = trimmed_plan.get("soft_context_terms", [])
        if isinstance(soft_terms, list):
            trimmed_plan["soft_context_terms"] = soft_terms[:8]
        query_variants = trimmed_plan.get("query_variants", [])
        if isinstance(query_variants, list):
            trimmed_plan["query_variants"] = query_variants[:5]
        compact["retrieval_plan"] = trimmed_plan

    cell_hypothesis = compact.get("cell_hypothesis", {})
    if isinstance(cell_hypothesis, dict):
        trimmed_hypothesis = dict(cell_hypothesis)
        entity_hypotheses = trimmed_hypothesis.get("entity_hypotheses", [])
        if isinstance(entity_hypotheses, list):
            trimmed_hypothesis["entity_hypotheses"] = entity_hypotheses[:3]
        compact["cell_hypothesis"] = trimmed_hypothesis

    return compact


def _should_retry_request_error(exc: Exception) -> bool:
    message = str(exc).casefold()
    retry_tokens = (
        "timed out",
        "timeout",
        "connection",
        "temporarily unavailable",
        "rate limit",
        "429",
        "500",
        "502",
        "503",
        "504",
        "overloaded",
    )
    return any(token in message for token in retry_tokens)
