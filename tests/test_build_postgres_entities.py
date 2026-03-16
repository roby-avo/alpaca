from __future__ import annotations

import unittest

from src.build_postgres_entities import (
    _build_entity_parse_context,
    _build_entity_context_string,
    _extract_cross_refs,
    _pick_primary_label,
    _resolve_sample_cache_qids,
    extract_entity_triples,
    infer_item_category,
    transform_entity_to_record,
)
from src.common import extract_multilingual_payload


def _statement_for_item_qid(
    numeric_id: int,
    *,
    rank: str = "normal",
) -> dict[str, object]:
    return {
        "rank": rank,
        "mainsnak": {
            "snaktype": "value",
            "datavalue": {
                "value": {
                    "entity-type": "item",
                    "numeric-id": int(numeric_id),
                }
            },
        }
    }


class BuildPostgresEntitiesCategoryTests(unittest.TestCase):
    class _StubSampleStore:
        def __init__(self, qids: list[str]) -> None:
            self.qids = list(qids)

        def list_sample_entity_ids(self, *, limit: int) -> list[str]:
            return self.qids[:limit]

    def test_primary_label_prefers_english_then_mul(self) -> None:
        self.assertEqual(
            _pick_primary_label({"it": "Roma", "mul": "Rome"}),
            "Rome",
        )
        self.assertEqual(
            _pick_primary_label({"it": "Roma", "en": "Rome", "mul": "Rome (mul)"}),
            "Rome",
        )

    def test_cross_refs_fallback_to_non_english_wikipedia(self) -> None:
        entity = {
            "sitelinks": {
                "itwiki": {"title": "Roma"},
            }
        }
        self.assertEqual(
            _extract_cross_refs(entity, preferred_language="mul"),
            {
                "wikipedia": "https://it.wikipedia.org/wiki/Roma",
                "dbpedia": "https://it.dbpedia.org/resource/Roma",
            },
        )

    def test_context_string_combines_related_labels_labels_and_aliases_without_duplicates(self) -> None:
        self.assertEqual(
            _build_entity_context_string(
                related_labels=["capital city", "Italy"],
            ),
            "capital city; Italy",
        )

    def test_transform_entity_keeps_english_description_and_human_occupations_in_types(self) -> None:
        entity = {
            "id": "Q42",
            "type": "item",
            "labels": {
                "en": {"language": "en", "value": "Douglas Adams"},
            },
            "descriptions": {
                "en": {"language": "en", "value": "English writer and humorist"},
                "it": {"language": "it", "value": "scrittore inglese"},
            },
            "claims": {
                "P31": [_statement_for_item_qid(5)],
                "P106": [_statement_for_item_qid(36_180)],
            },
        }

        record = transform_entity_to_record(
            entity,
            language_allowlist=("en",),
            max_aliases_per_language=8,
            disable_ner_classifier=True,
        )

        assert record is not None
        self.assertEqual(record.description, "English writer and humorist")
        self.assertEqual(record.types, ["Q5", "Q36180"])

    def test_transform_entity_description_is_null_without_english(self) -> None:
        entity = {
            "id": "Q1",
            "type": "item",
            "labels": {
                "mul": {"language": "mul", "value": "Universe"},
            },
            "descriptions": {
                "it": {"language": "it", "value": "universo"},
            },
        }

        record = transform_entity_to_record(
            entity,
            language_allowlist=("en",),
            max_aliases_per_language=8,
            disable_ner_classifier=True,
        )

        assert record is not None
        self.assertIsNone(record.description)

    def test_extract_multilingual_payload_removes_aliases_duplicated_by_labels(self) -> None:
        payload = extract_multilingual_payload(
            {
                "labels": {
                    "en": {"language": "en", "value": "Rome"},
                    "it": {"language": "it", "value": "Roma"},
                },
                "aliases": {
                    "en": [
                        {"language": "en", "value": "Rome"},
                        {"language": "en", "value": "Rome city"},
                    ],
                    "it": [
                        {"language": "it", "value": "Roma"},
                        {"language": "it", "value": "Roma capitale"},
                    ],
                },
            }
        )

        self.assertEqual(payload["labels"], {"en": "Rome", "it": "Roma"})
        self.assertEqual(payload["aliases"], {"en": ["Rome city"], "it": ["Roma capitale"]})

    def test_extract_entity_triples_keeps_subject_predicate_object_edges(self) -> None:
        entity = {
            "id": "Q42",
            "type": "item",
            "claims": {
                "P31": [_statement_for_item_qid(5)],
                "P106": [_statement_for_item_qid(36_180)],
            },
        }

        triples = extract_entity_triples(entity)

        self.assertEqual(
            [(triple.subject_qid, triple.predicate_pid, triple.object_qid) for triple in triples],
            [("Q42", "P106", "Q36180"), ("Q42", "P31", "Q5")],
        )

    def test_extract_entity_triples_prunes_to_diverse_informative_edges(self) -> None:
        entity = {
            "id": "Q42",
            "type": "item",
            "claims": {
                "P31": [_statement_for_item_qid(5, rank="preferred")],
                "P106": [
                    _statement_for_item_qid(36_180, rank="preferred"),
                    _statement_for_item_qid(49_670, rank="normal"),
                    _statement_for_item_qid(33_999, rank="deprecated"),
                ],
                "P166": [
                    _statement_for_item_qid(1),
                    _statement_for_item_qid(2),
                    _statement_for_item_qid(3),
                ],
            },
        }

        triples = extract_entity_triples(
            entity,
            max_triples=4,
            max_triples_per_predicate=2,
        )

        self.assertEqual(
            [(triple.predicate_pid, triple.object_qid) for triple in triples],
            [
                ("P106", "Q36180"),
                ("P31", "Q5"),
                ("P166", "Q1"),
                ("P106", "Q49670"),
            ],
        )

    def test_extract_entity_triples_prefers_person_disambiguators_over_generic_instance_of(self) -> None:
        entity = {
            "id": "Q42",
            "type": "item",
            "claims": {
                "P31": [_statement_for_item_qid(5)],
                "P106": [_statement_for_item_qid(36_180)],
                "P27": [_statement_for_item_qid(145)],
                "P39": [_statement_for_item_qid(11_691)],
                "P166": [_statement_for_item_qid(37922)],
            },
        }

        triples = extract_entity_triples(
            entity,
            max_triples=4,
            max_triples_per_predicate=1,
        )

        self.assertEqual(
            [(triple.predicate_pid, triple.object_qid) for triple in triples],
            [
                ("P106", "Q36180"),
                ("P27", "Q145"),
                ("P39", "Q11691"),
                ("P31", "Q5"),
            ],
        )

    def test_extract_entity_triples_uses_subject_kind_to_prioritize_location_context(self) -> None:
        entity = {
            "id": "Q84",
            "type": "item",
            "labels": {
                "en": {"language": "en", "value": "London"},
            },
            "descriptions": {
                "en": {"language": "en", "value": "capital city in England"},
            },
            "claims": {
                "P31": [_statement_for_item_qid(515)],
                "P17": [_statement_for_item_qid(145)],
                "P131": [_statement_for_item_qid(21)],
                "P361": [_statement_for_item_qid(22)],
                "P166": [_statement_for_item_qid(37922)],
            },
        }

        parse_context = _build_entity_parse_context(
            entity,
            language_allowlist=("en",),
            max_aliases_per_language=8,
            disable_ner_classifier=False,
        )

        self.assertIsNotNone(parse_context)
        triples = extract_entity_triples(
            entity,
            max_triples=4,
            max_triples_per_predicate=1,
            parse_context=parse_context,
        )

        self.assertEqual(
            [(triple.predicate_pid, triple.object_qid) for triple in triples],
            [
                ("P17", "Q145"),
                ("P131", "Q21"),
                ("P361", "Q22"),
                ("P31", "Q515"),
            ],
        )

    def test_extract_entity_triples_can_disable_pruning(self) -> None:
        entity = {
            "id": "Q42",
            "type": "item",
            "claims": {
                "P106": [
                    _statement_for_item_qid(36_180),
                    _statement_for_item_qid(49_670),
                    _statement_for_item_qid(33_999),
                ],
                "P166": [
                    _statement_for_item_qid(1),
                    _statement_for_item_qid(2),
                    _statement_for_item_qid(3),
                ],
            },
        }

        triples = extract_entity_triples(
            entity,
            max_triples=0,
            max_triples_per_predicate=0,
        )

        self.assertEqual(
            [(triple.predicate_pid, triple.object_qid) for triple in triples],
            [
                ("P106", "Q33999"),
                ("P166", "Q1"),
                ("P106", "Q36180"),
                ("P166", "Q2"),
                ("P106", "Q49670"),
                ("P166", "Q3"),
            ],
        )

    def test_resolve_sample_cache_qids_uses_cached_count_and_limit(self) -> None:
        qids = _resolve_sample_cache_qids(
            self._StubSampleStore(["Q1", "Q2", "Q3"]),
            sample_cache_ids=None,
            sample_cache_ids_file=None,
            sample_cache_count=3,
            limit=2,
        )

        self.assertEqual(qids, ["Q1", "Q2"])

    def test_property_is_predicate(self) -> None:
        self.assertEqual(infer_item_category({"id": "P31", "type": "property"}), "PREDICATE")

    def test_disambiguation_detected_from_instance_of(self) -> None:
        entity = {
            "id": "Q999",
            "type": "item",
            "claims": {
                "P31": [_statement_for_item_qid(4_167_410)],
            },
        }
        self.assertEqual(infer_item_category(entity), "DISAMBIGUATION")

    def test_type_detected_from_subclass_of(self) -> None:
        entity = {
            "id": "Q123",
            "type": "item",
            "claims": {
                "P279": [_statement_for_item_qid(35120)],
            },
        }
        self.assertEqual(infer_item_category(entity), "TYPE")

    def test_regular_item_defaults_to_entity(self) -> None:
        entity = {"id": "Q42", "type": "item", "claims": {}}
        self.assertEqual(infer_item_category(entity), "ENTITY")

    def test_lexeme_keeps_separate_category(self) -> None:
        self.assertEqual(infer_item_category({"id": "L1", "type": "lexeme"}), "LEXEME")


if __name__ == "__main__":
    unittest.main()
