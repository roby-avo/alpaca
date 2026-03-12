from __future__ import annotations

import unittest

from src.build_postgres_entities import (
    _build_entity_context_string,
    _extract_cross_refs,
    _pick_primary_label,
    extract_entity_triples,
    infer_item_category,
    transform_entity_to_record,
)


def _statement_for_item_qid(numeric_id: int) -> dict[str, object]:
    return {
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
