from __future__ import annotations

import unittest

from src.build_postgres_entities import infer_item_category


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
