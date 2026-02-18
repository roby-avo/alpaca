from __future__ import annotations

import unittest

from src.ner_typing import infer_ner_types


class NerTypingTests(unittest.TestCase):
    def test_property_id_maps_to_relation_property(self) -> None:
        coarse, fine, source = infer_ner_types(
            entity_id="P31",
            labels={"en": "instance of"},
            aliases={},
            descriptions={"en": "that class of which this subject is a particular example"},
        )
        self.assertEqual(coarse, ["RELATION"])
        self.assertEqual(fine, ["PROPERTY"])
        self.assertEqual(source, "lexical_v1")

    def test_person_description_maps_to_human(self) -> None:
        coarse, fine, _ = infer_ner_types(
            entity_id="Q42",
            labels={"en": "Douglas Adams"},
            aliases={},
            descriptions={"en": "English writer and author"},
        )
        self.assertIn("PERSON", coarse)
        self.assertIn("HUMAN", fine)

    def test_company_description_maps_to_company(self) -> None:
        coarse, fine, _ = infer_ner_types(
            entity_id="Q312",
            labels={"en": "Apple"},
            aliases={},
            descriptions={"en": "American technology company"},
        )
        self.assertIn("ORGANIZATION", coarse)
        self.assertIn("COMPANY", fine)

    def test_paris_like_description_maps_to_location_city(self) -> None:
        coarse, fine, _ = infer_ner_types(
            entity_id="Q90",
            labels={"en": "Paris"},
            aliases={},
            descriptions={"en": "capital and most populous city of France"},
        )
        self.assertIn("LOCATION", coarse)
        self.assertIn("CITY", fine)

    def test_president_maps_to_person_not_country(self) -> None:
        coarse, fine, _ = infer_ner_types(
            entity_id="Q76",
            labels={"en": "Barack Obama"},
            aliases={},
            descriptions={"en": "president of the United States from 2009 to 2017"},
        )
        self.assertIn("PERSON", coarse)
        self.assertIn("HUMAN", fine)
        self.assertNotIn("COUNTRY", fine)

    def test_us_state_maps_to_region_not_country(self) -> None:
        coarse, fine, _ = infer_ner_types(
            entity_id="Q99",
            labels={"en": "California"},
            aliases={},
            descriptions={"en": "state of the United States of America"},
        )
        self.assertIn("LOCATION", coarse)
        self.assertIn("REGION", fine)
        self.assertNotIn("COUNTRY", fine)

    def test_ocean_maps_to_landmark_not_conflict(self) -> None:
        coarse, fine, _ = infer_ner_types(
            entity_id="Q97",
            labels={"en": "Atlantic Ocean"},
            aliases={},
            descriptions={"en": "ocean between Europe, Africa and the Americas"},
        )
        self.assertIn("LOCATION", coarse)
        self.assertIn("LANDMARK", fine)
        self.assertNotIn("CONFLICT", fine)

    def test_charitable_foundation_maps_to_nonprofit_org(self) -> None:
        coarse, fine, _ = infer_ner_types(
            entity_id="Q180",
            labels={"en": "Wikimedia Foundation"},
            aliases={"en": ["Wikimedia Foundation, Inc."]},
            descriptions={"en": "American charitable organization"},
        )
        self.assertIn("ORGANIZATION", coarse)
        self.assertIn("NONPROFIT_ORG", fine)

    def test_continent_maps_to_region(self) -> None:
        coarse, fine, _ = infer_ner_types(
            entity_id="Q46",
            labels={"en": "Europe"},
            aliases={},
            descriptions={"en": "terrestrial continent located in north-western Eurasia"},
        )
        self.assertIn("LOCATION", coarse)
        self.assertIn("REGION", fine)

    def test_founder_description_maps_to_person(self) -> None:
        coarse, fine, _ = infer_ner_types(
            entity_id="Q181",
            labels={"en": "Jimmy Wales"},
            aliases={},
            descriptions={"en": "co-founder of Wikipedia (born 1966)"},
        )
        self.assertIn("PERSON", coarse)
        self.assertIn("HUMAN", fine)

    def test_internet_meme_maps_to_work_meme(self) -> None:
        coarse, fine, _ = infer_ner_types(
            entity_id="Q149",
            labels={"en": "Nyan Cat"},
            aliases={},
            descriptions={"en": "2011 Internet meme"},
        )
        self.assertIn("WORK", coarse)
        self.assertIn("INTERNET_MEME", fine)

    def test_mammal_maps_to_biological_taxon(self) -> None:
        coarse, fine, _ = infer_ner_types(
            entity_id="Q146",
            labels={"en": "cat"},
            aliases={},
            descriptions={"en": "small domesticated carnivorous mammal"},
        )
        self.assertIn("CONCEPT", coarse)
        self.assertIn("BIOLOGICAL_TAXON", fine)

    def test_english_preference_reduces_cross_language_noise(self) -> None:
        coarse, fine, _ = infer_ner_types(
            entity_id="Q2",
            labels={"en": "Earth"},
            aliases={},
            descriptions={
                "en": "third planet from the Sun in the Solar System",
                "es": "planeta habitado por humanos",
            },
        )
        self.assertNotIn("PERSON", coarse)
        self.assertNotIn("HUMAN", fine)
        self.assertIn("LOCATION", coarse)
        self.assertIn("CELESTIAL_BODY", fine)

    def test_fallback_when_no_clues(self) -> None:
        coarse, fine, _ = infer_ner_types(
            entity_id="Q999999",
            labels={"en": "Xyzz"},
            aliases={},
            descriptions={"en": "xkcdqv"},
        )
        self.assertEqual(coarse, ["MISC"])
        self.assertEqual(fine, ["ENTITY"])


if __name__ == "__main__":
    unittest.main()
