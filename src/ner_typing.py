from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .common import normalize_text, tokenize


@dataclass(frozen=True, slots=True)
class FineTypeRule:
    coarse: str
    fine: str
    token_clues: frozenset[str]
    phrase_clues: tuple[str, ...] = ()
    min_score: int = 1


# Deterministic, lightweight lexical rules for NER-ish typing.
# These are intentionally simple and fast: token/phrase matching only.
FINE_TYPE_RULES: tuple[FineTypeRule, ...] = (
    FineTypeRule(
        coarse="PERSON",
        fine="HUMAN",
        token_clues=frozenset(
            {
                "person",
                "actor",
                "actress",
                "singer",
                "musician",
                "politician",
                "writer",
                "author",
                "athlete",
                "footballer",
                "scientist",
                "artist",
                "director",
                "poet",
                "philosopher",
                "journalist",
                "engineer",
                "doctor",
                "composer",
                "president",
                "founder",
                "professor",
            }
        ),
        phrase_clues=("human being", "prime minister", "head of state", "president of"),
    ),
    FineTypeRule(
        coarse="PERSON",
        fine="FICTIONAL_CHARACTER",
        token_clues=frozenset({"fictional", "character", "superhero", "villain", "protagonist"}),
        phrase_clues=("fictional character",),
        min_score=2,
    ),
    FineTypeRule(
        coarse="ORGANIZATION",
        fine="COMPANY",
        token_clues=frozenset(
            {
                "company",
                "corporation",
                "business",
                "manufacturer",
                "enterprise",
                "startup",
                "firm",
                "multinational",
            }
        ),
    ),
    FineTypeRule(
        coarse="ORGANIZATION",
        fine="NONPROFIT_ORG",
        token_clues=frozenset({"foundation", "charitable", "nonprofit", "non-profit", "ngo"}),
        phrase_clues=("charitable organization", "non-profit organization"),
    ),
    FineTypeRule(
        coarse="ORGANIZATION",
        fine="GOVERNMENT_ORG",
        token_clues=frozenset(
            {
                "government",
                "ministry",
                "department",
                "agency",
                "parliament",
                "senate",
                "council",
                "municipality",
            }
        ),
    ),
    FineTypeRule(
        coarse="ORGANIZATION",
        fine="EDUCATIONAL_ORG",
        token_clues=frozenset({"university", "college", "school", "institute", "academy"}),
    ),
    FineTypeRule(
        coarse="ORGANIZATION",
        fine="SPORTS_TEAM",
        token_clues=frozenset(
            {"team", "fc", "athletic", "basketball", "baseball", "soccer", "hockey"}
        ),
        phrase_clues=("football club",),
    ),
    FineTypeRule(
        coarse="LOCATION",
        fine="COUNTRY",
        token_clues=frozenset(
            {
                "country",
                "nation",
                "republic",
                "kingdom",
                "sovereign",
            }
        ),
        phrase_clues=("sovereign state", "independent state", "country in"),
    ),
    FineTypeRule(
        coarse="LOCATION",
        fine="CITY",
        token_clues=frozenset(
            {
                "city",
                "town",
                "municipality",
                "capital",
                "village",
                "metropolis",
                "megacity",
                "commune",
                "arrondissement",
                "borough",
                "suburb",
                "settlement",
                "cidade",
                "ciudad",
                "stadt",
                "comune",
                "municipio",
            }
        ),
        phrase_clues=(
            "city in",
            "town in",
            "village in",
            "capital of",
            "county seat",
            "census-designated place",
            "global city",
            "national capital",
            "primate city",
            "largest city",
        ),
    ),
    FineTypeRule(
        coarse="LOCATION",
        fine="REGION",
        token_clues=frozenset({"region", "province", "district", "county", "territory", "continent"}),
        phrase_clues=(
            "state of the united states",
            "state in the united states",
            "federal state",
            "autonomous region",
        ),
    ),
    FineTypeRule(
        coarse="LOCATION",
        fine="LANDMARK",
        token_clues=frozenset(
            {
                "ocean",
                "sea",
                "gulf",
                "bay",
                "strait",
                "mountain",
                "river",
                "lake",
                "island",
                "airport",
                "station",
                "bridge",
                "building",
                "monument",
                "desert",
                "valley",
                "volcano",
            }
        ),
    ),
    FineTypeRule(
        coarse="LOCATION",
        fine="CELESTIAL_BODY",
        token_clues=frozenset({"planet", "moon", "star", "galaxy", "asteroid", "comet", "universe"}),
        phrase_clues=("solar system", "celestial body"),
    ),
    FineTypeRule(
        coarse="EVENT",
        fine="CONFLICT",
        token_clues=frozenset({"war", "battle", "revolution", "uprising", "campaign"}),
        phrase_clues=("armed conflict", "military conflict", "civil war"),
        min_score=2,
    ),
    FineTypeRule(
        coarse="EVENT",
        fine="SPORT_EVENT",
        token_clues=frozenset({"tournament", "championship", "olympics", "cup", "season"}),
        min_score=2,
    ),
    FineTypeRule(
        coarse="EVENT",
        fine="EVENT_GENERIC",
        token_clues=frozenset({"event", "festival", "conference", "election", "summit"}),
        min_score=2,
    ),
    FineTypeRule(
        coarse="WORK",
        fine="FILM",
        token_clues=frozenset({"film", "movie", "documentary", "cinema"}),
    ),
    FineTypeRule(
        coarse="WORK",
        fine="BOOK",
        token_clues=frozenset({"book", "novel", "poem", "literature"}),
    ),
    FineTypeRule(
        coarse="WORK",
        fine="MUSIC_WORK",
        token_clues=frozenset({"song", "album", "opera", "symphony", "anthem"}),
    ),
    FineTypeRule(
        coarse="WORK",
        fine="SOFTWARE",
        token_clues=frozenset({"software", "application", "app", "program", "library", "framework"}),
        phrase_clues=("operating system",),
    ),
    FineTypeRule(
        coarse="WORK",
        fine="INTERNET_MEME",
        token_clues=frozenset({"meme"}),
        phrase_clues=("internet meme",),
    ),
    FineTypeRule(
        coarse="PRODUCT",
        fine="DEVICE",
        token_clues=frozenset(
            {
                "device",
                "smartphone",
                "phone",
                "laptop",
                "hardware",
                "vehicle",
                "aircraft",
                "airliner",
                "automobile",
                "printer",
                "train",
            }
        ),
    ),
    FineTypeRule(
        coarse="PRODUCT",
        fine="MEDICATION",
        token_clues=frozenset({"drug", "medicine", "vaccine", "antibiotic", "treatment"}),
    ),
    FineTypeRule(
        coarse="PRODUCT",
        fine="FOOD_BEVERAGE",
        token_clues=frozenset(
            {
                "beverage",
                "drink",
                "food",
                "dish",
                "cuisine",
                "snack",
                "meal",
                "alcoholic",
                "non-alcoholic",
                "nonalcoholic",
            }
        ),
        phrase_clues=("alcoholic beverage", "non-alcoholic beverage"),
    ),
    FineTypeRule(
        coarse="PRODUCT",
        fine="PRODUCT_GENERIC",
        token_clues=frozenset({"product", "brand", "model"}),
    ),
    FineTypeRule(
        coarse="CONCEPT",
        fine="LANGUAGE",
        token_clues=frozenset({"language", "dialect"}),
    ),
    FineTypeRule(
        coarse="CONCEPT",
        fine="LAW",
        token_clues=frozenset({"law", "statute", "treaty", "regulation", "directive", "constitution", "code"}),
        phrase_clues=("law of", "act of", "treaty of", "regulation of"),
        min_score=2,
    ),
    FineTypeRule(
        coarse="CONCEPT",
        fine="SCIENTIFIC_THEORY",
        token_clues=frozenset({"theory", "principle", "equation", "theorem", "hypothesis"}),
    ),
    FineTypeRule(
        coarse="CONCEPT",
        fine="BIOLOGICAL_TAXON",
        token_clues=frozenset({"species", "genus", "taxon", "subspecies", "clade", "mammal"}),
    ),
    FineTypeRule(
        coarse="CONCEPT",
        fine="ANATOMY",
        token_clues=frozenset({"organ", "anatomy", "anatomical", "muscle", "bone", "artery", "vein"}),
        phrase_clues=("part of the body", "part of body", "sexual organ", "anatomical structure"),
        min_score=2,
    ),
)


def _has_english_text(
    labels: Mapping[str, str],
    aliases: Mapping[str, Sequence[str]],
    descriptions: Mapping[str, str],
) -> bool:
    if isinstance(descriptions.get("en"), str):
        return True
    if isinstance(labels.get("en"), str):
        return True
    aliases_en = aliases.get("en")
    if not isinstance(aliases_en, Sequence) or isinstance(aliases_en, (str, bytes, bytearray)):
        return False
    return any(isinstance(value, str) and normalize_text(value) for value in aliases_en)


def _iter_text_values(
    labels: Mapping[str, str],
    aliases: Mapping[str, Sequence[str]],
    descriptions: Mapping[str, str],
) -> list[str]:
    use_english_only = _has_english_text(labels, aliases, descriptions)

    values: list[str] = []
    seen: set[str] = set()

    description_languages = ("en",) if use_english_only else sorted(descriptions)
    for language in description_languages:
        if language not in descriptions:
            continue
        candidate = normalize_text(descriptions[language])
        if candidate and candidate not in seen:
            seen.add(candidate)
            values.append(candidate)

    label_languages = ("en",) if use_english_only else sorted(labels)
    for language in label_languages:
        if language not in labels:
            continue
        candidate = normalize_text(labels[language])
        if candidate and candidate not in seen:
            seen.add(candidate)
            values.append(candidate)

    alias_languages = ("en",) if use_english_only else sorted(aliases)
    for language in alias_languages:
        if language not in aliases:
            continue
        for alias in aliases[language]:
            candidate = normalize_text(alias)
            if candidate and candidate not in seen:
                seen.add(candidate)
                values.append(candidate)

    return values


def infer_ner_types(
    entity_id: str,
    labels: Mapping[str, str],
    aliases: Mapping[str, Sequence[str]],
    descriptions: Mapping[str, str],
) -> tuple[list[str], list[str], str]:
    # Properties are deterministic relation-like symbols for retrieval typing.
    if entity_id.startswith("P"):
        return ["RELATION"], ["PROPERTY"], "lexical_v1"

    text_values = _iter_text_values(labels, aliases, descriptions)
    if not text_values:
        return ["MISC"], ["ENTITY"], "lexical_v1"

    normalized_text = "\n".join(value.casefold() for value in text_values)
    token_set = set(tokenize(normalized_text))

    scored_rules: list[tuple[int, FineTypeRule]] = []
    for rule in FINE_TYPE_RULES:
        score = 0
        score += sum(1 for token in rule.token_clues if token in token_set)
        score += sum(2 for phrase in rule.phrase_clues if phrase and phrase in normalized_text)

        if score >= rule.min_score:
            scored_rules.append((score, rule))

    if not scored_rules:
        return ["MISC"], ["ENTITY"], "lexical_v1"

    scored_rules.sort(key=lambda item: (-item[0], item[1].fine))
    top_score = scored_rules[0][0]
    selected: list[tuple[int, FineTypeRule]] = [
        item for item in scored_rules if item[0] == top_score
    ][:2]

    coarse_scores: dict[str, int] = {}
    fine_types: list[str] = []
    seen_fine: set[str] = set()

    for score, rule in selected:
        coarse_scores[rule.coarse] = coarse_scores.get(rule.coarse, 0) + score
        if rule.fine not in seen_fine:
            seen_fine.add(rule.fine)
            fine_types.append(rule.fine)

    sorted_coarse = sorted(coarse_scores.items(), key=lambda item: (-item[1], item[0]))
    coarse_types = [coarse for coarse, _ in sorted_coarse[:2]]

    if not coarse_types:
        coarse_types = ["MISC"]
    if not fine_types:
        fine_types = ["ENTITY"]

    return coarse_types, fine_types, "lexical_v1"
