"""
Part 3 of Clustering: creating relationship edges through kinship extraction.
'my aunt Maria' is a fact: it provides information like RELATED_TO(interviewee,
Maria, detail='aunt'). We use this in surrogate generation later. 
Right now we just have entities: Entity E0 (Interviewee), 
Entity E1 (Maria, "she", "my aunt") but we need to mark relations between them.
"""


from __future__ import annotations
import re
from .models import Edge, Entity, Relation


# kinship word: gender it implies for the TARGET person, or None
KINSHIP_GENDER = {
    "aunt": "F", "uncle": "M", "mother": "F", "father": "M",
    "mom": "F", "dad": "M", "grandmother": "F", "grandfather": "M",
    "grandma": "F", "grandpa": "M", "sister": "F", "brother": "M",
    "wife": "F", "husband": "M", "daughter": "F", "son": "M",
    "niece": "F", "nephew": "M", "cousin": None,
}
KIN_WORDS = "|".join(KINSHIP_GENDER)  # "aunt|uncle|mother..."


"""
Pattern 1: "my aunt Maria" --> edge from INTERVIEWEE to the named person
#if you call this, it returns a Match object
Regex: my (aunt|uncle|mother) ([A-Z][a-z]+) --> Use on this text -->
my aunt Maria
   |      |
   |      +------ group(2)
   |
   +------------- group(1)
"""
_PAT_MY = re.compile(rf"\bmy ({KIN_WORDS})[,\s]+([A-Z][a-z]+)")


"""
Pattern 2: "his brother John", "her sister Denise"
Edge from an ANTECEDENT (nearest preceding person mention) to the named
person. The pronoun tells us whose relative it is. The pattern is below:
\b(his|her|their) ({KIN_WORDS})[,\s]+([A-Z][a-z]+)
   ^-------------^  ^-----------^       ^---------^
      Group 1         Group 2            Group 3
"""
_PAT_PRONOUN = re.compile(rf"\b(his|her|their) ({KIN_WORDS})[,\s]+([A-Z][a-z]+)")


"""Which entity owns a mention containing the range [start, end)?"""
def _entity_at(entities: list[Entity], char_pos_start: int, char_pos_end: int):
    for e in entities:
        for m in e.mentions:
            if m.start < char_pos_end and char_pos_start < m.end:
                return e
    return None


"""Which PERSON entity was mentioned most recently before this position? (char_pos)"""
def _entity_before(entities: list[Entity], char_pos: int, exclude_id: str):
    best, best_end = None, -1
    for e in entities:
        if e.category != "PERSON" or e.entity_id == exclude_id:
            continue
        for m in e.mentions:
            if m.end <= char_pos and m.end > best_end:
                best, best_end = e, m.end
    return best


def extract_kinship(
    transcript: str,
    person_entities: list[Entity],
    interviewee: Entity,  # a special entity
) -> list[Edge]:
    edges: list[Edge] = []

    for match in _PAT_MY.finditer(transcript):  # eg. "my aunt Maria"
        kin_word, _name = match.group(1), match.group(2)  # "aunt", "Maria"
        # retrieve / match to existing entity
        target = _entity_at(person_entities, match.start(2), match.end(2))
        if target is None:
            continue        # detector never flagged this name; file it as a miss
        edges.append(
            Edge(
                source=interviewee.entity_id,
                target=target.entity_id,
                relation=Relation.RELATED_TO,
                detail=kin_word,
                evidence=match.group(0),
            )
        )
        gender = KINSHIP_GENDER[kin_word]
        if gender and not target.attributes.get("gender"):
            target.attributes["gender"] = gender

    for match in _PAT_PRONOUN.finditer(transcript):
        kin_word, _name = match.group(2), match.group(3)
        target = _entity_at(person_entities, match.start(3), match.end(3))
        if target is None:
            continue
        anchor = _entity_before(person_entities, match.start(), target.entity_id)
        if anchor is None:
            continue
        edges.append(
            Edge(
                source=anchor.entity_id,
                target=target.entity_id,
                relation=Relation.RELATED_TO,
                detail=kin_word,
                evidence=match.group(0),
            )
        )
        gender = KINSHIP_GENDER[kin_word]
        if gender and not target.attributes.get("gender"):
            target.attributes["gender"] = gender

    return edges