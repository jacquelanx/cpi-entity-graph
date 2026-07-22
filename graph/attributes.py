"""
Fill each entity's attribute dict for the surrogate generator later.
We accept None values if the value is unknown.
"""


from __future__ import annotations
import re
from .models import Edge, Entity, Relation
from .merge_strings import KINSHIP_AND_TITLES


# Names that should usually NOT be replaced (public figures give no PII about
# the interviewee). Prefer full names / distinctive surnames over common names to
# avoid colliding with a private individual who shares the name.
PUBLIC_FIGURES = {
    # politicians / historical
    "obama", "barack obama", "michelle obama", "biden", "joe biden",
    "trump", "donald trump", "hillary clinton", "bill clinton",
    "george bush", "reagan", "nixon", "jfk", "kennedy",
    "mlk", "martin luther king", "malcolm x", "mandela", "nelson mandela",
    "putin", "gandhi",
    # entertainers
    "beyonce", "jay-z", "oprah", "oprah winfrey", "kanye", "drake",
    "rihanna", "madonna", "elvis", "michael jackson", "taylor swift",
    "kim kardashian",
    # athletes
    "lebron", "lebron james", "michael jordan", "kobe", "kobe bryant",
    "serena williams", "tom brady", "muhammad ali",
}


# Possessive / relational framing just before a name ("my Michelle", "our
# James") signals a PRIVATE individual, even if the name matches a public
# figure. Small trailing window before the mention.
_OWNED_BEFORE = re.compile(r"\b(my|our|his|her|their)\s+(?:\w+\s+){0,2}$", re.I)


PROFESSIONAL_CONTEXT = re.compile(
    r"\b(my|our|the)\s+(caseworker|case worker|social worker|doctor|dr\.?|"
    r"nurse|therapist|counselor|counsellor|psychiatrist|psychologist|"
    r"physician|surgeon|pediatrician|dentist|midwife|teacher|professor|"
    r"instructor|tutor|advisor|adviser|mentor|principal|dean|coach|boss|"
    r"manager|supervisor|landlord|lawyer|attorney|pastor|priest|rabbi|imam|"
    r"chaplain|parole officer|probation officer|po|sponsor|babysitter|"
    r"nanny|caregiver)\b", re.I)


"""
Infer high level attributes about PERSON entities. Earlier, we've detected PERSON
entities, clustered mentions, and built RELATED_TO edges and etc. 
"""
def infer_person_attributes(
    transcript: str, person_entities: list[Entity], edges: list[Edge]
) -> None:
    kin_targets = {e.target for e in edges if e.relation == Relation.RELATED_TO}
    kin_sources = {e.source for e in edges if e.relation == Relation.RELATED_TO}

    for ent in person_entities:
        attrs = ent.attributes

        # longest name form usually contains the most complete name
        longest = ent.sorted_mentions[0] if ent.sorted_mentions else ""

        # split that form into tokens while removing titles and kinship words such as
        # "Dr.", "Mr.", "Mom", "Aunt", etc
        tokens = [t for t in re.split(r"[\s,]+", longest)
                  if t and t.lower() not in KINSHIP_AND_TITLES]

        # assume first token is the given name and last token is the surname; middle names 
        # are ignored for surrogate generation
        if len(tokens) >= 2:
            attrs.setdefault("given_name", tokens[0])
            attrs.setdefault("surname", tokens[-1])
        # eg. only "maria"
        elif len(tokens) == 1:
            attrs.setdefault("given_name", tokens[0])
            attrs.setdefault("surname", None)

        # NOW WE DEAL WITH SPECIAL TYPES OF PEOPLE
        forms_lower = {f.lower() for f in ent.sorted_mentions}  # lowercase everything

        if forms_lower & PUBLIC_FIGURES:
            # if interviewee frames this name possessively ("my Michelle"), 
            # it's a private person who happens to share the name so we DON'T
            # replace it
            owned = any(
                _OWNED_BEFORE.search(transcript[max(0, m.start - 40):m.start])
                for m in ent.mentions
            )
            if not owned:
                ent.subtype = "PUBLIC_FIGURE"
                attrs["replace"] = False    # public figures usually stay
                continue
            ent.flag_entity(
                "name matches a public figure but is used possessively; "
                "treated as private"
            )

        if ent.entity_id in kin_targets or ent.entity_id in kin_sources:
            ent.subtype = ent.subtype or "FAMILY"
        else:
            # professional-context words within 60 chars of any mention?
            for m in ent.mentions:
                window = transcript[max(0, m.start - 60):m.end + 60]
                if PROFESSIONAL_CONTEXT.search(window):
                    ent.subtype = "PROFESSIONAL"
                    break
        attrs.setdefault("replace", True)

        # Kinship.py may have set the gender already; also pronouns could refine it
        # once coref runs. Leave None otherwise for gender-neutral substitutions.
        attrs.setdefault("gender", None)