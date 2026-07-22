"""
Fill each entity's attribute dict for the surrogate generator later.
We accept None values if the value is unknown.
"""


from __future__ import annotations
import re
from .models import Edge, Entity, Relation
from .merge_strings import KINSHIP_AND_TITLES


# Names that should usually NOT be replaced (public figures give no PII about
# the interviewee); HARDCODED
PUBLIC_FIGURES = {"obama", "barack obama", "trump", "biden", "beyonce", "mlk"}


PROFESSIONAL_CONTEXT = re.compile(
    r"\b(my|our|the)\s+(caseworker|case worker|doctor|nurse|therapist|"
    r"counselor|teacher|professor|boss|manager|landlord|lawyer|attorney|"
    r"pastor|social worker|parole officer|po)\b", re.I)


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
            ent.subtype = "PUBLIC_FIGURE"
            attrs["replace"] = False        # public figures usually stay
            continue

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