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


# Signals that a name matching PUBLIC_FIGURES is actually a PRIVATE individual who
# merely shares the name (so it must still be redacted): a first-person possessive
# directly on the name ("my Beyonce"), a naming construction that introduces a
# person ("her name is Beyonce", "a friend named Beyonce"), or an explicit
# "no relation to the famous one" aside. Deliberately NARROW: third-person
# ATTRIBUTIVE use of a famous name ("his Elvis impression", "my dad admired Obama")
# is NOT a personal signal -- the old broad window over-redacted those. Kinship and
# professional context are stronger signals and checked separately.
_POSS_NAME = re.compile(r"\b(?:my|our)\s+$", re.I)
_NAMING = re.compile(r"\b(?:name\s+(?:is|was)|named)\s+$", re.I)
_NOT_FAMOUS = re.compile(r"^\W{0,3}(?:no relation|not the (?:famous|real|actual|celebrity))", re.I)


PROFESSIONAL_CONTEXT = re.compile(
    r"\b(my|our|the)\s+(caseworker|case worker|social worker|doctor|dr\.?|"
    r"nurse|therapist|counselor|counsellor|psychiatrist|psychologist|"
    r"physician|surgeon|pediatrician|dentist|midwife|teacher|professor|"
    r"instructor|tutor|advisor|adviser|mentor|principal|dean|coach|boss|"
    r"manager|supervisor|landlord|lawyer|attorney|pastor|priest|rabbi|imam|"
    r"chaplain|parole officer|probation officer|po|sponsor|babysitter|"
    r"nanny|caregiver)\b", re.I)


# ------------------------- interviewee gender -------------------------
# First-person self-description of the SPEAKER'S OWN gender. We deliberately use
# ONLY gendered nouns the interviewee applies to THEMSELVES ("I'm a mother",
# "call me Grandpa") -- never relational terms about other people ("my husband"),
# which would encode an assumption about the speaker's own gender/orientation. The
# trigger is anchored to an explicit first-person "I ..." (or a "call me <kin>"),
# so third-person predicates ("he was known as a good man") never leak in.
_IV_TRIG = (r"I['’]?m|I am|I was|I['’]ve been|I have been|I became|I['’]d become|"
            r"I grew up|I ended up|I['’]ll be")
# article / possessive introducing the predicate noun ("I'm a mother", "I'm her mom")
_IV_SLOT = r"(?:a|an|the|her|his|their|your|our|my)\s+"
# optional adjectives between the article and the noun ("I was the only girl")
_IV_MOD = (r"(?:(?:only|oldest|youngest|eldest|middle|single|young|old|little|proud|"
           r"new|first|second|working|widowed|divorced|stay[\s-]at[\s-]home|good|"
           r"poor|country|former)\s+){0,3}")
_IV_NOUN_F = (r"woman|girl|lady|gal|mother|mom|mommy|mum|mama|momma|grandmother|"
              r"grandma|granny|nana|grandmom|gramma|daughter|sister|widow|housewife")
_IV_NOUN_M = (r"man|boy|guy|gentleman|fella|fellow|father|dad|daddy|papa|grandfather|"
              r"grandpa|granddad|gramps|son|brother|widower")
_IV_SELF_F = re.compile(r"\b(?:" + _IV_TRIG + r")\s+" + _IV_SLOT + _IV_MOD +
                        r"(?:" + _IV_NOUN_F + r")\b", re.I)
_IV_SELF_M = re.compile(r"\b(?:" + _IV_TRIG + r")\s+" + _IV_SLOT + _IV_MOD +
                        r"(?:" + _IV_NOUN_M + r")\b", re.I)
# "(the kids) call me Grandma" / "everybody calls me Pop"
_IV_CALLME_F = re.compile(r"\bcall(?:ed|s)?\s+me\s+(?:their\s+)?(?:mom|mommy|mama|"
                          r"momma|mother|grandma|granny|nana|meemaw|gramma)\b", re.I)
_IV_CALLME_M = re.compile(r"\bcall(?:ed|s)?\s+me\s+(?:their\s+)?(?:dad|daddy|papa|"
                          r"pop|pops|papaw|pawpaw|grandpa|gramps|granddad)\b", re.I)


def infer_interviewee_gender(transcript: str, interviewee: Entity) -> None:
    """Set the interviewee's OWN gender (rule layer) from first-person self-
    description. Conflicting cues (both F and M matched) leave it unset and raise a
    review flag -- the LLM second line (extract_pass) then fills or confirms it.
    Never overwrites a gender already set."""
    if interviewee.attributes.get("gender"):
        return
    genders, evidence = set(), None
    for rx, g in ((_IV_SELF_F, "F"), (_IV_CALLME_F, "F"),
                  (_IV_SELF_M, "M"), (_IV_CALLME_M, "M")):
        m = rx.search(transcript)
        if m:
            genders.add(g)
            if evidence is None:
                evidence = m.group(0).strip()
    if len(genders) == 1:
        interviewee.attributes["gender"] = next(iter(genders))
        interviewee.attributes.setdefault("gender_evidence", evidence)
    elif len(genders) == 2:
        interviewee.flag_entity("conflicting first-person gender cues for the "
                                "interviewee; left unset for the LLM / a human to resolve")


def _personal_signal(transcript: str, ent, kin_ids: set) -> bool:
    """True when a name that matches the public-figure list is, in THIS transcript,
    a private individual (a relative, someone with a professional role in the
    interviewee's life, or a namesake introduced personally) -> must be redacted."""
    if ent.entity_id in kin_ids:                          # the interviewee's relative
        return True
    for m in ent.mentions:
        pre = transcript[max(0, m.start - 24):m.start]
        if _POSS_NAME.search(pre) or _NAMING.search(pre):
            return True
        if PROFESSIONAL_CONTEXT.search(transcript[max(0, m.start - 60):m.end + 60]):
            return True
        if _NOT_FAMOUS.match(transcript[m.end:m.end + 40]):
            return True
    return False


"""
Infer high level attributes about PERSON entities. Earlier, we've detected PERSON
entities, clustered mentions, and built RELATED_TO edges and etc. 
"""
def infer_person_attributes(
    transcript: str, person_entities: list[Entity], edges: list[Edge]
) -> None:
    kin_targets = {e.target for e in edges if e.relation == Relation.RELATED_TO}
    kin_sources = {e.source for e in edges if e.relation == Relation.RELATED_TO}
    kin_ids = kin_targets | kin_sources

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
            # Default for a listed famous name is to KEEP it (it reveals nothing
            # about the interviewee). Redact only when a personal signal says this
            # is a private namesake -- leaking a private person is the unacceptable
            # error; over-redacting a celebrity is the safe one.
            #
            # This keep is PROVISIONAL: the closed PUBLIC_FIGURES list is not the
            # sole authority. When the LLM layer is active, openworld_pass CO-SIGNS
            # -- it raises redaction back on unless the model also affirms this is a
            # public figure (see llm_layer/openworld.py). Rules-only behavior here is
            # unchanged.
            if not _personal_signal(transcript, ent, kin_ids):
                ent.subtype = "PUBLIC_FIGURE"
                attrs["replace"] = False
                continue
            ent.flag_entity(
                "name matches a public figure but is used personally "
                "(relative / professional / namesake); treated as private"
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