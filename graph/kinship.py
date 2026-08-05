"""
Part 3 of Clustering: creating relationship edges through kinship extraction.
'my aunt Maria' is a fact: it provides information like RELATED_TO(interviewee,
Maria, detail='aunt'). We use this in surrogate generation later.
Right now we just have entities: Entity E0 (Interviewee), or
Entity E1 (Maria, "she", "my aunt") but we need to mark relations between them.

We handle these surface forms:
  1. "my aunt Maria"                    -> interviewee RELATED_TO Maria
  2. "his brother John"                 -> antecedent RELATED_TO John
  3. "my cousin named Trey"             -> "... named/called X"
  4. "Maria, my aunt"                   -> appositive
  5. "Maria's brother John"             -> named possessor
  6. "my mom's sister Denise"           -> possessive chain
Each form tolerates optional descriptive modifiers ("my *older* sister Jen"),
multi-token names ("Maria Rodriguez"), and step/half/in-law/grand kin terms.
"""


from __future__ import annotations
import re
from .models import Edge, Entity, Relation


# Kinship word (lowercased, spaces/hyphens normalized) -> gender it implies for
# the TARGET person, or None when the term is gender-neutral. 
KINSHIP_GENDER = {
    # female
    "mother": "F", "mom": "F", "mommy": "F", "mum": "F", "mummy": "F",
    "mama": "F", "mamma": "F", "momma": "F", "ma": "F",
    "aunt": "F", "auntie": "F", "aunty": "F",
    "grandmother": "F", "grandma": "F", "grandmom": "F", "granny": "F",
    "nana": "F", "nanna": "F", "gramma": "F", "grammy": "F", "meemaw": "F",
    "sister": "F", "sis": "F",
    "wife": "F", "daughter": "F", "niece": "F", "granddaughter": "F",
    "stepmother": "F", "stepmom": "F", "stepsister": "F", "stepdaughter": "F",
    "half-sister": "F", "mother-in-law": "F", "sister-in-law": "F",
    "daughter-in-law": "F", "godmother": "F", "goddaughter": "F",
    "great-grandmother": "F", "fiancee": "F", "girlfriend": "F", "ex-wife": "F",
    # male 
    "father": "M", "dad": "M", "daddy": "M", "papa": "M", "poppa": "M",
    "pop": "M", "pops": "M", "pa": "M",
    "uncle": "M",
    "grandfather": "M", "grandpa": "M", "granddad": "M", "grandad": "M",
    "grandpop": "M", "gramps": "M", "papaw": "M", "pawpaw": "M", "pappy": "M",
    "brother": "M", "bro": "M",
    "husband": "M", "son": "M", "nephew": "M", "grandson": "M",
    "stepfather": "M", "stepdad": "M", "stepbrother": "M", "stepson": "M",
    "half-brother": "M", "father-in-law": "M", "brother-in-law": "M",
    "son-in-law": "M", "godfather": "M", "godson": "M",
    "great-grandfather": "M", "fiance": "M", "boyfriend": "M", "ex-husband": "M",
    # gender-neutral / unknown
    "cousin": None, "parent": None, "sibling": None, "partner": None,
    "spouse": None, "child": None, "kid": None, "grandchild": None,
    "grandkid": None, "grandparent": None, "godparent": None, "godchild": None,
    "twin": None, "relative": None, "in-law": None,
    "stepparent": None, "stepchild": None, "stepkid": None, "ex": None,
}


"""
Accounts for spacing/hyphen variants: 'in-law' vs 'in law', 'stepmom' 
vs 'step mom', 'grandma' vs 'grand ma'.
"""
def _flex(word: str) -> str:
    w = re.escape(word).replace(r"\-", r"[\s\-]")           # in-law hyphen/space
    w = re.sub(r"^(step|grand|great|god|half)",             # optional prefix gap
               lambda m: m.group(1) + r"[\s\-]?", w)
    return w


# Longest-first so "grandmother" is tried before "mother", "brother-in-law"
# before "brother", etc. 
_KIN_ALT = "|".join(_flex(w) for w in sorted(KINSHIP_GENDER, key=len, reverse=True))
KIN = rf"(?:{_KIN_ALT})"


# Optional descriptive modifiers between a possessive/article and the kin word,
# e.g. "my *older* sister", "her *late* husband".
_MOD = (r"(?:(?:older|elder|younger|little|big|baby|oldest|youngest|eldest|"
        r"only|middle|twin|maternal|paternal|biological|bio|adoptive|adopted|"
        r"foster|late|dear|beloved|first|second|third|current|former|ex)\s+)*")


# one name token: a capitalized word, optionally with internal apostrophes or
# hyphens ("Maria", "De'Andre", "O'Brien", "Mary-Jane"); deliberately excludes
# "." so a match can't run past a sentence boundary into the next name
_NAME_TOKEN = r"[A-Z](?:[a-z]+|(?=['’\-]))(?:['’\-][A-Za-z]+)*"
# a proper name with 1-3 such tokens ("Maria", "Maria Rodriguez")
_NAME = rf"{_NAME_TOKEN}(?:\s+{_NAME_TOKEN}){{0,2}}"
# name with no apostrophe, used as a possessor so we don't swallow the "'s"
_NAME_PLAIN = r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}"


_POSS = r"my|our"                     # first person -> interviewee
_PRON = r"his|her|their"              # third person -> nearest antecedent


"""Look up implied gender, tolerating spacing/hyphen variants."""
def _kin_gender(raw: str):
    key = re.sub(r"\s+", "-", raw.strip().lower())
    if key in KINSHIP_GENDER:
        return KINSHIP_GENDER[key]
    return KINSHIP_GENDER.get(key.replace("-", ""))


"""Which entity owns a mention overlapping the range [start, end)?"""
def _entity_at(entities: list[Entity], start: int, end: int):
    for e in entities:
        for m in e.mentions:
            if m.start < end and start < m.end:
                return e
    return None


"""Which PERSON entity was mentioned most recently before position `pos`?"""
def _entity_before(entities: list[Entity], pos: int, exclude_id: str | None = None):
    best, best_end = None, -1
    for e in entities:
        if e.category != "PERSON" or e.entity_id == exclude_id:
            continue
        for m in e.mentions:
            if m.end <= pos and m.end > best_end:
                best, best_end = e, m.end
    return best


def extract_kinship(
    transcript: str,
    person_entities: list[Entity],
    interviewee: Entity,  # a special entity
) -> list[Edge]:
    edges: list[Edge] = []
    seen: set[tuple] = set()  # (source_id, target_id) dedup across patterns


    """
    HELPER FUNCTION! Used for the six patterns that we handle below: this function
    creates a RELATED_TO edge (deduped) and infer the target's gender.
    """
    def add(source_id, target, detail, evidence, gender_word=None):
        if source_id is None or target is None:
            return
        if source_id == target.entity_id:
            return
        key = (source_id, target.entity_id)
        if key in seen:
            return
        seen.add(key)
        edges.append(Edge(
            source=source_id,
            target=target.entity_id,
            relation=Relation.RELATED_TO,
            detail=detail,
            evidence=evidence,
        ))
        gender = _kin_gender(gender_word or detail)
        if gender and not target.attributes.get("gender"):
            target.attributes["gender"] = gender


    # Pattern 6: "my mom's sister Denise" (possessive chain); runs first
    for m in re.finditer(
        rf"(?i:\b(?:{_POSS})\s+{_MOD}({KIN})['’]s\s+{_MOD}({KIN}))[\s,]+({_NAME})",
        transcript,
    ):
        target = _entity_at(person_entities, m.start(3), m.end(3))
        add(interviewee.entity_id, target,
            f"{m.group(1)}'s {m.group(2)}", m.group(0), gender_word=m.group(2))


    # Pattern 1: "my aunt Maria" -> interviewee RELATED_TO Maria
    for m in re.finditer(
        rf"(?i:\b(?:{_POSS})\s+{_MOD}({KIN}))[\s,]+({_NAME})", transcript
    ):
        target = _entity_at(person_entities, m.start(2), m.end(2))
        add(interviewee.entity_id, target, m.group(1), m.group(0))


    # Pattern 2: "his brother John" -> antecedent RELATED_TO John
    for m in re.finditer(
        rf"(?i:\b({_PRON})\s+{_MOD}({KIN}))[\s,]+({_NAME})", transcript
    ):
        target = _entity_at(person_entities, m.start(3), m.end(3))
        exclude = target.entity_id if target else None
        anchor = _entity_before(person_entities, m.start(), exclude_id=exclude)
        add(anchor.entity_id if anchor else None, target, m.group(2), m.group(0))


    # Pattern 3: "my cousin named Trey" / "an aunt called Maria"
    for m in re.finditer(
        rf"(?i:\b(?:({_POSS}|{_PRON})\s+)?{_MOD}({KIN})\s+(?:named|called)\s+)({_NAME})",
        transcript,
    ):
        poss = (m.group(1) or "").lower()
        target = _entity_at(person_entities, m.start(3), m.end(3))
        if poss in ("my", "our"):
            add(interviewee.entity_id, target, m.group(2), m.group(0))
        else:
            exclude = target.entity_id if target else None
            anchor = _entity_before(person_entities, m.start(), exclude_id=exclude)
            add(anchor.entity_id if anchor else None, target, m.group(2), m.group(0))


    # Pattern 4: "Maria, my aunt" / "Denise, his sister" (appositive)
    # STRICT closer: the appositive must END cleanly -- the kin word is followed by
    # punctuation, "and"/"who"/"whom", or end of text. This rejects the ambiguous
    # case where the kin word is actually the SUBJECT of the next clause
    # ("Lewis, my Papaw would say ...", which is NOT an appositive). Anything that
    # doesn't clear this bar produces NO rule edge and is left to the LLM relation
    # second line (extract_pass -> relation_verify), which reads the full context.
    for m in re.finditer(
        rf"({_NAME}),\s+(?i:(?:who\s+(?:is|was)\s+|who's\s+)?({_POSS}|{_PRON})\s+{_MOD}({KIN}))"
        rf"\b(?=\s*(?:[,.;:!?)\]\"'’”]|and\b|who\b|whom\b|$))",
        transcript,
    ):
        target = _entity_at(person_entities, m.start(1), m.end(1))
        poss = m.group(2).lower()
        if poss in ("my", "our"):
            add(interviewee.entity_id, target, m.group(3), m.group(0))
        else:
            exclude = target.entity_id if target else None
            anchor = _entity_before(person_entities, m.start(1), exclude_id=exclude)
            add(anchor.entity_id if anchor else None, target, m.group(3), m.group(0))


    # Pattern 5: "Maria's brother John" (named possessor)
    for m in re.finditer(
        rf"({_NAME_PLAIN})['’]s\s+(?i:{_MOD}({KIN}))[\s,]+({_NAME})", transcript
    ):
        source = _entity_at(person_entities, m.start(1), m.end(1))
        target = _entity_at(person_entities, m.start(3), m.end(3))
        add(source.entity_id if source else None, target, m.group(2), m.group(0))

    return edges