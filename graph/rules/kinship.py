"""
Kinship extraction: turn family words in the text into RELATED_TO edges.

PURPOSE
    "my aunt Maria" is a fact worth keeping: it means
    RELATED_TO(interviewee, Maria, detail="aunt"). Clustering gives us the NODES
    -- E0 (the interviewee), E1 (Maria, "she", "my aunt") -- and this module adds
    the EDGES between them. Surrogate generation needs them so that a fake name
    for Maria stays consistent with her being the speaker's aunt. As a side
    effect it also fills in gender, which a kin word states directly ("aunt" is
    female).

FIT
    Called by `graph/pipeline.run_pipeline` right after clustering settles and
    BEFORE the attribute stages, because `attributes.infer_person_role` reads the
    kin edges this produces. Its `KINSHIP_GENDER` table and `_entity_at` helper
    are imported by `rules/aliases.py`, and the derived `KIN` regex is reused by
    `checks/ownership.py` and `checks/relation_evidence.py` -- so the kin
    vocabulary is defined once, here.

HOW
    A vocabulary table (`KINSHIP_GENDER`) is compiled into one big alternation
    regex (`KIN`), which is then dropped into six sentence patterns, each matching
    a different English way of naming a relative. Every pattern extracts a
    (source person, kin word, target name) triple and hands it to the local `add`
    helper, which creates the edge and infers gender. Patterns run in a fixed
    order -- the most specific first -- and `add` dedupes, so an earlier, more
    precise reading wins over a later, looser one.

    We handle these surface forms:
      1. "my aunt Maria"                -> interviewee RELATED_TO Maria
      2. "his brother John"             -> antecedent RELATED_TO John
      3. "my cousin named Trey"         -> "... named/called X"
      4. "Maria, my aunt"               -> appositive
      5. "Maria's brother John"         -> named possessor
      6. "my mom's sister Denise"       -> possessive chain

    Each form tolerates optional descriptive modifiers ("my *older* sister Jen"),
    multi-token names ("Maria Rodriguez"), and step/half/in-law/grand kin terms.

    A form that does NOT clear one of these patterns produces no rule edge at
    all. That is deliberate: the LLM relation second line can still propose it,
    and `checks/relation_evidence.py` verifies the proposal against the
    transcript. Abstaining is safe; a wrong family tie is not.
"""


from __future__ import annotations
import re
from ..models import Edge, Entity, Relation


# Kinship word (lowercased, spaces/hyphens normalized) -> gender it implies for
# the TARGET person, or None when the term is gender-neutral. 
KINSHIP_GENDER = {
    # female
    "mother": "F", "mom": "F", "mommy": "F", "mum": "F", "mummy": "F",
    "mama": "F", "mamma": "F", "momma": "F", "ma": "F",
    "aunt": "F", "auntie": "F", "aunty": "F",
    "grandmother": "F", "grandma": "F", "grandmom": "F", "granny": "F",
    "nana": "F", "nanna": "F", "gramma": "F", "grammy": "F", "meemaw": "F",
    # The Appalachian/Southern feminine grandparent terms. `mamaw` was absent
    # while its masculine counterpart `papaw` was present in every table, and this
    # is the table that builds the `KIN` regex -- so "my Mamaw Opal" produced no
    # RELATED_TO edge, no FAMILY subtype and no rule gender, while "my Papaw
    # Clarence" produced all three. That asymmetry was the single missing relation
    # in interview_002 (recall 6/7). `checks/names._EXTRA_KIN` already listed these
    # words, which meant the CHECKER knew a vocabulary the rules did not.
    "mamaw": "F", "mawmaw": "F", "memaw": "F", "mammaw": "F", "mimi": "F",
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


def _flex(word: str) -> str:
    """Turn one kin word into a regex tolerant of how people actually write it.

    English kin compounds are spelled inconsistently, and a table entry has to
    match all of them: "in-law" / "in law", "stepmom" / "step mom" / "step-mom",
    "grandma" / "grand ma".

    HOW: escape the word so no character is treated as regex syntax, then relax
    two things -- any hyphen becomes "space or hyphen", and a leading
    step/grand/great/god/half prefix may be followed by an optional space or
    hyphen. So "mother-in-law" compiles to a pattern also matching "mother in
    law".
    """
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


def _kin_gender(raw: str):
    """The gender a kin word implies, or None if the word is neutral or unknown.

    The inverse of `_flex`: whatever spelling the transcript used has to be mapped
    back onto one table key. Whitespace is collapsed to hyphens first, so "mother
    in law" becomes "mother-in-law"; if that misses, hyphens are removed
    altogether so "step mom" becomes "stepmom". Returns None both for genuinely
    neutral terms ("cousin") and for unrecognized words -- callers treat both as
    "no gender evidence".
    """
    key = re.sub(r"\s+", "-", raw.strip().lower())
    if key in KINSHIP_GENDER:
        return KINSHIP_GENDER[key]
    return KINSHIP_GENDER.get(key.replace("-", ""))


def _entity_at(entities: list[Entity], start: int, end: int):
    """The entity whose mention overlaps the character range `[start, end)`.

    How a regex capture group is turned into a graph node: the pattern matched
    "Maria" at some offsets, and this finds the entity clustering built for it.
    Overlap rather than exact equality, because the detector's span and the
    pattern's capture group need not align exactly. None means the matched text
    is not a person we know about, and the caller skips the edge.
    """
    for e in entities:
        for m in e.mentions:
            if m.start < end and start < m.end:
                return e
    return None


def _entity_before(entities: list[Entity], pos: int, exclude_id: str | None = None):
    """The PERSON entity mentioned most recently before offset `pos`.

    Resolves the antecedent of a third-person possessive: in "his brother John",
    whose brother? The answer is taken to be the last person named before the
    phrase, tracked by keeping the mention with the largest `end` that still
    falls at or before `pos`.

    `exclude_id` skips the TARGET of the relation being built, so "John" in "his
    brother John" cannot end up as its own antecedent.
    """
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
    """Scan the transcript for kin phrases and return the RELATED_TO edges found.

    Also sets `gender` on target entities as a side effect, since a kin word
    states it outright.

    HOW: six regex passes over the full text, in a deliberate order. Pattern 6
    (the possessive CHAIN, "my mom's sister Denise") runs FIRST because pattern 1
    would otherwise match its tail ("my mom") and claim the wrong relation; `add`
    dedupes by (source, target), so whichever pattern gets there first wins. The
    remaining patterns go from most to least constrained.

    Each pass does the same three things: match, resolve the captured name to an
    entity with `_entity_at`, and resolve the OWNER -- the interviewee for a
    first-person possessive ("my"/"our"), or the nearest preceding person for a
    third-person one ("his"/"her"/"their"), or the named possessor in pattern 5.
    """
    edges: list[Edge] = []
    seen: set[tuple] = set()  # (source_id, target_id) dedup across patterns


    def add(source_id, target, detail, evidence, gender_word=None):
        """Record one kin relation: create the edge, then infer the target's gender.

        Shared by all six patterns below. Silently does nothing when either side
        is unresolved, when a person would be related to themselves, or when this
        (source, target) pair was already recorded -- which is what makes pattern
        order meaningful, since the first pattern to claim a pair keeps it.

        `detail` is the kin word as written ("aunt", "mom's sister"), stored on
        the edge. `gender_word` overrides which word gender is read from: for the
        possessive chain the relation detail is "mom's sister" but the gender to
        infer is the SECOND word's ("sister" -> female). Gender is only filled in
        when the target has none yet, so an earlier, more direct statement wins.
        """
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
    # second line (extract_pass proposes, graph/checks/relation_evidence.py verifies),
    # which reads the full context.
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