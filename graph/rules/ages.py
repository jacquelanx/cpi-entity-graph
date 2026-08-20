"""
Age parsing and the age <-> date consistency constraint.

Convert spoken ages to an int ("nineteen", "in his forties", "twenty-something"),
and record STATED_WITH edges when an age and a date are stated in the same
sentence ("I was nineteen in 2009") so date-shifting keeps the implied birth
year arithmetically consistent.
"""

from __future__ import annotations

import re
from ..models import Edge, Entity, Relation
from ..text.sentences import sentence_spans


_AGE_ONES = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
             "seven": 7, "eight": 8, "nine": 9}


_AGE_TEENS = {"ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
              "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
              "eighteen": 18, "nineteen": 19}


_AGE_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
             "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}


# Decade expressions: "in my twenties", "mid-thirties", "late forties".
# These are approximate; we store a representative age within the decade.
_AGE_DECADES = {"twenties": 20, "thirties": 30, "forties": 40, "fifties": 50,
                "sixties": 60, "seventies": 70, "eighties": 80, "nineties": 90}


def parse_age_value(text: str) -> tuple[int | None, bool]:
    """THE rule parser for an age expression, as a pure function.

    Returns `(value, approximate)`; `value` is None when the expression cannot be
    parsed. Split out of `resolve_age_entity` so `graph/checks/` can re-run the
    rule's own arithmetic as the deterministic check on an LLM `value` /
    `approximate` proposal -- the same arrangement `checks/identifiers.py` uses
    with `identifiers._normalize`. Keeping ONE parser means the rule layer and
    the checker can never disagree about what the text says.
    """
    text = (text or "").lower()
    m = re.search(r"\d{1,3}", text)
    if m:
        return int(m.group(0)), False

    dec = re.search(r"(early|mid|middle|late)?[\s\-]*(" +
                    "|".join(_AGE_DECADES) + r")\b", text)
    if dec:
        bump = {"early": 2, "mid": 5, "middle": 5, "late": 8}.get(dec.group(1) or "", 5)
        return _AGE_DECADES[dec.group(2)] + bump, True

    # "twenty-something", "thirty-something"
    som = re.search(r"(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)"
                    r"[\s\-]*something", text)
    if som:
        return _AGE_TENS[som.group(1)] + 5, True

    total, found = 0, False
    for tok in re.split(r"[\s\-]+", text):
        if tok in _AGE_TEENS:
            total += _AGE_TEENS[tok]; found = True
        elif tok in _AGE_TENS:
            total += _AGE_TENS[tok]; found = True
        elif tok in _AGE_ONES:
            total += _AGE_ONES[tok]; found = True
    return (total, False) if found else (None, False)


"""
Used for AGE entities; we want a value to store in `entity.attributes`.
Thin wrapper over `parse_age_value` so the parser stays reusable by the checkers.
"""
def resolve_age_entity(entity: Entity) -> None:
    value, approximate = parse_age_value(entity.mentions[0].text)
    if value is None:
        entity.flag_entity("could not parse age value")
        return
    entity.attributes["value"] = value
    if approximate:
        entity.attributes["approximate"] = True


"""
STATED_WITH edges: an age and the date it was co-stated with must stay
arithmetically consistent after date-shifting ("I enlisted in September 2008. I
was eighteen" -> shifting 2008 must move the implied birth year too).

Scoped deliberately: each AGE links to the SINGLE NEAREST date within `window`
sentences (by character distance), not to every date in range. The old
"every age x every date in window" produced quadratic, duplicated, and spurious
links -- e.g. "eighteen" tying to both September 2008 (the real anchor) and a
nearby "9/11". One age has one temporal anchor, so one edge per age.
"""
def age_date_constraints(
    transcript: str, entities: list[Entity], window: int = 1
) -> list[Edge]:

    sentences = sentence_spans(transcript)

    """Return the index of the sentence containing character position `pos`."""
    def sentence_of(pos: int) -> int:
        for i, (s, e) in enumerate(sentences):
            if s <= pos < e:
                return i
        # fallback: if position isn't found, treat it as belonging to the final sentence
        return len(sentences) - 1

    ages = [e for e in entities if e.category == "AGE"]
    dates = [e for e in entities
             if e.category in ("DATE_ABSOLUTE", "DATE_RELATIVE", "DATE_ANCHOR")]
    edges = []
    for a in ages:
        best = None                     # (char_gap, date_entity, age_mention)
        for am in a.mentions:
            a_sent = sentence_of(am.start)
            for d in dates:
                for dm in d.mentions:
                    if abs(sentence_of(dm.start) - a_sent) <= window:
                        gap = abs(dm.start - am.start)
                        if best is None or gap < best[0]:
                            best = (gap, d, am)
        if best is not None:
            _, d, am = best
            s, e = sentences[sentence_of(am.start)]
            edges.append(Edge(
                source=a.entity_id, target=d.entity_id,
                relation=Relation.STATED_WITH,
                detail="age and date co-stated; keep arithmetic",
                evidence=transcript[s:e].strip(),
            ))
    return edges
