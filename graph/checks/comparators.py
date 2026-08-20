"""
Comparators: "do the rule and the LLM agree?", one function per kind of value.

PURPOSE
    When BOTH layers produced a value for a field, something has to decide whether
    they AGREE (outcome: confirm) or CONFLICT. That is not string equality --
    "Papaw" and "grandfather" are the same answer, "parish" and "county" are the
    same answer, and 1975-03-20 vs 1975-04-01 may or may not be, depending on how
    precise the source text could possibly have been. This module holds those
    judgments.

FIT
    Referenced by `graph/second_line/policies.py` (each `FieldPolicy` names its
    comparator) and called by `graph/second_line/engine.py`. Some comparators are
    plain functions `(a, b, ctx) -> bool`; others are FACTORIES that take a
    tolerance and return such a function (`date_close(3)`, `age_close(1)`), which
    is how per-field tolerances stay in the policy table rather than hard-coded
    here.

HOW
    Most comparators canonicalize both sides through a lookup table and compare
    the canonical forms -- `LOC_CANON` for place types, `_KIN_CANON` for kin words,
    `canon_kind` for identifier types. The date and age comparators instead widen
    their tolerance based on how vague the SOURCE TEXT was, which is the idea worth
    understanding here: agreement is judged at the precision the transcript can
    actually support.

Every field policy must name a comparator explicitly -- there is deliberately no
default. A missing comparator on a free-text field would make `confirm` silently
unreachable, which looks like working code and measures like a bug.
"""

from __future__ import annotations
import re
from datetime import date as _date

_YEAR_RE = re.compile(r"\b(?:1[89]\d{2}|20\d{2})\b")
_MONTH_RE = re.compile(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", re.I)
_SEASON_RE = re.compile(r"\b(?:spring|summer|fall|autumn|winter)\b", re.I)
# A spoken/dialect year carries no digits: "nineteen and sixty", "nineteen
# sixty-five", "eighteen ninety". It is still YEAR granularity -- without this the
# text looked day-precise and an LLM fill of '1960-05-02' passed unchallenged.
_SPELLED_YEAR = re.compile(
    r"\b(?:eighteen|nineteen|twenty)\b[\s\-]+(?:and[\s\-]+)?"
    r"(?:hundred|oh|o|zero|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
    r"twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)\b", re.I)


def exact(a, b, ctx=None) -> bool:
    """Identity: the two values must be equal as-is. For enums and booleans."""
    return a == b


def ci(a, b, ctx=None) -> bool:
    """Case-insensitive string match, ignoring surrounding whitespace."""
    return str(a).strip().lower() == str(b).strip().lower()


def upper(a, b, ctx=None) -> bool:
    """Case-insensitive match for values conventionally stored uppercase (subtypes)."""
    return str(a).strip().upper() == str(b).strip().upper()


def parse_iso(s):
    """Parse the leading `YYYY-MM-DD` of a string into a `date`, or None.

    Slicing to 10 characters tolerates a full timestamp. Returns None rather than
    raising on anything unparseable, so a comparator can simply treat "could not
    read it" as disagreement.
    """
    try:
        return _date.fromisoformat(str(s)[:10])
    except Exception:
        return None


def source_granularity(text: str) -> str:
    """How precise can the SOURCE TEXT possibly be?

    'March 4th, 1951' -> day; 'spring of 1975' -> month (a season pins a month,
    not a day); '1921' -> year. This replaces the old `_year_only` guard, which
    treated 'spring of 1975' as year-only and therefore silently CONFIRMED a
    rule value that was 78 days wrong.
    """
    t = text or ""
    if _MONTH_RE.search(t):
        return "day"
    if _SEASON_RE.search(t):
        return "month"
    if _YEAR_RE.search(t) or _SPELLED_YEAR.search(t):
        return "year"
    return "day"


def date_close(tol_days: int):
    """Build a date comparator that judges agreement at the SOURCE's granularity.

    Returns a comparator function; `tol_days` is the slack allowed only when the
    text really was day-precise.

    The point: comparing two ISO dates exactly is wrong, because a resolved date is
    always more precise than the words it came from. If the transcript says "1921",
    then 1921-01-01 and 1921-06-15 are the SAME answer and only the year should be
    compared. If it says "spring of 1975", compare year and month. Only when a
    month name is present is a day-level comparison meaningful, and then
    `tol_days` absorbs off-by-a-few differences.

    So the comparator looks at the entity's first mention (via `ctx`), asks
    `source_granularity` how precise that text could be, and compares at exactly
    that level.
    """
    def cmp(rule_value, llm_value, ctx=None) -> bool:
        """Do these two ISO dates agree at the source text's granularity?"""
        rd, ld = parse_iso(rule_value), parse_iso(llm_value)
        if rd is None or ld is None:
            return False
        src = ""
        if ctx is not None:
            m = ctx.first_mention()
            src = m.text if m else ""
        gran = source_granularity(src)
        if gran == "year":
            return rd.year == ld.year
        if gran == "month":
            return (rd.year, rd.month) == (ld.year, ld.month)
        return abs((rd - ld).days) <= tol_days
    return cmp


def age_close(tol: int, approx_tol: int = 5):
    """Compare ages, loosening the tolerance for an APPROXIMATE expression.

    'her eighties' and 'his fifties' have no exact answer -- the rule picks a
    representative age (85, 55) and the model picks another (80, 50). At tol=1
    every decade expression produced a conflict flag that told a reviewer nothing.

    Returns a comparator that uses `tol` normally and `approx_tol` when the
    entity carries `approximate=True` -- the flag `rules/ages.parse_age_value`
    sets for exactly these vague expressions.
    """
    def cmp(rule_value, llm_value, ctx=None) -> bool:
        """Do these two ages agree, allowing more slack for a vague expression?"""
        try:
            r, l = int(rule_value), int(llm_value)
        except (TypeError, ValueError):
            return False
        t = tol
        if ctx is not None and getattr(ctx, "entity", None) is not None:
            if ctx.entity.attributes.get("approximate"):
                t = approx_tol
        return abs(r - l) <= t
    return cmp


# Location types: the gazetteer and the model use overlapping but different
# vocabularies. Compare on a canonical bucket so 'parish' vs 'county' or
# 'university' vs 'institution' is agreement, not a spurious conflict.
LOC_CANON = {
    "country": "country", "territory": "country",
    "state": "state", "district": "state", "province": "state",
    "region": "region", "county": "region", "parish": "region",
    "island": "region", "metro": "region",
    "city": "city", "town": "city", "village": "city", "borough": "city",
    "neighborhood": "neighborhood", "neighbourhood": "neighborhood",
    "street": "street", "road": "street", "avenue": "street", "boulevard": "street",
    "institution": "institution", "university": "institution", "college": "institution",
    "school": "institution", "hospital": "institution", "church": "institution",
    "store": "institution", "business": "institution", "airport": "institution",
    "installation": "institution", "base": "institution",
    "landmark": "landmark", "park": "landmark", "monument": "landmark",
    "feature": "feature", "river": "feature", "creek": "feature", "lake": "feature",
    "gulf": "feature", "bay": "feature", "ocean": "feature", "sea": "feature",
    "mountain": "feature", "valley": "feature", "delta": "feature", "bayou": "feature",
    "other": "other",
}


def loc_type(a, b, ctx=None) -> bool:
    """Do two place-type words denote the same GRANULARITY bucket?

    Both sides are looked up in `LOC_CANON`, so "parish" and "county" both become
    "region" and count as agreement. An unrecognized word on either side yields
    None and therefore disagreement -- unknown is not agreement.
    """
    ca = LOC_CANON.get(str(a).strip().lower())
    cb = LOC_CANON.get(str(b).strip().lower())
    return ca is not None and ca == cb


def id_kind(a, b, ctx=None) -> bool:
    """Compare an entity CATEGORY against the model's `kind` word.

    Both sides go through `canon_kind`, so this works whichever vocabulary each
    side arrived in -- the rule value is a category ('SSN_OR_ID'), the LLM value is
    a word ('ssn'), and 'id' and 'ssn' both denote SSN_OR_ID.
    """
    from .identifiers import canon_kind
    ca, cb = canon_kind(a), canon_kind(b)
    return ca is not None and ca == cb


def boolean(a, b, ctx=None) -> bool:
    """Truthiness match, so `True`/`1`/`"yes"`-shaped values compare as equal.

    `bool(a) is bool(b)` uses `is` safely because `bool()` always returns one of
    the two singleton objects `True`/`False`.
    """
    return bool(a) is bool(b)


# Kin synonyms collapse to one canonical term, so the rule's transcript word
# ("Papaw", "mama") and the model's normalized word ("grandfather", "mother") count
# as AGREEMENT instead of a conflict.
_KIN_CANON = {}
for _base, _variants in {
    "mother": "mother mom mommy mum mummy mama mamma momma ma",
    "father": "father dad daddy papa poppa pop pops pa",
    "grandmother": "grandmother grandma grandmom granny nana nanna gramma grammy meemaw mamaw",
    "grandfather": "grandfather grandpa granddad grandad grandpop gramps papaw pawpaw pappy",
    "sister": "sister sis",
    "brother": "brother bro",
    "child": "child kid",
    "partner": "partner spouse",
}.items():
    for _v in _variants.split():
        _KIN_CANON[_v] = _base


def kin_canon(word: str) -> str:
    """The canonical kin term for a word ("papaw" -> "grandfather").

    Unknown words pass through unchanged, so two identical unlisted terms still
    compare equal.
    """
    w = str(word or "").strip().lower()
    return _KIN_CANON.get(w, w)


def kin_synonym(a, b, ctx=None) -> bool:
    """Do two kin words name the same relationship? ("mama" vs "mother" -> True)"""
    return kin_canon(a) == kin_canon(b)
