"""
Deterministic checkers for `stated_with` -- which DATE an AGE was co-stated with.

PURPOSE
    Bound the LLM's answer to "which date dates this age?". The pairing feeds a
    STATED_WITH edge, and the date-shifter relies on that edge to keep age
    arithmetic consistent after shifting.

FIT
    Named by the `stated_with` policy in `graph/second_line/policies.py`. The rule
    side is `rules/ages.age_date_constraints`. Uses `comparators.parse_iso` for
    the arithmetic check.

HOW
    All three checkers start by resolving the model's quoted date expression back
    to a real DATE entity (`date_entity_for`). If that lookup fails, the FIRST
    checker reports the failure and the other two return `na` -- a deliberate
    convention in this package, so one root cause produces one explanation rather
    than three.

The rule (`rules/ages.age_date_constraints`) is purely positional: the single
nearest date within one sentence, by character distance. That is a reasonable guess
and often right, but it has no notion of whether the sentence actually ties the two
together -- and the pairing matters, because date-shifting must keep the age
arithmetic consistent ("I enlisted in September 2008. I was eighteen" -> moving 2008
must move the implied birth year too).

The LLM proposes the date expression it reads as the anchor; these checkers bound
it: the quote must be real transcript text, it must belong to a DATE entity this
pipeline actually built, it must sit within one sentence of the age, and -- when
both resolve -- the implied birth year must be plausible.
"""

from __future__ import annotations
import re
from . import CheckOutcome, ok, fail, na
from .comparators import parse_iso

_DATE_CATS = ("DATE_ABSOLUTE", "DATE_RELATIVE", "DATE_ANCHOR", "DATE_OF_BIRTH")
_WINDOW_SENTENCES = 1


def date_entity_for(value, ctx):
    """The DATE entity whose mention text matches the proposed quote, or None.

    The model answers with a QUOTE ("September 2008"), not an entity id, so it has
    to be matched back to a node. Both sides are whitespace-collapsed and
    lowercased before comparing, so "September  2008" still matches. A miss means
    the model paraphrased or invented the expression.
    """
    if not value:
        return None
    want = re.sub(r"\s+", " ", str(value).strip().lower())
    for e in ctx.entities:
        if e.category not in _DATE_CATS:
            continue
        for m in e.mentions:
            if re.sub(r"\s+", " ", m.text.strip().lower()) == want:
                return e
    return None


def anchor_is_a_date_entity(value, ctx) -> CheckOutcome:
    """The quote must name a date THIS pipeline detected -- otherwise there is no
    node to point a STATED_WITH edge at, and the model may have paraphrased."""
    name = "anchor_is_a_date_entity"
    if date_entity_for(value, ctx) is None:
        return fail(name, f"{value!r} is not a detected date expression")
    return ok(name)


def anchor_is_near(value, ctx) -> CheckOutcome:
    """The age and its anchor must sit within one sentence of each other -- the same
    locality bound the rule uses, applied to the model's answer.

    HOW: convert both positions to SENTENCE INDICES and compare, so "within one
    sentence" is a subtraction. Passes if ANY (age mention, date mention) pair is
    close enough, since either entity may be mentioned several times.
    """
    name = "anchor_is_near"
    d = date_entity_for(value, ctx)
    if d is None:
        return na(name, "the entity checker owns that failure")
    sents = ctx.sents

    def idx(pos):
        """The sentence NUMBER containing `pos`, so nearness is a subtraction."""
        for i, (s, e) in enumerate(sents):
            if s <= pos < e:
                return i
        return len(sents) - 1

    for am in ctx.entity.mentions:
        for dm in d.mentions:
            if abs(idx(dm.start) - idx(am.start)) <= _WINDOW_SENTENCES:
                return ok(name)
    return fail(name, f"{value!r} is more than {_WINDOW_SENTENCES} sentence(s) "
                      f"from the age")


def implied_birth_year_plausible(value, ctx) -> CheckOutcome:
    """If both the age and the anchor date resolved, the implied birth year must be
    plausible. This is the arithmetic the STATED_WITH edge exists to protect.

    Subtracting the age from the anchor year gives the implied year of birth. It
    must fall between 1850 and the anchor year itself -- earlier is impossible for
    a living interviewee, later would mean being born after the event. So "I was
    eighteen" beside "September 2008" implies 1990, which passes, while the same
    age beside "1900" would imply 1882 and fail.

    Returns `na` rather than failing when either side did not resolve: there is no
    arithmetic to do, and a different checker owns that gap.
    """
    name = "implied_birth_year_plausible"
    d = date_entity_for(value, ctx)
    if d is None:
        return na(name, "the entity checker owns that failure")
    age = ctx.entity.attributes.get("value")
    iso = d.attributes.get("resolved_value")
    if not isinstance(age, int) or not iso:
        return na(name, "age and anchor are not both resolved")
    dt = parse_iso(iso)
    if dt is None:
        return na(name, "anchor date is not ISO")
    birth = dt.year - age
    if not (1850 <= birth <= dt.year):
        return fail(name, f"age {age} at {iso} implies birth year {birth}")
    return ok(name)
