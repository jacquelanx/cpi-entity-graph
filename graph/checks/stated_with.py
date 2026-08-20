"""
Deterministic checkers for `stated_with` -- which DATE an AGE was co-stated with.

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
    """The DATE entity whose mention text matches the proposed quote, or None."""
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
    locality bound the rule uses, applied to the model's answer."""
    name = "anchor_is_near"
    d = date_entity_for(value, ctx)
    if d is None:
        return na(name, "the entity checker owns that failure")
    sents = ctx.sents

    def idx(pos):
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
    plausible. This is the arithmetic the STATED_WITH edge exists to protect."""
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
