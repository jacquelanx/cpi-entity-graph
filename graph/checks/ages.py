"""
Deterministic checkers for AGE `value`.

The rule layer parses digits, spelled-out numbers, decade expressions and
'-something' forms. When it misses and the LLM fills in, the guess must be a
plausible whole-year age and must not contradict a DOB the graph already knows
for the same person (via ATTRIBUTE_OF), given the interview date.
"""

from __future__ import annotations
import re
from . import CheckOutcome, ok, fail, na
from .comparators import parse_iso
from ..models import Relation

_MIN_AGE, _MAX_AGE = 0, 115

# Units that make a bare number a MEASUREMENT rather than an age. A detector that
# tags every "twelve" in a transcript hands us "the water came up twelve feet" as an
# AGE, and nothing downstream could tell it from a person's age: it parses, it is in
# range, and once ownership resolves it becomes somebody's age in the graph. Since the
# speaker's own ages feed their surrogate identity directly, a measurement adopted as
# an age is a real corruption, and it is cheap to refute from the following word.
_UNIT_AFTER = re.compile(
    r"^\W{0,3}(?:feet|foot|ft|inch(?:es)?|yards?|miles?|acres?|pounds?|lbs?|"
    r"tons?|ounces?|gallons?|quarts?|barrels?|bushels?|dollars?|cents?|bucks?|"
    r"percent|degrees?|hours?|minutes?|seconds?|days?|weeks?|months?|miles|"
    r"head|rows?|feet\b)\b", re.I)
# "years old" / "year old" IS an age; a bare "years" is a duration.
_AGE_UNIT_AFTER = re.compile(r"^\W{0,3}(?:years?|yrs?)\s+old\b", re.I)


def plausible_range(value, ctx) -> CheckOutcome:
    name = "plausible_age_range"
    if not isinstance(value, int):
        return fail(name, f"{value!r} is not an integer")
    if not (_MIN_AGE <= value <= _MAX_AGE):
        return fail(name, f"age {value} out of range")
    return ok(name)


def not_a_measurement(value, ctx) -> CheckOutcome:
    """Refute an AGE whose span is immediately followed by a measurement unit.

    "the water came up twelve feet" is not an age. Deliberately narrow: only the word
    RIGHT AFTER the span is inspected, and "years old" is explicitly excluded so a
    real "sixty-eight years old" still passes. Unanimity is not required across
    mentions because AGE entities are now one-per-mention (see
    `pipeline._simple_entities`), so each span is judged on its own text.
    """
    name = "not_a_measurement"
    m = ctx.first_mention()
    if m is None:
        return na(name, "no span to read")
    after = ctx.transcript[m.end:m.end + 24]
    if _AGE_UNIT_AFTER.match(after):
        return ok(name, "followed by 'years old'")
    hit = _UNIT_AFTER.match(after)
    if hit:
        return fail(name, f"{m.text!r} is followed by {hit.group(0).strip()!r}; "
                          f"this is a measurement, not an age")
    return ok(name, "not followed by a measurement unit")


def _owner_of(ctx, entity_id):
    for ed in ctx.edges:
        if ed.relation == Relation.ATTRIBUTE_OF and ed.source == entity_id:
            return ed.target
    return None


def consistent_with_dob(value, ctx) -> CheckOutcome:
    """If this age belongs to a person whose DOB is known, and the age is stated
    at the interview ('Now I'm sixty-eight'), the arithmetic must work.

    Deliberately narrow: only applied when the age and the DOB are attributed to
    the SAME person and an interview date exists. A retrospective age ('I was
    seven') legitimately disagrees with age-at-interview, so we only refute when
    the stated age EXCEEDS the age at interview -- which is impossible.
    """
    name = "consistent_with_dob"
    if not isinstance(value, int) or ctx.interview_date is None:
        return na(name, "no integer age or no interview_date")
    owner = _owner_of(ctx, ctx.entity.entity_id)
    if owner is None:
        return na(name, "age not attributed to a person")
    dob = None
    for e in ctx.entities:
        if e.category != "DATE_OF_BIRTH":
            continue
        if _owner_of(ctx, e.entity_id) != owner:
            continue
        dob = parse_iso(e.attributes.get("resolved_value"))
        if dob:
            break
    if dob is None:
        return na(name, "no DOB for this person")
    age_at_interview = (ctx.interview_date - dob).days / 365.25
    if value > age_at_interview + 1:
        return fail(name, f"age {value} exceeds age at interview "
                          f"({age_at_interview:.0f}) implied by DOB {dob}")
    return ok(name)
