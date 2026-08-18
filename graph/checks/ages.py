"""
Deterministic checkers for AGE `value` and `replace_age`.

The rule layer parses digits, spelled-out numbers, decade expressions and
'-something' forms. When it misses and the LLM fills in, the guess must be a
plausible whole-year age and must not contradict a DOB the graph already knows
for the same person (via ATTRIBUTE_OF), given the interview date.

`keep_only_if_refuted_as_an_age` / `keep_not_an_explicit_age_phrase` are the KEEP
gate for `replace_age`, the field that tells surrogate generation whether an age
span's surface text may survive. AGE entities used to reach the generator with no
`replace` key AND no `shiftable` key -- the only category in the graph with no
redaction directive of any kind -- so a consumer keying off `replace` emitted the
speaker's ages verbatim. An age is a quasi-identifier, so the default is to replace,
and the ONLY keepable span is one a deterministic check proved is not an age at all.
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


# ------------------------------------------------------- replace / keep the text

# The age checks whose FAILURE means "this span is not an age", as opposed to "this
# span is an age and the number is wrong". Only the first kind licenses keeping the
# text: `plausible_age_range` failing says the VALUE is unusable, and an age we could
# not read is still an age. Getting this distinction wrong would keep "a hundred and
# twenty" -- out of range, therefore refuted, therefore (wrongly) safe to print.
_NOT_AN_AGE_CHECKS = ("not_a_measurement",)


def age_reading_refuted(entity) -> str:
    """The deterministic check that proved this span is NOT an age, or "".

    Reads the `value` Resolution off the entity's provenance, so the RULE layer
    (`second_line._rule_value` for `replace_age`), the safe-direction function and
    the checker below all ask one question of one record -- the same anti-drift
    arrangement `owner_survivors` and `support_for` use.
    """
    res = (getattr(entity, "provenance", None) or {}).get("value")
    failed = tuple(getattr(res, "checks_failed", ()) or ())
    return ", ".join(f for f in failed if f in _NOT_AN_AGE_CHECKS)


def keep_only_if_refuted_as_an_age(value, ctx) -> CheckOutcome:
    """Checker for `replace_age=False` (let an age span's text survive verbatim).

    An age is a quasi-identifier -- "sixty-eight" plus a holler plus "miner" is one
    household -- so the only span this pipeline will print back is one it has
    positively established is NOT an age: "the water came up twelve feet". Anything
    else, including an age neither layer could parse, is replaced.
    """
    name = "keep_only_if_refuted_as_an_age"
    if value is not False:
        return na(name, "not a keep claim")
    why = age_reading_refuted(ctx.entity)
    if not why:
        return fail(name, "nothing proved this span is anything other than a person's "
                          "age, and an age is a quasi-identifier; only a span refuted "
                          f"by {'/'.join(_NOT_AN_AGE_CHECKS)} may be kept")
    return ok(name, f"the age reading was refuted by {why}")


def keep_not_an_explicit_age_phrase(value, ctx) -> CheckOutcome:
    """A span written as an age is never keepable, whatever the value resolution said.

    The discriminating half of the gate: `keep_only_if_refuted_as_an_age` trusts a
    checker's verdict, and this one reads the text directly. "sixty-eight years old"
    is an age in anybody's reading, so if some future refutation ever knocked its
    value out, the span still must not survive. Same `_AGE_UNIT_AFTER` marker
    `not_a_measurement` uses, so the two cannot disagree about what an age phrase
    looks like.
    """
    name = "keep_not_an_explicit_age_phrase"
    if value is not False:
        return na(name, "not a keep claim")
    m = ctx.first_mention()
    if m is None:
        return na(name, "no span to read")
    if _AGE_UNIT_AFTER.match(ctx.transcript[m.end:m.end + 24]):
        return fail(name, f"{m.text!r} is followed by 'years old'; whatever the value "
                          f"resolution concluded, this reads as a person's age")
    return ok(name, "the span is not written as an explicit age phrase")
