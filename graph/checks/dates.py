"""
Deterministic checkers for DATE fields (`resolved_value`, `shiftable`).

The rule layer parses with dateutil, a relative-date regex set, and the
`ANCHOR_EVENTS` table. When a rule misses and the LLM fills the gap, these
checkers bound the guess: it must be a real ISO date, it must not sit after the
interview, a DOB must precede it by a plausible lifespan, and an anchor the table
DOES list must not be contradicted.

These run on the FILL path, which previously had no validation at all -- an
LLM date went straight into `suggested_value` unchecked.
"""

from __future__ import annotations
from datetime import date as _date
from . import CheckOutcome, ok, fail, na
from .comparators import parse_iso, source_granularity
from ..location_dates import ANCHOR_EVENTS, _strip_article

_MIN_YEAR = 1850
_MAX_LIFESPAN_YEARS = 120


def iso_valid(value, ctx) -> CheckOutcome:
    name = "iso_valid"
    d = parse_iso(value)
    if d is None:
        return fail(name, f"{value!r} is not an ISO date")
    if not (_MIN_YEAR <= d.year <= _date.today().year + 1):
        return fail(name, f"year {d.year} out of plausible range")
    return ok(name)


def not_after_interview(value, ctx) -> CheckOutcome:
    """A date recounted in a past-tense life-history interview cannot postdate it.
    Skipped when no `interview_date` is available."""
    name = "not_after_interview"
    if ctx.interview_date is None:
        return na(name, "no interview_date to bound against")
    d = parse_iso(value)
    if d is None:
        return na(name, "iso_valid owns that failure")
    if d > ctx.interview_date:
        return fail(name, f"{value} is after the interview ({ctx.interview_date})")
    return ok(name)


def dob_plausible(value, ctx) -> CheckOutcome:
    """A DATE_OF_BIRTH must precede the interview by less than a max lifespan."""
    name = "dob_plausible"
    if getattr(ctx.entity, "category", "") != "DATE_OF_BIRTH":
        return na(name, "not a DOB")
    if ctx.interview_date is None:
        return na(name, "no interview_date to bound against")
    d = parse_iso(value)
    if d is None:
        return na(name, "iso_valid owns that failure")
    years = (ctx.interview_date - d).days / 365.25
    if not (0 < years < _MAX_LIFESPAN_YEARS):
        return fail(name, f"implies age {years:.0f} at interview")
    return ok(name)


def anchor_table_not_contradicted(value, ctx) -> CheckOutcome:
    """If the mention text DOES match a public event the table lists, the model
    must not disagree with the table by more than a few days. Public events have
    exact dates, so a real divergence means one of the two is simply wrong."""
    name = "anchor_table_not_contradicted"
    if getattr(ctx.entity, "category", "") != "DATE_ANCHOR":
        return na(name, "not an anchor")
    m = ctx.first_mention()
    if m is None:
        return na(name, "no mention to read")
    text = m.text.lower()
    text_na = _strip_article(text)
    d = parse_iso(value)
    if d is None:
        return na(name, "iso_valid owns that failure")
    for phrase, iso in sorted(ANCHOR_EVENTS.items(), key=lambda kv: -len(kv[0])):
        pna = _strip_article(phrase)
        if phrase in text or pna in text_na or pna in text:
            td = parse_iso(iso)
            if td and abs((td - d).days) > 3:
                return fail(name, f"table says {iso} for {phrase!r}, model said {value}")
            return ok(name)
    return na(name, "phrase not in the anchor table")


def granularity_respected(value, ctx) -> CheckOutcome:
    """Reject over-specification. A source that names only a year must not produce
    a fill claiming a specific month/day beyond Jan 1 -- that is invented
    precision, and date-shifting would propagate it."""
    name = "granularity_respected"
    m = ctx.first_mention()
    if m is None:
        return na(name, "no mention to read")
    d = parse_iso(value)
    if d is None:
        return na(name, "iso_valid owns that failure")
    gran = source_granularity(m.text)
    if gran == "year" and (d.month, d.day) != (1, 1):
        return fail(name, f"source {m.text!r} names only a year; {value} invents a day")
    return ok(name)


def is_real_public_event(value, ctx) -> CheckOutcome:
    """Checker for `shiftable=False`: a date may only be pinned as non-shiftable
    when a PUBLIC EVENT is corroborated -- either the anchor table lists the
    phrase, or the model NAMED an event and also produced a date for it.

    The old fallback was `resolved_value or suggested_event`, which is too loose in
    two ways. `resolved_value` alone is true of every date the rules parsed, so
    once `shiftable` is arbitrated for DATE_ABSOLUTE / DATE_RELATIVE /
    DATE_OF_BIRTH as well as DATE_ANCHOR (it now is -- the rule sets `shiftable`
    for all four, so all four need the second line), a plain birthday like
    "March 4th, 1951" would have cleared this check and been pinned as
    non-shiftable, freezing a real calendar point in place. And `suggested_event`
    alone lets a bare event NAME with no date do the work. Both are now required,
    and only for a DATE_ANCHOR.
    """
    name = "is_real_public_event"
    if value is not False:
        return na(name, "not a non-shiftable claim")
    m = ctx.first_mention()
    if m is None:
        return fail(name, "no mention")
    text = m.text.lower()
    text_na = _strip_article(text)
    for phrase in ANCHOR_EVENTS:
        pna = _strip_article(phrase)
        if phrase in text or pna in text_na or pna in text:
            return ok(name, "in the anchor table")
    if getattr(ctx.entity, "category", "") != "DATE_ANCHOR":
        return fail(name, f"a {ctx.entity.category} that is not in the anchor "
                          f"table is a private date; it must stay shiftable")
    a = ctx.entity.attributes
    if a.get("suggested_event") and a.get("resolved_value"):
        return ok(name, f"event {a['suggested_event']!r} corroborated by the "
                        f"resolved date {a['resolved_value']}")
    return fail(name, "no table entry, and no named event with a resolved date")
