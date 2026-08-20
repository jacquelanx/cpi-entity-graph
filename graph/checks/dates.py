"""
Deterministic checkers for DATE fields (`resolved_value`, `shiftable`,
`replace_date`).

The rule layer parses with dateutil, a relative-date regex set, and the
`ANCHOR_EVENTS` table. When a rule misses and the LLM fills the gap, these
checkers bound the guess: it must be a real ISO date, it must not sit after the
interview, a DOB must precede it by a plausible lifespan, and an anchor the table
DOES list must not be contradicted.

These run on the FILL path, which previously had no validation at all -- an
LLM date went straight into `suggested_value` unchecked.

`keep_only_if_public_event` / `keep_is_not_a_local_event` are the KEEP gate for
`replace_date`, the field that tells surrogate generation whether a date's surface
text may survive. Shaped exactly like `checks/location.py`'s keep gate, for the same
reason: keeping is the leak-prone direction, so it is the direction that must be
proved.
"""

from __future__ import annotations
from datetime import date as _date
from . import CheckOutcome, ok, fail, na
from .comparators import parse_iso, source_granularity
from ..rules.dates import ANCHOR_EVENTS, _strip_article

_MIN_YEAR = 1850
_MAX_LIFESPAN_YEARS = 120


# Anchor-table events whose fame is REGIONAL. Naming one identifies nobody the way a
# phone number does, but it pins the speaker to one small community in one year --
# interview_002's "Buffalo Creek flood" plus "West Virginia" plus an age plus "miner"
# is a valley of a few thousand people in 1972, and every one of them is findable.
#
# Membership forces the safe direction (replace) rather than asserting anything about
# the date itself, which is the same shape `rules/locations.AMBIGUOUS_BROAD_NAMES` uses
# for the identical problem on place names. `shiftable` is deliberately UNAFFECTED: a
# local disaster still has one fixed calendar date and still constrains the shift, so
# the timeline stays consistent while the phrase gets replaced.
#
# Keep it to events whose recognition is confined to one state or metro area. A
# nationally or internationally known event belongs in neither this set nor a review
# queue -- see `second_line._safer_replace_date`.
LOCAL_ANCHOR_EVENTS = {
    "buffalo creek flood", "buffalo creek", "matewan massacre",
    "farmington mine disaster", "sago mine", "upper big branch",
}


def anchor_phrase_for(text: str) -> str:
    """The `ANCHOR_EVENTS` phrase this span names, or "".

    Article-insensitive and longest-first, so "hurricane katrina" wins over "katrina".
    Four call sites used to carry their own copy of this loop (two here, one in
    `checks/approximate.py`, one in `rules/dates.resolve_date_entity`), which is
    exactly how the rule layer and a checker drift apart.
    """
    t = (text or "").lower()
    t_na = _strip_article(t)
    for phrase in sorted(ANCHOR_EVENTS, key=len, reverse=True):
        pna = _strip_article(phrase)
        if phrase in t or pna in t_na or pna in t:
            return phrase
    return ""


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
    d = parse_iso(value)
    if d is None:
        return na(name, "iso_valid owns that failure")
    phrase = anchor_phrase_for(m.text)
    if not phrase:
        return na(name, "phrase not in the anchor table")
    td = parse_iso(ANCHOR_EVENTS[phrase])
    if td and abs((td - d).days) > 3:
        return fail(name, f"table says {ANCHOR_EVENTS[phrase]} for {phrase!r}, "
                          f"model said {value}")
    return ok(name)


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
    if anchor_phrase_for(m.text):
        return ok(name, "in the anchor table")
    if getattr(ctx.entity, "category", "") != "DATE_ANCHOR":
        return fail(name, f"a {ctx.entity.category} that is not in the anchor "
                          f"table is a private date; it must stay shiftable")
    a = ctx.entity.attributes
    if a.get("suggested_event") and a.get("resolved_value"):
        return ok(name, f"event {a['suggested_event']!r} corroborated by the "
                        f"resolved date {a['resolved_value']}")
    return fail(name, "no table entry, and no named event with a resolved date")


# ------------------------------------------------------- replace / keep the text

def keep_only_if_public_event(value, ctx) -> CheckOutcome:
    """Checker for `replace_date=False` (let a date's surface text survive verbatim).

    Mirrors `checks/location.keep_only_if_broad_place`: keeping is the leak-prone
    direction, so it is the direction that must be proved. A date may only be kept
    when OUR OWN TABLE names it as a public event.

    Deliberately stricter than `is_real_public_event`, which also accepts a
    DATE_ANCHOR the MODEL corroborated. The two questions differ in what a wrong
    answer costs: refusing to shift a date the model calls public is conservative,
    while KEEPING one on the model's word alone leaves a real calendar reference from
    the speaker's life in the transcript. So the shift gate may take the model's
    corroboration and the keep gate may not.
    """
    name = "keep_only_if_public_event"
    if value is not False:
        return na(name, "not a keep claim")
    m = ctx.first_mention()
    if m is None:
        return fail(name, "no span to read")
    phrase = anchor_phrase_for(m.text)
    if not phrase:
        return fail(name, f"{m.text!r} names no event in the anchor table, so it is a "
                          f"private date in this family's life; its surface text must "
                          f"not survive")
    return ok(name, f"{phrase!r} is in the anchor table")


def keep_is_not_a_local_event(value, ctx) -> CheckOutcome:
    """A keep may not rest on an event that is only REGIONALLY famous.

    The discriminating half of the gate, and the analogue of
    `location.keep_only_if_broad_place`'s `AMBIGUOUS_BROAD_NAMES` clause: "the table
    lists it" is not the same fact as "naming it identifies nobody". A named mine
    disaster or valley flood is a public event with a fixed date AND a pointer to a
    community of a few thousand people. See `LOCAL_ANCHOR_EVENTS`.
    """
    name = "keep_is_not_a_local_event"
    if value is not False:
        return na(name, "not a keep claim")
    m = ctx.first_mention()
    phrase = anchor_phrase_for(m.text if m is not None else "")
    if not phrase:
        return na(name, "keep_only_if_public_event owns that failure")
    if phrase in LOCAL_ANCHOR_EVENTS:
        return fail(name, f"{phrase!r} is known regionally, not nationally; naming it "
                          f"pins the speaker to one small community in one year")
    return ok(name, f"{phrase!r} is nationally known; naming it identifies nobody")
