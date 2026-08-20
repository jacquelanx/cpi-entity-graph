"""
Deterministic checkers for `approximate` -- is this age / date an estimate?

PURPOSE
    Decide whether a resolved age or date is exact or a best guess. Downstream
    this changes how much a value may be perturbed, and it changes what a reviewer
    is being told: presenting an estimate as exact is a quiet accuracy claim
    nothing supports.

FIT
    Named by the `approximate` policy in `graph/second_line/policies.py`. Reads
    `rules/ages.parse_age_value` and the anchor table via `checks/dates.py`, so
    every "is this vague?" judgment traces back to one implementation.

HOW
    The answer is written in the source text, so each checker looks at the SPAN
    and its immediate surroundings. `_hedged` is the shared primitive: a vagueness
    marker inside the span, or a qualifier in the ~20 characters before it. Each
    checker then guards one direction for one category and returns `na` outside
    it, so nothing is counted as verified that was not actually inspected.

`approximate` was the ONE policy in the registry with `checkers=()`. It has both
layers (the rule's own marker, and an LLM proposal on every age and date), so it
looked second-lined, but an LLM fill was accepted with the reason literally "NO
deterministic check applied to this value" -- 16 times across the two sample
transcripts. Nothing could refute a model that called "her eighties" exact or
"March 4th, 1951" an estimate.

The field is cheap to check deterministically, because the answer is written in
the source text. Four checkers, each guarding one direction:

  source_hedge_agrees        a hedge in or just before the span ("maybe", "about",
                             "-odd", "a few", a decade word) proves an estimate;
                             a full month-day-year date proves an exact one.
  age_parser_agrees          re-runs `rules/ages.parse_age_value` -- THE rule
                             parser -- and refuses a value that contradicts what
                             it says about the same span.
  relative_date_is_approx    a DATE_RELATIVE resolved against the interview date
                             is an estimate by construction.
  anchor_in_table_is_exact   a public event the anchor table lists has a fixed
                             calendar date, so it is not approximate.

Policy is the usual one: a checker only ever REFUTES, and returns `na` outside
its own direction, so silence is not counted as verification.
"""

from __future__ import annotations
import re
from . import CheckOutcome, ok, fail, na
from .comparators import source_granularity, _MONTH_RE

# Vagueness markers. Split into two groups because they are matched in different
# places: `_HEDGE_SPAN` can sit inside the span itself ("thirty-odd years ago",
# "forty-something", "her eighties"), while `_HEDGE_LEFT` is a qualifier the
# speaker puts in FRONT of an otherwise exact number ("I was maybe nineteen").
_HEDGE_SPAN = re.compile(
    r"\b(?:odd|something|ish|thereabouts|twenties|thirties|forties|fifties|"
    r"sixties|seventies|eighties|nineties|early|mid|middle|late|couple|few|"
    r"several|some)\b|\bor\s+so\b|-odd\b", re.I)
_HEDGE_LEFT = re.compile(
    r"\b(?:maybe|about|around|roughly|approximately|approx|nearly|almost|"
    r"close\s+to|somewhere\s+(?:around|near)|give\s+or\s+take|sometime|"
    r"round\s+about|like)\s*$", re.I)

# How far back we look for a left-hand hedge. One short clause: far enough for
# "I was maybe nineteen", short enough that a hedge attached to a DIFFERENT
# number earlier in the sentence cannot be read as this span's.
_LEFT_WINDOW = 20

_DATE_CATS = ("DATE_ABSOLUTE", "DATE_RELATIVE", "DATE_ANCHOR", "DATE_OF_BIRTH")


def _span_text(ctx) -> str:
    """The literal text of the span under judgment, or "" if there is none."""
    m = ctx.first_mention()
    return m.text if m is not None else ""


def _hedged(ctx) -> bool:
    """A vagueness marker inside the span, or a qualifier immediately before it.

    Two places to look, because English hedges in two ways. INSIDE the span:
    "thirty-odd", "forty-something", "her eighties". IMMEDIATELY BEFORE it: "I was
    maybe nineteen" -- the number itself is exact-looking, and only the preceding
    word makes it an estimate. The lookback is capped at `_LEFT_WINDOW` characters
    so a hedge attached to a DIFFERENT number earlier in the sentence cannot be
    misread as this one's.
    """
    m = ctx.first_mention()
    if m is None:
        return False
    if _HEDGE_SPAN.search(m.text):
        return True
    left = ctx.transcript[max(0, m.start - _LEFT_WINDOW):m.start]
    return bool(_HEDGE_LEFT.search(left))


def source_hedge_agrees(value, ctx) -> CheckOutcome:
    """The source text says whether this is an estimate. Refute a value that
    contradicts it, in either direction."""
    name = "source_hedge_agrees"
    if not isinstance(value, bool):
        return fail(name, f"{value!r} is not a boolean")
    text = _span_text(ctx)
    if not text:
        return na(name, "no span to read")
    if _hedged(ctx):
        if value is False:
            return fail(name, f"{text!r} carries a vagueness marker; not exact")
        return ok(name, "hedged expression, marked approximate")
    # No hedge. A span that pins a month AND a day is exact; anything vaguer
    # (a bare year, a season, a lone number) is not provably either way.
    if getattr(ctx.entity, "category", "") in _DATE_CATS:
        if _MONTH_RE.search(text) and re.search(r"\d{1,2}\b", text) and \
                source_granularity(text) == "day":
            if value is True:
                return fail(name, f"{text!r} names a month and a day; not an estimate")
            return ok(name, "day-precise date, marked exact")
    return na(name, "no hedge and no day-precise date; not decidable from the span")


def age_parser_agrees(value, ctx) -> CheckOutcome:
    """Re-run the rule's own age parser over the span. It reports approximation
    for decade and '-something' forms, so it can refute a contradicting claim
    without the checker reimplementing the vocabulary."""
    name = "age_parser_agrees"
    if getattr(ctx.entity, "category", "") != "AGE":
        return na(name, "not an age")
    if not isinstance(value, bool):
        return fail(name, f"{value!r} is not a boolean")
    from ..rules.ages import parse_age_value
    text = _span_text(ctx)
    parsed, approx = parse_age_value(text)
    if parsed is None:
        return na(name, "the rule parser could not read this span")
    if approx and value is False:
        return fail(name, f"the rule parser reads {text!r} as approximate")
    if not approx and value is True and not _hedged(ctx):
        return fail(name, f"the rule parser reads {text!r} as an exact age and "
                          f"nothing hedges it")
    return ok(name, f"rule parser and value agree ({'approx' if approx else 'exact'})")


def relative_date_is_approximate(value, ctx) -> CheckOutcome:
    """A relative expression is resolved by arithmetic against the interview date,
    so the result is an estimate whatever the model thinks -- except for the three
    day-exact forms the rule handles precisely."""
    name = "relative_date_is_approximate"
    if getattr(ctx.entity, "category", "") != "DATE_RELATIVE":
        return na(name, "not a relative date")
    text = _span_text(ctx).lower()
    if re.search(r"\b(?:yesterday|today|tonight|tomorrow|this\s+morning|"
                 r"this\s+afternoon)\b", text):
        return na(name, "a day-exact relative form; the rule resolves it precisely")
    if value is False:
        return fail(name, f"{text!r} is resolved by arithmetic against the "
                          f"interview date; the result is an estimate")
    return ok(name, "relative expression, marked approximate")


def anchor_in_table_is_exact(value, ctx) -> CheckOutcome:
    """A public event the table lists has one fixed calendar date."""
    name = "anchor_in_table_is_exact"
    if getattr(ctx.entity, "category", "") != "DATE_ANCHOR":
        return na(name, "not an anchor")
    # THE shared table match (`checks/dates.anchor_phrase_for`), not a fourth copy of
    # the article-insensitive longest-first loop.
    from .dates import anchor_phrase_for
    phrase = anchor_phrase_for(_span_text(ctx))
    if not phrase:
        return na(name, "phrase not in the anchor table")
    if value is True:
        return fail(name, f"{phrase!r} is in the anchor table with a "
                          f"fixed date; not an estimate")
    return ok(name, "listed public event, marked exact")
