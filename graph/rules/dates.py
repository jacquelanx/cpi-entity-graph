"""
Putting every date mention on the timeline.

DATE_ABSOLUTE  --> parse directly (dateutil).
DATE_RELATIVE  --> resolve against the interview_date from metadata.
DATE_ANCHOR    --> match against `ANCHOR_EVENTS`; public events cannot move, so
                   they CONSTRAIN the date-shift offset.
"""

from __future__ import annotations

import re
from datetime import timedelta, datetime
from dateutil import parser as dateparser
from ..models import Entity


# ----------------------------- Dates -----------------------------

# Public events that can NEVER be shifted. Matched as substrings of the mention
# text so phrases are kept SPECIFIC (e.g. "hurricane sandy", not "sandy") to
# avoid colliding with personal names. Longest phrases are tried first.
ANCHOR_EVENTS = {
    # natural disasters
    "hurricane katrina": "2005-08-29", "katrina": "2005-08-29",
    "hurricane sandy": "2012-10-29",
    "hurricane harvey": "2017-08-25",
    "deepwater horizon": "2010-04-20", "bp oil spill": "2010-04-20",
    "fukushima": "2011-03-11",
    # Named disasters that recur in US oral history and that the table's own
    # "anchor phrase not in ANCHOR_EVENTS table - add it" flag was asking for.
    # Without a row here BOTH layers went silent on interview_002's "Buffalo Creek
    # flood" -- no rule value, and the LLM's date was not accepted either -- so the
    # date-shifter got no constraint for a fixed public event.
    "buffalo creek flood": "1972-02-26", "buffalo creek": "1972-02-26",
    "matewan massacre": "1920-05-19",
    "farmington mine disaster": "1968-11-20",
    "sago mine": "2006-01-02",
    "upper big branch": "2010-04-05",
    "hurricane rita": "2005-09-24",
    "hurricane ike": "2008-09-13",
    "hurricane camille": "1969-08-17",
    "mount st. helens": "1980-05-18", "mount saint helens": "1980-05-18",
    "dust bowl": "1935-04-14",
    "saigon fell": "1975-04-30", "fall of saigon": "1975-04-30",
    # political
    "obama got elected": "2008-11-04", "obama was elected": "2008-11-04",
    "obama's election": "2008-11-04", "obama inauguration": "2009-01-20",
    "capitol riot": "2021-01-06", "january 6th": "2021-01-06",
    "jan 6th": "2021-01-06",
    # public health
    "covid": "2020-03-01", "the pandemic": "2020-03-01",
    "the lockdown": "2020-03-01", "quarantine started": "2020-03-01",
    # terror / violence
    "9/11": "2001-09-11", "september 11th": "2001-09-11",
    "sept 11th": "2001-09-11", "twin towers": "2001-09-11",
    "boston marathon bombing": "2013-04-15",
    "george floyd": "2020-05-25",
    # milestones
    "y2k": "2000-01-01", "the new millennium": "2000-01-01",
    "the great recession": "2008-09-15", "financial crisis": "2008-09-15",
    "london olympics": "2012-07-27",
}


def _strip_article(s: str) -> str:
    """Drop a leading article so anchor matching is article-insensitive."""
    return re.sub(r"^(?:the|a|an)\s+", "", s.strip())


# Fixed default for dateutil so missing components resolve DETERMINISTICALLY
# (missing day -> 1, missing month -> January) instead of dateutil silently
# defaulting to "today", which made "March 1990" resolve to today's day-of-month
# and change with the run date.
_ABS_DATE_DEFAULT = datetime(2000, 1, 1)


# a plausible explicit 4-digit year; used to detect year-less absolute dates
_YEAR_RE = re.compile(r"\b(?:1[89]\d{2}|20\d{2})\b")


# approximate mid-season dates
_SEASONS = {"spring": "03-20", "summer": "06-21", "fall": "09-22",
            "autumn": "09-22", "winter": "12-21"}


# a written month name, used to tell "nineteen sixty" (year only) from a spoken year
# that also pins a month
_MONTH_NAME_RE = re.compile(
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", re.I)


# "two years ago", "a couple of months ago", "several weeks ago", "3 days ago"
_REL_AGO = re.compile(
    r"\b(a|an|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"a\s+couple(?:\s+of)?|a\s+few|several|\d+)\s+"
    r"(day|week|month|year|decade)s?\s+ago\b", re.I)


# "last year", "this past month", "next week"
_REL_UNIT = re.compile(
    r"\b(last|this\s+past|this|next)\s+(week|month|year|decade)\b", re.I)


# "last spring", "this past summer", "last winter"
_REL_SEASON = re.compile(
    r"\b(?:last|this\s+past|this)\s+(spring|summer|fall|autumn|winter)\b", re.I)


_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday",
             "saturday", "sunday"]


_REL_WEEKDAY = re.compile(rf"\b(?:last|this\s+past)\s+({'|'.join(_WEEKDAYS)})\b", re.I)


_UNIT_DAYS = {"day": 1, "week": 7, "month": 30, "year": 365, "decade": 3650}


_WORD_NUM = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}


"""Turn the count group of _REL_AGO into an integer (vague -> estimate)."""
def _count_from_word(raw: str) -> int:
    raw = raw.strip().lower()
    if "couple" in raw:
        return 2
    if "few" in raw:
        return 3
    if "several" in raw:
        return 4
    return _WORD_NUM.get(raw, None) or int(raw)


"""
Fill entity.attributes with resolved_value / shiftable / approximate.
DATE_ABSOLUTE / DATE_OF_BIRTH:
Parse explicit dates (e.g. "March 4, 2020") into ISO format.

DATE_ANCHOR:
Resolve references to fixed public events and ISO (e.g. "9/11") using the
`ANCHOR_EVENTS` lookup table (hardcoded). These dates are marked as
`shiftable=False`.

DATE_RELATIVE:
Resolve relative expressions (e.g. "2 weeks ago", "last spring", etc)
relative to the interview date (metadata). These resolutions are marked
`approximate=True` since they represent estimates.
"""
# A SPOKEN year, which carries no digits: "nineteen and sixty", "nineteen
# sixty-five", "eighteen ninety". dateutil cannot read these, so the rule layer
# produced nothing for them and the field fell to the LLM -- which the checkers then
# had to gate with no rule value to compare against. `checks/comparators` already
# recognized the shape for GRANULARITY purposes; this makes the rule parser able to
# actually resolve it.
_SPOKEN_UNITS = {
    "hundred": 0, "oh": 0, "o": 0, "zero": 0, "naught": 0, "aught": 0,
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90,
}


_SPOKEN_CENTURY = {"eighteen": 1800, "nineteen": 1900, "twenty": 2000}


_SPOKEN_YEAR_RE = re.compile(
    r"\b(eighteen|nineteen|twenty)\b[\s\-]+(?:and[\s\-]+)?"
    r"((?:" + "|".join(_SPOKEN_UNITS) + r")(?:[\s\-]+(?:"
    + "|".join(_SPOKEN_UNITS) + r"))?)\b", re.I)


def parse_spoken_year(raw: str) -> int | None:
    """The 4-digit year a spelled-out year expression denotes, else None.
    "nineteen and sixty" -> 1960, "nineteen sixty-five" -> 1965,
    "nineteen hundred" -> 1900."""
    m = _SPOKEN_YEAR_RE.search(raw or "")
    if not m:
        return None
    base = _SPOKEN_CENTURY[m.group(1).lower()]
    offset = 0
    for tok in re.split(r"[\s\-]+", m.group(2).lower()):
        if tok in _SPOKEN_UNITS:
            offset += _SPOKEN_UNITS[tok]
    if not (0 <= offset <= 99):
        return None
    return base + offset


def parse_absolute_date(raw: str) -> tuple[str | None, bool]:
    """THE rule parser for an absolute date expression, as a pure function.

    Returns `(iso_or_None, has_explicit_year)`. Split out of
    `resolve_date_entity` for the same reason as `parse_age_value`: the
    deterministic checkers in `graph/checks/` re-run the rule's own parser rather
    than reimplementing it, so the two layers cannot drift apart.

    Handles three things dateutil alone does not:

      * a SEASON with a year ("spring of 1975", "the winter of 1972"). `_SEASONS`
        existed but was consulted only on the DATE_RELATIVE branch, so an absolute
        season fell through to dateutil's fixed default and resolved to JANUARY 1 --
        78 days out. Worse, `resolved_value` is RULE_WINS, so the model's correct
        "1975-04" lost the conflict to the rule's wrong "1975-01-01". Verified: the
        one date failure in interview_001.
      * a SPOKEN year with no digits (`parse_spoken_year`).
      * a two-digit year, which dateutil silently pivots into the FUTURE ("winter of
        '72" -> 2072). Reported as year-less so the caller flags it; the ISO value is
        still returned because `checks/dates.iso_valid` refutes an implausible year
        and `verify_always` then erases it.
    """
    text = (raw or "").strip()
    has_year = bool(_YEAR_RE.search(text))

    spoken = None if has_year else parse_spoken_year(text)
    year = int(_YEAR_RE.search(text).group(0)) if has_year else spoken
    if spoken is not None:
        has_year = True

    if year is not None:
        m = re.search(r"\b(spring|summer|fall|autumn|winter)\b", text, re.I)
        if m:
            return f"{year:04d}-{_SEASONS[m.group(1).lower()]}", True
        # a spoken year with nothing finer than the year itself
        if spoken is not None and not _MONTH_NAME_RE.search(text):
            return f"{year:04d}-01-01", True

    try:
        iso = dateparser.parse(text, fuzzy=True,
                               default=_ABS_DATE_DEFAULT).date().isoformat()
    except (ValueError, OverflowError, TypeError):
        return None, has_year
    return iso, has_year


def resolve_date_entity(entity: Entity, interview_date) -> None:
    text = entity.mentions[0].text.lower()
    attrs = entity.attributes
    attrs.setdefault("shiftable", True)

    if entity.category == "DATE_ABSOLUTE" or entity.category == "DATE_OF_BIRTH":
        raw = entity.mentions[0].text
        iso, _has_year = parse_absolute_date(raw)
        if iso is None:
            entity.flag_entity("absolute date failed to parse")
            return
        attrs["resolved_value"] = iso
        # A missing day now defaults to the 1st (deterministic); but a missing
        # YEAR pulls from the default and is almost certainly wrong -- flag it.
        if not _YEAR_RE.search(raw):
            entity.flag_entity("absolute date has no explicit year; "
                               "resolved value used a default and may be wrong")
        return

    if entity.category == "DATE_ANCHOR":
        attrs["shiftable"] = False          # public events do not move
        # Match article-insensitively: a detected span often drops the leading
        # article ("Great Recession" vs the table key "the great recession"), which
        # used to break the substring match. Longest phrase first so "hurricane
        # katrina" still wins over "katrina".
        text_na = _strip_article(text)
        for phrase, iso in sorted(ANCHOR_EVENTS.items(), key=lambda kv: -len(kv[0])):
            pna = _strip_article(phrase)
            if phrase in text or pna in text_na or pna in text:
                attrs["resolved_value"] = iso
                attrs["anchor_event"] = phrase
                # A listed public event has ONE fixed calendar date, so this is the
                # rule's answer for `approximate` too. Without it every anchor was a
                # "neither layer produced a value" row -- the anchor classifier does
                # not return an approximation flag -- which is a review flag on a
                # field the table already settles.
                attrs.setdefault("approximate", False)
                return
        entity.flag_entity("anchor phrase not in ANCHOR_EVENTS table - add it")
        return

    if entity.category == "DATE_RELATIVE":
        if interview_date is None:
            entity.flag_entity("relative date but no interview_date in metadata")
            return
        attrs["approximate"] = True

        # "yesterday" / "today" / "tomorrow"
        if re.search(r"\byesterday\b", text):
            attrs["resolved_value"] = (interview_date - timedelta(days=1)).isoformat()
            return
        if re.search(r"\b(today|tonight|this morning|this afternoon)\b", text):
            attrs["resolved_value"] = interview_date.isoformat()
            return
        if re.search(r"\btomorrow\b", text):
            attrs["resolved_value"] = (interview_date + timedelta(days=1)).isoformat()
            return

        # "two years ago", "a couple of months ago", "several weeks ago"
        m = _REL_AGO.search(text)
        if m:
            days = _UNIT_DAYS[m.group(2).lower()] * _count_from_word(m.group(1))
            attrs["resolved_value"] = (interview_date - timedelta(days=days)).isoformat()
            return

        # "last year", "this past month", "next week"
        m = _REL_UNIT.search(text)
        if m:
            direction, unit = m.group(1).lower(), m.group(2).lower()
            step = _UNIT_DAYS[unit]
            offset = step if direction != "next" else -step
            attrs["resolved_value"] = (interview_date - timedelta(days=offset)).isoformat()
            return

        # "last spring", "this past summer"
        m = _REL_SEASON.search(text)
        if m:
            season_md = _SEASONS[m.group(1).lower()]
            year = interview_date.year
            candidate = dateparser.parse(f"{year}-{season_md}").date()
            if candidate >= interview_date:  # not actually in the past yet
                candidate = dateparser.parse(f"{year - 1}-{season_md}").date()
            attrs["resolved_value"] = candidate.isoformat()
            return

        # "last Tuesday" -> most recent past occurrence of that weekday
        m = _REL_WEEKDAY.search(text)
        if m:
            target = _WEEKDAYS.index(m.group(1).lower())
            delta = (interview_date.weekday() - target) % 7 or 7
            attrs["resolved_value"] = (interview_date - timedelta(days=delta)).isoformat()
            return

        entity.flag_entity("relative date pattern not recognized")
