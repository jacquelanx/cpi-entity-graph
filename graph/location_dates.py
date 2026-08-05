"""
Normalization and hierarchy assignment for locations and dates.

Locations: resolve aliases via the gazetteer ("NOLA" -> "New Orleans"), then
build LOCATED_IN edges. When locations are replaced later, walk these edges
upward/downward for consistency within the surrogate generator. 

Dates: we put every date mention on the timeline.
DATE_ABSOLUTE  --> parse directly (dateutil).
DATE_RELATIVE  --> resolve against the interview_date from metadata.
DATE_ANCHOR    --> match against a public-events table; public events can't move
so they CONSTRAIN the date-shift offset. 

Ages: convert words to int, and record STATED_WITH edges when an age and a
date are stated in the same sentence ("I was nineteen in 2009"); again this is
for internal consistency. 
"""


from __future__ import annotations
import csv
import re
from datetime import timedelta, datetime
from pathlib import Path
from dateutil import parser as dateparser
from .models import Edge, Entity, Relation
from .sentences import sentence_spans


"""
gazetteer.csv columns: name, type, parent, alises (| separated)
Returns (records_by_canonical_name, alias->canonical map).
"""
def load_gazetteer(path: str | Path):
    records: dict[str, dict] = {}
    aliases: dict[str, str] = {}
    path = Path(path)
    if not path.exists():
        return records, aliases
    
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row["name"].strip().lower()
            records[name] = {
                "name": row["name"].strip(),
                "type": row["type"].strip().lower(),
                "parent": row["parent"].strip().lower(),
            }
            for a in (row.get("aliases") or "").split("|"):
                if a.strip():
                    aliases[a.strip().lower()] = name
    return records, aliases


"""
Creates LOCATED_IN edges between location/institution entities per gazetteer.
"""
def build_location_edges(
    entities: list[Entity],
    gazetteer: dict[str, dict],
    aliases: dict[str, str] | None = None,
) -> list[Edge]:
    aliases = aliases or {}
    by_name = {}
    for e in entities:
        if e.category in ("LOCATION", "INSTITUTION"):
            key = e.sorted_mentions[-1].lower() if e.sorted_mentions else ""
            for form in e.sorted_mentions:
                candidate = aliases.get(form.lower(), form.lower())
                if candidate in gazetteer:
                    key = candidate
                    break
            by_name[key] = e
            rec = gazetteer.get(key)
            if rec:
                e.subtype = rec["type"].upper()
    # now we have a lookup table that resolves the key to an alias in the
    # gazetteer (if it exists), else just uses the default key

    edges = []
    for key, e in by_name.items():  # iterate through lookup table
        rec = gazetteer.get(key)
        if rec and rec["parent"] and rec["parent"] in by_name:  # rec AND parent must be Entities
            edges.append(
                Edge(
                    source=e.entity_id,
                    target=by_name[rec["parent"]].entity_id,
                    relation=Relation.LOCATED_IN,
                    detail=f"{rec['type']} in {gazetteer[rec['parent']]['type']}",
                    evidence="gazetteer",
                )
            )
    return edges


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
def resolve_date_entity(entity: Entity, interview_date) -> None:
    text = entity.mentions[0].text.lower()
    attrs = entity.attributes
    attrs.setdefault("shiftable", True)

    if entity.category == "DATE_ABSOLUTE" or entity.category == "DATE_OF_BIRTH":
        raw = entity.mentions[0].text
        try:
            attrs["resolved_value"] = dateparser.parse(
                raw, fuzzy=True, default=_ABS_DATE_DEFAULT
            ).date().isoformat()
        except (ValueError, OverflowError):
            entity.flag_entity("absolute date failed to parse")
            return
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


"""
Used for AGE entities; we want a value to store in `entity.attributes`.
The function first looks for a numeric age (e.g. "42 years old"). If no
digits are present, it parses simple spelled-out ages (e.g. "ten",
"twenty-three", "thirty seven"). On success, the integer is stored as:
entity.attributes["value"]
"""
def resolve_age_entity(entity: Entity) -> None:
    text = entity.mentions[0].text.lower()
    m = re.search(r"\d{1,3}", text)
    if m:
        entity.attributes["value"] = int(m.group(0))
        return
    
    # spelled-out ages: "twenty-three", "thirty-seven", "ten"
    ones = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
            "seven": 7, "eight": 8, "nine": 9}
    teens = {"ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
             "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
             "eighteen": 18, "nineteen": 19}
    tens = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
            "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}

    # Decade expressions: "in my twenties", "mid-thirties", "late forties".
    # These are approximate; we store a representative age within the decade.
    _DECADES = {"twenties": 20, "thirties": 30, "forties": 40, "fifties": 50,
                "sixties": 60, "seventies": 70, "eighties": 80, "nineties": 90}
    dec = re.search(r"(early|mid|middle|late)?[\s\-]*(" +
                    "|".join(_DECADES) + r")\b", text)
    if dec:
        bump = {"early": 2, "mid": 5, "middle": 5, "late": 8}.get(dec.group(1) or "", 5)
        entity.attributes["value"] = _DECADES[dec.group(2)] + bump
        entity.attributes["approximate"] = True
        return

    # "twenty-something", "thirty-something"
    som = re.search(r"(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)"
                    r"[\s\-]*something", text)
    if som:
        entity.attributes["value"] = tens[som.group(1)] + 5
        entity.attributes["approximate"] = True
        return

    total, found = 0, False
    for tok in re.split(r"[\s\-]+", text):
        if tok in teens:
            total += teens[tok]; found = True
        elif tok in tens:
            total += tens[tok]; found = True
        elif tok in ones:
            total += ones[tok]; found = True
    if found:
        entity.attributes["value"] = total
    else:
        entity.flag_entity("could not parse age value")


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