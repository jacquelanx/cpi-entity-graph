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
from pathlib import Path
from dateutil import parser as dateparser
from .models import Edge, Entity, Relation


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

# Public events that can NEVER be shifted (need to be extended)
ANCHOR_EVENTS = {
    "katrina": "2005-08-29",
    "obama got elected": "2008-11-04",
    "obama was elected": "2008-11-04",
    "covid": "2020-03-01",
    "the pandemic": "2020-03-01",
    "9/11": "2001-09-11",
}

# Arbitrary...
_SEASONS = {"spring": "03-20", "summer": "06-21", "fall": "09-22",
            "autumn": "09-22", "winter": "12-21"}

_REL_AGO = re.compile(
    r"\b(a|an|one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+"
    r"(day|week|month|year)s?\s+ago\b", re.I)

_WORD_NUM = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}


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
        try:
            attrs["resolved_value"] = dateparser.parse(
                entity.mentions[0].text, fuzzy=True
            ).date().isoformat()
        except (ValueError, OverflowError):
            entity.flag_entity("absolute date failed to parse")
        return

    if entity.category == "DATE_ANCHOR":
        attrs["shiftable"] = False          # public events do not move
        for phrase, iso in ANCHOR_EVENTS.items():
            if phrase in text:
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
        m = _REL_AGO.search(text)
        if m:
            n = _WORD_NUM.get(m.group(1).lower()) or int(m.group(1))
            unit = m.group(2).lower()
            days = {"day": 1, "week": 7, "month": 30, "year": 365}[unit] * n
            from datetime import timedelta
            # calculates date
            attrs["resolved_value"] = (interview_date - timedelta(days=days)).isoformat()
            return
        m = re.search(r"last (spring|summer|fall|autumn|winter)", text)
        if m:
            season_md = _SEASONS[m.group(1)]
            year = interview_date.year
            candidate = dateparser.parse(f"{year}-{season_md}").date()
            if candidate >= interview_date:  # date is not actually in the past
                candidate = dateparser.parse(f"{year - 1}-{season_md}").date()
            attrs["resolved_value"] = candidate.isoformat()
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
This takes care of STATED_WITH edges. If an AGE and a DATE is stated in the
same sentence, they must stay arithmetically consistent after date-shifting.
IMPORTANT: sentence might not be enough... what about context that builds?
The current implementation links an AGE to every DATE appearing within
`window` sentences (default = same sentence).
"""
def age_date_constraints(
    transcript: str, entities: list[Entity], window: int = 1
) -> list[Edge]:
    
    boundaries = [0] + [m.end() for m in re.finditer(r"[.!?]", transcript)]
    sentences = list(zip(boundaries, boundaries[1:] + [len(transcript)]))

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
        for am in a.mentions:
            s_idx = sentence_of(am.start)
            for d in dates:
                for dm in d.mentions:
                    # if mentions occur within `window` sentences
                    if abs(sentence_of(dm.start) - s_idx) <= window:
                        s, e = sentences[s_idx]
                        edges.append(Edge(
                            source=a.entity_id, target=d.entity_id,
                            relation=Relation.STATED_WITH,
                            detail="age and date co-stated; keep arithmetic",
                            evidence=transcript[s:e].strip(),
                        ))
    return edges