"""
Gazetteer lookup and the LOCATED_IN hierarchy.

Resolve aliases via the gazetteer ("NOLA" -> "New Orleans"), then build
LOCATED_IN edges. When locations are replaced later, the surrogate generator
walks these edges upward/downward for consistency.

`BROAD_LOCATION_TYPES` / `AMBIGUOUS_BROAD_NAMES` are the rule layer behind the
`replace_location` decision; the checkers in `graph/checks/location.py` gate the
KEEP direction.
"""

from __future__ import annotations

import csv
from pathlib import Path
from ..models import Edge, Entity, Relation


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
                # Expose the gazetteer's parent as the RULE value for the
                # `location_parent` field, so the second line CHECKS it instead of
                # only filling gaps. Recorded even when the parent isn't a mention in
                # this transcript (so no edge is built): an LLM parent that
                # contradicts a known gazetteer parent is then a conflict rather than
                # an unchallenged fill -- which is what let "Columbus -> West
                # Virginia" and "Charleston -> West Virginia" through.
                if rec["parent"]:
                    e.attributes.setdefault(
                        "location_parent", gazetteer[rec["parent"]]["name"]
                        if rec["parent"] in gazetteer else rec["parent"])
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


# Geographic granularity coarse enough that keeping it cannot single out a
# household: a country, a state/province/district, or a region/county/parish.
# Everything at CITY level or finer -- and every INSTITUTION, and every place the
# gazetteer could not type -- is replaced. Deliberately keyed on the RAW type
# rather than the canonical bucket in `checks/comparators.LOC_CANON`: that bucket
# folds "island" and "metro" into "region", and an island or a metro area can be
# small enough to identify one family.
BROAD_LOCATION_TYPES = {
    "country", "territory",
    "state", "province", "district",
    "region", "county", "parish",
}


# Names the gazetteer types BROAD but that, said aloud, just as often mean a
# CITY-level place. The gazetteer is keyed on a bare lowercase name, so it answers
# with whichever sense it happens to hold, and "keep it, the table says it is a
# state" is then unfalsifiable: `checks/location.type_corroborated` confirms the type
# against the same table that produced it, so the two agree by construction.
#
# Verified: interview_002's "Every light that ever came on in Charleston or in
# Washington" means Washington D.C., and the row above types "washington" as a state,
# so the name was KEPT -- the one place leak in the sample transcripts.
#
# Membership forces the safe direction (replace) rather than asserting a type, so the
# cost of a name being here is over-redacting a genuine region reference, and the
# benefit is that an ambiguous name can never be kept on a table collision. Keep it to
# names where BOTH senses are common in ordinary speech.
AMBIGUOUS_BROAD_NAMES = {
    "washington",       # state / D.C.
    "new york",         # state / city
    "georgia",          # US state / country
    "mexico",           # country / Mexico City, and Mexico, Missouri
    "kansas",           # state / Kansas City
    "panama",           # country / Panama City
    "quebec",           # province / Quebec City
    "luxembourg",       # country / city
    "singapore", "monaco", "san marino",   # country == city
}


def infer_location_replace(entities: list[Entity]) -> None:
    """RULE layer for LOCATION / INSTITUTION `replace`.

    Nothing used to decide this at all: location entities reached surrogate
    generation with no `replace` key, so a consumer keying off `replace` skipped
    every place name while redacting every person -- and a hamlet plus an age plus
    an occupation is exactly how an interviewee gets re-identified.

    The rule is the gazetteer's own granularity. A place typed country / state /
    region is kept; a city or anything finer is replaced; an INSTITUTION (a named
    organisation, church, school, employer) is always replaced; and a place the
    gazetteer could not type at all is replaced, because an unknown place is far
    more likely to be a small local one than a country.

    Over-redaction is the safe error here, so the default is always `True`. The
    LLM's own identifying-ness judgment second-lines this in
    `graph.second_line` under `replace_location`, with `conflict_policy=
    safe_direction` -- so a disagreement always resolves toward more redaction --
    and the checkers in `graph/checks/location.py` gate the KEEP direction.
    """
    for e in entities:
        if e.category not in ("LOCATION", "INSTITUTION"):
            continue
        if e.category == "INSTITUTION":
            e.attributes.setdefault("replace", True)
            continue
        raw = (e.subtype or "").strip().lower()
        e.attributes.setdefault("replace", raw not in BROAD_LOCATION_TYPES)
