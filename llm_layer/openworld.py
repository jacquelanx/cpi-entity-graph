"""
LLM use #2: the open-world classifier (Class B -- flagged suggestions).
The deterministic tables (gazetteer, public-figure list, public-event table,
kinship/professional word sets, date + age parsers) are small and can never
cover everything. Whenever a rule MISSES -- a list doesn't contain the value, or
a parser leaves it unresolved -- this module asks the local LLM and records the
answer as a FLAGGED SUGGESTION on the entity.

Policy (identical to the rest of the layer): suggestions are additive and never
authoritative. The LLM never overwrites a rule-set value and never lowers
redaction; it only fills a gap the rules left and raises a review flag.

Second-line coverage added here, one per rule table that could miss:
  PERSON      -- public-figure list  -> `classify_person_public`
              -- FAMILY/PROFESSIONAL subtype (kinship/professional regex)
                 -> relationship hint in the same call -> `suggested_subtype`
  LOCATION    -- gazetteer            -> `classify_location`
  DATE_ANCHOR -- public-event table   -> `resolve_public_event`
  DATE_ABSOLUTE / DATE_OF_BIRTH / DATE_RELATIVE
              -- dateutil / relative-date regex -> `resolve_date`
  AGE         -- word-number / decade maps      -> `resolve_age`
"""

from __future__ import annotations
from .llm import _windows


def _ctx(transcript, entity) -> str:
    return "\n".join(_windows(transcript, entity))


# --------------------------------------------------------------------------
# Sub-classifier 1: is an unlisted PERSON a public figure?
# --------------------------------------------------------------------------
_PERSON_SYS = (
    "You are a de-identification assistant. For a person named in an interview, "
    "decide two things. (1) Is this a WIDELY-KNOWN PUBLIC FIGURE (a politician, "
    "celebrity, athlete, or historical figure recognizable to the general "
    "public), as opposed to a private individual in the interviewee's own life? "
    "Be conservative: a regular person who merely holds a title (a local teacher, "
    "a family doctor, a neighbor) is NOT a public figure. (2) What is this "
    "person's relationship to the interviewee (the speaker)? Use \"family\" for a "
    "relative, \"professional\" for someone in a work/service/care role in the "
    "interviewee's life (boss, coworker, teacher, doctor, caseworker, landlord, "
    "sponsor...), \"acquaintance\" for a friend/neighbor, \"public\" for a public "
    "figure, or \"unknown\". Reply with ONLY a JSON object: "
    '{"public_figure": true or false, "who": "<name or role, or empty>", '
    '"relationship": "family"|"professional"|"acquaintance"|"public"|"unknown", '
    '"confidence": "high" or "low"}.'
)


def classify_person_public(client, transcript, entity) -> dict | None:
    if client is None or not client.available():
        return None
    name = entity.sorted_mentions[0] if entity.sorted_mentions else "?"
    prompt = (f'Person named "{name}" in an interview transcript.\nContexts:\n'
              + _ctx(transcript, entity)
              + "\n\nIs this a widely-known public figure, or a private individual?")
    return client.judge(prompt, system=_PERSON_SYS)


# --------------------------------------------------------------------------
# Sub-classifier 2: type/parent of a location not in the gazetteer
# --------------------------------------------------------------------------
_LOC_SYS = (
    "You classify a place mentioned in an interview. Give its geographic TYPE and "
    "the larger place it sits inside, if clear. Reply with ONLY a JSON object: "
    '{"type": one of ["country","state","region","city","neighborhood","street",'
    '"institution","landmark","other"], "parent": "<larger place, or empty>", '
    '"confidence": "high" or "low"}.'
)


def classify_location(client, transcript, entity) -> dict | None:
    if client is None or not client.available():
        return None
    name = entity.sorted_mentions[0] if entity.sorted_mentions else "?"
    prompt = (f'Place named "{name}" in an interview transcript.\nContexts:\n'
              + _ctx(transcript, entity)
              + "\n\nWhat type of place is it, and what larger place is it in?")
    return client.judge(prompt, system=_LOC_SYS)


# --------------------------------------------------------------------------
# Sub-classifier 3: fixed date of a public event not in the anchor table
# --------------------------------------------------------------------------
_EVENT_SYS = (
    "A date reference in an interview names a public 'anchor' event. If it is a "
    "well-known public event with a fixed calendar date, give that date. If you "
    "are not certain of the exact date, or it is not a public event, leave the "
    "date empty. Reply with ONLY a JSON object: "
    '{"event": "<name, or empty>", "date": "YYYY-MM-DD or empty", '
    '"confidence": "high" or "low"}.'
)


def resolve_public_event(client, transcript, entity) -> dict | None:
    if client is None or not client.available():
        return None
    phrase = entity.mentions[0].text if entity.mentions else "?"
    prompt = (f'Date/event phrase: "{phrase}"\nContexts:\n'
              + _ctx(transcript, entity)
              + "\n\nWhat public event is this and what is its fixed date?")
    return client.judge(prompt, system=_EVENT_SYS)


# --------------------------------------------------------------------------
# Sub-classifier 4: a date the rule parser could not resolve
# (absolute date that failed dateutil, or a relative expression the regex set
# didn't recognize, e.g. "a few years back").
# --------------------------------------------------------------------------
_DATE_SYS = (
    "You resolve a single date expression from an interview to a calendar date. "
    "You are given the expression and, for relative expressions, the interview "
    "date to anchor against. If you can resolve it to a specific day with "
    "confidence, give it; otherwise leave the date empty. Do NOT guess. Reply "
    'with ONLY a JSON object: {"date": "YYYY-MM-DD or empty", '
    '"approximate": true or false, "confidence": "high" or "low"}.'
)


def resolve_date(client, transcript, entity, interview_date=None) -> dict | None:
    if client is None or not client.available():
        return None
    phrase = entity.mentions[0].text if entity.mentions else "?"
    anchor = (f"The interview took place on {interview_date}. "
              if interview_date else "")
    prompt = (f'{anchor}Date expression: "{phrase}" (type {entity.category}).\n'
              "Contexts:\n" + _ctx(transcript, entity)
              + "\n\nResolve this expression to a calendar date if you can.")
    return client.judge(prompt, system=_DATE_SYS)


# --------------------------------------------------------------------------
# Sub-classifier 5: an age the rule parser could not turn into a number.
# --------------------------------------------------------------------------
_AGE_SYS = (
    "You read an age expression from an interview and return the age in whole "
    "years as an integer. For a vague expression (\"in his sixties\", "
    "\"twenty-something\") give a representative whole number and mark it "
    "approximate. If you cannot tell, leave value null. Reply with ONLY a JSON "
    'object: {"value": <integer or null>, "approximate": true or false, '
    '"confidence": "high" or "low"}.'
)


def resolve_age(client, transcript, entity) -> dict | None:
    if client is None or not client.available():
        return None
    phrase = entity.mentions[0].text if entity.mentions else "?"
    prompt = (f'Age expression: "{phrase}"\nContexts:\n'
              + _ctx(transcript, entity)
              + "\n\nWhat whole-year age does this express?")
    return client.judge(prompt, system=_AGE_SYS)


# --------------------------------------------------------------------------
# The pass: run the right sub-classifier on each LIST MISS, add flagged
# suggestions. Only acts on high-confidence answers to keep review noise down.
# --------------------------------------------------------------------------
def openworld_pass(transcript: str, entities: list, llm, interview_date=None) -> None:
    if llm is None or not llm.available():
        return

    for e in entities:
        if e.category == "PERSON":
            # skip the interviewee (no mentions) and anyone the rules already tied
            # to the interviewee's own life (family / professional). PUBLIC_FIGURE
            # still passes -- it gets the safety re-check below -- as do unset-
            # subtype persons, which get the relationship suggestion.
            if not e.mentions or e.subtype in ("FAMILY", "PROFESSIONAL"):
                continue
            v = classify_person_public(llm, transcript, e)
            if not v:
                continue
            if e.subtype == "PUBLIC_FIGURE":
                # the rules KEPT this listed name unredacted. Safety re-check: if
                # the model judges it a private individual, RAISE redaction. The LLM
                # only ever moves toward MORE redaction, never less.
                if v.get("public_figure") is False and v.get("confidence") == "high":
                    e.attributes["replace"] = True
                    e.flag_entity("LLM: rules kept this as a public figure, but "
                                  "context suggests a private individual; redacted "
                                  "for safety")
                continue
            # unlisted person: flag (for human review) if the model thinks it's a
            # public figure -- but NEVER lower redaction automatically
            if v.get("public_figure") and v.get("confidence") == "high":
                e.attributes["candidate_public_figure"] = v.get("who") or True
                e.attributes["replace"] = True     # NEVER flip to False here
                who = v.get("who") or "public figure"
                e.flag_entity(f"LLM suggests this is a public figure ({who}); "
                              f"review whether to keep it unredacted")
            # second line for the FAMILY/PROFESSIONAL subtype (kinship + professional
            # regex tables): when the rules left the subtype UNSET, record the LLM's
            # relationship read as a suggestion. Never overwrites a rule subtype.
            elif e.subtype is None and v.get("confidence") == "high":
                rel = (v.get("relationship") or "").strip().lower()
                sub = {"professional": "PROFESSIONAL", "family": "FAMILY"}.get(rel)
                if sub:
                    e.attributes["suggested_subtype"] = sub
                    e.flag_entity(f"LLM suggests this person is {rel} "
                                  f"(rule left subtype unset); review")

        elif e.category in ("LOCATION", "INSTITUTION"):
            if e.subtype:                           # already typed by the gazetteer
                continue
            v = classify_location(llm, transcript, e)
            if v and v.get("type") and v.get("confidence") == "high":
                e.attributes["suggested_type"] = v["type"]
                msg = f"LLM-suggested location type '{v['type']}'"
                if v.get("parent"):
                    e.attributes["suggested_parent"] = v["parent"]
                    msg += f", inside '{v['parent']}'"
                e.flag_entity(msg + "; review")

        elif e.category == "DATE_ANCHOR":
            if e.attributes.get("resolved_value"):  # rule already resolved it
                continue
            v = resolve_public_event(llm, transcript, e)
            if v and v.get("date") and v.get("confidence") == "high":
                e.attributes["suggested_value"] = v["date"]
                msg = f"LLM-suggested anchor date {v['date']}"
                if v.get("event"):
                    e.attributes["suggested_event"] = v["event"]
                    msg += f" ({v['event']})"
                e.flag_entity(msg + "; review")

        elif e.category in ("DATE_ABSOLUTE", "DATE_OF_BIRTH", "DATE_RELATIVE"):
            # second line for the date parsers (dateutil + relative-date regex):
            # only when the rule left it UNRESOLVED. Suggestion only, never applied
            # to resolved_value (so date-shifting won't act on an LLM guess).
            if e.attributes.get("resolved_value"):
                continue
            v = resolve_date(llm, transcript, e, interview_date)
            if v and v.get("date") and v.get("confidence") == "high":
                e.attributes["suggested_value"] = v["date"]
                if v.get("approximate"):
                    e.attributes["suggested_approximate"] = True
                e.flag_entity(f"LLM-suggested date {v['date']} "
                              f"(rule could not resolve it); review")

        elif e.category == "AGE":
            # second line for the age parser (word-number / decade maps): only when
            # the rule produced no numeric value. Suggestion only.
            if e.attributes.get("value") is not None:
                continue
            v = resolve_age(llm, transcript, e)
            if v and isinstance(v.get("value"), int) and v.get("confidence") == "high":
                e.attributes["suggested_value"] = v["value"]
                if v.get("approximate"):
                    e.attributes["suggested_approximate"] = True
                e.flag_entity(f"LLM-suggested age {v['value']} "
                              f"(rule could not parse it); review")
