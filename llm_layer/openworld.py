"""
LLM use #2: the open-world classifier (Class B -- flagged suggestions).
The deterministic lists (gazetteer, public-figure list, public-event table) are
small and can never cover everything. When one of them MISSES on an entity, this
module asks the local LLM to classify it and records the answer as a FLAGGED
SUGGESTION on the entity.
"""

from __future__ import annotations
from .llm import _windows


def _ctx(transcript, entity) -> str:
    return "\n".join(_windows(transcript, entity))


# --------------------------------------------------------------------------
# Sub-classifier 1: is an unlisted PERSON a public figure?
# --------------------------------------------------------------------------
_PERSON_SYS = (
    "You are a de-identification assistant. Decide whether a person named in an "
    "interview is a WIDELY-KNOWN PUBLIC FIGURE (a politician, celebrity, athlete, "
    "or historical figure recognizable to the general public), as opposed to a "
    "private individual in the interviewee's own life. Be conservative: a regular "
    "person who merely holds a title (a local teacher, a family doctor, a "
    "neighbor) is NOT a public figure. Reply with ONLY a JSON object: "
    '{"public_figure": true or false, "who": "<name or role, or empty>", '
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
# The pass: run the right sub-classifier on each LIST MISS, add flagged
# suggestions. Only acts on high-confidence answers to keep review noise down.
# --------------------------------------------------------------------------
def openworld_pass(transcript: str, entities: list, llm) -> None:
    if llm is None or not llm.available():
        return

    for e in entities:
        if e.category == "PERSON":
            # skip the interviewee (no mentions) and anyone the rules already
            # tied to the interviewee's own life (family / professional)
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
