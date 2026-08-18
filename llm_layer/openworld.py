"""
The open-world PROPOSER. The deterministic tables (gazetteer, public-figure list,
public-event table, kinship/professional word sets, date + age parsers) are small
and can never cover everything. This module asks the local LLM the same questions
the tables answer, for EVERY entity, and returns the answers as plain data:

    {entity_id: {field: {"value": ..., "confidence": ..., <extras>}}}

It decides nothing and mutates nothing. `graph.second_line.resolve_all` arbitrates:
a field the rules filled gets CHECKED, a field they left empty gets FILLED only if
every deterministic checker in `graph/checks/` passes.

Two behaviours changed when this became a proposer, both deliberate:

  * every field is proposed UNCONDITIONALLY. The old pass skipped locations the
    gazetteer had typed (`if e.subtype: continue`) and people the rules had typed
    FAMILY/PROFESSIONAL, so on those the rule tables were the sole authority and a
    wrong table hit was unfalsifiable.
  * the public-figure CO-SIGN is no longer implemented here. It is now the
    `replace` field's `conflict_policy=safe_direction` plus its checkers, which
    generalizes the same guarantee: agreement is required to KEEP a name, and any
    disagreement resolves toward more redaction.

Fields proposed, and the rule table each one second-lines:
  PERSON       replace          <- PUBLIC_FIGURES + `_personal_signal`
               subtype_person   <- kinship edges / PROFESSIONAL_CONTEXT
  LOCATION     subtype_location <- gazetteer type
               location_parent  <- gazetteer parent (-> LOCATED_IN)
  DATE_ANCHOR  resolved_value   <- ANCHOR_EVENTS
               shiftable        <- the hardcoded per-category default
  DATE_*       resolved_value   <- dateutil / relative-date regex
               approximate      <- the rule's own approximation marker
  AGE          value            <- word-number / decade maps
               approximate      <- ditto
"""

from __future__ import annotations
from .llm import _windows

# Per-category context sizing. A judgment's window is sized to how NON-LOCAL its
# evidence is: whether someone is a public figure (vs a private namesake) often
# turns on cues far from the name, so it gets the widest read; a place's type is
# usually stated right beside it; a date/age expression is self-contained, so extra
# context there is cost without benefit.
_RADIUS = {"person": 450, "location": 250, "date": 180, "age": 180}
_MAX_SNIPS = {"person": 5, "location": 3, "date": 3, "age": 3}


def _ctx(transcript, entity, kind="date") -> str:
    return "\n".join(_windows(transcript, entity,
                              radius=_RADIUS[kind], max_snips=_MAX_SNIPS[kind]))


def _conf_str(confidence) -> str:
    """Normalize the model's confidence to a short tag for review triage. Review-only
    suggestions are accepted at ANY confidence (they never leak or auto-act); the tag
    lets a human sort high- from low-confidence flags."""
    c = str(confidence or "").strip().lower()
    return c or "unstated"


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
              + _ctx(transcript, entity, "person")
              + "\n\nIs this a widely-known public figure, or a private individual?")
    return client.judge(prompt, system=_PERSON_SYS)


# --------------------------------------------------------------------------
# Sub-classifier 2: type/parent of a location not in the gazetteer
# --------------------------------------------------------------------------
_LOC_SYS = (
    "You classify a place mentioned in an interview. Give its geographic TYPE, "
    "the larger place it sits inside if clear, and whether the place is specific "
    "enough that naming it would help identify the particular family or person "
    "being interviewed. A country, state or large region identifies nobody; a "
    "small town, a village, a holler, a neighborhood, a street, a named church, "
    "school or employer often identifies a household. Reply with ONLY a JSON "
    'object: {"type": one of ["country","state","region","city","neighborhood",'
    '"street","institution","landmark","other"], "parent": "<larger place, or '
    'empty>", "identifying": true or false, "confidence": "high" or "low"}.'
)


def classify_location(client, transcript, entity) -> dict | None:
    if client is None or not client.available():
        return None
    name = entity.sorted_mentions[0] if entity.sorted_mentions else "?"
    prompt = (f'Place named "{name}" in an interview transcript.\nContexts:\n'
              + _ctx(transcript, entity, "location")
              + "\n\nWhat type of place is it, and what larger place is it in?")
    return client.judge(prompt, system=_LOC_SYS)


# --------------------------------------------------------------------------
# Sub-classifier 3: fixed date of a public event not in the anchor table
# --------------------------------------------------------------------------
# `approximate` is asked for here as well as on the other date categories. It was
# missing, and `second_line._fields_for` resolves `approximate` for EVERY date
# category -- so for any anchor phrase outside `ANCHOR_EVENTS` the rule set nothing
# and the model was never asked, making it the one field in the registry with no
# second layer at all. It rendered as clean because `llm_report` drops
# both-layers-blind rows.
_EVENT_SYS = (
    "A date reference in an interview names a public 'anchor' event. If it is a "
    "well-known public event with a fixed calendar date, give that date. If you "
    "are not certain of the exact date, or it is not a public event, leave the "
    "date empty. Also say whether the reference is APPROXIMATE: a named public "
    "event has one fixed calendar date and is NOT approximate, while a vague or "
    "hedged reference to a period is. Reply with ONLY a JSON object: "
    '{"event": "<name, or empty>", "date": "YYYY-MM-DD or empty", '
    '"approximate": true or false, "confidence": "high" or "low"}.'
)


def resolve_public_event(client, transcript, entity) -> dict | None:
    if client is None or not client.available():
        return None
    phrase = entity.mentions[0].text if entity.mentions else "?"
    prompt = (f'Date/event phrase: "{phrase}"\nContexts:\n'
              + _ctx(transcript, entity, "date")
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
    "confidence, give it; otherwise leave the date empty. Do NOT guess. Also say "
    "whether the expression refers to a WELL-KNOWN PUBLIC EVENT with a fixed "
    "calendar date (a hurricane, an election, an attack, a disaster) as opposed to "
    "a private date in this family's life; give the event's name if so, and leave "
    "it empty otherwise. Reply with ONLY a JSON object: "
    '{"date": "YYYY-MM-DD or empty", "approximate": true or false, '
    '"public_event": "<event name, or empty>", "confidence": "high" or "low"}.'
)


def resolve_date(client, transcript, entity, interview_date=None) -> dict | None:
    if client is None or not client.available():
        return None
    phrase = entity.mentions[0].text if entity.mentions else "?"
    anchor = (f"The interview took place on {interview_date}. "
              if interview_date else "")
    prompt = (f'{anchor}Date expression: "{phrase}" (type {entity.category}).\n'
              "Contexts:\n" + _ctx(transcript, entity, "date")
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
              + _ctx(transcript, entity, "age")
              + "\n\nWhat whole-year age does this express?")
    return client.judge(prompt, system=_AGE_SYS)


# --------------------------------------------------------------------------
# Sub-classifier 6: which date expression is an age stated WITH?
# Second line for `location_dates.age_date_constraints`, whose rule is purely
# positional (the single nearest date within one sentence). An age and the date it
# was co-stated with must stay arithmetically consistent after date-shifting, so a
# wrong pairing silently corrupts the shift.
# --------------------------------------------------------------------------
_ANCHOR_SYS = (
    "An interview mentions someone's AGE. If the surrounding text also states the "
    "date or year that age was true (\"I enlisted in September 2008. I was "
    "eighteen\"), quote that date expression EXACTLY as it appears in the text. If "
    "no date in this passage pins the age, leave it empty. Do NOT guess and do NOT "
    'invent a date. Reply with ONLY a JSON object: {"date_text": "<exact quote or '
    'empty>", "confidence": "high" or "low"}.'
)


def resolve_age_anchor(client, transcript, entity) -> dict | None:
    if client is None or not client.available():
        return None
    phrase = entity.mentions[0].text if entity.mentions else "?"
    prompt = (f'Age expression: "{phrase}"\nContexts:\n'
              + _ctx(transcript, entity, "date")
              + "\n\nWhich date expression in this passage is that age stated with?")
    return client.judge(prompt, system=_ANCHOR_SYS)


# --------------------------------------------------------------------------
# The PROPOSER. This module no longer mutates entities and no longer decides
# anything: it runs the right sub-classifier per entity and returns plain dicts
#
#     {entity_id: {field: {"value": ..., "confidence": ..., <extras>}}}
#
# for `graph.second_line.resolve_all` to arbitrate. Plain dicts keep the one-way
# dependency (graph -> llm_layer) intact.
#
# Every field is proposed UNCONDITIONALLY, whether or not the rule already filled
# it. That is the point of the unification: a filled field gets CHECKED, an empty
# one gets FILLED, and the old `if e.subtype: continue` short-circuit -- which
# meant a gazetteer HIT was never double-checked -- is gone.
# --------------------------------------------------------------------------
def openworld_propose(transcript: str, entities: list, llm, interview_date=None) -> dict:
    out: dict = {}
    if llm is None or not llm.available():
        return out

    def put(e, field, value, confidence=None, **extra):
        if value is None or value == "":
            return
        out.setdefault(e.entity_id, {})[field] = {
            "value": value, "confidence": _conf_str(confidence), **extra}

    for e in entities:
        if e.category == "PERSON":
            # The interviewee is never a public-figure candidate and never gets a
            # FAMILY/PROFESSIONAL subtype -- it is the subject of the interview. The
            # `role` marker rather than an import keeps this module free of any
            # dependency on `graph`; the mention check alone stopped working once the
            # identification stage started giving the interviewee real name spans.
            if e.attributes.get("role") == "interviewee" or not e.mentions:
                continue
            v = classify_person_public(llm, transcript, e)
            if not v:
                continue
            conf = v.get("confidence")
            is_public = v.get("public_figure") is True
            # `replace` is proposed for EVERY named person, including ones the rules
            # already typed FAMILY/PROFESSIONAL -- previously those were skipped, so
            # the rules were the sole authority on them.
            put(e, "replace", not is_public, conf, who=v.get("who") or "")
            rel = (v.get("relationship") or "").strip().lower()
            sub = {"professional": "PROFESSIONAL", "family": "FAMILY"}.get(rel)
            if sub:
                put(e, "subtype_person", sub, conf)

        elif e.category in ("LOCATION", "INSTITUTION"):
            v = classify_location(llm, transcript, e)
            if not v:
                continue
            put(e, "subtype_location", (v.get("type") or "").strip(),
                v.get("confidence"))
            put(e, "location_parent", (v.get("parent") or "").strip(),
                v.get("confidence"))
            # `replace_location`: the model's identifying-ness judgment, second-lined
            # against the gazetteer granularity rule in
            # `location_dates.infer_location_replace`. `put` skips falsy values, so
            # the boolean is passed explicitly.
            if isinstance(v.get("identifying"), bool):
                put(e, "replace_location", bool(v["identifying"]),
                    v.get("confidence"))

        elif e.category == "DATE_ANCHOR":
            v = resolve_public_event(llm, transcript, e)
            if not v:
                continue
            put(e, "resolved_value", v.get("date"), v.get("confidence"),
                event=v.get("event") or "")
            # a named event with a date is the model's claim that this is a real
            # public event -> propose shiftable=False; otherwise propose nothing and
            # let the rule's own value stand or be flagged.
            if v.get("date") and v.get("event"):
                put(e, "shiftable", False, v.get("confidence"))
            # `approximate` for an anchor. Taken from the model when it answered, and
            # otherwise DERIVED: a named event with a fixed calendar date is exact by
            # construction, which is the same reasoning
            # `location_dates.resolve_date_entity` applies to a phrase it finds in
            # ANCHOR_EVENTS. `checks/approximate.anchor_in_table_is_exact` gates it.
            if v.get("approximate") is not None:
                put(e, "approximate", bool(v["approximate"]), v.get("confidence"))
            elif v.get("date") and v.get("event"):
                put(e, "approximate", False, v.get("confidence"))

        elif e.category in ("DATE_ABSOLUTE", "DATE_OF_BIRTH", "DATE_RELATIVE"):
            v = resolve_date(llm, transcript, e, interview_date)
            if not v:
                continue
            event = (v.get("public_event") or "").strip()
            put(e, "resolved_value", v.get("date"), v.get("confidence"),
                event=event)
            if v.get("approximate") is not None:
                put(e, "approximate", bool(v["approximate"]), v.get("confidence"))
            # `shiftable` is now second-lined for EVERY date category, not just
            # DATE_ANCHOR -- the rule sets it on all four. A private date is
            # shiftable; a named public event is the model's claim that it is not,
            # and `checks/dates.is_real_public_event` refuses that claim outright
            # for a non-anchor category.
            put(e, "shiftable", not bool(event), v.get("confidence"))

        elif e.category == "AGE":
            v = resolve_age(llm, transcript, e)
            if v:
                if isinstance(v.get("value"), int):
                    put(e, "value", v["value"], v.get("confidence"))
                if v.get("approximate") is not None:
                    put(e, "approximate", bool(v["approximate"]), v.get("confidence"))
            a = resolve_age_anchor(llm, transcript, e)
            if a:
                put(e, "stated_with", (a.get("date_text") or "").strip(),
                    a.get("confidence"))

    return out
