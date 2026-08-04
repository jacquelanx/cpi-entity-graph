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
                 (CO-SIGN: a name the rules KEPT unredacted on the strength of the
                 closed PUBLIC_FIGURES list is kept only if the LLM ALSO affirms it
                 is a public figure; otherwise redaction is raised. The closed list
                 is never the sole authority for keeping a name unredacted.)
              -- FAMILY/PROFESSIONAL subtype (kinship/professional regex)
                 -> relationship hint in the same call -> `suggested_subtype`
  LOCATION    -- gazetteer            -> `classify_location`
  DATE_ANCHOR -- public-event table   -> `resolve_public_event`
  DATE_ABSOLUTE / DATE_OF_BIRTH / DATE_RELATIVE
              -- dateutil / relative-date regex -> `resolve_date`
  AGE         -- word-number / decade maps      -> `resolve_age`

For DATE_* and AGE the second line now does BOTH jobs: it CHECKS a value the rule
resolved (flagging a meaningful divergence for review, never overwriting it) and
FILLS a value the rule missed (`suggested_value`). Only the deterministic value is
ever used downstream (e.g. date-shifting); the LLM's is advisory.
"""

from __future__ import annotations
from datetime import date as _date
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


def _parse_iso(s):
    """Lenient ISO-date parse (both rule values and the LLM's dates are ISO)."""
    try:
        return _date.fromisoformat(str(s)[:10])
    except Exception:
        return None


def _conf_str(confidence) -> str:
    """Normalize the model's confidence to a short tag for review triage. Review-only
    suggestions are accepted at ANY confidence (they never leak or auto-act); the tag
    lets a human sort high- from low-confidence flags."""
    c = str(confidence or "").strip().lower()
    return c or "unstated"


# --------------------------------------------------------------------------
# Reconcilers: the date/age second line now both CHECKS a rule-resolved value
# and FILLS a rule miss.
#   * rule value UNSET  -> FILL: record `suggested_value` (+ approximate) as before.
#   * rule value SET    -> CHECK: never overwrite it (date-shifting must act only on
#     the deterministic value). Agreement records `*_confirmed`; a divergence raises
#     a CONFLICT flag, but only when it is meaningful -- the model is confident OR
#     the gap is large -- so an unsure model disagreeing by a day doesn't spam review.
# --------------------------------------------------------------------------
def _reconcile_date(e, llm_date, approximate, confidence, tol_days, large_days,
                    event=None):
    if not llm_date:
        return
    rule_val = e.attributes.get("resolved_value")
    if rule_val is None:                                    # FILL (rule missed)
        conf = _conf_str(confidence)                        # any confidence; tagged
        e.attributes["suggested_value"] = llm_date
        e.attributes["suggested_value_confidence"] = conf
        if approximate:
            e.attributes["suggested_approximate"] = True
        msg = f"LLM-suggested date {llm_date} (rule could not resolve it; confidence {conf})"
        if event:
            e.attributes["suggested_event"] = event
            msg += f" [{event}]"
        e.flag_entity(msg + "; review")
        return
    rd, ld = _parse_iso(rule_val), _parse_iso(llm_date)     # CHECK (rule resolved)
    if rd is None or ld is None:
        return
    diff = abs((rd - ld).days)
    if diff <= tol_days:
        e.attributes["date_confirmed"] = True
        return
    if confidence == "high" or diff > large_days:
        e.attributes["llm_check_value"] = llm_date
        e.flag_entity(f"LLM resolved this date to {llm_date}, differs from the rule "
                      f"value {rule_val} by {diff} days; rule value kept -- review")


def _reconcile_age(e, llm_val, approximate, confidence, tol, large):
    if not isinstance(llm_val, int):
        return
    rule_val = e.attributes.get("value")
    if rule_val is None:                                    # FILL (rule missed)
        conf = _conf_str(confidence)                        # any confidence; tagged
        e.attributes["suggested_value"] = llm_val
        e.attributes["suggested_value_confidence"] = conf
        if approximate:
            e.attributes["suggested_approximate"] = True
        e.flag_entity(f"LLM-suggested age {llm_val} "
                      f"(rule could not parse it; confidence {conf}); review")
        return
    diff = abs(rule_val - llm_val)                          # CHECK (rule resolved)
    if diff <= tol:
        e.attributes["age_confirmed"] = True
        return
    if confidence == "high" or diff > large:
        e.attributes["llm_check_value"] = llm_val
        e.flag_entity(f"LLM read this age as {llm_val}, differs from the rule value "
                      f"{rule_val} by {diff}; rule value kept -- review")


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
              + _ctx(transcript, entity, "location")
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
# The pass: run the right sub-classifier per entity and add flagged suggestions.
# Review-only suggestions (public-figure candidate, subtype, location type, and the
# date/age FILL of a rule miss) are accepted at ANY confidence -- they can never leak
# or auto-act -- and are tagged with the model's confidence for triage. The one place
# a confidence bar remains is the date/age CHECK path (_reconcile_*), which only
# ACCUSES a resolved rule value of being wrong when the model is confident OR the gap
# is large, so review isn't spammed by an unsure model.
# --------------------------------------------------------------------------
def openworld_pass(transcript: str, entities: list, llm, interview_date=None) -> None:
    if llm is None or not llm.available():
        return

    for e in entities:
        if e.category == "PERSON":
            # skip the interviewee (no mentions) and anyone the rules already tied
            # to the interviewee's own life (family / professional -- already
            # replace=True). PUBLIC_FIGURE still passes -- it gets the CO-SIGN below
            # -- as do unset-subtype persons, which get the relationship suggestion.
            if not e.mentions or e.subtype in ("FAMILY", "PROFESSIONAL"):
                continue
            v = classify_person_public(llm, transcript, e)
            if not v:
                continue                                # no LLM answer -> rules stand
            if e.subtype == "PUBLIC_FIGURE":
                # CO-SIGN. The rules KEPT this listed name unredacted on the strength
                # of the closed PUBLIC_FIGURES list. That list must NOT be the sole
                # authority for KEEPING a name, because keeping is the leak-prone
                # direction (a private namesake of a celebrity would leak). Require
                # the LLM to AFFIRM the name is a public figure; if it does not --
                # says private, names a personal role, or is simply unsure -- raise
                # redaction. Confidence is deliberately NOT required here: moving
                # toward MORE redaction is always the safe direction, so any answer
                # short of an explicit "public figure" tips us back to redacting.
                if v.get("public_figure") is True:
                    e.attributes["public_figure_cosign"] = True
                else:
                    e.attributes["replace"] = True
                    e.attributes["public_figure_cosign"] = False
                    who = v.get("who") or "context suggests a private individual"
                    e.flag_entity("LLM did not confirm this listed name as a public "
                                  f"figure ({who}); redacted for safety -- the "
                                  "public-figure list alone is not sufficient to "
                                  "keep a name unredacted")
                continue
            # unlisted person: flag (for human review) if the model thinks it's a
            # public figure -- but NEVER lower redaction automatically. Accepted at
            # ANY confidence (a review-only flag that only ever KEEPS redaction), and
            # tagged so a reviewer can triage high- vs low-confidence flags.
            if v.get("public_figure"):
                conf = _conf_str(v.get("confidence"))
                e.attributes["candidate_public_figure"] = v.get("who") or True
                e.attributes["candidate_public_figure_confidence"] = conf
                e.attributes["replace"] = True     # NEVER flip to False here
                who = v.get("who") or "public figure"
                e.flag_entity(f"LLM suggests this is a public figure ({who}; confidence "
                              f"{conf}); review whether to keep it unredacted")
            # second line for the FAMILY/PROFESSIONAL subtype (kinship + professional
            # regex tables): when the rules left the subtype UNSET, record the LLM's
            # relationship read as a suggestion (any confidence, tagged). Never
            # overwrites a rule subtype.
            elif e.subtype is None:
                rel = (v.get("relationship") or "").strip().lower()
                sub = {"professional": "PROFESSIONAL", "family": "FAMILY"}.get(rel)
                if sub:
                    conf = _conf_str(v.get("confidence"))
                    e.attributes["suggested_subtype"] = sub
                    e.attributes["suggested_subtype_confidence"] = conf
                    e.flag_entity(f"LLM suggests this person is {rel} (rule left "
                                  f"subtype unset; confidence {conf}); review")

        elif e.category in ("LOCATION", "INSTITUTION"):
            if e.subtype:                           # already typed by the gazetteer
                continue
            v = classify_location(llm, transcript, e)
            if v and v.get("type"):
                conf = _conf_str(v.get("confidence"))       # any confidence; tagged
                e.attributes["suggested_type"] = v["type"]
                e.attributes["suggested_type_confidence"] = conf
                msg = f"LLM-suggested location type '{v['type']}'"
                if v.get("parent"):
                    e.attributes["suggested_parent"] = v["parent"]
                    msg += f", inside '{v['parent']}'"
                e.flag_entity(f"{msg} (confidence {conf}); review")

        elif e.category == "DATE_ANCHOR":
            # second line for the ANCHOR_EVENTS table: CHECK a resolved anchor date
            # against the model (a wrong/colliding table entry gets flagged) and
            # FILL an event the table doesn't list. Public events are exact, so the
            # agreement tolerance is tight.
            v = resolve_public_event(llm, transcript, e)
            if v:
                _reconcile_date(e, v.get("date"), False, v.get("confidence"),
                                tol_days=3, large_days=30, event=v.get("event"))

        elif e.category in ("DATE_ABSOLUTE", "DATE_OF_BIRTH", "DATE_RELATIVE"):
            # second line for the date parsers (dateutil + relative-date regex):
            # CHECK the rule's resolved value and FILL a miss. A CHECK never touches
            # resolved_value, so date-shifting still acts only on the rule value.
            # Absolute/DOB dates allow a month of slack (year-less / day-less inputs);
            # relative dates are estimates, so more.
            v = resolve_date(llm, transcript, e, interview_date)
            if v:
                tol, large = ((60, 400) if e.category == "DATE_RELATIVE"
                              else (31, 366))
                _reconcile_date(e, v.get("date"), v.get("approximate"),
                                v.get("confidence"), tol_days=tol, large_days=large)

        elif e.category == "AGE":
            # second line for the age parser (word-number / decade maps): CHECK the
            # rule's numeric value and FILL a miss. One year of slack absorbs
            # decade-rounding ("late forties" -> 48 vs 47).
            v = resolve_age(llm, transcript, e)
            if v:
                _reconcile_age(e, v.get("value"), v.get("approximate"),
                               v.get("confidence"), tol=1, large=5)
