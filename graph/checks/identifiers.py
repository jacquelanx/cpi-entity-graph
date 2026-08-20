"""
Deterministic checkers for identifier `kind`, the normalized sub-attributes
(`digits`, `local`, `domain`, `handle`, `occupation`), and `identifying`.

PURPOSE
    Verify what the LLM says a span IS ("that's a phone number") and whether a
    job title is rare enough to single somebody out (`identifying`). Also
    regenerate the derived sub-attributes so an accepted `kind` actually changes
    the entity.

FIT
    Named by the `kind` and `identifying` policies in
    `graph/second_line/policies.py`; the proposer is
    `llm_layer/identifier_judge.py`. `canon_kind` is also imported by
    `checks/comparators.id_kind`, so the two vocabularies stay reconciled in one
    place.

HOW -- the rule parser IS the checker
    Rather than describing what a phone number looks like a second time, the
    checker re-runs `rules/identifiers._normalize` under the CLAIMED category and
    fails if that parser rejects the span. AGE and DATE_OF_BIRTH -- which
    `_normalize` does not own -- are verified by `rules/ages.parse_age_value` and
    `rules/dates.parse_absolute_date` instead. One parser per kind of value, used
    by both layers.

The rule layer's own parsers ARE the checkers: if the model claims a span is a
phone, the span must normalize as a phone. Re-running `identifiers._normalize`
under the claimed category is therefore a complete, deterministic verification.

Three gaps this module used to have, all closed here:

  * the vocabulary was smaller than the proposer's. `llm_layer/identifier_judge`
    proposes `kind` for AGE and DATE_OF_BIRTH spans too (their owner is a
    second-lined field, so they ride in the same pass), but `KIND_TO_CAT` knew
    nothing about "age" or "dob" -- so those proposals were dropped, and would
    have been refuted by `kind_is_known` if they hadn't been. Both are now first
    class, verified by the rule parsers in `graph/rules/dates.py and graph/rules/ages.py`.
  * `kind` was compared across two different vocabularies. The RULE value is an
    entity CATEGORY (`SSN_OR_ID`), the LLM value is a lowercase word (`ssn`).
    `canon_kind` maps both onto one category so a checker can run on either.
  * `identifying` had no rule source and no checker at all.
"""

from __future__ import annotations
from . import CheckOutcome, ok, fail, na
from ..rules.identifiers import _normalize, COMMON_OCCUPATIONS, _is_common_occupation

# The model's `kind` word -> the category this pipeline emits.
KIND_TO_CAT = {"phone": "PHONE", "email": "EMAIL", "ssn": "SSN_OR_ID",
               "id": "SSN_OR_ID", "handle": "USERNAME_HANDLE",
               "occupation": "OCCUPATION", "age": "AGE", "dob": "DATE_OF_BIRTH"}

# Categories `_normalize` (the identifier rule layer) can re-type. AGE and
# DATE_OF_BIRTH are verified by their own rule parsers instead.
_NORMALIZABLE = ("PHONE", "EMAIL", "SSN_OR_ID", "USERNAME_HANDLE", "OCCUPATION")


def canon_kind(value) -> str | None:
    """The category a `kind` value denotes, whether it arrived as the model's word
    ('ssn') or as an entity category ('SSN_OR_ID'). None when unrecognized.

    The two layers speak different vocabularies -- the rule side stores an entity
    CATEGORY, the model answers with a lowercase word -- and several words map to
    one category ("id" and "ssn" both mean SSN_OR_ID). Normalizing both onto the
    category makes them comparable. Tries the category form first (already
    canonical), then the word table.
    """
    s = str(value or "").strip()
    if not s:
        return None
    if s.upper() in KIND_TO_CAT.values():
        return s.upper()
    return KIND_TO_CAT.get(s.lower())


def kind_is_known(value, ctx) -> CheckOutcome:
    """The claimed kind must be one this pipeline actually emits.

    A kind outside the closed set cannot be verified or acted on, so it is
    refused outright rather than passed through.
    """
    name = "kind_is_known"
    if canon_kind(value) is None:
        return fail(name, f"kind {value!r} is not a category this pipeline emits")
    return ok(name)


def kind_renormalizes(value, ctx) -> CheckOutcome:
    """Re-type the span under the claimed kind; the rule parsers must accept it.

    The substantive check. If the model says "phone", the characters had better
    parse as a phone number. Three routes by category:

      AGE            `parse_age_value` must read it, and the result must fall in
                     0..115.
      DATE_OF_BIRTH  `parse_absolute_date` must read it AND find an explicit year
                     -- a birth date with a defaulted year is not usable as one.
      everything else  `_normalize` must return no failure flag.
    """
    name = "kind_renormalizes"
    cat = canon_kind(value)
    if cat is None:
        return na(name, "kind_is_known owns that failure")
    m = ctx.first_mention()
    if m is None:
        return fail(name, "no span to re-normalize")

    if cat == "AGE":
        from ..rules.ages import parse_age_value
        parsed, _approx = parse_age_value(m.text)
        if parsed is None:
            return fail(name, f"{m.text!r} does not parse as an age")
        if not (0 <= parsed <= 115):
            return fail(name, f"{m.text!r} parses as {parsed}, out of age range")
        return ok(name, f"parses as age {parsed}")

    if cat == "DATE_OF_BIRTH":
        from ..rules.dates import parse_absolute_date
        iso, has_year = parse_absolute_date(m.text)
        if iso is None:
            return fail(name, f"{m.text!r} does not parse as a date")
        if not has_year:
            return fail(name, f"{m.text!r} carries no explicit year; not usable "
                              f"as a date of birth")
        return ok(name, f"parses as the date {iso}")

    _sub, _attrs, flag = _normalize(cat, m.text)
    if flag:
        return fail(name, f"span does not normalize as {cat}: {flag}")
    return ok(name)


def renormalized_attrs(kind: str, ctx) -> dict:
    """The sub-attributes implied by an accepted `kind`, so the derived values are
    regenerated from the rule regexes rather than left stale.

    For AGE / DATE_OF_BIRTH -- the categories `_normalize` does not own, whose values
    live in `value` / `resolved_value` and are second-lined in their own right -- this
    still records `kind` itself. Returning a bare `{}` made the whole `kind`
    resolution a NO-OP on those two categories: `apply_resolution` skips the attribute
    write for `attr="category"` (an entity is never re-typed from a suggestion), so a
    verified `kind` on an age or a DOB produced a ledger row and changed nothing at
    all -- 30 confirms across the two samples that wrote nothing.
    """
    cat = canon_kind(kind)
    m = ctx.first_mention()
    if cat is None or m is None:
        return {}
    if cat not in _NORMALIZABLE:
        return {"kind": cat}
    _sub, attrs, flag = _normalize(cat, m.text)
    return {} if flag else attrs


# ------------------------------------------------------------- identifying-ness

def identifying_only_for_occupation(value, ctx) -> CheckOutcome:
    """`identifying` is a judgment about how rare a JOB TITLE is. It says nothing
    about a phone number or an email -- those are identifying by construction and
    are always replaced -- so a claim on any other category is out of scope.

    Note the asymmetry: only `identifying=True` is inspected. A False claim, or a
    missing one, is `na` -- the field only ever adds a review signal, so there is
    nothing to refute in its absence.
    """
    name = "identifying_only_for_occupation"
    if value is not True:
        return na(name, "not an identifying claim")
    if getattr(ctx.entity, "category", "") != "OCCUPATION":
        return fail(name, f"{ctx.entity.category} is not an occupation; "
                          f"`identifying` does not apply")
    return ok(name)


def identifying_not_a_common_occupation(value, ctx) -> CheckOutcome:
    """Refute `identifying=True` for a job the rule layer lists as common.

    This is the checker that stops the model from flagging "miners" in a
    coal-mining interview and "preacher" in a church one. `COMMON_OCCUPATIONS` is
    the rule table, so the checker and the rule cannot drift apart.
    """
    name = "identifying_not_a_common_occupation"
    if value is not True:
        return na(name, "not an identifying claim")
    m = ctx.first_mention()
    text = (ctx.entity.attributes.get("occupation")
            or (m.text if m is not None else ""))
    if _is_common_occupation(text):
        return fail(name, f"{text!r} is a common occupation "
                          f"({len(COMMON_OCCUPATIONS)}-entry rule table); knowing "
                          f"it does not single out a person")
    return ok(name, f"{text!r} is not in the common-occupation table")
