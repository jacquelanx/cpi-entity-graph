"""
Deterministic handling for the direct-identifier types that were previously
dropped: PHONE, EMAIL, SSN_OR_ID, USERNAME_HANDLE, OCCUPATION.

Each detected span becomes an entity so it reaches surrogate generation (never
silently dropped -- that would be a leak), typed/normalized by rule, and marked
`replace=True` by default (over-redaction is safe; under-redaction leaks). The
LLM adds the *contextual* judgment on top (owner, whether an occupation is
identifying) -- see `llm_layer/identifier_judge.py`.

INSTITUTION is NOT handled here -- it goes through the location path (gazetteer +
open-world classifier).
"""

from __future__ import annotations
import re
from .models import Entity

ID_CATS = ("PHONE", "EMAIL", "SSN_OR_ID", "USERNAME_HANDLE", "OCCUPATION")

_SSN = re.compile(r"^\d{3}-?\d{2}-?\d{4}$")
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# RULE layer for `identifying`: occupations common enough that knowing one tells
# you nothing about WHICH person you are looking at. `identifying` used to be
# LLM-only with no rule to run first and no checker to refute it, and the model
# duly returned True for seven of the nine occupations in the sample transcripts
# -- including "miners" in a coal-mining interview and "preacher" in a church
# one. A signal that fires on everything carries no information.
#
# Membership here means NOT identifying (the rule fills `identifying=False`); an
# occupation absent from the list is left UNSET so the LLM can fill it, gated by
# `checks/identifiers.identifying_not_a_common_occupation`. Kept deliberately
# generous: the cost of calling a common job common is a missed review flag on a
# span that is redacted anyway (every OCCUPATION is `replace=True`), while the
# cost of the reverse is the noise that made this field useless.
COMMON_OCCUPATIONS = {
    # extraction / trades / manual
    "miner", "miners", "coal miner", "farmer", "farmers", "fisherman", "fishermen",
    "shrimper", "deckhand", "welder", "mechanic", "carpenter", "plumber",
    "electrician", "painter", "roofer", "mason", "logger", "trucker", "millwright",
    "truck driver", "driver", "laborer", "labourer", "factory worker", "millworker",
    "mill worker", "steelworker", "foreman", "machinist", "operator", "janitor",
    "custodian", "housekeeper", "maid", "seamstress", "tailor", "butcher", "baker",
    # service / retail / food
    "waitress", "waiter", "server", "cook", "chef", "dishwasher", "bartender",
    "cashier", "clerk", "salesman", "saleswoman", "shopkeeper", "barber",
    "hairdresser", "beautician",
    # care / education / clerical
    "nurse", "nurses", "nurse's aide", "aide", "caregiver", "babysitter", "nanny",
    "teacher", "teachers", "schoolteacher", "substitute teacher", "secretary",
    "receptionist", "bookkeeper", "accountant", "typist",
    # Social services. `caseworker` is the gap the evaluation found: the rule table
    # did not list it, so `identifying_not_a_common_occupation` had nothing to refute
    # the model's `True` with, and a diocese caseworker was flagged as a job rare
    # enough to single somebody out. Every agency in the country employs them.
    "caseworker", "case worker", "social worker", "counselor", "counsellor",
    "therapist", "midwife", "orderly", "dispatcher", "librarian",
    # uniformed / civic / faith
    "soldier", "sailor", "marine", "policeman", "police officer", "officer",
    "firefighter", "fireman", "mailman", "postman", "preacher", "pastor",
    "minister", "priest",
    # household / status
    "housewife", "homemaker", "student", "retired", "unemployed", "volunteer",
}


def _is_common_occupation(text: str) -> bool:
    t = re.sub(r"\s+", " ", (text or "").strip().lower()).strip(".,;:")
    return t in COMMON_OCCUPATIONS


def _normalize(cat: str, text: str):
    """Return (subtype, attributes) for one identifier span. `kind` mirrors the
    category so the surrogate generator knows what shape of fake to mint."""
    t = text.strip()
    attrs = {"kind": cat, "replace": True}
    subtype, flag = cat, None

    if cat == "PHONE":
        digits = re.sub(r"\D", "", t)
        attrs["digits"] = digits
        subtype = "US_PHONE" if len(digits) in (10, 11) else "PHONE"
        if len(digits) < 7:
            flag = "phone has too few digits"
    elif cat == "EMAIL":
        if _EMAIL.match(t):
            attrs["local"], attrs["domain"] = (p.lower() for p in t.split("@", 1))
        else:
            flag = "email failed to parse"
    elif cat == "SSN_OR_ID":
        subtype = "SSN" if _SSN.match(t) else "ID"
        if not any(c.isdigit() for c in t):
            flag = "SSN_OR_ID has no digits"
    elif cat == "USERNAME_HANDLE":
        attrs["handle"] = t.lstrip("@").lower()
        subtype = "HANDLE"
    elif cat == "OCCUPATION":
        attrs["occupation"] = t.lower()
        subtype = "OCCUPATION"
        # rule layer for `identifying`: a common job is not identifying. An
        # uncommon one is left UNSET for the LLM to fill and the checkers to gate.
        if _is_common_occupation(t):
            attrs["identifying"] = False
    return subtype, attrs, flag


def build_identifier_entities(transcript_id: str, mentions: list) -> list[Entity]:
    """One entity per distinct (category, lowercased text) so repeated identical
    identifiers cluster and get replaced consistently."""
    groups: dict[tuple, list] = {}
    for m in mentions:
        if m.entity_type in ID_CATS:
            groups.setdefault((m.entity_type, m.text.lower()), []).append(m)

    ents = []
    for i, ((cat, _), ms) in enumerate(
        sorted(groups.items(), key=lambda kv: min(x.start for x in kv[1])), start=1
    ):
        sub, attrs, flag = _normalize(cat, ms[0].text)
        e = Entity(entity_id=f"{transcript_id}_ID{i:03d}", category=cat, subtype=sub,
                   mentions=sorted(ms, key=lambda x: x.start), attributes=attrs)
        if flag:
            e.flag_entity(flag)
        ents.append(e)
    return ents
