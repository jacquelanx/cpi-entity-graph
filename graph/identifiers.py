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
