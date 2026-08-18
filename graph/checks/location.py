"""
Deterministic checkers for LOCATION `subtype` (type) and `parent` (hierarchy).

The gazetteer is the rule layer. Two failures motivated these checkers, both
verified in the mock output:

  * the model returned types OUTSIDE the enum its own prompt specified
    ('island', 'gulf', 'river', 'county') and nothing rejected them;
  * the model returned free-text parents that were wrong ('Logan' -> 'Wyoming'),
    circular ('Palawan' -> 'Palawan Island'), or placeholders
    ('Freewill Baptist' -> 'creek road, [an unspecified city or town]'), and none
    was ever resolved or turned into a LOCATED_IN edge.

`parent_resolves` is what promotes a suggestion into a real edge: the parent must
name a place the gazetteer knows AND that appears in THIS transcript, otherwise
there is no node to point at.
"""

from __future__ import annotations
import re
from . import CheckOutcome, ok, fail, na
from .comparators import LOC_CANON

_PLACEHOLDER = re.compile(r"\[|\bunspecified\b|\bunknown\b|\bn/?a\b|\?", re.I)


def type_in_enum(value, ctx) -> CheckOutcome:
    """The type must map onto the canonical bucket set. This is the check the
    prompt's own enum was never enforced against.

    NOTE: this is a VOCABULARY test, not a truth test -- it says the word is one we
    know, not that the place is that kind of place. `keep_rests_on_a_verified_type`
    must not treat it as corroboration; see `type_corroborated`.
    """
    name = "type_in_enum"
    if LOC_CANON.get(str(value).strip().lower()) is None:
        return fail(name, f"type {value!r} is outside the accepted vocabulary")
    return ok(name)


# Words in the transcript that CORROBORATE a claimed geographic type. Only the
# buckets whose type can actually be spoken aloud beside a place name are listed; a
# bucket with no linguistic marker stays unprovable rather than being waved through.
# Each cue must imply the bucket it is filed under -- a cue that implies a DIFFERENT
# bucket is not corroboration, it is a near-miss, and filing it loosely turns this
# checker back into the rubber stamp it replaced. "Delta" is the case that taught
# this: with `delta` listed under `region`, the model's uncorroborated "Mekong Delta
# is a region" was corroborated by the word `Delta` in the place's own name -- and
# `LOC_CANON` files `delta` under `feature`, so the two disagreed. A toponymic suffix
# IS real evidence ("Tug Fork" is a fork, "Buffalo Creek" is a creek), which is why
# the entity's own mention text is deliberately NOT masked out here; it just has to
# corroborate the bucket actually claimed.
_TYPE_CUES = {
    "country": (r"countr(?:y|ies)", r"nation"),
    "state": (r"state", r"province", r"commonwealth"),
    "region": (r"region", r"count(?:y|ies)", r"parish", r"coast", r"territory",
               r"peninsula", r"panhandle"),
    "city": (r"cit(?:y|ies)", r"town", r"village", r"hamlet", r"borough"),
    "neighborhood": (r"neighb(?:o|ou)rhood", r"district", r"quarter", r"projects",
                     r"subdivision", r"holler", r"hollow"),
    "street": (r"street", r"road", r"avenue", r"boulevard", r"lane", r"drive",
               r"route", r"highway"),
    "institution": (r"church", r"parish", r"school", r"college", r"universit(?:y|ies)",
                    r"hospital", r"clinic", r"compan(?:y|ies)", r"plant", r"mill",
                    r"mine", r"store", r"diocese", r"base", r"chapel", r"academy"),
    "landmark": (r"park", r"monument", r"landmark", r"memorial", r"cemetery"),
    "feature": (r"river", r"creek", r"lake", r"bayou", r"gulf", r"bay", r"ocean",
                r"sea", r"mountains?", r"fork", r"branch", r"delta", r"valley",
                r"ridge", r"hollow", r"holler", r"island", r"swamp", r"marsh"),
}
# How far from a mention a type word may sit and still describe it.
_TYPE_NEAR = 60


def type_corroborated(value, ctx) -> CheckOutcome:
    """Is the claimed geographic type backed by the gazetteer or by the transcript?

    `subtype_location`'s only checker used to be `type_in_enum`, and
    `keep_rests_on_a_verified_type` accepts any `fill` whose `checks_passed` is
    non-empty -- so "the model used a word from our vocabulary" was the entire
    evidential basis on which a place name could be KEPT unredacted. That is the
    single-layer decision the second line exists to prevent, and it sits on the
    highest-risk category: a model calling a hamlet a "region" gets the hamlet kept.
    (Verified on interview_001, where `Mekong Delta` was kept on a lone
    `type_in_enum` pass.)

    Two deterministic routes to corroboration, plus an explicit abstention:

      * the gazetteer knows this place, and its type agrees at bucket level;
      * a type word for the claimed bucket appears within one clause of a mention
        ("down in the Mekong Delta", "a little two-room school over in Logan");
      * otherwise `na` -- unprovable, NOT refuted. Refusing an uncorroborated type
        outright would strip the geography out of every off-gazetteer place, and the
        type is still useful for reporting. `na` keeps the type while leaving
        `checks_passed` EMPTY, which is exactly what stops it from funding a keep.
    """
    name = "type_corroborated"
    canon = LOC_CANON.get(str(value).strip().lower())
    if canon is None:
        return na(name, "type_in_enum owns that failure")

    ent = ctx.entity
    for form in getattr(ent, "sorted_mentions", []):
        key = _resolve(form, ctx)
        if not key:
            continue
        gaz_type = LOC_CANON.get(str(ctx.gazetteer[key].get("type", "")).lower())
        if gaz_type is None:
            continue
        if gaz_type == canon:
            return ok(name, f"the gazetteer types {form!r} as {gaz_type!r}")
        return fail(name, f"the gazetteer types {form!r} as {gaz_type!r}, "
                          f"model said {value!r}")

    cues = _TYPE_CUES.get(canon)
    if cues:
        rx = re.compile(r"(?<![a-z])(?:" + "|".join(cues) + r")(?![a-z])", re.I)
        for m in getattr(ent, "mentions", []):
            window = ctx.transcript[max(0, m.start - _TYPE_NEAR):m.end + _TYPE_NEAR]
            if rx.search(window):
                return ok(name, f"a {canon!r} type word appears beside a mention")
    return na(name, f"nothing in the gazetteer or the transcript corroborates "
                    f"{value!r}")


def _resolve(candidate: str, ctx):
    """Canonical gazetteer key for a candidate place name, or None."""
    key = str(candidate).strip().lower()
    key = ctx.gaz_aliases.get(key, key)
    return key if key in ctx.gazetteer else None


def parent_no_placeholder(value, ctx) -> CheckOutcome:
    name = "parent_no_placeholder"
    if not value:
        return na(name, "no parent claimed")
    if _PLACEHOLDER.search(str(value)):
        return fail(name, f"parent {value!r} is a placeholder, not a place")
    return ok(name)


def parent_not_self(value, ctx) -> CheckOutcome:
    """Reject circular hierarchies ('Palawan' inside 'Palawan Island')."""
    name = "parent_not_self"
    if not value:
        return na(name, "no parent claimed")
    forms = {f.lower() for f in getattr(ctx.entity, "sorted_mentions", [])}
    for part in re.split(r"\s*,\s*", str(value).lower()):
        part = part.strip()
        if not part:
            continue
        if part in forms or any(part.startswith(f) or f.startswith(part) for f in forms):
            return fail(name, f"parent {value!r} restates the place itself")
    return ok(name)


def parent_resolves(value, ctx) -> CheckOutcome:
    """The parent must resolve to a gazetteer record AND be present in this
    transcript, so a LOCATED_IN edge has a real node to target.

    The model often returns a comma-separated chain ('Mingo County, West
    Virginia, United States'); we accept the FIRST component that resolves and
    is present, which is the nearest enclosing place.
    """
    name = "parent_resolves"
    if not value:
        return na(name, "no parent claimed")
    present = set()
    for e in ctx.entities:
        if e.category not in ("LOCATION", "INSTITUTION"):
            continue
        for f in e.sorted_mentions:
            k = _resolve(f, ctx)
            if k:
                present.add(k)
    for part in re.split(r"\s*,\s*", str(value)):
        k = _resolve(part, ctx)
        if k and k in present:
            return ok(name, f"resolved to {k!r}")
    return fail(name, f"no component of {value!r} is a gazetteer place present here")


# ------------------------------------------------- replace / identifying-ness

def keep_only_if_broad_place(value, ctx) -> CheckOutcome:
    """Checker for `replace_location=False` (keep a place name unredacted).

    Mirrors `checks/persons.personal_signal_absent`: keeping is the leak-prone
    direction, so it is the direction that must be proved. A name may only be kept
    when the place is coarse enough that it cannot single out a household --
    country / state / region, per the rule layer's own
    `location_dates.BROAD_LOCATION_TYPES`. An INSTITUTION is never keepable (a
    named church, school or employer is a direct pointer to a person), and neither
    is a place nothing could type.
    """
    name = "keep_only_if_broad_place"
    if value is not False:
        return na(name, "not a keep claim")
    from ..location_dates import BROAD_LOCATION_TYPES, AMBIGUOUS_BROAD_NAMES
    ent = ctx.entity
    if getattr(ent, "category", "") == "INSTITUTION":
        return fail(name, "an INSTITUTION is a named organisation; never kept")
    # A bare name whose broad gazetteer type is a COLLISION, not a reading of this
    # transcript, cannot fund a keep -- the table cannot tell the state from the city.
    for form in getattr(ent, "sorted_mentions", []):
        if form.strip().lower() in AMBIGUOUS_BROAD_NAMES:
            return fail(name, f"{form!r} names a broad place AND a city-level one; "
                              f"the gazetteer cannot tell which is meant here, so the "
                              f"keep rests on a table collision")
    raw = str(getattr(ent, "subtype", "") or "").strip().lower()
    if not raw:
        return fail(name, "no geographic type could be established for this place")
    if raw in BROAD_LOCATION_TYPES:
        return ok(name, f"type {raw!r} is coarser than a city")
    canon = LOC_CANON.get(raw)
    if canon is None:
        return fail(name, f"type {raw!r} is outside the accepted vocabulary")
    return fail(name, f"type {raw!r} is city-level or finer; keeping it could "
                      f"identify a household")


def keep_rests_on_a_verified_type(value, ctx) -> CheckOutcome:
    """A keep may not rest on a type nothing verified.

    `keep_only_if_broad_place` reads `entity.subtype`, which may have been FILLED
    from the LLM earlier in the same pass. That is fine only when the fill was
    itself checked, so this reads the `subtype_location` Resolution out of
    `entity.provenance` and refuses a keep built on an unverified type. Without
    this, a model that calls a village a "state" could talk the pipeline into
    keeping it.
    """
    name = "keep_rests_on_a_verified_type"
    if value is not False:
        return na(name, "not a keep claim")
    prov = getattr(ctx.entity, "provenance", None) or {}
    res = prov.get("subtype_location")
    if res is None:
        # no resolution recorded yet -> the type came straight from the gazetteer
        return ok(name, "type came from the gazetteer") if ctx.entity.subtype \
            else fail(name, "no type and no resolution to rest on")
    # The test is WHERE the surviving type came from, not which action produced it.
    # A gazetteer type is a deterministic table lookup and stays verified even when
    # the model disagreed with it: `subtype_location` for "Vietnam" resolves to
    # CONFLICT (the model called a country something else) with the rule's value
    # surviving, and treating that as unverified cost Vietnam its keep -- the model's
    # wrong answer was silently deciding a redaction it had just lost.
    if res.source in ("rule", "rule_confirmed"):
        return ok(name, f"gazetteer type ({res.action}); the table is deterministic")
    # An LLM-supplied type funds a keep only when a checker actually CORROBORATED it.
    # `checks_passed` alone is not enough: `type_in_enum` passes on any word in the
    # vocabulary, so requiring merely "some check passed" let the model keep a place
    # by naming it a region. `type_corroborated` is the truth test, and it is the one
    # that has to be in there.
    if res.action == "fill" and "type_corroborated" in tuple(res.checks_passed):
        return ok(name, f"type filled from the LLM and CORROBORATED by the "
                        f"gazetteer or the transcript")
    return fail(name, f"the geographic type this keep depends on came from the LLM "
                      f"({res.action}) with no deterministic corroboration -- "
                      f"passed only {list(res.checks_passed) or 'nothing'}")


def resolved_parent_key(value, ctx):
    """The gazetteer key `parent_resolves` accepted -- used to build the edge."""
    if not value:
        return None
    present = {}
    for e in ctx.entities:
        if e.category not in ("LOCATION", "INSTITUTION"):
            continue
        for f in e.sorted_mentions:
            k = _resolve(f, ctx)
            if k:
                present[k] = e
    for part in re.split(r"\s*,\s*", str(value)):
        k = _resolve(part, ctx)
        if k and k in present:
            return k, present[k]
    return None
