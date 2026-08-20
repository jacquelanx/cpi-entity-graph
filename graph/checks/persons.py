"""
Deterministic checkers for PERSON `replace` / PUBLIC_FIGURE and the
FAMILY / PROFESSIONAL subtype.

`replace` is the one field where the safe direction is asymmetric: keeping a name
is the leak-prone move, redacting it is the harmless one. So the policy is
`safe_direction` (see `graph/second_line/policies.py`) and the checker on the KEEP
direction is strict -- `personal_signal_absent` reuses `attributes._personal_signal`,
which the rules already compute, and refuses any keep for a name used personally.

Subtype checkers corroborate the model's relationship read against the same two
tables the rules use: the kinship edges and `PROFESSIONAL_CONTEXT`. Verified
failure this catches: `John L. Lewis` (a union leader) proposed as FAMILY -- but only
since the kin-word test started requiring the word to be BOUND to the name. With a
bare 40-char proximity window it did the OPPOSITE of what this line claimed: "And
John L. Lewis, my Papaw would say that name like a prayer" put a kin word inside the
window, so the union leader was typed FAMILY with the checker reporting success. See
`_kin_word_binds_a_mention`.

`role_corroborated` deliberately keeps a proximity window, because a role word is a
DESCRIPTION of the person named ("Dr. Combs ... out of a little clinic") rather than a
possessive construction binding them to the speaker. The two questions differ, so the
two tests do.
"""

from __future__ import annotations
import re
from . import CheckOutcome, ok, fail, na
from ..rules.attributes import _personal_signal, PROFESSIONAL_CONTEXT
from ..models import Relation


def personal_signal_absent(value, ctx) -> CheckOutcome:
    """A name may only be KEPT (replace=False) when no personal signal marks it as
    a private individual in THIS transcript."""
    name = "personal_signal_absent"
    if value is not False:
        return na(name, "not a keep claim")
    if _personal_signal(ctx.transcript, ctx.entity, ctx.kin_ids):
        return fail(name, "name is used personally (relative / professional / namesake)")
    return ok(name)


def not_kin_of_interviewee(value, ctx) -> CheckOutcome:
    """A person the rules tied to the interviewee by a kinship edge is never a
    public figure, whatever the model says."""
    name = "not_kin_of_interviewee"
    if value is not False:
        return na(name, "not a keep claim")
    if ctx.entity.entity_id in ctx.kin_ids:
        return fail(name, "entity has a kinship edge to the interviewee")
    return ok(name)


def role_corroborated(value, ctx) -> CheckOutcome:
    """Checker for `role` -- the short relationship/job word for a named person.

    `role` used to be LLM-only and unchecked, so "father" for a Catholic priest
    (verified: `Father Nguyen` in interview_001 got `role: father`) was
    indistinguishable from "father" for the speaker's actual father. The rule layer
    (`attributes.infer_person_role`) now derives the word from a kinship edge or a
    professional construction, and this refuses any role word the transcript does
    not corroborate in one of those two ways:

      * it matches the detail of a RELATED_TO edge for this person (kin synonyms
        collapse, so the model's "mother" corroborates the rule's "mama"); or
      * the word itself -- or a kin synonym of it -- appears near a mention.

    A word with neither is world knowledge, which is exactly what we do not take.
    """
    name = "role_corroborated"
    from .comparators import kin_canon
    word = str(value or "").strip().lower().rstrip(".")
    if not word:
        return na(name, "no role claimed")
    canon = kin_canon(word)
    eid = ctx.entity.entity_id

    for ed in ctx.edges:
        if ed.relation != Relation.RELATED_TO or eid not in (ed.source, ed.target):
            continue
        if kin_canon(str(ed.detail or "").strip().lower()) == canon:
            return ok(name, f"matches the RELATED_TO detail {ed.detail!r}")

    import re
    from .comparators import _KIN_CANON
    from .relation_words import KIN_WORDS

    # A KINSHIP role is a RELATION to somebody, so the word has to be BOUND to this
    # person -- the same test `subtype_corroborated` applies to FAMILY. Proximity is
    # not enough and cannot be made enough: "Father Nguyen, no relation, ... he found
    # my father his first work" puts a genuine "my father" 60 characters from the
    # priest's name, so masking the honorific inside the mention (which this function
    # does) still leaves the speaker's own father corroborating `role="father"` for a
    # Catholic priest. Only a binding construction distinguishes them.
    #
    # A PROFESSIONAL role is a DESCRIPTION of the person named ("a caseworker, Ms.
    # Boudreaux", "Brother Estep's been the preacher there"), so it keeps the
    # proximity window. The two questions differ, so the two tests do.
    if canon in KIN_WORDS or canon in set(_KIN_CANON.values()):
        bound = _kin_word_binds_a_mention(ctx)
        if bound and kin_canon(re.sub(r"^\W+|\W+$", "", bound.split()[-1])) == canon:
            return ok(name, f"kin construction {bound!r} names this person")
        for ed in ctx.edges:
            if ed.relation == Relation.RELATED_TO and eid in (ed.source, ed.target):
                return ok(name, "a kinship edge ties this person to somebody")
        return fail(name, f"kin role {value!r} is not BOUND to this person by any "
                          f"possessive or appositive construction, and they have no "
                          f"kinship edge (a kin word merely nearby can belong to "
                          f"somebody else)")

    synonyms = {w for w, base in _KIN_CANON.items() if base == canon} | {word, canon}
    rx = re.compile(r"(?<![a-z])(?:" +
                    "|".join(re.escape(s) for s in sorted(synonyms, key=len, reverse=True))
                    + r")s?(?![a-z])", re.I)
    # The role word must appear NEAR a mention, not INSIDE one. A word that is part of
    # the person's own name form is not corroboration -- it is the same string being
    # read twice.
    #
    # This module's header cites `Father Nguyen -> role: father` as the failure
    # `role_corroborated` was written to catch, and without this masking it did the
    # opposite: the honorific sits inside the mention, so the proximity search always
    # found it and the priest was confirmed as somebody's father with the checker
    # reporting success. Same for `Governor Barbour -> role: Governor` and
    # `Brother Estep -> role: brother`. Legitimate corroboration is unaffected,
    # because it comes from text AROUND the name ("a caseworker, Ms. Boudreaux",
    # "Brother Estep's been the preacher there").
    own = [(mm.start, mm.end) for mm in ctx.entity.mentions]
    for m in ctx.entity.mentions:
        a, b = max(0, m.start - 80), m.end + 80
        window = list(ctx.transcript[a:b])
        for (s, e) in own:
            for i in range(max(a, s), min(b, e)):
                window[i - a] = "\x00"        # blank out this entity's own spans
        if rx.search("".join(window)):
            return ok(name, f"{word!r} (or a synonym) appears beside a mention")
    return fail(name, f"role {value!r} is corroborated by no kinship edge and by no "
                      f"occurrence of the word near any mention (occurrences inside "
                      f"the person's own name do not count)")


def _has_kin_edge(ctx) -> bool:
    eid = ctx.entity.entity_id
    return any(ed.relation == Relation.RELATED_TO and eid in (ed.source, ed.target)
               for ed in ctx.edges)


# A kin word near a name does NOT make that name a relative. It has to be BOUND to
# the name by one of two constructions:
#
#   "my Papaw Clarence"          possessive + kin word immediately before the name
#   "Clarence, my Papaw"         appositive immediately after, closing cleanly
#
# A bare proximity window accepted neither test and duly mistyped a union leader as
# family: "And John L. Lewis, my Papaw would say that name like a prayer" puts "my
# Papaw" 3 characters after the mention, so a 40-char window found it -- but that kin
# word is the SUBJECT of the following clause, not an appositive describing Lewis.
# `checks/persons.py` claimed to catch this case and did the opposite.
#
# The strict appositive closer is lifted from `kinship.py`'s Pattern 4, which already
# had to solve this exact ambiguity: the kin word must be followed by punctuation,
# "and"/"who"/"whom", or end of text -- never by a verb continuing the sentence.
def _kin_binding_re():
    from ..rules.kinship import KIN, _MOD
    before = re.compile(rf"\b(?:my|our|his|her|their)\s+{_MOD}{KIN}[\s,]*$", re.I)
    after = re.compile(
        rf"^\s*,\s*(?:who\s+(?:is|was)\s+|who's\s+)?"
        rf"(?:my|our|his|her|their)\s+{_MOD}{KIN}\b"
        rf"(?=\s*(?:[,.;:!?)\]\"'’”]|and\b|who\b|whom\b|$))", re.I)
    return before, after


_KIN_BOUND = None


def _kin_word_binds_a_mention(ctx) -> str:
    """The bound kin construction naming this entity, else ""."""
    global _KIN_BOUND
    if _KIN_BOUND is None:
        _KIN_BOUND = _kin_binding_re()
    before, after = _KIN_BOUND
    for m in ctx.entity.mentions:
        pre = ctx.transcript[max(0, m.start - 60):m.start]
        hit = before.search(pre)
        if hit:
            return hit.group(0).strip()
        hit = after.match(ctx.transcript[m.end:m.end + 60])
        if hit:
            return hit.group(0).strip()
    return ""


def subtype_corroborated(value, ctx) -> CheckOutcome:
    """FAMILY needs a kinship edge or a kin word beside a mention; PROFESSIONAL
    needs a professional-context word near a mention. Otherwise the model is
    guessing from world knowledge, which is exactly what we do not accept.

    PUBLIC_FIGURE markers are OUT OF SCOPE, not refuted. This checker now runs on
    rule values as well as LLM proposals (`verify_always`), and the rule sets
    `PUBLIC_FIGURE` / `PUBLIC_FIGURE_UNCONFIRMED` -- a vocabulary this function has no
    corroboration test for. Reporting those as failures would erase the marker that
    `checks/interviewee.not_a_public_figure` and the report both read, for a claim
    this checker was never the right judge of: the `replace` policy owns the
    public-figure decision, with its own two checkers and its own safe direction.
    """
    name = "subtype_corroborated"
    sub = str(value).strip().upper()
    if sub.startswith("PUBLIC_FIGURE"):
        return na(name, "a public-figure marker is decided by the `replace` policy, "
                        "not by kin/professional corroboration")
    if sub == "FAMILY":
        if _has_kin_edge(ctx):
            return ok(name, "kinship edge present")
        bound = _kin_word_binds_a_mention(ctx)
        if bound:
            return ok(name, f"kin construction {bound!r} names this person")
        return fail(name, "no kinship edge, and no kin word is BOUND to any mention "
                          "(a kin word merely nearby can belong to somebody else)")
    if sub == "PROFESSIONAL":
        for m in ctx.entity.mentions:
            if PROFESSIONAL_CONTEXT.search(ctx.transcript[max(0, m.start - 80):m.end + 80]):
                return ok(name, "professional-context word near a mention")
        return fail(name, "no professional-context word near any mention")
    return fail(name, f"subtype {value!r} is not one this checker recognizes")
