"""
Deterministic checkers for `ethnicity`.

This was the worst-behaved field in the pipeline, and it was invisible because it
sat outside the second line entirely: "LLM-only, no rule to check against". The
consequence, verified on both sample transcripts, was that EVERY named person
inherited the speaker's ethnicity as an `inferred` guess from their name alone --
Bao, Hoa, Minh, Thao, Hai, Trang, Khanh and Duc all came out "Vietnamese
(inferred)", Mr. Landry came out "Cajun", and the interviewee of interview_002
came out "Scotch-Irish (inferred)" -- with no Resolution, no ledger row, no
checker and no way for a reviewer to see that nothing had been verified.

Ethnicity IS rule-checkable, because the only trustworthy source for it is the
transcript saying so. `attributes.ethnicity_claims` owns the constructions; these
three checkers bound whatever the model proposes:

  label_is_known_ethnonym    the label must be a recognized ethnonym, not free text
  label_stated_in_transcript the label must actually occur in the transcript --
                             this is what refutes a guess from a name
  attributed_to_this_person  the occurrence must be tied to THIS person: in the
                             subject's own speech for the interviewee, or near one
                             of their mentions / in an accepted construction for a
                             named person. This is what stops the speaker's
                             ethnicity spreading to everyone they mention.

Name-based inference is not merely unverified, it is *refutable*, and refuting it
is the point: a surrogate name minted from a hallucinated heritage is a worse
error than no ethnicity at all.
"""

from __future__ import annotations
import re
from . import CheckOutcome, ok, fail, na

# How far from a mention an ethnicity statement may sit and still be read as that
# person's. One clause either side -- the same locality bound the rule layer uses.
_NEAR = 80


def _canon(value):
    from ..rules.attributes import normalize_ethnonym
    return normalize_ethnonym(value)


def _label_re(canon: str):
    """Matches the label in text, tolerating the hyphen/space variant."""
    alts = {canon, canon.replace("-", " "), canon.replace(" ", "-")}
    return re.compile(r"(?<![a-z])(?:" +
                      "|".join(re.escape(a) for a in sorted(alts, key=len, reverse=True))
                      + r")(?![a-z])", re.I)


def label_is_known_ethnonym(value, ctx) -> CheckOutcome:
    name = "label_is_known_ethnonym"
    if not value:
        return na(name, "no ethnicity claimed")
    if _canon(value) is None:
        return fail(name, f"{value!r} is not a recognized ethnicity/heritage label")
    return ok(name)


def label_stated_in_transcript(value, ctx) -> CheckOutcome:
    """The label has to be in the text. A model that reads 'Bao' and answers
    'Vietnamese' is guessing from the name, and this refutes it."""
    name = "label_stated_in_transcript"
    canon = _canon(value)
    if canon is None:
        return na(name, "label_is_known_ethnonym owns that failure")
    if _label_re(canon).search(ctx.transcript):
        return ok(name, f"{canon!r} appears in the transcript")
    return fail(name, f"{canon!r} never appears in the transcript; this is an "
                      f"inference from a name, not a statement")


def attributed_to_this_person(value, ctx) -> CheckOutcome:
    """The statement must be about THIS person.

    For the interviewee: the label must appear in the SUBJECT'S OWN speech, so the
    interviewer's heritage cannot become the subject's.

    For a named person: the label must appear within one clause of one of their
    mentions, or inside an accepted ethnicity construction that sits beside a
    mention. A label that only ever occurs in sentences about somebody else is
    refuted -- which is the whole reason this checker exists.
    """
    name = "attributed_to_this_person"
    canon = _canon(value)
    if canon is None:
        return na(name, "label_is_known_ethnonym owns that failure")
    rx = _label_re(canon)
    ent = ctx.entity
    iv = ctx.interviewee

    if ent is iv or getattr(ent, "entity_id", None) == getattr(iv, "entity_id", None):
        from ..rules.attributes import ethnicity_claims
        spoken = ctx.subject_transcript
        if any(c == canon for c, _ev in ethnicity_claims(spoken)):
            return ok(name, "self-applied in the subject's own speech")
        if rx.search(spoken):
            return ok(name, f"{canon!r} occurs in the subject's own speech")
        return fail(name, f"{canon!r} never occurs in the subject's own speech "
                          f"(only the interviewer's, or nowhere)")

    for m in getattr(ent, "mentions", []):
        window = ctx.transcript[max(0, m.start - _NEAR):m.end + _NEAR]
        if rx.search(window):
            return ok(name, f"{canon!r} appears beside a mention of this person")
    return fail(name, f"{canon!r} appears in the transcript but never near a "
                      f"mention of this person; it is somebody else's")
