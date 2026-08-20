"""
Deterministic checkers for `interviewee_identity` -- is this named PERSON the
speaker?

Merging the wrong person into `e000` is the worst single error this pipeline can
make: the interviewee's surrogate identity would be built from a relative's name
while their own name went to a different node, and every owned identifier, age and
DOB would hang off the wrong human. So the accepted value is gated hard, and the
gate is deliberately POSITIVE -- unlike most checkers here, silence is not enough.
A name only becomes the interviewee on evidence.

The three checks:

  named_in_self_reference_or_address   the required positive evidence: the name is
      the speaker's turn label, appears in a first-person self-introduction inside
      a SUBJECT turn, or is used by the interviewer to address the listener. Uses
      `interviewee.support_for`, the same function the rule proposer uses, so the
      proposer and the verifier cannot disagree about what counts.
  not_introduced_as_a_relative        "my aunt Maria" proves Maria is NOT the
      speaker. Reuses the kinship vocabulary rather than a private list.
  not_a_public_figure                 a listed celebrity is never the subject of an
      oral-history interview; if the rules or the model land there, something else
      has gone wrong upstream.
"""

from __future__ import annotations
from . import CheckOutcome, ok, fail, na
from ..rules.interviewee import support_for, kin_introduction_before


def _candidate(value, ctx):
    if not value:
        return None
    return ctx.ent_by_id.get(str(value))


def named_in_self_reference_or_address(value, ctx) -> CheckOutcome:
    name = "named_in_self_reference_or_address"
    ent = _candidate(value, ctx)
    if ent is None:
        return fail(name, f"{value!r} is not an entity in this transcript")
    kind, ev = support_for(ent, ctx.transcript)
    if not kind:
        return fail(name, "this name is never the speaker's turn label, never "
                          "self-introduced in a speaker turn, and never used by "
                          "the interviewer to address the listener")
    return ok(name, f"{kind}: {ev[:80]}")


def not_introduced_as_a_relative(value, ctx) -> CheckOutcome:
    name = "not_introduced_as_a_relative"
    ent = _candidate(value, ctx)
    if ent is None:
        return na(name, "no candidate to inspect")
    for m in ent.mentions:
        kin = kin_introduction_before(ctx.transcript, m.start)
        if kin:
            return fail(name, f"introduced as the speaker's relative: {kin!r}")
    return ok(name)


def not_a_public_figure(value, ctx) -> CheckOutcome:
    name = "not_a_public_figure"
    ent = _candidate(value, ctx)
    if ent is None:
        return na(name, "no candidate to inspect")
    from ..rules.attributes import PUBLIC_FIGURES
    if str(ent.subtype or "").startswith("PUBLIC_FIGURE"):
        return fail(name, "the rules typed this name as a public figure")
    if {f.lower() for f in ent.sorted_mentions} & PUBLIC_FIGURES:
        return fail(name, "the name matches the public-figure table")
    return ok(name)
