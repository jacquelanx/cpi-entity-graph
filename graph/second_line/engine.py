"""
The decision procedure itself: given a rule value, an LLM proposal
and a policy, produce exactly one `Resolution`.

`second_line` is the entry point. Everything else here is one step of it --
running the checkers, guarding a surviving value that no checker examined
(`_guard_unsafe`), and putting the remaining candidates of a closed-set field to
the same checkers before abstaining (`_try_alternatives`).

Knows nothing about WHICH fields exist; `policies.py` supplies that.
"""

from __future__ import annotations

import dataclasses
from ..checks import CheckContext, CheckOutcome
from .outcomes import CONFIRM, CONFLICT, FILL, FieldPolicy, KEEP, REJECT, REQUIRED_VERIFIED, RULE_WINS, Resolution, SAFE_DIRECTION, _UNSET


def _norm_conf(c) -> str:
    return (str(c or "").strip().lower() or "unstated")


def _empty(v) -> bool:
    return v is None or v == "" or v == ()


# ------------------------------------------------------------------ arbitration

def _run_checks(policy: FieldPolicy, value, ctx: CheckContext):
    """Returns (passed, failed, skipped). `passed` holds ONLY outcomes from checkers
    that actually inspected `value`; a checker that did not apply lands in
    `skipped` and is reported as such rather than inflating the passed count."""
    passed, failed, skipped = [], [], []
    for check in policy.checkers:
        out: CheckOutcome = check(value, ctx)
        if not out.applicable:
            skipped.append(out)
        elif out.passed:
            passed.append(out)
        else:
            failed.append(out)
    return passed, failed, skipped


def _names(outs) -> tuple:
    return tuple(o.name for o in outs)


def _is_unsafe(policy: FieldPolicy, value) -> bool:
    if policy.verify_always and not _empty(value):
        return True
    if policy.unsafe_when is not None:
        return bool(policy.unsafe_when(value))
    return policy.unsafe is not _UNSET and value == policy.unsafe


def _guard_unsafe(res: Resolution, policy: FieldPolicy, ctx: CheckContext) -> Resolution:
    """A field that lands on a value the checkers can REFUTE must clear them,
    however it got there -- including by rule/LLM agreement, and including when the
    rules alone put it there.

    This is the ONLY place a checker can examine a value the LLM did not supply, so
    it is what makes "every field has a deterministic checker" true on every path
    rather than only on `fill`. It applies to a field's consequential direction
    (`unsafe` / `unsafe_when`) and to every non-empty value of a field whose checkers
    are truth tests (`verify_always`).

    Three fields need the directional form, for the same reason. Without it the
    checkers would never bind on the dangerous path:

      * `replace=False` (keep a name). The safe direction already forces more
        redaction on a conflict, so the only way to reach a keep is AGREEMENT --
        and agreement is exactly the case that leaks when both layers are fooled by
        a private namesake of a celebrity. This is what makes the closed
        PUBLIC_FIGURES list non-authoritative in the keep direction.
      * `owner="interviewee"`. Claiming the speaker owns an identifier is the
        consequential direction: it feeds their surrogate identity. A rule value
        that the LLM never contradicted would otherwise pass unexamined.
      * `interviewee_identity`. ANY non-empty value merges a named person into
        e000, so `unsafe_when` marks the whole non-empty direction consequential:
        rule/LLM agreement on the wrong person must not skip the gate.
    """
    if not _is_unsafe(policy, res.value):
        return res
    passed, failed, skipped = _run_checks(policy, res.value, ctx)
    if not failed:
        # A REQUIRED_VERIFIED field blocks because its value was UNVERIFIED, not
        # because of the action that produced it. If a checker has now actually
        # examined the surviving value and not refuted it, that requirement is met and
        # the block must lift -- otherwise the guard reports success and the pipeline
        # still stops. Three of interview_001's five blocking rows were exactly this:
        # the speaker's phone, DOB and boat licence, each with four passing checkers,
        # blocking only because the model had answered "other" or stayed silent.
        #
        # `not passed` keeps the block when every checker returned `na`: nothing looked
        # at the value, so nothing verified it, so the tier's demand is unsatisfied.
        blocking = bool(res.blocking and not passed)
        # ...and a REQUIRED_VERIFIED field must block on an UNVERIFIED value however
        # it got here, including on the FILL path.
        #
        # `_resolve.blocking()` only ever set `blocking` for KEEP / CONFLICT / REJECT,
        # so a FILL was structurally exempt from the one tier whose whole meaning is
        # "this value must be verified". interview_001's `interviewee_gender` was
        # exactly that: `F` accepted from the model with BOTH checkers returning `na`,
        # reported as "filled from the LLM with NO deterministic check applicable",
        # and NOT blocking -- an unverified guess about the subject's gender flowing
        # straight into surrogate generation.
        #
        # Narrow by construction: only three policies carry this tier, and for
        # `replace` and `owner` a non-empty value always has an applicable checker, so
        # this adds no rows there. It fires on `interviewee_gender`, at most once per
        # transcript, on the field that most deserves a human.
        if policy.tier == REQUIRED_VERIFIED and not passed and not _empty(res.value):
            blocking = True
        return dataclasses.replace(res, checks_passed=_names(passed),
                                   checks_failed=(), checks_skipped=_names(skipped),
                                   blocking=blocking)
    return Resolution(res.field, CONFLICT, policy.safe_value, "rule", res.confidence,
                      _names(passed), _names(failed),
                      f"{res.action} resolved to {res.value!r} but a deterministic "
                      f"check refuted it: "
                      + "; ".join(f"{o.name}: {o.detail}" for o in failed),
                      blocking=(policy.tier == REQUIRED_VERIFIED),
                      checks_skipped=_names(skipped))


def second_line(policy: FieldPolicy, rule_value, llm, ctx: CheckContext,
                llm_ran: bool = True) -> Resolution:
    """Resolve ONE field, then guard the leak-prone direction.

    `llm` is `{"value":..., "confidence":...}` or None.
    """
    res = _guard_unsafe(_resolve(policy, rule_value, llm, ctx, llm_ran), policy, ctx)
    res = _try_alternatives(res, policy, rule_value, ctx, llm_ran)
    return _canonicalize(res, policy)


def _canonicalize(res: Resolution, policy: FieldPolicy) -> Resolution:
    """Put the surviving value in the field's canonical form (see `FieldPolicy.canon`).
    A canon that cannot read the value leaves it alone -- normalization must never be
    able to destroy a decision the checkers already accepted."""
    if policy.canon is None or _empty(res.value):
        return res
    try:
        canon = policy.canon(res.value)
    except Exception:
        return res
    if canon is None or canon == res.value:
        return res
    return dataclasses.replace(res, value=canon)


# Fields whose value comes from a small CLOSED set, where every member has its own
# checker family. For these, "the answer the model happened to give is refuted" does
# not mean "there is no answer" -- so before abstaining, the remaining candidates are
# put to the checkers too.
_ALTERNATIVES: dict[str, tuple] = {
    # `owner` is the field this exists for. `checks/ownership.py` carries a positive
    # checker family for EACH direction, and the arbitration used to consult only the
    # one the proposal happened to name: in interview_001 the model answered "other"
    # for the speaker's own shop phone and email, `third_party_identifiable` correctly
    # refuted it, and the field went empty and BLOCKING -- even though "interviewee"
    # would have cleared all four of its checkers on the very same span. A refuted
    # guess was silently costing the interviewee their own identifiers.
    "owner": ("interviewee", "other"),
    # Same argument for the SPEAKER'S OWN GENDER, which is a two-member closed set
    # and the field surrogate generation leans on hardest. In interview_002 the model
    # answered `F` for a speaker who says "My wife wanted to";
    # `interviewee_spouse_term_agrees` correctly refuted it and the field went empty,
    # with `M` never put to the checkers at all.
    #
    # This cannot manufacture an answer: `_survivors` demands at least one checker
    # actually PASS, and the only checker here that can pass is the honorific-address
    # one. So a refuted guess is rescued only when the interviewer's own form of
    # address supports the alternative -- never from a spouse term, which stays
    # refute-only on purpose.
    "interviewee_gender": ("F", "M"),
}


def _survivors(policy: FieldPolicy, candidates, ctx: CheckContext) -> list:
    """The candidates that clear EVERY applicable checker, with at least one checker
    having actually examined them. Silence is not evidence: a checker family that
    short-circuits on a span says nothing in favour of its direction."""
    out = []
    for cand in candidates:
        passed, failed, _skipped = _run_checks(policy, cand, ctx)
        if not failed and passed:
            out.append(cand)
    return out


def owner_survivors(policy: FieldPolicy, ctx: CheckContext) -> list:
    """Which `owner` directions the deterministic checkers support for `ctx.entity`.

    Exported so the RULE layer (`pipeline._link_interviewee_pii`) decides with exactly
    this function rather than a private copy of the predicates -- one implementation,
    one answer, no drift.
    """
    return _survivors(policy, _ALTERNATIVES["owner"], ctx)


def _try_alternatives(res: Resolution, policy: FieldPolicy, rule_value,
                      ctx: CheckContext, llm_ran: bool) -> Resolution:
    """When a closed-set field abstained, test the OTHER candidates deterministically.

    Only ever reached when the second line has already concluded "no value" -- so this
    cannot override the rules, cannot override a value the model gave that survived its
    checkers, and cannot weaken a conflict that resolved to something. It fills a hole
    that was previously left empty.

    A candidate is accepted only when it clears EVERY applicable checker and at least
    one checker actually examined it (`passed` non-empty): silence is not evidence
    here, which is what stops both directions being "accepted" on a span that supports
    neither. If two candidates both clear the checkers the field stays empty -- an
    ambiguous owner is a decision for a human, not a coin flip.
    """
    candidates = _ALTERNATIVES.get(policy.field)
    if candidates is None or res.value is not None or not _empty(rule_value):
        return res

    survivors = _survivors(policy, candidates, ctx)
    if len(survivors) != 1:
        return res

    cand = survivors[0]
    passed, _failed, skipped = _run_checks(policy, cand, ctx)
    return Resolution(
        policy.field, FILL, cand, "checker_derived", res.confidence,
        _names(passed), (),
        f"neither layer produced a usable value ({res.reason[:80]}), but "
        f"{cand!r} clears every applicable deterministic check and the other "
        f"candidates do not",
        blocking=False, checks_skipped=_names(skipped))


def _resolve(policy: FieldPolicy, rule_value, llm, ctx: CheckContext,
             llm_ran: bool = True) -> Resolution:
    f = policy.field
    llm_value = None if llm is None else llm.get("value")
    conf = _norm_conf(None if llm is None else llm.get("confidence"))

    def blocking(action: str) -> bool:
        # Tiers only bind when the second line actually ran; with the LLM off the
        # pipeline degrades to rules-only behaviour instead of blocking everything.
        if not llm_ran or policy.tier != REQUIRED_VERIFIED:
            return False
        return action in (KEEP, CONFLICT, REJECT)

    # --- no LLM answer -------------------------------------------------------
    if _empty(llm_value):
        if not _empty(rule_value):
            return Resolution(f, KEEP, rule_value, "rule", conf,
                              reason="rules stand; no LLM answer to confirm them",
                              blocking=blocking(KEEP))
        return Resolution(f, REJECT, None, "", conf,
                          reason="neither the rules nor the LLM produced a value",
                          blocking=blocking(REJECT))

    # --- rule filled -> CHECK ------------------------------------------------
    if not _empty(rule_value):
        if policy.comparator(rule_value, llm_value, ctx):
            return Resolution(f, CONFIRM, rule_value, "rule_confirmed", conf,
                              reason="rule and LLM agree")
        if policy.conflict_policy == RULE_WINS:
            return Resolution(f, CONFLICT, rule_value, "rule", conf,
                              reason=f"LLM says {llm_value!r}, rule kept {rule_value!r}",
                              blocking=blocking(CONFLICT))
        if policy.conflict_policy == SAFE_DIRECTION:
            safe = policy.safer(rule_value, llm_value, ctx)
            return Resolution(f, CONFLICT, safe,
                              "rule" if safe == rule_value else "llm_checked", conf,
                              reason=f"LLM says {llm_value!r}, rule said {rule_value!r}; "
                                     f"resolved to the safer value {safe!r}")
        return Resolution(f, CONFLICT, None, "", conf,
                          reason=f"LLM says {llm_value!r}, rule said {rule_value!r}; "
                                 f"unresolvable without a human",
                          blocking=blocking(CONFLICT))

    # --- rule empty -> FILL, gated by every APPLICABLE checker ---------------
    passed, failed, skipped = _run_checks(policy, llm_value, ctx)
    if failed:
        return Resolution(f, REJECT, None, "", conf,
                          checks_passed=_names(passed),
                          checks_failed=_names(failed),
                          reason="; ".join(f"{o.name}: {o.detail}" for o in failed),
                          blocking=blocking(REJECT), checks_skipped=_names(skipped))
    reason = ("rule left it unset; LLM value passed every applicable check"
              if passed else
              "rule left it unset; NO deterministic check applied to this value")
    return Resolution(f, FILL, llm_value, "llm_checked", conf,
                      checks_passed=_names(passed), reason=reason,
                      checks_skipped=_names(skipped))
