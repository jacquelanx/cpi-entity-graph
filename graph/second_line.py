"""
THE unified second line. One arbitration point for every field.

The rules run first. For each field this module then asks one of five questions
and records exactly one outcome:

    rule set,   no LLM answer      -> keep      rules stand, unconfirmed
    rule set,   LLM agrees         -> confirm   provenance rule_confirmed
    rule set,   LLM disagrees      -> conflict  resolved per conflict_policy
    rule empty, LLM passes checks  -> fill      provenance llm_checked
    rule empty, LLM fails a check  -> reject    stays empty, failed check named
    rule empty, no LLM answer      -> reject    BOTH LAYERS BLIND, made visible

Whatever the outcome, the value that SURVIVES is then verified -- see `_guard_unsafe`
and `FieldPolicy.verify_always`. That is not a fourth step bolted on; it is what makes
"a deterministic checker stands behind every field" true on the `keep` and `conflict`
paths as well as on `fill`. And when a closed-set field abstains, `_try_alternatives`
puts the OTHER candidates to the same checkers before giving up, because "the answer
the model happened to give was refuted" is not the same fact as "there is no answer".

Three properties the previous architecture did not have:

1. **Arbitration is policy, not `dict.setdefault`.** `owner`, `suggested_subtype`
   and `suggested_kind` used to be settled by `setdefault`, which discarded the
   LLM's answer before it could be compared -- so "the LLM double-checks a filled
   field" was structurally impossible, and a WRONG rule value was unfalsifiable.
2. **`conflict_policy` is per field.** A blanket "rule wins" is wrong for
   `replace`: if the rule says keep and the LLM says private, safety requires the
   LLM to win. `safe_direction` encodes that. `block` refuses to pick at all.
3. **Checkers can reject.** An LLM fill is accepted only when EVERY checker
   passes, and the failing check names itself in the review flag.

NOTHING is out of scope any more. Two classes used to be:

  * "LLM-only fields with no rule to check against" -- `role`, `ethnicity`,
    `identifying`. That was wrong on the facts: a kinship-edge detail or a
    professional construction IS a rule for `role`; self-identification is a closed
    construction set for `ethnicity` (`graph/attributes.py`); and a
    common-occupation table is a rule for `identifying` (`graph/identifiers.py`).
    The cost of the exemption was visible in the output -- every named person in
    both sample transcripts inherited the speaker's ethnicity as an unchecked guess
    from their name, and the model called seven of nine occupations "identifying".
  * "Structural identity decisions" -- alias/nickname merges, clustering, coref.
    The question shape really is different (there is no field to fill), but the
    consequence is not, and these were the one decision class with no Resolution, no
    provenance and no ledger row. They are now resolved per PAIR under
    `same_person`, exactly as `relation` is, with checkers in
    `graph/checks/merges.py`. Merging is still NEVER automatic from the LLM: a
    checked claim becomes a review flag, so identity changes stay a human decision.

Every field on every category has all three layers -- a rule, an LLM proposal, and at
least one deterministic checker. `_fields_for` is the map. Two things that sentence
does NOT claim, because they are not true and the previous wording implied they were:

  * a checker does not run on every PATH of every field. `location_parent` and
    `relation` carry no `verify_always`, so their rule values are never re-examined.
    That is deliberate and argued at each policy (both checkers are PROVABILITY gates,
    not truth tests, and verifying rule values with them would delete correct data) --
    but it does mean a wrong rule kinship edge is unfalsifiable, and it propagates into
    `ctx.kin_ids`, `role`, the FAMILY subtype and `checks/persons`.
  * "checked" is not "corroborated". A checker that returns `na` says nothing, and
    `Resolution.verified` is the only thing that distinguishes a value some checker
    actually examined from one that merely nothing refuted. Read `checks_passed`, not
    `action`, when the answer matters -- `checks/location.keep_rests_on_a_verified_type`
    is the worked example of getting that distinction wrong and back again.
"""

from __future__ import annotations
import dataclasses
from dataclasses import dataclass
from typing import Callable

from .models import Edge, Relation
from .checks import CheckContext, CheckOutcome
from .checks import comparators as C
from .checks import (ages as chk_age, approximate as chk_approx,
                     dates as chk_date, ethnicity as chk_eth,
                     gender as chk_gender, identifiers as chk_id,
                     interviewee as chk_iv, location as chk_loc,
                     merges as chk_merge, names as chk_name,
                     ownership as chk_own, persons as chk_person,
                     relationcheck as chk_rel, statedwith as chk_sw)
from .checks.relwords import KIN_WORDS

# ---------------------------------------------------------------- outcome types

KEEP, CONFIRM, FILL, CONFLICT, REJECT = "keep", "confirm", "fill", "conflict", "reject"

# tiers
REQUIRED_VERIFIED = "REQUIRED_VERIFIED"
REQUIRED_OR_ABSTAIN = "REQUIRED_OR_ABSTAIN"
OPTIONAL = "OPTIONAL"

# conflict policies
RULE_WINS, SAFE_DIRECTION, BLOCK = "rule_wins", "safe_direction", "block"


class _Unset:
    def __repr__(self):
        return "<unset>"


_UNSET = _Unset()


@dataclass(frozen=True)
class Resolution:
    field: str
    action: str
    value: object = None
    source: str = ""                  # rule | rule_confirmed | llm_checked
    confidence: str = "unstated"
    checks_passed: tuple = ()
    checks_failed: tuple = ()
    reason: str = ""
    blocking: bool = False
    # Checkers that DID NOT APPLY to this value. Kept separate from
    # `checks_passed` so a fill nothing actually verified cannot advertise itself
    # as having passed N checks -- see the note in graph/checks/__init__.py.
    checks_skipped: tuple = ()

    @property
    def verified(self) -> bool:
        """True when at least one checker actually examined the value and none
        refuted it."""
        return bool(self.checks_passed) and not self.checks_failed

    def to_dict(self) -> dict:
        return {"field": self.field, "action": self.action, "value": self.value,
                "source": self.source, "confidence": self.confidence,
                "checks_passed": list(self.checks_passed),
                "checks_failed": list(self.checks_failed),
                "checks_skipped": list(self.checks_skipped),
                "reason": self.reason, "blocking": self.blocking}


@dataclass(frozen=True)
class FieldPolicy:
    field: str
    tier: str
    conflict_policy: str
    comparator: Callable              # mandatory, never defaulted
    checkers: tuple = ()
    safer: Callable | None = None     # required when conflict_policy == SAFE_DIRECTION
    attr: str | None = None           # attribute key, when != field name
    unsafe: object = _UNSET           # the consequential value: must clear the checkers
    safe_value: object = None         # what to fall back to when a checker refutes it
    # Predicate form of `unsafe`, for a field whose consequential direction is not
    # a single literal ("any non-empty interviewee identity is consequential").
    unsafe_when: Callable | None = None
    # Canonical form for the SURVIVING value, applied after arbitration so the
    # Resolution, the provenance record and the attribute all carry one spelling.
    #
    # Without it the two layers wrote the same field in two formats: the rule stores a
    # canonical lowercase ethnonym ("vietnamese") while an LLM fill stored the model's
    # raw label ("Cajun", "Scotch-Irish"), and the rule stores a lowercase role while
    # the model returned "Caseworker" / "Governor". The comparators are
    # case-insensitive so this never showed up as a conflict -- it just handed
    # surrogate generation the same field in two shapes.
    canon: Callable | None = None

    # Verify EVERY non-empty resolved value, not just the leak-prone direction.
    #
    # Set on every field whose checkers are TRUTH TESTS -- predicates that say the
    # value is wrong, not merely unusable. Without it the checkers bound only on the
    # `fill` path: a rule value the LLM never contradicted (`keep`) or one that
    # survived a RULE_WINS conflict reached the graph with nothing having examined
    # it, which is how `given_name="Papaw"` and the `her eighties` <-> `Last spring`
    # pairing both got through with `checks_passed=[]`.
    #
    # Deliberately NOT set on `location_parent`: `parent_resolves` asks "is there a
    # node here to point a LOCATED_IN edge at?", which is a usability gate, not a
    # truth test -- the gazetteer's parent for "West Virginia" is "United States"
    # whether or not this transcript happens to mention it, and verifying it would
    # discard a correct value.
    verify_always: bool = False


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


# ------------------------------------------------------------- safe directions

def _safer_replace(rule_value, llm_value, ctx=None):
    """More redaction is always safer: True (replace) beats False (keep)."""
    return True if (rule_value is True or llm_value is True) else False


def _safer_shiftable(rule_value, llm_value, ctx=None):
    """Refusing to shift a date is the conservative choice for a public event, but
    a date wrongly pinned as non-shiftable leaks a real calendar point. Treat
    'shiftable' (True) as the safer value when the two disagree."""
    return True if (rule_value is True or llm_value is True) else False


# Granularity at which the deterministic rule OUTRANKS a cautious model. A country
# or a state/province/district holds millions of people; no amount of model
# nervousness makes naming one identify a household, and deferring to the model
# there strips the region out of every transcript -- the sample runs had the model
# calling "Vietnam", "West Virginia" and "Washington" identifying. At region /
# county level and below the model's local knowledge is worth more than the
# gazetteer's bucket, so there the usual "more redaction wins" applies.
_RULE_OUTRANKS_TYPES = {"country", "territory", "state", "province", "district"}


def _safer_replace_location(rule_value, llm_value, ctx=None):
    """Safe direction for a PLACE name, tempered by granularity.

    Identical to `_safer_replace` except in one case: the rule says KEEP because the
    gazetteer typed the place country- or state-level, and the model says replace.
    There the rule wins, because the threshold is deterministic and auditable and
    the model's answer is a guess about a question the granularity already settles.
    The keep still has to clear `keep_only_if_broad_place` and
    `keep_rests_on_a_verified_type` via `_guard_unsafe`, so this loosens the
    POLICY, never the verification.
    """
    if rule_value is False and llm_value is True:
        ent = getattr(ctx, "entity", None) if ctx is not None else None
        raw = str(getattr(ent, "subtype", "") or "").strip().lower()
        if raw in _RULE_OUTRANKS_TYPES:
            return False
    return True if (rule_value is True or llm_value is True) else False


def _canon_ethnonym(v):
    """The canonical lowercase ethnonym, or None to leave the value untouched.
    Shares `attributes.normalize_ethnonym` with the rule layer and with
    `checks/ethnicity`, so all three agree on the spelling."""
    from .attributes import normalize_ethnonym
    return normalize_ethnonym(v)


# -------------------------------------------------------------- policy registry

POLICIES: dict[str, FieldPolicy] = {

    # ---- PERSON -----------------------------------------------------------
    "gender": FieldPolicy(
        "gender", OPTIONAL, RULE_WINS, C.exact,
        checkers=(chk_gender.not_refuted_by_honorific,
                  chk_gender.not_refuted_by_kin_word,
                  chk_gender.not_refuted_by_pronoun),
        verify_always=True),

    # `role` and `ethnicity` were the two "LLM-only, nothing to check against"
    # fields. Both DO have a rule source -- a kinship-edge detail or a professional
    # construction for `role`, a closed set of self-identification constructions
    # for `ethnicity` -- so both are now ordinary second-lined fields rather than
    # advisory text written in place. See graph/attributes.py.
    "role": FieldPolicy(
        "role", OPTIONAL, RULE_WINS, C.kin_synonym,
        checkers=(chk_person.role_corroborated,), verify_always=True,
        canon=lambda v: str(v).strip().lower().rstrip(".") or None),

    "ethnicity": FieldPolicy(
        "ethnicity", OPTIONAL, RULE_WINS, C.ci,
        checkers=(chk_eth.label_is_known_ethnonym,
                  chk_eth.label_stated_in_transcript,
                  chk_eth.attributed_to_this_person),
        verify_always=True, canon=_canon_ethnonym),

    # `interviewee_honorific_address_agrees` is the only checker here that can
    # POSITIVELY support a value; the other two refute only. Without it every
    # candidate scored zero applicable checks on a transcript with no first-person
    # self-description, so the field was unverifiable in principle -- and under
    # `REQUIRED_VERIFIED` that is now (correctly) a blocking outcome rather than a
    # silently accepted guess. See `_guard_unsafe` and `_ALTERNATIVES`.
    "interviewee_gender": FieldPolicy(
        "interviewee_gender", REQUIRED_VERIFIED, BLOCK, C.exact,
        checkers=(chk_gender.interviewee_self_description_agrees,
                  chk_gender.interviewee_honorific_address_agrees,
                  chk_gender.interviewee_spouse_term_agrees),
        attr="gender", verify_always=True),

    # WHICH named person is the speaker. `unsafe_when` makes EVERY non-empty value
    # consequential, because any of them merges a named person into e000 -- so
    # rule/LLM agreement cannot skip the gate. BLOCK on conflict: abstaining costs
    # one review, merging the wrong person corrupts the whole surrogate identity.
    # REQUIRED_OR_ABSTAIN, not REQUIRED_VERIFIED: a transcript that genuinely never
    # names its speaker is normal (both samples are like this) and must not block.
    "interviewee_identity": FieldPolicy(
        "interviewee_identity", REQUIRED_OR_ABSTAIN, BLOCK, C.exact,
        checkers=(chk_iv.named_in_self_reference_or_address,
                  chk_iv.not_introduced_as_a_relative,
                  chk_iv.not_a_public_figure),
        attr="identity_entity_id",
        unsafe_when=lambda v: bool(v), safe_value=None),

    # `verify_always`: the rule's token split is exactly what these checkers exist to
    # refute ("Father Nguyen" -> given_name "Nguyen", "Papaw Clarence" -> given_name
    # "Papaw"), and on the RULE_WINS conflict path they never saw it. Verifying the
    # resolved value fixes the mis-slotting at the source instead of leaving it to
    # `_cross_field_consistency` to notice after the fact.
    # `part_not_a_titled_surname` is on THIS field only: a lone token behind "Father" /
    # "Dr." / "Mr." is a surname, so the claim is wrong in the given-name slot and
    # right in the surname slot.
    "given_name": FieldPolicy(
        "given_name", REQUIRED_OR_ABSTAIN, RULE_WINS, C.ci,
        checkers=(chk_name.part_is_token_of_mention,
                  chk_name.part_not_a_title,
                  chk_name.part_not_a_kin_word,
                  chk_name.part_not_a_titled_surname), verify_always=True),

    "surname": FieldPolicy(
        "surname", REQUIRED_OR_ABSTAIN, RULE_WINS, C.ci,
        checkers=(chk_name.part_is_token_of_mention,
                  chk_name.part_not_a_title,
                  chk_name.part_not_a_kin_word), verify_always=True),

    "replace": FieldPolicy(
        "replace", REQUIRED_VERIFIED, SAFE_DIRECTION, C.boolean,
        checkers=(chk_person.personal_signal_absent,
                  chk_person.not_kin_of_interviewee),
        safer=_safer_replace, unsafe=False, safe_value=True),

    "subtype_person": FieldPolicy(
        "subtype_person", OPTIONAL, RULE_WINS, C.upper,
        checkers=(chk_person.subtype_corroborated,), attr="subtype",
        verify_always=True),

    # ---- LOCATION ---------------------------------------------------------
    # `verify_always` is safe here: every type in data/gazetteer.csv canonicalizes
    # under `comparators.LOC_CANON`, so verifying the rule's own value cannot discard
    # a gazetteer hit -- it only refuses a type outside the vocabulary, whichever
    # layer produced it. `replace_location`'s keep gate depends on this type, so a
    # type nothing can canonicalize must not silently support a keep.
    "subtype_location": FieldPolicy(
        "subtype_location", OPTIONAL, RULE_WINS, C.loc_type,
        checkers=(chk_loc.type_in_enum, chk_loc.type_corroborated),
        attr="subtype", verify_always=True),

    # Whether a PLACE NAME must be replaced. Nothing decided this before: location
    # entities reached surrogate generation with no `replace` key at all, so a
    # consumer keying off `replace` redacted every person and kept every place --
    # and "Red Jacket" plus an age plus "miner" identifies one household. Shaped
    # exactly like the PERSON `replace` policy, for the same reason: keeping is the
    # leak-prone direction, so disagreement resolves toward more redaction and the
    # KEEP direction must clear its checkers however it was reached.
    # REQUIRED_OR_ABSTAIN rather than REQUIRED_VERIFIED, unlike PERSON `replace`.
    # The rule always supplies a value here, so the only outcome a verified tier
    # would newly block is KEEP -- "the gazetteer decided and the model said
    # nothing" -- and for a place that resolves to replace=True that is pure noise:
    # the safe outcome was already taken. The dangerous direction is covered
    # regardless of tier, because `unsafe=False` makes `_guard_unsafe` run both
    # checkers on ANY keep however it was reached. Same reasoning as
    # `interviewee_identity`.
    "replace_location": FieldPolicy(
        "replace_location", REQUIRED_OR_ABSTAIN, SAFE_DIRECTION, C.boolean,
        checkers=(chk_loc.keep_only_if_broad_place,
                  chk_loc.keep_rests_on_a_verified_type),
        safer=_safer_replace_location, attr="replace", unsafe=False,
        safe_value=True),

    "location_parent": FieldPolicy(
        "location_parent", OPTIONAL, RULE_WINS, C.ci,
        checkers=(chk_loc.parent_no_placeholder,
                  chk_loc.parent_not_self,
                  chk_loc.parent_resolves)),

    # ---- DATES ------------------------------------------------------------
    "resolved_value": FieldPolicy(
        "resolved_value", REQUIRED_OR_ABSTAIN, RULE_WINS, C.date_close(31),
        checkers=(chk_date.iso_valid,
                  chk_date.granularity_respected,
                  chk_date.not_after_interview,
                  chk_date.dob_plausible,
                  chk_date.anchor_table_not_contradicted),
        verify_always=True),

    "shiftable": FieldPolicy(
        "shiftable", OPTIONAL, SAFE_DIRECTION, C.boolean,
        checkers=(chk_date.is_real_public_event,), safer=_safer_shiftable,
        unsafe=False, safe_value=True),

    # Was the ONLY policy in this registry with `checkers=()` -- so it had both
    # layers on paper while an LLM fill was accepted with the reason "NO
    # deterministic check applied to this value" (16 times on the two samples).
    # The answer is written in the source text, so it is cheaply checkable.
    # `verify_always`, and note it must fire on `False` too -- `_empty(False)` is
    # False, so "this date is exact" is a real value that the source text can refute.
    "approximate": FieldPolicy(
        "approximate", OPTIONAL, RULE_WINS, C.boolean,
        checkers=(chk_approx.source_hedge_agrees,
                  chk_approx.age_parser_agrees,
                  chk_approx.relative_date_is_approximate,
                  chk_approx.anchor_in_table_is_exact),
        verify_always=True),

    # ---- AGE --------------------------------------------------------------
    "value": FieldPolicy(
        "value", REQUIRED_OR_ABSTAIN, RULE_WINS, C.age_close(1),
        checkers=(chk_age.plausible_range, chk_age.consistent_with_dob,
                  chk_age.not_a_measurement),
        verify_always=True),

    # ---- IDENTIFIERS ------------------------------------------------------
    # BOTH directions are checked. Claiming the INTERVIEWEE owns an identifier is
    # the consequential direction, so that value must clear its checkers however it
    # was reached (`unsafe`). But `owner="other"` is not free either: it is what
    # EXCLUDES a span from the speaker's surrogate identity, and it used to be
    # accepted with no evidence at all, so it now needs positive proof that
    # somebody else is the referent. With neither direction provable the field
    # stays empty -- and for the categories in `_OWNER_VERIFIED_CATS` an empty
    # owner BLOCKS (tier promoted at call time in the driver below), because
    # "unknown owner" on the speaker's phone is a decision for a human.
    # `unsafe_when` marks EVERY non-empty owner consequential, not just
    # "interviewee". The module docstring in graph/checks/ownership.py says "BOTH
    # DIRECTIONS ARE CHECKED", but with `unsafe="interviewee"` that held only on the
    # FILL path: an `owner="other"` the rules asserted, or that survived a
    # RULE_WINS conflict, reached the graph with its two `other`-direction checkers
    # never run -- and `other` is what EXCLUDES a span from the speaker's surrogate
    # identity, which is not a free direction either.
    "owner": FieldPolicy(
        "owner", REQUIRED_OR_ABSTAIN, RULE_WINS, C.exact,
        checkers=(chk_own.first_person_cue_present,
                  chk_own.no_kin_noun_between,
                  chk_own.no_nearer_named_person,
                  chk_own.no_third_person_subject,
                  chk_own.third_party_identifiable,
                  chk_own.not_bound_by_first_person),
        unsafe_when=lambda v: bool(v), safe_value=None),

    # `unsafe_when` here means the span is re-normalized under its accepted kind
    # WHATEVER the outcome, not only on a fill. The derived sub-attributes
    # (`digits`, `local`, `domain`, `handle`) are minted from that normalization and
    # feed surrogate generation, so a category nothing re-verified is a category
    # that can hand the generator a malformed value.
    "kind": FieldPolicy(
        "kind", OPTIONAL, RULE_WINS, C.id_kind,
        checkers=(chk_id.kind_is_known, chk_id.kind_renormalizes),
        attr="category", unsafe_when=lambda v: bool(v), safe_value=None),

    # Whether an OCCUPATION is rare enough to help identify someone. Previously
    # LLM-only AND unchecked, and the model duly returned True for seven of the
    # nine occupations across both transcripts -- a signal that fires on everything.
    # The rule layer is `identifiers.COMMON_OCCUPATIONS`; RULE_WINS so the common
    # list is authoritative and the LLM only fills what it does not cover.
    "identifying": FieldPolicy(
        "identifying", OPTIONAL, RULE_WINS, C.boolean,
        checkers=(chk_id.identifying_only_for_occupation,
                  chk_id.identifying_not_a_common_occupation),
        verify_always=True),

    # ---- AGE <-> DATE pairing (STATED_WITH) -------------------------------
    # `verify_always`: the rule here is a bare positional guess (the nearest date in
    # the sentence), and it is the value that survives when the model stays quiet --
    # so before this the arithmetic constraint the date-shifter relies on was the one
    # thing NOTHING ever checked. A refuted pairing now drops its edge, see
    # `resolve_all`.
    "stated_with": FieldPolicy(
        "stated_with", OPTIONAL, RULE_WINS, C.ci,
        checkers=(chk_sw.anchor_is_a_date_entity,
                  chk_sw.anchor_is_near,
                  chk_sw.implied_birth_year_plausible),
        verify_always=True),

    # ---- RELATIONS -------------------------------------------------------
    # Resolved per PAIR rather than per entity (see _resolve_relations). The
    # comparator canonicalizes kin synonyms so the rule's "mama" and the model's
    # "mother" count as agreement rather than a conflict.
    #
    # NO `verify_always`, on the same grounds as `location_parent`. `locally_provable`
    # is a PROVABILITY gate, not a truth test -- its own failure text is "plausible but
    # not locally provable" -- and the rule patterns in `extract_kinship` legitimately
    # produce details the verifier's vocabulary cannot score ("mother's sister",
    # "Papaw"), which it reports as unprovable rather than false. Verifying rule edges
    # with it would delete correct relations, which is the opposite of the point.
    "relation": FieldPolicy(
        "relation", OPTIONAL, RULE_WINS, C.kin_synonym,
        checkers=(chk_rel.locally_provable,)),

    # ---- IDENTITY / CLUSTERING -------------------------------------------
    # Are these two PERSON entities one human? Resolved per PAIR, like `relation`.
    # Alias / nickname / coref merges were the one remaining class outside this
    # registry, so they produced no Resolution, no provenance and no ledger row.
    # They do now. Merging is still never automatic from the LLM -- a checked
    # proposal becomes a `suggested_merge_with` flag (see `_resolve_merges`) -- but
    # the decision is at last recorded and reviewable like every other field.
    #
    # `unsafe_when=(v is True)`: a MERGE is the consequential direction (it changes
    # who the graph thinks exists), so it must clear the checkers however it was
    # reached. Without this the checkers never ran on a rule/coref merge at all --
    # the rule value is `True` and the LLM value is always `True`, so `C.boolean`
    # always agreed and every applied merge resolved `confirm`, which skips them.
    # A split (`False`) is the safe direction and needs no verification.
    #
    # `SAFE_DIRECTION` rather than RULE_WINS, with "keep them separate" as the safer
    # value. This is what lets the containment veto in `merge_strings` be a POLICY
    # instead of an out-of-band side effect: the rule says "merge" (the bare name
    # matched exactly one full name), the LLM says "different people", and the
    # disagreement resolves the way the veto already behaves -- no merge -- but now
    # with a Resolution, a provenance record and a ledger row behind it.
    "same_person": FieldPolicy(
        "same_person", OPTIONAL, SAFE_DIRECTION, C.boolean,
        checkers=(chk_merge.quote_is_transcript_text,
                  chk_merge.quote_grounds_the_pair,
                  chk_merge.names_share_a_token,
                  chk_merge.alias_cue_present,
                  chk_merge.genders_do_not_conflict,
                  chk_merge.not_co_occurring_without_a_cue),
        safer=lambda rule_value, llm_value, ctx=None: bool(rule_value) and bool(llm_value),
        unsafe_when=lambda v: v is True, safe_value=None),
}

# A relative date is an estimate, so agreement is judged loosely; an absolute date
# is not. Handled by swapping the comparator per category at call time.
_DATE_TOL = {"DATE_RELATIVE": 60, "DATE_ABSOLUTE": 31, "DATE_OF_BIRTH": 31,
             "DATE_ANCHOR": 3}


# ---------------------------------------------------------------- apply / ledger

def _legacy_mirror(ent, res: Resolution, policy: FieldPolicy) -> None:
    """Keep writing the old `suggested_*` / `*_confirmed` keys.

    The HTML report and the demo scripts read those keys. Mirroring them means the
    unification can land without touching the reporting layer, and a reviewer sees
    the same page with better decisions behind it.
    """
    a = ent.attributes
    f = policy.field
    if res.action == CONFIRM:
        if f in ("gender", "interviewee_gender"):
            a["gender_confirmed"] = True
        elif f == "resolved_value":
            a["date_confirmed"] = True
        elif f == "value":
            a["age_confirmed"] = True
        elif f == "ethnicity":
            # the report reads `suggested_ethnicity`, so mirror a confirmed value
            # too -- otherwise a rule value the LLM agreed with would be the one
            # outcome that disappeared from the page
            a["suggested_ethnicity"] = res.value
            a["ethnicity_confirmed"] = True
        return
    if res.action == FILL:
        if f in ("gender", "interviewee_gender"):
            a["suggested_gender"] = res.value
        elif f == "subtype_person":
            a["suggested_subtype"] = res.value
            a["suggested_subtype_confidence"] = res.confidence
        elif f == "subtype_location":
            a["suggested_type"] = res.value
            a["suggested_type_confidence"] = res.confidence
        elif f == "location_parent":
            a["suggested_parent"] = res.value
        elif f in ("resolved_value", "value"):
            a["suggested_value"] = res.value
            a["suggested_value_confidence"] = res.confidence
        elif f == "kind":
            a["suggested_kind"] = res.value
        elif f == "role":
            a["suggested_role"] = res.value
        elif f == "ethnicity":
            # the report reads `suggested_ethnicity`; `ethnicity_basis` now records
            # that a CHECKER verified the label, not merely that the model claimed to
            # have read it somewhere
            a["suggested_ethnicity"] = res.value
            a["ethnicity_basis"] = "stated" if res.checks_passed else "unverified"
        return
    if res.action == CONFLICT and f in ("resolved_value", "value"):
        # only when there IS a competing value to show. Writing the key with None put a
        # null-valued attribute on the entity, which reads to a consumer as "the LLM
        # checked this and answered nothing" rather than "no comparison was recorded".
        other = None if res.source == "llm_checked" else res.value
        if other is None:
            a.pop("llm_check_value", None)
        else:
            a["llm_check_value"] = other


# Attributes the RULE layer writes ALONGSIDE a field's own value, which must be
# erased with it. `apply_resolution` popped only `policy.attr`, so a refuted
# `ethnicity` left `ethnicity_basis="stated"` and an `ethnicity_evidence` quote on the
# entity -- a consumer reading the basis key saw a verified claim with no value behind
# it. Same shape for the speaker's `gender_evidence`, for an anchor's `anchor_event`,
# and for the identifier sub-attributes minted from a `kind` the checkers refused.
_COMPANION_ATTRS: dict[str, tuple] = {
    "ethnicity": ("ethnicity_basis", "ethnicity_evidence", "suggested_ethnicity",
                  "ethnicity_confirmed"),
    "gender": ("gender_evidence", "suggested_gender", "gender_confirmed"),
    "interviewee_gender": ("gender_evidence", "suggested_gender",
                           "gender_confirmed"),
    "resolved_value": ("anchor_event", "suggested_value", "date_confirmed"),
    "value": ("suggested_value", "age_confirmed"),
    "role": ("suggested_role",),
    "kind": ("kind", "digits", "local", "domain", "handle", "occupation"),
    "location_parent": ("suggested_parent",),
    "stated_with": (),
}


def apply_resolution(ent, res: Resolution, policy: FieldPolicy) -> None:
    """Write the resolved value, its provenance, and a flag when a human is needed."""
    if not hasattr(ent, "provenance"):
        ent.provenance = {}
    ent.provenance[policy.field] = res

    attr = policy.attr or policy.field
    if res.action in (FILL, CONFIRM) or (res.action == CONFLICT and res.value is not None):
        if attr == "subtype":
            ent.subtype = str(res.value).upper() if res.value is not None else None
        elif attr == "category":
            pass                              # never re-type an entity from a suggestion
        else:
            ent.attributes[attr] = res.value
    elif res.value is None and res.action in (CONFLICT, REJECT):
        # ABSTENTION MUST ERASE. The rules write their answers onto the entity before
        # arbitration runs (`attributes.setdefault`, `pipeline._link_interviewee_pii`,
        # the gazetteer's `e.subtype`), so a resolution that concludes "no verified
        # value" used to leave the refuted rule value sitting on the entity for a
        # consumer to read. That is the difference between a checker that reports a
        # problem and a checker that prevents one: a refuted `owner="interviewee"`
        # still handed the speaker somebody else's phone number.
        #
        # Only ever erases when the second line landed on NO value -- a conflict that
        # resolved to a real value (`safe_value=True` on `replace`) is written above.
        if attr == "subtype":
            ent.subtype = None
        elif attr == "category":
            pass                              # the category is the detector's, not ours
        else:
            ent.attributes.pop(attr, None)
        # ...and the rule's COMPANION keys, or the erasure is only half done
        for k in _COMPANION_ATTRS.get(policy.field, ()):
            ent.attributes.pop(k, None)

    # `kind` acceptance regenerates the derived sub-attributes from the rule
    # regexes rather than leaving stale ones in place. Also on CONFIRM: the
    # sub-attributes feed surrogate minting, so they are regenerated whenever the
    # category was re-verified, not only when the LLM supplied it.
    if policy.field == "kind" and res.action in (FILL, CONFIRM):
        from .checks.identifiers import renormalized_attrs
        ent.attributes.update({k: v for k, v in
                               renormalized_attrs(res.value, _CTX[0]).items()
                               if k not in ("replace",)})

    _legacy_mirror(ent, res, policy)

    if res.action == CONFIRM:
        return
    if res.action == FILL:
        n = len(res.checks_passed)
        if res.source == "checker_derived":
            # Not an LLM fill: neither layer gave a usable answer and the checkers
            # themselves singled this value out (see `_try_alternatives`). Saying
            # "filled from the LLM" here would misreport where the value came from.
            ent.flag_entity(f"{policy.field}: neither layer produced a usable value; "
                            f"{n} deterministic check(s) single out {res.value!r} and "
                            f"refute the alternatives; review")
        elif n:
            ent.flag_entity(f"{policy.field}: filled from the LLM; {n} deterministic "
                            f"check(s) verified it (confidence {res.confidence}); "
                            f"review")
        else:
            # No checker inspected this value. Say so: the old wording counted
            # non-applicable checkers as passes, which advertised an unverified
            # fill as a checked one.
            ent.flag_entity(f"{policy.field}: filled from the LLM with NO "
                            f"deterministic check applicable to this value "
                            f"(confidence {res.confidence}); review")
        return
    if res.action == CONFLICT:
        ent.flag_entity(f"{policy.field}: {res.reason}"
                        + ("; BLOCKING" if res.blocking else ""))
        return
    if res.action == REJECT and res.reason:
        ent.flag_entity(f"{policy.field}: no verified value -- {res.reason}"
                        + ("; BLOCKING" if res.blocking else ""))
        return
    if res.action == KEEP and res.blocking:
        ent.flag_entity(f"{policy.field}: rule value is unconfirmed and this field "
                        f"must be verified; BLOCKING")


# `apply_resolution` needs the live context for `kind` re-normalization; kept in a
# one-slot list so the module stays import-cycle free.
_CTX: list = [None]


def _drop_edges(edges: list, new_edges: list, relation, source_id: str) -> None:
    """Remove every `relation` edge leaving `source_id`, from both edge lists.

    The counterpart to "abstention must erase" in `apply_resolution`: a rule stage
    can write an EDGE before arbitration runs, and dropping the attribute while
    leaving the edge would just move the stale claim somewhere a consumer still
    walks. Mutates `edges` in place because the caller's list is the pipeline's.
    """
    for bucket in (edges, new_edges):
        bucket[:] = [e for e in bucket
                     if not (e.relation == relation and e.source == source_id)]


# ------------------------------------------------------------------- the driver

_PERSON_FIELDS = ("gender", "given_name", "surname", "replace", "subtype_person",
                  "role", "ethnicity")
_DATE_CATS = ("DATE_ABSOLUTE", "DATE_RELATIVE", "DATE_ANCHOR", "DATE_OF_BIRTH")
_ID_CATS = ("PHONE", "EMAIL", "SSN_OR_ID", "USERNAME_HANDLE", "OCCUPATION")
_OWNED_CATS = _ID_CATS + ("AGE", "DATE_OF_BIRTH")

# Categories whose OWNER feeds the interviewee's surrogate identity directly. On
# these, an owner neither layer could establish is a BLOCKING gap rather than a
# silent None -- a phone number nobody can attribute is exactly the case a human
# must settle before surrogates are minted. OCCUPATION is deliberately left at
# REQUIRED_OR_ABSTAIN: transcripts mention many jobs belonging to many people, and
# blocking on each would bury the rows that matter.
_OWNER_VERIFIED_CATS = ("PHONE", "EMAIL", "SSN_OR_ID", "USERNAME_HANDLE",
                        "DATE_OF_BIRTH", "AGE")


def _rule_value(ent, policy: FieldPolicy, ctx):
    attr = policy.attr or policy.field
    if policy.field == "replace_location":
        # Recomputed from the CURRENT subtype rather than read from the attribute.
        # `location_dates.infer_location_replace` runs in the pipeline, before the
        # second line has typed the off-gazetteer places, so at that point every
        # unknown place defaults to replace=True. Reading that stale default here
        # made the rule layer contradict its own threshold: "Mekong Delta" resolved
        # to REGION (a keepable type) and the rule still said replace, purely
        # because the gazetteer had not heard of it. `subtype_location` is resolved
        # BEFORE this field (see `_fields_for`), so by now the type is the verified
        # one, and the rule is a pure function of it.
        from .location_dates import BROAD_LOCATION_TYPES
        if ent.category == "INSTITUTION":
            return True
        raw = str(ent.subtype or "").strip().lower()
        if raw:
            return raw not in BROAD_LOCATION_TYPES
        return ent.attributes.get("replace", True)
    if attr == "subtype":
        return ent.subtype
    if attr == "category":
        # `kind`. The rule's answer is the detector's category -- but ONLY when the
        # rule's own normalizer accepts the span under it. When `_normalize` raised a
        # flag ("email failed to parse", "phone has too few digits", "SSN_OR_ID has no
        # digits"), the rules have no trustworthy answer and must say so, which is
        # what opens the FILL path so the LLM can re-type the span and
        # `checks/identifiers.kind_renormalizes` can gate the guess.
        #
        # Reading `ent.category` unconditionally made `kind` the one field whose FILL
        # branch was unreachable: the value was never empty, so a malformed span could
        # only ever be flagged, never corrected, and `_legacy_mirror`'s
        # `suggested_kind` was dead code. The entity's CATEGORY is still never
        # rewritten (see `apply_resolution`) -- what a checked fill supplies is
        # `attributes["kind"]` and the derived sub-attributes the surrogate generator
        # actually mints from.
        from .checks.identifiers import _NORMALIZABLE
        if ent.category in _NORMALIZABLE and ent.mentions:
            from .identifiers import _normalize
            _sub, _attrs, flag = _normalize(ent.category, ent.mentions[0].text)
            if flag:
                return None
        return ent.category
    if attr == "stated_with":
        # the rule value lives on the STATED_WITH edge age_date_constraints built
        for ed in ctx.edges:
            if ed.relation == Relation.STATED_WITH and ed.source == ent.entity_id:
                tgt = next((e for e in ctx.entities if e.entity_id == ed.target), None)
                if tgt is not None and tgt.mentions:
                    return tgt.mentions[0].text
        return None
    return ent.attributes.get(attr)


def _fields_for(ent, interviewee) -> list[str]:
    """Which policies apply to this entity.

    ORDER MATTERS within a category: a later field's checkers may read the value an
    earlier one settled. `replace_location` must come after `subtype_location`
    because the keep gate is a function of the resolved geographic type, and
    `ethnicity` comes last on a person so its checkers see final mentions.
    """
    if ent is interviewee:
        # `interviewee_identity` is resolved EARLY (graph/interviewee.py) because the
        # merge has to happen before kinship and attributes run; the driver records
        # its Resolution rather than re-resolving it here. Name parts only exist as
        # a field once identification actually found a name.
        out = ["interviewee_gender"]
        if getattr(ent, "mentions", None):
            out += ["given_name", "surname"]
        # the speaker's OWN ethnicity is second-lined like anyone else's -- it feeds
        # surrogate name selection, so an unchecked guess is not acceptable here
        out.append("ethnicity")
        return out
    # NICKNAME cannot reach here today (merge_person_mentions relabels every
    # PERSON/NICKNAME mention "PERSON"), but treating it as a person keeps this
    # defensive rather than silently returning no fields if that ever changes.
    if ent.category in ("PERSON", "NICKNAME"):
        return list(_PERSON_FIELDS)
    if ent.category in ("LOCATION", "INSTITUTION"):
        return ["subtype_location", "location_parent", "replace_location"]
    if ent.category in _DATE_CATS:
        # `shiftable` is set by the rule for EVERY date category
        # (location_dates.resolve_date_entity), so every category needs the second
        # line. Previously only DATE_ANCHOR was arbitrated, which left the rule's
        # `shiftable=True` on absolute, relative and DOB dates double-checked by
        # nothing at all.
        out = ["resolved_value", "approximate", "shiftable"]
        if ent.category == "DATE_OF_BIRTH":
            out += ["kind", "owner"]
        return out
    if ent.category == "AGE":
        return ["value", "approximate", "kind", "owner", "stated_with"]
    if ent.category in _ID_CATS:
        out = ["kind", "owner"]
        if ent.category == "OCCUPATION":
            out.append("identifying")
        return out
    return []


def _resolve_relations(ctx, edges, ledger, relation_proposals, llm_ran):
    """Arbitrate RELATED_TO pairs, then let the deterministic kin rule run again.

    Resolved per PAIR: the rule value is the detail of the kinship edge the rules
    already produced for that pair, the LLM value is the raw proposal, and the
    checker is the verifier in `graph/checks/relations.py`. Runs BEFORE the
    per-entity field loop so that a relation the second line adds is visible to the
    kin-derived FAMILY subtype and to every checker that reads `kin_ids`.
    """
    iv = ctx.interviewee
    rule_rel = {(e.source, e.target): e for e in edges
                if e.relation == Relation.RELATED_TO}
    llm_rel = {}
    for r in (relation_proposals or []):
        llm_rel[(r["source"], r["target"])] = r

    policy = POLICIES["relation"]
    new_edges = []
    for pair in list(rule_rel) + [p for p in llm_rel if p not in rule_rel]:
        src, tgt = pair
        prop = llm_rel.get(pair)
        rule_edge = rule_rel.get(pair)
        ctx.pair = (src, tgt, (prop or {}).get("evidence", ""))
        ctx.relation_verdict = None
        ctx.entity = ctx.ent_by_id.get(tgt) or ctx.ent_by_id.get(src)
        res = second_line(policy,
                          rule_edge.detail if rule_edge else None,
                          {"value": prop["detail"], "confidence": prop["confidence"]}
                          if prop else None,
                          ctx, llm_ran=llm_ran)

        if res.action == FILL:
            v = ctx.relation_verdict
            new_edges.append(Edge(source=v.source, target=v.target,
                                  relation=Relation.RELATED_TO, detail=v.detail,
                                  evidence=f"(llm) {v.evidence}"))
        elif res.action == REJECT and res.checks_failed:
            # a proposal that was merely UNPROVABLE (not refuted) still reaches a
            # human, tagged on the named person -- same policy as before the move.
            v = ctx.relation_verdict
            if v is not None and v.action == "suggest":
                named = ctx.ent_by_id.get(v.target)
                other = ctx.ent_by_id.get(v.source)
                if named is iv:
                    named, other = other, named
                if named is not None and named is not iv:
                    with_nm = ("the interviewee" if other is iv
                               else (other.sorted_mentions[0]
                                     if other and other.sorted_mentions else "someone"))
                    named.attributes.setdefault(
                        "suggested_relation",
                        {"detail": v.detail, "with": with_nm,
                         "evidence": v.evidence[:120]})
                    named.flag_entity(
                        f"LLM suggests relation '{v.detail}' with {with_nm} but it "
                        f"couldn't be verified locally; review")

        key = f"relation:{src}" if ctx.entity and ctx.entity.entity_id == tgt \
            else f"relation:{tgt}"
        if ctx.entity is not None:
            if not hasattr(ctx.entity, "provenance"):
                ctx.entity.provenance = {}
            ctx.entity.provenance[key] = res
            ledger.setdefault(ctx.entity.entity_id, {})[key] = res

    ctx.pair, ctx.relation_verdict, ctx.entity = None, None, None

    # deterministic rule, re-applied now that relations are final: a person tied to
    # someone by a KIN relation is FAMILY unless the rules already typed them.
    for ed in edges + new_edges:
        if ed.relation != Relation.RELATED_TO:
            continue
        if C.kin_canon(ed.detail) not in KIN_WORDS and ed.detail.lower() not in KIN_WORDS:
            continue
        for eid in (ed.source, ed.target):
            e = ctx.ent_by_id.get(eid)
            if e is not None and e is not ctx.interviewee and e.subtype is None:
                e.subtype = "FAMILY"
    return new_edges


def _resolve_merges(ctx, merge_records, ledger, llm_ran) -> None:
    """Arbitrate same-person (alias / nickname / coref) claims.

    `merge_records` is a list of plain dicts, one per CLAIMED pair:

        {"a": kept_entity_id, "b": other_entity_id, "evidence": str,
         "source": "rule" | "llm", "confidence": str,
         "applied": bool,      # did this claim change the entity set?
         "value": bool}        # an LLM record's verdict; defaults to True

    A record with `source="rule"` is a claim the rule layer made -- one it CARRIED OUT
    (`applied=True`, alias and coref merges) or one it wanted and the LLM vetoed
    (`applied=False`, the containment veto in `merge_strings`). A record with
    `source="llm"` is the model's own answer: `value=True` for an alias/containment
    proposal, `value=False` for a veto. Both sources for one pair means the field was
    filled by the rules and DOUBLE-CHECKED by the model, which is the ordinary
    two-layer shape -- coref's LLM double-gate emits exactly that pair of records, so
    an LLM-confirmed coref merge is no longer recorded as "rules stand; no LLM answer
    to confirm them".

    NOTHING here merges. A checked LLM proposal becomes `suggested_merge_with` plus
    a review flag -- the same policy as before -- so identity changes remain a
    human decision. What changes is that the decision is now recorded, the
    checkers name themselves, and a reviewer can tell a grounded nickname claim
    from an ungrounded one.
    """
    policy = POLICIES["same_person"]
    by_pair: dict = {}
    for r in (merge_records or []):
        key = (r["a"], r["b"])
        by_pair.setdefault(key, {})[r.get("source", "llm")] = r

    for (a_id, b_id), rec in by_pair.items():
        rule_rec = rec.get("rule")
        llm_rec = rec.get("llm")
        ev = (llm_rec or rule_rec or {}).get("evidence", "") or ""
        ctx.pair = (a_id, b_id, ev)
        ctx.entity = ctx.ent_by_id.get(a_id)
        # `applied` distinguishes a merge the rules CARRIED OUT from a containment
        # merge they wanted and the LLM vetoed. Both are rule claims of "same
        # person"; only the first changed the entity set.
        rule_applied = bool(rule_rec) and rule_rec.get("applied", True)
        # An LLM record carries its own verdict. `extract_pass` aliases always claim
        # True; the containment veto in `merge_strings` claims False, which is the
        # model DOUBLE-CHECKING a filled rule value and disagreeing -- an ordinary
        # conflict, resolved by `same_person`'s safe direction (do not merge).
        llm_value = None if llm_rec is None else llm_rec.get("value", True)
        res = second_line(
            policy,
            True if rule_rec else None,
            {"value": llm_value, "confidence": llm_rec.get("confidence", "unstated")}
            if llm_rec is not None else None,
            ctx, llm_ran=llm_ran)

        a = ctx.ent_by_id.get(a_id)
        b = ctx.ent_by_id.get(b_id)
        a_nm = (a.sorted_mentions[0] if a is not None and a.sorted_mentions else a_id)
        b_nm = (b.sorted_mentions[0] if b is not None and b.sorted_mentions else b_id)

        # An APPLIED merge the checkers refuted. Nothing here can un-merge -- the fold
        # happened in `aliases` / `coref` before this ran -- so the only honest thing
        # is to say so loudly. This case was unreachable before: an applied merge
        # always resolved `confirm`, and `confirm` ran no checkers.
        if rule_applied and res.value is None and res.checks_failed:
            (a if a is not None else b or ctx.interviewee).flag_entity(
                f"same_person: {a_nm} and {b_nm} were MERGED by the rules, but a "
                f"deterministic check refutes it ({', '.join(res.checks_failed)}); "
                f"the merge is already applied -- review and split if wrong")

        # A proposal the rules did not make, that cleared every applicable checker:
        # FLAG it, never merge it.
        if res.action == FILL and res.value is True and a is not None and b is not None:
            a.attributes["suggested_merge_with"] = b_nm
            b.attributes["suggested_merge_with"] = a_nm
            detail = (f"{len(res.checks_passed)} deterministic check(s) verified it"
                      if res.checks_passed else
                      "NO deterministic check applied to this claim")
            a.flag_entity(f"same_person: LLM suggests this is {b_nm}; {detail} "
                          f"(confidence {res.confidence}); review to merge")
            b.flag_entity(f"same_person: LLM suggests this is {a_nm}; {detail} "
                          f"(confidence {res.confidence}); review to merge")
        elif res.action == REJECT and res.checks_failed and a is not None:
            a.flag_entity(f"same_person: an LLM merge suggestion for {b_nm} was "
                          f"refuted ({', '.join(res.checks_failed)}); kept separate")
        elif res.action == CONFLICT and llm_value is False and not rule_applied \
                and a is not None:
            # A veto: the rules/coref proposed the merge and the LLM refused it. Now
            # recorded as a resolution rather than only as a free-text flag written
            # inside the clustering rule (containment) or nowhere at all (coref).
            a.flag_entity(f"same_person: the rules proposed merging this into "
                          f"{b_nm}; the LLM judged them different people and the "
                          f"safe direction kept them separate; review")

        for ent, other in ((a, b_id), (b, a_id)):
            if ent is None:
                continue
            if not hasattr(ent, "provenance"):
                ent.provenance = {}
            ent.provenance[f"same_person:{other}"] = res
            ledger.setdefault(ent.entity_id, {})[f"same_person:{other}"] = res

    ctx.pair, ctx.entity = None, None


def resolve_all(transcript, entities, edges, interviewee, proposals, *,
                interview_date=None, gazetteer=None, gaz_aliases=None,
                relation_proposals=None, llm_ran=True,
                identity_resolution=None, merge_records=None,
                folded_entities=None) -> dict:
    """Arbitrate every in-scope field on every entity.

    `proposals` is `{entity_id: {field: {"value":..., "confidence":...}}}` --
    plain dicts, so `llm_layer` needs no import from `graph`.

    Returns a ledger: `{entity_id: {field: Resolution}}`. New edges that checked
    proposals earned -- RELATED_TO, LOCATED_IN, ATTRIBUTE_OF, STATED_WITH -- come
    back under the `_edges` key.
    """
    ctx = CheckContext(transcript=transcript, entities=entities, edges=edges,
                       interviewee=interviewee, interview_date=interview_date,
                       gazetteer=gazetteer or {}, gaz_aliases=gaz_aliases or {},
                       extra_entities=dict(folded_entities or {}))
    _CTX[0] = ctx

    ledger: dict = {}
    new_edges: list[Edge] = []

    # `interviewee_identity` was resolved before kinship ran (the merge has to
    # precede every stage that reads the entity set), so it is threaded in here to
    # get its ledger row and its blocking status like any other field.
    if identity_resolution is not None:
        ledger.setdefault(interviewee.entity_id, {})["interviewee_identity"] = \
            identity_resolution

    # RELATIONS first: they feed `kin_ids` and the kin-derived FAMILY subtype, both
    # of which later checkers read.
    new_edges += _resolve_relations(ctx, edges, ledger, relation_proposals, llm_ran)
    ctx.edges = edges + new_edges
    ctx.kin_ids = ({e.source for e in ctx.edges if e.relation == Relation.RELATED_TO}
                   | {e.target for e in ctx.edges if e.relation == Relation.RELATED_TO})

    # Same-person claims, recorded and checked. Runs after relations (so a gender
    # inferred from kinship is available to `genders_do_not_conflict`) and before
    # the per-entity loop (so a merge flag is on the entity before its own fields
    # are resolved).
    _resolve_merges(ctx, merge_records, ledger, llm_ran)

    for ent in entities:
        ctx.entity = ent
        per_ent = proposals.get(ent.entity_id, {})
        # A `resolved_value` proposal for an anchor carries the event NAME as an
        # extra. `checks/dates.is_real_public_event` needs it to tell a corroborated
        # public event from any date the rules happened to parse, so surface it
        # before `shiftable` is resolved. Nothing wrote this key before, which is
        # why that checker's fallback was effectively "has a resolved_value".
        rv = per_ent.get("resolved_value") or {}
        if rv.get("event"):
            ent.attributes["suggested_event"] = rv["event"]
        # Same treatment for the ethnicity proposal's evidence quote: it is
        # provenance for a human, not part of the decision, so it is surfaced rather
        # than arbitrated. `ethnicity_basis` is written by the mirror once the
        # CHECKERS have spoken, so it now records verification rather than the
        # model's own claim about where it read the label.
        eth = per_ent.get("ethnicity") or {}
        if eth.get("evidence"):
            ent.attributes.setdefault("ethnicity_evidence", eth["evidence"])
        for fname in _fields_for(ent, interviewee):
            # An AGE span the checkers proved is NOT an age has no owner and no
            # temporal anchor, so those questions are not asked of it. `value` is
            # resolved first (see `_fields_for`), so by now we know.
            #
            # Without this, "the water came up twelve feet" -- an AGE only because the
            # detector tagged every "twelve" -- had its `value` correctly refused by
            # `checks/ages.not_a_measurement` and then went on to acquire
            # `owner="interviewee"`, putting a measurement into the speaker's own
            # identity. Skipping beats adding an ownership checker for it: a refutation
            # there would leave the field empty on a REQUIRED_VERIFIED category and
            # raise a BLOCKING row that no human can resolve, for a span that should
            # never have been an age.
            if ent.category == "AGE" and fname in ("owner", "stated_with") \
                    and ent.attributes.get("value") is None:
                if fname == "owner":
                    # `pipeline._link_interviewee_pii` writes `owner` onto AGE
                    # entities BEFORE arbitration, so skipping the field is not
                    # enough -- the rule's claim has to be erased, the same
                    # "abstention must erase" rule `apply_resolution` applies.
                    ent.attributes.pop("owner", None)
                    ent.flag_entity(
                        "no verified age value, so this span is not treated as an "
                        "age: ownership and the age<->date pairing were not resolved "
                        "for it; check the detector's label")
                    _drop_edges(edges, new_edges, Relation.ATTRIBUTE_OF,
                                ent.entity_id)
                else:
                    _drop_edges(edges, new_edges, Relation.STATED_WITH,
                                ent.entity_id)
                continue
            policy = POLICIES[fname]
            # per-category date tolerance
            if fname == "resolved_value":
                policy = dataclasses.replace(
                    policy, comparator=C.date_close(_DATE_TOL.get(ent.category, 31)))
            # ownership of the speaker's own quasi-identifiers must be VERIFIED, not
            # merely attempted: promote the tier so an unresolved owner blocks.
            #
            # ...but only for a span the SUBJECT actually uttered. An interviewer echo
            # ("You said your father started at fourteen") is not the speaker's data
            # and carries no ownership evidence either way, so demanding a verified
            # owner for it manufactures a blocking row a human cannot resolve. This
            # matters now that AGE entities are per-mention (see
            # `pipeline._simple_entities`): the echo used to ride along in the same
            # entity as the subject's own span, and `checks/ownership._claim` skipped
            # it; on its own it would fail both directions and block.
            if fname == "owner" and ent.category in _OWNER_VERIFIED_CATS and \
                    any(ctx.in_subject_turn(m.start) for m in ent.mentions):
                policy = dataclasses.replace(policy, tier=REQUIRED_VERIFIED)
            res = second_line(policy, _rule_value(ent, policy, ctx),
                              per_ent.get(fname), ctx, llm_ran=llm_ran)
            apply_resolution(ent, res, policy)
            ledger.setdefault(ent.entity_id, {})[fname] = res

            # `subtype` and `replace` must never disagree. When redaction is raised
            # on a name the rules had typed PUBLIC_FIGURE, downgrade the subtype --
            # otherwise a consumer keying off `subtype` keeps a name that a consumer
            # keying off `replace` redacts (the verified Kennedy / Reagan bug).
            if fname == "replace" and res.value is True and ent.subtype == "PUBLIC_FIGURE":
                ent.subtype = "PUBLIC_FIGURE_UNCONFIRMED"

            # An owner the second line could NOT establish must not leave the rule's
            # ATTRIBUTE_OF edge behind. `pipeline._link_interviewee_pii` writes that
            # edge before arbitration, so a claim the checkers refute was still
            # reachable by any consumer walking edges -- the edge half of the same
            # "abstention must erase" rule `apply_resolution` applies to attributes.
            if fname == "owner" and res.value is None:
                _drop_edges(edges, new_edges, Relation.ATTRIBUTE_OF, ent.entity_id)

            # A CHECKED owner becomes a real ATTRIBUTE_OF edge. The rule path
            # (pipeline._link_interviewee_pii) runs BEFORE this, so without these
            # lines an owner the second line filled set `attributes["owner"]` and
            # nothing else -- leaving two classes of ownership with no marker, and
            # the speaker's own phone/email invisible to any consumer walking edges.
            if fname == "owner" and res.value == "interviewee" and \
                    res.action in (FILL, CONFIRM, CONFLICT):
                if not any(e.relation == Relation.ATTRIBUTE_OF
                           and e.source == ent.entity_id
                           and e.target == interviewee.entity_id
                           for e in edges + new_edges):
                    m = ent.mentions[0] if ent.mentions else None
                    new_edges.append(Edge(
                        source=ent.entity_id, target=interviewee.entity_id,
                        relation=Relation.ATTRIBUTE_OF,
                        detail=(ent.subtype or ent.category),
                        evidence=(f"(second line: {res.source}, "
                                  f"{len(res.checks_passed)} check(s) passed) "
                                  + (ctx.sentence_text(m.start).strip() if m else ""))))

            # A pairing the checkers refuted must not leave the rule's positional
            # STATED_WITH edge in place: the date-shifter reads that edge as a hard
            # arithmetic constraint, so a wrong pairing is worse than none.
            if fname == "stated_with" and res.value is None:
                _drop_edges(edges, new_edges, Relation.STATED_WITH, ent.entity_id)

            # A CHECKED age<->date pairing becomes a real STATED_WITH edge, so the
            # arithmetic constraint the date-shifter needs exists even when the
            # rule's positional guess found nothing.
            if fname == "stated_with" and res.action == FILL:
                from .checks.statedwith import date_entity_for
                d = date_entity_for(res.value, ctx)
                if d is not None and not any(
                        e.relation == Relation.STATED_WITH
                        and e.source == ent.entity_id and e.target == d.entity_id
                        for e in edges + new_edges):
                    new_edges.append(Edge(
                        source=ent.entity_id, target=d.entity_id,
                        relation=Relation.STATED_WITH,
                        detail="age and date co-stated; keep arithmetic",
                        evidence=f"(llm pairing, {len(res.checks_passed)} "
                                 f"check(s) passed) {res.value}"))

            # a CHECKED parent suggestion becomes a real LOCATED_IN edge -- the
            # step the old code never took, which is why 002 had zero hierarchy
            if fname == "location_parent" and res.action == FILL:
                from .checks.location import resolved_parent_key
                hit = resolved_parent_key(res.value, ctx)
                if hit is not None:
                    _key, parent_ent = hit
                    if parent_ent is not ent and not any(
                            e.source == ent.entity_id and e.target == parent_ent.entity_id
                            for e in edges + new_edges):
                        new_edges.append(Edge(
                            source=ent.entity_id, target=parent_ent.entity_id,
                            relation=Relation.LOCATED_IN,
                            detail=f"{ent.subtype or 'place'} in "
                                   f"{parent_ent.subtype or 'place'}",
                            evidence="(llm parent, gazetteer-verified)"))

    _cross_field_consistency(entities, ledger)
    ledger["_edges"] = new_edges
    return ledger


def _cross_field_consistency(entities, ledger) -> None:
    """Deterministic checks that span two resolved fields.

    `given_name == surname` is impossible for one person, and it is exactly what a
    single-token honorific-prefixed name produces: the rule puts the surname in the
    given slot ('Father Nguyen' -> given_name 'Nguyen'), while the LLM correctly
    fills `surname` with the same token. The surname slot is the better-evidenced
    one, so the given name abstains rather than carrying a value we can prove wrong.

    This does not fix the underlying rule -- `attributes.infer_person_attributes`
    still mis-slots -- but it stops the wrong value reaching a consumer.
    """
    for ent in entities:
        a = ent.attributes
        gn, sn = a.get("given_name"), a.get("surname")
        if not gn or not sn or str(gn).lower() != str(sn).lower():
            continue
        # POP, not `= None`. Leaving the key present with a null value hands a
        # consumer a `given_name` that exists and is empty, which reads differently
        # from a name the pipeline never established.
        a.pop("given_name", None)
        ent.flag_entity(
            f"given_name and surname both resolved to {gn!r}, which cannot both be "
            f"true; the given name was dropped (the rule mis-slots a surname when a "
            f"title precedes a single-token name)")
        res = ledger.get(ent.entity_id, {}).get("given_name")
        if res is not None:
            ledger[ent.entity_id]["given_name"] = dataclasses.replace(
                res, action=REJECT, value=None,
                checks_failed=res.checks_failed + ("given_surname_collision",),
                reason=f"dropped: identical to surname {sn!r}")


def blocking_fields(ledger: dict) -> list[tuple[str, str, str]]:
    """(entity_id, field, reason) for every resolution that blocks the transcript."""
    out = []
    for eid, fields in ledger.items():
        if eid == "_edges":
            continue
        for fname, res in fields.items():
            if res.blocking:
                out.append((eid, fname, res.reason))
    return out
