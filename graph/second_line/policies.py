"""
THE registry: one `FieldPolicy` per field.

PURPOSE
    This table is the answer to "what does this pipeline promise about field X?" --
    its tier, its comparator, its checkers, and what happens when the two layers
    disagree. Read it top to bottom to audit the stage.

FIT
    The declarative half of `graph/second_line/`. `engine.py` implements the
    decision procedure but knows no field names; `walk.py` looks each field up
    here. Imports every module in `graph/checks/` (to name checkers), the
    vocabulary from `outcomes.py`, and the tie-breakers from `safe_direction.py`.
    Adding a field to this pipeline means adding a row HERE, not editing the
    engine.

HOW TO READ A ROW
    `FieldPolicy(name, tier, conflict_policy, comparator, ...)`:

      name             the field, as it appears in `Entity.provenance`.
      tier             REQUIRED_VERIFIED (must end up verified or it BLOCKS
                       review) / REQUIRED_OR_ABSTAIN (verified if present, but may
                       legitimately be absent) / OPTIONAL.
      conflict_policy  who wins a disagreement: RULE_WINS, SAFE_DIRECTION (ask the
                       `safer=` function) or BLOCK (refuse to choose).
      comparator       what counts as the two layers AGREEING -- `C.exact`,
                       `C.ci`, `C.kin_synonym`, `C.date_close(3)`, ...
      checkers=        the deterministic predicates that can refuse a value.
      verify_always=   True when the checkers are TRUTH tests, so they bind on
                       every path and not only on `fill`.
      unsafe= /
      unsafe_when=     which direction is consequential enough to require
                       verification however it was reached.
      safe_value=      what to fall back to when a checker refutes it.
      attr=            the attribute key, when it differs from the field name.
      canon=           canonical spelling for the surviving value.

    Grouped by category (PERSON, then LOCATION, DATE, AGE, identifiers, and the
    two pair-shaped fields `relation` and `same_person`). The per-row comments
    record WHY each choice is what it is; those are the load-bearing part of this
    file.
"""

from __future__ import annotations

from ..checks import comparators as C
from ..checks import ages as chk_age, approximate as chk_approx, dates as chk_date, ethnicity as chk_eth, gender as chk_gender, identifiers as chk_id, interviewee as chk_iv, location as chk_loc, merges as chk_merge, names as chk_name, ownership as chk_own, persons as chk_person, relations as chk_rel, stated_with as chk_sw
from .outcomes import BLOCK, FieldPolicy, OPTIONAL, REQUIRED_OR_ABSTAIN, REQUIRED_VERIFIED, RULE_WINS, SAFE_DIRECTION
from .safe_direction import _canon_ethnonym, _safer_replace, _safer_replace_age, _safer_replace_date, _safer_replace_location, _safer_shiftable


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
    # advisory text written in place. See graph/rules/attributes.py.
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

    # Whether a DATE's SURFACE TEXT must be replaced. Nothing decided this: DATE
    # entities reached surrogate generation with no `replace` key at all -- the exact
    # hole `replace_location` was added to close for places -- so a consumer keying
    # off `replace` redacted every person and every place and then emitted the
    # speaker's date of birth verbatim.
    #
    # `shiftable` is the rule, and the two fields are NOT the same question. Shifting
    # is about the timeline (may the date-shifter move this point?); replacing is
    # about the text (may these words survive?). They agree in the common case, which
    # is why `shiftable` can serve as the rule -- a date the shifter will move cannot
    # survive as written, and a public event it cannot move may stay -- but they come
    # apart on a REGIONALLY famous event, which has a fixed date the shifter must
    # respect AND a phrase that points at one small community. `keep_is_not_a_local_event`
    # is where that difference lives.
    #
    # Resolved AFTER `shiftable` (see `_fields_for`), so the rule is a pure function of
    # a value the second line has already verified -- the same arrangement
    # `replace_location` has with `subtype_location`. REQUIRED_OR_ABSTAIN and
    # `unsafe=False` for the reasons argued at `replace_location`: the rule always
    # supplies a value, so the only outcome a verified tier would newly block is a
    # keep, and the dangerous direction is gated regardless of tier.
    "replace_date": FieldPolicy(
        "replace_date", REQUIRED_OR_ABSTAIN, SAFE_DIRECTION, C.boolean,
        checkers=(chk_date.keep_only_if_public_event,
                  chk_date.keep_is_not_a_local_event),
        safer=_safer_replace_date, attr="replace", unsafe=False, safe_value=True),

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

    # Whether an AGE's SURFACE TEXT must be replaced. AGE was the one category in the
    # graph with NO redaction directive of any kind -- no `replace`, and no
    # `shiftable` either -- so surrogate generation had nothing at all to key off and
    # "Now I'm sixty-eight" went through untouched, next to a holler and an
    # occupation. An age is a quasi-identifier, so the rule defaults to replace and
    # keeps only what a deterministic check proved is not an age (see
    # `checks/ages.age_reading_refuted`).
    #
    # Resolved LAST on an AGE (see `_fields_for`), because the rule reads the `value`
    # Resolution, and that has to exist first.
    "replace_age": FieldPolicy(
        "replace_age", REQUIRED_OR_ABSTAIN, SAFE_DIRECTION, C.boolean,
        checkers=(chk_age.keep_only_if_refuted_as_an_age,
                  chk_age.keep_not_an_explicit_age_phrase),
        safer=_safer_replace_age, attr="replace", unsafe=False, safe_value=True),

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
    # value. This is what lets the containment veto in `name_matching` be a POLICY
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
