"""
THE unified second line. One arbitration point for every field.

PURPOSE
    Every field on every entity is decided in exactly one place: here. A rule
    proposes, the LLM proposes, deterministic checkers verify, and the outcome is
    recorded as a `Resolution` so the decision can be audited afterwards. This
    package is also where the review-blocking list and the ledger come from.

FIT
    Called by `graph/pipeline.run_pipeline` (once, as `resolve_all`) and by
    `graph/rules/interviewee.py` (for its one field, before the main walk).
    Depends on `graph/checks/` for the verification predicates and on
    `graph/models.py`; the `llm_layer` proposals arrive as plain dicts, so the
    dependency stays one-way.

HOW
    See the five questions below, then "WHERE THINGS LIVE" at the bottom of this
    docstring for the file-by-file map.

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
    construction set for `ethnicity` (`graph/rules/attributes.py`); and a
    common-occupation table is a rule for `identifying` (`graph/rules/identifiers.py`).
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

----------------------------------------------------------------------------
WHERE THINGS LIVE. This was one 1600-line module; it is now one package with
the same public surface, split along the seams the docstring above describes:

    outcomes.py        the vocabulary: outcomes, tiers, conflict strategies,
                       `Resolution`, `FieldPolicy`. Pure data, imports nothing.
    engine.py          `second_line` -- decide ONE field. The five questions,
                       `_guard_unsafe`, `_try_alternatives`.
    safe_direction.py  the `conflict_policy=SAFE_DIRECTION` resolvers.
    policies.py        THE registry: one `FieldPolicy` per field.
    apply.py           `apply_resolution` -- write a decision back onto an entity.
    walk.py            `resolve_all` -- drive all of the above over a transcript.

Dependencies run strictly downward in that order; there are no cycles.
"""

from .outcomes import (
    KEEP, CONFIRM, FILL, CONFLICT, REJECT,
    REQUIRED_VERIFIED, REQUIRED_OR_ABSTAIN, OPTIONAL,
    RULE_WINS, SAFE_DIRECTION, BLOCK,
    Resolution, FieldPolicy,
)
from .engine import second_line, owner_survivors
from .policies import POLICIES
from .apply import apply_resolution
from .walk import resolve_all, blocking_fields, _fields_for, _rule_value

__all__ = [
    "KEEP", "CONFIRM", "FILL", "CONFLICT", "REJECT",
    "REQUIRED_VERIFIED", "REQUIRED_OR_ABSTAIN", "OPTIONAL",
    "RULE_WINS", "SAFE_DIRECTION", "BLOCK",
    "Resolution", "FieldPolicy",
    "second_line", "owner_survivors", "POLICIES", "apply_resolution",
    "resolve_all", "blocking_fields",
]
