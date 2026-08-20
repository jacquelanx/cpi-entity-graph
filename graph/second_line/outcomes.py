"""
The vocabulary of an arbitration: outcomes, tiers, conflict
strategies, and the two records everything else passes around.

PURPOSE
    Defines the words the rest of `second_line` reasons in, plus the two records
    it passes around: `Resolution` (what ONE field decision produced) and
    `FieldPolicy` (the declaration of how that field must be decided).

FIT
    The bottom of this package's dependency stack -- pure data, importing nothing
    from the project, so `policies.py` can freely name checkers from
    `graph/checks/` without a cycle. `engine.py`, `apply.py` and `walk.py` all
    read these types, and the whole vocabulary is re-exported from
    `graph/second_line/__init__.py`.

THE THREE VOCABULARIES
    OUTCOMES -- what happened to a field:
        keep      the rule's value stands, with no LLM answer to confirm it
        confirm   both layers produced the same value
        fill      the rule had nothing; the LLM's value cleared the checkers
        conflict  the layers disagreed; resolved per `conflict_policy`
        reject    no value survives (a checker refused, or both layers were blind)

    TIERS -- how much a field is allowed to abstain:
        REQUIRED_VERIFIED     must end up with a verified value, or it BLOCKS
                              review. For fields where a missing answer is unsafe.
        REQUIRED_OR_ABSTAIN   must be verified if present, but may legitimately be
                              absent.
        OPTIONAL              nice to have; absence is unremarkable.

    CONFLICT POLICIES -- who wins a disagreement:
        rule_wins        the deterministic layer prevails.
        safe_direction   whichever answer is SAFER prevails, per the field's own
                         `safer` function (see `safe_direction.py`). Correct for
                         redaction directives, where "keep" and "replace" carry
                         very different costs.
        block            refuse to choose; the field goes to a human.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


# ---------------------------------------------------------------- outcome types

KEEP, CONFIRM, FILL, CONFLICT, REJECT = "keep", "confirm", "fill", "conflict", "reject"


# tiers
REQUIRED_VERIFIED = "REQUIRED_VERIFIED"


REQUIRED_OR_ABSTAIN = "REQUIRED_OR_ABSTAIN"


OPTIONAL = "OPTIONAL"


# conflict policies
RULE_WINS, SAFE_DIRECTION, BLOCK = "rule_wins", "safe_direction", "block"


class _Unset:
    """Sentinel type for "this policy field was not configured".

    Needed because `None`, `False` and `""` are all legitimate VALUES a policy
    might designate as unsafe, so none of them can double as "unspecified". A
    unique object can. `__repr__` makes it readable in debug output.
    """
    def __repr__(self):
        return "<unset>"


_UNSET = _Unset()


@dataclass(frozen=True)
class Resolution:
    """The record of one field decision -- the unit of provenance in this codebase.

    Stored on `Entity.provenance[field]` and serialized into the artifact, so a
    reviewer can see not just WHAT the graph concluded but how. Frozen, because a
    decision is a historical fact.

    Fields worth calling out:
      `action`     one of the five outcomes above.
      `source`     which layer the surviving value came from -- "rule",
                   "rule_confirmed" (rule value, LLM agreed) or "llm_checked".
      `blocking`   True when this field must reach a human before surrogates are
                   minted. `walk.blocking_fields` collects these.
      the three check tuples -- see `verified` below, and note that
      `checks_skipped` exists precisely so a value nothing examined cannot
      advertise itself as checked.
    """
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
        refuted it.

        The distinction that matters: "no checker refuted it" is NOT verification
        if no checker looked. `checks_passed` must be non-empty, so a value every
        checker skipped reads as unverified -- which is the honest answer.
        """
        return bool(self.checks_passed) and not self.checks_failed

    def to_dict(self) -> dict:
        """JSON-safe view for the artifact. Tuples become lists; `verified` is
        omitted because it is derivable from the two check lists."""
        return {"field": self.field, "action": self.action, "value": self.value,
                "source": self.source, "confidence": self.confidence,
                "checks_passed": list(self.checks_passed),
                "checks_failed": list(self.checks_failed),
                "checks_skipped": list(self.checks_skipped),
                "reason": self.reason, "blocking": self.blocking}


@dataclass(frozen=True)
class FieldPolicy:
    """The DECLARATION of how one field must be decided -- a row in the registry.

    `policies.py` holds one of these per field, and `engine.second_line` reads it
    rather than branching on field names. Adding a field means adding a policy,
    not editing the engine.

    The mandatory four: `field` (its name), `tier` (how much abstention is
    allowed), `conflict_policy` (who wins a disagreement) and `comparator` (what
    counts as the two layers AGREEING -- never defaulted, because a missing
    comparator would make `confirm` silently unreachable).

    The optional rest, each documented at its own line below: `checkers`, `safer`,
    `attr`, the `unsafe` / `safe_value` / `unsafe_when` trio for designating a
    field's consequential direction, `canon` for normalizing the surviving value,
    and `verify_always`.
    """
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
