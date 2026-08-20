"""
The vocabulary of an arbitration: outcomes, tiers, conflict
strategies, and the two records everything else passes around.

`Resolution` is what ONE field decision produces; `FieldPolicy` is the
declaration of how that field must be decided. Both are pure data -- nothing
here imports the engine, so the policy table can name checkers freely.
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
