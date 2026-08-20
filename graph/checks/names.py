"""
Deterministic checkers for `given_name` / `surname`.

The rule layer splits the longest mention form on whitespace after stripping
titles and kinship words. That produces a surname in the given-name slot for
every honorific-prefixed person -- verified: `Father Nguyen`, `Mr. Landry`,
`Ms. Boudreaux`, `Dr. Combs`, `Brother Estep` all yield `given_name = <surname>`
-- and a kin word in the given-name slot when the kin term is missing from the
strip set (`Papaw Clarence` -> `given_name="Papaw"`).

The checkers are cheap and complete: a proposed name part must actually appear as
a token of one of the entity's mentions (no invention), and must not be a title
or kin word (no mis-slotting).
"""

from __future__ import annotations
import re
from . import CheckOutcome, ok, fail, na
from ..rules.name_matching import KINSHIP_AND_TITLES, HONORIFIC_TITLES

# Kin terms absent from the strip set but common in oral history. Listed here so
# the CHECKER catches the mis-slotting even while the rule table stays as-is.
_EXTRA_KIN = {"mamaw", "papaw", "meemaw", "pawpaw", "pappy", "memaw", "mawmaw"}


def _tokens_of(entity) -> set[str]:
    toks = set()
    for form in getattr(entity, "sorted_mentions", []):
        for raw in re.split(r"[\s,]+", form):
            t = raw.strip("'\".()-").lower()
            if t:
                toks.add(t)
    return toks


def part_is_token_of_mention(value, ctx) -> CheckOutcome:
    """The proposed part must be a token the transcript actually wrote for this
    person -- the model may re-slot tokens, never invent them."""
    name = "part_is_token_of_mention"
    if not value:
        return na(name, "no part claimed")
    if str(value).strip().lower() in _tokens_of(ctx.entity):
        return ok(name)
    return fail(name, f"{value!r} is not a token of any mention of this person")


def part_not_a_title(value, ctx) -> CheckOutcome:
    name = "part_not_a_title"
    if not value:
        return na(name, "no part claimed")
    t = str(value).strip().lower().rstrip(".")
    if t in HONORIFIC_TITLES:
        return fail(name, f"{value!r} is an honorific title, not a name part")
    return ok(name)


def part_not_a_kin_word(value, ctx) -> CheckOutcome:
    name = "part_not_a_kin_word"
    if not value:
        return na(name, "no part claimed")
    t = str(value).strip().lower().rstrip(".")
    if t in KINSHIP_AND_TITLES or t in _EXTRA_KIN:
        return fail(name, f"{value!r} is a kinship/title word, not a name part")
    return ok(name)


def part_not_a_titled_surname(value, ctx) -> CheckOutcome:
    """For the GIVEN-NAME slot only: a lone name token behind a form of address is a
    SURNAME, so it may not be proposed as a given name.

    `name_matching.split_name_parts` applies this reasoning on the rule side, but the
    rule abstaining does not stop the model re-proposing the same mis-slot -- and the
    other three checkers cannot refuse it, because the token really is a token of the
    mention, really is not a title, and really is not a kin word. Verified: with the
    rule fixed, "Father Nguyen" still came out `given_name="Nguyen"`, filled from the
    LLM with "3 deterministic check(s) verified it".

    Registered on `given_name` only (see the policy registry) -- for `surname` the very
    same claim is CORRECT.
    """
    name = "part_not_a_titled_surname"
    if not value:
        return na(name, "no part claimed")
    from ..rules.name_matching import HONORIFIC_TITLES, _TITLE_ONLY, _TITLE_OR_KIN
    titles = HONORIFIC_TITLES | _TITLE_ONLY | _TITLE_OR_KIN
    want = str(value).strip().lower()
    for form in getattr(ctx.entity, "sorted_mentions", []):
        toks = [t for t in re.split(r"[\s,]+", form) if t]
        if len(toks) != 2:
            continue
        lead, tail = toks[0].lower().strip("."), toks[1].lower().strip(".")
        if lead in titles and tail == want:
            return fail(name, f"{form!r} is a form of address plus ONE name token, so "
                              f"{value!r} is this person's surname, not their given name")
    return ok(name)
