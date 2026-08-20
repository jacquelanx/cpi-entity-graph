"""
`conflict_policy=SAFE_DIRECTION` resolvers.

PURPOSE
    Break a rule-vs-LLM tie for the REDACTION DIRECTIVES, where the two possible
    errors cost wildly different amounts. Over-redacting a place name loses some
    colour from a transcript; under-redacting one can identify a household. So
    these functions do not ask "who is more likely right?" -- they ask "which
    answer fails more safely?".

FIT
    Referenced by name from `graph/second_line/policies.py` (as each policy's
    `safer=`) and called by `engine._resolve` on the conflict path. Reads
    `checks/ages.py` and `checks/dates.py` for the two verdicts that let a rule
    outrank the model.

HOW
    The baseline is one line: `True if either side says True else False` -- i.e.
    more redaction wins. Three variants then carve out a NARROW exception where the
    rule's answer is a deterministic, auditable table lookup and the model's is a
    guess about a question the table already settles (a country-level place, a
    nationally-known event, a span proved not to be an age). In every case the
    keep must still clear its checkers via `engine._guard_unsafe`, so these loosen
    the POLICY and never the verification.

One resolver per field shape, named in the policy table.
"""

from __future__ import annotations

from ..checks import ages as chk_age, dates as chk_date


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


def _safer_replace_date(rule_value, llm_value, ctx=None):
    """Safe direction for a DATE expression, tempered by the anchor table.

    Identical to `_safer_replace` except in one case: the rule says KEEP because the
    span names a NATIONALLY known event the table lists, and the model calls the date
    identifying. There the rule wins -- the table is deterministic and auditable, and
    no amount of model nervousness makes "9/11" or "Hurricane Katrina" point at one
    household, while deferring to the model there would strip the historical anchors
    out of every transcript. A REGIONAL table entry gets no such protection (see
    `checks/dates.LOCAL_ANCHOR_EVENTS`), and the keep still has to clear both checkers
    via `_guard_unsafe`, so this loosens the POLICY, never the verification.

    Same shape and same argument as `_safer_replace_location`.
    """
    if rule_value is False and llm_value is True and ctx is not None:
        ent = getattr(ctx, "entity", None)
        ms = getattr(ent, "mentions", None)
        phrase = chk_date.anchor_phrase_for(ms[0].text) if ms else ""
        if phrase and phrase not in chk_date.LOCAL_ANCHOR_EVENTS:
            return False
    return True if (rule_value is True or llm_value is True) else False


def _safer_replace_age(rule_value, llm_value, ctx=None):
    """Safe direction for an AGE span, tempered by the age checkers' own verdict.

    The rule only ever keeps a span a deterministic check REFUTED as an age ("the
    water came up twelve feet"). A model answering "yes, that is a person's age" about
    a number the transcript immediately follows with a unit of measurement is simply
    wrong, and replacing the number would corrupt the narrative to protect nothing --
    so the refutation outranks it. `not_a_measurement` is narrow by construction (it
    reads only the word directly after the span), which is what makes that safe.

    Every other disagreement resolves toward more redaction, and the keep still clears
    both checkers via `_guard_unsafe`.
    """
    if rule_value is False and llm_value is True and ctx is not None:
        if chk_age.age_reading_refuted(getattr(ctx, "entity", None)):
            return False
    return True if (rule_value is True or llm_value is True) else False


def _canon_ethnonym(v):
    """The canonical lowercase ethnonym, or None to leave the value untouched.
    Shares `attributes.normalize_ethnonym` with the rule layer and with
    `checks/ethnicity`, so all three agree on the spelling."""
    from ..rules.attributes import normalize_ethnonym
    return normalize_ethnonym(v)
