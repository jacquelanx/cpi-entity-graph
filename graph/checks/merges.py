"""
Deterministic checkers for `same_person` -- are these two PERSON entities one human?

This is the "structural identity" class the second line used to exclude, on the
grounds that "is the field filled?" is meaningless for a merge. That reasoning
holds for the *question shape*, but not for the consequence: alias, nickname and
coref merges change WHO the graph thinks exists, and they were the one class of
decision with no Resolution, no provenance and no ledger row. A reviewer could
not see that `Minh` and `Sonny` are the same man, that the rule failed to merge
them, and that the LLM's alias pass missed it too -- the only visible trace was a
clustering number one point lower than it should be.

The analogue of the four-step pattern for a merge is the one the module header of
`graph/second_line/` names: rule proposes, LLM adjudicates, checkers confirm.
That is what these implement. A pair rides on `ctx.pair` the way a relation does
(`(a_entity_id, b_entity_id, evidence_quote)`), and the value is the boolean
"same person".

POLICY: these checkers gate *visibility*, not merging. A proposal that clears them
becomes a `suggested_merge_with` flag for a human, never an automatic merge --
identity changes stay a human decision. The checkers still matter: they are what
separates "the LLM found a real nickname the rules missed" from "the LLM
cross-referenced two ids that do not appear in its own evidence quote", and only
the former is worth a reviewer's time.

They now run on merges the rules ALREADY APPLIED as well as on proposals. They did
not before: `same_person`'s rule value is `True if a rule merged the pair else None`
and its LLM value is always `True`, so `C.boolean(True, True)` ALWAYS agreed, every
rule/coref merge resolved `confirm`, and `confirm` skips the checkers. The result
was a ledger row with zero verification behind it -- the one outcome the unification
was supposed to make impossible. `same_person` now carries `unsafe_when`, so a true
resolution is verified however it was reached. Two consequences for the checkers
themselves, both handled here:

  * a rule merge has FOLDED one entity into the other, so `_own_mentions` /
    `_forms` subtract the other side's spans to recover the pre-merge view;
  * a containment or coref merge is not a NICKNAME claim, so it is judged on the
    shared-name route (`names_share_a_token`) rather than being refuted by
    `alias_cue_present` for lacking a cue it was never going to have.
"""

from __future__ import annotations
import re
from . import CheckOutcome, ok, fail, na


def _pair_entities(ctx):
    pair = getattr(ctx, "pair", None)
    if pair is None:
        return None, None, ""
    a_id, b_id, ev = pair
    return ctx.ent_by_id.get(a_id), ctx.ent_by_id.get(b_id), (ev or "")


def _own_mentions(ent, other):
    """`ent`'s mentions MINUS any that belong to `other`.

    These checkers now run on merges the rules ALREADY APPLIED, and a rule merge
    folds `other`'s mentions into `ent` (see `aliases._merge` / `coref._merge_into`).
    Reading `ent.mentions` directly after a fold therefore reports `other`'s spans as
    `ent`'s, which makes every applied merge look like it co-occurs with itself and
    turns `not_co_occurring_without_a_cue` into a guaranteed failure. Subtracting by
    `mention_id` recovers the pre-merge view, so a checker behaves identically
    whether it sees a claim or a completed merge.
    """
    theirs = {m.mention_id for m in getattr(other, "mentions", [])}
    mine = [m for m in getattr(ent, "mentions", []) if m.mention_id not in theirs]
    # A fold can leave NOTHING that is uniquely `ent`'s (the kept entity was itself
    # empty of spans). Fall back to everything rather than reporting "no mentions".
    return mine or list(getattr(ent, "mentions", []))


def _forms(ent, other=None):
    """Distinct surface forms, excluding any contributed by `other` after a fold."""
    if other is None:
        return [f.strip() for f in getattr(ent, "sorted_mentions", []) if f.strip()]
    return sorted({m.text.strip() for m in _own_mentions(ent, other) if m.text.strip()},
                  key=len, reverse=True)


def _tokens(ent, other=None) -> set[str]:
    """Normalized name tokens (titles / kin words stripped) for one side."""
    from ..rules.name_matching import normalize
    toks: set[str] = set()
    for f in _forms(ent, other):
        toks |= set(normalize(f))
    return toks


def _share_a_name_token(a, b) -> str:
    """The shared (or prefix-compatible) name token linking these two, else "".

    This is the OTHER evidence route for a same-person claim. An alias construction
    links two DIFFERENT names ("we called Roberto Beto"); a shared name token links
    two spellings of the SAME name ("Bill" / "Bill Ratliff", "Will" / "William").
    Containment and coref merges rest on this route, and demanding an alias cue of
    them would refute every one of them.
    """
    ta, tb = _tokens(a, b), _tokens(b, a)
    shared = ta & tb
    if shared:
        return sorted(shared)[0]
    for x in ta:
        for y in tb:
            if len(x) >= 3 and len(y) >= 3 and (x.startswith(y) or y.startswith(x)):
                return f"{x}/{y}"
    return ""


def _names_in(ent, text_lower: str, other=None) -> bool:
    """Whole-word match, so a short name cannot ground on a longer one
    ('ruth' must not match inside 'ruthie')."""
    for f in _forms(ent, other):
        if re.search(r"(?<![a-z0-9])" + re.escape(f.lower()) + r"(?![a-z0-9])",
                     text_lower):
            return True
    return False


def _alias_cue_re():
    """The closed alias-construction set the RULE layer owns, reused verbatim so
    the checker and `graph/rules/aliases.py` cannot disagree about what counts as a cue."""
    from ..rules.aliases import _CALL, _AS, _REAL, _BY, _NICK
    return (_CALL, _AS, _REAL, _BY, _NICK)


def quote_is_transcript_text(value, ctx) -> CheckOutcome:
    """The evidence must be real transcript text -- not a paraphrase."""
    name = "quote_is_transcript_text"
    if value is not True:
        return na(name, "not a same-person claim")
    _a, _b, ev = _pair_entities(ctx)
    if not ev:
        return na(name, "no evidence quote supplied (a rule merge carries none)")
    q = re.sub(r"\s+", " ", ev.strip()).lower()
    hay = re.sub(r"\s+", " ", ctx.transcript.lower())
    if len(q) < 4:
        return fail(name, "evidence quote is too short to verify")
    if q not in hay:
        return fail(name, f"evidence {ev[:60]!r} is not verbatim transcript text")
    return ok(name)


def quote_grounds_the_pair(value, ctx) -> CheckOutcome:
    """The evidence quote must actually be about THIS pair.

    Two shapes count, because two valid constructions exist:

      * the quote names BOTH sides ("we called Roberto Beto"); or
      * the quote names ONE side and carries an alias cue, which is what an alias
        construction with a PRONOUN subject looks like -- "we all called her Sissy"
        cannot mention Loretta, because the antecedent is the pronoun.

    Requiring both names unconditionally (as this did) refuted every pronoun-subject
    alias merge the rule layer makes, including the one correct alias merge in the
    sample transcripts. It still kills what it was written for: a quote that names
    neither side is a cross-referenced hallucination whatever else it contains.
    """
    name = "quote_grounds_the_pair"
    if value is not True:
        return na(name, "not a same-person claim")
    a, b, ev = _pair_entities(ctx)
    if a is None or b is None:
        return fail(name, "one side of the pair is not an entity here")
    if not ev:
        return na(name, "no evidence quote supplied")
    evl = re.sub(r"\s+", " ", ev.lower())
    in_a, in_b = _names_in(a, evl, b), _names_in(b, evl, a)
    if in_a and in_b:
        return ok(name, "the quote names both sides")
    if (in_a or in_b) and any(rx.search(ev) for rx in _alias_cue_re()):
        return ok(name, "the quote names one side inside an alias construction "
                        "(pronoun antecedent)")
    if not (in_a or in_b):
        return fail(name, "the evidence quote names neither side of the pair")
    return fail(name, "the evidence quote names only one side and carries no alias "
                      "construction to tie it to the other")


def names_share_a_token(value, ctx) -> CheckOutcome:
    """The SHARED-NAME evidence route: the two sides are two spellings of one name.

    Complementary to `alias_cue_present`, and exactly one of the two applies to any
    given pair -- a pair whose names share a token is a containment/coref claim and
    needs no alias construction; a pair whose names differ entirely is an alias claim
    and needs one. Splitting the routes this way is what stops a legitimate
    'Bill' / 'Bill Ratliff' claim being refuted for lacking a nickname cue, while
    keeping `checks_passed` non-empty for every same-person claim.
    """
    name = "names_share_a_token"
    if value is not True:
        return na(name, "not a same-person claim")
    a, b, _ev = _pair_entities(ctx)
    if a is None or b is None:
        return fail(name, "one side of the pair is not an entity here")
    shared = _share_a_name_token(a, b)
    if not shared:
        return na(name, "the two names share no token; alias_cue_present owns this pair")
    return ok(name, f"both sides are written with the name token {shared!r}")


def alias_cue_present(value, ctx) -> CheckOutcome:
    """Somewhere in the transcript, one of the closed alias constructions must link
    these two -- 'we called Roberto Beto', 'everybody knew him as Big Jim', 'his
    real name was Terrence', 'goes by Debbie', 'nicknamed X'.

    Deliberately checked against the whole transcript rather than the quote: the
    cue and the two names often sit in one long sentence the model quotes only part
    of. A pair with no cue anywhere is a coref guess, not an alias.

    Does NOT apply when the two names share a token: that pair rests on the
    shared-name route (`names_share_a_token`), and an alias construction is not the
    evidence it should be judged against. Before that carve-out this checker
    refuted every containment and coref merge -- which is what those merges are,
    and none of them is a nickname claim.
    """
    name = "alias_cue_present"
    if value is not True:
        return na(name, "not a same-person claim")
    a, b, _ev = _pair_entities(ctx)
    if a is None or b is None:
        return fail(name, "one side of the pair is not an entity here")
    if _share_a_name_token(a, b):
        return na(name, "the two names share a token; this is not an alias claim")
    forms = [f.lower() for f in _forms(a, b) + _forms(b, a)]
    if not forms:
        return na(name, "neither side has a surface form")
    for rx in _alias_cue_re():
        for m in rx.finditer(ctx.transcript):
            hit = m.group(0).lower()
            if any(f in hit for f in forms):
                return ok(name, f"alias construction {m.group(0).strip()[:50]!r}")
    return fail(name, "no alias/nickname construction in the transcript links "
                      "these two names")


def genders_do_not_conflict(value, ctx) -> CheckOutcome:
    """One human has one gender. A conflict is a hard refutation -- the same
    two-way signal `graph/rules/coref.py` already treats as a hard block."""
    name = "genders_do_not_conflict"
    if value is not True:
        return na(name, "not a same-person claim")
    a, b, _ev = _pair_entities(ctx)
    if a is None or b is None:
        return fail(name, "one side of the pair is not an entity here")
    ga = a.attributes.get("gender")
    gb = b.attributes.get("gender")
    if ga and gb and ga != gb:
        return fail(name, f"genders differ ({ga} vs {gb})")
    if ga and gb:
        return ok(name, f"both {ga}")
    return na(name, "at least one side has no gender to compare")


def not_co_occurring_without_a_cue(value, ctx) -> CheckOutcome:
    """Two names in ONE sentence are usually two people ('Sarah's brother Danny').
    They can still be an alias ('we called Roberto Beto'), but only when the
    sentence itself carries the cue -- so co-occurrence without a cue refutes."""
    name = "not_co_occurring_without_a_cue"
    if value is not True:
        return na(name, "not a same-person claim")
    a, b, _ev = _pair_entities(ctx)
    if a is None or b is None:
        return fail(name, "one side of the pair is not an entity here")
    a_sents = {ctx.sentence_bounds(m.start) for m in _own_mentions(a, b)}
    b_sents = {ctx.sentence_bounds(m.start) for m in _own_mentions(b, a)}
    shared = a_sents & b_sents
    if not shared:
        return ok(name, "the two names never share a sentence")
    cues = _alias_cue_re()
    for (s, e) in shared:
        seg = ctx.transcript[s:e]
        if not any(rx.search(seg) for rx in cues):
            return fail(name, f"both names appear in one sentence with no alias "
                              f"cue: {seg.strip()[:70]!r}")
    return ok(name, "co-occur only in sentences that carry an alias cue")
