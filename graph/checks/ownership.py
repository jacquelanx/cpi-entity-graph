"""
Deterministic checkers for OWNER -- whose identifier / age / DOB is this?

This is the highest-stakes field for interviewee-only de-identification: a false
'interviewee' attribution mints a surrogate from someone else's data, and a false
'other' leaves the speaker's own identifier out of their surrogate identity
entirely.

BOTH DIRECTIONS ARE CHECKED. They used to not be. `_every_mention` opened with

    if value != "interviewee": return ok(name, "not an interviewee claim")

so every `owner="other"` fill cleared all four checkers without a single line of
evidence being read, and then reported itself as "filled from the LLM after 4
deterministic check(s) passed". On the sample transcripts that vacuously accepted
10 of 18 owner fills, including the interviewee's own phone and email in
interview_001 ("we're easy to find. The shop line is 228-555-0143.") -- which
silently blocked the ATTRIBUTE_OF edge to e000 that surrogate generation needs.

So there are two checker families, and each returns `na` outside its direction:

  interviewee : first_person_cue_present, no_kin_noun_between,
                no_nearer_named_person, no_third_person_subject
  other       : third_party_identifiable, not_bound_by_first_person

'other' now requires POSITIVE evidence that somebody else is the referent. With
none, the field stays empty -- which is the honest outcome, and (for the
categories that feed the speaker's identity) a blocking one, so a human decides
rather than the pipeline quietly asserting "not the speaker".

Every predicate is lifted to require UNANIMITY across the entity's mentions, and
every cue search runs over `ctx.subject_transcript`, so the interviewer's
first-person speech can never be read as the subject's.
"""

from __future__ import annotations
import re
from . import CheckOutcome, ok, fail, na
from ..rules.kinship import KINSHIP_GENDER

_FP_CUE = re.compile(
    r"\b(?:my|our)\b|\bI\b|\bme\b|\bwe\b|"
    r"\b(?:reach|call|email|text|message)\s+me\b", re.I)

_KIN_NOUN = re.compile(
    r"\b(?:" + "|".join(re.escape(w) for w in
                        sorted(KINSHIP_GENDER, key=len, reverse=True)) + r")\b",
    re.I)

# A third-person SUBJECT governing the clause ('She was a schoolteacher',
# 'She started a page') means the span belongs to that person, not the speaker.
#
# Deliberately restricted to subject pronouns and possessive+noun. Object pronouns
# ('him', 'them') cannot be the governing subject, and matching them produced a
# false refutation on the dialectal 'one of them email addresses' -- which is the
# speaker's own email. This is the check that catches the two verified
# mis-attributions ('She was a schoolteacher', 'She started a page').
_THIRD_SUBJ = re.compile(r"\b(?:he|she|they)\b|\b(?:his|her|their)\s+\w+", re.I)
_FP_POSS = re.compile(r"\b(?:my|our)\b", re.I)

# A possessive PERSON governing the span names its owner outright: "Mamaw's
# ninety-one", "Trang's phone". First person is excluded -- "my" is a cue, not a
# third party. This is the signal that covers a kin term `KINSHIP_GENDER` does not
# list ("Mamaw" is absent from it), so `owner="other"` need not abstain merely
# because a dialect word is missing from a table.
_POSS_NOUN = re.compile(r"\b(?!my\b|our\b|I\b|we\b)([A-Za-z][\w'’\-]*)['’]s\b", re.I)

# ...but the possessor has to BE a person. "Email's nguyenfamilyshrimp@example.net"
# and "Phone's 304-555-0176" are possessives whose head is the identifier itself,
# and reading them as a person named "Email" handed the speaker's own contact
# details to a third party. Personhood is established from the transcript rather
# than assumed: a kin word, a common person noun, or a token the text itself
# introduces with a possessive ("my Mamaw Opal" proves "Mamaw" is a person here).
_PERSON_NOUNS = {
    "guy", "man", "woman", "boy", "girl", "lady", "kid", "person", "dude", "fella",
    "fellow", "gentleman", "baby", "child", "friend", "buddy", "neighbor", "boss",
    "foreman", "preacher", "pastor", "doctor", "teacher", "nurse", "granny",
}


def _is_person_possessor(tok: str, ctx) -> bool:
    t = tok.lower()
    if t in KINSHIP_GENDER or t in _PERSON_NOUNS:
        return True
    for (s, e, _eid) in ctx.named_person_spans():
        if ctx.transcript[s:e].lower().split()[:1] == [t]:
            return True
    return bool(re.search(r"\b(?:my|our|his|her|their)\s+" + re.escape(tok) + r"\b",
                          ctx.transcript, re.I))


def _claim(name: str, direction: str, per_mention):
    """Lift a per-mention predicate to the whole entity for ONE claim direction.

    Requires UNANIMITY. Entities are grouped by surface form, so one AGE entity
    can cover two spans in different ownership contexts -- 'the water came up
    twelve feet' and 'my daughter Trang was maybe twelve' are the same entity.
    Inspecting only `mentions[0]` attributed that age to the speaker. If any
    mention refutes the claim, ownership is ambiguous and we abstain.

    A span the SUBJECT DID NOT UTTER is SKIPPED, not refused -- provided at least one
    span of this entity is the subject's. The interviewer echoing a value ("You said
    your father started at fourteen") neither supports nor refutes ownership: it is not
    evidence, so it must not behave like counter-evidence.

    It used to refuse in both directions, and because entities are grouped by surface
    TEXT, one interviewer echo poisoned the whole entity. That is what made `fourteen`
    in interview_002 unattributable and BLOCKING even though the speaker's own span --
    "My father, Earl, went in the mines at fourteen" -- states the owner plainly. With
    every span the interviewer's, there is genuinely nothing to read and the checker
    still fails.
    """
    def checker(value, ctx) -> CheckOutcome:
        if str(value) != direction:
            return na(name, f"not an {direction!r} claim")
        mentions = getattr(ctx.entity, "mentions", [])
        if not mentions:
            return fail(name, "no mention to anchor on")
        spoken = [m for m in mentions if ctx.in_subject_turn(m.start)]
        if not spoken:
            roles = sorted({ctx.role_at(m.start) for m in mentions})
            return fail(name, f"every span of this entity was spoken by the "
                              f"{'/'.join(roles)}, not the interviewee")
        for m in spoken:
            out = per_mention(m, ctx, name)
            if not out.passed:
                extra = "" if len(spoken) == 1 else f" (at {m.start})"
                return fail(name, out.detail + extra)
        return ok(name)
    return checker


# How far back a first-person cue may sit and still bind a span. Bounded three
# ways -- by the TURN (never read another speaker's "my"), by a sentence count, and
# by a character budget -- so the reach is wide enough for real speech and still
# local.
#
# This used to be exactly ONE sentence of lookback, and the arbitrary line it drew
# cost the interviewee their own data in both sample transcripts:
#
#   "Oh, it's a business, we're easy to find. The shop line is 228-555-0143.
#    Email's nguyenfamilyshrimp@example.net."
#
# The phone is one sentence from the cue and was attributed; the EMAIL is two, and
# came out unattributable and BLOCKING -- same speaker, same breath, same
# construction. Likewise "I did. Course I did. What else was there. Went in at
# eighteen, came out at fifty-five", where the speaker's own life-course ages sat
# three short sentences after the cue and both blocked.
#
# Widening is safe because the GUARDS, not the distance, are what carry the
# precision: a third-person subject, a kin noun or a named person appearing between
# the cue and the span still refutes, and the further back the cue, the likelier one
# of those intervenes.
_LOOKBACK_SENTENCES = 4
_LOOKBACK_CHARS = 600


def _nearest_cue_end(ctx, m):
    """Absolute end offset of the NEAREST first-person cue at or before the span,
    or None. Searched over `ctx.subject_transcript`, so a cue the interviewer
    uttered is invisible, and clamped to the span's own turn.

    The clamp is a clamp, not a rejection: sentence spans tile the transcript, so
    the sentence preceding a span often STARTS inside the previous speaker's turn
    (it opens with "\\nSPEAKER: "). Testing that start for subject-hood would throw
    away a legitimate same-turn lookback.
    """
    from ..text.turns import turn_bounds
    ts, _te = turn_bounds(m.start, ctx.turns)
    floor = max(ts, m.start - _LOOKBACK_CHARS)
    ss, _se = ctx.sentence_bounds(m.start)

    # nearest first: the span's own sentence up to the span, then back one
    # sentence at a time while still inside the turn and the char budget
    windows = [(max(ss, floor), m.start)]
    prev = []
    for s, e in ctx.sents:
        if e > ss:
            break
        if e > floor:
            prev.append((max(s, floor), e))
    windows += list(reversed(prev))[:_LOOKBACK_SENTENCES]

    for (a, b) in windows:
        if b <= a:
            continue
        cues = list(_FP_CUE.finditer(ctx.subject_transcript[a:b]))
        if cues:
            return a + cues[-1].end()
    return None


def _interposed(ctx, cue_abs: int, m):
    """What stands between the cue and the span and takes the span away from the
    speaker, as a reason string -- else "". The precision half of the lookback."""
    between = ctx.transcript[cue_abs:m.start]
    if _THIRD_SUBJ.search(between):
        return "a third-person subject intervenes after the first-person cue"
    if _KIN_NOUN.search(between):
        return "a kin noun intervenes after the first-person cue"
    if any(cue_abs <= s < m.start for (s, _e, _eid) in ctx.named_person_spans()):
        return "a named person intervenes after the first-person cue"
    return ""


# `_cues_before` lived here. Every caller now goes through `_nearest_cue_end`, which
# is the point: the four predicates used to disagree about WHERE the binding cue was
# (two searched the span's sentence, one searched the sentence before it as well),
# and that disagreement is what let a widened lookback gain reach without gaining
# scrutiny. One cue position, four predicates.


# ----------------------------------------------------- 'interviewee' direction

def _fp_cue(m, ctx, name) -> CheckOutcome:
    """A first-person cue binding the span, from the span's own sentence or from a
    bounded lookback within the same turn (see `_nearest_cue_end`).

    A cue in the span's own sentence is accepted as before. A cue further back only
    counts when nothing could have taken over as the referent in between -- no
    third-person subject, no kin noun, no named person (see `_interposed`). The
    remaining sentence-local ambiguity ("my daughter Trang was maybe twelve", which
    has a cue in its own sentence) is what `no_kin_noun_between` and
    `no_nearer_named_person` exist to catch.
    """
    cue_abs = _nearest_cue_end(ctx, m)
    if cue_abs is None:
        return fail(name, f"no first-person cue in the span's sentence or the "
                          f"{_LOOKBACK_SENTENCES} preceding sentences of this turn")
    ss, _se = ctx.sentence_bounds(m.start)
    if cue_abs >= ss:
        return ok(name, "first-person cue in the span's own sentence")
    reason = _interposed(ctx, cue_abs, m)
    if reason:
        return fail(name, reason)
    return ok(name, "first-person cue earlier in the same turn, nothing intervening")


def _no_kin_noun(m, ctx, name) -> CheckOutcome:
    """A kin noun between the cue and the span means it belongs to THAT relative
    ("my daughter runs a page, @handle").

    Reads the NEAREST cue wherever it is, rather than only a cue inside the span's
    own sentence. With the sentence-local version this checker silently returned
    `ok` for every cross-sentence cue -- so the widened lookback in `_fp_cue` would
    have gained reach with no matching gain in scrutiny.
    """
    cue_abs = _nearest_cue_end(ctx, m)
    if cue_abs is None:
        return ok(name)                       # the cue checker owns that failure
    gap = ctx.transcript[cue_abs:m.start]
    if _KIN_NOUN.search(gap):
        return fail(name, f"kin noun between cue and span: {gap.strip()[:40]!r}")
    return ok(name)


def _no_nearer_named_person(m, ctx, name) -> CheckOutcome:
    """The nearest thing before the span decides who it describes.

    If a NAMED person sits between the closest first-person cue and the span -- or
    before the span with no first-person cue in front of it at all -- the span
    describes that person, not the speaker. This is what separates

        "By twenty-three I was working the docks."     -> the speaker
        "The year Trang turned sixteen ..."            -> Trang
        "my daughter Trang was maybe twelve"           -> Trang

    all three of which contain a first-person cue somewhere in the sentence, so a
    whole-sentence cue search alone cannot tell them apart.
    """
    ss, _se = ctx.sentence_bounds(m.start)
    cue_end = _nearest_cue_end(ctx, m)
    # search from whichever comes first: the span's sentence, or the cue that binds
    # it. A cue reached by lookback must not skip over a named person sitting
    # between it and the span.
    search_from = min(ss, cue_end) if cue_end is not None else ss
    nearest = None
    for (s, _e, _eid) in ctx.named_person_spans():
        if search_from <= s < m.start and (nearest is None or s > nearest):
            nearest = s
    if nearest is None:
        return ok(name)
    if cue_end is None:
        return fail(name, "a named person precedes the span and no first-person "
                          "cue comes before it")
    if nearest >= cue_end:
        return fail(name, "a named person sits between the first-person cue "
                          "and the span")
    return ok(name)


def _no_third_person(m, ctx, name) -> CheckOutcome:
    """Reject an 'interviewee' claim when a third-person subject governs the span
    and no first-person possessive binds it more closely.

    Catches the two verified mis-attributions: `schoolteacher` in interview_002
    ("She was a schoolteacher" -- the mother) and `@lebateaushrimp` in
    interview_001 ("She started a page for the business" -- the daughter).
    """
    ss, _se = ctx.sentence_bounds(m.start)
    before = ctx.transcript[ss:m.start]
    third = None
    for c in _THIRD_SUBJ.finditer(before):
        third = c                             # nearest to the span wins
    if third is None:
        return ok(name)
    fp = None
    for c in _FP_POSS.finditer(ctx.subject_transcript[ss:m.start]):
        fp = c
    if fp is not None and fp.start() > third.start():
        return ok(name, "first-person possessive binds the span")
    return fail(name, f"third-person subject {third.group(0)!r} governs the span")


# ------------------------------------------------------------ 'other' direction

def _third_party(m, ctx, name) -> CheckOutcome:
    """SOMEBODY ELSE must be identifiable as the referent.

    Any one of four deterministic signals will do, in decreasing locality:

      * a named person mentioned in the span's sentence ("The foreman when I
        started was a man named Bill Ratliff." -> `foreman` is his);
      * a kin noun between the nearest first-person cue and the span ("My father
        ... as a deckhand" -> the father's);
      * a possessive noun governing the span ("Mamaw's ninety-one");
      * a third-person subject before the span ("She was a schoolteacher").

    With none of the four, 'other' is an unsupported assertion. We refuse it and
    let the field stay empty, because "unknown owner" is recoverable and "someone
    else's" is not.
    """
    ss, se = ctx.sentence_bounds(m.start)
    for (s, _e, eid) in ctx.named_person_spans():
        if ss <= s < se and not (m.start <= s < m.end):
            return ok(name, "a named person is mentioned in the same sentence")
    # same cue position the 'interviewee' family uses, so the two directions cannot
    # disagree about WHERE the binding cue is
    cue_end = _nearest_cue_end(ctx, m)
    cue_end = ss if cue_end is None else cue_end
    if _KIN_NOUN.search(ctx.transcript[cue_end:m.start]):
        return ok(name, "a kin noun stands between the first-person cue and the span")
    # a PERSON possessor other than the speaker themselves
    own_toks = {t.lower() for f in getattr(ctx.interviewee, "sorted_mentions", [])
                for t in re.split(r"[\s,]+", f) if t}
    for p in _POSS_NOUN.finditer(ctx.transcript[ss:m.start]):
        tok = p.group(1)
        if tok.lower() not in own_toks and _is_person_possessor(tok, ctx):
            return ok(name, f"possessive {p.group(0)!r} names another owner")
    if _THIRD_SUBJ.search(ctx.transcript[ss:m.start]):
        return ok(name, "a third-person subject governs the span")
    return fail(name, "no named person, kin noun, possessive or third-person subject "
                      "identifies another owner for this span")


def _not_fp_bound(m, ctx, name) -> CheckOutcome:
    """Refute 'other' when a first-person possessive binds the span with nothing
    in between -- the mirror of `no_third_person_subject`. "my claim number's
    5-338-2201" cannot belong to somebody else."""
    ss, _se = ctx.sentence_bounds(m.start)
    fp = None
    for c in _FP_POSS.finditer(ctx.subject_transcript[ss:m.start]):
        fp = c
    if fp is None:
        return ok(name, "no first-person possessive binds the span")
    gap = ctx.transcript[ss + fp.end():m.start]
    if _KIN_NOUN.search(gap):
        return ok(name, "a kin noun follows the first-person possessive")
    third = None
    for c in _THIRD_SUBJ.finditer(ctx.transcript[ss:m.start]):
        third = c
    if third is not None and third.start() > fp.start():
        return ok(name, "a third-person subject binds the span more closely")
    if any(ss + fp.end() <= s < m.start for (s, _e, _eid) in ctx.named_person_spans()):
        return ok(name, "a named person follows the first-person possessive")
    return fail(name, "a first-person possessive binds the span and nothing else "
                      "intervenes; this looks like the interviewee's own")


# Public checkers, each lifted to require UNANIMITY across every mention and to
# return `na` outside its own claim direction.
first_person_cue_present = _claim("first_person_cue", "interviewee", _fp_cue)
no_kin_noun_between = _claim("no_kin_noun_between", "interviewee", _no_kin_noun)
no_nearer_named_person = _claim("no_nearer_named_person", "interviewee",
                                _no_nearer_named_person)
no_third_person_subject = _claim("no_third_person_subject", "interviewee",
                                 _no_third_person)

third_party_identifiable = _claim("third_party_identifiable", "other", _third_party)
not_bound_by_first_person = _claim("not_bound_by_first_person", "other", _not_fp_bound)
