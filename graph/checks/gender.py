"""
Deterministic checkers for GENDER (named persons and the interviewee).

The rule layer reads gender from kinship words (`kinship.KINSHIP_GENDER`). These
checkers add the OTHER deterministic signals the rules never used, and apply them
as refutations of an LLM guess: an honorific ('Mr.', 'Ms.', 'Father'), a kinship
word adjacent to the mention, and -- for the interviewee -- a spouse term.

Policy: a checker only ever REFUTES. Silence is not support, but it is not a
refutation either, so an unrefuted guess passes. That keeps recall while making
the one thing we can prove -- a contradiction -- fatal.
"""

from __future__ import annotations
import re
from . import CheckOutcome, ok, fail, na
from ..kinship import KINSHIP_GENDER

# Honorifics that carry gender. `merge_strings.HONORIFIC_TITLES` exists but was
# never mapped to gender; this is that mapping.
HONORIFIC_GENDER = {
    "mr": "M", "mister": "M", "sir": "M", "father": "M", "fr": "M",
    "brother": "M", "monsignor": "M", "rabbi": None, "imam": None,
    "mrs": "F", "ms": "F", "miss": "F", "madam": "F", "maam": "F",
    "ma'am": "F", "mother": "F", "sister": "F", "nun": "F",
    # deliberately genderless
    "dr": None, "mx": None, "prof": None, "professor": None, "rev": None,
    "reverend": None, "pastor": None, "captain": None, "capt": None,
    "sgt": None, "sergeant": None, "officer": None, "judge": None,
    "senator": None, "governor": None, "gov": None, "mayor": None,
    "president": None, "pres": None, "coach": None, "principal": None,
}

_WORD_BEFORE = re.compile(r"([A-Za-z][\w'’-]*)[\s,]+$")

# First-person spouse reference. This is NOT evidence of the speaker's own
# gender (that would encode an assumption about their orientation), but a
# same-gender spouse term makes a *specific* LLM guess unprovable, so we treat a
# conflicting pair as unverifiable rather than as support. See policy note in
# `attributes.infer_interviewee_gender`.
_SPOUSE_F = re.compile(r"\bmy\s+(?:wife|late\s+wife|ex-wife)\b", re.I)
_SPOUSE_M = re.compile(r"\bmy\s+(?:husband|late\s+husband|ex-husband)\b", re.I)

# Unambiguous first-person self-description of the SPEAKER'S own gender, reusing
# the rule layer's own patterns AND the same subject-turn masking, so the checker
# and the rule cannot disagree about either the pattern or who was speaking.
def _self_described(transcript: str):
    from ..attributes import _IV_SELF_F, _IV_SELF_M, _IV_CALLME_F, _IV_CALLME_M
    found = set()
    for rx, g in ((_IV_SELF_F, "F"), (_IV_CALLME_F, "F"),
                  (_IV_SELF_M, "M"), (_IV_CALLME_M, "M")):
        if rx.search(transcript):
            found.add(g)
    return found


def honorific_gender(text: str):
    """Gender implied by a leading honorific in a name form, else None."""
    for raw in re.split(r"[\s,]+", text or ""):
        t = raw.strip("'\".()-").lower()
        if t in HONORIFIC_GENDER:
            return HONORIFIC_GENDER[t]
    return None


def not_refuted_by_honorific(value, ctx) -> CheckOutcome:
    name = "honorific_agreement"
    seen = False
    for form in getattr(ctx.entity, "sorted_mentions", []):
        h = honorific_gender(form)
        if not h:
            continue
        seen = True
        if h != value:
            return fail(name, f"'{form}' implies {h}, model said {value}")
    if not seen:
        return na(name, "no gendered honorific in any name form")
    return ok(name)


def not_refuted_by_kin_word(value, ctx) -> CheckOutcome:
    """A kinship word immediately before a mention ('my father James') fixes the
    gender deterministically; an LLM guess that contradicts it is wrong."""
    name = "kin_word_agreement"
    seen = False
    for m in getattr(ctx.entity, "mentions", []):
        w = _WORD_BEFORE.search(ctx.transcript[max(0, m.start - 24):m.start])
        if not w:
            continue
        g = KINSHIP_GENDER.get(w.group(1).lower().replace(".", ""))
        if not g:
            continue
        seen = True
        if g != value:
            return fail(name, f"'{w.group(1)}' implies {g}, model said {value}")
    if not seen:
        return na(name, "no gendered kin word beside any mention")
    return ok(name)


# Third-person pronouns and gendered person-nouns. These are the signals the rule
# layer never used and the reason `gender` fills were landing with ZERO applicable
# checks: `not_refuted_by_honorific` and `not_refuted_by_kin_word` only apply to a
# titled or kin-prefixed name, which most named people are not. Verified: on the
# sample transcripts Opal, Dr. Combs and Bill Ratliff all got an LLM gender with
# `checks_passed=[]` -- nothing had looked at the value at all.
_PRON_GENDER = {"he": "M", "him": "M", "his": "M", "himself": "M",
                "she": "F", "her": "F", "hers": "F", "herself": "F"}
_PRON_RE = re.compile(r"\b(he|him|his|himself|she|her|hers|herself)\b", re.I)
_NOUN_GENDER = {"man": "M", "boy": "M", "guy": "M", "gentleman": "M", "fella": "M",
                "fellow": "M", "widower": "M",
                "woman": "F", "girl": "F", "lady": "F", "gal": "F", "widow": "F"}
_NOUN_RE = re.compile(r"\b(" + "|".join(_NOUN_GENDER) + r")\b", re.I)


def _next_sentence(ctx, pos: int):
    """The sentence AFTER the one containing `pos`, clipped to the same turn."""
    from ..turns import turn_bounds
    _ts, te = turn_bounds(pos, ctx.turns)
    _s, e = ctx.sentence_bounds(pos)
    for a, b in ctx.sents:
        if a >= e:
            return (a, min(b, te)) if a < te else None
    return None


def _other_person_starts(ctx, entity_id):
    return [s for (s, _e, eid) in ctx.named_person_spans() if eid != entity_id]


def not_refuted_by_pronoun(value, ctx) -> CheckOutcome:
    """A third-person pronoun or gendered noun bound to this person's mention.

    Scanned in two places, both bounded so another person cannot steal the signal:

      * AFTER the mention, to the end of its sentence and through the next
        sentence in the same turn ("Dr. Combs, who's treated three generations ...,
        he'll tell you"; "was a man named Bill Ratliff. ... He'd been down there"),
        stopping at the next named person's mention.
      * a gendered person-noun immediately BEFORE the mention, within one short
        clause ("a man named Bill Ratliff", "a woman we never saw again").

    Conflicting signals make the value unverifiable rather than refuted, so the
    outcome is `na` -- consistent with the rest of this module, where only a clean
    contradiction is fatal.
    """
    name = "pronoun_agreement"
    if value not in ("F", "M"):
        return fail(name, f"{value!r} is not a gender this pipeline emits")
    eid = getattr(ctx.entity, "entity_id", None)
    others = _other_person_starts(ctx, eid)
    found = set()
    for m in getattr(ctx.entity, "mentions", []):
        # (1) forward window: rest of this sentence + the next, cut at the next person
        _ss, se = ctx.sentence_bounds(m.start)
        windows = [(m.end, se)]
        nxt = _next_sentence(ctx, m.start)
        if nxt is not None:
            windows.append(nxt)
        for (a, b) in windows:
            if b <= a:
                continue
            cut = min([s for s in others if a <= s < b], default=b)
            for p in _PRON_RE.finditer(ctx.transcript[a:cut]):
                found.add(_PRON_GENDER[p.group(1).lower()])
        # (2) a gendered noun just before the mention ("a man named Bill Ratliff")
        pre_start = max(_ss, m.start - 30)
        if not any(pre_start <= s < m.start for s in others):
            for n in _NOUN_RE.finditer(ctx.transcript[pre_start:m.start]):
                found.add(_NOUN_GENDER[n.group(1).lower()])

    if not found:
        return na(name, "no pronoun or gendered noun bound to this person")
    if len(found) > 1:
        return na(name, "both masculine and feminine cues nearby; not decidable")
    got = found.pop()
    if got != value:
        return fail(name, f"nearby pronoun/noun implies {got}, model said {value}")
    return ok(name, f"a nearby pronoun/noun implies {got}")


def interviewee_self_description_agrees(value, ctx) -> CheckOutcome:
    """If the speaker described their own gender in the first person, the LLM must
    agree with it. Conflicting self-descriptions (the case the rule abstains on)
    make ANY specific guess unverifiable."""
    name = "interviewee_self_description"
    found = _self_described(ctx.subject_transcript)
    if len(found) == 1 and value not in found:
        return fail(name, f"first-person self-description implies {found.pop()}")
    if len(found) > 1:
        return fail(name, "conflicting first-person gender cues; not verifiable")
    if not found:
        return na(name, "the speaker never described their own gender")
    return ok(name)


def interviewee_honorific_address_agrees(value, ctx) -> CheckOutcome:
    """The INTERVIEWER's honorific for the subject ("Thank you, Ms. Boudreaux").

    The only checker for this field that can POSITIVELY support a value, which is
    what makes it matter. The other two are refute-only by design -- self-description
    is rare and a spouse term is deliberately not read as evidence (that would encode
    an assumption about the speaker's orientation) -- so on a transcript with neither,
    every candidate gender scored zero applicable checks. `interviewee_gender` was
    therefore unverifiable in principle: a model's guess could never be confirmed,
    and `second_line._try_alternatives` could never single out a survivor.

    An honorific used to ADDRESS the subject is different in kind: it is the
    interviewer's own gendered form of address, deterministic, and it carries no
    inference about anybody's relationships. It only exists once identification has
    given the speaker a name, which is precisely the real-transcript case -- both
    sample transcripts leave the speaker unnamed, so this stays `na` there.
    """
    name = "interviewee_honorific_address"
    from ..interviewee import ADDRESS_TITLED, _is_address
    from ..turns import parse_turns, in_interviewer_turn
    from ..merge_strings import normalize

    iv = ctx.interviewee
    toks = set()
    for form in getattr(iv, "sorted_mentions", []):
        toks |= set(normalize(form))
    if not toks:
        return na(name, "the transcript never names the speaker, so no honorific "
                        "address exists")

    turns = parse_turns(ctx.transcript)
    found = set()
    for m in ADDRESS_TITLED.finditer(ctx.transcript):
        if not in_interviewer_turn(m.start(), turns):
            continue
        if not (set(normalize(m.group(1))) & toks):
            continue
        if not _is_address(ctx.transcript, m.start(1), m.end(1), m.group(1),
                           titled=True, phrase_start=m.start()):
            continue
        g = honorific_gender(m.group(0))
        if g:
            found.add(g)

    if not found:
        return na(name, "no gendered honorific is used to address the speaker")
    if len(found) > 1:
        return na(name, "the interviewer uses both masculine and feminine "
                        "honorifics; not decidable")
    got = found.pop()
    if got != value:
        return fail(name, f"the interviewer addresses the speaker with a {got} "
                          f"honorific, model said {value}")
    return ok(name, f"the interviewer addresses the speaker with a {got} honorific")


def interviewee_spouse_term_agrees(value, ctx) -> CheckOutcome:
    """A first-person spouse term is weak evidence, so it cannot CONFIRM -- but a
    guess that a spouse term contradicts under the common (heterosexual) reading
    is not locally provable either. We require the model's guess not to be the
    only reading a spouse term makes doubtful.

    Concretely: 'my wife' present and the model says the speaker is F -> not
    provable from this transcript, so the fill is rejected and left for a human.
    This is the check that catches the verified `interview_002` failure, where the
    model returned F for a speaker who says 'My wife wanted to'.
    """
    name = "interviewee_spouse_term"
    # subject turns only: the INTERVIEWER's "my wife" says nothing about the
    # person being interviewed.
    f_spouse = bool(_SPOUSE_F.search(ctx.subject_transcript))
    m_spouse = bool(_SPOUSE_M.search(ctx.subject_transcript))
    if f_spouse and m_spouse:
        return na(name, "both spouse terms present -> uninformative")
    if not (f_spouse or m_spouse):
        return na(name, "no first-person spouse term in the subject's speech")
    if f_spouse and value == "F":
        return fail(name, "'my wife' present; speaker gender F not locally provable")
    if m_spouse and value == "M":
        return fail(name, "'my husband' present; speaker gender M not locally provable")
    # NOT `ok`. A spouse term that fails to contradict the guess is not evidence FOR
    # it -- reading "my husband" as proof the speaker is female is an assumption about
    # their orientation, which the docstring above is explicit about refusing. But
    # returning `ok` put this checker in `checks_passed`, and `checks_passed` is the
    # definition of `Resolution.verified`: interview_001's interviewee gender was
    # reported as "1 deterministic check verified it" when the sole check was that
    # heteronormative inference. `na` keeps the refutation (which IS sound -- a
    # same-gender spouse term makes a specific guess unprovable) while refusing to
    # count silence as support, so an unverified gender now looks unverified.
    return na(name, "a spouse term is present and does not contradict this value, "
                    "which is not evidence for it")
