"""
WHICH named person is the interviewee?

Nothing in the pipeline used to ask. `e000` is a synthetic node with no detected
span, and no stage ever linked a named PERSON entity to it -- an assumption that
holds in both sample transcripts only because their speaker is never named. Real
transcripts name the speaker constantly:

    INTERVIEWER: Thank you, Ms. Boudreaux.
    SPEAKER: My name's Rosa Boudreaux, and I was born up the holler.
    ROSA: ...

Left unhandled, that produces TWO nodes for one human -- `e000` carrying the
gender / DOB / ethnicity / owned identifiers, and a separate PERSON carrying the
name -- and surrogate generation mints two different fake identities for the one
person we are trying to de-identify.

This module is the missing stage, built to the same four-step shape as every
other field:

  1. RULE      `rule_candidate` -- three closed, high-precision constructions
                 (speaker label, first-person self-introduction in a subject
                 turn, interviewer address in an interviewer turn).
  2. LLM       `llm_layer.interviewee_id.propose_interviewee` reads the roster and
                 names one id, or none.
  3. CHECKERS  `graph/checks/interviewee.py` gates whatever survives: the name
                 must actually appear as a self-reference in a SPEAKER turn or as
                 an interviewer address, must not be introduced as the speaker's
                 relative, and must not be a public figure.
  4. MERGE     only a value that cleared step 3 is folded into `e000`.

`support_for` is deliberately shared between the rule and the checker -- the same
arrangement `checks/gender.py` uses -- so the layer that proposes and the layer
that verifies cannot disagree about what counts as evidence.

Because merging changes the entity set, this runs EARLY (right after coref, before
kinship) so every later stage sees one interviewee. Its `Resolution` is threaded
into `resolve_all` afterwards, so it still gets a provenance record and a ledger
row like every other field.
"""

from __future__ import annotations
import re

from .merge_strings import normalize
from .turns import (parse_turns, mask_to_subject, in_subject_turn,
                    in_interviewer_turn, subject_labels, INTERVIEWER)

# ---------------------------------------------------------------- constructions

_NAME_TOKEN = r"[A-Z](?:[a-z]+|(?=['’\-]))(?:['’\-][A-Za-z]+)*"
_NAME = rf"{_NAME_TOKEN}(?:\s+{_NAME_TOKEN}){{0,2}}"
# Honorifics an interviewer can use to ADDRESS the subject. Widened past the
# original civil set: oral-history subjects are addressed as "Reverend", "Pastor",
# "Sister", "Sergeant" constantly, and a title the list did not know made the name
# invisible to BOTH layers -- `checks/interviewee` gates on this same function, so a
# gap here is a gap in the checker too (see `support_for`).
_TITLE = (r"(?:Mr|Mrs|Ms|Miss|Mx|Dr|Prof|Professor|Rev|Reverend|Pastor|Father|"
          r"Sister|Brother|Elder|Deacon|Sgt|Sergeant|Capt|Captain|Judge|Sheriff|"
          r"Coach|Chief|Officer|Senator|Governor|Mayor)\.?")

# (1) The speaker names themselves. Matched against SUBJECT turns only.
#
# `call(?:s|ed)?` and the perfect forms matter: "Boudreaux is what everybody's
# CALLED me" is the construction `graph/interviewee.py`'s own docstring offers as
# the reason the LLM layer exists, and bare `call\s+me` did not match it. Since
# `checks/interviewee.named_in_self_reference_or_address` gates the LLM's answer
# with THIS function, a construction missing here could not be recovered by the
# model either -- the second layer was capped at the first layer's vocabulary.
SELF_INTRO = re.compile(
    r"\b(?:my\s+(?:full\s+|real\s+|given\s+)?name(?:['’]s|\s+is|\s+was)|name['’]s|"
    r"I\s*['’]m|I\s+am|"
    r"(?:they|folks|everybody|everyone|people|you|you\s+can)\s+"
    r"(?:all\s+|just\s+|always\s+)?(?:call|calls|called)\s+me|"
    r"(?:['’]ve|have|has|['’]s|had)\s+(?:always\s+)?called\s+me|"
    r"go(?:es)?\s+by)\s+"
    rf"(?:{_TITLE}\s+)?({_NAME})")

# (1b) The same self-introduction with the NAME FIRST. English puts the name in
# front just as often ("Boudreaux is what everybody's called me since the mine",
# "Rosa, that's me"), and a forward-only pattern could not see any of it.
SELF_INTRO_REVERSED = re.compile(
    rf"\b({_NAME})\b\s*(?:,\s*)?"
    r"(?:is|was|['’]s)\s+(?:what|the\s+name)\s+"
    r"(?:everybody|everyone|folks|people|they|the\s+\w+)\s*"
    r"(?:['’]s|['’]ve|has|have|had)?\s*(?:always\s+)?call(?:s|ed)?\s+me\b"
    rf"|\b({_NAME})\b\s*,\s*that['’]?s\s+(?:me\b|my\s+name\b)")

# (2) The interviewer addresses the subject BY NAME. Matched against INTERVIEWER
# turns only, and -- titled or bare -- the occurrence must actually be an ADDRESS:
# set off as a vocative, or sitting in a sentence that speaks to the listener.
#
# The titled route used to skip both tests, accepting any "Mr./Dr. <Name>" an
# interviewer uttered. That was already loose for the civil titles and becomes
# unsafe with the widened `_TITLE` set above (an interviewer mentioning "Father
# Nguyen" is discussing a priest, not addressing the subject), so the two routes
# now share one predicate instead of applying different standards.
ADDRESS_TITLED = re.compile(rf"\b{_TITLE}\s+({_NAME})")

# (2b) A bare name in an interviewer turn counts only when the same sentence
# addresses the listener -- "Thank you, Rosa" / "And you, Rosa?" -- AND the name is
# used vocatively, i.e. set off by a comma or standing at the start of the
# sentence. Without the vocative test, "Is your husband Earl joining us?" reads as
# an address to Earl: the sentence does contain "your", but the name is the third
# party being asked about, not the person being spoken to.
_SECOND_PERSON = re.compile(r"\b(?:you|your|yours|yourself)\b", re.I)
# A vocative is SET OFF BY A COMMA -- "Thank you, Rosa." / "Rosa, tell me about it."
# Trailing sentence punctuation is NOT enough: "You mentioned two people named Hai?"
# ends with the name and a question mark, and reading that as an address to Hai is
# what made the rule nominate the speaker's uncle in interview_001.
_VOCATIVE_BEFORE = re.compile(r"(?:^|[,;:]|\b(?:so|and|now|well|okay|ok)\s*,?)\s*$", re.I)
_VOCATIVE_AFTER = re.compile(r"^\s*,")
# The name is being REFERRED to, not addressed.
_REFERRING = re.compile(r"\b(?:named|called|about|mentioned|know|knew|remember)\s+$", re.I)

_SENT_SPLIT = re.compile(r"[.?!]+")

# A name introduced as somebody's relative is not the speaker. ANY possessive
# counts, not just first person: "my mother, Gloria" (the speaker's), "your husband
# Earl" (the interviewer asking about the subject's), "his brother John". A speaker
# cannot be introduced as their own relative, so all of them refute. Uses the
# kinship vocabulary rather than a fresh list so the two cannot drift apart.
def _kin_before_re():
    from .kinship import KIN, _MOD
    return re.compile(
        rf"\b(?:my|our|your|his|her|their)\s+{_MOD}{KIN}[\s,]*$", re.I)


_KIN_BEFORE = None


def kin_introduction_before(transcript: str, pos: int) -> str:
    """The "my <kin>" construction immediately preceding `pos`, else ""."""
    global _KIN_BEFORE
    if _KIN_BEFORE is None:
        _KIN_BEFORE = _kin_before_re()
    m = _KIN_BEFORE.search(transcript[max(0, pos - 60):pos])
    return m.group(0).strip() if m else ""


# ------------------------------------------------------------------- evidence

def support_for(entity, transcript: str) -> tuple[str, str]:
    """Is `entity` evidenced as the SPEAKER? Returns (kind, evidence quote).

    kind is one of "label", "self_intro", "address", or "" for no support. Shared
    by the rule proposer and the deterministic checker.
    """
    turns = parse_turns(transcript)
    spoken = mask_to_subject(transcript)
    forms = [f for f in entity.sorted_mentions if f.strip()]
    toks = set()
    for f in forms:
        toks |= set(normalize(f))
    if not toks:
        return "", ""

    # (1) the speaker's turn label IS this name ("ROSA:" / "Ms. Boudreaux:")
    for label in subject_labels(transcript):
        if toks & set(normalize(label)):
            return "label", f"speaker label {label!r}"

    # (2) first-person self-introduction inside a SUBJECT turn, name after the cue
    #     ("my name's Rosa") or before it ("Boudreaux is what they call me")
    for rx in (SELF_INTRO, SELF_INTRO_REVERSED):
        for m in rx.finditer(spoken):
            if not in_subject_turn(m.start(), turns):
                continue
            captured = next((g for g in m.groups() if g), None)
            if captured and set(normalize(captured)) & toks:
                return "self_intro", m.group(0).strip()

    # (3) the interviewer addressing the subject -- titled or bare, both held to
    #     the same standard by `_is_address`
    for m in ADDRESS_TITLED.finditer(transcript):
        if not in_interviewer_turn(m.start(), turns):
            continue
        if not (set(normalize(m.group(1))) & toks):
            continue
        ev = _is_address(transcript, m.start(1), m.end(1), m.group(1), titled=True)
        if ev:
            return "address", ev
    for mention in entity.mentions:
        if not in_interviewer_turn(mention.start, turns):
            continue
        ev = _is_address(transcript, mention.start, mention.end, mention.text,
                         titled=False)
        if ev:
            return "address", ev
    return "", ""


def _is_address(transcript: str, start: int, end: int, text: str,
                titled: bool) -> str:
    """The evidence quote if this name occurrence is the interviewer ADDRESSING the
    listener, else "". Shared by the titled and bare routes so the two cannot drift.

    Two refutations apply to both routes: the name must not be introduced as
    somebody's relative ("your husband Earl") and must not be being REFERRED to
    ("two people named Hai", "tell me about Bill Ratliff").

    The positive test is deliberately STRICTER for a bare name than for a titled
    one, because merging the wrong person into e000 is the worst error this pipeline
    can make (see `checks/interviewee.py`):

      * titled  -- vocative OR a second-person sentence. An honorific is itself an
        address marker, which is what makes the widened `_TITLE` set safe here.
      * bare    -- vocative AND a second-person sentence, unchanged. With neither
        signal a bare name in an interviewer turn is just a third party, and with
        only one of them it is a coin flip ("Did you and Loretta stay close?" reads
        as vocative purely because of the coordinating "and").
    """
    if kin_introduction_before(transcript, start):
        return ""                         # "your husband Earl" -- a third party
    before = transcript[max(0, start - 40):start]
    after = transcript[end:end + 4]
    if _REFERRING.search(before):
        return ""                         # "two people named Hai" -- referred to
    vocative = bool(_VOCATIVE_BEFORE.search(before) or _VOCATIVE_AFTER.match(after))
    if not vocative and not titled:
        return ""
    seg_start = max(0, start - 160)
    seg = transcript[seg_start:end + 160]
    for part in _SENT_SPLIT.split(seg):
        if text not in part:
            continue
        if _SECOND_PERSON.search(part) or (titled and vocative):
            return part.strip()[:120]
    return ""


# ------------------------------------------------------------------ rule layer

def rule_candidate(transcript: str, persons: list) -> tuple[str | None, str, list]:
    """The RULE's answer: (entity_id or None, evidence, all_supported_ids).

    Conservative by construction. Several distinct people supported by these cues
    means the transcript is not a two-party interview we can read deterministically,
    so the rule abstains and lets the LLM propose against the checkers.
    """
    supported = []
    for e in persons:
        kind, ev = support_for(e, transcript)
        if kind:
            supported.append((e.entity_id, kind, ev))
    if len(supported) == 1:
        eid, kind, ev = supported[0]
        return eid, f"{kind}: {ev}", [s[0] for s in supported]
    # a label match outranks the weaker cues when several names are supported
    labelled = [s for s in supported if s[1] == "label"]
    if len(labelled) == 1:
        eid, kind, ev = labelled[0]
        return eid, f"{kind}: {ev}", [s[0] for s in supported]
    return None, "", [s[0] for s in supported]


# -------------------------------------------------------------------- the merge

def _name_parts(entity) -> tuple[str | None, str | None]:
    """given_name / surname from the longest name form -- literally the same
    function `attributes.infer_person_attributes` uses, so the speaker's own name
    cannot be split differently from everybody else's. This was a second copy of
    the split, and it carried the same honorific mis-slotting bug."""
    from .merge_strings import split_name_parts
    return split_name_parts(entity.sorted_mentions[0] if entity.sorted_mentions else "")


def merge_into_interviewee(interviewee, named, persons: list) -> None:
    """Fold the named PERSON entity into `e000`.

    Runs before kinship / attributes / identifiers, so there are no edges to
    retarget yet -- the whole point of doing identification early.
    """
    interviewee.mentions.extend(named.mentions)
    interviewee.mentions.sort(key=lambda m: m.start)
    for k, v in named.attributes.items():
        if v is not None and k not in ("role", "replace"):
            interviewee.attributes.setdefault(k, v)
    gn, sn = _name_parts(named)
    if gn:
        interviewee.attributes.setdefault("given_name", gn)
    if sn:
        interviewee.attributes.setdefault("surname", sn)
    interviewee.attributes["replace"] = True          # never negotiable
    interviewee.attributes["identified_from"] = named.entity_id
    if named.needs_review:
        interviewee.flag_entity(named.review_reason)
    if named in persons:
        persons.remove(named)


# ---------------------------------------------------------------- arbitration

def resolve_interviewee_identity(transcript: str, persons: list, interviewee,
                                 llm=None, llm_ran: bool = False):
    """Run the four steps and return the `Resolution`.

    Rules first, LLM second, deterministic checkers third, merge last. Imported
    lazily inside the function to keep `graph.interviewee` free of an import cycle
    with `graph.second_line`.
    """
    from .second_line import POLICIES, second_line, apply_resolution, FILL, CONFIRM, KEEP
    from .checks import CheckContext

    policy = POLICIES["interviewee_identity"]
    ctx = CheckContext(transcript=transcript, entities=list(persons),
                       edges=[], interviewee=interviewee, entity=interviewee)

    rule_value, rule_ev, supported = rule_candidate(transcript, persons)
    if rule_value is None and len(supported) > 1:
        interviewee.flag_entity(
            "several named people are addressed or self-introduce; the rule could "
            "not tell which one is the speaker")

    llm_proposal = None
    if llm is not None and getattr(llm, "available", lambda: False)():
        from llm_layer import propose_interviewee
        v = propose_interviewee(llm, transcript, persons, interviewee)
        if v:
            llm_proposal = v

    res = second_line(policy, rule_value, llm_proposal, ctx, llm_ran=llm_ran)
    apply_resolution(interviewee, res, policy)

    if res.action in (FILL, CONFIRM, KEEP) and res.value:
        named = next((e for e in persons if e.entity_id == res.value), None)
        if named is not None:
            # `apply_resolution` only writes the attribute on fill/confirm, because
            # for every other field a KEEP means the value is already on the entity.
            # This one is computed, not read, so record it here for all three.
            interviewee.attributes["identity_entity_id"] = res.value
            merge_into_interviewee(interviewee, named, persons)
            if rule_ev and res.action != FILL:
                interviewee.attributes.setdefault("identity_evidence", rule_ev)
    return res
