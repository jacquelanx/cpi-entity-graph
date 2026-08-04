"""
LLM use #6: windowed ownership resolution for AGE and DATE_OF_BIRTH.

The rules attach an age / date-of-birth to the INTERVIEWEE only, and only on a
clear first-person cue (see graph/pipeline._link_interviewee_pii). That leaves a
RELATIVE'S age or birth date unowned -- "my brother turned forty in 2019",
"my mom was born in 1962" -- because the rule can't safely say which person a
non-first-person span belongs to.

This pass closes that gap. It reads windows of the transcript with BOTH the people
(tagged [P3 Ronnie]; P0 is the interviewee) AND the age / birth-date spans (tagged
[A2 nineteen] for an age, [B1 March 1990] for a birth date) marked, and asks the
model WHOSE age / birth date each one is -- returning the SPECIFIC person id. The
winning owner (majority vote across windows) becomes an ATTRIBUTE_OF edge from the
age/dob entity to that person.

Policy (identical to the rest of the layer):
  - Additive and conservative. The rule's owner stays authoritative: we only fill an
    UNSET owner; if the LLM disagrees with a rule-assigned owner we FLAG it and keep
    the rule value.
  - Ownership never touches `replace` -- age/dob stay redacted exactly as before.
    This only enriches the graph (whose age/dob is it), never lowers redaction.
  - Imports nothing from `graph`: it mutates the Entity objects it is handed and
    RETURNS ATTRIBUTE_OF edge tuples for the caller to build, preserving the one-way
    graph -> llm_layer dependency (same contract as extract_pass).
"""

from __future__ import annotations
import re
from collections import Counter, defaultdict

from .extract import _sentences, _pack_windows, _WINDOW_CHARS

_OWN_CATS = ("AGE", "DATE_OF_BIRTH")

_SYS = (
    "You read ONE window of an interview transcript. People are tagged like "
    "[P3 Ronnie]; P0 is the interviewee (the speaker -- 'I', 'me', 'my'). Age and "
    "birth-date spans are tagged like [A2 nineteen] (an age) or [B1 March 1990] "
    "(a birth date). Using ONLY this text, for EACH A#/B# tag decide WHOSE age or "
    "birth date it is and give that person's id (P0 for the interviewee, or a P#). "
    'If the text does not make the owner clear, use "unknown". Be conservative: '
    "only name an owner the text clearly supports. Reply with ONLY a JSON object "
    'mapping each tag to {"owner": "P0"|"P#"|"unknown"}, e.g. '
    '{"A2": {"owner": "P0"}, "B1": {"owner": "P3"}}.'
)


def _owner_pid(s: str):
    """Parse an owner answer ('P3', 'p0', '3') to an int id; None for unknown/junk."""
    m = re.fullmatch(r"[Pp]?(\d+)", str(s).strip())
    return int(m.group(1)) if m else None


def _tag(transcript, ws, we, person_roster, pii_roster):
    """`transcript[ws:we]` with person mentions wrapped [P<id> ...] and age/dob
    spans wrapped [<tag> ...]. Person and PII spans never overlap (different
    categories), so simple right-to-left insertion is safe."""
    marks = []
    for pid, e in person_roster:
        for m in e.mentions:
            if ws <= m.start < we:
                marks.append((m.start, m.end, f"P{pid}"))
    for tag, e in pii_roster:
        for m in e.mentions:
            if ws <= m.start < we:
                marks.append((m.start, m.end, tag))
    seg = transcript[ws:we]
    for (ms, me, tag) in sorted(marks, key=lambda x: -x[0]):
        rs, re_ = ms - ws, me - ws
        seg = seg[:rs] + f"[{tag} " + seg[rs:re_] + "]" + seg[re_:]
    return seg.strip()


def pii_owner_pass(transcript: str, entities: list, interviewee, llm) -> list[tuple]:
    """Resolve ownership of AGE / DATE_OF_BIRTH spans. Mutates owner attributes /
    flags in place and RETURNS ATTRIBUTE_OF edge tuples (source, target, detail,
    evidence) for the caller. No-op (returns []) if the LLM is unavailable."""
    if llm is None or not llm.available():
        return []

    persons = [e for e in entities if e.category == "PERSON" and e.mentions]
    pii = [e for e in entities if e.category in _OWN_CATS and e.mentions]
    if not pii:
        return []

    # stable global person ids: P0 = interviewee, P1.. = named persons (in order)
    ent_by_pid = {0: interviewee}
    pid_by_eid = {interviewee.entity_id: 0}
    for i, e in enumerate(persons, start=1):
        ent_by_pid[i] = e
        pid_by_eid[e.entity_id] = i

    # stable tags: AGE -> A1.., DATE_OF_BIRTH -> B1..
    tag_by_eid, ent_by_tag = {}, {}
    ai = bi = 0
    for e in pii:
        if e.category == "AGE":
            ai += 1
            tag = f"A{ai}"
        else:
            bi += 1
            tag = f"B{bi}"
        tag_by_eid[e.entity_id] = tag
        ent_by_tag[tag] = e

    votes: dict = defaultdict(Counter)      # pii entity_id -> Counter(owner pid)

    for (ws, we) in _pack_windows(_sentences(transcript), _WINDOW_CHARS):
        here_pii = [e for e in pii if any(ws <= m.start < we for m in e.mentions)]
        if not here_pii:
            continue
        here_persons = [e for e in persons if any(ws <= m.start < we for m in e.mentions)]
        person_roster = [(pid_by_eid[e.entity_id], e) for e in here_persons]
        pii_roster = [(tag_by_eid[e.entity_id], e) for e in here_pii]

        people_lines = "P0 = the interviewee (speaker)\n" + "\n".join(
            f"P{pid} = {e.sorted_mentions[0]}" for pid, e in person_roster)
        pii_lines = "\n".join(
            f"{tag} = {e.mentions[0].text}  ({e.category})" for tag, e in pii_roster)
        ctx = _tag(transcript, ws, we, person_roster, pii_roster)
        prompt = (f"People:\n{people_lines}\n\nAge/birth-date spans:\n{pii_lines}\n\n"
                  f"Text:\n{ctx}\n\nWho owns each age/birth-date span?")
        res = llm.judge(prompt, system=_SYS)
        if not res:
            continue
        for k, v in (res.get("owners", res) or {}).items():
            e = ent_by_tag.get(str(k).strip())
            if e is None or not isinstance(v, dict):
                continue
            pid = _owner_pid(v.get("owner", ""))
            if pid is None or pid not in ent_by_pid:     # unknown / not a real id
                continue
            votes[e.entity_id][pid] += 1

    edges = []
    for e in pii:
        c = votes.get(e.entity_id)
        if not c:
            continue
        pid = c.most_common(1)[0][0]
        owner_ent = ent_by_pid.get(pid)
        if owner_ent is None:
            continue
        owner_label = "interviewee" if pid == 0 else "other"
        rule_owner = e.attributes.get("owner")
        if rule_owner is None:
            # rule left ownership unset -> the LLM fills it (this is the relative's
            # age/dob case the rule can't reach). Record owner + a specific edge.
            e.attributes["owner"] = owner_label
            if pid != 0:
                e.attributes["owner_person"] = owner_ent.entity_id
            edges.append((e.entity_id, owner_ent.entity_id,
                          e.subtype or e.category,
                          f"(llm) owner of {e.category.lower()}"))
        elif rule_owner != owner_label:
            # rule already assigned an owner (high-precision first-person cue) --
            # keep it, but surface the disagreement for review.
            nm = (owner_ent.sorted_mentions[0] if pid != 0 and owner_ent.sorted_mentions
                  else "the interviewee")
            e.flag_entity(f"LLM thinks this {e.category.lower()} belongs to {nm}, but "
                          f"the rule assigned it to '{rule_owner}'; rule kept -- review")
    return edges
