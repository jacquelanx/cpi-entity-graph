"""
LLM use #4: windowed judgment over direct identifiers.

The deterministic layer (`graph/identifiers.py`) already types/normalizes PHONE /
EMAIL / SSN_OR_ID / USERNAME_HANDLE / OCCUPATION. This pass adds the *contextual*
judgment a regex can't make, reading windows of the transcript with the
identifier spans tagged (`[ID2 555-123-4567]`):

  - **owner**  — whose identifier is it (interviewee / other)? Useful so the
    surrogate generator can keep one person's fakes consistent.
  - **identifying** — is an OCCUPATION specific/rare enough to help identify
    someone? (common jobs are not.)

Both are written as conservative **suggestions/flags** (`owner`, `identifying`
+ a review flag). It never changes `replace` — direct identifiers stay
`replace=True`; this only *raises* attention, never lowers redaction.

Windowed and bounded like the other passes; one call per window that actually
contains an identifier (identifiers are sparse, so this is a handful of calls).
Imports nothing from `graph` (mutates the Entity objects it's handed).
"""

from __future__ import annotations
import re
from collections import Counter, defaultdict

from .extract import _sentences, _pack_windows, _WINDOW_CHARS

# AGE and DATE_OF_BIRTH are included: their OWNER is a Variant A field with a rule
# source (`pipeline._link_interviewee_pii`), so it needs an LLM proposer to be
# second-lined at all. Previously they were absent, leaving age/DOB ownership --
# a primary quasi-identifier for the speaker -- with no second line whatsoever.
_ID_CATS = ("PHONE", "EMAIL", "SSN_OR_ID", "USERNAME_HANDLE", "OCCUPATION",
            "AGE", "DATE_OF_BIRTH")

_SYS = (
    "You classify identifier spans in one window of an interview transcript. Each "
    "is tagged like [ID2 555-123-4567], [ID3 nurse] or [ID4 sixty-eight]. Using "
    'ONLY the text, for each id give: "owner" ("interviewee" if it belongs to or '
    'describes the speaker/"I"/"my", "other" if it belongs to someone else, else '
    '"unknown" -- an AGE or DATE OF BIRTH belongs to whoever it describes); '
    '"identifying" (true ONLY for an OCCUPATION that is specific or rare enough to '
    "help identify a particular person; false for common jobs and for "
    'phones/emails/IDs/handles); and "kind" (what the span actually is: one of '
    '"phone", "email", "ssn", "id", "handle", "occupation", "age", "dob", "other"). '
    'Reply with ONLY JSON: {"ID2": {"owner": "...", "identifying": false, '
    '"kind": "phone"}, ...}.'
)

# map the LLM "kind" word onto the pipeline's entity categories, so we can tell
# whether the model disagrees with the rule's typing of a malformed span.
_KIND_TO_CAT = {"phone": "PHONE", "email": "EMAIL", "ssn": "SSN_OR_ID",
                "id": "SSN_OR_ID", "handle": "USERNAME_HANDLE",
                "occupation": "OCCUPATION", "age": "AGE", "dob": "DATE_OF_BIRTH"}


def _tagged(transcript, ws, we, roster):
    marks = [(m.start, m.end, gid) for gid, e in roster for m in e.mentions
             if ws <= m.start < we]
    seg = transcript[ws:we]
    for (ms, me, gid) in sorted(marks, key=lambda x: -x[0]):
        rs, re_ = ms - ws, me - ws
        seg = seg[:rs] + f"[ID{gid} " + seg[rs:re_] + "]" + seg[re_:]
    return seg.strip()


def identifier_judge_pass(transcript: str, entities: list, llm) -> dict:
    """Returns `{entity_id: {field: {"value","confidence"}}}` for `owner`, `kind`
    and `identifying`, which `graph.second_line` arbitrates.

    `identifying` used to be written in place here as an advisory flag, on the
    grounds that it had no rule source. It does now --
    `identifiers.COMMON_OCCUPATIONS` fills `identifying=False` for a common job --
    so it is proposed like everything else and gated by
    `checks/identifiers.identifying_not_a_common_occupation`. That checker is what
    stops the model flagging "miners" in a coal-mining interview: on the sample
    transcripts it returned True for seven of nine occupations, and nothing could
    refute it.
    """
    proposals: dict = {}
    if llm is None or not llm.available():
        return proposals
    ids = [e for e in entities if e.category in _ID_CATS and e.mentions]
    if not ids:
        return proposals

    gid_by_eid = {e.entity_id: i + 1 for i, e in enumerate(ids)}
    ent_by_gid = {i + 1: e for i, e in enumerate(ids)}
    votes = defaultdict(lambda: {"owner": Counter(), "identifying": Counter(),
                                 "kind": Counter()})

    for (ws, we) in _pack_windows(_sentences(transcript), _WINDOW_CHARS):
        here = [e for e in ids if any(ws <= m.start < we for m in e.mentions)]
        if not here:
            continue
        roster = [(gid_by_eid[e.entity_id], e) for e in here]
        lines = "\n".join(f"ID{g} = {e.mentions[0].text}  ({e.category})" for g, e in roster)
        res = llm.judge(f"Identifiers:\n{lines}\n\nText:\n{_tagged(transcript, ws, we, roster)}"
                        "\n\nClassify each id.", system=_SYS)
        if not res:
            continue
        for k, v in (res.get("identifiers", res) or {}).items():
            m = re.search(r"(\d+)", str(k))
            e = ent_by_gid.get(int(m.group(1))) if m else None
            if e is None or not isinstance(v, dict):
                continue
            owner = str(v.get("owner", "")).lower()
            if owner in ("interviewee", "other"):
                votes[e.entity_id]["owner"][owner] += 1
            if isinstance(v.get("identifying"), bool):
                votes[e.entity_id]["identifying"][bool(v["identifying"])] += 1
            kind = str(v.get("kind", "")).lower()
            if kind in _KIND_TO_CAT:
                votes[e.entity_id]["kind"][kind] += 1

    def propose(e, field, counter):
        if not counter:
            return
        value, n = counter.most_common(1)[0]
        proposals.setdefault(e.entity_id, {})[field] = {
            "value": value, "confidence": "high" if n > 1 else "low"}

    for e in ids:
        vo = votes[e.entity_id]
        # `owner` is now ALWAYS proposed, including when the rule already set it.
        # The old `setdefault` discarded the model's answer whenever the rule had
        # spoken, which made a WRONG rule attribution unfalsifiable -- and ownership
        # of the interviewee's own identifiers is the highest-stakes field here.
        propose(e, "owner", vo["owner"])
        # `kind` is likewise proposed unconditionally, not only when the rule
        # flagged the span as malformed. graph/checks/identifiers.py re-runs the
        # rule's own regexes as the verification.
        propose(e, "kind", vo["kind"])
        # `identifying` is a proposal now, not an in-place write -- it has a rule
        # source (COMMON_OCCUPATIONS) and two checkers.
        if e.category == "OCCUPATION":
            propose(e, "identifying", vo["identifying"])

    return proposals
