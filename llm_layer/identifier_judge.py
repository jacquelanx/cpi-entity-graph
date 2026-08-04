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

_ID_CATS = ("PHONE", "EMAIL", "SSN_OR_ID", "USERNAME_HANDLE", "OCCUPATION")

_SYS = (
    "You classify identifier spans in one window of an interview transcript. Each "
    "is tagged like [ID2 555-123-4567] or [ID3 nurse]. Using ONLY the text, for "
    'each id give: "owner" ("interviewee" if it belongs to the speaker/"I"/"my", '
    '"other" if it belongs to someone else, else "unknown") and "identifying" '
    "(true ONLY for an OCCUPATION that is specific or rare enough to help identify "
    "a particular person; false for common jobs and for phones/emails/IDs/handles). "
    'Reply with ONLY JSON: {"ID2": {"owner": "...", "identifying": false}, ...}.'
)


def _tagged(transcript, ws, we, roster):
    marks = [(m.start, m.end, gid) for gid, e in roster for m in e.mentions
             if ws <= m.start < we]
    seg = transcript[ws:we]
    for (ms, me, gid) in sorted(marks, key=lambda x: -x[0]):
        rs, re_ = ms - ws, me - ws
        seg = seg[:rs] + f"[ID{gid} " + seg[rs:re_] + "]" + seg[re_:]
    return seg.strip()


def identifier_judge_pass(transcript: str, entities: list, llm) -> None:
    if llm is None or not llm.available():
        return
    ids = [e for e in entities if e.category in _ID_CATS and e.mentions]
    if not ids:
        return

    gid_by_eid = {e.entity_id: i + 1 for i, e in enumerate(ids)}
    ent_by_gid = {i + 1: e for i, e in enumerate(ids)}
    votes = defaultdict(lambda: {"owner": Counter(), "identifying": Counter()})

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
            if v.get("identifying") is True:
                votes[e.entity_id]["identifying"]["true"] += 1

    for e in ids:
        vo = votes[e.entity_id]
        if vo["owner"]:
            # rules may already have set owner deterministically (see
            # graph/pipeline._link_interviewee_pii); keep the rule value if so.
            e.attributes.setdefault("owner", vo["owner"].most_common(1)[0][0])
        if e.category == "OCCUPATION" and vo["identifying"].get("true"):
            e.attributes["identifying"] = True
            e.flag_entity("LLM: this occupation may be specific enough to identify "
                          "someone; review")
