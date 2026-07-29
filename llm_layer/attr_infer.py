"""
LLM use #3: the attribute inferrer (Class B: flagged suggestions). Infers 
attributes the rules miss (gender + a short role/relationship) for each
person, via the LLM reading local context.

IMPORTANT: We never send the whole transcript in one call. Instead we walk
the transcript in fixed-size WINDOWS; for each window, run ONE roster call 
over just the people present in it, with every mention tagged by a local id 
so the model can tell them apart(holistic *within* the window), accumulate 
per-entity votes and reconcile once at the end.
An entity is dropped from later windows once it has enough evidence (early-exit), 
so most people are resolved in their first window.

Reconciliation with the rules:
  - rule set the value and the LLM AGREES  -> keep it (trusted).
  - rule left it unset and the LLM infers one -> store as a SUGGESTION
    (`suggested_gender` / `suggested_role`), NOT the trusted attribute.
  - rule set it and the LLM CONFLICTS -> keep the rule's value, FLAG the conflict.
"""

from __future__ import annotations
import re
from collections import Counter

_WINDOW_CHARS = 4000        # per-call context budget (~1k tokens), length-independent
_MAX_VISITS = 2             # query any one person in at most this many windows
_ROLE_JUNK = {"unknown", "none", "n/a", ""}


def _sentences(text: str):
    stops = [0] + [m.end() for m in re.finditer(r"[.!?]", text)]
    return list(zip(stops, stops[1:] + [len(text)]))


def _pack_windows(sents, budget):
    """Greedily pack consecutive sentences into (start, end) char windows."""
    windows, cs, ce = [], None, None
    for (s, e) in sents:
        if cs is None:
            cs, ce = s, e
        elif e - cs <= budget:
            ce = e
        else:
            windows.append((cs, ce))
            cs, ce = s, e
    if cs is not None:
        windows.append((cs, ce))
    return windows


def _tagged_window(transcript, ws, we, roster):
    """`transcript[ws:we]` with each rostered mention wrapped as `[P<id> text]`."""
    marks = []
    for lid, e in roster:
        for m in e.mentions:
            if ws <= m.start < we:
                marks.append((m.start, m.end, lid))
    if not marks:
        return ""
    seg = transcript[ws:we]
    for (ms, me, lid) in sorted(marks, key=lambda x: -x[0]):   # right-to-left
        rs, re_ = ms - ws, me - ws
        seg = seg[:rs] + f"[P{lid} " + seg[rs:re_] + "]" + seg[re_:]
    return seg.strip()


_ROSTER_SYS = (
    "You extract attributes for a FIXED roster of people in an interview excerpt. "
    "Each person has an id (P1, P2, ...). In the text, every mention is tagged like "
    "[P2 Ronnie]. Using ONLY the text, for each roster id give the person's gender "
    "and a short role/relationship to the interviewee. Give gender only when the "
    'text makes it clear; otherwise leave it "". Do not invent people or ids. Reply '
    "with ONLY a JSON object mapping every roster id to "
    '{"gender": "F"|"M"|"", "role": "<short noun or empty>"}.'
)


def infer_attributes_pass(transcript: str, entities: list, llm) -> None:
    if llm is None or not llm.available():
        return

    persons = [e for e in entities
               if e.category == "PERSON" and e.mentions and e.subtype != "PUBLIC_FIGURE"]
    if not persons:
        return

    votes = {e.entity_id: {"g": Counter(), "r": Counter()} for e in persons}
    visits: Counter = Counter()   # how many windows we've queried each person in
    resolved: set = set()

    for (ws, we) in _pack_windows(_sentences(transcript), _WINDOW_CHARS):
        local = [e for e in persons
                 if e.entity_id not in resolved
                 and any(ws <= m.start < we for m in e.mentions)]
        if not local:
            continue
        roster = [(i + 1, e) for i, e in enumerate(local)]     # local ids per window
        ctx = _tagged_window(transcript, ws, we, roster)
        if not ctx:
            continue
        lines = "\n".join(f"P{lid} = {e.sorted_mentions[0]}" for lid, e in roster)
        prompt = (f"Roster:\n{lines}\n\nText (mentions tagged with their id):\n{ctx}"
                  "\n\nReturn attributes for EVERY roster id.")
        result = llm.judge(prompt, system=_ROSTER_SYS)
        if not result:
            continue
        for lid, e in roster:
            visits[e.entity_id] += 1
            entry = result.get(f"P{lid}")
            if isinstance(entry, dict):
                g = (entry.get("gender") or "").strip().upper()
                if g in ("F", "M"):
                    votes[e.entity_id]["g"][g] += 1
                role = (entry.get("role") or "").strip()
                if role.lower() not in _ROLE_JUNK:
                    votes[e.entity_id]["r"][role] += 1
            v = votes[e.entity_id]
            # drop from later windows once we have both attributes, OR after a
            # capped number of tries -- so a person who never fully resolves isn't
            # re-queried in every window they appear in
            if (v["g"] and v["r"]) or visits[e.entity_id] >= _MAX_VISITS:
                resolved.add(e.entity_id)

    # reconcile the accumulated votes against the rules, once
    for e in persons:
        v = votes[e.entity_id]
        if v["g"]:
            g = v["g"].most_common(1)[0][0]
            rule_g = e.attributes.get("gender")
            if rule_g is None:
                e.attributes["suggested_gender"] = g
            elif rule_g != g:
                e.flag_entity(f"LLM-inferred gender {g} conflicts with rule-derived "
                              f"{rule_g}; kept the rule value")
            else:
                e.attributes["gender_confirmed"] = True   # rule + LLM agree
        if v["r"]:
            e.attributes["suggested_role"] = v["r"].most_common(1)[0][0]
