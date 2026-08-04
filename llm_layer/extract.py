"""
LLM use #3 (broadened): windowed "read-along" extraction.

One bounded pass over the transcript that lets the LLM actually READ the text and
extract, per window, several things at once -- instead of only judging narrow,
pre-filtered candidates. In each window every detected person is tagged by a
stable id (`[P3 Ronnie]`; `P0` is the interviewee), and the model returns ONE
JSON with:
  - attributes  : gender + role per person
  - relations   : family/social relations between people (incl. the interviewee)
  - aliases     : id pairs that are the SAME person written differently

One LLM call per window -> linear in transcript length, and it folds the old
per-person attribute pass into the same call (fewer calls, not more).

Reconciliation stays conservative and ON TOP of the rules (the rules run first
and stay authoritative). Per the chosen policy:
  - attributes -> agree/keep, unset->suggest (`suggested_*`), conflict->flag.
  - relations  -> APPLIED as edges, but ONLY when the evidence quote is verifiably
    present in the transcript (anti-hallucination) and the rules didn't already
    have that edge. Additive / non-destructive.
  - aliases    -> FLAGGED for review (`suggested_merge_with` + flag), NEVER auto-
    merged. Identity changes stay a human decision.
Nothing here touches `replace` (no under-redaction).

This module imports nothing from `graph`: it mutates the Entity objects it is
handed (attributes / flags) and RETURNS relation tuples for the caller to turn
into edges, so the one-way dependency (graph -> llm_layer) is preserved.
"""

from __future__ import annotations
import re
from collections import Counter, defaultdict

_WINDOW_CHARS = 4000
_ROLE_JUNK = {"unknown", "none", "n/a", ""}
# local kin set (kept here so we don't import from `graph`) -- only used to mark
# a relation target as FAMILY; relation words outside it (friend, boss...) still
# create an edge, just no FAMILY subtype.
_KIN_WORDS = {
    "mother", "mom", "father", "dad", "parent", "son", "daughter", "child",
    "kid", "brother", "sister", "sibling", "aunt", "uncle", "cousin", "niece",
    "nephew", "grandmother", "grandfather", "grandma", "grandpa", "grandson",
    "granddaughter", "wife", "husband", "spouse", "partner", "in-law",
    "mother-in-law", "father-in-law", "brother-in-law", "sister-in-law",
    "stepmother", "stepfather", "stepbrother", "stepsister", "half-brother",
    "half-sister", "godmother", "godfather", "grandparent", "grandchild",
}


def _sentences(text: str):
    stops = [0] + [m.end() for m in re.finditer(r"[.!?]", text)]
    return list(zip(stops, stops[1:] + [len(text)]))


def _pack_windows(sents, budget):
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
    """`transcript[ws:we]` with each rostered person's mentions wrapped as
    `[P<id> text]`. roster: [(pid, entity)] with GLOBAL, stable pids."""
    marks = []
    for pid, e in roster:
        for m in e.mentions:
            if ws <= m.start < we:
                marks.append((m.start, m.end, pid))
    seg = transcript[ws:we]
    for (ms, me, pid) in sorted(marks, key=lambda x: -x[0]):
        rs, re_ = ms - ws, me - ws
        seg = seg[:rs] + f"[P{pid} " + seg[rs:re_] + "]" + seg[re_:]
    return seg.strip()


_SYS = (
    "You read ONE window of an interview transcript. People are tagged like "
    "[P3 Ronnie]; P0 is the interviewee (the speaker -- 'I', 'me', 'my'). Using "
    "ONLY this text, and referring to people ONLY by their given ids, extract:\n"
    '- "attributes": object mapping each tagged person id to '
    '{"gender": "F"|"M"|"", "role": "<short role/relationship word or empty>"}. '
    "Gender only when clear.\n"
    '- "relations": list of family/social relations, each '
    '{"from": id, "rel": "<relationship word, e.g. son, aunt, wife, friend>", '
    '"to": id, "evidence": "<exact quote from THIS text>"}. Anchor on the "from" '
    'person: for "my aunt Maria" -> {"from":"P0","rel":"aunt","to":"<Maria id>"}.\n'
    '- "aliases": list of id pairs that are the SAME person written differently '
    '(a nickname/alias), each {"a": id, "b": id, "evidence": "<exact quote>"}.\n'
    "Be conservative: only include a relation or alias the text CLEARLY states, and "
    "always quote the exact supporting text. Leave lists/fields empty when unsure. "
    'Reply with ONLY a JSON object: {"attributes": {}, "relations": [], "aliases": []}.'
)


def _pid(s):
    m = re.fullmatch(r"[Pp]?(\d+)", str(s).strip())
    return int(m.group(1)) if m else None


def extract_pass(transcript: str, entities: list, interviewee, llm) -> list[dict]:
    """Run the windowed extraction. Mutates entity attributes/flags in place;
    RETURNS a list of relation dicts {source, target, detail, evidence} for the
    caller to turn into edges. No-op (returns []) if the LLM is unavailable."""
    if llm is None or not llm.available():
        return []

    persons = [e for e in entities
               if e.category == "PERSON" and e.mentions and e.subtype != "PUBLIC_FIGURE"]
    if not persons:
        return []

    # stable global ids: P0 = interviewee, P1.. = persons (in order)
    ent_by_pid = {0: interviewee}
    pid_by_eid = {interviewee.entity_id: 0}
    for i, e in enumerate(persons, start=1):
        ent_by_pid[i] = e
        pid_by_eid[e.entity_id] = i

    attr_votes = {e.entity_id: {"g": Counter(), "r": Counter()} for e in persons}
    rel_votes: dict = defaultdict(lambda: {"rel": Counter(), "ev": ""})   # (from_eid,to_eid)
    alias_ev: dict = {}                                                    # frozenset(eids) -> quote

    tnorm = re.sub(r"\s+", " ", transcript.lower())

    def verified(quote):
        # the model quotes from the TAGGED window, so strip `[P# ...]` tags first,
        # then require the (whitespace-normalized) quote to be real transcript text
        if not quote:
            return ""
        q = re.sub(r"\[P\d+\s+([^\]]*)\]", r"\1", str(quote))   # untag
        q = re.sub(r"\s+", " ", q.strip())
        return q if len(q) >= 4 and q.lower() in tnorm else ""

    def names_in(ent, text_lower):
        # whole-word match so a short name can't ground on a longer one
        # ("ruth" must not match inside "ruthie")
        for f in ent.sorted_mentions:
            f = f.lower().strip()
            if f and re.search(r"(?<![a-z0-9])" + re.escape(f) + r"(?![a-z0-9])", text_lower):
                return True
        return False

    for (ws, we) in _pack_windows(_sentences(transcript), _WINDOW_CHARS):
        here = [e for e in persons if any(ws <= m.start < we for m in e.mentions)]
        if not here:
            continue
        roster = [(pid_by_eid[e.entity_id], e) for e in here]
        ctx = _tagged_window(transcript, ws, we, roster)
        lines = "P0 = the interviewee (speaker)\n" + "\n".join(
            f"P{pid} = {e.sorted_mentions[0]}" for pid, e in roster)
        prompt = f"Roster:\n{lines}\n\nText:\n{ctx}\n\nExtract attributes, relations, aliases."
        res = llm.judge(prompt, system=_SYS)
        if not res:
            continue

        for k, v in (res.get("attributes") or {}).items():
            pid = _pid(k)
            e = ent_by_pid.get(pid)
            if e is None or pid == 0 or not isinstance(v, dict):
                continue
            g = (v.get("gender") or "").strip().upper()
            if g in ("F", "M"):
                attr_votes[e.entity_id]["g"][g] += 1
            role = (v.get("role") or "").strip()
            if role.lower() not in _ROLE_JUNK:
                attr_votes[e.entity_id]["r"][role] += 1

        for r in (res.get("relations") or []):
            if not isinstance(r, dict):
                continue
            fe, te = ent_by_pid.get(_pid(r.get("from"))), ent_by_pid.get(_pid(r.get("to")))
            ev = verified(r.get("evidence"))
            rel = (r.get("rel") or "").strip().lower()
            if fe is None or te is None or fe is te or not ev or not rel:
                continue
            # ground it: the target (the named relative) must appear in the quote
            if te is not interviewee and not names_in(te, ev.lower()):
                continue
            slot = rel_votes[(fe.entity_id, te.entity_id)]
            slot["rel"][rel] += 1
            slot["ev"] = slot["ev"] or ev

        for a in (res.get("aliases") or []):
            if not isinstance(a, dict):
                continue
            ae, be = ent_by_pid.get(_pid(a.get("a"))), ent_by_pid.get(_pid(a.get("b")))
            ev = verified(a.get("evidence"))
            if ae is None or be is None or ae is be or 0 in (_pid(a.get("a")), _pid(a.get("b"))) or not ev:
                continue
            # ground it: BOTH names must appear in the quote (kills cross-referenced
            # hallucinations where the ids don't match the evidence)
            evl = ev.lower()
            if not (names_in(ae, evl) and names_in(be, evl)):
                continue
            alias_ev.setdefault(frozenset((ae.entity_id, be.entity_id)), (ae, be, ev))

    # ---- reconcile attributes (agree/keep, unset->suggest, conflict->flag) ----
    for e in persons:
        v = attr_votes[e.entity_id]
        if v["g"]:
            g = v["g"].most_common(1)[0][0]
            rg = e.attributes.get("gender")
            if rg is None:
                e.attributes["suggested_gender"] = g
            elif rg != g:
                e.flag_entity(f"LLM-inferred gender {g} conflicts with rule-derived "
                              f"{rg}; kept the rule value")
            else:
                e.attributes["gender_confirmed"] = True
        if v["r"]:
            e.attributes["suggested_role"] = v["r"].most_common(1)[0][0]

    # ---- aliases -> FLAG only (never auto-merge) ----
    for (ae, be, ev) in alias_ev.values():
        an = ae.sorted_mentions[0] if ae.sorted_mentions else ae.entity_id
        bn = be.sorted_mentions[0] if be.sorted_mentions else be.entity_id
        ae.attributes["suggested_merge_with"] = bn
        be.attributes["suggested_merge_with"] = an
        ae.flag_entity(f"LLM suggests same person as {bn}: \"{ev[:80]}\"; review to merge")
        be.flag_entity(f"LLM suggests same person as {an}: \"{ev[:80]}\"; review to merge")

    # ---- relations -> VERIFY, then return survivors as tuples for the caller ----
    # Every proposal passes a deterministic gate (llm_layer/relation_verify): it is
    # applied only when locally provable, downgraded to a review suggestion when
    # merely plausible, and dropped when refuted or out of scope.
    from .relation_verify import RelationContext, verify_relation
    ctx = RelationContext(transcript, persons, interviewee)
    out = []
    for (src, tgt), slot in rel_votes.items():
        rel = slot["rel"].most_common(1)[0][0]
        v = verify_relation(src, tgt, rel, slot["ev"], ctx)
        if v.action == "apply":
            out.append({"source": v.source, "target": v.target,
                        "detail": v.detail, "evidence": v.evidence})
            tgt_ent = ctx.ent_by_id.get(v.target)
            if tgt_ent is not None and tgt_ent.subtype is None and v.detail in _KIN_WORDS:
                tgt_ent.subtype = "FAMILY"
        elif v.action == "suggest":
            named = ctx.ent_by_id.get(v.target)
            other = ctx.ent_by_id.get(v.source)
            if named is interviewee:                 # keep the flag on the named person
                named, other = other, named
            if named is not None and named is not interviewee:
                with_nm = ("the interviewee" if other is interviewee
                           else (other.sorted_mentions[0] if other and other.sorted_mentions
                                 else "someone"))
                named.attributes.setdefault(
                    "suggested_relation",
                    {"detail": v.detail, "with": with_nm, "evidence": v.evidence[:120]})
                named.flag_entity(f"LLM suggests relation '{v.detail}' with {with_nm} "
                                  f"but it couldn't be verified locally; review")
        # v.action == "reject" -> dropped
    return out
