"""
LLM use #3 (broadened): windowed "read-along" extraction.

One bounded pass over the transcript that lets the LLM actually READ the text and
extract, per window, several things at once -- instead of only judging narrow,
pre-filtered candidates. In each window every detected person is tagged by a
stable id (`[P3 Ronnie]`; `P0` is the interviewee), and the model returns ONE
JSON with:
  - attributes  : gender + role + ethnicity per person
  - relations   : family/social relations between people (incl. the interviewee)
  - aliases     : id pairs that are the SAME person written differently

One LLM call per window -> linear in transcript length, and it folds the old
per-person attribute pass into the same call (fewer calls, not more).

Reconciliation stays conservative and ON TOP of the rules (the rules run first
and stay authoritative). Per the chosen policy:
  - attributes -> agree/keep, unset->suggest (`suggested_*`), conflict->flag.
    Ethnicity has no rule source, so it is ALWAYS a suggestion: a quote-grounded
    `stated` label, or an `inferred` guess tagged low-confidence (`ethnicity_basis`).
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


# Abbreviation-aware sentence segmentation. This is a deliberate mirror of
# graph/sentences.py:sentence_spans -- llm_layer imports nothing from `graph`
# (one-way graph -> llm_layer dependency), so the small pure function is copied
# here rather than imported. Keep the two in sync. Used by this module and by
# relation_verify (which imports `_sentences` from here).
_ABBREV = {
    "mr", "mrs", "ms", "mx", "dr", "prof", "st", "sr", "jr", "rev", "fr",
    "hon", "gov", "sen", "rep", "pres", "capt", "sgt", "lt", "col", "gen",
    "cmdr", "cpl", "det", "ofc", "supt", "atty", "messrs", "mmes",
    "etc", "vs", "al", "inc", "ltd", "co", "corp", "dept", "est", "fig",
    "no", "nos", "vol", "pp", "approx", "apt", "ave", "blvd", "rd", "ste",
    "cf", "viz", "ibid",
}
_WORD_BEFORE_DOT = re.compile(r"([A-Za-z][A-Za-z'&.]*)$")
_INITIALISM = re.compile(r"^[a-z](?:\.[a-z])+$")
_TERMINATORS = ".?!"
_CLOSERS = "\"'”’)]"


def _ends_sentence(text, seg_start, dot, run):
    if run != ".":
        return set(run) != {"."}                     # ellipsis "..." -> not a boundary
    n = len(text)
    if 0 < dot < n - 1 and text[dot - 1].isdigit() and text[dot + 1].isdigit():
        return False                                 # decimal number
    wb = _WORD_BEFORE_DOT.search(text[seg_start:dot])
    if wb:
        tok = wb.group(1).lower()
        if tok in _ABBREV or len(tok) == 1 or _INITIALISM.match(tok):
            return False                             # abbreviation / initial / acronym
    return True


def _sentences(text: str):
    n = len(text)
    spans, start, i = [], 0, 0
    while i < n:
        if text[i] in _TERMINATORS:
            j = i
            while j < n and text[j] in _TERMINATORS:
                j += 1
            if _ends_sentence(text, start, i, text[i:j]):
                end = j
                while end < n and text[end] in _CLOSERS:
                    end += 1
                spans.append((start, end))
                start = i = end
                continue
            i = j
            continue
        i += 1
    if start < n:
        spans.append((start, n))
    return spans


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
    '{"gender": "F"|"M"|"", "role": "<short role/relationship word or empty>", '
    '"ethnicity": "<short free-text ethnicity/heritage label, e.g. Cuban, African '
    'American, Vietnamese, Creole, or empty>", '
    '"ethnicity_basis": "stated"|"inferred"|"", '
    '"ethnicity_evidence": "<exact quote from THIS text if stated, else empty>"}. '
    'Gender only when clear. For ethnicity use "stated" ONLY when the person '
    "explicitly self-identifies or the text plainly states it, and put the exact "
    'words in ethnicity_evidence; use "inferred" if you are merely guessing from a '
    "name or surrounding context; leave ethnicity empty when you cannot tell.\n"
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


def _eth_evidenced(quote: str, label: str) -> bool:
    """A 'stated' ethnicity must be BACKED by the quote -- the quote must actually
    mention the ethnicity/heritage, not merely be real transcript text. (Before this
    check the model could attach any verbatim-but-irrelevant quote and get 'stated'.)
    True when a >=4-char token of the label appears in the quote, or a heritage cue
    is present; otherwise the claim is demoted to 'inferred'."""
    q = quote.lower()
    if any(tok in q for tok in re.findall(r"[a-z]{4,}", label.lower())):
        return True
    return bool(re.search(r"\b(?:descent|heritage|ancestr|immigrant|immigrated|refugee)\b", q))


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

    attr_votes = {e.entity_id: {"g": Counter(), "r": Counter(),
                                "eth_stated": Counter(), "eth_inferred": Counter(),
                                "eth_ev": {}} for e in persons}
    # the interviewee (P0) gets an ethnicity vote bucket too -- first-person
    # self-identification is the strongest ethnicity signal and was previously dropped.
    attr_votes[interviewee.entity_id] = {"g": Counter(), "r": Counter(),
                                         "eth_stated": Counter(), "eth_inferred": Counter(),
                                         "eth_ev": {}}
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
            if e is None or not isinstance(v, dict):
                continue
            av = attr_votes.get(e.entity_id)
            if av is None:
                continue
            if pid != 0:                               # gender/role: named persons only
                g = (v.get("gender") or "").strip().upper()
                if g in ("F", "M"):
                    av["g"][g] += 1
                role = (v.get("role") or "").strip()
                if role.lower() not in _ROLE_JUNK:
                    av["r"][role] += 1
            # ethnicity: recorded for EVERYONE incl. the interviewee (P0).
            eth = (v.get("ethnicity") or "").strip()
            if eth and eth.lower() not in _ROLE_JUNK:
                basis = (v.get("ethnicity_basis") or "").strip().lower()
                ev = verified(v.get("ethnicity_evidence")) if basis == "stated" else ""
                if basis == "stated" and ev and _eth_evidenced(ev, eth):
                    av["eth_stated"][eth] += 1
                    av["eth_ev"].setdefault(eth, ev)
                else:
                    # "inferred", or a "stated" claim whose quote isn't in the text or
                    # doesn't actually mention the ethnicity -> treat as a guess.
                    av["eth_inferred"][eth] += 1

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

    # ---- ethnicity (named persons AND the interviewee) -> always a SUGGESTION ----
    # Prefer a STATED, quote-grounded label; otherwise fall back to an INFERRED guess
    # tagged low-confidence. Name-based inference is unreliable and ethnicity is
    # sensitive, so a consumer should treat 'inferred' as a soft hint only and
    # 'stated' (with its evidence quote) as the trustworthy one.
    for e in persons + [interviewee]:
        v = attr_votes[e.entity_id]
        if v["eth_stated"]:
            label = v["eth_stated"].most_common(1)[0][0]
            e.attributes["suggested_ethnicity"] = label
            e.attributes["ethnicity_basis"] = "stated"
            e.attributes["ethnicity_evidence"] = v["eth_ev"].get(label, "")
        elif v["eth_inferred"]:
            label = v["eth_inferred"].most_common(1)[0][0]
            e.attributes["suggested_ethnicity"] = label
            e.attributes["ethnicity_basis"] = "inferred"
            e.attributes["ethnicity_confidence"] = "low"

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
