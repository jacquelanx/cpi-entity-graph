"""
Windowed "read-along" PROPOSER.

One bounded pass over the transcript that lets the LLM actually READ the text and
extract, per window, several things at once. In each window every detected person is
tagged by a stable id (`[P3 Ronnie]`; `P0` is the interviewee), and the model returns
ONE JSON with attributes, relations and aliases. One call per window -> linear in
transcript length.

EVERYTHING it produces is RETURNED as data and arbitrated by `graph.second_line`.
This module mutates nothing:

  * proposals -- gender (incl. the speaker's own), given_name, surname, `role`,
    `ethnicity`
  * relations -- RAW, unverified. `graph/checks/relation_evidence.py` is the checker; it used
    to live in this package and verify its own proposals, which made relations the
    one field with no Resolution and no provenance.
  * merges -- RAW same-person (alias/nickname) claims, checked by
    `graph/checks/merges.py`. Still never applied automatically: a claim that clears
    the checkers becomes a `suggested_merge_with` review flag, so identity changes
    stay a human decision. What is new is that the decision is RECORDED.

`role`, `ethnicity` and aliases used to be written in place here, on the grounds
that they were "LLM-only, with no rule to run first and nothing to check against".
That was wrong on both counts, and the cost was visible in the output: every named
person in the sample transcripts inherited the speaker's ethnicity as an unchecked
`inferred` guess from their name alone, and a priest addressed as "Father Nguyen"
got `role: father` with nothing able to refute it. Both fields now have a rule
layer in `graph/rules/attributes.py` and checkers in `graph/checks/`.

Nothing here touches `replace`. This module imports nothing from `graph`, so the
one-way dependency (graph -> llm_layer) is preserved.
"""

from __future__ import annotations
import re
from collections import Counter, defaultdict

_WINDOW_CHARS = 4000
_ROLE_JUNK = {"unknown", "none", "n/a", ""}
# Abbreviation-aware sentence segmentation. This is a deliberate mirror of
# graph/text/sentences.py:sentence_spans -- llm_layer imports nothing from `graph`
# (one-way graph -> llm_layer dependency), so the small pure function is copied
# here rather than imported. Keep the two in sync. Used by this module and by
# identifier_judge (which imports `_sentences` from here).
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
    '"given_name": "<the person\'s FIRST/personal name as written here, or empty>", '
    '"surname": "<the person\'s FAMILY name as written here, or empty>", '
    '"ethnicity": "<short free-text ethnicity/heritage label, e.g. Cuban, African '
    'American, Vietnamese, Creole, or empty>", '
    '"ethnicity_basis": "stated"|"inferred"|"", '
    '"ethnicity_evidence": "<exact quote from THIS text if stated, else empty>"}. '
    "Include P0 (the interviewee) when the speaker's OWN gender/ethnicity is clear "
    'from first-person context (e.g. "I\'m a grandmother", "as a Vietnamese refugee"). '
    "For given_name/surname, split ONLY the name tokens actually written in this "
    "text -- never invent a name, and never put a title (Mr., Dr., Father) or a "
    "kinship word (Mamaw, Papaw, Aunt) in either slot; leave a slot empty if the "
    "text does not supply it. "
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


def extract_pass(transcript: str, entities: list, interviewee, llm,
                 subject_mask: str | None = None):
    """Run the windowed extraction.

    `subject_mask` is a same-length copy of the transcript with the interviewer's
    speech masked out (`graph.text.turns.mask_to_subject`), passed in as a plain string
    so this module still imports nothing from `graph`. It is used to decide whether
    a window contains any of the SPEAKER's own words -- a window that is entirely
    interviewer speech says nothing about P0. When omitted, every window counts as
    the subject's, which is the pre-turn-awareness behaviour.

    Returns `(proposals, relations, merges)`:
      * `proposals` -- `{entity_id: {field: {"value","confidence"}}}` for gender,
        given_name, surname, role and ethnicity, arbitrated by `graph.second_line`.
      * `relations` -- RAW relation proposals {source, target, detail, evidence,
        confidence}. Unverified: `graph.second_line` runs
        `graph/checks/relation_evidence.py` over them as the `relation` field's checker.
      * `merges` -- RAW same-person (alias/nickname) claims {a, b, evidence,
        source, confidence}, checked by `graph/checks/merges.py`. Never applied
        automatically; a checked claim becomes a review flag.

    This module no longer mutates any entity. Every field it reads is returned as
    data and decided in `graph.second_line`, which is what gives `role`,
    `ethnicity` and alias claims a Resolution, provenance and a ledger row for the
    first time.
    """
    if llm is None or not llm.available():
        return {}, [], []

    # `e is not interviewee` matters now that the identification stage can give the
    # interviewee real name mentions: without it the speaker would appear on the
    # roster BOTH as P0 and as a numbered third party, and the model would be asked
    # to relate them to themselves.
    # NO public-figure exclusion. `e.subtype != "PUBLIC_FIGURE"` used to sit here, and
    # it removed the model's second line from `gender`, `given_name`, `surname`, `role`
    # and `ethnicity` for every name the RULE table happened to match -- leaving those
    # five fields rule-only, with `checks_passed=[]` and the flag "neither the rules nor
    # the LLM produced a value" (verified on Obama, Kennedy and Reagan across the two
    # samples).
    #
    # It failed hardest in exactly the case that matters. This pass runs BEFORE
    # `replace` is arbitrated, so the subtype it read was the rule's provisional guess;
    # Kennedy and Reagan were then downgraded to PUBLIC_FIGURE_UNCONFIRMED and treated
    # as private individuals -- with no LLM second line on any of their attributes. A
    # private namesake of a celebrity is the one person here who most needs one.
    persons = [e for e in entities
               if e.category == "PERSON" and e.mentions and e is not interviewee]
    if not persons and not getattr(interviewee, "mentions", None):
        # No named third party AND no name for the speaker -- but P0's own gender and
        # ethnicity are still readable from first-person speech, so the pass must
        # still run. Returning early here is what left `interviewee_gender` with no
        # LLM proposal at all on a transcript that names nobody.
        if not (transcript or "").strip():
            return {}, [], []

    # stable global ids: P0 = interviewee, P1.. = persons (in order)
    ent_by_pid = {0: interviewee}
    pid_by_eid = {interviewee.entity_id: 0}
    for i, e in enumerate(persons, start=1):
        ent_by_pid[i] = e
        pid_by_eid[e.entity_id] = i

    def _bucket():
        return {"g": Counter(), "r": Counter(), "gn": Counter(), "sn": Counter(),
                "eth_stated": Counter(), "eth_inferred": Counter(), "eth_ev": {}}

    attr_votes = {e.entity_id: _bucket() for e in persons}
    # the interviewee (P0) gets buckets too -- first-person self-identification is
    # the strongest signal for its own gender/ethnicity and was previously dropped.
    attr_votes[interviewee.entity_id] = _bucket()
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
        roster = [(pid_by_eid[e.entity_id], e) for e in here]
        # the interviewee joins the roster as P0 when its own name appears here, so
        # the model can split the speaker's name into given/surname parts
        if any(ws <= m.start < we for m in getattr(interviewee, "mentions", [])):
            roster.insert(0, (0, interviewee))
        subject_here = (True if subject_mask is None
                        else bool(subject_mask[ws:we].replace("\x00", "").strip()))
        if not roster and not subject_here:
            continue                     # pure interviewer speech, nobody tagged
        ctx = _tagged_window(transcript, ws, we, roster)
        lines = "P0 = the interviewee (speaker)\n" + "\n".join(
            f"P{pid} = {e.sorted_mentions[0]}" for pid, e in roster if pid != 0)
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
            # gender: recorded for EVERYONE incl. the interviewee (P0) -- the
            # interviewee's own gender was previously dropped, leaving it with no
            # rule OR LLM source. Role stays named-persons-only (P0's role is just
            # "interviewee").
            g = (v.get("gender") or "").strip().upper()
            if g in ("F", "M"):
                av["g"][g] += 1
            if pid != 0:
                role = (v.get("role") or "").strip()
                if role.lower() not in _ROLE_JUNK:
                    av["r"][role] += 1
            # name parts: for any person the transcript actually names -- which now
            # includes the interviewee, once the identification stage has found the
            # speaker's own name. Verified by graph/checks/names.py.
            if pid != 0 or e.mentions:
                for key, slot in (("given_name", "gn"), ("surname", "sn")):
                    part = (v.get(key) or "").strip()
                    if part and part.lower() not in _ROLE_JUNK:
                        av[slot][part] += 1
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

    # ---- gender / name parts -> PROPOSALS for graph.second_line -------------
    # These three fields have a rule source, so they are arbitrated centrally:
    # gender from `kinship.KINSHIP_GENDER` (named persons) and
    # `attributes.infer_interviewee_gender` (the speaker); name parts from the
    # token split in `attributes.infer_person_attributes`. This module no longer
    # writes them.
    proposals: dict = {}

    def propose(e, field, counter):
        if not counter:
            return
        value, votes = counter.most_common(1)[0]
        proposals.setdefault(e.entity_id, {})[field] = {
            "value": value,
            "confidence": "high" if votes > 1 else "low",
        }

    for e in persons + [interviewee]:
        v = attr_votes[e.entity_id]
        # the interviewee's gender has its own policy (blocking tier, spouse-term
        # checker), so it is keyed under a distinct field name
        propose(e, "interviewee_gender" if e is interviewee else "gender", v["g"])
        if e is not interviewee or e.mentions:
            propose(e, "given_name", v["gn"])
            propose(e, "surname", v["sn"])
        if e is not interviewee:
            # `role` is a PROPOSAL now. It does have a rule source -- the kinship-edge
            # detail, or a professional construction beside a mention
            # (`graph.attributes.infer_person_role`) -- so it is arbitrated centrally
            # and checked by `checks/persons.role_corroborated`, rather than being
            # written here as advisory text nothing could refute.
            propose(e, "role", v["r"])

    # ---- ethnicity -> a PROPOSAL, arbitrated and CHECKED -----------------------
    # This used to be written in place, unchecked, for everyone: on the sample
    # transcripts every named person inherited the speaker's ethnicity as an
    # `inferred` guess from their name alone. It is now an ordinary second-lined
    # field. A quote-grounded `stated` label is proposed at the model's confidence;
    # an `inferred` one is proposed as low-confidence, and
    # `checks/ethnicity.attributed_to_this_person` refutes it unless the label is
    # actually tied to THIS person in the text.
    for e in persons + [interviewee]:
        v = attr_votes[e.entity_id]
        if v["eth_stated"]:
            label, votes = v["eth_stated"].most_common(1)[0]
            proposals.setdefault(e.entity_id, {})["ethnicity"] = {
                "value": label, "confidence": "high" if votes > 1 else "low",
                "basis": "stated", "evidence": v["eth_ev"].get(label, "")}
        elif v["eth_inferred"]:
            label = v["eth_inferred"].most_common(1)[0][0]
            proposals.setdefault(e.entity_id, {})["ethnicity"] = {
                "value": label, "confidence": "low", "basis": "inferred",
                "evidence": ""}

    # ---- aliases -> MERGE RECORDS for graph.second_line ------------------------
    # Still never auto-merged. But the claim is now DATA rather than a flag written
    # here, so `graph.second_line._resolve_merges` can arbitrate it, run the
    # checkers in `graph/checks/merges.py`, and give the decision a Resolution and a
    # ledger row -- which alias/coref merges never had.
    merges = []
    for (ae, be, ev) in alias_ev.values():
        merges.append({"a": ae.entity_id, "b": be.entity_id, "evidence": ev,
                       "source": "llm", "confidence": "unstated"})

    # ---- relations -> RAW proposals, verified by graph/checks/relation_evidence.py -------
    # This module no longer verifies its own relation proposals. Verification is a
    # deterministic checker, so it moved to `graph/checks/relation_evidence.py` and runs
    # behind the `relation` field in `graph.second_line` -- which is what gives
    # relations a Resolution, a provenance record and a row in the ledger, like
    # every other field.
    relations = []
    for (src, tgt), slot in rel_votes.items():
        rel, votes = slot["rel"].most_common(1)[0]
        relations.append({"source": src, "target": tgt, "detail": rel,
                          "evidence": slot["ev"],
                          "confidence": "high" if votes > 1 else "low"})
    return proposals, relations, merges
