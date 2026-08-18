"""
Deterministic verifier for proposed relations -- the CHECKER behind the `relation`
field in `graph/second_line.py`.

This module used to live in `llm_layer/` and was invoked by the proposer on itself,
which meant relations were the one field that never passed through the single
arbitration point: they produced no Resolution, carried no provenance, and did not
appear in the ledger. It is deterministic and rule-shaped, so it belongs here with
the other checkers; `llm_layer/extract.py` now returns RAW relation proposals and
this decides their fate.

For each candidate (source, target, rel, evidence) `verify_relation` returns:
  apply   -- strongly supported locally; the relation becomes an edge. Direction is
             canonicalized (the interviewee is forced to the SOURCE) and the
             evidence is re-grounded to a single nearby sentence (kills run-on
             quotes).
  suggest -- plausible but not locally provable (e.g. needs coreference, like
             "My mom ... Her name was Gloria"); surfaced for review, NO edge.
  reject  -- refuted or out of scope; dropped.

Refutations / drops:
  * either end is a public figure                  -> drops celebrity edges
  * a rel word outside the kin/social vocabulary   -> NOT an edge, but surfaced as a
    review SUGGESTION tagged with the raw word, so an out-of-table relation the
    rules can't score still reaches a human
  * a named/third-person possessor of the kin word -> drops "Carla's mom" read as
    the interviewee's mother, "Her husband, Ronnie" read as the interviewee's
  * whole-word grounding                           -> "ruth" no longer matches "ruthie"
"""


from __future__ import annotations
import re
from dataclasses import dataclass

from ..sentences import sentence_spans as _sentences
from .relwords import KIN_WORDS as _KIN_WORDS


# Social ties we keep as edges (single-token, matched as whole words). Anything
# outside KIN u SOCIAL is treated as out of scope and dropped.
_SOCIAL_WORDS = {
    "friend", "buddy", "boss", "manager", "supervisor", "coworker", "colleague",
    "neighbor", "teacher", "professor", "instructor", "tutor", "mentor", "coach",
    "counselor", "counsellor", "therapist", "doctor", "nurse", "sponsor",
    "landlord", "sergeant", "captain", "officer", "roommate", "classmate",
    "partner", "pastor", "priest", "rabbi", "imam", "caseworker",
}
_REL_WORDS = _KIN_WORDS | _SOCIAL_WORDS

# kin variants that mean the same relation, so a canon of "mother" is still
# supported by the transcript word "mama" / "mom".
_KIN_GROUPS = [
    {"mother", "mom", "mommy", "mum", "mummy", "mama", "mamma", "momma", "ma"},
    {"father", "dad", "daddy", "papa", "poppa", "pop", "pops", "pa"},
    {"grandmother", "grandma", "grandmom", "granny", "nana", "nanna", "gramma",
     "grammy", "meemaw", "mamaw", "mawmaw", "memaw", "mammaw", "mimi"},
    {"grandfather", "grandpa", "granddad", "grandad", "grandpop", "gramps",
     "papaw", "pawpaw", "pappy"},
    {"sister", "sis"},
    {"brother", "bro"},
]
_GROUP_OF = {w: g for g in _KIN_GROUPS for w in g}

# first-person cue (interviewee) and possessive markers just before a kin word
_FP_CUE = re.compile(r"\b(?:i|me|my|our|we|us)\b", re.I)
_POSS_FIRST = re.compile(r"\b(?:my|our)\s*$", re.I)
_POSS_THIRD = re.compile(r"\b(?:his|her|their)\s*$", re.I)
_POSS_NAMED = re.compile(r"[A-Za-z][\w'’-]*['’]s\s*$")


def _canon_rel(detail: str):
    """First whitelisted token in the (possibly modified) rel phrase, else None.
    'twin brother' -> 'brother', 'best friend' -> 'friend', 'musical idol' -> None."""
    for tok in re.split(r"\s+", (detail or "").strip().lower()):
        tok = tok.strip(",.;:")
        if tok in _REL_WORDS:
            return tok
    return None


def _search_terms(canon: str) -> set[str]:
    """Transcript words that would corroborate `canon` (its kin synonym group)."""
    return set(_GROUP_OF.get(canon, {canon}))


def _term_re(terms):
    # allow an optional plural 's' ("twin brothers", "my grandparents")
    alt = "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True))
    return re.compile(r"(?<![a-z])(?:" + alt + r")s?(?![a-z])", re.I)


@dataclass
class _Verdict:
    action: str                 # "apply" | "suggest" | "reject"
    source: str = ""
    target: str = ""
    detail: str = ""
    evidence: str = ""


class RelationContext:
    """Precomputed transcript structure shared across all candidate checks."""

    def __init__(self, transcript, persons, interviewee):
        self.transcript = transcript
        self.interviewee = interviewee
        self.sents = _sentences(transcript)
        self.ent_by_id = {e.entity_id: e for e in persons}
        self.ent_by_id[interviewee.entity_id] = interviewee
        # (start, end, entity_id) for every NAMED person mention
        self.person_spans = [(m.start, m.end, e.entity_id)
                             for e in persons for m in e.mentions]

    def sentence_of(self, pos: int) -> int:
        for i, (s, e) in enumerate(self.sents):
            if s <= pos < e:
                return i
        return len(self.sents) - 1

    def sentence_text(self, pos: int) -> str:
        s, e = self.sents[self.sentence_of(pos)]
        return self.transcript[s:e].strip()

    def person_between(self, a: int, b: int, exclude_id: str) -> bool:
        return any(a <= s < b and eid != exclude_id
                   for (s, e, eid) in self.person_spans)


def _possessor_before(transcript: str, floor: int, rel_start: int) -> str:
    pre = transcript[max(floor, rel_start - 18):rel_start]
    if _POSS_FIRST.search(pre):
        return "first"
    if _POSS_THIRD.search(pre):
        return "third"
    if _POSS_NAMED.search(pre):
        return "named"
    return "neutral"


def _interviewee_support(named, canon: str, ctx: RelationContext) -> str:
    """Is a kin/social tie between the interviewee and `named` locally provable?
    Returns 'supported' | 'refuted' | 'weak'."""
    terms = _search_terms(canon)
    term_re = _term_re(terms)
    apposite = re.compile(
        r"^[\s,]*(?:who\s+(?:is|was)\s+|who's\s+)?(my|our|his|her|their)\s+"
        r"(?:\w+\s+){0,2}?(?:" + "|".join(re.escape(t) for t in terms) + r")\b", re.I)
    refuted = False

    for m in named.mentions:
        si = ctx.sentence_of(m.start)
        ss, se = ctx.sents[si]

        # (1) kin word BEFORE the name, bound to THIS name ("my aunt Maria")
        pre_start = max(ss, m.start - 60)
        pre = ctx.transcript[pre_start:m.start]
        hits = list(term_re.finditer(pre))
        if hits:
            h = hits[-1]                                    # nearest to the name
            rel_abs_start, rel_abs_end = pre_start + h.start(), pre_start + h.end()
            if not ctx.person_between(rel_abs_end, m.start, named.entity_id):
                poss = _possessor_before(ctx.transcript, ss, rel_abs_start)
                if poss == "first":
                    return "supported"
                if poss in ("third", "named"):
                    refuted = True
                elif _FP_CUE.search(ctx.transcript[ss:rel_abs_start]):
                    return "supported"

        # (2) appositive AFTER the name ("Rosa, my aunt" / "..., who is my aunt")
        post = ctx.transcript[m.end:min(se, m.end + 45)]
        ap = apposite.match(post)
        if ap:
            if ap.group(1).lower() in ("my", "our"):
                return "supported"
            refuted = True

    return "refuted" if refuted else "weak"


def _mentions_in(ent, ss: int, se: int) -> bool:
    return any(ss <= m.start < se for m in ent.mentions)


def _pair_evidence(se_ent, te_ent, canon: str, ctx: RelationContext):
    """A single sentence holding BOTH people and a corroborating rel word."""
    term_re = _term_re(_search_terms(canon))
    for (ss, se) in ctx.sents:
        seg = ctx.transcript[ss:se]
        if _mentions_in(se_ent, ss, se) and _mentions_in(te_ent, ss, se) and term_re.search(seg):
            return seg.strip()
    return None


def verify_relation(src_eid, tgt_eid, rel, evidence, ctx: RelationContext) -> _Verdict:
    se = ctx.ent_by_id.get(src_eid)
    te = ctx.ent_by_id.get(tgt_eid)
    if se is None or te is None or se is te:
        return _Verdict("reject")

    iv = ctx.interviewee
    for end in (se, te):
        if end is not iv and end.subtype == "PUBLIC_FIGURE":
            return _Verdict("reject")                   # no edges to/from celebrities

    canon = _canon_rel(rel)
    if canon is None:
        # rel word is outside the known kin/social vocabulary table. The rules
        # can't corroborate it, but the LLM proposed it with grounded evidence, so
        # the second-line policy surfaces it as a REVIEW SUGGESTION (never an edge),
        # tagged with the raw relationship word. A proposal with no usable word is
        # dropped.
        m = re.search(r"[a-z][a-z\-']+", (rel or "").lower())
        if not m:
            return _Verdict("reject")
        raw = m.group(0)
        if se is iv or te is iv:
            named = te if se is iv else se
            if named is iv:
                return _Verdict("reject")
            ev = ctx.sentence_text(named.mentions[0].start) if named.mentions else (evidence or "")
            return _Verdict("suggest", iv.entity_id, named.entity_id, raw, ev)
        ev = ctx.sentence_text(te.mentions[0].start) if te.mentions else (evidence or "")
        return _Verdict("suggest", se.entity_id, te.entity_id, raw, ev)

    if se is iv or te is iv:
        named = te if se is iv else se
        if named is iv:
            return _Verdict("reject")
        support = _interviewee_support(named, canon, ctx)
        if support == "refuted":
            return _Verdict("reject")
        ev = ctx.sentence_text(named.mentions[0].start) if named.mentions else (evidence or "")
        # canonical direction: interviewee is always the SOURCE
        if support == "supported":
            return _Verdict("apply", iv.entity_id, named.entity_id, canon, ev)
        return _Verdict("suggest", iv.entity_id, named.entity_id, canon, ev)

    # person <-> person
    ev = _pair_evidence(se, te, canon, ctx)
    if ev is not None:
        return _Verdict("apply", se.entity_id, te.entity_id, canon, ev)
    fallback = ctx.sentence_text(te.mentions[0].start) if te.mentions else (evidence or "")
    return _Verdict("suggest", se.entity_id, te.entity_id, canon, fallback)
