"""
Explicit alias / nickname resolution by RULE -- part of clustering, independent
of coref. Oral-history transcripts introduce nicknames with a small, CLOSED, and
distinctive set of constructions:

  "we called Roberto Beto"          call(ed) <name> <alias>
  "everybody just called her Glo"   call(ed) <pronoun> <alias>
  "everyone knew him as Big Jim"    knew <pronoun> as <alias>
  "His real name was Terrence"      <pronoun> real name was <name>
  "goes by Debbie" / "nicknamed X"  implicit-subject cues

Because the constructions are closed and distinctive, a rule can own them at high
precision -- and, unlike the coref path, it fires even when the two surface forms
are NOT name-compatible (Tank/Terrence, Gloria/Glo). Crucially, "named her after
my grandmother, Ruthie" is NOT one of these cues, so grandmother/granddaughter
stays split.

Runs right after part-1 clustering and BEFORE kinship/coref, so every downstream
stage sees the merged entity. Pronoun antecedents are resolved conservatively: the
nearest preceding person whose gender (read from an adjacent kinship word, since
attribute inference hasn't run yet) doesn't conflict with the pronoun, and only
when no fresh person reference ("a brother", "a baby") sits in between -- which is
what keeps "I have a brother too, we call him Chip" from merging Chip into a
nearby nephew.
"""

from __future__ import annotations
import re
from .kinship import KINSHIP_GENDER, _entity_at

# one capitalized name token / a 1-3 token proper name (case-sensitive)
_NAME_TOKEN = r"[A-Z][a-z]+(?:['’\-][A-Za-z]+)*"
_NAME = rf"{_NAME_TOKEN}(?:\s+{_NAME_TOKEN}){{0,2}}"
# subject of a cue: a third-person pronoun (case-insensitive) or a single name
_PRON = r"(?i:him|his|he|her|hers|she|them|their|they)"
_SUBJ = rf"(?:{_PRON}|{_NAME_TOKEN})"
_Q = r"[\"'“”‘’]?"

# call / called / calls / calling / callin' -- "calling" was previously missed
# ("everybody ended up calling him Sonny"), leaving the alias unmerged.
_CALL = re.compile(r"(?i:\bcall(?:ed|s|ing|in['’]?)?)\s+(" + _SUBJ + r")\s+" + _Q + r"(" + _NAME + r")" + _Q)
_AS = re.compile(r"(?i:\b(?:knew|know|knows|known))\s+(" + _SUBJ + r")(?i:\s+as\s+)"
                 + _Q + r"(" + _NAME + r")" + _Q)
_REAL = re.compile(r"(" + _SUBJ + r")(?i:\s+real\s+name\s+(?:was|is)\s+)" + _Q + r"(" + _NAME + r")" + _Q)
_BY = re.compile(r"(?i:\b(?:goes|go|went|going)\s+by\s+)" + _Q + r"(" + _NAME + r")" + _Q)
_NICK = re.compile(r"(?i:\bnicknamed\s+)" + _Q + r"(" + _NAME + r")" + _Q)

_PRON_GENDER = {"him": "M", "his": "M", "he": "M",
                "her": "F", "hers": "F", "she": "F",
                "them": None, "their": None, "they": None}

# person common-nouns; a fresh "a/my/... <noun>" between a candidate and the
# pronoun means the pronoun probably binds to that new referent, not the candidate
_PERSON_NOUNS = set(KINSHIP_GENDER) | {
    "guy", "man", "woman", "boy", "girl", "lady", "kid", "person", "dude",
    "fella", "fellow", "gentleman", "baby", "child", "friend", "buddy", "neighbor",
}
_NEW_PERSON = re.compile(
    r"\b(?:a|an|another|my|your|his|her|their|our|this|that)\s+(?:\w+\s+){0,2}?"
    r"(?:" + "|".join(re.escape(w) for w in sorted(_PERSON_NOUNS, key=len, reverse=True))
    + r")\b", re.I)

_WORD_BEFORE = re.compile(r"([A-Za-z][\w'’-]*)[\s,]+$")


def _local_gender(transcript: str, m) -> str | None:
    """Gender implied by a kinship word right before a mention ('my father James')."""
    w = _WORD_BEFORE.search(transcript[max(0, m.start - 24):m.start])
    if w:
        return KINSHIP_GENDER.get(w.group(1).lower().replace(".", ""))
    return None


def _intervening_person(transcript: str, a: int, b: int) -> bool:
    return a < b and bool(_NEW_PERSON.search(transcript[a:b]))


def _nearest_person(transcript, persons, pos: int, want_gender, exclude=None):
    """Nearest preceding person the pronoun could refer to, or None.

    `exclude` is the ALIAS entity itself. A pronoun in an alias cue refers to whoever
    the alias renames, never to the alias -- but the alias name very often appears just
    before the cue that introduces it:

        "My little brother, Minh -- Sonny, everybody ended up calling him Sonny"

    Here the nearest preceding mention is the FIRST "Sonny", so the antecedent resolved
    to the alias entity, `do()` saw `primary is alias_ent` and returned, and Minh/Sonny
    stayed split -- the single clustering split in the sample transcripts, which no
    later layer can repair because `same_person` never auto-merges. Excluding the alias
    lets the search reach Minh.
    """
    ex = getattr(exclude, "entity_id", None)
    cands = sorted(
        ((pos - m.end, m, e) for e in persons for m in e.mentions
         if m.end <= pos and e.entity_id != ex),
        key=lambda t: t[0])
    for gap, m, e in cands:
        if gap > 140:
            break
        g = _local_gender(transcript, m)
        if want_gender and g and g != want_gender:
            continue                        # wrong gender -> not the referent
        if _intervening_person(transcript, m.end, pos):
            return None                     # a nearer fresh referent intervenes
        return e
    return None


def _merge(base, other, persons) -> None:
    base.mentions.extend(other.mentions)
    base.mentions.sort(key=lambda m: m.start)
    base.attributes.update({k: v for k, v in other.attributes.items() if v is not None})
    if other.needs_review:
        base.flag_entity(other.review_reason)
    if other in persons:
        persons.remove(other)


def apply_alias_cues(transcript: str, persons: list) -> list[dict]:
    """Merge explicit alias/nickname pairs in place.

    Returns one MERGE RECORD per applied merge:

        {"a": kept_id, "b": folded_id, "evidence": <cue quote>,
         "source": "rule", "folded": <the folded Entity object>}

    `graph.second_line._resolve_merges` arbitrates these so a rule merge gets a
    Resolution, provenance and a ledger row like every other decision -- clustering
    used to be the one class of decision with none. The folded Entity rides along
    because it has been removed from `persons` by then, and the `same_person`
    checkers need both sides of the pair; the pipeline registers it on the
    CheckContext as an `extra_entities` entry.
    """
    merged: list[dict] = []
    if not persons:
        return merged

    def subject_entity(subj, s, e, alias_ent=None):
        if subj.lower() in _PRON_GENDER:
            return _nearest_person(transcript, persons, s, _PRON_GENDER[subj.lower()],
                                   exclude=alias_ent)
        return _entity_at(persons, s, e)

    def do(primary, alias_ent, evidence):
        if primary is None or alias_ent is None or primary is alias_ent:
            return
        rec = {"a": primary.entity_id, "b": alias_ent.entity_id,
               "evidence": (evidence or "").strip(), "source": "rule",
               "folded": alias_ent}
        _merge(primary, alias_ent, persons)
        merged.append(rec)

    # subject + alias, both spans present in the match. The alias is resolved FIRST so
    # the pronoun search can exclude it -- see `_nearest_person`.
    for rx in (_CALL, _AS, _REAL):
        for m in rx.finditer(transcript):
            alias_ent = _entity_at(persons, m.start(2), m.end(2))
            primary = subject_entity(m.group(1), m.start(1), m.end(1), alias_ent)
            do(primary, alias_ent, m.group(0))

    # implicit-subject cues: resolve the antecedent to the nearest person
    for rx in (_BY, _NICK):
        for m in rx.finditer(transcript):
            alias_ent = _entity_at(persons, m.start(1), m.end(1))
            primary = _nearest_person(transcript, persons, m.start(), None,
                                      exclude=alias_ent)
            do(primary, alias_ent, m.group(0))

    return merged
