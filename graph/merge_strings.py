"""
Part 1 of Clustering. The following rules are applied:
1. Exact normalized match          "Maria" == "maria" == "Aunt Maria"(stripped)
2. Token containment               "Maria" is a token of "Maria Rodriguez"
IMPORTANT: a short form merges into a long form ONLY if exactly one
candidate exists. If both "Maria Rodriguez" and "Maria Hayes" are present,
just "Maria" stays unmerged and gets flagged for review.
"""


from __future__ import annotations
from .models import Entity, Mention
import re


# Words we strip before comparing name strings; used later elsewhere. Covers
# possessives, honorific titles, kinship terms, and the descriptive modifiers
# that can precede a name ("my *older* sister Jen"). Kept as single tokens
# because normalize() splits on whitespace / commas / periods.
KINSHIP_AND_TITLES = {
    # possessive determiners
    "my", "his", "her", "their", "our", "your",
    # honorific / occupational titles
    "mr", "mr.", "mrs", "mrs.", "ms", "ms.", "miss", "mx", "dr", "dr.",
    "prof", "prof.", "professor", "sir", "madam", "ma'am", "rev", "rev.",
    "reverend", "pastor", "father", "sister", "brother", "captain", "capt",
    "sergeant", "sgt", "officer", "judge", "senator", "governor", "gov",
    "mayor", "president", "pres", "coach", "principal", "auntie", "uncle",
    # kinship terms (single-token and hyphenated forms)
    "aunt", "aunty", "mother", "mom", "mommy", "mum", "mama", "momma", "ma",
    "dad", "daddy", "papa", "poppa", "pop", "pops", "pa",
    "grandma", "grandpa", "grandmother", "grandfather", "granny", "nana",
    "grandmom", "grandad", "granddad", "gramps", "cousin", "sis", "bro",
    # Dialect grandparent terms. Absent here, `normalize("Papaw Clarence")` kept
    # "papaw" as a name token, so the form did not exact-match a bare "Clarence"
    # and the token split handed `given_name="Papaw"` to the surrogate generator.
    # `checks/names._EXTRA_KIN` caught the second symptom; nothing caught the first.
    "mamaw", "papaw", "meemaw", "pawpaw", "memaw", "mawmaw", "mammaw", "pappy",
    "gramma", "grammy", "mimi",
    "husband", "wife", "son", "daughter", "niece", "nephew", "grandson",
    "granddaughter", "twin", "partner", "spouse", "sibling", "parent",
    "child", "kid", "boyfriend", "girlfriend", "fiance", "fiancee",
    "stepmom", "stepdad", "stepmother", "stepfather", "stepsister",
    "stepbrother", "stepson", "stepdaughter", "godmother", "godfather",
    "brother-in-law", "sister-in-law", "mother-in-law", "father-in-law",
    "son-in-law", "daughter-in-law", "half-brother", "half-sister",
    # descriptive modifiers that can sit between a title/possessive and a name
    "older", "elder", "younger", "little", "big", "baby", "oldest",
    "youngest", "eldest", "only", "middle", "maternal", "paternal",
    "biological", "bio", "adoptive", "adopted", "foster", "late", "dear",
    "beloved", "step", "half", "great", "grand", "former", "current", "ex",
}


# Honorific titles (subset of the strip-set above). Unlike kinship prefixes,
# an honorific marks social distance, so a titled form + a bare form that share
# only one given name ("Miss Rosa" vs "Rosa") can be different people.
HONORIFIC_TITLES = {
    "mr", "mrs", "ms", "miss", "mx", "dr", "prof", "professor", "sir",
    "madam", "maam", "ma'am", "rev", "reverend", "pastor", "captain", "capt",
    "sergeant", "sgt", "officer", "judge", "senator", "governor", "gov",
    "mayor", "president", "pres", "coach", "principal",
}


"""The honorific title leading a mention ('Miss Rosa' -> 'miss'), else ''."""
def _honorific(text: str) -> str:
    for raw in re.split(r"[\s,]+", text):
        t = raw.strip("'\".()-").lower()
        if t in HONORIFIC_TITLES:
            return t
    return ""


# "H-A-Y-E-S", "H A Y E S", "H.A.Y.E.S" --> collapse into the word "hayes"
_SPELLOUT = re.compile(r"^(?:[A-Za-z][\s\-\.]){2,}[A-Za-z]\.?$")


"""
Converts spelled out name to regular name; eg. "S-A-M" to "sam"
"""
def collapse_spellout(text: str) -> str:
    if _SPELLOUT.match(text.strip()):
        return re.sub(r"[\s\-\.]", "", text).lower()
    return text


"""
Takes a span and makes it lowercase; also strips punctuation and prefix words.
Returns name tokens in a tuple (list may be empty). Example usage:
"my aunt, Dr. Sarah Hayes" --> ("sarah", "hayes")
"""
def normalize(text: str) -> tuple[str, ...]:
    text = collapse_spellout(text)
    tokens = []
    for raw in text.replace(",", " ").replace(".", " ").split():
        t = raw.strip("'\"()-").lower()
        if t and t not in KINSHIP_AND_TITLES:
            tokens.append(t)
    # use tuples bc they're immutable & bc you can use them for dict keys
    return tuple(tokens)


# Religious/professional forms of address that are NOT also kinship terms. A single
# name token behind one of these is a SURNAME. Kept separate from
# `HONORIFIC_TITLES` because that set drives clustering (`_honorific`), and widening
# it there would change which bare names are held apart as possible collisions.
_TITLE_ONLY = {
    "rev", "reverend", "pastor", "deacon", "elder", "bishop", "monsignor",
    "rabbi", "imam", "chaplain", "fr", "pr",
}
# Words that are BOTH a kinship term and a form of religious address. "Father
# Nguyen" is a priest's surname; "my father, Earl" is a given name. Nothing in a bare
# surface form distinguishes them, and English religious usage genuinely goes both
# ways -- "Brother Estep" is a surname, "Sister Agnes" is a given name -- so the rule
# ABSTAINS on the single-token case rather than guessing. Both name-part policies are
# REQUIRED_OR_ABSTAIN, so an abstention is safe: the LLM proposes and
# `checks/names.py` gates the answer.
_TITLE_OR_KIN = {"father", "mother", "brother", "sister", "padre"}


def split_name_parts(text: str) -> tuple[str | None, str | None]:
    """THE rule split of one surface form into (given_name, surname).

    Shared by `attributes.infer_person_attributes` and
    `interviewee._name_parts`, which each carried their own copy of it.

    A SINGLE remaining token is slotted by WHAT PRECEDED IT, which is the fix for
    the mis-slotting `checks/names.py` could only report after the fact:

      * honorific + one token  -> that token is a SURNAME. "Father Nguyen",
        "Mr. Landry", "Dr. Combs", "Ms. Boudreaux", "Brother Estep" all name the
        family name; slotting it as a given name meant the surrogate generator
        minted a fake FIRST name to stand in for a surname. Verified: every
        titled person in both sample transcripts came out with
        `given_name = <their surname>`, and `_cross_field_consistency` could only
        drop it afterwards -- and only when the LLM happened to fill `surname`
        with the same token, which it did not do for "Father Nguyen".
      * kinship word + one token -> that token is a GIVEN name ("Aunt Maria",
        "Papaw Clarence"), which is the pre-existing behaviour.

    Titles and kin words are stripped from the token list either way, so a form
    with two or more real name tokens is unchanged: first is given, last is surname.
    """
    raw = [t for t in re.split(r"[\s,]+", text or "") if t]
    kept = [t for t in raw if t.lower().strip(".") not in KINSHIP_AND_TITLES]
    if len(kept) >= 2:
        return kept[0], kept[-1]
    if len(kept) == 1:
        # was the stripped prefix an HONORIFIC (social distance -> surname), a
        # kinship term (intimacy -> given name), or a word that is both?
        dropped = [t.lower().strip(".") for t in raw if t not in kept]
        if any(d in _TITLE_OR_KIN for d in dropped):
            return None, None                 # genuinely ambiguous -- abstain
        if any(d in HONORIFIC_TITLES or d in _TITLE_ONLY for d in dropped):
            return None, kept[0]
        return kept[0], None
    return None, None


"""
Implements an union find to later find/merge mention indices. Supports two
functions: find (an element in a group0 and union (two groups).
"""
class _UnionFind:
    # initially every element is its own group
    # eg. UnionFind(5) produces [0, 1, 2, 3, 4] --> {0} {1} {2} {3} {4}
    def __init__(self, n: int):
        self.parent = list(range(n))

    # say parent = [0,0,0,3,3] --> this means node 0 has parent = 0,
    # node 1 has parent 0,... node 4 has parent 3, node 5 has parent 3
    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        self.parent[self.find(a)] = self.find(b)


def _llm_says_distinct(persons, uf, bare_root, cand_root, transcript, llm):
    """Ask the LLM adjudicator whether the bare-name group and the full-name group are
    the SAME person. Returns `(veto, verdict)`; `veto` is True ONLY on a confident
    'different' verdict -> veto the containment merge. Any other outcome (same /
    unsure / no verdict) allows the merge, so rules-only behavior and clustering
    recall are preserved.

    The verdict is returned as well as consumed, because this LLM decision changes
    who the graph thinks exists and therefore owes the second line a record. It used
    to be applied and forgotten: the veto wrote a free-text review flag and nothing
    else, so the one decision class the arbitration layer could not see was an LLM
    call inside the clustering rule. `merge_person_mentions` now emits a
    `same_person` record for it (see the module docstring in
    `graph/second_line.py`).
    """
    from llm_layer import adjudicate_same_person
    bare = Entity(entity_id="_bare", category="PERSON",
                  mentions=[persons[j] for j in range(len(persons)) if uf.find(j) == bare_root])
    cand = Entity(entity_id="_cand", category="PERSON",
                  mentions=[persons[j] for j in range(len(persons)) if uf.find(j) == cand_root])
    v = adjudicate_same_person(llm, transcript, bare, cand)
    veto = bool(v) and v.get("same") is False and v.get("confidence") == "high"
    return veto, (v or {})


"""
Group PERSON/NICKNAME mentions into entities. Returns a tuple of
(person_entities, ambiguous_mentions_left_unmerged, veto_records).

When `transcript` and `llm` are supplied and the LLM is up, a bare given name that
would merge into a SINGLE full-name candidate is first adjudicated: the merge is
vetoed only if the LLM is confident the two are different people (e.g. uncle "Bill"
vs. foreman "Bill Ratliff"). With no LLM this is a no-op -- the rule merges as before.

`veto_records` are `same_person` merge records for the pairs that adjudication kept
apart: one `source="rule"` record with `applied=False` (the containment rule DID want
to merge them) and one `source="llm"` record with `value=False` (the model
disagreed). `graph.second_line._resolve_merges` arbitrates the pair, so the split
gets a Resolution, provenance and a ledger row -- which an LLM decision this
consequential should never have gone without.
"""
def merge_person_mentions(transcript_id: str, mentions: list[Mention],
                          transcript: str | None = None, llm=None
                          ) -> tuple[list[Entity], list[Mention], list[dict]]:
    persons = [m for m in mentions if m.entity_type in ("PERSON", "NICKNAME")]
    if not persons:
        return [], [], []

    norm = [normalize(m.text) for m in persons]  # eg. ("sarah", "hayes")
    canon = norm
    honor = [_honorific(m.text) for m in persons]  # "" unless a title leads it
    uf = _UnionFind(len(persons))  # just initialize, not merged yet

    # Exact normalized/canonical match. For a distinctive (multi-token) name we
    # merge on the name alone. For a BARE given name we additionally key on the
    # honorific, so "Miss Rosa" and "Rosa" land in DIFFERENT groups instead of
    # silently over-merging two people who share a first name. Bare "Aunt Maria"
    # and "Maria" still merge (kinship prefix carries no honorific).
    seen: dict[tuple, int] = {}
    honorifics_by_key: dict[tuple, set[str]] = {}  # canon key -> honorifics seen
    for i, key in enumerate(canon):  # eg. 0: ("maria",), 1: ("maria",)
        if len(key) == 1:
            honorifics_by_key.setdefault(key, set()).add(honor[i])
            merge_key = (key, honor[i])
        else:
            merge_key = (key, "")
        if merge_key in seen:
            uf.union(i, seen[merge_key])
        else:
            seen[merge_key] = i

    # bare-name keys seen with >1 distinct honorific profile = a likely collision
    collision_keys = {k for k, hs in honorifics_by_key.items() if len(hs) > 1}

    # now single-token forms merge into multi-token forms that
    # contain them but only if exactly ONE candidate group exists
    ambiguous: list[Mention] = []
    multi_groups: dict[int, set[str]] = {}
    for i, key in enumerate(canon):
        if len(key) >= 2:
            multi_groups.setdefault(uf.find(i), set()).update(key)

    llm_on = transcript is not None and llm is not None and llm.available()
    withheld_roots: set[int] = set()      # bare groups the LLM judged distinct
    veto_memo: dict[tuple, bool] = {}
    # bare_root -> (candidate_root, verdict), for the records emitted below. Kept
    # keyed by ROOT because entity ids do not exist until the groups are built.
    vetoed_pairs: dict[int, tuple] = {}
    for i, key in enumerate(canon):
        if len(key) != 1:
            continue
        token = key[0]
        candidates = {root for root, toks in multi_groups.items() if token in toks}
        candidates.discard(uf.find(i))
        if len(candidates) == 1:
            cand = candidates.pop()
            bare_root = uf.find(i)
            if llm_on:
                mk = (bare_root, cand)
                if mk not in veto_memo:
                    veto_memo[mk] = _llm_says_distinct(persons, uf, bare_root, cand,
                                                       transcript, llm)
                veto, verdict = veto_memo[mk]
                if veto:
                    withheld_roots.add(bare_root)     # keep the bare name separate
                    vetoed_pairs[bare_root] = (cand, verdict)
                    continue
            uf.union(i, cand)
        elif len(candidates) > 1:
            ambiguous.append(persons[i])

    # use the union-find groups to create Entity objects (track member indices
    # so we can look up each group's canon keys for collision flagging)
    groups: dict[int, list[int]] = {}
    for i in range(len(persons)):
        groups.setdefault(uf.find(i), []).append(i)

    entities = []
    id_by_root: dict[int, str] = {}
    for n, (root, idxs) in enumerate(
        sorted(groups.items(), key=lambda kv: min(persons[i].start for i in kv[1]))
    ):
        group = [persons[i] for i in idxs]
        e = Entity(
            entity_id=f"{transcript_id}_e{n + 1:03d}",
            category="PERSON",
            mentions=sorted(group, key=lambda m: m.start),
        )
        id_by_root[root] = e.entity_id
        if any(m in ambiguous for m in group):
            e.flag_entity("short name matches multiple long names")
        # this bare name also appears with a different honorific elsewhere
        # (e.g. "Rosa" here vs "Miss Rosa"): possibly a different person
        if any(canon[i] in collision_keys for i in idxs):
            e.flag_entity("bare given name also appears with a title elsewhere; "
                          "possible distinct people")
        entities.append(e)

    # `same_person` records for the withheld pairs, so the split is arbitrated rather
    # than merely annotated. The review flag itself is now written by
    # `second_line._resolve_merges` alongside the Resolution, so it is not duplicated
    # here.
    veto_records: list[dict] = []
    for bare_root, (cand_root, verdict) in vetoed_pairs.items():
        a_id, b_id = id_by_root.get(bare_root), id_by_root.get(cand_root)
        if a_id is None or b_id is None:
            continue
        ev = str((verdict or {}).get("evidence") or "")
        conf = str((verdict or {}).get("confidence") or "unstated")
        veto_records.append({"a": a_id, "b": b_id, "evidence": ev,
                             "source": "rule", "applied": False, "folded": None})
        veto_records.append({"a": a_id, "b": b_id, "evidence": ev, "value": False,
                             "source": "llm", "confidence": conf, "applied": False})
    return entities, ambiguous, veto_records