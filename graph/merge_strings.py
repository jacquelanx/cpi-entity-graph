"""
Part 1 of Clustering: merges PERSON/NICKNAME. The following rules are applied:
1. Exact normalized match          "Maria" == "maria" == "Aunt Maria"(stripped)
2. Token containment               "Maria" is a token of "Maria Rodriguez"
3. Nickname table                  "Mar" -> "Maria", "Bill" -> "William"
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


# Nickname / diminutive -> canonical given name. We try to be CONSERVATIVE to
# avoid overmerging. Not sure if we should keep this though... 
NICKNAMES = {
    "johnny": "john", "steve": "steven", "stevie": "steven",
    "eddie": "edward", "charlie": "charles", "ben": "benjamin", 
    "benny": "benjamin", "sam": "samuel", "sammy": "samuel",
    "nick": "nicholas", "greg": "gregory", "fred": "frederick",
    "ronnie": "ronald", "donnie": "donald",
    # female
    "susie": "susan", "suzy": "susan", "maggie": "margaret",
    "cathy": "catherine", "tricia": "patricia", "barb": "barbara",
    "deb": "deborah", "debbie": "deborah", "becca": "rebecca",
    "abby": "abigail", "vicky": "victoria", "steph": "stephanie",
    "jess": "jessica", "jessie": "jessica", "kim": "kimberly", 
}


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


"""
Resolve a nickname to the associated given name (or itself).
"""
def canonical_token(token: str) -> str:
    return NICKNAMES.get(token, token)


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


"""
Group PERSON/NICKNAME mentions into entities. Returns tuple in the form of
(person_entities, ambiguous_mentions_left_unmerged).
"""
def merge_person_mentions(transcript_id: str, mentions: list[Mention]) -> tuple[list[Entity], list[Mention]]:
    persons = [m for m in mentions if m.entity_type in ("PERSON", "NICKNAME")]
    if not persons:
        return [], []

    norm = [normalize(m.text) for m in persons]  # eg. ("sarah", "hayes")
    canon = [tuple(canonical_token(t) for t in toks) for toks in norm]  
    # eg. canon = [("maria",), ("maria",), ("maria","rodriguez")]
    uf = _UnionFind(len(persons))  # just initialize, not merged yet

    # exact normalized/canonical match
    seen: dict[tuple, int] = {}
    for i, key in enumerate(canon):  # eg. 0: ("maria",), 1: ("maria",)
        if key in seen:
            uf.union(i, seen[key])
        else:
            seen[key] = i

    # now single-token forms merge into multi-token forms that
    # contain them but only if exactly ONE candidate group exists
    ambiguous: list[Mention] = []
    multi_groups: dict[int, set[str]] = {}
    for i, key in enumerate(canon):
        if len(key) >= 2:
            multi_groups.setdefault(uf.find(i), set()).update(key)

    for i, key in enumerate(canon):
        if len(key) != 1:
            continue
        token = key[0]
        candidates = {root for root, toks in multi_groups.items() if token in toks}
        candidates.discard(uf.find(i))
        if len(candidates) == 1:
            uf.union(i, candidates.pop())
        elif len(candidates) > 1:
            ambiguous.append(persons[i])

    # use the union-find groups to create Entity objects
    groups: dict[int, list[Mention]] = {}
    for i, m in enumerate(persons):
        groups.setdefault(uf.find(i), []).append(m)

    entities = []
    for n, (_, group) in enumerate(
        sorted(groups.items(), key=lambda kv: min(m.start for m in kv[1]))
    ):
        e = Entity(
            entity_id=f"{transcript_id}_e{n + 1:03d}",
            category="PERSON",
            mentions=sorted(group, key=lambda m: m.start),
        )
        if any(m in ambiguous for m in group):
            e.flag_entity("short name matches multiple long names")
        entities.append(e)
    return entities, ambiguous