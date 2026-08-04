"""
Part 2 of Clustering: coreference resolution (the ML layer). This part focuses
on merging duplicate entities from the last stage (eg. merges "Peanut" / Maria).
When fastcoref reads the transcript, it produces something like this:
clusters = [
    [(226, 231), (250, 255), (315, 318), (320, 337), (394, 397)],
    [(478, 485), (497, 510), ...],
]
Where [(226, 231), (250, 255), (315, 318), (320, 337), (394, 397)] might refer
to "Maria", "My mom's sister", "Peanut", "Her", "Mar" etc (the SAME person). We refine
our entities list by merging Maria and Peanut and dropping non-identifying phrases
like "my mom's sister" and "her". "Mar"/"Maria" (nicknames) were resolved in the
last stage.
"""

from __future__ import annotations
import re
from .models import Entity
from .merge_strings import normalize
from llm_layer import adjudicate_same_person
from fastcoref import FCoref


def _overlapping_entity(entities: list[Entity], start: int, end: int):
    for e in entities:
        for m in e.mentions:
            if m.start < end and start < m.end:
                return e
    return None


"""(start, end) for each sentence, split on . ! ? (same scheme as elsewhere)."""
def _sentence_bounds(transcript: str) -> list[tuple[int, int]]:
    stops = [0] + [m.end() for m in re.finditer(r"[.!?]", transcript)]
    return list(zip(stops, stops[1:] + [len(transcript)]))


def _sent_index(bounds: list[tuple[int, int]], pos: int) -> int:
    for i, (s, e) in enumerate(bounds):
        if s <= pos < e:
            return i
    return len(bounds) - 1


"""
True if a mention of `a` and a mention of `b` sit in the same sentence
(strong signal they are two distinct people, not one).
"""
def _same_sentence(a: Entity, b: Entity, bounds) -> bool:
    a_sents = {_sent_index(bounds, m.start) for m in a.mentions}
    b_sents = {_sent_index(bounds, m.start) for m in b.mentions}
    return bool(a_sents & b_sents)


"""Lowercased name tokens across all of an entity's forms, titles/kin stripped."""
def _name_tokens(ent: Entity) -> set[str]:
    toks: set[str] = set()
    for form in ent.sorted_mentions:
        toks.update(normalize(form))
    return toks


"""
Could these two be the SAME name written differently? Used to corroborate
a coref-suggested merge. Conservative; requires positive evidence.
"""
def _name_compatible(a: Entity, b: Entity) -> bool:
    ta, tb = _name_tokens(a), _name_tokens(b)
    if not ta or not tb:
        return True                                   # one side is descriptor-only
    if ta & tb:
        return True                                   # shared token ("Maria Lopez"/"Maria")
    for x in ta:                                      # one a prefix of the other ("Will"/"William")
        for y in tb:
            if len(x) >= 3 and len(y) >= 3 and (x.startswith(y) or y.startswith(x)):
                return True
    return False


def _genders_conflict(a: Entity, b: Entity) -> bool:
    ga, gb = a.attributes.get("gender"), b.attributes.get("gender")
    return ga is not None and gb is not None and ga != gb


"""Fold `other` into `base` (mentions, non-null attributes, review flag)."""
def _merge_into(base: Entity, other: Entity, person_entities: list[Entity]) -> None:
    base.mentions.extend(other.mentions)
    base.mentions.sort(key=lambda m: m.start)
    base.attributes.update(
        {k: v for k, v in other.attributes.items() if v is not None})
    if other.needs_review:
        base.flag_entity(other.review_reason)
    person_entities.remove(other)


"""
Run coref and fold its clusters into our entities.
Returns (entities, merged_pairs, ran_flag). merged_pairs lists only the pairs
that were ACTUALLY merged (suggestions that were merely flagged are not counted).
"""
def apply_coref(transcript: str, person_entities: list[Entity], llm=None) -> tuple[list[Entity], list[tuple[str, str]], bool]:
    model = FCoref()
    pred = model.predict(texts=[transcript])[0]  # pass in an one-item list and extract the only item
    clusters = pred.get_clusters(as_strings=False)  # see header comment
    bounds = _sentence_bounds(transcript)

    merged_pairs: list[tuple[str, str]] = []
    for cluster in clusters:
        # Which of OUR entities does this cluster touch?
        # ALL OF THE BELOW IS FOR ONE CLUSTER (REFS TO ONE ENTITY)
        touched: list[Entity] = []
        for (s, e) in cluster:  # start, end
            ent = _overlapping_entity(person_entities, s, e)
            if ent is not None and ent not in touched:
                touched.append(ent)

        if len(touched) < 2:  # nothing to merge
            continue

        base = touched[0]  # a compatible/confirmed `other` is merged into base
        for other in touched[1:]:
            llm_on = llm is not None and llm.available()

            # A REJECTED coref link is (almost always) fastcoref over-linking two
            # distinct people -- keeping them apart is the CORRECT outcome, so we do
            # NOT flag it for review. Only a genuinely UNRESOLVED ambiguity (coref
            # says same, names differ, and no LLM to adjudicate) is worth a human.

            # Gender conflict is ALWAYS a hard block (strong two-way signal).
            if _genders_conflict(base, other):
                continue                                  # correct separation; no flag

            # Same-sentence co-occurrence is a hard block ONLY when there's no LLM.
            # With the LLM we let it read the sentence--it reliably tells an
            # explicit alias ("we called Roberto Beto") from two distinct people
            # in one sentence ("Sarah's brother Danny").
            if _same_sentence(base, other, bounds) and not llm_on:
                continue                                  # likely distinct; no flag

            # DOUBLE-GATE. The coref link is the deterministic half; when the LLM
            # is active it must CONFIRM the two are the same person (with evidence);
            # this recovers alias/nickname merges that we don't want to hardcode while
            # staying conservative. With no LLM we fall back to rules-only behavior.
            verdict = adjudicate_same_person(llm, transcript, base, other) if llm_on else None
            if verdict is not None:
                same = bool(verdict.get("same")) and (
                    verdict.get("confidence") == "high" or verdict.get("evidence"))
                if same:
                    if verdict.get("evidence"):
                        base.attributes.setdefault("merge_evidence", verdict["evidence"])
                    _merge_into(base, other, person_entities)
                    merged_pairs.append((base.entity_id, other.entity_id))
                # LLM declined -> correct separation; no flag
            elif _name_compatible(base, other):
                _merge_into(base, other, person_entities)
                merged_pairs.append((base.entity_id, other.entity_id))
            else:
                # coref says same, names differ, no LLM to adjudicate: genuinely
                # unresolved -- THIS is the one case worth a human's review.
                base.flag_entity(f"coref suggests same as {other.entity_id} but "
                                 f"names differ; needs review")
                other.flag_entity(f"coref suggests same as {base.entity_id} but "
                                  f"names differ; needs review")

    return person_entities, merged_pairs, True
