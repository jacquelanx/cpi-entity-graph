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
from .models import Entity
from fastcoref import FCoref


def _overlapping_entity(entities: list[Entity], start: int, end: int):
    for e in entities:
        for m in e.mentions:
            if m.start < end and start < m.end:
                return e
    return None


"""
Run coref and fold its clusters into our entities. 
Returns (entities, merge_suggestions_applied_or_flagged, ran_flag).
"""
def apply_coref(transcript: str, person_entities: list[Entity]) -> tuple[list[Entity], list[tuple[str, str]], bool]:
    model = FCoref()
    pred = model.predict(texts=[transcript])[0]  # pass in an one-item list and extract the only item
    clusters = pred.get_clusters(as_strings=False)  # see header comment

    merged_pairs: list[tuple[str, str]] = []
    for cluster in clusters:
        # Which of OUR entities does this cluster touch?
        # ALL OF THE BELOW IS FOR ONE CLUSTER (REFS TO ONE ENTITY)

        touched: list[Entity] = []  # list of entities mentioned in this cluster (we're going to merge them)
        for (s, e) in cluster:  # start, end
            ent = _overlapping_entity(person_entities, s, e)  # does this span mention our entity?
            if ent is not None and ent not in touched:
                touched.append(ent)

        if len(touched) < 2:  # nothing to merge
            continue

        # Merge! Only when their surface forms don't contradict (no surname conflict)
        base = touched[0]  # suppose base = Entity A; everything will be merged into A
        for other in touched[1:]:
            base_tokens = {t for f in base.sorted_mentions for t in f.lower().split()}
            other_tokens = {t for f in other.sorted_mentions for t in f.lower().split()}

            # crude conflict test: two DIFFERENT multi-token names
            conflict = (
                any(len(f.split()) > 1 for f in base.sorted_mentions)           # does base have full name? (eg. maria lopez)
                and any(len(f.split()) > 1 for f in other.sorted_mentions)      # what ab the other entity?
                and not (base_tokens & other_tokens)                            # base and other share NO tokens
            )
            
            if conflict:
                base.flag_entity(f"coref links to {other.entity_id} but names conflict")
                other.flag_entity(f"coref links to {base.entity_id} but names conflict")
            else:
                base.mentions.extend(other.mentions)
                base.mentions.sort(key=lambda m: m.start)
                base.attributes.update(
                    {k: v for k, v in other.attributes.items() if v is not None}
                )
                person_entities.remove(other)
                merged_pairs.append((base.entity_id, other.entity_id))

    return person_entities, merged_pairs, True