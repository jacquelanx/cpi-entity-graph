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
from ..models import Entity
from .name_matching import normalize
from ..text.sentences import sentence_spans
from llm_layer import adjudicate_same_person
from fastcoref import FCoref


def _overlapping_entity(entities: list[Entity], start: int, end: int):
    for e in entities:
        for m in e.mentions:
            if m.start < end and start < m.end:
                return e
    return None


"""(start, end) for each sentence (abbreviation-aware; see graph/text/sentences.py)."""
def _sentence_bounds(transcript: str) -> list[tuple[int, int]]:
    return sentence_spans(transcript)


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
Returns (entities, merge_records, ran_flag). `merge_records` covers every pair the
coref link PROPOSED and the LLM adjudicated -- the ones it merged (`applied=True`)
and the ones the LLM vetoed (`applied=False`, `value=False`) -- in the same shape
`graph/aliases.apply_alias_cues` returns, so `graph.second_line._resolve_merges` can
give each one a Resolution and a ledger row. Pairs blocked by the deterministic
gender / same-sentence rules before the LLM was ever consulted are not recorded:
no LLM decision was made, so there is nothing to arbitrate.
"""
def apply_coref(transcript: str, person_entities: list[Entity], llm=None) -> tuple[list[Entity], list[dict], bool]:
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
                    ev = str(verdict.get("evidence") or "")
                    # TWO records: the coref link is the rule/ML half, the adjudication
                    # is the LLM half. Emitting only the first labelled the merge
                    # `source="rule"`, so `second_line._resolve_merges` saw no LLM
                    # answer and recorded a merge the MODEL had actually decided as
                    # "rules stand; no LLM answer to confirm them" -- provenance that
                    # was not merely thin but wrong. With both, the pair resolves
                    # `confirm` and (via `same_person`'s `unsafe_when`) the checkers
                    # in graph/checks/merges.py run on it.
                    rec = {"a": base.entity_id, "b": other.entity_id,
                           "evidence": ev, "source": "rule", "applied": True,
                           "folded": other}
                    merged_pairs.append(rec)
                    merged_pairs.append({
                        "a": base.entity_id, "b": other.entity_id, "evidence": ev,
                        "value": True, "source": "llm", "applied": True,
                        "confidence": str(verdict.get("confidence") or "unstated")})
                    _merge_into(base, other, person_entities)
                else:
                    # LLM DECLINED. Keeping them apart is (almost always) the right
                    # call, but it is still an LLM decision that changed who the graph
                    # thinks exists, and it used to leave NO trace at all: no merge
                    # record, no Resolution, no ledger row. `name_matching` emits
                    # paired rule/llm records for the exactly analogous containment
                    # veto, so this was the one class of LLM identity decision still
                    # outside the single arbitration point.
                    #
                    # Same shape as the containment veto: the coref link is the rule
                    # half that WANTED the merge (`applied=False`), the adjudication is
                    # the LLM half that refused it (`value=False`).
                    ev = str(verdict.get("evidence") or "")
                    merged_pairs.append({
                        "a": base.entity_id, "b": other.entity_id, "evidence": ev,
                        "source": "rule", "applied": False, "folded": None})
                    merged_pairs.append({
                        "a": base.entity_id, "b": other.entity_id, "evidence": ev,
                        "value": False, "source": "llm", "applied": False,
                        "confidence": str(verdict.get("confidence") or "unstated")})
            elif _name_compatible(base, other):
                rec = {"a": base.entity_id, "b": other.entity_id, "evidence": "",
                       "source": "rule", "applied": True, "folded": other}
                _merge_into(base, other, person_entities)
                merged_pairs.append(rec)
            else:
                # coref says same, names differ, no LLM to adjudicate: genuinely
                # unresolved -- THIS is the one case worth a human's review.
                base.flag_entity(f"coref suggests same as {other.entity_id} but "
                                 f"names differ; needs review")
                other.flag_entity(f"coref suggests same as {base.entity_id} but "
                                  f"names differ; needs review")

    return person_entities, merged_pairs, True
