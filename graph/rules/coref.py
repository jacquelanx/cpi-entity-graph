"""
Clustering, part 2: coreference resolution (the ML layer).

PURPOSE
    Catch the same-person merges that name matching cannot see, because the two
    surface forms share no name -- "Maria" and "Peanut". A coreference model
    reads the whole transcript and reports which spans refer to the same thing;
    this module maps that back onto our entities.

FIT
    The last clustering stage, after `rules/name_matching.py` and
    `rules/aliases.py`, called from `graph/pipeline.run_pipeline` (skippable via
    `run_coref=False`). This is the ONE module in `graph/rules/` that imports a
    heavyweight ML dependency (`fastcoref`) and the only rule module that calls
    `llm_layer` directly. Its merge records feed `graph/second_line`.

HOW -- a DOUBLE GATE, which is the whole design
    Coreference models over-link: they will happily fuse two different people who
    are discussed similarly, and an over-merge is unrecoverable (two real people
    become one). So a coref link is treated as a PROPOSAL that must be
    corroborated, never as a decision:

      * a gender conflict blocks the merge outright;
      * two entities in the SAME SENTENCE block it too, unless an LLM is
        available to read that sentence (co-occurrence usually means "Sarah's
        brother Danny" -- two people -- but sometimes means "we called Roberto
        Beto");
      * with an LLM, the model must positively CONFIRM the pair;
      * without one, the names must at least be compatible (`_name_compatible`).

    A blocked merge is normally the CORRECT outcome, so it is not flagged for
    review. Only the genuinely unresolved case -- coref says same, names differ,
    no LLM to ask -- goes to a human.

COREFERENCE, CONCRETELY
    When fastcoref reads the transcript, it produces clusters of character spans:

        clusters = [
            [(226, 231), (250, 255), (315, 318), (320, 337), (394, 397)],
            [(478, 485), (497, 510), ...],
        ]

    Each inner list is one referent. The first might be "Maria", "My mom's
    sister", "Peanut", "Her", "Mar" -- all the SAME person. We refine our entity
    list by merging Maria and Peanut, and drop the non-identifying phrases ("my
    mom's sister", "her") because they overlap no detected mention.
    "Mar"/"Maria" (nicknames) were already resolved by name matching.

NOTE ON THE NUMBERING
    "Part 2" follows the repo README, even though `aliases.py` also runs between
    this and part 1.
"""

from __future__ import annotations
from ..models import Entity
from .name_matching import normalize
from ..text.sentences import sentence_spans
from llm_layer import adjudicate_same_person
from fastcoref import FCoref


def _overlapping_entity(entities: list[Entity], start: int, end: int):
    """The first entity that has a mention overlapping the span `[start, end)`.

    Coref reports raw character spans; this is how a span is translated back into
    "which of OUR entities is that?". Uses overlap rather than equality because
    the coref model's span boundaries rarely match the detector's exactly -- it
    may report "my aunt Maria" where we detected "Maria". Returns None when the
    span covers no known person (a pronoun, or a descriptive phrase like "my
    mom's sister"), which is how such spans get dropped.
    """
    for e in entities:
        for m in e.mentions:
            if m.start < end and start < m.end:
                return e
    return None


def _sentence_bounds(transcript: str) -> list[tuple[int, int]]:
    """(start, end) for each sentence (abbreviation-aware; see graph/text/sentences.py).

    A thin named wrapper, so the same-sentence test below reads clearly and the
    splitter can be swapped in one place.
    """
    return sentence_spans(transcript)


def _sent_index(bounds: list[tuple[int, int]], pos: int) -> int:
    """Which sentence number contains offset `pos`.

    Returns an index into `bounds`, so two positions are in the same sentence
    exactly when they yield the same number. Falls back to the last sentence for
    a position past the end of the text.
    """
    for i, (s, e) in enumerate(bounds):
        if s <= pos < e:
            return i
    return len(bounds) - 1


def _same_sentence(a: Entity, b: Entity, bounds) -> bool:
    """True if any mention of `a` shares a sentence with any mention of `b`.

    A strong signal they are two DISTINCT people. Speakers do not usually
    introduce the same person twice in one breath, so "Sarah's brother Danny"
    naming both in one sentence means they are two entities.

    HOW: collect the set of sentence numbers each entity appears in, then
    intersect them -- `a_sents & b_sents` is non-empty exactly when some sentence
    contains both.
    """
    a_sents = {_sent_index(bounds, m.start) for m in a.mentions}
    b_sents = {_sent_index(bounds, m.start) for m in b.mentions}
    return bool(a_sents & b_sents)


def _name_tokens(ent: Entity) -> set[str]:
    """Every lowercased name token this entity was ever written with.

    Runs `normalize` over each distinct surface form and unions the results, so
    an entity written as both "Maria" and "Maria Lopez" yields
    `{"maria", "lopez"}`. Titles and kin words are stripped by `normalize`.
    """
    toks: set[str] = set()
    for form in ent.sorted_mentions:
        toks.update(normalize(form))
    return toks


def _name_compatible(a: Entity, b: Entity) -> bool:
    """Could these two entities be the SAME name written differently?

    The no-LLM corroboration for a coref link. Three ways to say yes:

      * ONE SIDE HAS NO NAME at all (only descriptors like "my mom's sister"), so
        there is nothing to contradict -- permissive by design, since coref is the
        only thing that could ever attach such a phrase to a person.
      * A SHARED TOKEN ("Maria Lopez" / "Maria").
      * ONE TOKEN IS A PREFIX OF THE OTHER ("Will" / "William"). The 3-character
        minimum keeps initials and very short fragments from matching almost
        anything.

    Otherwise no. Conservative on purpose: this is the gate that stops coref
    fusing two unrelated people when no LLM is available to arbitrate.
    """
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
    """True only when both entities have a KNOWN and DIFFERENT gender.

    A missing gender on either side is not a conflict -- unknown is not evidence.
    Used as a hard block on merging, since a gender disagreement is a strong
    two-way signal that coref linked two different people.
    """
    ga, gb = a.attributes.get("gender"), b.attributes.get("gender")
    return ga is not None and gb is not None and ga != gb


def _merge_into(base: Entity, other: Entity, person_entities: list[Entity]) -> None:
    """Fold `other` into `base` and remove it from the person list.

    Same operation as `aliases._merge`: `base` absorbs the mentions (kept in
    transcript order) and any non-None attributes, and inherits a review flag so
    a concern about the folded entity is not lost with it.
    """
    base.mentions.extend(other.mentions)
    base.mentions.sort(key=lambda m: m.start)
    base.attributes.update(
        {k: v for k, v in other.attributes.items() if v is not None})
    if other.needs_review:
        base.flag_entity(other.review_reason)
    person_entities.remove(other)


def apply_coref(transcript: str, person_entities: list[Entity], llm=None) -> tuple[list[Entity], list[dict], bool]:
    """Run the coreference model and fold its clusters into the person entities.

    Returns `(entities, merge_records, ran_flag)`. Entities are modified IN PLACE
    and the same list is returned, minus anything folded away. `ran_flag` is
    always True on return -- reaching this point means the model ran.

    HOW: predict clusters over the whole transcript, then for each cluster map its
    spans onto our entities (`touched`). A cluster touching fewer than two
    entities has nothing to merge. Otherwise the FIRST entity touched becomes
    `base` and each remaining one is considered for folding into it, subject to
    the double gate described in the module docstring.

    `merge_records` covers every pair the coref link PROPOSED and the LLM
    adjudicated -- the ones it merged (`applied=True`) and the ones the LLM vetoed
    (`applied=False`, `value=False`) -- in the same shape
    `graph/aliases.apply_alias_cues` returns, so `graph.second_line._resolve_merges`
    can give each one a Resolution and a ledger row. Pairs blocked by the
    deterministic gender / same-sentence rules before the LLM was ever consulted are
    not recorded: no LLM decision was made, so there is nothing to arbitrate.
    """
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
