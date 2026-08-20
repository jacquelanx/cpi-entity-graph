"""
Orchestrator that wires the individual graph stages together.
Given a transcript and its DETECTED mentions (from the detection stage),
it runs every stage and returns the full (entities, edges) the surrogate 
generator will consume:

merge_person_mentions  (part 1 clustering)
apply_coref            (part 2 clustering, ML)   -- optional
extract_kinship        (RELATED_TO edges)
infer_person_attributes
build_location_edges   (LOCATED_IN edges, gazetteer)
resolve_date_entity / resolve_age_entity
age_date_constraints   (STATED_WITH edges)

Non-person mentions (LOCATION / DATE_* / AGE) are grouped into entities here,
since the per-mention modules only clustered PERSON/NICKNAME.
"""


from __future__ import annotations
import os
import re
from dateutil import parser as dateparser
from .models import Entity, Edge, Relation
from .rules.name_matching import merge_person_mentions, normalize
from .rules.aliases import apply_alias_cues
from .rules.coref import apply_coref
from .rules.kinship import extract_kinship
from .rules.attributes import (infer_person_attributes, infer_interviewee_gender,
                         infer_person_role, infer_ethnicity)
from .rules.identifiers import build_identifier_entities
from .rules.locations import (load_gazetteer, build_location_edges,
                              infer_location_replace)
from .rules.dates import resolve_date_entity
from .rules.ages import resolve_age_entity, age_date_constraints
from .second_line import resolve_all, blocking_fields
from .rules.interviewee import resolve_interviewee_identity
from .text.turns import mask_to_subject

_DATE_CATS = ("DATE_ABSOLUTE", "DATE_RELATIVE", "DATE_ANCHOR", "DATE_OF_BIRTH")

# Personal-PII categories we attribute to a specific person. Events
# (DATE_ABSOLUTE/RELATIVE/ANCHOR) are excluded -- they rarely "belong" to the
# speaker in a way that's deterministically recoverable.
# OCCUPATION is included. It used to be excluded, which meant `owner` on an
# occupation had NO rule layer at all -- step 1 of the pattern was simply absent,
# and every occupation ownership decision in both sample transcripts was an
# unchallenged LLM fill. The blockers in `graph/checks/ownership.py` are exactly what
# makes "my father ... as a deckhand" attribute to the father rather than the
# speaker, so the rule can be trusted here too.
_PII_CATS = ("PHONE", "EMAIL", "SSN_OR_ID", "USERNAME_HANDLE", "DATE_OF_BIRTH",
             "AGE", "OCCUPATION")


def _link_interviewee_pii(transcript, entities, interviewee):
    """Deterministically attach the interviewee's OWN identifiers / age / DOB to
    the (otherwise mention-less) interviewee node via ATTRIBUTE_OF edges, so the
    graph can answer 'what PII belongs to the speaker?'. Conservative: ownership is
    claimed only when every `interviewee`-direction predicate holds -- otherwise it
    is left unset (safe).

    THE RULE IS THE CHECKERS. This function used to re-implement the predicates with
    its own regexes, and the copy had drifted from the original in four ways that all
    cost the speaker their own data:

      * its first-person cue set omitted `we` and bare `me`, so "we're easy to find.
        The shop line is 228-555-0143." matched nothing -- while
        `checks/ownership._FP_CUE` matches both;
      * it searched only the span's own sentence, with no one-sentence lookback, so
        the commonest contact-details construction ("I'm not hard to find, never have
        been. Phone's 304-555-0176.") was invisible to it;
      * it inspected `mentions[0]` alone, where the checker requires unanimity across
        every mention;
      * its own comment claimed a third-person-subject blocker that was never
        written, so "She was a schoolteacher" could be claimed for the speaker.

    Calling `graph.checks.ownership` directly makes the drift impossible: the layer
    that proposes and the layer that verifies are now literally the same predicates,
    which is the arrangement `checks/gender.py` and `checks/interviewee.py` already
    use via `support_for` / `_self_described`.
    """
    from .checks import CheckContext
    from .second_line import POLICIES, owner_survivors

    ctx = CheckContext(transcript=transcript, entities=entities, edges=[],
                       interviewee=interviewee)
    policy = POLICIES["owner"]
    edges = []
    for ent in entities:
        if ent.category not in _PII_CATS or not ent.mentions:
            continue
        ctx.entity = ent
        # EXCLUSIVE evidence, not merely sufficient evidence. The two checker
        # families are not mutually exclusive: "The foreman when I started was a man
        # named Bill Ratliff" carries a first-person cue in the sentence AND a named
        # third party in it, so both directions clear. Claiming `interviewee` on the
        # strength of one family alone therefore handed the speaker Bill Ratliff's job
        # title and Ms. Boudreaux's -- the rule was more aggressive than the old regex
        # precisely where the old regex happened to be right. When both directions
        # survive the span is ambiguous and the rule must abstain, which is the same
        # test `second_line._try_alternatives` applies.
        if owner_survivors(policy, ctx) != ["interviewee"]:
            continue
        ent.attributes.setdefault("owner", "interviewee")
        edges.append(Edge(source=ent.entity_id, target=interviewee.entity_id,
                          relation=Relation.ATTRIBUTE_OF,
                          detail=(ent.subtype or ent.category),
                          evidence=ctx.sentence_text(ent.mentions[0].start).strip()))
    return edges


def _ambiguous_merge_claims(transcript, persons, llm):
    """`same_person` MERGE RECORDS for the clustering table's ambiguous short names.

    When the containment rule leaves a bare given name unmerged because it matched
    SEVERAL full names ('short name matches multiple long names'), the LLM is asked
    which full-name entity it belongs to. That answer used to be written straight onto
    the entity here -- `suggested_merge_with` plus a review flag -- with no checkers,
    no Resolution and no ledger row, which made it one of the last LLM decisions
    outside the single arbitration point (and a duplicate of the logic in
    `second_line._resolve_merges`).

    It now RETURNS records instead of writing anything. `_resolve_merges` runs the
    five checkers in `graph/checks/merges.py` over each claim and writes the flag
    itself, so an ungrounded guess is refuted rather than surfaced as if it were
    evidence. Still never merges: identity changes stay a human decision.
    """
    from llm_layer import adjudicate_same_person

    def tokens(e):
        toks = set()
        for f in e.sorted_mentions:
            toks |= set(normalize(f))
        return toks

    claims: list[dict] = []
    multi = [(e, tokens(e)) for e in persons if len(tokens(e)) >= 2]
    for e in persons:
        if "matches multiple long names" not in (e.review_reason or ""):
            continue
        etoks = tokens(e)
        if len(etoks) != 1:
            continue
        token = next(iter(etoks))
        for c, ctoks in multi:
            if c is e or token not in ctoks:
                continue
            v = adjudicate_same_person(llm, transcript, e, c)
            if v and bool(v.get("same")) and (
                    v.get("confidence") == "high" or v.get("evidence")):
                claims.append({"a": e.entity_id, "b": c.entity_id,
                               "evidence": str(v.get("evidence") or ""),
                               "value": True, "source": "llm", "applied": False,
                               "confidence": str(v.get("confidence") or "unstated")})
                break
    return claims


def _merge_claims(records):
    """The merge records minus the Entity object, so `llm_layer` and `graph` still
    exchange plain data across the boundary."""
    return [{k: v for k, v in r.items() if k != "folded"} for r in records]


def _folded(records):
    """`{entity_id: Entity}` for every entity a merge REMOVED from the person list.

    A rule/coref merge folds one entity into another and drops it, so the
    `same_person` checkers -- which need both sides of the pair -- would otherwise
    see `None` for the folded side and report a real merge as unverifiable.
    """
    return {r["b"]: r["folded"] for r in records if r.get("folded") is not None}


"""
One entity per distinct (lowercased) surface form within `categories`.
Used for LOCATION / DATE / AGE, which the clustering modules don't group.

`group_by_text=False` gives one entity per MENTION instead. AGE uses it, because an
age is a fact about ONE moment and one person, not a repeated identifier: grouping
by text made "twelve" in "the water came up twelve feet" and "my daughter Trang was
maybe twelve" a single entity with a single `owner`, and since `checks/ownership`
requires UNANIMITY across an entity's mentions, the two contexts refuted each other
and the age came out unattributable and BLOCKING. The same collapse hits any
transcript where two people are the same age.

Text grouping is kept for LOCATION and DATE, where it is load-bearing: two
mentions of "Biloxi" are one place and must get one surrogate, and two mentions of
"1975" must take one date shift.
"""
def _simple_entities(transcript_id, mentions, categories, prefix,
                     group_by_text=True):
    groups: dict[str, list] = {}
    for i, m in enumerate(mentions):
        if m.entity_type in categories:
            key = m.text.lower() if group_by_text else f"{i}"
            groups.setdefault(key, []).append(m)
    ents = []
    for i, (_, ms) in enumerate(
        sorted(groups.items(), key=lambda kv: min(x.start for x in kv[1])), start=1
    ):
        ents.append(Entity(
            entity_id=f"{transcript_id}_{prefix}{i:03d}",
            category=ms[0].entity_type,
            mentions=sorted(ms, key=lambda x: x.start),
        ))
    return ents


"""
Return (entities, edges, info). `info` carries the interviewee entity,
the coref flag, and any ambiguous person mentions.
    """
def run_pipeline(transcript_id, transcript, mentions, metadata=None,
                 gazetteer_path="data/gazetteer.csv", run_coref=True, trace=False,
                 llm=None):
    metadata = metadata or {}
    interview_date = None
    if metadata.get("interview_date"):
        interview_date = dateparser.parse(metadata["interview_date"]).date()

    # opt-in local LLM: explicit client, or KG_USE_LLM=1 in the environment.
    # off by default.
    if llm is None and os.environ.get("KG_USE_LLM") == "1":
        from llm_layer import default_client
        llm = default_client()

    # people (LLM double-gate on single-candidate containment merges when available).
    # `veto_records` carry the pairs that double-gate kept APART, so the split is
    # arbitrated under `same_person` like every other decision instead of only
    # leaving a review flag behind.
    persons, ambiguous, veto_records = merge_person_mentions(
        transcript_id, mentions, transcript, llm)
    # rule-based alias/nickname merges (closed cue set), independent of coref.
    # The records are kept so `second_line` can arbitrate each merge and give it a
    # ledger row -- clustering used to be the one decision class with no provenance.
    merge_records: list[dict] = list(apply_alias_cues(transcript, persons))
    merge_records += veto_records

    # snapshot the rule-based clustering BEFORE coref, so the trace can show the
    # coref (ML) stage's effect separately
    pre_coref = None
    if trace:
        pre_coref = {p.entity_id: {"forms": list(p.sorted_mentions),
                                   "flag": p.review_reason} for p in persons}

    coref_ran, coref_merges, coref_flags = False, [], []
    if run_coref and persons:
        # LLM use #1 (merge adjudication)
        # passing `llm=llm` here activates it 
        persons, _pairs, coref_ran = apply_coref(transcript, persons, llm=llm)
        merge_records += list(_pairs)
        if trace:
            for rec in _pairs:
                base_id, other_id = rec["a"], rec["b"]
                b = pre_coref.get(base_id, {}).get("forms") or [base_id]
                o = pre_coref.get(other_id, {}).get("forms") or [other_id]
                coref_merges.append({"kept": b[0], "merged": o[0]})
            seen = set()
            for p in persons:                       # collect coref "review" notes
                for line in filter(None, (p.review_reason or "").split("; ")):
                    if "coref" not in line.lower():
                        continue
                    m = re.search(r"(\w+_e\d{3})", line)
                    other = (pre_coref.get(m.group(1), {}).get("forms")
                             if m else None)
                    other_name = other[0] if other else "?"
                    this_name = p.sorted_mentions[0] if p.sorted_mentions else p.entity_id
                    tail = re.sub(r"\w+_e\d{3}", other_name, line)
                    key = frozenset({this_name, other_name}) | {tail}
                    if key in seen:
                        continue
                    seen.add(key)
                    coref_flags.append({"a": this_name, "b": other_name, "note": tail})

    # The interviewee ("I") starts as a synthetic entity with no detected span.
    interviewee = Entity(entity_id=f"{transcript_id}_e000", category="PERSON",
                         attributes={"role": "interviewee", "replace": True})

    # WHICH named person is the speaker? Rules first (speaker label / self-
    # introduction / interviewer address), then the LLM proposes, then the checkers
    # in graph/checks/interviewee.py gate it; only a value that clears them is
    # merged into e000. Runs HERE -- after clustering and coref have settled the
    # person entities, before kinship / attributes / identifiers -- so every later
    # stage sees ONE interviewee instead of a synthetic node plus a named twin.
    llm_up = llm is not None and llm.available()
    identity_res = resolve_interviewee_identity(transcript, persons, interviewee,
                                                llm=llm, llm_ran=llm_up)

    # rule layer for the interviewee's own gender (first-person self-description,
    # subject turns only); the LLM second line in extract_pass fills/confirms it
    infer_interviewee_gender(transcript, interviewee)

    edges = list(extract_kinship(transcript, persons, interviewee))
    infer_person_attributes(transcript, persons, edges)
    # rule layers for `role` and `ethnicity`. Both fields used to be LLM-only and
    # unchecked; the rules for them are the kinship-edge detail / professional cue,
    # and a closed set of self-identification constructions. See attributes.py.
    infer_person_role(transcript, persons, edges)
    infer_ethnicity(transcript, persons, interviewee)

    # locations
    records, aliases = load_gazetteer(gazetteer_path)
    locations = _simple_entities(transcript_id, mentions,
                                 ("LOCATION", "INSTITUTION"), "L")
    edges += build_location_edges(locations, records, aliases)
    # rule layer for LOCATION/INSTITUTION `replace`: gazetteer granularity. Runs
    # AFTER build_location_edges, which is what assigns the gazetteer type.
    infer_location_replace(locations)

    # dates / ages
    dates = _simple_entities(transcript_id, mentions, _DATE_CATS, "D")
    for d in dates:
        resolve_date_entity(d, interview_date)
    # one entity per AGE MENTION -- see `_simple_entities`
    ages = _simple_entities(transcript_id, mentions, ("AGE",), "A",
                            group_by_text=False)
    for a in ages:
        resolve_age_entity(a)

    # direct identifiers (PHONE/EMAIL/SSN_OR_ID/USERNAME_HANDLE/OCCUPATION): typed
    # + normalized by rule, passed through so they reach surrogate generation
    idents = build_identifier_entities(transcript_id, mentions)

    entities = [interviewee] + persons + locations + dates + ages + idents
    edges += age_date_constraints(transcript, entities)
    # attach the interviewee's own identifiers / age / DOB to the e000 node
    edges += _link_interviewee_pii(transcript, entities, interviewee)

    # ------------------------------------------------------------------ LLM
    # The LLM layer PROPOSES; `graph.second_line` ARBITRATES. Nothing below
    # writes a field decision directly except the two out-of-scope classes noted
    # in graph/second_line/ (LLM-only fields with no rule to check against, and
    # structural identity suggestions).
    llm_ran = False
    ledger: dict = {}
    if llm_up:
        from llm_layer import (openworld_propose, extract_pass,
                               identifier_judge_pass)
        llm_ran = True

        proposals: dict = {}

        def _merge(more):
            for eid, fields in (more or {}).items():
                proposals.setdefault(eid, {}).update(fields)

        # per-entity classifiers: public figure / subtype / location / dates / ages
        _merge(openworld_propose(transcript, entities, llm,
                                 interview_date=interview_date))
        # windowed read-along: gender + name parts (proposed); role / ethnicity /
        # aliases written in place (out of scope); relations already verified.
        # `subject_mask` lets the pass tell the speaker's words from the
        # interviewer's without importing anything from `graph`.
        attr_proposals, llm_rels, llm_merges = extract_pass(
            transcript, entities, interviewee, llm,
            subject_mask=mask_to_subject(transcript))
        _merge(attr_proposals)
        merge_records += list(llm_merges)
        # windowed identifier judgment: owner + kind (proposed); identifying in place
        _merge(identifier_judge_pass(transcript, entities, llm))

        # ambiguous short names: the LLM's answer joins the `same_person` records so
        # `resolve_all` checks it, instead of being written onto the entity here
        merge_records += _ambiguous_merge_claims(transcript, persons, llm)

        # Relations are NOT added here any more. They are a field like any other:
        # `resolve_all` arbitrates each pair (rule detail vs LLM proposal) with the
        # verifier in graph/checks/relation_evidence.py as the checker, and returns the
        # surviving edges. That is what gives relations a Resolution, a provenance
        # record and a row in the ledger.
        ledger = resolve_all(transcript, entities, edges, interviewee, proposals,
                             interview_date=interview_date,
                             gazetteer=records, gaz_aliases=aliases,
                             relation_proposals=llm_rels, llm_ran=True,
                             identity_resolution=identity_res,
                             merge_records=_merge_claims(merge_records),
                             folded_entities=_folded(merge_records))
        edges += ledger.pop("_edges", [])
    else:
        # rules-only: still run the arbitration so every field gets a provenance
        # record (every outcome will be `keep` or `reject`), and so the ledger shape
        # is identical whether or not the LLM was available.
        ledger = resolve_all(transcript, entities, edges, interviewee, {},
                             interview_date=interview_date,
                             gazetteer=records, gaz_aliases=aliases,
                             llm_ran=False, identity_resolution=identity_res,
                             merge_records=_merge_claims(merge_records),
                             folded_entities=_folded(merge_records))
        ledger.pop("_edges", None)

    info = {
        "interviewee": interviewee,
        "coref_ran": coref_ran,
        "ambiguous": ambiguous,
        "identity": identity_res,
        # carried out so the artifact writer needs nothing but `info`: the
        # date-shifter validates every shift against this, and it used to be
        # readable only by the caller that supplied the metadata.
        "interview_date": interview_date,
        "llm_ran": llm_ran,
        "llm_model": getattr(llm, "model", None) if llm_ran else None,
        "ledger": ledger,
        "blocking": blocking_fields(ledger),
    }
    if trace:
        info["pre_coref"] = list((pre_coref or {}).values())
        info["coref_merges"] = coref_merges
        info["coref_flags"] = coref_flags
    return entities, edges, info