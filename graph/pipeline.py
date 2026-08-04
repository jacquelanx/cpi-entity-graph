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
from .merge_strings import merge_person_mentions
from .aliases import apply_alias_cues
from .coref import apply_coref
from .kinship import extract_kinship, KINSHIP_GENDER
from .attributes import infer_person_attributes
from .identifiers import build_identifier_entities
from .location_dates import (
    load_gazetteer, build_location_edges,
    resolve_date_entity, resolve_age_entity, age_date_constraints,
)

_DATE_CATS = ("DATE_ABSOLUTE", "DATE_RELATIVE", "DATE_ANCHOR", "DATE_OF_BIRTH")

# Personal-PII categories we attribute to a specific person. Events
# (DATE_ABSOLUTE/RELATIVE/ANCHOR) and OCCUPATION are excluded -- they rarely
# "belong" to the speaker in a way that's deterministically recoverable.
_PII_CATS = ("PHONE", "EMAIL", "SSN_OR_ID", "USERNAME_HANDLE", "DATE_OF_BIRTH", "AGE")

# First-person cue that governs a PII span: a possessive ("my cell", "our file
# number"), a first-person subject ("I was nineteen"), or a reach-me object
# ("email me at ...", "call me").
_FP_CUE = re.compile(
    r"\b(?:my|our)\b|\bI\b|\b(?:reach|call|email|text|message)\s+me\b|\bme\s+at\b",
    re.I)

# A kin noun between the cue and the PII means it belongs to THAT relative, not
# the speaker ("my daughter runs a page, @handle" -> the daughter's).
_KIN_NOUN = re.compile(
    r"\b(?:" + "|".join(re.escape(w) for w in
                        sorted(KINSHIP_GENDER, key=len, reverse=True)) + r")\b",
    re.I)


def _link_interviewee_pii(transcript, entities, interviewee, persons):
    """Deterministically attach the interviewee's OWN identifiers / age / DOB to
    the (otherwise mention-less) interviewee node via ATTRIBUTE_OF edges, so the
    graph can answer 'what PII belongs to the speaker?'. Conservative: an owner is
    claimed only on a clear first-person cue with no intervening relative or other
    named person -- otherwise ownership is left unset (safe)."""
    stops = [0] + [m.end() for m in re.finditer(r"[.!?]", transcript)]
    sents = list(zip(stops, stops[1:] + [len(transcript)]))

    def sentence_of(pos):
        for s, e in sents:
            if s <= pos < e:
                return s, e
        return (sents[-1] if sents else (0, len(transcript)))

    edges = []
    for ent in entities:
        if ent.category not in _PII_CATS or not ent.mentions:
            continue
        m = ent.mentions[0]
        ss, se = sentence_of(m.start)
        cues = list(_FP_CUE.finditer(transcript[ss:m.start]))
        if not cues:
            continue
        cue_end = ss + cues[-1].end()               # closest cue to the PII
        gap = transcript[cue_end:m.start]
        # blocker: a kin noun, or another named person, sits between cue and PII
        if _KIN_NOUN.search(gap):
            continue
        if any(cue_end <= pm.start < m.start for p in persons for pm in p.mentions):
            continue
        ent.attributes.setdefault("owner", "interviewee")
        edges.append(Edge(source=ent.entity_id, target=interviewee.entity_id,
                          relation=Relation.ATTRIBUTE_OF,
                          detail=(ent.subtype or ent.category),
                          evidence=transcript[ss:se].strip()))
    return edges


"""
One entity per distinct (lowercased) surface form within `categories`.
Used for LOCATION / DATE / AGE, which the clustering modules don't group.
"""
def _simple_entities(transcript_id, mentions, categories, prefix):
    groups: dict[str, list] = {}
    for m in mentions:
        if m.entity_type in categories:
            groups.setdefault(m.text.lower(), []).append(m)
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

    # people
    persons, ambiguous = merge_person_mentions(transcript_id, mentions)
    # rule-based alias/nickname merges (closed cue set), independent of coref
    apply_alias_cues(transcript, persons)

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
        if trace:
            for base_id, other_id in _pairs:
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

    # the interviewee ("I") is a synthetic entity: no detected span, but kinship
    # edges hang off it
    interviewee = Entity(entity_id=f"{transcript_id}_e000", category="PERSON",
                         attributes={"role": "interviewee", "replace": True})

    edges = list(extract_kinship(transcript, persons, interviewee))
    infer_person_attributes(transcript, persons, edges)

    # locations
    records, aliases = load_gazetteer(gazetteer_path)
    locations = _simple_entities(transcript_id, mentions,
                                 ("LOCATION", "INSTITUTION"), "L")
    edges += build_location_edges(locations, records, aliases)

    # dates / ages
    dates = _simple_entities(transcript_id, mentions, _DATE_CATS, "D")
    for d in dates:
        resolve_date_entity(d, interview_date)
    ages = _simple_entities(transcript_id, mentions, ("AGE",), "A")
    for a in ages:
        resolve_age_entity(a)

    # direct identifiers (PHONE/EMAIL/SSN_OR_ID/USERNAME_HANDLE/OCCUPATION): typed
    # + normalized by rule, passed through so they reach surrogate generation
    idents = build_identifier_entities(transcript_id, mentions)

    entities = [interviewee] + persons + locations + dates + ages + idents
    edges += age_date_constraints(transcript, entities)
    # attach the interviewee's own identifiers / age / DOB to the e000 node
    edges += _link_interviewee_pii(transcript, entities, interviewee, persons)

    # LLM uses #2, #3, #4
    if llm is not None:
        from llm_layer import openworld_pass, extract_pass, identifier_judge_pass
        openworld_pass(transcript, entities, llm)        # #2: fill list misses
        # #3: windowed read-along -> attributes (in place), aliases (flagged in
        # place), and relations returned as tuples. Relations are added as edges
        # ONLY if the rules didn't already have that pair (rules stay authoritative).
        llm_rels = extract_pass(transcript, entities, interviewee, llm)
        have = {(e.source, e.target) for e in edges if e.relation == Relation.RELATED_TO}
        for r in llm_rels:
            if (r["source"], r["target"]) not in have:
                edges.append(Edge(source=r["source"], target=r["target"],
                                  relation=Relation.RELATED_TO,
                                  detail=r["detail"], evidence=f"(llm) {r['evidence']}"))
                have.add((r["source"], r["target"]))
        # #4: contextual judgment on direct identifiers (owner / occupation
        # identifying-ness) -- suggestions only, never lowers redaction
        identifier_judge_pass(transcript, entities, llm)

    info = {
        "interviewee": interviewee,
        "coref_ran": coref_ran,
        "ambiguous": ambiguous,
    }
    if trace:
        info["pre_coref"] = list((pre_coref or {}).values())
        info["coref_merges"] = coref_merges
        info["coref_flags"] = coref_flags
    return entities, edges, info