"""
The per-transcript walk: `resolve_all`.

Decides which fields apply to each entity (`_fields_for`), reads what the rule
layer already put there (`_rule_value`), drives `second_line` over every field,
every proposed relation and every proposed merge, applies the results, and
reports what still blocks surrogate generation (`blocking_fields`).
"""

from __future__ import annotations

import dataclasses
from ..models import Edge, Relation
from ..checks import CheckContext
from ..checks import comparators as C
from ..checks.relation_words import KIN_WORDS
from .apply import _CTX, _drop_edges, apply_resolution
from .engine import second_line
from .outcomes import CONFIRM, CONFLICT, FILL, FieldPolicy, REJECT, REQUIRED_VERIFIED
from .policies import POLICIES, _DATE_TOL


# ------------------------------------------------------------------- the driver

_PERSON_FIELDS = ("gender", "given_name", "surname", "replace", "subtype_person",
                  "role", "ethnicity")


_DATE_CATS = ("DATE_ABSOLUTE", "DATE_RELATIVE", "DATE_ANCHOR", "DATE_OF_BIRTH")


_ID_CATS = ("PHONE", "EMAIL", "SSN_OR_ID", "USERNAME_HANDLE", "OCCUPATION")


_OWNED_CATS = _ID_CATS + ("AGE", "DATE_OF_BIRTH")


# Categories whose OWNER feeds the interviewee's surrogate identity directly. On
# these, an owner neither layer could establish is a BLOCKING gap rather than a
# silent None -- a phone number nobody can attribute is exactly the case a human
# must settle before surrogates are minted. OCCUPATION is deliberately left at
# REQUIRED_OR_ABSTAIN: transcripts mention many jobs belonging to many people, and
# blocking on each would bury the rows that matter.
_OWNER_VERIFIED_CATS = ("PHONE", "EMAIL", "SSN_OR_ID", "USERNAME_HANDLE",
                        "DATE_OF_BIRTH", "AGE")


def _rule_value(ent, policy: FieldPolicy, ctx):
    attr = policy.attr or policy.field
    if policy.field == "replace_location":
        # Recomputed from the CURRENT subtype rather than read from the attribute.
        # `rules/locations.infer_location_replace` runs in the pipeline, before the
        # second line has typed the off-gazetteer places, so at that point every
        # unknown place defaults to replace=True. Reading that stale default here
        # made the rule layer contradict its own threshold: "Mekong Delta" resolved
        # to REGION (a keepable type) and the rule still said replace, purely
        # because the gazetteer had not heard of it. `subtype_location` is resolved
        # BEFORE this field (see `_fields_for`), so by now the type is the verified
        # one, and the rule is a pure function of it.
        from ..rules.locations import BROAD_LOCATION_TYPES
        if ent.category == "INSTITUTION":
            return True
        raw = str(ent.subtype or "").strip().lower()
        if raw:
            return raw not in BROAD_LOCATION_TYPES
        return ent.attributes.get("replace", True)
    if policy.field == "replace_date":
        # A pure function of the RESOLVED `shiftable` -- resolved first, see
        # `_fields_for`. A date the shifter will move cannot survive as written; one
        # it cannot move may stay, subject to the keep checkers. Defaults to shiftable
        # (so, replace) if the attribute is somehow missing, which is the safe error.
        return bool(ent.attributes.get("shiftable", True))
    if policy.field == "replace_age":
        # Keep ONLY a span a deterministic check proved is not an age. An age nobody
        # could parse is still an age, so it is replaced.
        from ..checks.ages import age_reading_refuted
        return not bool(age_reading_refuted(ent))
    if attr == "subtype":
        return ent.subtype
    if attr == "category":
        # `kind`. The rule's answer is the detector's category -- but ONLY when the
        # rule's own normalizer accepts the span under it. When `_normalize` raised a
        # flag ("email failed to parse", "phone has too few digits", "SSN_OR_ID has no
        # digits"), the rules have no trustworthy answer and must say so, which is
        # what opens the FILL path so the LLM can re-type the span and
        # `checks/identifiers.kind_renormalizes` can gate the guess.
        #
        # Reading `ent.category` unconditionally made `kind` the one field whose FILL
        # branch was unreachable: the value was never empty, so a malformed span could
        # only ever be flagged, never corrected, and `_legacy_mirror`'s
        # `suggested_kind` was dead code. The entity's CATEGORY is still never
        # rewritten (see `apply_resolution`) -- what a checked fill supplies is
        # `attributes["kind"]` and the derived sub-attributes the surrogate generator
        # actually mints from.
        from ..checks.identifiers import _NORMALIZABLE
        if ent.category in _NORMALIZABLE and ent.mentions:
            from ..rules.identifiers import _normalize
            _sub, _attrs, flag = _normalize(ent.category, ent.mentions[0].text)
            if flag:
                return None
        return ent.category
    if attr == "stated_with":
        # the rule value lives on the STATED_WITH edge age_date_constraints built
        for ed in ctx.edges:
            if ed.relation == Relation.STATED_WITH and ed.source == ent.entity_id:
                tgt = next((e for e in ctx.entities if e.entity_id == ed.target), None)
                if tgt is not None and tgt.mentions:
                    return tgt.mentions[0].text
        return None
    return ent.attributes.get(attr)


def _fields_for(ent, interviewee) -> list[str]:
    """Which policies apply to this entity.

    ORDER MATTERS within a category: a later field's checkers may read the value an
    earlier one settled. `replace_location` must come after `subtype_location`
    because the keep gate is a function of the resolved geographic type;
    `replace_date` after `shiftable` and `replace_age` after `value` for exactly the
    same reason (each of those rules is a pure function of the field before it); and
    `ethnicity` comes last on a person so its checkers see final mentions.
    """
    if ent is interviewee:
        # `interviewee_identity` is resolved EARLY (graph/rules/interviewee.py) because the
        # merge has to happen before kinship and attributes run; the driver records
        # its Resolution rather than re-resolving it here. Name parts only exist as
        # a field once identification actually found a name.
        out = ["interviewee_gender"]
        if getattr(ent, "mentions", None):
            out += ["given_name", "surname"]
        # the speaker's OWN ethnicity is second-lined like anyone else's -- it feeds
        # surrogate name selection, so an unchecked guess is not acceptable here
        out.append("ethnicity")
        return out
    # NICKNAME cannot reach here today (merge_person_mentions relabels every
    # PERSON/NICKNAME mention "PERSON"), but treating it as a person keeps this
    # defensive rather than silently returning no fields if that ever changes.
    if ent.category in ("PERSON", "NICKNAME"):
        return list(_PERSON_FIELDS)
    if ent.category in ("LOCATION", "INSTITUTION"):
        return ["subtype_location", "location_parent", "replace_location"]
    if ent.category in _DATE_CATS:
        # `shiftable` is set by the rule for EVERY date category
        # (rules/dates.resolve_date_entity), so every category needs the second
        # line. Previously only DATE_ANCHOR was arbitrated, which left the rule's
        # `shiftable=True` on absolute, relative and DOB dates double-checked by
        # nothing at all.
        out = ["resolved_value", "approximate", "shiftable", "replace_date"]
        if ent.category == "DATE_OF_BIRTH":
            out += ["kind", "owner"]
        return out
    if ent.category == "AGE":
        return ["value", "approximate", "kind", "owner", "stated_with", "replace_age"]
    if ent.category in _ID_CATS:
        out = ["kind", "owner"]
        if ent.category == "OCCUPATION":
            out.append("identifying")
        return out
    return []


def _resolve_relations(ctx, edges, ledger, relation_proposals, llm_ran):
    """Arbitrate RELATED_TO pairs, then let the deterministic kin rule run again.

    Resolved per PAIR: the rule value is the detail of the kinship edge the rules
    already produced for that pair, the LLM value is the raw proposal, and the
    checker is the verifier in `graph/checks/relation_evidence.py`. Runs BEFORE the
    per-entity field loop so that a relation the second line adds is visible to the
    kin-derived FAMILY subtype and to every checker that reads `kin_ids`.
    """
    iv = ctx.interviewee
    rule_rel = {(e.source, e.target): e for e in edges
                if e.relation == Relation.RELATED_TO}
    llm_rel = {}
    for r in (relation_proposals or []):
        llm_rel[(r["source"], r["target"])] = r

    policy = POLICIES["relation"]
    new_edges = []
    for pair in list(rule_rel) + [p for p in llm_rel if p not in rule_rel]:
        src, tgt = pair
        prop = llm_rel.get(pair)
        rule_edge = rule_rel.get(pair)
        ctx.pair = (src, tgt, (prop or {}).get("evidence", ""))
        ctx.relation_verdict = None
        ctx.entity = ctx.ent_by_id.get(tgt) or ctx.ent_by_id.get(src)
        res = second_line(policy,
                          rule_edge.detail if rule_edge else None,
                          {"value": prop["detail"], "confidence": prop["confidence"]}
                          if prop else None,
                          ctx, llm_ran=llm_ran)

        if res.action == FILL:
            v = ctx.relation_verdict
            new_edges.append(Edge(source=v.source, target=v.target,
                                  relation=Relation.RELATED_TO, detail=v.detail,
                                  evidence=f"(llm) {v.evidence}"))
        elif res.action == REJECT and res.checks_failed:
            # a proposal that was merely UNPROVABLE (not refuted) still reaches a
            # human, tagged on the named person -- same policy as before the move.
            v = ctx.relation_verdict
            if v is not None and v.action == "suggest":
                named = ctx.ent_by_id.get(v.target)
                other = ctx.ent_by_id.get(v.source)
                if named is iv:
                    named, other = other, named
                if named is not None and named is not iv:
                    with_nm = ("the interviewee" if other is iv
                               else (other.sorted_mentions[0]
                                     if other and other.sorted_mentions else "someone"))
                    named.attributes.setdefault(
                        "suggested_relation",
                        {"detail": v.detail, "with": with_nm,
                         "evidence": v.evidence[:120]})
                    named.flag_entity(
                        f"LLM suggests relation '{v.detail}' with {with_nm} but it "
                        f"couldn't be verified locally; review")

        key = f"relation:{src}" if ctx.entity and ctx.entity.entity_id == tgt \
            else f"relation:{tgt}"
        if ctx.entity is not None:
            if not hasattr(ctx.entity, "provenance"):
                ctx.entity.provenance = {}
            ctx.entity.provenance[key] = res
            ledger.setdefault(ctx.entity.entity_id, {})[key] = res

    ctx.pair, ctx.relation_verdict, ctx.entity = None, None, None

    # deterministic rule, re-applied now that relations are final: a person tied to
    # someone by a KIN relation is FAMILY unless the rules already typed them.
    for ed in edges + new_edges:
        if ed.relation != Relation.RELATED_TO:
            continue
        if C.kin_canon(ed.detail) not in KIN_WORDS and ed.detail.lower() not in KIN_WORDS:
            continue
        for eid in (ed.source, ed.target):
            e = ctx.ent_by_id.get(eid)
            if e is not None and e is not ctx.interviewee and e.subtype is None:
                e.subtype = "FAMILY"
    return new_edges


def _resolve_merges(ctx, merge_records, ledger, llm_ran) -> None:
    """Arbitrate same-person (alias / nickname / coref) claims.

    `merge_records` is a list of plain dicts, one per CLAIMED pair:

        {"a": kept_entity_id, "b": other_entity_id, "evidence": str,
         "source": "rule" | "llm", "confidence": str,
         "applied": bool,      # did this claim change the entity set?
         "value": bool}        # an LLM record's verdict; defaults to True

    A record with `source="rule"` is a claim the rule layer made -- one it CARRIED OUT
    (`applied=True`, alias and coref merges) or one it wanted and the LLM vetoed
    (`applied=False`, the containment veto in `name_matching`). A record with
    `source="llm"` is the model's own answer: `value=True` for an alias/containment
    proposal, `value=False` for a veto. Both sources for one pair means the field was
    filled by the rules and DOUBLE-CHECKED by the model, which is the ordinary
    two-layer shape -- coref's LLM double-gate emits exactly that pair of records, so
    an LLM-confirmed coref merge is no longer recorded as "rules stand; no LLM answer
    to confirm them".

    NOTHING here merges. A checked LLM proposal becomes `suggested_merge_with` plus
    a review flag -- the same policy as before -- so identity changes remain a
    human decision. What changes is that the decision is now recorded, the
    checkers name themselves, and a reviewer can tell a grounded nickname claim
    from an ungrounded one.
    """
    policy = POLICIES["same_person"]
    by_pair: dict = {}
    for r in (merge_records or []):
        key = (r["a"], r["b"])
        by_pair.setdefault(key, {})[r.get("source", "llm")] = r

    for (a_id, b_id), rec in by_pair.items():
        rule_rec = rec.get("rule")
        llm_rec = rec.get("llm")
        ev = (llm_rec or rule_rec or {}).get("evidence", "") or ""
        ctx.pair = (a_id, b_id, ev)
        ctx.entity = ctx.ent_by_id.get(a_id)
        # `applied` distinguishes a merge the rules CARRIED OUT from a containment
        # merge they wanted and the LLM vetoed. Both are rule claims of "same
        # person"; only the first changed the entity set.
        rule_applied = bool(rule_rec) and rule_rec.get("applied", True)
        # An LLM record carries its own verdict. `extract_pass` aliases always claim
        # True; the containment veto in `name_matching` claims False, which is the
        # model DOUBLE-CHECKING a filled rule value and disagreeing -- an ordinary
        # conflict, resolved by `same_person`'s safe direction (do not merge).
        llm_value = None if llm_rec is None else llm_rec.get("value", True)
        res = second_line(
            policy,
            True if rule_rec else None,
            {"value": llm_value, "confidence": llm_rec.get("confidence", "unstated")}
            if llm_rec is not None else None,
            ctx, llm_ran=llm_ran)

        a = ctx.ent_by_id.get(a_id)
        b = ctx.ent_by_id.get(b_id)
        a_nm = (a.sorted_mentions[0] if a is not None and a.sorted_mentions else a_id)
        b_nm = (b.sorted_mentions[0] if b is not None and b.sorted_mentions else b_id)

        # An APPLIED merge the checkers refuted. Nothing here can un-merge -- the fold
        # happened in `aliases` / `coref` before this ran -- so the only honest thing
        # is to say so loudly. This case was unreachable before: an applied merge
        # always resolved `confirm`, and `confirm` ran no checkers.
        if rule_applied and res.value is None and res.checks_failed:
            (a if a is not None else b or ctx.interviewee).flag_entity(
                f"same_person: {a_nm} and {b_nm} were MERGED by the rules, but a "
                f"deterministic check refutes it ({', '.join(res.checks_failed)}); "
                f"the merge is already applied -- review and split if wrong")

        # A proposal the rules did not make, that cleared every applicable checker:
        # FLAG it, never merge it.
        if res.action == FILL and res.value is True and a is not None and b is not None:
            a.attributes["suggested_merge_with"] = b_nm
            b.attributes["suggested_merge_with"] = a_nm
            detail = (f"{len(res.checks_passed)} deterministic check(s) verified it"
                      if res.checks_passed else
                      "NO deterministic check applied to this claim")
            a.flag_entity(f"same_person: LLM suggests this is {b_nm}; {detail} "
                          f"(confidence {res.confidence}); review to merge")
            b.flag_entity(f"same_person: LLM suggests this is {a_nm}; {detail} "
                          f"(confidence {res.confidence}); review to merge")
        elif res.action == REJECT and res.checks_failed and a is not None:
            a.flag_entity(f"same_person: an LLM merge suggestion for {b_nm} was "
                          f"refuted ({', '.join(res.checks_failed)}); kept separate")
        elif res.action == CONFLICT and llm_value is False and not rule_applied \
                and a is not None:
            # A veto: the rules/coref proposed the merge and the LLM refused it. Now
            # recorded as a resolution rather than only as a free-text flag written
            # inside the clustering rule (containment) or nowhere at all (coref).
            a.flag_entity(f"same_person: the rules proposed merging this into "
                          f"{b_nm}; the LLM judged them different people and the "
                          f"safe direction kept them separate; review")

        for ent, other in ((a, b_id), (b, a_id)):
            if ent is None:
                continue
            if not hasattr(ent, "provenance"):
                ent.provenance = {}
            ent.provenance[f"same_person:{other}"] = res
            ledger.setdefault(ent.entity_id, {})[f"same_person:{other}"] = res

    ctx.pair, ctx.entity = None, None


def resolve_all(transcript, entities, edges, interviewee, proposals, *,
                interview_date=None, gazetteer=None, gaz_aliases=None,
                relation_proposals=None, llm_ran=True,
                identity_resolution=None, merge_records=None,
                folded_entities=None) -> dict:
    """Arbitrate every in-scope field on every entity.

    `proposals` is `{entity_id: {field: {"value":..., "confidence":...}}}` --
    plain dicts, so `llm_layer` needs no import from `graph`.

    Returns a ledger: `{entity_id: {field: Resolution}}`. New edges that checked
    proposals earned -- RELATED_TO, LOCATED_IN, ATTRIBUTE_OF, STATED_WITH -- come
    back under the `_edges` key.
    """
    ctx = CheckContext(transcript=transcript, entities=entities, edges=edges,
                       interviewee=interviewee, interview_date=interview_date,
                       gazetteer=gazetteer or {}, gaz_aliases=gaz_aliases or {},
                       extra_entities=dict(folded_entities or {}))
    _CTX[0] = ctx

    ledger: dict = {}
    new_edges: list[Edge] = []

    # `interviewee_identity` was resolved before kinship ran (the merge has to
    # precede every stage that reads the entity set), so it is threaded in here to
    # get its ledger row and its blocking status like any other field.
    if identity_resolution is not None:
        ledger.setdefault(interviewee.entity_id, {})["interviewee_identity"] = \
            identity_resolution

    # RELATIONS first: they feed `kin_ids` and the kin-derived FAMILY subtype, both
    # of which later checkers read.
    new_edges += _resolve_relations(ctx, edges, ledger, relation_proposals, llm_ran)
    ctx.edges = edges + new_edges
    ctx.kin_ids = ({e.source for e in ctx.edges if e.relation == Relation.RELATED_TO}
                   | {e.target for e in ctx.edges if e.relation == Relation.RELATED_TO})

    # Same-person claims, recorded and checked. Runs after relations (so a gender
    # inferred from kinship is available to `genders_do_not_conflict`) and before
    # the per-entity loop (so a merge flag is on the entity before its own fields
    # are resolved).
    _resolve_merges(ctx, merge_records, ledger, llm_ran)

    for ent in entities:
        ctx.entity = ent
        per_ent = proposals.get(ent.entity_id, {})
        # A `resolved_value` proposal for an anchor carries the event NAME as an
        # extra. `checks/dates.is_real_public_event` needs it to tell a corroborated
        # public event from any date the rules happened to parse, so surface it
        # before `shiftable` is resolved. Nothing wrote this key before, which is
        # why that checker's fallback was effectively "has a resolved_value".
        rv = per_ent.get("resolved_value") or {}
        if rv.get("event"):
            ent.attributes["suggested_event"] = rv["event"]
        # Same treatment for the ethnicity proposal's evidence quote: it is
        # provenance for a human, not part of the decision, so it is surfaced rather
        # than arbitrated. `ethnicity_basis` is written by the mirror once the
        # CHECKERS have spoken, so it now records verification rather than the
        # model's own claim about where it read the label.
        eth = per_ent.get("ethnicity") or {}
        if eth.get("evidence"):
            ent.attributes.setdefault("ethnicity_evidence", eth["evidence"])
        for fname in _fields_for(ent, interviewee):
            # An AGE span the checkers proved is NOT an age has no owner and no
            # temporal anchor, so those questions are not asked of it. `value` is
            # resolved first (see `_fields_for`), so by now we know.
            #
            # Without this, "the water came up twelve feet" -- an AGE only because the
            # detector tagged every "twelve" -- had its `value` correctly refused by
            # `checks/ages.not_a_measurement` and then went on to acquire
            # `owner="interviewee"`, putting a measurement into the speaker's own
            # identity. Skipping beats adding an ownership checker for it: a refutation
            # there would leave the field empty on a REQUIRED_VERIFIED category and
            # raise a BLOCKING row that no human can resolve, for a span that should
            # never have been an age.
            if ent.category == "AGE" and fname in ("owner", "stated_with") \
                    and ent.attributes.get("value") is None:
                if fname == "owner":
                    # `pipeline._link_interviewee_pii` writes `owner` onto AGE
                    # entities BEFORE arbitration, so skipping the field is not
                    # enough -- the rule's claim has to be erased, the same
                    # "abstention must erase" rule `apply_resolution` applies.
                    ent.attributes.pop("owner", None)
                    ent.flag_entity(
                        "no verified age value, so this span is not treated as an "
                        "age: ownership and the age<->date pairing were not resolved "
                        "for it; check the detector's label")
                    _drop_edges(edges, new_edges, Relation.ATTRIBUTE_OF,
                                ent.entity_id)
                else:
                    _drop_edges(edges, new_edges, Relation.STATED_WITH,
                                ent.entity_id)
                continue
            policy = POLICIES[fname]
            # per-category date tolerance
            if fname == "resolved_value":
                policy = dataclasses.replace(
                    policy, comparator=C.date_close(_DATE_TOL.get(ent.category, 31)))
            # ownership of the speaker's own quasi-identifiers must be VERIFIED, not
            # merely attempted: promote the tier so an unresolved owner blocks.
            #
            # ...but only for a span the SUBJECT actually uttered. An interviewer echo
            # ("You said your father started at fourteen") is not the speaker's data
            # and carries no ownership evidence either way, so demanding a verified
            # owner for it manufactures a blocking row a human cannot resolve. This
            # matters now that AGE entities are per-mention (see
            # `pipeline._simple_entities`): the echo used to ride along in the same
            # entity as the subject's own span, and `checks/ownership._claim` skipped
            # it; on its own it would fail both directions and block.
            if fname == "owner" and ent.category in _OWNER_VERIFIED_CATS and \
                    any(ctx.in_subject_turn(m.start) for m in ent.mentions):
                policy = dataclasses.replace(policy, tier=REQUIRED_VERIFIED)
            res = second_line(policy, _rule_value(ent, policy, ctx),
                              per_ent.get(fname), ctx, llm_ran=llm_ran)
            apply_resolution(ent, res, policy)
            ledger.setdefault(ent.entity_id, {})[fname] = res

            # `subtype` and `replace` must never disagree. When redaction is raised
            # on a name the rules had typed PUBLIC_FIGURE, downgrade the subtype --
            # otherwise a consumer keying off `subtype` keeps a name that a consumer
            # keying off `replace` redacts (the verified Kennedy / Reagan bug).
            if fname == "replace" and res.value is True and ent.subtype == "PUBLIC_FIGURE":
                ent.subtype = "PUBLIC_FIGURE_UNCONFIRMED"

            # An owner the second line could NOT establish must not leave the rule's
            # ATTRIBUTE_OF edge behind. `pipeline._link_interviewee_pii` writes that
            # edge before arbitration, so a claim the checkers refute was still
            # reachable by any consumer walking edges -- the edge half of the same
            # "abstention must erase" rule `apply_resolution` applies to attributes.
            if fname == "owner" and res.value is None:
                _drop_edges(edges, new_edges, Relation.ATTRIBUTE_OF, ent.entity_id)

            # A CHECKED owner becomes a real ATTRIBUTE_OF edge. The rule path
            # (pipeline._link_interviewee_pii) runs BEFORE this, so without these
            # lines an owner the second line filled set `attributes["owner"]` and
            # nothing else -- leaving two classes of ownership with no marker, and
            # the speaker's own phone/email invisible to any consumer walking edges.
            if fname == "owner" and res.value == "interviewee" and \
                    res.action in (FILL, CONFIRM, CONFLICT):
                if not any(e.relation == Relation.ATTRIBUTE_OF
                           and e.source == ent.entity_id
                           and e.target == interviewee.entity_id
                           for e in edges + new_edges):
                    m = ent.mentions[0] if ent.mentions else None
                    new_edges.append(Edge(
                        source=ent.entity_id, target=interviewee.entity_id,
                        relation=Relation.ATTRIBUTE_OF,
                        detail=(ent.subtype or ent.category),
                        evidence=(f"(second line: {res.source}, "
                                  f"{len(res.checks_passed)} check(s) passed) "
                                  + (ctx.sentence_text(m.start).strip() if m else ""))))

            # A pairing the checkers refuted must not leave the rule's positional
            # STATED_WITH edge in place: the date-shifter reads that edge as a hard
            # arithmetic constraint, so a wrong pairing is worse than none.
            if fname == "stated_with" and res.value is None:
                _drop_edges(edges, new_edges, Relation.STATED_WITH, ent.entity_id)

            # A CHECKED age<->date pairing becomes a real STATED_WITH edge, so the
            # arithmetic constraint the date-shifter needs exists even when the
            # rule's positional guess found nothing.
            if fname == "stated_with" and res.action == FILL:
                from ..checks.stated_with import date_entity_for
                d = date_entity_for(res.value, ctx)
                if d is not None and not any(
                        e.relation == Relation.STATED_WITH
                        and e.source == ent.entity_id and e.target == d.entity_id
                        for e in edges + new_edges):
                    new_edges.append(Edge(
                        source=ent.entity_id, target=d.entity_id,
                        relation=Relation.STATED_WITH,
                        detail="age and date co-stated; keep arithmetic",
                        evidence=f"(llm pairing, {len(res.checks_passed)} "
                                 f"check(s) passed) {res.value}"))

            # a CHECKED parent suggestion becomes a real LOCATED_IN edge -- the
            # step the old code never took, which is why 002 had zero hierarchy
            if fname == "location_parent" and res.action == FILL:
                from ..checks.location import resolved_parent_key
                hit = resolved_parent_key(res.value, ctx)
                if hit is not None:
                    _key, parent_ent = hit
                    if parent_ent is not ent and not any(
                            e.source == ent.entity_id and e.target == parent_ent.entity_id
                            for e in edges + new_edges):
                        new_edges.append(Edge(
                            source=ent.entity_id, target=parent_ent.entity_id,
                            relation=Relation.LOCATED_IN,
                            detail=f"{ent.subtype or 'place'} in "
                                   f"{parent_ent.subtype or 'place'}",
                            evidence="(llm parent, gazetteer-verified)"))

    _cross_field_consistency(entities, ledger)
    ledger["_edges"] = new_edges
    return ledger


def _cross_field_consistency(entities, ledger) -> None:
    """Deterministic checks that span two resolved fields.

    `given_name == surname` is impossible for one person, and it is exactly what a
    single-token honorific-prefixed name produces: the rule puts the surname in the
    given slot ('Father Nguyen' -> given_name 'Nguyen'), while the LLM correctly
    fills `surname` with the same token. The surname slot is the better-evidenced
    one, so the given name abstains rather than carrying a value we can prove wrong.

    This does not fix the underlying rule -- `attributes.infer_person_attributes`
    still mis-slots -- but it stops the wrong value reaching a consumer.
    """
    for ent in entities:
        a = ent.attributes
        gn, sn = a.get("given_name"), a.get("surname")
        if not gn or not sn or str(gn).lower() != str(sn).lower():
            continue
        # POP, not `= None`. Leaving the key present with a null value hands a
        # consumer a `given_name` that exists and is empty, which reads differently
        # from a name the pipeline never established.
        a.pop("given_name", None)
        ent.flag_entity(
            f"given_name and surname both resolved to {gn!r}, which cannot both be "
            f"true; the given name was dropped (the rule mis-slots a surname when a "
            f"title precedes a single-token name)")
        res = ledger.get(ent.entity_id, {}).get("given_name")
        if res is not None:
            ledger[ent.entity_id]["given_name"] = dataclasses.replace(
                res, action=REJECT, value=None,
                checks_failed=res.checks_failed + ("given_surname_collision",),
                reason=f"dropped: identical to surname {sn!r}")


def blocking_fields(ledger: dict) -> list[tuple[str, str, str]]:
    """(entity_id, field, reason) for every resolution that blocks the transcript."""
    out = []
    for eid, fields in ledger.items():
        if eid == "_edges":
            continue
        for fname, res in fields.items():
            if res.blocking:
                out.append((eid, fname, res.reason))
    return out
