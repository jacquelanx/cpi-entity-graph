"""
Writing a settled `Resolution` back onto the entity.

`apply_resolution` is the only place a decision becomes state: the attribute,
the provenance record, the review flag, and the legacy mirror keys older
consumers still read.
"""

from __future__ import annotations

from .outcomes import CONFIRM, CONFLICT, FILL, FieldPolicy, KEEP, REJECT, Resolution


# ---------------------------------------------------------------- apply / ledger

def _legacy_mirror(ent, res: Resolution, policy: FieldPolicy) -> None:
    """Keep writing the old `suggested_*` / `*_confirmed` keys.

    The HTML report and the demo scripts read those keys. Mirroring them means the
    unification can land without touching the reporting layer, and a reviewer sees
    the same page with better decisions behind it.
    """
    a = ent.attributes
    f = policy.field
    if res.action == CONFIRM:
        if f in ("gender", "interviewee_gender"):
            a["gender_confirmed"] = True
        elif f == "resolved_value":
            a["date_confirmed"] = True
        elif f == "value":
            a["age_confirmed"] = True
        elif f == "ethnicity":
            # the report reads `suggested_ethnicity`, so mirror a confirmed value
            # too -- otherwise a rule value the LLM agreed with would be the one
            # outcome that disappeared from the page
            a["suggested_ethnicity"] = res.value
            a["ethnicity_confirmed"] = True
        return
    if res.action == FILL:
        if f in ("gender", "interviewee_gender"):
            a["suggested_gender"] = res.value
        elif f == "subtype_person":
            a["suggested_subtype"] = res.value
            a["suggested_subtype_confidence"] = res.confidence
        elif f == "subtype_location":
            a["suggested_type"] = res.value
            a["suggested_type_confidence"] = res.confidence
        elif f == "location_parent":
            a["suggested_parent"] = res.value
        elif f in ("resolved_value", "value"):
            a["suggested_value"] = res.value
            a["suggested_value_confidence"] = res.confidence
        elif f == "kind":
            a["suggested_kind"] = res.value
        elif f == "role":
            a["suggested_role"] = res.value
        elif f == "ethnicity":
            # the report reads `suggested_ethnicity`; `ethnicity_basis` now records
            # that a CHECKER verified the label, not merely that the model claimed to
            # have read it somewhere
            a["suggested_ethnicity"] = res.value
            a["ethnicity_basis"] = "stated" if res.checks_passed else "unverified"
        return
    if res.action == CONFLICT and f in ("resolved_value", "value"):
        # only when there IS a competing value to show. Writing the key with None put a
        # null-valued attribute on the entity, which reads to a consumer as "the LLM
        # checked this and answered nothing" rather than "no comparison was recorded".
        other = None if res.source == "llm_checked" else res.value
        if other is None:
            a.pop("llm_check_value", None)
        else:
            a["llm_check_value"] = other


# Attributes the RULE layer writes ALONGSIDE a field's own value, which must be
# erased with it. `apply_resolution` popped only `policy.attr`, so a refuted
# `ethnicity` left `ethnicity_basis="stated"` and an `ethnicity_evidence` quote on the
# entity -- a consumer reading the basis key saw a verified claim with no value behind
# it. Same shape for the speaker's `gender_evidence`, for an anchor's `anchor_event`,
# and for the identifier sub-attributes minted from a `kind` the checkers refused.
_COMPANION_ATTRS: dict[str, tuple] = {
    "ethnicity": ("ethnicity_basis", "ethnicity_evidence", "suggested_ethnicity",
                  "ethnicity_confirmed"),
    "gender": ("gender_evidence", "suggested_gender", "gender_confirmed"),
    "interviewee_gender": ("gender_evidence", "suggested_gender",
                           "gender_confirmed"),
    "resolved_value": ("anchor_event", "suggested_value", "date_confirmed"),
    "value": ("suggested_value", "age_confirmed"),
    "role": ("suggested_role",),
    "kind": ("kind", "digits", "local", "domain", "handle", "occupation"),
    "location_parent": ("suggested_parent",),
    "stated_with": (),
}


def apply_resolution(ent, res: Resolution, policy: FieldPolicy) -> None:
    """Write the resolved value, its provenance, and a flag when a human is needed."""
    if not hasattr(ent, "provenance"):
        ent.provenance = {}
    ent.provenance[policy.field] = res

    attr = policy.attr or policy.field
    # KEEP writes too. For most fields that is a no-op -- the rule layer already put
    # its answer on the entity with `setdefault`, so "the rules stand" needs nothing
    # written. But a rule value that is COMPUTED during arbitration rather than read
    # off the entity has no other way to land, and with the LLM off every one of those
    # fields resolves KEEP:
    #
    #   * `replace_date` (a function of the resolved `shiftable`) and `replace_age` (a
    #     function of the `value` Resolution) reached the graph with NO `replace` key
    #     at all in rules-only runs -- the exact hole they were added to close, back
    #     again through a different door. The evaluation caught it: date redaction 37%,
    #     age redaction 0%, every span reported as kept.
    #   * `interviewee_identity` hit this first and `graph/rules/interviewee.py` worked around
    #     it locally by writing `identity_entity_id` itself.
    #
    # Writing the resolved value on KEEP is also just the honest semantics: the
    # Resolution says what the field IS, so the attribute should say the same thing.
    if res.action in (FILL, CONFIRM, KEEP) or (res.action == CONFLICT and res.value is not None):
        if attr == "subtype":
            ent.subtype = str(res.value).upper() if res.value is not None else None
        elif attr == "category":
            pass                              # never re-type an entity from a suggestion
        else:
            ent.attributes[attr] = res.value
    elif res.value is None and res.action in (CONFLICT, REJECT):
        # ABSTENTION MUST ERASE. The rules write their answers onto the entity before
        # arbitration runs (`attributes.setdefault`, `pipeline._link_interviewee_pii`,
        # the gazetteer's `e.subtype`), so a resolution that concludes "no verified
        # value" used to leave the refuted rule value sitting on the entity for a
        # consumer to read. That is the difference between a checker that reports a
        # problem and a checker that prevents one: a refuted `owner="interviewee"`
        # still handed the speaker somebody else's phone number.
        #
        # Only ever erases when the second line landed on NO value -- a conflict that
        # resolved to a real value (`safe_value=True` on `replace`) is written above.
        if attr == "subtype":
            ent.subtype = None
        elif attr == "category":
            pass                              # the category is the detector's, not ours
        else:
            ent.attributes.pop(attr, None)
        # ...and the rule's COMPANION keys, or the erasure is only half done
        for k in _COMPANION_ATTRS.get(policy.field, ()):
            ent.attributes.pop(k, None)

    # `kind` acceptance regenerates the derived sub-attributes from the rule
    # regexes rather than leaving stale ones in place. Also on CONFIRM: the
    # sub-attributes feed surrogate minting, so they are regenerated whenever the
    # category was re-verified, not only when the LLM supplied it.
    if policy.field == "kind" and res.action in (FILL, CONFIRM):
        from ..checks.identifiers import renormalized_attrs
        ent.attributes.update({k: v for k, v in
                               renormalized_attrs(res.value, _CTX[0]).items()
                               if k not in ("replace",)})

    _legacy_mirror(ent, res, policy)

    if res.action == CONFIRM:
        return
    if res.action == FILL:
        n = len(res.checks_passed)
        if res.source == "checker_derived":
            # Not an LLM fill: neither layer gave a usable answer and the checkers
            # themselves singled this value out (see `_try_alternatives`). Saying
            # "filled from the LLM" here would misreport where the value came from.
            ent.flag_entity(f"{policy.field}: neither layer produced a usable value; "
                            f"{n} deterministic check(s) single out {res.value!r} and "
                            f"refute the alternatives; review")
        elif n:
            ent.flag_entity(f"{policy.field}: filled from the LLM; {n} deterministic "
                            f"check(s) verified it (confidence {res.confidence}); "
                            f"review")
        else:
            # No checker inspected this value. Say so: the old wording counted
            # non-applicable checkers as passes, which advertised an unverified
            # fill as a checked one.
            ent.flag_entity(f"{policy.field}: filled from the LLM with NO "
                            f"deterministic check applicable to this value "
                            f"(confidence {res.confidence}); review")
        return
    if res.action == CONFLICT:
        ent.flag_entity(f"{policy.field}: {res.reason}"
                        + ("; BLOCKING" if res.blocking else ""))
        return
    if res.action == REJECT and res.reason:
        ent.flag_entity(f"{policy.field}: no verified value -- {res.reason}"
                        + ("; BLOCKING" if res.blocking else ""))
        return
    if res.action == KEEP and res.blocking:
        ent.flag_entity(f"{policy.field}: rule value is unconfirmed and this field "
                        f"must be verified; BLOCKING")


# `apply_resolution` needs the live context for `kind` re-normalization; kept in a
# one-slot list so the module stays import-cycle free.
_CTX: list = [None]


def _drop_edges(edges: list, new_edges: list, relation, source_id: str) -> None:
    """Remove every `relation` edge leaving `source_id`, from both edge lists.

    The counterpart to "abstention must erase" in `apply_resolution`: a rule stage
    can write an EDGE before arbitration runs, and dropping the attribute while
    leaving the edge would just move the stale claim somewhere a consumer still
    walks. Mutates `edges` in place because the caller's list is the pipeline's.
    """
    for bucket in (edges, new_edges):
        bucket[:] = [e for e in bucket
                     if not (e.relation == relation and e.source == source_id)]
