"""
The `relation` field's checker: a thin adapter that runs the deterministic verifier
in `graph/checks/relation_evidence.py` over one proposed (source, target, rel) triple.

Checkers take `(value, ctx)`, but a relation is about a PAIR, so the pair rides on
`ctx.pair` -- the same way the entity under resolution rides on `ctx.entity`. The
verifier's three-way verdict is stashed on `ctx.relation_verdict` so
`second_line.resolve_all` can tell a REFUTED proposal (drop it) from a merely
UNPROVABLE one (record a review suggestion, no edge).
"""

from __future__ import annotations
from . import CheckOutcome, ok, fail
from .relation_evidence import RelationContext, verify_relation


def relation_context(ctx) -> RelationContext:
    """Built once per transcript and cached on the CheckContext."""
    rc = getattr(ctx, "_relctx", None)
    if rc is None:
        persons = [e for e in ctx.entities
                   if e.category == "PERSON" and e is not ctx.interviewee]
        rc = RelationContext(ctx.transcript, persons, ctx.interviewee)
        ctx._relctx = rc
    return rc


def locally_provable(value, ctx) -> CheckOutcome:
    name = "relation_locally_provable"
    pair = getattr(ctx, "pair", None)
    if pair is None:
        return fail(name, "no pair on the context")
    src, tgt, evidence = pair
    v = verify_relation(src, tgt, value, evidence, relation_context(ctx))
    ctx.relation_verdict = v
    if v.action == "apply":
        return ok(name, "supported by a local kin/social construction")
    if v.action == "suggest":
        return fail(name, "plausible but not locally provable "
                          "(needs coreference or a distant antecedent)")
    return fail(name, "refuted or out of scope")
