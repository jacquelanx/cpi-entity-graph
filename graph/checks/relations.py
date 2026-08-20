"""
The `relation` field's checker (adapter around the evidence verifier).

PURPOSE
    Let a proposed family/social relationship go through the same checker
    machinery as every scalar field, even though a relation is about a PAIR of
    entities rather than one.

FIT
    Named by the `relation` policy in `graph/second_line/policies.py`; the real
    work is in `graph/checks/relation_evidence.py`. `graph/second_line/walk.py`
    reads `ctx.relation_verdict` afterwards to decide between dropping a proposal
    and recording it as a review suggestion.

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
    """The relation verifier's own context, built once per transcript and cached.

    Building it means indexing the transcript for the verifier, so it is stashed
    on `ctx._relctx` and reused across every relation proposal in that transcript.
    The interviewee is excluded from `persons` because the verifier treats the
    speaker separately (a first-person construction, not a named mention).
    """
    rc = getattr(ctx, "_relctx", None)
    if rc is None:
        persons = [e for e in ctx.entities
                   if e.category == "PERSON" and e is not ctx.interviewee]
        rc = RelationContext(ctx.transcript, persons, ctx.interviewee)
        ctx._relctx = rc
    return rc


def locally_provable(value, ctx) -> CheckOutcome:
    """Is this proposed relation actually SAID somewhere in the transcript?

    `value` is the relation word ("aunt"); the pair it applies to arrives on
    `ctx.pair` as `(source_id, target_id, evidence_quote)`.

    Delegates to `verify_relation`, which returns one of three actions, and
    stashes the full verdict on `ctx.relation_verdict` so the caller can tell the
    two failure kinds apart:

      apply    -- a local kin/social construction supports it        -> ok
      suggest  -- plausible, but proving it needs coreference or a
                  distant antecedent this verifier will not chase    -> fail
      (other)  -- refuted by the text, or out of scope               -> fail

    Both non-`apply` outcomes fail the check, so no edge is created; the
    difference is that `suggest` still earns a review note.
    """
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
