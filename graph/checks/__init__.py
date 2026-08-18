"""
Deterministic checkers -- the "verify" half of the unified second line.

A checker answers ONE question about a value the LLM proposed: is it locally
supportable by the transcript and by the rule tables? Checkers use only
deterministic evidence (regex, table lookup, arithmetic) so they can refute the
model's mistakes without inheriting its guesses.

These live in `graph/` on purpose. They are the same predicates the rules
already compute, and keeping them here preserves the one-way dependency
(`graph` -> `llm_layer`): `llm_layer` proposes plain dicts, `graph` arbitrates.

A checker is a callable `(value, ctx: CheckContext) -> CheckOutcome`. A field's
LLM fill is accepted only when EVERY APPLICABLE checker passes; the first failure
names itself in the review flag, so a rejection is always explainable.

An outcome has THREE states, not two. `ok` and `fail` both mean the checker
examined the value; `na` means it did not apply and therefore says nothing about
it. That distinction is not cosmetic: most checkers guard one direction of a
claim ("not a keep claim", "not a DOB") and used to return `ok`, so a value no
checker had actually inspected was reported as having passed N checks. The
ownership checkers were the worst case -- every `owner="other"` fill was
advertised as "4 deterministic check(s) passed" when all four had short-circuited
on the first line. `checks_passed` now counts only real verification, and a fill
with an empty `checks_passed` is visibly unverified.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from ..sentences import sentence_spans
from ..turns import (parse_turns, mask_to_subject, in_subject_turn, role_at,
                     turn_bounds)


@dataclass(frozen=True)
class CheckOutcome:
    name: str
    passed: bool
    detail: str = ""
    applicable: bool = True             # False -> the checker did not inspect the value


def ok(name: str, detail: str = "") -> CheckOutcome:
    return CheckOutcome(name, True, detail)


def fail(name: str, detail: str = "") -> CheckOutcome:
    return CheckOutcome(name, False, detail)


def na(name: str, detail: str = "") -> CheckOutcome:
    """The checker does not apply to this value: it neither supports nor refutes
    it. Counted under `checks_skipped`, never under `checks_passed`."""
    return CheckOutcome(name, True, detail, applicable=False)


@dataclass
class CheckContext:
    """Everything a checker may need, computed once per transcript.

    `entity` is rebound per entity as `resolve_all` walks the graph; everything
    else is shared and read-only.
    """
    transcript: str
    entities: list
    edges: list
    interviewee: object
    entity: object = None
    interview_date: object = None
    gazetteer: dict = field(default_factory=dict)
    gaz_aliases: dict = field(default_factory=dict)
    kin_ids: set = field(default_factory=set)
    # a relation is about a PAIR, so it rides here the way `entity` does:
    # (source_entity_id, target_entity_id, evidence_quote)
    pair: tuple | None = None
    relation_verdict: object = None      # set by checks.relationcheck
    # Entities that are no longer in `entities` but that a checker still has to be
    # able to resolve by id. A rule-applied alias/coref merge FOLDS one entity into
    # another and drops it from the list, so the `same_person` checkers -- which
    # need both sides of the pair -- would see `None` for the folded side and report
    # a real merge as refuted. The pipeline registers the folded objects here.
    extra_entities: dict = field(default_factory=dict)
    _relctx: object = None               # cached RelationContext
    _sents: list | None = None

    @property
    def ent_by_id(self) -> dict:
        d = {e.entity_id: e for e in self.entities}
        d[self.interviewee.entity_id] = self.interviewee
        d.update(self.extra_entities)
        return d

    @property
    def sents(self) -> list:
        if self._sents is None:
            self._sents = sentence_spans(self.transcript)
        return self._sents

    # ---------------------------------------------------------------- turns
    @property
    def turns(self) -> tuple:
        """Speaker turns, so a checker can tell the interviewee's words from the
        interviewer's. Cached in `graph.turns` per transcript string."""
        return parse_turns(self.transcript)

    @property
    def subject_transcript(self) -> str:
        """The transcript with every non-subject character masked out, offsets
        preserved. Any first-person cue search MUST run over this rather than
        `transcript`, or the interviewer's 'I' / 'my' counts as the speaker's."""
        return mask_to_subject(self.transcript)

    def in_subject_turn(self, pos: int) -> bool:
        return in_subject_turn(pos, self.turns)

    def role_at(self, pos: int) -> str:
        return role_at(pos, self.turns)

    def sentence_bounds(self, pos: int) -> tuple[int, int]:
        """The sentence containing `pos`, CLIPPED to its speaker turn.

        A turn that ends without a terminator ("INTERVIEWER: And you?\\nSPEAKER:
        304-555-0176.") otherwise merges into one sentence span, letting a cue
        from one speaker bind a span uttered by another.
        """
        s, e = self.sents[-1] if self.sents else (0, len(self.transcript))
        for a, b in self.sents:
            if a <= pos < b:
                s, e = a, b
                break
        ts, te = turn_bounds(pos, self.turns)
        return max(s, ts), min(e, te) if te > ts else e

    def sentence_text(self, pos: int) -> str:
        s, e = self.sentence_bounds(pos)
        return self.transcript[s:e]

    def first_mention(self):
        ms = getattr(self.entity, "mentions", None)
        return ms[0] if ms else None

    def named_person_spans(self):
        """(start, end, entity_id) for every OTHER named person's mention.

        The interviewee is excluded on purpose. Once the identification stage
        merges the speaker's own name into e000, that name becomes a PERSON
        mention -- and an ownership checker asking "does a named person sit
        between the cue and this span?" would refute the speaker's claim to their
        own phone number because the speaker's own name precedes it.
        """
        iv = getattr(self.interviewee, "entity_id", None)
        return [(m.start, m.end, e.entity_id) for e in self.entities
                if e.category == "PERSON" and e.entity_id != iv
                for m in e.mentions]
