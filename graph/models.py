"""
Core data structures for the knowledge graph.

PURPOSE
    Defines the four value types every other module in `graph/` passes around:
    `Mention` (one detected span of text), `Entity` (a node -- one real-world
    person/place/date), `Relation` (the edge vocabulary) and `Edge` (a link
    between two entities). Also defines how each of them serializes to JSON.

FIT
    This is the bottom of the dependency stack: it imports nothing from the
    project and virtually everything imports it. `graph/loader.py` produces
    `Mention`s from detector JSON; `graph/rules/*` group those mentions into
    `Entity`s and emit `Edge`s; `graph/checks/*` and `graph/second_line/*` read
    and annotate entities (via `attributes` and `provenance`);
    `graph/serialize.py` calls the `to_dict` methods here to build the artifact
    that surrogate generation consumes.

HOW
    Plain `@dataclass`es rather than a graph library, so the pipeline can be a
    sequence of ordinary function calls over lists. Two conventions carry most
    of the design:

      * `Entity.attributes` is an open dict rather than fixed fields. Stages add
        keys as they infer them (`gender`, `surname`, `owner`, `replace`, ...),
        which lets `second_line` treat "a field" uniformly by name without this
        module knowing every field that exists.
      * `Entity.provenance` records HOW each attribute was decided, keyed by the
        same field name. That is what makes a decision auditable after the fact
        instead of just present.
"""


from __future__ import annotations
from dataclasses import dataclass, field  # use dataclass so we don't have to write init manually
from enum import Enum


@dataclass
class Mention:
    """A single detected span of transcript text that refers to some entity.

    One `Mention` is one hit from the upstream detection stage -- e.g. the eight
    characters "Aunt Maria" at offset 42. Mentions are the raw material of the
    pipeline: clustering decides which mentions are the same person, so several
    mentions ("Maria", "Aunt Maria", "she") end up attached to one `Entity`.

    Offsets are CHARACTER offsets into the raw transcript text, 0-indexed and
    end-exclusive, so `transcript[start:end] == text` holds exactly.
    `graph/loader.load_detections` enforces that invariant on the way in, and
    `graph/serialize.py` re-checks it on the way out -- surrogate generation
    splices replacement text at these positions, so a drifted offset would
    corrupt the output silently.
    """
    transcript_id: str          # the transcript this is from
    start: int                  # character offset of the first character, 0-indexed
    end: int                    # character offset one PAST the last character (exclusive)
    text: str                   # the exact substring transcript[start:end]
    entity_type: str            # eg. PERSON, NICKNAME, LOCATION
    mention_id: str             # eg. "t014_m0042"
    score: float = 1.0          # detector confidence, 0..1 (1.0 when unstated)
    recognizer: str = ""        # which recognizer was used

    # define custom exporter to create relevant JSON object
    def to_dict(self) -> dict:
        """Return the JSON-safe view of this mention for the output artifact.

        Deliberately narrower than the dataclass: `transcript_id`, `score` and
        `recognizer` are dropped because the artifact already names its
        transcript once at the top level, and detector bookkeeping is not
        something a downstream consumer should key off.
        """
        return {
            "mention_id": self.mention_id,
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "entity_type": self.entity_type,
        }


@dataclass
class Entity:
    """One node in the graph: a single real-world thing the transcript talks about.

    An entity gathers every `Mention` judged to refer to the same thing, plus
    whatever the pipeline has inferred about it. A person entity might hold
    three mentions ("Maria", "Aunt Maria", "my aunt"), `attributes` of
    `{"gender": "female", "given_name": "Maria"}`, and a `provenance` entry per
    attribute saying whether a rule, the LLM, or both agreed on it.

    Non-person categories use the same shape: a LOCATION entity carries a
    gazetteer `subtype` (CITY / STATE / ...), a DATE entity carries a
    `resolved_value`, an AGE entity carries an `owner`.
    """
    entity_id: str                                          # eg. "t014_e007"
    category: str                                           # PERSON, LOCATION, ...
    subtype: str | None = None                              # FAMILY, CITY, ...
    mentions: list[Mention] = field(default_factory=list)   # default factory so each instance gets own copy
    attributes: dict = field(default_factory=dict)          # gender, surname, nationality, etc.
    needs_review: bool = False
    review_reason: str = ""
    # field -> second_line.Resolution: how each field was decided (rule / confirmed
    # / llm_checked), which deterministic checks ran, and whether it blocks.
    provenance: dict = field(default_factory=dict)

    @property  # lets you access function like attribute
    def sorted_mentions(self) -> list[str]:
        """The distinct surface FORMS this entity was written as, longest first.

        Returns strings, not `Mention` objects -- the set comprehension collapses
        repeats, so three occurrences of "Maria" contribute one entry. Longest
        first because the longest form is the most informative label for the
        entity: given `{"Maria", "Aunt Maria"}` you get
        `["Aunt Maria", "Maria"]`, and callers that just want a display name can
        take element 0. Several rules also rely on that ordering when comparing
        a short name against a fuller one.
        """
        return sorted({m.text for m in self.mentions}, key=len, reverse=True)

    def flag_entity(self, reason: str) -> None:
        """Mark this entity for human review, accumulating reasons rather than
        overwriting them.

        Several independent stages can each find something suspect about the same
        entity, so reasons are joined with "; " instead of replaced. The
        `strip("; ")` handles the first call, where `review_reason` is still ""
        and naive concatenation would leave a leading separator.
        """
        self.needs_review = True
        self.review_reason = (self.review_reason + "; " + reason).strip("; ") # appends new reason

    def to_dict(self) -> dict:
        """Return the JSON-safe view of this entity for the output artifact.

        Two details matter. Mentions are emitted in transcript order (sorted by
        `start`) so a consumer reading the artifact top-to-bottom follows the
        interview rather than whatever order clustering happened to append in.
        And `provenance` values are `Resolution` objects, so each is converted
        via its own `to_dict`; the `hasattr` guard passes through anything a
        stage stored as a plain dict instead.
        """
        return {
            "entity_id": self.entity_id,
            "category": self.category,
            "subtype": self.subtype,
            "attributes": self.attributes,
            "mentions": [m.to_dict() for m in sorted(self.mentions, key=lambda m: m.start)],
            "needs_review": self.needs_review,
            "review_reason": self.review_reason,
            "provenance": {k: v.to_dict() if hasattr(v, "to_dict") else v
                           for k, v in self.provenance.items()},
        }


class Relation (str, Enum):
    """The closed vocabulary of edge kinds in the graph.

    Mixes in `str` so a member compares equal to its own wire value. That is not
    cosmetic: `serialize.build_nx_graph` stores the ENUM on each networkx edge while
    `serialize.location_chain` filtered those edges with `d.get("relation") ==
    "LOCATED_IN"`. A plain Enum member never equals a string, so the filter matched
    nothing and `location_chain` always returned `[entity_id]` -- the LOCATED_IN walk
    the surrogate generator needs for consistent place substitution was dead code.
    `__str__` is pinned to `str.__str__` so `f"{rel}"` yields "LOCATED_IN" rather than
    "Relation.LOCATED_IN"; `.value` keeps working, so both existing call sites
    (`models.Edge.to_dict`, `render._rel`) are unaffected.
    """
    RELATED_TO = "RELATED_TO"       # person to person
    LOCATED_IN = "LOCATED_IN"       # place to place
    STATED_WITH = "STATED_WITH"     # age <-> the date it was co-stated with
    ATTRIBUTE_OF = "ATTRIBUTE_OF"   # identifier/age/dob -> the person it belongs to

    __str__ = str.__str__


@dataclass
class Edge:
    """A directed link between two entities, carrying the quote that justifies it.

    `source` and `target` are `entity_id` strings rather than object references,
    which keeps edges cheap to copy and safe to serialize, and means a merge that
    folds one entity into another only has to rewrite ids.

    `detail` narrows the relation -- for a RELATED_TO edge it is the kin term
    ("aunt"), for an ATTRIBUTE_OF edge the kind of thing owned ("PHONE").
    `evidence` is the sentence the inference came from, so a reviewer can judge
    the edge without re-reading the transcript.
    """
    source: str             # entity_id
    target: str             # entity_id
    relation: Relation
    detail: str = ""
    evidence: str = ""      # the quote that justifies this edge, for review

    def to_dict(self) -> dict:
        """Return the JSON-safe view of this edge for the output artifact."""
        return {
            "source": self.source,
            "target": self.target,
            # serialize enum to its string value; tolerate a bare string too
            "relation": self.relation.value if isinstance(self.relation, Relation) else self.relation,
            "detail": self.detail,
            "evidence": self.evidence,
        }
