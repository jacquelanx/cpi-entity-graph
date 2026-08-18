"""
Core data structures for knowledge graph.
"""


from __future__ import annotations
from dataclasses import dataclass, field  # use dataclass so we don't have to write init manually
from enum import Enum


"""
A single detected span that mentions an entity.
"""
@dataclass
class Mention:
    transcript_id: str          # the transcript this is from
    start: int                  # starting line number, zero-indexed
    end: int                    # ending line number, inclusive
    text: str                   # span this is from
    entity_type: str            # eg. PERSON, NICKNAME, LOCATION
    mention_id: str             # eg. "t014_m0042"
    score: float = 1.0          # 
    recognizer: str = ""        # which recognizer was used

    # define custom exporter to create relevant JSON object
    def to_dict(self) -> dict:
        return {
            "mention_id": self.mention_id,
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "entity_type": self.entity_type,
        }


"""
An entity that functions as a node in the graph.
"""
@dataclass
class Entity:
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

    """All distinct ways this entity was written with longest first."""
    @property  # lets you access function like attribute
    def sorted_mentions(self) -> list[str]:
        return sorted({m.text for m in self.mentions}, key=len, reverse=True)
    
    """Does this entity need to be reviewed?"""
    def flag_entity(self, reason: str) -> None:
        self.needs_review = True
        self.review_reason = (self.review_reason + "; " + reason).strip("; ") # appends new reason
    
    def to_dict(self) -> dict:
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


"""
Set enum for relationships between nodes.

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
class Relation (str, Enum):
    RELATED_TO = "RELATED_TO"       # person to person
    LOCATED_IN = "LOCATED_IN"       # place to place
    STATED_WITH = "STATED_WITH"     # age <-> the date it was co-stated with
    ATTRIBUTE_OF = "ATTRIBUTE_OF"   # identifier/age/dob -> the person it belongs to

    __str__ = str.__str__


"""
An edge between two entities.
"""
@dataclass
class Edge:
    source: str             # entity_id
    target: str             # entity_id
    relation: Relation
    detail: str = ""
    evidence: str = ""      # the quote that justifies this edge, for review

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "target": self.target,
            # serialize enum to its string value; tolerate a bare string too
            "relation": self.relation.value if isinstance(self.relation, Relation) else self.relation,
            "detail": self.detail,
            "evidence": self.evidence,
        }