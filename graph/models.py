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
        }


"""
Set enum for relationships between nodes.
"""
class Relation (Enum):
    RELATED_TO = "RELATED_TO"       # person to person
    LOCATED_IN = "LOCATED_IN"       # place to place
    STATED_WITH = "STATED_WITH"     # age <-> the date it was co-stated with
    ATTRIBUTE_OF = "ATTRIBUTE_OF"   # identifier/age/dob -> the person it belongs to


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