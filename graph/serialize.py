"""
Output boundary: assemble the networkx graph and write the ARTIFACT that the
surrogate-generation stage consumes.

PURPOSE
    Two related jobs. `build_nx_graph` / `location_chain` turn the flat
    (entities, edges) lists into a real graph so callers can WALK it -- notably
    following LOCATED_IN upward to get a place's full hierarchy.
    `build_payload` / `validate_payload` / `serialize` produce the on-disk JSON
    artifact, and refuse to emit or accept one that cannot be trusted.

FIT
    Last stage in the flow, and the mirror image of `graph/loader.py`: loader
    checks the contract coming IN from detection, this checks the contract going
    OUT to surrogate generation. Called by `scripts/build_graph.py` (writes
    `out/graphs/<id>.json`), by `demo/render/stages_graph.py` (renders the graph
    in the HTML reports) and by `tests/test_invariants.py`. Depends on
    `graph/models.py` for the dataclasses and on `graph/loader.Violation` so both
    boundaries raise the same error type.

HOW
    The artifact is a plain dict, versioned by `GRAPH_VERSION`, and every
    guarantee it makes is re-derived from the transcript at validation time
    rather than trusted. `validate_payload` is the interesting part: it checks
    referential integrity (no edge or blocking row may point at an entity that
    is not present), the presence of the interviewee node, and -- when given the
    transcript -- that the source hash and every single mention offset still
    agree with the text. See "the digest" below for why that last check exists.

WHY THE EXTRA FIELDS
    `serialize` used to emit entities and edges alone, and nothing ever called it. Four
    things the pipeline computes and then dropped on the floor:

  * `blocking` -- the (entity, field, reason) rows the second line could not settle.
    This is the "do NOT mint surrogates yet, a human must decide" signal, and it is
    the single most important field in a de-identification handoff.
  * `interview_date` -- the date-shifter validates shifts against it.
  * the interviewee's entity id -- inferrable from `attributes["role"]`, but a
    convention is not a contract, and interviewee-only de-identification runs
    entirely off this one node.
  * whether an LLM touched the output, and which one.

And one thing nobody computed -- THE DIGEST: A HASH OF THE SOURCE TEXT. Every mention is a pair of
character offsets, so the artifact is meaningless -- worse, silently wrong -- against
any text but the exact bytes it was built from. A re-transcription, a re-encoding or a
stripped BOM shifts every offset, and the surrogate stage would splice at the wrong
positions with no error anywhere. `validate_payload` refuses that.

The transcript itself is deliberately NOT embedded. The artifact already carries every
detected span verbatim, so it is sensitive and has to be handled as such; there is no
reason for it to be a second complete copy of the interview as well.

`validate_payload` / `load_graph` are the read-side counterpart, and they are the
mirror of `loader.load_detections`: the INPUT contract to this stage has been checked
since the beginning (offsets must match the text, labels must be known), and the
OUTPUT contract was not checked at all.
"""


from __future__ import annotations
import hashlib
import json
from pathlib import Path
import networkx as nx
from .loader import Violation
from .models import Edge, Entity

# 0.2 adds `source`, `interview_date`, `interviewee_id`, `run` and `blocking`.
# Bumped because a consumer written against 0.1 would silently miss the review gate.
GRAPH_VERSION = "0.2"


def build_nx_graph(entities: list[Entity], edges: list[Edge]) -> nx.MultiDiGraph:
    """Load the flat entity/edge lists into a networkx graph so they can be walked.

    A `MultiDiGraph` is directed (RELATED_TO "aunt" runs one way) and MULTI, i.e.
    it permits several distinct edges between the same pair of nodes -- which is
    required here, because two entities can legitimately be linked more than once
    (a person can be both RELATED_TO and the ATTRIBUTE_OF target of an
    identifier, and even two RELATED_TO edges with different `detail` are
    meaningful).

    Node and edge attributes are copied across so a walker never has to look
    anything up in the original lists. Note `attributes=e.attributes` shares the
    same dict object rather than copying it, so mutating an entity after building
    the graph is visible through the graph too.
    """
    g = nx.MultiDiGraph()
    for e in entities:
        g.add_node(e.entity_id, category=e.category, subtype=e.subtype,
                   attributes=e.attributes, needs_review=e.needs_review)
    for edge in edges:
        g.add_edge(edge.source, edge.target,
                   relation=edge.relation, detail=edge.detail,
                   evidence=edge.evidence)
    return g


def location_chain(g: nx.MultiDiGraph, entity_id: str) -> list[str]:
    """Walk LOCATED_IN edges upward and return the containment chain, innermost first.

    For a graph holding "Ninth Ward" -> "New Orleans" -> "Louisiana", asking about
    the Ninth Ward returns all three ids in that order. Surrogate generation needs
    this so a fake place substitution stays internally consistent: if the city
    becomes a different city, the neighbourhood inside it has to move with it.

    HOW: start at `entity_id` and repeatedly look for an outgoing LOCATED_IN edge,
    stepping to its target each time, until there is none left. When a node has
    several parents (a gazetteer can offer more than one) it takes the first --
    arbitrary but deterministic given a fixed input order.

    The `current in chain` test is a CYCLE GUARD. A malformed gazetteer row could
    make A contained in B while B is contained in A, and following that blindly
    would loop forever; detecting a node we have already visited stops the walk
    and returns what was found so far.
    """
    chain = [entity_id]
    current = entity_id

    while True:
        # out_edges looks something like (source, target, edge_attributes)
        # _ = source "Street", v = target "Neighborhood", d = dictionary {"relation": "LOCATED_IN"}
        parents = [
            v for _, v, d in g.out_edges(current, data=True)
            if d.get("relation") == "LOCATED_IN"
        ]
        if not parents:
            return chain

        current = parents[0]  # move up
        if current in chain:  # guard against bad gazetteer row
            return chain
        chain.append(current)


# ------------------------------------------------------------------ the artifact

def source_digest(transcript: str) -> str:
    """SHA-256 of the transcript: the fingerprint every offset is relative to.

    A hex digest of the exact UTF-8 bytes. Two texts that differ by even one
    character -- a re-transcription, a stripped byte-order mark, CRLF instead of
    LF -- produce completely different digests, which is what lets
    `validate_payload` tell "the same interview" from "the same bytes". `or ""`
    keeps a `None` transcript hashable so callers can validate structure alone.
    """
    return hashlib.sha256((transcript or "").encode("utf-8")).hexdigest()


def _name(ent, interviewee=None) -> str:
    """A human-readable label for an entity, for review rows a person will read.

    Prefers the longest surface form the transcript actually used ("Aunt Maria").
    Falls back to the entity id, except for the interviewee -- see below.
    """
    forms = getattr(ent, "sorted_mentions", None) or []
    if forms:
        return forms[0]
    # The speaker is a synthetic node with no detected span until identification
    # finds them a name, and on a transcript that never names them it stays that way.
    # "the interviewee" is what a reviewer needs to read there, not "t_e000".
    if interviewee is not None and ent is interviewee:
        return "the interviewee"
    return ent.entity_id


def build_payload(transcript_id: str, transcript: str, entities: list[Entity],
                  edges: list[Edge], info: dict) -> dict:
    """Build the artifact as a plain dict, ready to validate and write.

    `info` is `run_pipeline`'s third return value -- the bag holding the
    interviewee node, the interview date, whether coref/LLM ran, and the blocking
    rows. Everything this function emits beyond `entities`/`edges` comes from
    there; see the module docstring for why each field is worth carrying.

    The one piece of real work is reshaping `info["blocking"]` from bare
    `(entity_id, field, reason)` tuples into dicts that also carry the entity's
    display name, resolved through the `by_id` index built at the top.
    """
    interviewee = info.get("interviewee")
    by_id = {e.entity_id: e for e in entities}

    # Each blocking row carries the entity's DISPLAY NAME as well as its id: a
    # reviewer resolving "interview_002_e000 / interviewee_gender" should not have to
    # grep the entity list to find out who that is.
    blocking = [{"entity_id": eid,
                 "entity": _name(by_id[eid], interviewee) if eid in by_id else eid,
                 "field": field, "reason": reason}
                for (eid, field, reason) in info.get("blocking", [])]

    iv_date = info.get("interview_date")
    return {
        "graph_version": GRAPH_VERSION,
        "transcript_id": transcript_id,
        # what these offsets mean. See the module docstring.
        "source": {"sha256": source_digest(transcript), "chars": len(transcript or "")},
        "interview_date": iv_date.isoformat() if hasattr(iv_date, "isoformat")
                          else iv_date,
        "interviewee_id": getattr(interviewee, "entity_id", None),
        "run": {
            "coref_ran": bool(info.get("coref_ran")),
            "llm_ran": bool(info.get("llm_ran")),
            "llm_model": info.get("llm_model"),
        },
        # The review gate. Empty list means every field either resolved or was
        # allowed to abstain; a non-empty list means a human decides before
        # surrogates are minted.
        "blocking": blocking,
        "entities": [e.to_dict() for e in entities],
        "edges": [e.to_dict() for e in edges],
    }


def validate_payload(payload: dict, transcript: str | None = None) -> dict:
    """Check an artifact and return it, or raise `Violation`.

    The mirror of `loader.load_detections`. Pass `transcript` to check the two
    things that make offsets trustworthy -- the digest and every mention span. Those
    are the checks that stop the surrogate stage splicing at the wrong positions
    against a text that merely LOOKS like the one this was built from.
    """
    def need(cond, msg):
        """Assert an artifact invariant; raise `Violation(msg)` if it does not hold."""
        if not cond:
            raise Violation(msg)

    need(isinstance(payload, dict), "artifact is not a JSON object")
    got = payload.get("graph_version")
    need(got == GRAPH_VERSION,
         f"graph_version {got!r} != {GRAPH_VERSION!r}; this artifact was written by a "
         f"different version of the graph stage")
    for key in ("transcript_id", "source", "interviewee_id", "run", "blocking",
                "entities", "edges"):
        need(key in payload, f"artifact is missing {key!r}")

    ids = [e.get("entity_id") for e in payload["entities"]]
    need(all(ids), "every entity needs an entity_id")
    need(len(ids) == len(set(ids)), "entity_ids are not unique")
    known = set(ids)

    # An edge into nothing is a dangling pointer for anything that walks the graph --
    # LOCATED_IN chains for consistent place substitution, ATTRIBUTE_OF for the
    # speaker's own identifiers.
    for i, ed in enumerate(payload["edges"]):
        for end in ("source", "target"):
            need(ed.get(end) in known,
                 f"edge #{i} ({ed.get('relation')}) points {end} at "
                 f"{ed.get(end)!r}, which is not an entity in this artifact")

    # A blocking row a reviewer cannot resolve to an entity is not a review item.
    for i, b in enumerate(payload["blocking"]):
        need(b.get("entity_id") in known,
             f"blocking row #{i} ({b.get('field')}) names {b.get('entity_id')!r}, "
             f"which is not an entity in this artifact")

    need(payload["interviewee_id"] in known,
         f"interviewee_id {payload['interviewee_id']!r} is not an entity in this "
         f"artifact; interviewee-only surrogate generation has nothing to run on")

    if transcript is not None:
        digest = source_digest(transcript)
        need(payload["source"].get("sha256") == digest,
             "this artifact was built from a DIFFERENT transcript "
             f"({payload['source'].get('sha256', '')[:12]}... != {digest[:12]}...); "
             "every mention offset in it would point at the wrong characters")
        for e in payload["entities"]:
            for m in e.get("mentions", []):
                actual = transcript[m.get("start", 0):m.get("end", 0)]
                need(actual == m.get("text"),
                     f"{e['entity_id']} mention {m.get('mention_id')}: artifact says "
                     f"{m.get('text')!r}, transcript[{m.get('start')}:{m.get('end')}] "
                     f"is {actual!r}")
    return payload


def serialize(transcript_id: str, transcript: str, entities: list[Entity],
              edges: list[Edge], info: dict, out_dir: str | Path) -> Path:
    """Write the artifact to `out_dir/<transcript_id>.json` and return the path.

    Validated before it is written: a malformed artifact that reaches disk is one a
    later stage has to discover the hard way.
    """
    payload = validate_payload(build_payload(transcript_id, transcript, entities,
                                             edges, info), transcript)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{transcript_id}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def load_graph(path: str | Path, transcript: str | None = None) -> dict:
    """Read an artifact back and validate it. Raises `Violation` on any problem."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_payload(payload, transcript)
