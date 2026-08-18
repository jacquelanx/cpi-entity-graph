"""
Assembles the networkx graph and serializes it to the ARTIFACT that the
surrogate-generation stage consumes.

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

And one thing nobody computed: A HASH OF THE SOURCE TEXT. Every mention is a pair of
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
    g = nx.MultiDiGraph()
    for e in entities:
        g.add_node(e.entity_id, category=e.category, subtype=e.subtype,
                   attributes=e.attributes, needs_review=e.needs_review)
    for edge in edges:
        g.add_edge(edge.source, edge.target,
                   relation=edge.relation, detail=edge.detail,
                   evidence=edge.evidence)
    return g


"""
Walk LOCATED_IN upward: street -> neighborhood -> city -> ...
"""
def location_chain(g: nx.MultiDiGraph, entity_id: str) -> list[str]:
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
    """The hash every offset in this artifact is relative to."""
    return hashlib.sha256((transcript or "").encode("utf-8")).hexdigest()


def _name(ent, interviewee=None) -> str:
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
    """The artifact, as a plain dict. `info` is `run_pipeline`'s third return value."""
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
