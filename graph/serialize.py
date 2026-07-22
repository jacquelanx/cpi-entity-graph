"""
This file assembles the networkx graph and serialize to the proper output JSON
format. networkx is what we are using for surrogate generation later.
"""


from __future__ import annotations
import json
from pathlib import Path
import networkx as nx
from .models import Edge, Entity
GRAPH_VERSION = "0.1"


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


def serialize(
    transcript_id: str,
    entities: list[Entity],
    edges: list[Edge],
    out_dir: str | Path,
    coref_ran: bool,
) -> Path:
    payload = {
        "transcript_id": transcript_id,
        "graph_version": GRAPH_VERSION,
        "coref_ran": coref_ran,
        "entities": [e.to_dict() for e in entities],
        "edges": [e.to_dict() for e in edges],
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{transcript_id}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path
