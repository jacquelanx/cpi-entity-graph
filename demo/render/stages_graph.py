"""
Panels 10-12: the entity-graph SVG, the edge table, the ledger, the artifact.

PURPOSE
    Render the finished product: a drawing of the relationship graph, every edge as
    a table, the complete arbitration ledger, and the serialized JSON payload a
    consumer would actually receive.

FIT
    Sits on `primitives.py` and `provenance.py`; assembled by
    `page.transcript_panel`. `stage_artifact` calls into `graph/serialize.py` to
    build and validate the real payload, so the page shows what would be written
    rather than a reconstruction.

HOW
    The SVG is laid out by hand (see `_graph_svg`) rather than by a graph library,
    since the shape being drawn is always the same: the interviewee at the centre,
    their direct relations around them, and anyone else hanging off those.
"""

from __future__ import annotations

import math
from html import escape
from .primitives import _PERSON_FILL, _names_map, _pname, _relname, _val, section
from .provenance import _ACTION, _checks_html


# ---------- 10 · graph ----------
def _graph_svg(case) -> str:
    """Draw the RELATED_TO graph as an SVG, centred on the interviewee.

    Hand-rolled layout in three tiers, which is possible because the graph always
    has the same shape -- a speaker, the people they are directly related to, and
    people related to those:

      * the INTERVIEWEE is the centre, labelled "You";
      * `level1` is everyone with a direct edge to them, placed around the centre,
        each captioned with the relation word from `cap`;
      * anyone else is attached to whichever placed node they connect to
        (`parent_of`).

    `adj` is an undirected adjacency map (each edge recorded in both directions),
    because "is this person connected to the speaker?" does not care which way the
    edge points. Returns a short message instead of an SVG when there are no
    relations to draw.
    """
    ents = {e.entity_id: e for e in case["entities"]}
    iv = case["info"]["interviewee"].entity_id
    rel = [(x.source, x.target, x.detail) for x in case["edges"]
           if _relname(x) == "RELATED_TO"]
    if not rel:
        return "<p class='muted'>No relationship graph for this transcript.</p>"
    adj, nodes = {}, set()
    for s, t, d in rel:
        nodes.update((s, t))
        adj.setdefault(s, []).append(t)
        adj.setdefault(t, []).append(s)

    def disp(nid):
        """Node label: "You" for the interviewee, otherwise the person's name."""
        return "You" if nid == iv else (_pname(ents[nid]) if nid in ents else nid)

    level1 = sorted([n for n in nodes if n != iv and iv in adj.get(n, [])], key=disp)
    cap = {}
    for s, t, d in rel:
        if s == iv:
            cap[t] = d
        elif t == iv:
            cap[s] = d
    placed = {iv, *level1}
    parent_of = {}
    for n in nodes:
        if n in placed:
            continue
        for o in adj.get(n, []):
            if o in level1:
                det = next((d for s, t, d in rel if {s, t} == {n, o}), "")
                parent_of[n] = o
                cap[n] = f"{det} of {disp(o)}"
                break

    pos = {iv: (0.0, 0.0)}
    R1 = 175
    for i, n in enumerate(level1):
        ang = 2 * math.pi * i / max(1, len(level1)) - math.pi / 2
        pos[n] = (R1 * math.cos(ang), R1 * math.sin(ang))
    for n, p in parent_of.items():
        px, py = pos[p]
        ang = math.atan2(py, px)
        pos[n] = (px + 118 * math.cos(ang), py + 118 * math.sin(ang))

    R = 29
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    pad = R + 46
    dx, dy = -min(xs) + pad, -min(ys) + pad
    W, H = (max(xs) - min(xs)) + 2 * pad, (max(ys) - min(ys)) + 2 * pad
    P = lambda n: (pos[n][0] + dx, pos[n][1] + dy)

    disp_w = min(W, 560.0)
    disp_h = disp_w * H / W
    parts = [f"<svg viewBox='0 0 {W:.0f} {H:.0f}' width='{disp_w:.0f}' "
             f"height='{disp_h:.0f}' role='img' "
             f"style='max-width:100%;height:auto;display:block;margin:0 auto' "
             f"xmlns='http://www.w3.org/2000/svg'><title>Relationship graph</title>"]
    for s, t, d in rel:
        if s in pos and t in pos:
            x1, y1 = P(s)
            x2, y2 = P(t)
            parts.append(f"<line x1='{x1:.0f}' y1='{y1:.0f}' x2='{x2:.0f}' "
                         f"y2='{y2:.0f}' stroke='#c9ced6' stroke-width='1.5'/>")
    for n in nodes:
        if n not in pos:
            continue
        x, y = P(n)
        if n == iv:
            fill, r = "#2C4A6E", 32
        else:
            e = ents.get(n)
            fill, r = _PERSON_FILL.get(e.subtype if e else None, "#5B7FA6"), R
        stroke = "#C0392B" if (n in ents and ents[n].needs_review) else "#ffffff"
        sw = 2.5 if stroke == "#C0392B" else 2
        parts.append(f"<circle cx='{x:.0f}' cy='{y:.0f}' r='{r}' fill='{fill}' "
                     f"stroke='{stroke}' stroke-width='{sw}'/>")
        parts.append(f"<text x='{x:.0f}' y='{y + 4:.0f}' text-anchor='middle' "
                     f"font-size='12' font-weight='500' fill='#fff'>"
                     f"{escape(disp(n))}</text>")
        if cap.get(n):
            parts.append(f"<text x='{x:.0f}' y='{y + r + 15:.0f}' "
                         f"text-anchor='middle' font-size='10.5' fill='#8a8f98'>"
                         f"{escape(cap[n])}</text>")
    parts.append("</svg>")
    return "".join(parts)


_GRAPH_LEGEND = (
    "<div class='glegend'>"
    "<span><i style='background:#2C4A6E'></i>interviewee</span>"
    "<span><i style='background:#5B7FA6'></i>family</span>"
    "<span><i style='background:#9CB6D2'></i>professional</span>"
    "<span><i style='background:#AEB3BA'></i>public figure</span>"
    "<span class='ring'>&#9711; red ring = flagged for review</span></div>"
)


_REL_HELP = {
    "RELATED_TO": "person &rarr; person, family or social tie",
    "LOCATED_IN": "place &rarr; the larger place containing it",
    "ATTRIBUTE_OF": "identifier / age / DOB &rarr; the person it belongs to",
    "STATED_WITH": "age &rarr; the date it was co-stated with (keep the arithmetic)",
}


def _edge_table(case) -> str:
    """Every edge in the graph as a table: relation, both ends, detail, evidence."""
    """EVERY edge, of all four relation types. The SVG shows the family graph
    because that is the readable part; this is the rest of the graph."""
    names = _names_map(case)
    counts = {}
    for ed in case["edges"]:
        counts[_relname(ed)] = counts.get(_relname(ed), 0) + 1
    rows = []
    for ed in sorted(case["edges"], key=lambda e: (_relname(e), e.source)):
        rel = _relname(ed)
        rows.append(
            f"<tr><td class='v'>{escape(names.get(ed.source, ed.source))}</td>"
            f"<td><span class='rel'>{escape(rel)}</span></td>"
            f"<td class='v'>{escape(names.get(ed.target, ed.target))}</td>"
            f"<td class='ptype'>{escape(str(ed.detail))}</td>"
            f"<td class='ev2'>{escape(str(ed.evidence)[:160])}</td></tr>")
    tally = " &middot; ".join(
        f"<b>{v}</b> {escape(k)} <i>({_REL_HELP.get(k, '')})</i>"
        for k, v in sorted(counts.items()))
    return (f"<div class='sub-h'>All edges &mdash; {len(case['edges'])} total</div>"
            f"<p class='muted tally2'>{tally}</p>"
            "<table class='prov'><thead><tr><th>from</th><th>relation</th><th>to</th>"
            "<th>detail</th><th>evidence</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>")


def stage_graph(case) -> str:
    """Panel 10: the relationship graph drawing, followed by the full edge table."""
    body = (f"<div class='graph'>{_graph_svg(case)}{_GRAPH_LEGEND}</div>"
            + _edge_table(case))
    return section("10", "Entity graph", body,
                   "The interviewee sits at the centre of the family graph; every "
                   "other edge the pipeline built is listed underneath, because the "
                   "surrogate generator walks all four kinds.")


# ---------- 11 · the full ledger ----------
def stage_ledger(case) -> str:
    """Panel 11: the complete arbitration ledger -- every field decision on every entity.

    The audit view. Rows are grouped by entity and each shows the outcome, the
    value, which layer it came from and which deterministic checks ran, so a
    reader can answer "why is this field what it is?" without reading any code.
    """
    """Every field resolution on every entity, grouped by outcome.

    This is the "nothing is hidden" view. The stages above choose what to foreground;
    this one is exhaustive, so a reviewer can always answer "what did the pipeline
    decide about X, and what backed it?" without reading JSON.
    """
    ledger = case["info"].get("ledger", {})
    names = _names_map(case)
    buckets = {"conflict": [], "fill": [], "keep": [], "confirm": [],
               "refuted": [], "blind": []}
    n_blocking = 0
    for eid, fields in ledger.items():
        if eid == "_edges":
            continue
        for fname, res in fields.items():
            key = ("refuted" if res.action == "reject" and res.checks_failed
                   else "blind" if res.action == "reject" else res.action)
            buckets.setdefault(key, []).append((eid, fname, res))
            if res.blocking:
                n_blocking += 1

    order = [
        ("conflict", "Layers disagreed",
         "the rule and the model gave different answers; the policy's conflict "
         "rule decided, and for a safety field that always means more redaction"),
        ("fill", "The LLM filled a gap",
         "the rules had no answer; the model's was accepted only after every "
         "applicable deterministic checker passed"),
        ("refuted", "A proposal was refuted",
         "the model answered and a deterministic checker proved it wrong, so the "
         "field stayed empty"),
        ("keep", "Rules stood, unconfirmed",
         "the rules answered and the model offered nothing to confirm it with"),
        ("blind", "Neither layer had an answer",
         "made visible rather than left silent -- this is the honest gap list"),
        ("confirm", "Rule and LLM agreed",
         "both layers independently reached the same value"),
    ]
    blocks = []
    for key, title, why in order:
        items = buckets.get(key) or []
        if not items:
            continue
        rows = []
        for eid, fname, res in sorted(items, key=lambda x: (x[1], x[0])):
            label = fname
            if ":" in fname:
                base, other = fname.split(":", 1)
                label = f"{base} &rarr; {escape(names.get(other, other))}"
            else:
                label = escape(fname)
            rows.append(
                f"<tr class='{'blk' if res.blocking else ''}'>"
                f"<td class='v'>{escape(names.get(eid, eid))}</td>"
                f"<td class='f'>{label}</td>"
                f"<td class='v'>{_val(res.value)}</td>"
                f"<td class='src2'>{escape(res.source or '&mdash;')}</td>"
                f"<td class='ck-cell'>{_checks_html(res)}"
                f"<div class='why'>{escape(res.reason)}</div></td></tr>")
        blocks.append(
            f"<details class='ldg' {'open' if key == 'conflict' else ''}>"
            f"<summary><span class='act {_ACTION.get(key, ('', 'a-rej'))[1] if key in _ACTION else ('a-rej' if key == 'refuted' else 'a-blind')}'>"
            f"{escape(title)}</span> <b>{len(items)}</b></summary>"
            f"<p class='muted why2'>{why}</p>"
            "<table class='prov'><thead><tr><th>entity</th><th>field</th>"
            "<th>value</th><th>source</th><th>checks &amp; reason</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></details>")

    total = sum(len(v) for v in buckets.values())
    return section("11", "Decision ledger — every field, every layer",
                   "".join(blocks),
                   f"{total} field resolutions on this transcript"
                   + (f", {n_blocking} of them blocking" if n_blocking else "")
                   + ". Every value the pipeline emits appears here with the layer "
                     "that produced it and the checkers that examined it.")


# ---------- 12 · the artifact ----------
def stage_artifact(case) -> str:
    """Panel 12: the serialized JSON artifact a downstream consumer would receive.

    Built through `graph/serialize.py` -- the real writer, including its validation
    -- so the page shows the actual payload rather than a reconstruction of it,
    and a validation failure surfaces here rather than downstream.
    """
    """What actually gets written to disk for the surrogate-generation stage."""
    from graph.serialize import build_payload

    p = build_payload(case["tid"], case["text"], case["entities"], case["edges"],
                      case["info"])
    run = p["run"]
    cats = {}
    for e in p["entities"]:
        cats[e["category"]] = cats.get(e["category"], 0) + 1
    cat_html = " &middot; ".join(f"<b>{v}</b> {escape(k)}"
                                 for k, v in sorted(cats.items()))

    meta = [
        ("graph_version", p["graph_version"]),
        ("transcript_id", p["transcript_id"]),
        ("source sha256", p["source"]["sha256"][:24] + "&hellip;"),
        ("source chars", f"{p['source']['chars']:,}"),
        ("interview_date", p["interview_date"] or "&mdash;"),
        ("interviewee_id", p["interviewee_id"]),
        ("coref ran", "yes" if run["coref_ran"] else "no"),
        ("LLM second line", (escape(str(run["llm_model"])) if run["llm_ran"]
                             else "off (rules only)")),
    ]
    meta_html = "".join(f"<div class='kv'><span>{escape(k)}</span><b>{v}</b></div>"
                        for k, v in meta)

    if p["blocking"]:
        items = "".join(
            f"<li><b>{escape(b['entity'])}</b> &middot; {escape(b['field'])} "
            f"&mdash; {escape(b['reason'])}</li>" for b in p["blocking"])
        gate = (f"<div class='blkbox'><b>Review gate: {len(p['blocking'])} field(s) "
                f"block surrogate generation.</b> The artifact is still written; "
                f"<code>scripts/build_graph.py</code> exits 2 so an automated run "
                f"stops here.<ul>{items}</ul></div>")
    else:
        gate = ("<div class='okbox'><b>Review gate clear.</b> Nothing blocks; this "
                "artifact is ready for surrogate generation.</div>")

    return section("12", "The artifact — what the next stage receives",
                   f"{gate}<div class='kvs'>{meta_html}</div>"
                   f"<p class='muted tally2'>{len(p['entities'])} entities "
                   f"({cat_html}) &middot; {len(p['edges'])} edges</p>"
                   "<p class='muted'>Written by <code>scripts/build_graph.py</code> "
                   "to <code>out/graphs/&lt;transcript_id&gt;.json</code>. Every "
                   "mention carries character offsets, so the artifact pins the "
                   "SHA-256 of the transcript it was built from and refuses to load "
                   "against any other text.</p>",
                   "The handoff. Entities carry their attributes, their mention "
                   "offsets and their full decision record; the blocking list is the "
                   "gate the next stage must respect.")
