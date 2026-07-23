"""
Shared HTML renderers for the demo pages (demo_artifact.py, dashboard.py).
Every function takes a `case` dict from demo_utils.load_case(..., trace=True) and
returns an HTML fragment. CSS lives in the CSS constant so both pages share one
minimalist stylesheet.
"""

from __future__ import annotations
import math
from html import escape

from graph.models import Relation

# ---- category hues (fixed; readable on light surfaces) ----
_HL = {
    "PERSON": ("#E9F0FA", "#234E86"), "LOCATION": ("#EAF2E0", "#315915"),
    "INSTITUTION": ("#EAF2E0", "#315915"), "AGE": ("#EFEDFA", "#413593"),
    "DATE_ABSOLUTE": ("#F8EFDD", "#6A4310"), "DATE_RELATIVE": ("#F8EFDD", "#6A4310"),
    "DATE_ANCHOR": ("#FBEAEA", "#7C2222"), "DATE_OF_BIRTH": ("#F8EFDD", "#6A4310"),
}
_PERSON_FILL = {"FAMILY": "#5B7FA6", "PROFESSIONAL": "#9CB6D2",
                "PUBLIC_FIGURE": "#AEB3BA"}


def _relname(ed):
    return ed.relation.value if isinstance(ed.relation, Relation) else str(ed.relation)


def _pname(e):
    return e.sorted_mentions[0] if e.sorted_mentions else "?"


def _pct(x):
    return "&mdash;" if x is None else f"{x * 100:.0f}%"


def section(num, title, body, note="", wide=False):
    note_html = f"<p class='s-note'>{note}</p>" if note else ""
    cls = "stage wide" if wide else "stage"
    return (f"<section class='{cls}'><div class='s-head'><span class='s-num'>{num}</span>"
            f"<h3>{escape(title)}</h3></div>{note_html}{body}</section>")


# ---------- metrics ----------
def metrics_grid(R) -> str:
    c, r, g, rp = R["cluster"], R["rel"], R["gender"], R["replace"]
    d, a, l = R["dates"], R["ages"], R["locations"]
    leak_cls = "ok" if rp["leaks"] == 0 else "bad"

    def tile(val, lab, sub="", cls=""):
        sub = f"<div class='m-sub'>{sub}</div>" if sub else ""
        return (f"<div class='m-tile'><div class='m-val {cls}'>{val}</div>"
                f"<div class='m-lab'>{lab}</div>{sub}</div>")

    tiles = [
        tile(f"{rp['leaks']}", "privacy leaks", "should-replace but kept", leak_cls),
        tile(_pct(c["recall"]), "clustering recall",
             f"{c['exact']}/{c['gold']} exact &middot; {c['over_merges']} over-merge &middot; {c['splits']} split"),
        tile(_pct(r["precision"]), "relation precision", f"{r['tp']}/{r['pred']} correct"),
        tile(_pct(r["recall"]), "relation recall", f"{r['tp']}/{r['gold']} found"),
        tile(_pct(g["recall"]), "gender recall", f"{g['missing']} missing"),
        tile(_pct(rp["accuracy"]), "replace accuracy", f"{rp['over_redactions']} over-redaction"),
        tile(_pct(d["accuracy"]), "date accuracy", f"{d['pass']}/{d['total']}"),
        tile(_pct(a["accuracy"]), "age accuracy", f"{a['pass']}/{a['total']}"),
    ]
    return "<div class='metrics'>" + "".join(tiles) + "</div>"


# ---------- 1 · detect ----------
def stage_detect(case) -> str:
    text, dets = case["text"], sorted(case["dets"], key=lambda d: d["start"])
    out, i = [], 0
    for d in dets:
        if d["start"] < i:
            continue
        out.append(escape(text[i:d["start"]]))
        fill, col = _HL.get(d["entity_type"], ("#eee", "#333"))
        out.append(f"<mark style='background:{fill};color:{col}'>"
                   f"{escape(text[d['start']:d['end']])}</mark>")
        i = d["end"]
    out.append(escape(text[i:]))
    # preserve the interview's line/turn breaks instead of one wall of text
    rendered = "".join(out).strip("\n").replace("\n", "<br>")
    legend = ("<div class='legend'>"
              "<span class='lg lg-p'>person</span><span class='lg lg-l'>location</span>"
              "<span class='lg lg-d'>date</span><span class='lg lg-a'>age</span></div>")
    body = f"<div class='excerpt'>{rendered}</div>{legend}"
    return section("01", "Detect identifier spans", body,
                   f"{len(dets)} spans from the (simulated perfect) detector, color-coded by type.",
                   wide=True)


# ---------- 2 · rule-based clustering ----------
def stage_cluster(case) -> str:
    pre = case["info"].get("pre_coref", [])
    n_ment = sum(1 for m in case["mentions"] if m.entity_type in ("PERSON", "NICKNAME"))
    merges = [p for p in pre if len(p["forms"]) > 1]
    flags = [p for p in pre if p["flag"]]
    rows = []
    for p in merges:
        rows.append(f"<li><b>{escape(' + '.join(p['forms']))}</b> "
                    f"<span class='tag ok'>merged by rule</span></li>")
    for p in flags:
        rows.append(f"<li><b>{escape(p['forms'][0] if p['forms'] else '?')}</b> "
                    f"<span class='tag warn'>&#9873;</span> "
                    f"<span class='muted'>{escape(p['flag'])}</span></li>")
    if not rows:
        rows.append("<li class='muted'>Every name appeared in a single form &mdash; nothing to merge or flag.</li>")
    body = f"<ul class='deltas'>{''.join(rows)}</ul>"
    return section("02", "Cluster mentions (rule-based)", body,
                   f"{n_ment} person mentions &rarr; {len(pre)} entities via exact / nickname / "
                   f"containment rules.")


# ---------- 3 · coreference (ML) ----------
def stage_coref(case) -> str:
    info = case["info"]
    if not info.get("coref_ran"):
        return section("03", "Coreference resolution (ML)",
                       "<p class='muted'>Coref did not run for this transcript.</p>")
    merges = info.get("coref_merges", [])
    flags = info.get("coref_flags", [])
    rows = []
    for m in merges:
        rows.append(f"<li><span class='tag ok'>merged</span> "
                    f"<b>{escape(m['merged'])}</b> &rarr; <b>{escape(m['kept'])}</b> "
                    f"<span class='muted'>(names compatible)</span></li>")
    for f in flags:
        rows.append(f"<li><span class='tag warn'>held apart</span> "
                    f"<b>{escape(f['a'])}</b> &harr; <b>{escape(f['b'])}</b> "
                    f"<span class='muted'>{escape(f['note'])}</span></li>")
    if not rows:
        rows.append("<li class='muted'>fastcoref suggested no cross-entity links here.</li>")
    body = f"<ul class='deltas'>{''.join(rows)}</ul>"
    return section("03", "Coreference resolution (ML)", body,
                   "fastcoref proposes links; we merge only name-compatible pairs and "
                   "flag the rest rather than over-merge.")


# ---------- 4 · relations ----------
def stage_relations(case) -> str:
    ents = {e.entity_id: e for e in case["entities"]}
    iv = case["info"]["interviewee"].entity_id

    def nm(i):
        return "Interviewee" if i == iv else (_pname(ents[i]) if i in ents else i)

    rows = []
    for ed in case["edges"]:
        rel = _relname(ed)
        if rel == "RELATED_TO":
            rows.append(f"<li><b>{escape(nm(ed.source))}</b> "
                        f"<span class='rel'>{escape(ed.detail)}</span> "
                        f"<b>{escape(nm(ed.target))}</b> "
                        f"<span class='ev'>&ldquo;{escape(ed.evidence)}&rdquo;</span></li>")
    body = (f"<ul class='rels'>{''.join(rows)}</ul>" if rows
            else "<p class='muted'>No family relationships were extracted.</p>")
    return section("04", "Extract relationships", body,
                   f"{len(rows)} kinship edges, each with the quote that justifies it.")


# ---------- 5 · resolved entities & attributes ----------
def stage_entities(case) -> str:
    iv = case["info"]["interviewee"].entity_id
    cards = []
    for e in case["entities"]:
        if e.category != "PERSON" or e.entity_id == iv or not e.mentions:
            continue
        chips = [f"<span class='chip c-person'>PERSON"
                 + (f" &middot; {e.subtype}" if e.subtype else "") + "</span>"]
        if e.attributes.get("gender"):
            chips.append(f"<span class='chip c-attr'>gender {e.attributes['gender']}</span>")
        chips.append(f"<span class='chip c-attr'>replace: "
                     f"{'yes' if e.attributes.get('replace', True) else 'no'}</span>")
        if e.needs_review:
            chips.append("<span class='chip c-flag'>&#9873; review</span>")
        forms = " / ".join(escape(f) for f in e.sorted_mentions)
        note = f"<div class='note'>{escape(e.review_reason)}</div>" if e.needs_review else ""
        cards.append(f"<div class='card{' flag' if e.needs_review else ''}'>"
                     f"<div class='nm'>{escape(_pname(e))}</div>"
                     f"<div class='forms'>{forms}</div>{''.join(chips)}{note}</div>")
    body = f"<div class='cards'>{''.join(cards)}</div>"
    return section("05", "Resolve entities &amp; attributes", body)


# ---------- 6 · graph ----------
def _graph_svg(case) -> str:
    ents = {e.entity_id: e for e in case["entities"]}
    iv = case["info"]["interviewee"].entity_id
    rel = [(x.source, x.target, x.detail) for x in case["edges"] if _relname(x) == "RELATED_TO"]
    if not rel:
        return "<p class='muted'>No relationship graph for this transcript.</p>"
    adj, nodes = {}, set()
    for s, t, d in rel:
        nodes.update((s, t))
        adj.setdefault(s, []).append(t)
        adj.setdefault(t, []).append(s)

    def disp(nid):
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

    parts = [f"<svg width='100%' viewBox='0 0 {W:.0f} {H:.0f}' role='img' "
             f"xmlns='http://www.w3.org/2000/svg'><title>Relationship graph</title>"]
    for s, t, d in rel:
        if s in pos and t in pos:
            x1, y1 = P(s); x2, y2 = P(t)
            parts.append(f"<line x1='{x1:.0f}' y1='{y1:.0f}' x2='{x2:.0f}' y2='{y2:.0f}' "
                         f"stroke='#c9ced6' stroke-width='1.5'/>")
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
        parts.append(f"<text x='{x:.0f}' y='{y + 4:.0f}' text-anchor='middle' font-size='12' "
                     f"font-weight='500' fill='#fff'>{escape(disp(n))}</text>")
        if cap.get(n):
            parts.append(f"<text x='{x:.0f}' y='{y + r + 15:.0f}' text-anchor='middle' "
                         f"font-size='10.5' fill='#8a8f98'>{escape(cap[n])}</text>")
    parts.append("</svg>")
    return "".join(parts)


def _places_times(case) -> str:
    ents = {e.entity_id: e for e in case["entities"]}
    loc = [(x.source, x.target) for x in case["edges"] if _relname(x) == "LOCATED_IN"]
    rows = []
    for s, t in loc:
        cs = _pname(ents[s]) if s in ents else s
        ct = _pname(ents[t]) if t in ents else t
        rows.append(f"<div class='locrow'><span class='pill loc'>{escape(cs)}</span>"
                    f"<span class='arrow'>in &rarr;</span>"
                    f"<span class='pill loc'>{escape(ct)}</span></div>")
    loc_html = (f"<div class='sub-h'>Location hierarchy</div>{''.join(rows)}") if rows else ""
    chips = []
    for e in case["entities"]:
        if e.category.startswith("DATE"):
            v = e.attributes.get("resolved_value") or "unresolved"
            chips.append(f"<span class='pill date'>{escape(e.mentions[0].text)} &rarr; {escape(str(v))}</span>")
        elif e.category == "AGE":
            chips.append(f"<span class='pill age'>{escape(e.mentions[0].text)} &rarr; "
                         f"{escape(str(e.attributes.get('value')))}</span>")
    dt_html = (f"<div class='sub-h'>Dates &amp; ages resolved</div><div class='pills'>{''.join(chips)}</div>"
               if chips else "")
    return loc_html + dt_html


def stage_graph(case) -> str:
    body = f"<div class='graph'>{_graph_svg(case)}{_places_times(case)}</div>"
    return section("06", "Entity graph", body,
                   "Interviewee at center; a red ring marks an entity flagged for review.",
                   wide=True)


def transcript_panel(case, metrics=None) -> str:
    head = metrics_grid(metrics) if metrics else ""
    stages = (stage_detect(case) + stage_cluster(case) + stage_coref(case)
              + stage_relations(case) + stage_entities(case) + stage_graph(case))
    return head + f"<div class='stages'>{stages}</div>"


CSS = """
:root{--ink:#1b1d21;--muted:#6b7280;--faint:#9ca3af;--line:#ececef;--panel:#fafafa;--accent:#2C4A6E;}
*{box-sizing:border-box;}
body{margin:0;background:#fff;color:var(--ink);-webkit-font-smoothing:antialiased;
  font-family:'Iowan Old Style','Palatino Linotype',Palatino,'Book Antiqua',Georgia,'Times New Roman',serif;
  font-size:16px;line-height:1.65;}
.wrap{max-width:1180px;margin:0 auto;padding:40px 32px 80px;}
h1{font-size:25px;font-weight:600;letter-spacing:-0.4px;margin:0 0 4px;}
.lede{color:var(--muted);font-size:14.5px;margin:0 0 8px;max-width:70ch;}
h3{font-size:16px;font-weight:600;margin:0;}
.muted{color:var(--muted);} .faint{color:var(--faint);}

/* tabs */
.tabs{display:flex;flex-wrap:wrap;gap:2px;border-bottom:1px solid var(--line);margin:26px 0 30px;}
.tab{appearance:none;background:none;border:none;cursor:pointer;font:inherit;font-size:14px;
  color:var(--muted);padding:10px 16px 12px;border-bottom:2px solid transparent;margin-bottom:-1px;}
.tab .t-sub{display:block;font-size:11.5px;color:var(--faint);margin-top:1px;}
.tab:hover{color:var(--ink);}
.tab.active{color:var(--ink);border-bottom-color:var(--accent);}
.tab.active .t-sub{color:var(--muted);}
.panel{display:none;} .panel.active{display:block;}

/* metrics */
.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:var(--line);
  border:1px solid var(--line);border-radius:12px;overflow:hidden;margin-bottom:38px;}
.m-tile{background:#fff;padding:16px 18px;}
.m-val{font-size:26px;font-weight:600;letter-spacing:-0.5px;}
.m-val.ok{color:#2f7a3d;} .m-val.bad{color:#b3261e;}
.m-lab{font-size:12.5px;color:var(--muted);margin-top:2px;}
.m-sub{font-size:11px;color:var(--faint);margin-top:3px;}

/* stages: two-column grid; wide stages span the full row */
.stages{display:grid;grid-template-columns:1fr 1fr;gap:34px 40px;align-items:start;}
.stage{margin:0;min-width:0;}
.stage.wide{grid-column:1 / -1;}
@media(max-width:880px){.stages{grid-template-columns:1fr;}.metrics{grid-template-columns:repeat(2,1fr);}}
.s-head{display:flex;align-items:baseline;gap:12px;margin-bottom:6px;}
.s-num{font-size:12px;font-weight:600;color:var(--faint);letter-spacing:1px;}
.s-note{color:var(--muted);font-size:13px;margin:0 0 14px;}

.excerpt{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:22px 26px;line-height:1.9;font-size:15px;}
mark{padding:1px 5px;border-radius:4px;font-weight:500;}
.legend{display:flex;gap:16px;margin-top:12px;font-size:12px;color:var(--muted);}
.lg::before{content:"";display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:6px;vertical-align:1px;}
.lg-p::before{background:#5B7FA6;}.lg-l::before{background:#6f9a4a;}.lg-d::before{background:#c99a4a;}.lg-a::before{background:#7d72c4;}

.deltas,.rels{list-style:none;padding:0;margin:0;}
.deltas li,.rels li{padding:9px 0;border-bottom:1px solid var(--line);font-size:14px;}
.deltas li:last-child,.rels li:last-child{border-bottom:none;}
.tag{display:inline-block;font-size:11px;padding:1px 8px;border-radius:20px;margin-right:6px;}
.tag.ok{background:#e7f2e9;color:#2f6b3b;} .tag.warn{background:#fbeceb;color:#93362f;}
.rel{display:inline-block;font-size:12px;color:#234E86;background:#E9F0FA;padding:1px 8px;border-radius:20px;margin:0 4px;}
.ev{display:block;color:var(--faint);font-size:12.5px;margin-top:2px;font-style:italic;}

.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(228px,1fr));gap:12px;}
.card{background:#fff;border:1px solid var(--line);border-radius:12px;padding:14px 16px;}
.card.flag{border-left:3px solid #b3261e;}
.card .nm{font-size:15.5px;font-weight:600;}
.card .forms{font-size:12px;color:var(--muted);margin:1px 0 9px;}
.chip{display:inline-block;font-size:11px;padding:2px 8px;border-radius:20px;margin:0 4px 4px 0;}
.c-person{background:#E9F0FA;color:#234E86;}.c-attr{background:#f0f0ee;color:#4b4b47;}.c-flag{background:#fbeceb;color:#7C2222;}
.card .note{font-size:11.5px;color:#7C2222;margin-top:5px;line-height:1.5;}

.graph{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px;}
.sub-h{font-size:12px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:0.6px;margin:20px 0 10px;}
.locrow{display:flex;align-items:center;gap:10px;margin-bottom:8px;}
.arrow{font-size:12px;color:var(--faint);}
.pills{display:flex;flex-wrap:wrap;gap:8px;}
.pill{font-size:12.5px;padding:4px 11px;border-radius:20px;}
.pill.loc{background:#EAF2E0;color:#315915;}.pill.date{background:#F8EFDD;color:#6A4310;}.pill.age{background:#EFEDFA;color:#413593;}
.foot{font-size:12.5px;color:var(--muted);border-top:1px solid var(--line);padding-top:16px;margin-top:10px;}
"""
