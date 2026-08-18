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
_ID_CATS = ("PHONE", "EMAIL", "SSN_OR_ID", "USERNAME_HANDLE", "OCCUPATION")
_ID_LABEL = {"PHONE": "Phone", "EMAIL": "Email", "SSN_OR_ID": "SSN / ID",
             "USERNAME_HANDLE": "Handle", "OCCUPATION": "Occupation"}


def _relname(ed):
    return ed.relation.value if isinstance(ed.relation, Relation) else str(ed.relation)


def _pname(e):
    return e.sorted_mentions[0] if e.sorted_mentions else "?"


def _pct(x):
    return "&mdash;" if x is None else f"{x * 100:.0f}%"


def section(num, title, body, note="", wide=False):   # `wide` kept for callers; ignored
    note_html = f"<p class='s-note'>{note}</p>" if note else ""
    cls = "stage"
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
            ev = str(ed.evidence or "")
            llm = ev.startswith("(llm)")
            ev = ev[5:].strip() if llm else ev
            tag = "<span class='src llm'>llm</span>" if llm else "<span class='src rule'>rule</span>"
            rows.append(f"<li>{tag}<b>{escape(nm(ed.source))}</b> "
                        f"<span class='rel'>{escape(ed.detail)}</span> "
                        f"<b>{escape(nm(ed.target))}</b> "
                        f"<span class='ev'>&ldquo;{escape(ev)}&rdquo;</span></li>")
    body = (f"<ul class='rels'>{''.join(rows)}</ul>" if rows
            else "<p class='muted'>No family relationships were extracted.</p>")
    return section("04", "Extract relationships", body,
                   f"{len(rows)} relationships (rule-derived and LLM-added), each with its "
                   f"evidence.")


# ---------- 5 · resolved entities & attributes ----------
def _chip(text, cls=""):
    cls = f"chip {cls}".strip()
    return f"<span class='{cls}'>{text}</span>"


# attribute keys that are surfaced explicitly below; the catch-all skips these so
# it only ever shows things we haven't already rendered (keeps "show EVERYTHING"
# honest without duplicating).
_HANDLED_ATTR_KEYS = {
    "given_name", "surname", "gender", "gender_confirmed", "replace", "role",
    "subtype", "owner", "identifying", "suggested_gender", "gender_confirmed",
    "suggested_role", "suggested_subtype", "suggested_subtype_confidence",
    "public_figure_cosign", "suggested_merge_with", "suggested_ethnicity",
    "ethnicity_basis", "ethnicity_evidence", "ethnicity_confidence",
    "suggested_relation", "merge_evidence", "replace",
}


def _owned_by_person(case):
    """person entity_id -> [owned attribute entities] (AGE / DATE_OF_BIRTH /
    PHONE / EMAIL / SSN_OR_ID / USERNAME_HANDLE / OCCUPATION), gathered from the
    ATTRIBUTE_OF edges the pipeline builds, plus anything the LLM tagged
    owner=='interviewee' that has no edge (e.g. OCCUPATION)."""
    ents = {e.entity_id: e for e in case["entities"]}
    iv = case["info"]["interviewee"].entity_id
    owned, seen = {}, set()
    for ed in case["edges"]:
        if _relname(ed) == "ATTRIBUTE_OF" and ed.source in ents:
            owned.setdefault(ed.target, []).append(ents[ed.source])
            seen.add(ed.source)
    for e in case["entities"]:
        if e.entity_id in seen:
            continue
        if e.attributes.get("owner") == "interviewee":
            owned.setdefault(iv, []).append(e)
    return owned


def _rel_to_interviewee(case):
    """person entity_id -> relationship word to the interviewee (from RELATED_TO)."""
    iv = case["info"]["interviewee"].entity_id
    m = {}
    for ed in case["edges"]:
        if _relname(ed) != "RELATED_TO":
            continue
        if ed.source == iv:
            m.setdefault(ed.target, ed.detail)
        elif ed.target == iv:
            m.setdefault(ed.source, ed.detail)
    return m


def _owned_chip(e):
    """Render one owned attribute entity (age / DOB / occupation / identifier)."""
    a = e.attributes
    if e.category == "AGE":
        v = a.get("value", a.get("suggested_value"))
        txt = f"age {v}" if v is not None else f"age &ldquo;{escape(e.mentions[0].text)}&rdquo;"
        # `suggested_approximate` was read here and NEVER WRITTEN: `_legacy_mirror`
        # has no branch for `approximate`, so the alternative was dead. The resolved
        # attribute is the value the second line actually settled on.
        if a.get("approximate"):
            txt += " (approx)"
        return _chip(txt, "age")
    if e.category == "DATE_OF_BIRTH":
        v = a.get("resolved_value") or a.get("suggested_value") or e.mentions[0].text
        return _chip(f"DOB {escape(str(v))}", "date")
    if e.category == "OCCUPATION":
        return _chip(f"occupation: {escape(str(a.get('occupation', e.mentions[0].text)))}", "occ")
    label = _ID_LABEL.get(e.category, e.category)
    return _chip(f"{escape(label)}: {escape(e.mentions[0].text)}", "id")


def _person_card(e, case, owned, rel_map, is_iv=False) -> str:
    a = e.attributes
    chips, sugg, other = [], [], []

    # ---- identity / demographics (rule-known) ----
    rel = rel_map.get(e.entity_id)
    if is_iv:
        chips.append(_chip("role: interviewee", "ok"))
    elif rel:
        chips.append(_chip(f"relationship: {escape(str(rel))}", "rel-chip"))

    if a.get("gender"):
        g = _chip(f"gender {escape(str(a['gender']))}", "g")
        if a.get("gender_confirmed"):
            g = _chip(f"gender {escape(str(a['gender']))} &check;", "g ok")
        chips.append(g)
    if a.get("given_name"):
        chips.append(_chip(f"given name: {escape(str(a['given_name']))}", "faint"))
    if a.get("surname"):
        chips.append(_chip(f"surname: {escape(str(a['surname']))}", "faint"))

    # ethnicity (LLM-only; no rule source) -> show prominently, tagged by basis
    eth = a.get("suggested_ethnicity")
    if eth:
        basis = a.get("ethnicity_basis", "")
        tag = f" ({basis}{', low conf' if a.get('ethnicity_confidence') == 'low' else ''})" if basis else ""
        chips.append(_chip(f"ethnicity: {escape(str(eth))}{escape(tag)}", "eth"))
        if a.get("ethnicity_evidence"):
            sugg.append(f"ethnicity evidence: &ldquo;{escape(str(a['ethnicity_evidence']))}&rdquo;")

    # owned age / DOB / occupation / identifiers
    for oe in owned.get(e.entity_id, []):
        chips.append(_owned_chip(oe))

    # replace / redaction decision
    keep = a.get("replace", True) is False
    if e.subtype == "PUBLIC_FIGURE" or keep:
        chips.append(_chip("replace: no (kept)", "warn"))
        if a.get("public_figure_cosign") is True:
            chips.append(_chip("public figure &check; LLM co-signed", "ok"))
    else:
        chips.append(_chip("replace: yes", ""))

    # ---- LLM suggestions ----
    if a.get("suggested_gender"):
        sugg.append(f"gender? {escape(str(a['suggested_gender']))}")
    if a.get("suggested_role"):
        sugg.append(f"role &ldquo;{escape(str(a['suggested_role']))}&rdquo;")
    if a.get("suggested_subtype"):
        conf = a.get("suggested_subtype_confidence")
        sugg.append(f"maybe {escape(str(a['suggested_subtype']).lower())}"
                    + (f" ({escape(str(conf))} conf)" if conf else ""))
    # `candidate_public_figure` used to be read here. Nothing writes it: the
    # public-figure decision became the `replace` field's `safe_direction` policy plus
    # its two checkers, so what a reviewer needs is the RESOLUTION, not a stale
    # advisory key. Read it from provenance instead, where the answer actually lives.
    _rep = (getattr(e, "provenance", None) or {}).get("replace")
    if _rep is not None and _rep.action == "conflict":
        sugg.append(f"redaction disputed: {escape(_rep.reason[:90])}")
    if a.get("suggested_merge_with"):
        sugg.append(f"maybe same person as {escape(str(a['suggested_merge_with']))}")
    if a.get("suggested_relation"):
        sr = a["suggested_relation"]
        if isinstance(sr, dict):
            sugg.append(f"maybe &lsquo;{escape(str(sr.get('detail', '')))}&rsquo; "
                        f"with {escape(str(sr.get('with', '')))}")
    if a.get("merge_evidence"):
        sugg.append(f"merged on: &ldquo;{escape(str(a['merge_evidence']))}&rdquo;")

    # ---- catch-all: any attribute we didn't explicitly render, so nothing hides ----
    for k, v in a.items():
        if k in _HANDLED_ATTR_KEYS or v is None or v == "":
            continue
        other.append(f"{escape(k.replace('_', ' '))}: {escape(str(v))}")

    sug_html = (f"<div class='sug'>LLM: {' &middot; '.join(sugg)}</div>" if sugg else "")
    other_html = (f"<div class='other'>{' &middot; '.join(other)}</div>" if other else "")
    note = f"<div class='note'>&#9873; {escape(e.review_reason)}</div>" if e.needs_review else ""

    if is_iv:
        header_name, cat = "Interviewee", "PERSON &middot; you (speaker)"
        forms = "no named mention &mdash; first-person &ldquo;I / me / my&rdquo;"
    else:
        header_name = _pname(e)
        cat = "PERSON" + (f" &middot; {e.subtype}" if e.subtype else "")
        forms = " / ".join(escape(f) for f in e.sorted_mentions)

    card_cls = "card" + (" iv" if is_iv else "") + (" flag" if e.needs_review else "")
    return (
        f"<div class='{card_cls}'>"
        f"<div class='nm'>{escape(header_name)}<span class='cat'>{cat}</span></div>"
        f"<div class='forms'>{forms}</div>"
        f"<div class='chips'>{''.join(chips)}</div>{sug_html}{other_html}{note}</div>")


def stage_entities(case) -> str:
    iv = case["info"]["interviewee"].entity_id
    owned = _owned_by_person(case)
    rel_map = _rel_to_interviewee(case)

    interviewee = next((e for e in case["entities"] if e.entity_id == iv), None)
    people = [e for e in case["entities"]
              if e.category == "PERSON" and e.entity_id != iv and e.mentions]

    cards = []
    if interviewee is not None:
        cards.append(_person_card(interviewee, case, owned, rel_map, is_iv=True))
    for e in people:
        cards.append(_person_card(e, case, owned, rel_map))

    body = f"<div class='cards'>{''.join(cards)}</div>"
    return section("05", "Resolve entities &amp; attributes", body,
                   "Everything known about each person &mdash; the interviewee (highlighted) "
                   "and everyone they mention: demographics, owned age / DOB / occupation / "
                   "identifiers, relationship, and the redaction decision. Rule-known values "
                   "are solid chips; the &ldquo;LLM:&rdquo; line is advisory suggestions.")


# ---------- 6 · direct identifiers ----------
def stage_identifiers(case) -> str:
    ids = [e for e in case["entities"] if e.category in _ID_CATS]
    if not ids:
        return section("06", "Direct identifiers",
                       "<p class='muted'>No direct identifiers detected in this transcript.</p>",
                       "Phones, emails, IDs, handles, occupations &mdash; typed and passed "
                       "through so surrogate generation replaces them.")
    cards = []
    for e in sorted(ids, key=lambda x: _ID_CATS.index(x.category)):
        a = e.attributes
        label = _ID_LABEL.get(e.category, e.category)
        if e.subtype and e.subtype not in (e.category, label.upper()):
            label += f" &middot; {e.subtype}"
        chips = [f"<span class='chip g'>{label}</span>",
                 f"<span class='chip'>replace: {'yes' if a.get('replace', True) else 'no'}</span>"]
        if a.get("owner"):
            chips.append(f"<span class='chip'>owner: {escape(str(a['owner']))}</span>")
        if a.get("identifying"):
            chips.append("<span class='chip warn'>possibly identifying</span>")
        note = f"<div class='note'>&#9873; {escape(e.review_reason)}</div>" if e.needs_review else ""
        cards.append(f"<div class='card{' flag' if e.needs_review else ''}'>"
                     f"<div class='nm'>{escape(_pname(e))}</div>{''.join(chips)}{note}</div>")
    body = f"<div class='cards'>{''.join(cards)}</div>"
    return section("06", "Direct identifiers", body,
                   f"{len(ids)} identifier(s): typed by rule, with owner / identifying-ness "
                   f"judged by the LLM. All kept for replacement in surrogate generation.")


# ---------- 7 · graph ----------
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

    # render at natural size (capped), centered, responsive -- not stretched to
    # the full container width
    disp_w = min(W, 560.0)
    disp_h = disp_w * H / W
    parts = [f"<svg viewBox='0 0 {W:.0f} {H:.0f}' width='{disp_w:.0f}' height='{disp_h:.0f}' "
             f"role='img' style='max-width:100%;height:auto;display:block;margin:0 auto' "
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


def _owner_note(e) -> str:
    """`owner` and its blocking status, for an AGE / DATE entity.

    These entities were rendered as a bare "text -> value" pill, so NONE of their
    ownership decisions appeared anywhere on the page -- including the blocking ones,
    which are precisely the rows a human has to settle before surrogates are minted.
    An age nobody can attribute is not a resolved age.
    """
    res = (getattr(e, "provenance", None) or {}).get("owner")
    owner = e.attributes.get("owner")
    if owner:
        cls = "own-iv" if owner == "interviewee" else "own-other"
        return f"<i class='own {cls}'>{escape(str(owner))}</i>"
    if res is not None and getattr(res, "blocking", False):
        return "<i class='own own-block'>owner unresolved &mdash; BLOCKING</i>"
    if res is not None and res.action in ("reject", "conflict"):
        return "<i class='own own-none'>owner unresolved</i>"
    return ""


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

    # PLACE REDACTION. `replace_location` decides whether a place name survives into
    # the surrogate transcript, and nothing on this page showed it: a reviewer saw
    # every person redacted and could not tell that "Red Jacket" was too, or that
    # "Washington" was kept. A hamlet plus an age plus an occupation re-identifies a
    # household, so this belongs beside the people, not nowhere.
    place_rows = []
    for e in case["entities"]:
        if e.category not in ("LOCATION", "INSTITUTION"):
            continue
        keep = e.attributes.get("replace", True) is False
        res = (getattr(e, "provenance", None) or {}).get("replace_location")
        why = ""
        if res is not None:
            n = len(res.checks_passed)
            why = f"{res.action} &middot; {escape(res.source or 'unresolved')}"
            if n:
                why += f" &middot; {n} check(s)"
            elif keep:
                # Only a KEEP needs proving, so only a KEEP is alarming when nothing
                # checked it. Saying "unverified" against `replace=True` would flag the
                # safe direction, where the keep-gates correctly do not apply.
                why += " &middot; NOTHING VERIFIED THIS KEEP"
        place_rows.append(
            f"<div class='locrow'><span class='pill loc'>{escape(_pname(e))}</span>"
            f"<span class='ptype'>{escape(str(e.subtype or 'untyped').lower())}</span>"
            f"<span class='pill {'keepp' if keep else 'repl'}'>"
            f"{'kept' if keep else 'replace'}</span>"
            f"<span class='pwhy'>{why}</span></div>")
    place_html = (f"<div class='sub-h'>Place names &mdash; redaction decision</div>"
                  f"{''.join(place_rows)}") if place_rows else ""

    chips = []
    for e in case["entities"]:
        if e.category.startswith("DATE"):
            v = e.attributes.get("resolved_value") or "unresolved"
            # `shiftable` and `approximate` are what the date-shifter reads. Both were
            # arbitrated and neither was ever displayed, so a date pinned as
            # non-shiftable looked identical to one that can move.
            marks = []
            if e.attributes.get("shiftable") is False:
                marks.append("fixed")
            if e.attributes.get("approximate"):
                marks.append("approx")
            tail = f" <i class='dmark'>{' &middot; '.join(marks)}</i>" if marks else ""
            chips.append(f"<span class='pill date'>{escape(e.mentions[0].text)} &rarr; "
                         f"{escape(str(v))}{tail}{_owner_note(e)}</span>")
        elif e.category == "AGE":
            v = e.attributes.get("value")
            tail = " <i class='dmark'>approx</i>" if e.attributes.get("approximate") else ""
            chips.append(f"<span class='pill age'>{escape(e.mentions[0].text)} &rarr; "
                         f"{escape(str(v))}{tail}{_owner_note(e)}</span>")
    dt_html = (f"<div class='sub-h'>Dates &amp; ages resolved</div><div class='pills'>{''.join(chips)}</div>"
               if chips else "")
    return loc_html + place_html + dt_html


_GRAPH_LEGEND = (
    "<div class='glegend'>"
    "<span><i style='background:#2C4A6E'></i>interviewee</span>"
    "<span><i style='background:#5B7FA6'></i>family</span>"
    "<span><i style='background:#9CB6D2'></i>professional</span>"
    "<span><i style='background:#AEB3BA'></i>public figure</span>"
    "<span class='ring'>&#9711; red ring = flagged for review</span></div>"
)


def stage_graph(case) -> str:
    body = f"<div class='graph'>{_graph_svg(case)}{_GRAPH_LEGEND}{_places_times(case)}</div>"
    return section("07", "Entity graph", body,
                   "Interviewee at center; each line is a relationship.")


_STEPS = ["Detect", "Cluster", "Coref", "Relations", "Entities", "Identifiers", "Graph"]


def _stepper() -> str:
    chips = [f"<span class='st'><b>{i:02d}</b>{s}</span>" for i, s in enumerate(_STEPS, 1)]
    return "<div class='stepper'>" + "<span class='sep'>&rarr;</span>".join(chips) + "</div>"


def transcript_panel(case, metrics=None) -> str:
    head = metrics_grid(metrics) if metrics else ""
    stages = (stage_detect(case) + stage_cluster(case) + stage_coref(case)
              + stage_relations(case) + stage_entities(case) + stage_identifiers(case)
              + stage_graph(case))
    return head + _stepper() + f"<div class='stages'>{stages}</div>"


CSS = """
:root{--ink:#1f2328;--muted:#5b636c;--faint:#9aa3ad;--line:#e7e9ec;--panel:#f6f7f9;
  --accent:#3b6ea5;--accentbg:#eef3f9;}
*{box-sizing:border-box;}
body{margin:0;background:#fbfbfc;color:var(--ink);-webkit-font-smoothing:antialiased;
  font-family:'Iowan Old Style','Palatino Linotype',Palatino,'Book Antiqua',Georgia,'Times New Roman',serif;
  font-size:16px;line-height:1.6;}
.wrap{max-width:960px;margin:0 auto;padding:40px 28px 100px;}
h1{font-size:24px;font-weight:600;letter-spacing:-0.3px;margin:0 0 4px;}
.lede{color:var(--muted);font-size:14.5px;margin:0 0 8px;max-width:72ch;}
h3{font-size:15.5px;font-weight:600;margin:0;letter-spacing:-0.1px;}
.muted{color:var(--muted);} .faint{color:var(--faint);}

/* tabs */
.tabs{display:flex;flex-wrap:wrap;gap:4px;border-bottom:1px solid var(--line);margin:24px 0 26px;}
.tab{appearance:none;background:none;border:none;cursor:pointer;font:inherit;font-size:13.5px;
  color:var(--muted);padding:9px 14px 11px;border-bottom:2px solid transparent;margin-bottom:-1px;text-align:left;}
.tab .t-sub{display:block;font-size:11px;color:var(--faint);margin-top:1px;}
.tab:hover{color:var(--ink);}
.tab.active{color:var(--accent);border-bottom-color:var(--accent);}
.tab.active .t-sub{color:var(--muted);}
.panel{display:none;} .panel.active{display:block;}

/* metrics */
.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:24px;}
.m-tile{background:#fff;border:1px solid var(--line);border-radius:12px;padding:14px 16px;}
.m-val{font-size:23px;font-weight:600;letter-spacing:-0.5px;line-height:1.1;}
.m-val.ok{color:#2f7a3d;} .m-val.bad{color:#b3261e;}
.m-lab{font-size:12px;color:var(--muted);margin-top:4px;}
.m-sub{font-size:10.5px;color:var(--faint);margin-top:2px;}
@media(max-width:760px){.metrics{grid-template-columns:repeat(2,1fr);}}

/* stepper: at-a-glance pipeline sequence */
.stepper{display:flex;flex-wrap:wrap;align-items:center;gap:6px;margin:0 0 20px;}
.stepper .st{background:#fff;border:1px solid var(--line);border-radius:20px;padding:4px 12px;
  font-size:12px;color:var(--muted);}
.stepper .st b{color:var(--accent);font-weight:600;margin-right:5px;font-size:11px;}
.stepper .sep{color:var(--faint);font-size:12px;}

/* stages as a clean vertical sequence of cards */
.stages{display:flex;flex-direction:column;gap:18px;}
.stage{background:#fff;border:1px solid var(--line);border-radius:14px;padding:20px 22px;}
.s-head{display:flex;align-items:center;gap:11px;}
.s-num{flex:0 0 auto;width:26px;height:26px;border-radius:50%;background:var(--accentbg);color:var(--accent);
  font-size:11.5px;font-weight:600;display:flex;align-items:center;justify-content:center;}
.s-note{color:var(--muted);font-size:13px;margin:6px 0 0 37px;}

/* 01 detect */
.excerpt{background:var(--panel);border-radius:10px;padding:16px 18px;line-height:1.95;font-size:14.5px;margin-top:14px;}
mark{padding:1px 5px;border-radius:4px;font-weight:500;}
.legend{display:flex;gap:16px;margin-top:12px;font-size:12px;color:var(--muted);}
.lg::before{content:"";display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:6px;vertical-align:1px;}
.lg-p::before{background:#5B7FA6;}.lg-l::before{background:#6f9a4a;}.lg-d::before{background:#c99a4a;}.lg-a::before{background:#7d72c4;}

/* 02/03 lists, 04 relations */
.deltas,.rels{list-style:none;padding:0;margin:14px 0 0;}
.deltas li,.rels li{padding:8px 0;border-bottom:1px solid var(--line);font-size:14px;}
.deltas li:last-child,.rels li:last-child{border-bottom:none;}
.tag{display:inline-block;font-size:11px;padding:1px 8px;border-radius:20px;margin-right:6px;}
.tag.ok{background:#e7f2e9;color:#2f6b3b;} .tag.warn{background:#fbeceb;color:#93362f;}
.src{display:inline-block;font-size:9.5px;font-weight:600;letter-spacing:.5px;text-transform:uppercase;
  padding:1px 6px;border-radius:4px;margin-right:7px;}
.src.rule{background:#eceef1;color:#5b636c;} .src.llm{background:var(--accentbg);color:var(--accent);}
.rel{display:inline-block;font-size:12px;color:var(--accent);background:var(--accentbg);padding:1px 8px;border-radius:20px;margin:0 4px;}
.ev{display:block;color:var(--faint);font-size:12px;margin-top:2px;font-style:italic;}

/* 05 entities */
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(248px,1fr));gap:10px;margin-top:14px;}
.card{background:var(--panel);border:1px solid var(--line);border-radius:11px;padding:12px 14px;}
.card.flag{border-left:3px solid #c0574f;}
.card.iv{background:#eef3f9;border:1px solid #cdddef;border-left:3px solid #2C4A6E;grid-column:1/-1;}
.card .nm{font-size:14.5px;font-weight:600;}
.card .nm .cat{font-weight:400;font-size:11px;color:var(--muted);margin-left:4px;}
.card .forms{font-size:11.5px;color:var(--muted);margin:1px 0 8px;}
.card .chips{display:flex;flex-wrap:wrap;}
.chip{display:inline-block;font-size:11px;padding:1px 8px;border-radius:20px;margin:0 4px 4px 0;background:#eceef1;color:#4b5158;}
.chip.g{background:#E9F0FA;color:#234E86;}
.chip.ok{background:#e7f2e9;color:#2f6b3b;}
.chip.warn{background:#fbeceb;color:#8a2f28;}
.chip.eth{background:#F3E8F6;color:#6b2f7c;}
.chip.occ{background:#E4F1F1;color:#245b5b;}
.chip.id{background:#FBEAEA;color:#7C2222;}
.chip.age{background:#EFEDFA;color:#413593;}
.chip.date{background:#F8EFDD;color:#6A4310;}
.chip.rel-chip{background:#E9F0FA;color:#234E86;}
.chip.faint{background:#eef0f2;color:#7c828b;}
.card .sug{font-size:11.5px;color:#6A4310;margin-top:5px;}
.card .other{font-size:11px;color:#8a8f98;margin-top:5px;line-height:1.5;}
.card .note{font-size:11px;color:#8a2f28;margin-top:5px;line-height:1.5;}

/* 06 graph */
.graph{padding-top:8px;}
.glegend{display:flex;flex-wrap:wrap;gap:14px;justify-content:center;margin-top:14px;font-size:11.5px;color:var(--muted);}
.glegend i{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px;}
.glegend .ring{color:#8a2f28;}
.sub-h{font-size:11.5px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:0.5px;margin:20px 0 9px;}
.locrow{display:flex;align-items:center;gap:9px;margin-bottom:7px;}
.arrow{font-size:12px;color:var(--faint);}
.pills{display:flex;flex-wrap:wrap;gap:7px;}
.pill{font-size:12px;padding:3px 10px;border-radius:20px;}
.pill.loc{background:#EAF2E0;color:#315915;}.pill.date{background:#F8EFDD;color:#6A4310;}.pill.age{background:#EFEDFA;color:#413593;}
.pill.repl{background:#FBECEA;color:#7C2222;}.pill.keepp{background:#EAF3EC;color:#2F6B3B;}
.ptype{font-size:11px;color:var(--faint);min-width:82px;}
.pwhy{font-size:11px;color:var(--muted);}
.dmark{font-style:normal;font-size:10.5px;opacity:0.75;margin-left:4px;}
.own{font-style:normal;font-size:10.5px;margin-left:6px;padding:1px 6px;border-radius:20px;}
.own-iv{background:#2C4A6E;color:#fff;}
.own-other{background:#fff;color:var(--muted);border:1px solid var(--line);}
.own-none{background:#fff;color:#6A4310;border:1px solid #C99A4A;}
.own-block{background:#7C2222;color:#fff;}
.foot{font-size:12.5px;color:var(--muted);border-top:1px solid var(--line);padding-top:16px;margin-top:24px;}
"""
