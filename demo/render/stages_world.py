"""
Panels for the non-person entities: places and their
hierarchy, dates and ages, and direct identifiers.
"""

from __future__ import annotations

from html import escape
from .primitives import _DATE_CATS, _ID_CATS, _ID_LABEL, _names_map, _pname, _relname, _val, section
from .provenance import _checks_html, _prov_of


# ---------- 07 · places ----------
def stage_places(case) -> str:
    ents = {e.entity_id: e for e in case["entities"]}
    places = [e for e in case["entities"]
              if e.category in ("LOCATION", "INSTITUTION")]
    if not places:
        return section("07", "Places", "<p class='muted'>No places detected.</p>")

    parent_of = {ed.source: ed.target for ed in case["edges"]
                 if _relname(ed) == "LOCATED_IN"}
    names = _names_map(case)

    rows = []
    for e in sorted(places, key=lambda x: (x.attributes.get("replace") is False,
                                           _pname(x))):
        keep = e.attributes.get("replace", True) is False
        res = _prov_of(e).get("replace_location")
        parent = parent_of.get(e.entity_id)
        rows.append(
            f"<tr><td class='v'>{escape(_pname(e))}</td>"
            f"<td class='ptype'>{escape(str(e.subtype or 'untyped').lower())}</td>"
            f"<td class='ptype'>"
            + (escape(names.get(parent, "")) if parent else "<i class='none'>&mdash;</i>")
            + f"</td><td><span class='pill {'keepp' if keep else 'repl'}'>"
              f"{'kept' if keep else 'replace'}</span></td>"
              f"<td class='ck-cell'>"
            + (_checks_html(res, quiet=not keep) if res is not None else "")
            + (f"<div class='why'>{escape(res.reason)}</div>" if res is not None else "")
            + "</td></tr>")

    tree = _location_tree(case)
    body = ("<table class='prov'><thead><tr><th>place</th><th>type</th>"
            "<th>inside</th><th>redaction</th><th>how the decision was checked</th>"
            "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>" + tree)
    kept = sum(1 for e in places if e.attributes.get("replace") is False)
    return section("07", "Places — typing, hierarchy and redaction", body,
                   f"{len(places)} places &middot; {kept} kept. A place is kept only "
                   f"when it is coarse enough that naming it cannot single out a "
                   f"household, and only when the geographic type behind that "
                   f"decision was itself verified.")


def _location_tree(case) -> str:
    ents = {e.entity_id: e for e in case["entities"]}
    parent_of = {ed.source: ed.target for ed in case["edges"]
                 if _relname(ed) == "LOCATED_IN"}
    if not parent_of:
        return ""
    children = {}
    for c, p in parent_of.items():
        children.setdefault(p, []).append(c)
    roots = sorted({p for p in parent_of.values() if p not in parent_of})

    def walk(nid, depth):
        e = ents.get(nid)
        label = escape(_pname(e)) if e else escape(nid)
        typ = escape(str(e.subtype or "").lower()) if e else ""
        out = [f"<div class='trow' style='padding-left:{depth * 22}px'>"
               f"{'<span class=\"tw\">└</span>' if depth else ''}"
               f"<span class='pill loc'>{label}</span>"
               f"<span class='ptype'>{typ}</span></div>"]
        for ch in sorted(children.get(nid, []),
                         key=lambda x: _pname(ents[x]) if x in ents else x):
            out.append(walk(ch, depth + 1))
        return "".join(out)

    return ("<div class='sub-h'>Location hierarchy (LOCATED_IN)</div>"
            + "".join(walk(r, 0) for r in roots))


# ---------- 08 · dates & ages ----------
def stage_dates_ages(case) -> str:
    ents = {e.entity_id: e for e in case["entities"]}
    stated = {}
    for ed in case["edges"]:
        if _relname(ed) == "STATED_WITH":
            stated[ed.source] = ed.target
    owner_of = {}
    for ed in case["edges"]:
        if _relname(ed) == "ATTRIBUTE_OF":
            owner_of[ed.source] = ed.target
    names = _names_map(case)

    def row(e, resolved, value_field):
        a = e.attributes
        prov = _prov_of(e)
        keep = a.get("replace", True) is False
        marks = []
        if a.get("shiftable") is False:
            marks.append("<i class='dmark'>fixed point</i>")
        if a.get("approximate"):
            marks.append("<i class='dmark'>approx</i>")
        anchor = stated.get(e.entity_id)
        if anchor:
            marks.append(f"<i class='dmark'>stated with "
                         f"{escape(_pname(ents[anchor]) if anchor in ents else anchor)}</i>")
        own = a.get("owner")
        own_html = ("<i class='none'>&mdash;</i>" if not own else
                    f"<span class='own {'own-iv' if own == 'interviewee' else 'own-other'}'>"
                    f"{escape(str(own))}</span>")
        ores = prov.get("owner")
        if not own and ores is not None and ores.blocking:
            own_html = "<span class='own own-block'>unresolved &mdash; BLOCKING</span>"
        rres = prov.get("replace_date") or prov.get("replace_age")
        return (f"<tr><td class='v'>{escape(e.mentions[0].text)}</td>"
                f"<td class='ptype'>{escape(e.category.replace('DATE_', '').lower())}</td>"
                f"<td class='v'>{_val(resolved)} {' '.join(marks)}</td>"
                f"<td>{own_html}</td>"
                f"<td><span class='pill {'keepp' if keep else 'repl'}'>"
                f"{'kept' if keep else 'replace'}</span></td>"
                f"<td class='ck-cell'>"
                + (_checks_html(rres, quiet=not keep) if rres is not None else "")
                + (f"<div class='why'>{escape(rres.reason)}</div>"
                   if rres is not None else "")
                + "</td></tr>")

    d_rows = [row(e, e.attributes.get("resolved_value"), "resolved_value")
              for e in case["entities"] if e.category in _DATE_CATS]
    a_rows = [row(e, e.attributes.get("value"), "value")
              for e in case["entities"] if e.category == "AGE"]
    if not d_rows and not a_rows:
        return section("08", "Dates & ages",
                       "<p class='muted'>None detected.</p>")

    head = ("<thead><tr><th>span</th><th>kind</th><th>resolved</th><th>owner</th>"
            "<th>redaction</th><th>how the redaction was checked</th></tr></thead>")
    body = ""
    if d_rows:
        body += ("<div class='sub-h'>Dates</div><table class='prov'>" + head
                 + f"<tbody>{''.join(d_rows)}</tbody></table>")
    if a_rows:
        body += ("<div class='sub-h'>Ages</div><table class='prov'>" + head
                 + f"<tbody>{''.join(a_rows)}</tbody></table>")
    n_fixed = sum(1 for e in case["entities"]
                  if e.category in _DATE_CATS and e.attributes.get("shiftable") is False)
    return section("08", "Dates & ages — resolution, ownership, redaction", body,
                   f"{len(d_rows)} dates ({n_fixed} pinned as fixed public events, so "
                   f"the date-shifter must not move them) and {len(a_rows)} ages. "
                   f"&lsquo;Stated with&rsquo; marks an age whose arithmetic must "
                   f"survive the shift.")


# ---------- 09 · direct identifiers ----------
def stage_identifiers(case) -> str:
    ids = [e for e in case["entities"] if e.category in _ID_CATS]
    if not ids:
        return section("09", "Direct identifiers",
                       "<p class='muted'>No direct identifiers detected.</p>")
    names = _names_map(case)
    _PARTS = ("digits", "local", "domain", "handle", "occupation")
    rows = []
    for e in sorted(ids, key=lambda x: _ID_CATS.index(x.category)):
        a, prov = e.attributes, _prov_of(e)
        label = _ID_LABEL.get(e.category, e.category)
        if e.subtype and e.subtype not in (e.category, label.upper()):
            label += f" &middot; {escape(str(e.subtype))}"
        parts = " ".join(f"<span class='part'>{k}={escape(str(a[k]))}</span>"
                         for k in _PARTS if a.get(k))
        own = a.get("owner")
        own_html = ("<i class='none'>&mdash;</i>" if not own else
                    f"<span class='own {'own-iv' if own == 'interviewee' else 'own-other'}'>"
                    f"{escape(str(own))}</span>")
        ores = prov.get("owner")
        if not own and ores is not None and ores.blocking:
            own_html = "<span class='own own-block'>unresolved &mdash; BLOCKING</span>"
        idg = a.get("identifying")
        idg_html = ("<i class='none'>&mdash;</i>" if idg is None else
                    "<span class='pill repl'>rare enough to identify</span>" if idg
                    else "<span class='pill keepp'>common</span>")
        rows.append(
            f"<tr class='{'blk' if e.needs_review else ''}'>"
            f"<td class='v'>{escape(e.mentions[0].text)}</td>"
            f"<td class='ptype'>{label}</td>"
            f"<td class='parts'>{parts or '<i class=none>&mdash;</i>'}</td>"
            f"<td>{own_html}</td><td>{idg_html}</td>"
            f"<td><span class='pill {'repl' if a.get('replace', True) else 'keepp'}'>"
            f"{'replace' if a.get('replace', True) else 'kept'}</span></td>"
            f"<td class='ck-cell'>{_checks_html(ores) if ores is not None else ''}</td>"
            f"</tr>")
    body = ("<table class='prov'><thead><tr><th>span</th><th>type</th>"
            "<th>normalized parts</th><th>owner</th><th>identifying?</th>"
            "<th>redaction</th><th>ownership checks</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>")
    n_iv = sum(1 for e in ids if e.attributes.get("owner") == "interviewee")
    return section("09", "Direct identifiers", body,
                   f"{len(ids)} identifier(s), {n_iv} belonging to the subject. The "
                   f"normalized parts are what surrogate generation mints a "
                   f"replacement from, so they are regenerated whenever the type is "
                   f"re-verified.")
