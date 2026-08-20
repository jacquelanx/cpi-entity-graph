"""
Panels 05-06: who the interviewee is, and one card per named person.

PURPOSE
    Render the two person-centred panels. The interviewee panel is the important
    one -- interviewee-only de-identification runs entirely off that node, so its
    fields are shown with full provenance. The people panel is a grid of compact
    cards, one per named third party.

FIT
    Sits on `primitives.py` and `provenance.py`; assembled by
    `page.transcript_panel`. The three module-level helpers at the top build the
    cross-references both panels need -- who owns which identifier, and who is
    related to the speaker how.
"""

from __future__ import annotations

from html import escape
from .primitives import _DATE_CATS, _ID_LABEL, _chip, _names_map, _pname, _relname, _val, section
from .provenance import _action_badge, _checks_html, _flag_html, _prov_details, _prov_of


# ---------- shared: owned attributes ----------
def _owned_by_person(case):
    """person entity_id -> [owned attribute entities], from ATTRIBUTE_OF edges.

    Two sources, in this order. First the ATTRIBUTE_OF EDGES, which is the
    authoritative record. Then, as a fallback, any entity carrying
    `owner == "interviewee"` in its attributes that no edge already covered --
    `seen` tracks which those are. The fallback matters because an owner can be
    resolved on the attribute without an edge having been built for it.
    """
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
    """`{person entity_id: relation word}` for everyone tied to the speaker.

    Checks both edge directions, since a RELATED_TO edge may point either way, and
    uses `setdefault` so the FIRST relation found for a person wins if they are
    described more than one way.
    """
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
    """A compact label for one owned attribute, for display on a person card.

    Formats by category -- an age as "age 68 (approx)", a date as its resolved
    value, an identifier by its `_ID_LABEL` name -- and falls back to the raw
    mention text when no value was resolved, so an unresolved span is still
    visible rather than silently absent.
    """
    a = e.attributes
    if e.category == "AGE":
        v = a.get("value")
        txt = (f"age {v}" if v is not None
               else f"age &ldquo;{escape(e.mentions[0].text)}&rdquo;")
        if a.get("approximate"):
            txt += " (approx)"
        return _chip(txt, "age")
    if e.category == "DATE_OF_BIRTH":
        v = a.get("resolved_value") or e.mentions[0].text
        return _chip(f"DOB {escape(str(v))}", "date")
    if e.category == "OCCUPATION":
        return _chip(f"occupation: {escape(str(a.get('occupation', e.mentions[0].text)))}",
                     "occ")
    label = _ID_LABEL.get(e.category, e.category)
    return _chip(f"{escape(label)}: {escape(e.mentions[0].text)}", "id")


# ---------- 05 · the interviewee ----------
def stage_interviewee(case) -> str:
    """Panel 05: the interviewee node, field by field, with full provenance.

    The most important panel on the page: interviewee-only de-identification runs
    entirely off this node, so each field is shown with its outcome badge, the
    deterministic checks that examined it, and any blocking mark. The nested
    `field_row` helper renders one such row.
    """
    """The subject of de-identification, on their own. Everything the surrogate
    generator needs about the one person we are actually protecting, plus how each
    of those values was decided."""
    info = case["info"]
    iv = info["interviewee"]
    a = iv.attributes
    prov = _prov_of(iv)
    names = _names_map(case)
    owned = _owned_by_person(case).get(iv.entity_id, [])
    rel_map = _rel_to_interviewee(case)
    ents = {e.entity_id: e for e in case["entities"]}

    def field_row(label, value, res_key, extra=""):
        """One field row on the interviewee panel.

        `res_key` names the provenance entry to render alongside the value, so the
        row shows the decision as well as the answer.
        """
        res = prov.get(res_key)
        state = _action_badge(res) if res is not None else ""
        checks = _checks_html(res) if res is not None else ""
        why = f"<div class='why'>{escape(res.reason)}</div>" if res is not None else ""
        return (f"<tr class='{'blk' if (res is not None and res.blocking) else ''}'>"
                f"<td class='f'>{label}</td><td class='v big'>{value}{extra}</td>"
                f"<td>{state}</td><td class='ck-cell'>{checks}{why}</td></tr>")

    named = " / ".join(escape(f) for f in iv.sorted_mentions) or \
        "<i class='none'>never named &mdash; first-person &ldquo;I / me / my&rdquo;</i>"
    rows = [
        field_row("name in transcript", named, "interviewee_identity"),
        field_row("gender", _val(a.get("gender")), "interviewee_gender"),
        field_row("ethnicity", _val(a.get("ethnicity")), "ethnicity",
                  (f"<div class='ev2'>&ldquo;"
                   f"{escape(str(a['ethnicity_evidence']))}&rdquo;</div>"
                   if a.get("ethnicity_evidence") else "")),
    ]
    if a.get("given_name") or "given_name" in prov:
        rows.append(field_row("given name", _val(a.get("given_name")), "given_name"))
    if a.get("surname") or "surname" in prov:
        rows.append(field_row("surname", _val(a.get("surname")), "surname"))

    # owned PII -- the rows the surrogate generator mints from
    own_rows = []
    for oe in sorted(owned, key=lambda e: e.category):
        oprov = _prov_of(oe)
        ores = oprov.get("owner")
        v = (oe.attributes.get("value") if oe.category == "AGE"
             else oe.attributes.get("resolved_value") if oe.category in _DATE_CATS
             else None)
        own_rows.append(
            f"<tr><td class='f'>{escape(_ID_LABEL.get(oe.category, oe.category))}</td>"
            f"<td class='v'>{escape(oe.mentions[0].text)}"
            + (f" <i class='arrow2'>&rarr; {_val(v)}</i>" if v is not None else "")
            + f"</td><td>{'<span class=\"pill repl\">replace</span>' if oe.attributes.get('replace') is not False else '<span class=\"pill keepp\">kept</span>'}</td>"
              f"<td class='ck-cell'>{_checks_html(ores) if ores is not None else ''}</td></tr>")
    own_html = (
        "<div class='sub-h'>Identifiers, ages and dates that belong to the subject"
        "</div><table class='prov'><thead><tr><th>kind</th><th>span</th>"
        "<th>redaction</th><th>how ownership was proved</th></tr></thead><tbody>"
        + "".join(own_rows) + "</tbody></table>"
        if own_rows else
        "<div class='sub-h'>Owned identifiers</div>"
        "<p class='muted'>Nothing in this transcript was attributed to the subject.</p>")

    # family / social ties
    kin = []
    for eid, detail in rel_map.items():
        other = ents.get(eid)
        if other is None:
            continue
        kin.append(f"<span class='pill kin'>{escape(detail)}: "
                   f"{escape(_pname(other))}</span>")
    kin_html = (f"<div class='sub-h'>Ties to other people</div>"
                f"<div class='pills'>{''.join(kin)}</div>" if kin else "")

    blocking = [b for b in info.get("blocking", []) if b[0] == iv.entity_id]
    blk_html = ""
    if blocking:
        items = "".join(f"<li><b>{escape(f)}</b> &mdash; {escape(why)}</li>"
                        for _e, f, why in blocking)
        blk_html = (f"<div class='blkbox'><b>{len(blocking)} field(s) block surrogate "
                    f"generation for this person.</b><ul>{items}</ul></div>")

    flag = _flag_html(iv)

    body = (f"{blk_html}"
            f"<table class='prov iv-t'><thead><tr><th>field</th><th>value</th>"
            f"<th>outcome</th><th>deterministic checks</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
            f"{own_html}{kin_html}{flag}")
    return section("05", "The interviewee — the person being de-identified", body,
                   "Everything known about the subject, and how each value was "
                   "settled. This is the node interviewee-only surrogate generation "
                   "runs on: a blocking row here stops the next stage.")


# ---------- 06 · other people ----------
_HANDLED_ATTR_KEYS = {
    "given_name", "surname", "gender", "gender_confirmed", "replace", "role",
    "subtype", "owner", "identifying", "suggested_gender", "suggested_role",
    "suggested_subtype", "suggested_subtype_confidence", "suggested_merge_with",
    "ethnicity", "suggested_ethnicity", "ethnicity_basis", "ethnicity_evidence",
    "ethnicity_confirmed", "suggested_relation", "merge_evidence",
}


def _person_card(e, case, owned, rel_map, names) -> str:
    """One compact card for a named third party.

    Shows the name, subtype, relation to the speaker, inferred attributes, owned
    identifiers and review flags. Provenance uses the STACKED layout
    (`_prov_list`), because a five-column table does not fit a ~300px card -- see
    that function.
    """
    a = e.attributes
    chips, sugg, other = [], [], []

    rel = rel_map.get(e.entity_id)
    if rel:
        chips.append(_chip(f"relationship: {escape(str(rel))}", "rel-chip"))
    if a.get("role"):
        chips.append(_chip(f"role: {escape(str(a['role']))}", "occ"))
    if a.get("gender"):
        g = f"gender {escape(str(a['gender']))}"
        chips.append(_chip(g + (" &check;" if a.get("gender_confirmed") else ""),
                           "g ok" if a.get("gender_confirmed") else "g"))
    if a.get("given_name"):
        chips.append(_chip(f"given name: {escape(str(a['given_name']))}", "faint"))
    if a.get("surname"):
        chips.append(_chip(f"surname: {escape(str(a['surname']))}", "faint"))

    # the resolved value, not the legacy mirror key: a rule-only ethnicity never
    # writes `suggested_ethnicity`, so reading the mirror hid it entirely
    eth = a.get("ethnicity") or a.get("suggested_ethnicity")
    if eth:
        basis = a.get("ethnicity_basis", "")
        chips.append(_chip(f"ethnicity: {escape(str(eth))}"
                           + (f" ({escape(str(basis))})" if basis else ""), "eth"))
        if a.get("ethnicity_evidence"):
            sugg.append(f"ethnicity evidence: &ldquo;"
                        f"{escape(str(a['ethnicity_evidence']))}&rdquo;")

    for oe in owned.get(e.entity_id, []):
        chips.append(_owned_chip(oe))

    keep = a.get("replace", True) is False
    chips.append(_chip("replace: no (name kept)", "warn") if keep
                 else _chip("replace: yes", ""))

    if a.get("suggested_merge_with"):
        sugg.append(f"maybe the same person as "
                    f"{escape(str(a['suggested_merge_with']))}")
    if a.get("suggested_relation"):
        sr = a["suggested_relation"]
        if isinstance(sr, dict):
            sugg.append(f"unverified relation &lsquo;"
                        f"{escape(str(sr.get('detail', '')))}&rsquo; with "
                        f"{escape(str(sr.get('with', '')))}")
    if a.get("merge_evidence"):
        sugg.append(f"merged on: &ldquo;{escape(str(a['merge_evidence']))}&rdquo;")

    for k, v in a.items():
        if k in _HANDLED_ATTR_KEYS or v is None or v == "":
            continue
        other.append(f"{escape(k.replace('_', ' '))}: {escape(str(v))}")

    sug_html = f"<div class='sug'>{' &middot; '.join(sugg)}</div>" if sugg else ""
    other_html = f"<div class='other'>{' &middot; '.join(other)}</div>" if other else ""
    note = _flag_html(e)

    cat = "PERSON" + (f" &middot; {escape(str(e.subtype))}" if e.subtype else "")
    forms = " / ".join(escape(f) for f in e.sorted_mentions)
    card_cls = "card" + (" flag" if e.needs_review else "")
    return (f"<div class='{card_cls}'>"
            f"<div class='nm'>{escape(_pname(e))}<span class='cat'>{cat}</span></div>"
            f"<div class='forms'>{forms}</div>"
            f"<div class='chips'>{''.join(chips)}</div>{sug_html}{other_html}{note}"
            f"{_prov_details(e, names, stacked=True)}</div>")


def stage_people(case) -> str:
    """Panel 06: a grid of cards, one per named person other than the interviewee.

    Builds the three cross-reference maps once (`_owned_by_person`,
    `_rel_to_interviewee`, `_names_map`) and passes them to every card, rather than
    each card re-deriving them from the edge list.
    """
    iv = case["info"]["interviewee"].entity_id
    owned, rel_map, names = _owned_by_person(case), _rel_to_interviewee(case), _names_map(case)
    people = [e for e in case["entities"]
              if e.category == "PERSON" and e.entity_id != iv and e.mentions]
    if not people:
        return section("06", "Other people named in the interview",
                       "<p class='muted'>Nobody else is named.</p>")
    cards = [_person_card(e, case, owned, rel_map, names) for e in people]
    kept = sum(1 for e in people if e.attributes.get("replace") is False)
    body = f"<div class='cards'>{''.join(cards)}</div>"
    return section("06", "Other people named in the interview", body,
                   f"{len(people)} people &middot; {kept} kept unredacted (public "
                   f"figures) &middot; {len(people) - kept} replaced. Open a decision "
                   f"record to see which layer settled each field and what checked it.")
