"""
Shared HTML renderers for the demo pages (dashboard.py, llm_report.py).
Every function takes a `case` dict from demo_utils.load_case(..., trace=True) and
returns an HTML fragment. CSS lives in the CSS constant so both pages share one
minimalist stylesheet.

The page is a WALKTHROUGH of the pipeline in the order it runs, and its job is to
show everything the pipeline knows -- not a summary of it. Two rules make that
honest rather than overwhelming:

  * every stage renders the RESOLVED value AND the decision behind it. The
    interesting fact about `gender = F` is not the F, it is that the rules were
    silent, the model proposed it, and no deterministic checker could examine it.
    `_prov_details` puts that one click away on every entity.
  * nothing is summarised out of existence. Stage 11 is the complete ledger --
    every field, on every entity, with its action, source, value and checks -- and
    stage 10's edge table lists every edge of all four relation types. If the
    pipeline decided it, it is on this page somewhere.
"""

from __future__ import annotations
import math
import re
from html import escape

from graph.models import Relation

# ---- category hues (fixed; readable on light surfaces) ----
_HL = {
    "PERSON": ("#E9F0FA", "#234E86"), "LOCATION": ("#EAF2E0", "#315915"),
    "INSTITUTION": ("#EAF2E0", "#315915"), "AGE": ("#EFEDFA", "#413593"),
    "DATE_ABSOLUTE": ("#F8EFDD", "#6A4310"), "DATE_RELATIVE": ("#F8EFDD", "#6A4310"),
    "DATE_ANCHOR": ("#FBEAEA", "#7C2222"), "DATE_OF_BIRTH": ("#F8EFDD", "#6A4310"),
    "PHONE": ("#FBEAEA", "#7C2222"), "EMAIL": ("#FBEAEA", "#7C2222"),
    "SSN_OR_ID": ("#FBEAEA", "#7C2222"), "USERNAME_HANDLE": ("#FBEAEA", "#7C2222"),
    "OCCUPATION": ("#E4F1F1", "#245b5b"),
}
_PERSON_FILL = {"FAMILY": "#5B7FA6", "PROFESSIONAL": "#9CB6D2",
                "PUBLIC_FIGURE": "#AEB3BA", "PUBLIC_FIGURE_UNCONFIRMED": "#AEB3BA"}
_ID_CATS = ("PHONE", "EMAIL", "SSN_OR_ID", "USERNAME_HANDLE", "OCCUPATION")
_DATE_CATS = ("DATE_ABSOLUTE", "DATE_RELATIVE", "DATE_ANCHOR", "DATE_OF_BIRTH")
_ID_LABEL = {"PHONE": "Phone", "EMAIL": "Email", "SSN_OR_ID": "SSN / ID",
             "USERNAME_HANDLE": "Handle", "OCCUPATION": "Occupation"}


def _relname(ed):
    return ed.relation.value if isinstance(ed.relation, Relation) else str(ed.relation)


def _pname(e):
    return e.sorted_mentions[0] if e.sorted_mentions else "?"


def _pct(x):
    return "&mdash;" if x is None else f"{x * 100:.0f}%"


def _val(v):
    if v is None:
        return "<i class='none'>&mdash;</i>"
    if isinstance(v, bool):
        return "yes" if v else "no"
    return escape(str(v))


def section(num, title, body, note="", wide=False):   # `wide` kept for callers; ignored
    note_html = f"<p class='s-note'>{note}</p>" if note else ""
    return (f"<section class='stage'><div class='s-head'><span class='s-num'>{num}</span>"
            f"<h3>{escape(title)}</h3></div>{note_html}{body}</section>")


# ---------------------------------------------------------------- provenance
# The second line records, per field, HOW the value was reached. Everything below
# renders that record; nothing re-derives it.

_ACTION = {
    "confirm": ("rule + LLM agree", "a-ok"),
    "fill":    ("LLM filled the gap", "a-fill"),
    "keep":    ("rules only", "a-keep"),
    "conflict": ("layers disagreed", "a-conf"),
}


def _action_badge(res) -> str:
    if res.action == "reject":
        label, cls = (("proposal refuted", "a-rej") if res.checks_failed
                      else ("both layers blind", "a-blind"))
    else:
        label, cls = _ACTION.get(res.action, (res.action, ""))
    blk = " <b class='blkmark'>BLOCKING</b>" if res.blocking else ""
    return f"<span class='act {cls}'>{label}</span>{blk}"


def _checks_html(res, quiet=False) -> str:
    """What actually examined this value. `checks_passed` is the only one that means
    verification -- a skipped checker said nothing, which is why they are rendered
    differently rather than lumped together.

    `quiet` is for a value on a field's SAFE direction (replace=True). Those fields
    deliberately gate only the leak-prone direction, so "nothing verified this" is
    the designed outcome rather than a gap, and flagging it amber on every redacted
    span buries the one row where it does matter -- an unverified KEEP.
    """
    bits = []
    if res.checks_passed:
        bits.append(f"<span class='ck ok'>&check; {escape(', '.join(res.checks_passed))}</span>")
    if res.checks_failed:
        bits.append(f"<span class='ck bad'>&times; {escape(', '.join(res.checks_failed))}</span>")
    if res.checks_skipped:
        bits.append(f"<span class='ck na'>n/a {escape(', '.join(res.checks_skipped))}</span>")
    if not res.checks_passed and not res.checks_failed:
        bits.insert(0, "<span class='ck na'>safe direction &mdash; no keep-gate "
                       "applies</span>" if quiet else
                       "<span class='ck none'>nothing verified this</span>")
    return " ".join(bits)


def _prov_of(e) -> dict:
    return getattr(e, "provenance", None) or {}


def _prov_table(prov: dict, names=None) -> str:
    rows = []
    for fname, res in prov.items():
        label = fname
        if ":" in fname and names is not None:
            base, other = fname.split(":", 1)
            label = f"{base} &rarr; {escape(names.get(other, other))}"
        else:
            label = escape(fname)
        rows.append(
            f"<tr class='{'blk' if res.blocking else ''}'>"
            f"<td class='f'>{label}</td>"
            f"<td>{_action_badge(res)}</td>"
            f"<td class='v'>{_val(res.value)}</td>"
            f"<td class='src2'>{escape(res.source or '&mdash;')}"
            + (f"<i> &middot; {escape(res.confidence)} conf</i>"
               if res.confidence and res.confidence != "unstated" else "")
            + f"</td><td class='ck-cell'>{_checks_html(res)}"
              f"<div class='why'>{escape(res.reason)}</div></td></tr>")
    if not rows:
        return "<p class='muted'>No fields were resolved for this entity.</p>"
    return ("<table class='prov'><thead><tr><th>field</th><th>outcome</th>"
            "<th>value</th><th>source</th><th>deterministic checks</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>")


def _prov_list(prov: dict, names=None) -> str:
    """The same decision record as `_prov_table`, stacked instead of tabulated.

    A five-column table needs ~690px; a person card in the grid is ~300px, so the
    table version overflowed its card and rendered as an unreadable smear. Inside a
    card each field becomes a small block that wraps cleanly at any width; the table
    is kept for the full-width sections, where columns genuinely help comparison.
    """
    blocks = []
    for fname, res in prov.items():
        if ":" in fname and names is not None:
            base, other = fname.split(":", 1)
            label = f"{escape(base)} &rarr; {escape(names.get(other, other))}"
        else:
            label = escape(fname)
        src = escape(res.source or "&mdash;")
        if res.confidence and res.confidence != "unstated":
            src += f", {escape(res.confidence)} confidence"
        blocks.append(
            f"<div class='pv{' blk' if res.blocking else ''}'>"
            f"<div class='pv-h'><span class='pv-f'>{label}</span>"
            f"{_action_badge(res)}</div>"
            f"<div class='pv-v'>{_val(res.value)} <i>&middot; {src}</i></div>"
            f"<div class='pv-c'>{_checks_html(res)}</div>"
            f"<div class='why'>{escape(res.reason)}</div></div>")
    return f"<div class='pv-list'>{''.join(blocks)}</div>"


def _prov_details(e, names=None, label="decision record", stacked=False) -> str:
    prov = _prov_of(e)
    if not prov:
        return ""
    n_blk = sum(1 for r in prov.values() if r.blocking)
    tail = f" &middot; <b class='blkmark'>{n_blk} blocking</b>" if n_blk else ""
    inner = _prov_list(prov, names) if stacked else _prov_table(prov, names)
    return (f"<details class='prov-d'><summary>{label} &middot; {len(prov)} field(s)"
            f"{tail}</summary>{inner}</details>")


# Review flags are accumulated by `Entity.flag_entity`, which joins them with "; ".
# Rendered raw that is one long red run-on sentence -- and the reasons THEMSELVES
# contain semicolons and colons, so a naive split shreds them. Each flag written by
# `apply_resolution` starts with "<field>: ", so split only where the next fragment
# begins with a real policy field name.
def _flag_items(reason: str) -> list[str]:
    from graph.second_line import POLICIES
    fields = sorted(set(POLICIES) | {"same_person"}, key=len, reverse=True)
    pat = "|".join(re.escape(f) for f in fields)
    return [p.strip() for p in re.split(rf";\s+(?={pat}:)", reason or "") if p.strip()]


def _flag_html(e) -> str:
    """The review flags for one entity, as a readable list rather than a red blob."""
    if not e.needs_review or not e.review_reason:
        return ""
    items = []
    for item in _flag_items(e.review_reason):
        head, sep, rest = item.partition(": ")
        if sep and " " not in head:                # "<field>: <why>"
            items.append(f"<li><b>{escape(head)}</b> {escape(rest)}</li>")
        else:
            items.append(f"<li>{escape(item)}</li>")
    return (f"<div class='note'><div class='note-h'>&#9873; needs review "
            f"&middot; {len(items)}</div><ul>{''.join(items)}</ul></div>")


def _names_map(case) -> dict:
    names = {e.entity_id: _pname(e) for e in case["entities"]}
    iv = case["info"]["interviewee"]
    names[iv.entity_id] = _pname(iv) if iv.sorted_mentions else "the interviewee"
    return names


# ---------------------------------------------------------------- metrics
def metrics_grid(R) -> str:
    c, r, g, rp = R["cluster"], R["rel"], R["gender"], R["replace"]
    d, a, l = R["dates"], R["ages"], R["locations"]
    o, sl = R.get("owner") or {}, R.get("second_line") or {}
    iv = R.get("interviewee") or {}
    dr, ar = R.get("date_redaction") or {}, R.get("age_redaction") or {}

    leaks = rp["leaks"] + l.get("rep_leaks", 0) + dr.get("leaks", 0) + ar.get("leaks", 0)
    n_blk = len(sl.get("blocking") or [])

    def tile(val, lab, sub="", cls=""):
        sub = f"<div class='m-sub'>{sub}</div>" if sub else ""
        return (f"<div class='m-tile'><div class='m-val {cls}'>{val}</div>"
                f"<div class='m-lab'>{lab}</div>{sub}</div>")

    iv_ok = sum(1 for k in ("identity", "gender", "ethnicity", "dob")
                if iv.get(k) == "correct")
    iv_tot = sum(1 for k in ("identity", "gender", "ethnicity", "dob")
                 if iv.get(k) and iv[k] != "n/a")

    tiles = [
        # the two numbers that decide whether the next stage may run
        tile(f"{leaks}", "privacy leaks", "people + places + dates + ages",
             "ok" if leaks == 0 else "bad"),
        tile(f"{n_blk}", "blocking fields", "a human decides before surrogates",
             "ok" if n_blk == 0 else "warnv"),
        tile(_pct(o.get("accuracy")), "ownership accuracy",
             f"{o.get('wrong', 0)} wrong &middot; {o.get('missing', 0)} unresolved",
             "bad" if o.get("wrong") else ""),
        tile(f"{iv_ok}/{iv_tot}" if iv_tot else "&mdash;", "interviewee fields",
             "identity &middot; gender &middot; ethnicity &middot; DOB"),
        tile(_pct(c["recall"]), "clustering recall",
             f"{c['exact']}/{c['gold']} exact &middot; {c['over_merges']} over-merge "
             f"&middot; {c['splits']} split"),
        tile(_pct(r["recall"]), "relation recall",
             f"{r['tp']}/{r['gold']} found &middot; P {_pct(r['precision'])}"),
        tile(_pct(g["recall"]), "gender recall (others)", f"{g['missing']} missing"),
        tile(_pct(rp["accuracy"]), "person redaction",
             f"{rp['over_redactions']} over-redaction"),
        tile(_pct(l.get("rep_accuracy")), "place redaction",
             f"{l.get('rep_over', 0)} over-redaction"),
        tile(_pct(d["accuracy"]), "date accuracy", f"{d['pass']}/{d['total']} resolved"),
        tile(_pct(a["accuracy"]), "age accuracy", f"{a['pass']}/{a['total']} resolved"),
        tile(_pct(dr.get("accuracy")), "date + age redaction",
             f"{dr.get('total', 0) + ar.get('total', 0)} spans judged"),
    ]
    return "<div class='metrics'>" + "".join(tiles) + "</div>"


# ---------- 01 · detect ----------
def stage_detect(case) -> str:
    text, dets = case["text"], sorted(case["dets"], key=lambda d: d["start"])
    counts = {}
    for d in dets:
        counts[d["entity_type"]] = counts.get(d["entity_type"], 0) + 1
    out, i = [], 0
    for d in dets:
        if d["start"] < i:
            continue
        out.append(escape(text[i:d["start"]]))
        fill, col = _HL.get(d["entity_type"], ("#eee", "#333"))
        out.append(f"<mark style='background:{fill};color:{col}' "
                   f"title='{escape(d['entity_type'])}'>"
                   f"{escape(text[d['start']:d['end']])}</mark>")
        i = d["end"]
    out.append(escape(text[i:]))
    rendered = "".join(out).strip("\n").replace("\n", "<br>")
    tally = " &middot; ".join(f"{escape(k)} {v}" for k, v in sorted(counts.items()))
    legend = ("<div class='legend'>"
              "<span class='lg lg-p'>person</span><span class='lg lg-l'>location</span>"
              "<span class='lg lg-d'>date</span><span class='lg lg-a'>age</span>"
              "<span class='lg lg-i'>identifier</span></div>")
    body = (f"<div class='tally'>{tally}</div>"
            f"<div class='excerpt'>{rendered}</div>{legend}")
    return section("01", "Detect identifier spans", body,
                   f"{len(dets)} spans from the (simulated perfect) detector. Every "
                   f"guarantee on this page is conditional on this stage: a span that "
                   f"is never detected is never decided about.")


# ---------- 02 · rule-based clustering ----------
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
        rows.append("<li class='muted'>Every name appeared in a single form &mdash; "
                    "nothing to merge or flag.</li>")
    body = f"<ul class='deltas'>{''.join(rows)}</ul>"
    return section("02", "Cluster mentions (rule-based)", body,
                   f"{n_ment} person mentions &rarr; {len(pre)} entities via exact / "
                   f"alias / containment rules. A bare given name that matches exactly "
                   f"one full name is put to the LLM before it is merged.")


# ---------- 03 · coreference (ML) ----------
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
                    f"<span class='muted'>(coref link, LLM-confirmed)</span></li>")
    for f in flags:
        rows.append(f"<li><span class='tag warn'>held apart</span> "
                    f"<b>{escape(f['a'])}</b> &harr; <b>{escape(f['b'])}</b> "
                    f"<span class='muted'>{escape(f['note'])}</span></li>")
    if not rows:
        rows.append("<li class='muted'>fastcoref suggested no cross-entity links "
                    "here.</li>")
    body = f"<ul class='deltas'>{''.join(rows)}</ul>"
    return section("03", "Coreference resolution (ML)", body,
                   "fastcoref proposes links; a link becomes a merge only if the LLM "
                   "confirms the two are one person. Every merge and every veto is "
                   "arbitrated under `same_person` and appears in the ledger.")


# ---------- 04 · relations ----------
def stage_relations(case) -> str:
    ents = {e.entity_id: e for e in case["entities"]}
    iv = case["info"]["interviewee"].entity_id

    def nm(i):
        return "Interviewee" if i == iv else (_pname(ents[i]) if i in ents else i)

    rows = []
    for ed in case["edges"]:
        if _relname(ed) != "RELATED_TO":
            continue
        ev = str(ed.evidence or "")
        llm = ev.startswith("(llm)")
        ev = ev[5:].strip() if llm else ev
        tag = ("<span class='src llm'>llm</span>" if llm
               else "<span class='src rule'>rule</span>")
        rows.append(f"<li>{tag}<b>{escape(nm(ed.source))}</b> "
                    f"<span class='rel'>{escape(ed.detail)}</span> "
                    f"<b>{escape(nm(ed.target))}</b> "
                    f"<span class='ev'>&ldquo;{escape(ev)}&rdquo;</span></li>")
    body = (f"<ul class='rels'>{''.join(rows)}</ul>" if rows
            else "<p class='muted'>No family relationships were extracted.</p>")
    return section("04", "Extract relationships", body,
                   f"{len(rows)} relationship(s), each with the quote that justifies "
                   f"it. Rule edges come from kinship constructions; LLM edges must "
                   f"clear the deterministic verifier before they become edges.")


# ---------- shared: owned attributes ----------
def _owned_by_person(case):
    """person entity_id -> [owned attribute entities], from ATTRIBUTE_OF edges."""
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


def _chip(text, cls=""):
    return f"<span class='{('chip ' + cls).strip()}'>{text}</span>"


def _owned_chip(e):
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


# ---------- 10 · graph ----------
def _graph_svg(case) -> str:
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
    body = (f"<div class='graph'>{_graph_svg(case)}{_GRAPH_LEGEND}</div>"
            + _edge_table(case))
    return section("10", "Entity graph", body,
                   "The interviewee sits at the centre of the family graph; every "
                   "other edge the pipeline built is listed underneath, because the "
                   "surrogate generator walks all four kinds.")


# ---------- 11 · the full ledger ----------
def stage_ledger(case) -> str:
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


# ---------- assembly ----------
_STEPS = ["Detect", "Cluster", "Coref", "Relations", "Interviewee", "People",
          "Places", "Dates & ages", "Identifiers", "Graph", "Ledger", "Artifact"]


def _stepper() -> str:
    chips = [f"<span class='st'><b>{i:02d}</b>{escape(s)}</span>"
             for i, s in enumerate(_STEPS, 1)]
    return "<div class='stepper'>" + "<span class='sep'>&rarr;</span>".join(chips) + "</div>"


def transcript_panel(case, metrics=None) -> str:
    head = metrics_grid(metrics) if metrics else ""
    stages = (stage_detect(case) + stage_cluster(case) + stage_coref(case)
              + stage_relations(case) + stage_interviewee(case) + stage_people(case)
              + stage_places(case) + stage_dates_ages(case) + stage_identifiers(case)
              + stage_graph(case) + stage_ledger(case) + stage_artifact(case))
    return head + _stepper() + f"<div class='stages'>{stages}</div>"


CSS = """
:root{--ink:#1f2328;--muted:#5b636c;--faint:#9aa3ad;--line:#e7e9ec;--panel:#f6f7f9;
  --accent:#3b6ea5;--accentbg:#eef3f9;}
*{box-sizing:border-box;}
body{margin:0;background:#fbfbfc;color:var(--ink);-webkit-font-smoothing:antialiased;
  font-family:'Iowan Old Style','Palatino Linotype',Palatino,'Book Antiqua',Georgia,'Times New Roman',serif;
  font-size:16px;line-height:1.6;}
.wrap{max-width:1120px;margin:0 auto;padding:40px 28px 100px;}
h1{font-size:24px;font-weight:600;letter-spacing:-0.3px;margin:0 0 4px;}
.lede{color:var(--muted);font-size:14.5px;margin:0 0 8px;max-width:78ch;}
h3{font-size:15.5px;font-weight:600;margin:0;letter-spacing:-0.1px;}
.muted{color:var(--muted);} .faint{color:var(--faint);}
code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px;
  background:var(--panel);padding:1px 5px;border-radius:4px;}

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
.m-val.ok{color:#2f7a3d;} .m-val.bad{color:#b3261e;} .m-val.warnv{color:#8a5a12;}
.m-lab{font-size:12px;color:var(--muted);margin-top:4px;}
.m-sub{font-size:10.5px;color:var(--faint);margin-top:2px;}
@media(max-width:900px){.metrics{grid-template-columns:repeat(2,1fr);}}

/* stepper */
.stepper{display:flex;flex-wrap:wrap;align-items:center;gap:6px;margin:0 0 20px;}
.stepper .st{background:#fff;border:1px solid var(--line);border-radius:20px;padding:4px 12px;
  font-size:12px;color:var(--muted);}
.stepper .st b{color:var(--accent);font-weight:600;margin-right:5px;font-size:11px;}
.stepper .sep{color:var(--faint);font-size:12px;}

/* stages */
.stages{display:flex;flex-direction:column;gap:18px;}
.stage{background:#fff;border:1px solid var(--line);border-radius:14px;padding:20px 22px;}
.s-head{display:flex;align-items:center;gap:11px;}
.s-num{flex:0 0 auto;width:26px;height:26px;border-radius:50%;background:var(--accentbg);color:var(--accent);
  font-size:11.5px;font-weight:600;display:flex;align-items:center;justify-content:center;}
.s-note{color:var(--muted);font-size:13px;margin:6px 0 0 37px;max-width:86ch;}

/* 01 detect */
.tally{font-size:11.5px;color:var(--faint);margin-top:12px;}
.excerpt{background:var(--panel);border-radius:10px;padding:16px 18px;line-height:1.95;font-size:14.5px;margin-top:8px;}
mark{padding:1px 5px;border-radius:4px;font-weight:500;}
.legend{display:flex;gap:16px;margin-top:12px;font-size:12px;color:var(--muted);flex-wrap:wrap;}
.lg::before{content:"";display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:6px;vertical-align:1px;}
.lg-p::before{background:#5B7FA6;}.lg-l::before{background:#6f9a4a;}.lg-d::before{background:#c99a4a;}
.lg-a::before{background:#7d72c4;}.lg-i::before{background:#c0574f;}

/* lists */
.deltas,.rels{list-style:none;padding:0;margin:14px 0 0;}
.deltas li,.rels li{padding:8px 0;border-bottom:1px solid var(--line);font-size:14px;}
.deltas li:last-child,.rels li:last-child{border-bottom:none;}
.tag{display:inline-block;font-size:11px;padding:1px 8px;border-radius:20px;margin-right:6px;}
.tag.ok{background:#e7f2e9;color:#2f6b3b;} .tag.warn{background:#fbeceb;color:#93362f;}
.src{display:inline-block;font-size:9.5px;font-weight:600;letter-spacing:.5px;text-transform:uppercase;
  padding:1px 6px;border-radius:4px;margin-right:7px;}
.src.rule{background:#eceef1;color:#5b636c;} .src.llm{background:var(--accentbg);color:var(--accent);}
.rel{display:inline-block;font-size:11.5px;color:var(--accent);background:var(--accentbg);padding:1px 8px;border-radius:20px;}
.ev{display:block;color:var(--faint);font-size:12px;margin-top:2px;font-style:italic;}

/* cards */
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:10px;margin-top:14px;}
.card{background:var(--panel);border:1px solid var(--line);border-radius:11px;padding:12px 14px;}
.card.flag{border-left:3px solid #c0574f;}
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

/* review flags: a list, not a red run-on sentence */
.note{margin-top:9px;background:#fdf5f4;border:1px solid #f0dbd8;border-left:3px solid #c0574f;
  border-radius:8px;padding:8px 11px 9px;
  font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;}
.note-h{font-size:10px;font-weight:700;letter-spacing:.5px;text-transform:uppercase;color:#a2453c;}
.note ul{margin:5px 0 0;padding-left:15px;}
.note li{font-size:12px;line-height:1.55;color:#4b3330;margin:3px 0;overflow-wrap:anywhere;}
.note li b{color:#8a2f28;font-weight:600;}

/* decision record, stacked -- for narrow containers (person cards) */
.pv-list{margin-top:8px;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;}
.pv{border-top:1px solid var(--line);padding:7px 0 8px;overflow-wrap:anywhere;}
.pv:first-child{border-top:none;}
.pv.blk{background:#fdf4f3;border-radius:6px;padding-left:7px;padding-right:7px;}
.pv-h{display:flex;flex-wrap:wrap;align-items:center;gap:7px;}
.pv-f{font-size:12px;font-weight:700;color:var(--accent);}
.pv-v{font-size:12.5px;font-weight:600;margin-top:3px;}
.pv-v i{font-style:normal;font-weight:400;color:var(--muted);font-size:11.5px;}
.pv-c{margin-top:3px;line-height:1.6;}
.pv .why{margin-top:3px;}

/* provenance tables */
table.prov{width:100%;border-collapse:collapse;margin:10px 0 4px;font-size:12.5px;
  font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;}
table.prov th{text-align:left;font-size:10.5px;text-transform:uppercase;letter-spacing:.4px;
  color:var(--faint);font-weight:600;border-bottom:1px solid var(--line);padding:4px 8px 5px;}
table.prov td{border-bottom:1px solid var(--line);padding:6px 8px;vertical-align:top;}
table.prov tr:last-child td{border-bottom:none;}
table.prov tr.blk{background:#fdf4f3;}
table.prov td.f{color:var(--accent);font-weight:600;white-space:nowrap;}
table.prov td.v{font-weight:600;}
table.prov td.v.big{font-size:14px;}
table.prov td.src2{color:var(--muted);white-space:nowrap;}
table.prov td.src2 i{font-style:normal;color:var(--faint);}
table.prov td.parts{color:var(--muted);}
.part{display:inline-block;background:var(--panel);border-radius:4px;padding:0 5px;margin:0 3px 3px 0;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11px;}
.iv-t td.f{width:150px;}
.act{display:inline-block;font-size:10px;font-weight:600;letter-spacing:.3px;padding:1px 7px;
  border-radius:20px;white-space:nowrap;}
.act.a-ok{background:#e7f2e9;color:#2f6b3b;}
.act.a-fill{background:#F8EFDD;color:#6A4310;}
.act.a-keep{background:#eceef1;color:#5b636c;}
.act.a-conf{background:#FBEAEA;color:#7C2222;}
.act.a-rej{background:#F3E8F6;color:#6b2f7c;}
.act.a-blind{background:#eef0f2;color:#7c828b;}
.blkmark{color:#b3261e;font-size:10px;letter-spacing:.4px;}
.ck{display:inline-block;font-size:11px;margin-right:8px;}
.ck.ok{color:#2f6b3b;} .ck.bad{color:#b3261e;} .ck.na{color:var(--faint);}
.ck.none{color:#8a5a12;font-style:italic;}
.why{color:var(--faint);font-size:11px;margin-top:2px;line-height:1.45;}
.why2{font-size:12px;margin:2px 0 6px;}
.none{color:var(--faint);font-style:normal;}
.ev2{color:var(--faint);font-size:11px;font-style:italic;}
.arrow2{color:var(--muted);font-style:normal;font-weight:400;}
details.prov-d,details.ldg{margin-top:8px;}
details.prov-d summary,details.ldg summary{cursor:pointer;font-size:11.5px;color:var(--accent);
  list-style:none;padding:3px 0;}
details.ldg summary{font-size:13px;padding:7px 0;border-top:1px solid var(--line);}
details.prov-d summary::-webkit-details-marker,details.ldg summary::-webkit-details-marker{display:none;}
details.prov-d summary::before,details.ldg summary::before{content:"\\25B8 ";color:var(--faint);}
details.prov-d[open] summary::before,details.ldg[open] summary::before{content:"\\25BE ";}

/* graph + places + dates */
.graph{padding-top:8px;}
.glegend{display:flex;flex-wrap:wrap;gap:14px;justify-content:center;margin-top:14px;font-size:11.5px;color:var(--muted);}
.glegend i{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px;}
.glegend .ring{color:#8a2f28;}
.sub-h{font-size:11.5px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:0.5px;margin:20px 0 9px;}
.tally2{font-size:12px;margin:4px 0 0;}
.trow{display:flex;align-items:center;gap:9px;margin-bottom:5px;font-size:13px;}
.tw{color:var(--faint);}
.pills{display:flex;flex-wrap:wrap;gap:7px;}
.pill{font-size:12px;padding:3px 10px;border-radius:20px;display:inline-block;}
.pill.loc{background:#EAF2E0;color:#315915;}
.pill.kin{background:#E9F0FA;color:#234E86;}
.pill.repl{background:#FBECEA;color:#7C2222;}
.pill.keepp{background:#EAF3EC;color:#2F6B3B;}
.ptype{font-size:11.5px;color:var(--muted);}
.dmark{font-style:normal;font-size:10.5px;color:var(--muted);background:var(--panel);
  border-radius:4px;padding:0 5px;margin-left:3px;}
.own{font-style:normal;font-size:10.5px;padding:1px 7px;border-radius:20px;white-space:nowrap;}
.own-iv{background:#2C4A6E;color:#fff;}
.own-other{background:#fff;color:var(--muted);border:1px solid var(--line);}
.own-block{background:#7C2222;color:#fff;}

/* boxes */
.blkbox{background:#fdf4f3;border:1px solid #f0d5d2;border-left:3px solid #b3261e;
  border-radius:9px;padding:12px 15px;margin:14px 0 4px;font-size:13px;}
.blkbox ul{margin:6px 0 0;padding-left:18px;} .blkbox li{margin:3px 0;}
.okbox{background:#f2f8f3;border:1px solid #d5e7d9;border-left:3px solid #2f7a3d;
  border-radius:9px;padding:12px 15px;margin:14px 0 4px;font-size:13px;}
.kvs{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:6px 14px;margin:14px 0 0;}
.kv{display:flex;justify-content:space-between;gap:10px;font-size:12.5px;
  border-bottom:1px dotted var(--line);padding:3px 0;}
.kv span{color:var(--muted);}
.kv b{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11.5px;font-weight:500;}
.foot{font-size:12.5px;color:var(--muted);border-top:1px solid var(--line);padding-top:16px;margin-top:24px;}
"""
