"""
Panels 01-04: detection, name clustering, coreference, relation edges.

PURPOSE
    Render the first four stages of the pipeline -- the ones that decide WHO the
    transcript is about. Each `stage_*` takes a case dict and returns one HTML
    section.

FIT
    Sits on `primitives.py`; assembled by `page.transcript_panel`. Reads
    `case["info"]["pre_coref"]` / `["coref_merges"]`, the trace snapshots
    `graph/pipeline.py` records when `trace=True`, which is what lets the coref
    panel show its effect separately from rule-based clustering.
"""

from __future__ import annotations

from html import escape
from .primitives import _HL, _pname, _relname, section


# ---------- 01 · detect ----------
def stage_detect(case) -> str:
    """Panel 01: the transcript with every detected span highlighted by category.

    HOW: walk the detections in transcript order and alternate between plain text
    and a coloured `<mark>`, using `i` as a cursor to the last position emitted.
    `if d["start"] < i: continue` skips any span that overlaps one already
    rendered -- the cursor cannot move backwards, so an overlap would otherwise
    duplicate text.

    Newlines become `<br>` only at the very end, after all escaping, so the
    inserted markup is not itself escaped.
    """
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
    """Panel 02: what RULE-based clustering merged, and what it flagged.

    Reads the `pre_coref` trace snapshot -- the entity list as it stood BEFORE the
    ML stage -- so the panel shows the rule layer's work in isolation. Two kinds
    of row: an entity written more than one way (a merge happened) and an entity
    carrying a review flag (the rule declined to merge and said why).
    """
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
    """Panel 03: what the coreference model merged, and what the double gate held apart.

    Shows both outcomes, because a HELD-APART pair is usually the system working
    correctly rather than a failure -- coref over-links, and the gate exists to
    catch that. Returns a short placeholder section when coref did not run
    (`run_coref=False`, or no person entities).
    """
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
    """Panel 04: the RELATED_TO edges, with the quote that justifies each one.

    The evidence column is the point: a family tie the reader cannot trace back to
    a sentence is not reviewable.
    """
    ents = {e.entity_id: e for e in case["entities"]}
    iv = case["info"]["interviewee"].entity_id

    def nm(i):
        """A display name for an entity id; the interviewee reads as "You"."""
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
