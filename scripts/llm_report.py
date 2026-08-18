"""
Interactive two-layer report: runs BOTH the deterministic ruleset and the local
LLM layer on the sample transcripts and builds one self-contained HTML page
with, per transcript:
  - the full pipeline walkthrough (all stages) + the entity graph,
  - a "Two-layer review" panel: agreements / suggestions / conflicts between the
    rules and the LLM, where suggestions and conflicts can be resolved
    (accept / reject) and exported,
and, up top, a rules-only vs rules+LLM metrics comparison.

    ./venv/bin/python3 scripts/llm_report.py

Writes tests/llm_report.html and opens it. Needs Ollama running with the model
(see LLM.md); the persistent cache makes re-runs fast. First run takes a few
minutes.
"""

from __future__ import annotations
import os
os.environ["KG_USE_LLM"] = "1"          # turn the LLM layer on for this report

import importlib.util
import sys
import webbrowser
from html import escape
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

from demo_utils import all_tids, load_case, REPO_ROOT
import render

OUT = REPO_ROOT / "tests" / "llm_report.html"
TITLES = {
    "interview_001": "Gulf Coast Vietnamese shrimping family",
    "interview_002": "Appalachian coal-mining family",
}


def _load_eval():
    spec = importlib.util.spec_from_file_location("kg_eval", SCRIPTS / "eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------- pull agreements / suggestions / conflicts out of a finished run ----------
# Read from the LEDGER, not from `review_reason` strings and `suggested_*` mirror keys.
#
# The old version grepped each entity's review text for the literal phrase "conflicts
# with rule". Nothing in `graph/` or `llm_layer/` has ever written that phrase -- the
# unified second line writes "LLM says X, rule kept Y" -- so the Conflicts column was
# unreachable: both sample transcripts rendered "Conflicts 0" while the ledger held
# ~20 real disagreements (`identifying`, `subtype_person`, `replace`, `location_parent`,
# `owner`, `same_person`). It also missed every field `_legacy_mirror` has no branch
# for, which is most of them: `owner`, `identifying`, `replace`, `replace_location`,
# `shiftable`, `approximate`, `stated_with`, `given_name`, `surname`. And two of the
# keys it did read -- `candidate_public_figure`, `suggested_kind` -- are written by
# nothing at all.
#
# `info["ledger"]` is `{entity_id: {field: Resolution}}` and is the authoritative
# record of every decision, so every column is now derived from it directly. One row
# per resolution, bucketed by ACTION, which is what the second line already computes.
_FIELD_LABEL = {
    "gender": "gender", "interviewee_gender": "gender (speaker)",
    "given_name": "name part", "surname": "name part", "role": "role",
    "ethnicity": "ethnicity", "replace": "redaction",
    "subtype_person": "subtype", "subtype_location": "place type",
    "location_parent": "place parent", "replace_location": "place redaction",
    "resolved_value": "date", "shiftable": "date shiftable",
    "approximate": "approximate", "value": "age", "owner": "owner",
    "kind": "identifier kind", "identifying": "identifying",
    "stated_with": "age↔date pairing", "relation": "relation",
    "same_person": "merge", "interviewee_identity": "speaker identity",
}


def _val(v):
    return "—" if v is None else str(v)


def reconcile_items(case):
    info = case["info"]
    ledger = info.get("ledger", {})
    names = {e.entity_id: (e.sorted_mentions[0] if e.sorted_mentions else e.entity_id)
             for e in case["entities"]}
    names[info["interviewee"].entity_id] = "the interviewee"
    agree, sugg, conflict, unverified = [], [], [], []

    for eid, fields in ledger.items():
        if eid == "_edges":
            continue
        nm = names.get(eid, eid)
        for fname, res in fields.items():
            base = fname.split(":", 1)[0]
            kind = _FIELD_LABEL.get(base, base)
            checks = (f"{len(res.checks_passed)} check(s) passed"
                      if res.checks_passed else "NOTHING VERIFIED IT")
            if res.action == "confirm":
                agree.append({"kind": kind, "text": nm,
                              "detail": f"{base} = {_val(res.value)} — rule & LLM agree"})
            elif res.action == "fill":
                src = ("the checkers alone" if res.source == "checker_derived"
                       else f"the LLM (confidence {res.confidence})")
                sugg.append({"kind": kind, "text": nm,
                             "detail": f"{base} = {_val(res.value)} filled from {src}; "
                                       f"{checks}"})
            elif res.action == "conflict":
                conflict.append({"kind": kind, "text": nm,
                                 "detail": f"{base}: {res.reason}"
                                           + ("  [BLOCKING]" if res.blocking else "")})
            elif res.action == "reject" and res.checks_failed:
                conflict.append({"kind": kind, "text": nm,
                                 "detail": f"{base}: proposal refuted — {res.reason}"
                                           + ("  [BLOCKING]" if res.blocking else "")})
            elif res.action == "reject" and res.blocking:
                conflict.append({"kind": kind, "text": nm,
                                 "detail": f"{base}: neither layer produced a value, and "
                                           f"this field must be verified  [BLOCKING]"})
            elif res.action == "keep" and res.blocking:
                conflict.append({"kind": kind, "text": nm,
                                 "detail": f"{base} = {_val(res.value)} from the rules, "
                                           f"unconfirmed and unverified  [BLOCKING]"})
            elif res.action == "keep":
                # RULES STAND, NOTHING CONFIRMED THEM. Not a disagreement, so not a
                # conflict; not a suggestion either. It used to fall through every
                # branch and vanish from the page.
                unverified.append({
                    "kind": kind, "text": nm,
                    "detail": f"{base} = {_val(res.value)} from the rules; the LLM "
                              f"gave no answer to confirm it"
                              + ("" if res.checks_passed else
                                 "; no deterministic check applied either")})
            elif res.action == "reject":
                # BOTH LAYERS BLIND -- the outcome `graph/second_line.py` describes as
                # "made visible", and the one this function made invisible: a
                # non-blocking `reject` with no failed checks matched no branch at all.
                # That is why `approximate` on an off-table DATE_ANCHOR -- a field that
                # genuinely had no proposer -- rendered as a clean page.
                unverified.append({
                    "kind": kind, "text": nm,
                    "detail": f"{base}: neither the rules nor the LLM produced a "
                              f"value"})

    # Merges the coref stage actually applied, kept as their own agreement rows so the
    # clustering decisions stay visible next to the field decisions.
    for m in info.get("coref_merges", []):
        agree.append({"kind": "merge", "text": f"{m['merged']} → {m['kept']}",
                      "detail": "coref + LLM agree: same person, merged"})
    for f in info.get("coref_flags", []):
        conflict.append({"kind": "merge", "text": f"{f['a']} ↔ {f['b']}",
                         "detail": f["note"]})

    # Most-actionable first: blocking rows, then the rest.
    conflict.sort(key=lambda c: "BLOCKING" not in c["detail"])
    sugg.sort(key=lambda s: "NOTHING VERIFIED IT" not in s["detail"])
    unverified.sort(key=lambda u: "neither" not in u["detail"])
    return agree, sugg, conflict, unverified


_KIND = 0


def _rec_section(tid, agree, sugg, conflict, unverified):
    global _KIND

    def card(item, actionable):
        global _KIND
        _KIND += 1
        cid = f"{tid}-r{_KIND}"
        btns = ("<div class='rec-act'>"
                "<button class='acc'>Accept</button>"
                "<button class='rej'>Reject</button></div>") if actionable else ""
        return (f"<div class='rec-card' data-id='{cid}'>"
                f"<div class='rec-t'>{escape(item['text'])} "
                f"<span class='rec-k'>{escape(item['kind'])}</span></div>"
                f"<div class='rec-d'>{escape(item['detail'])}</div>{btns}</div>")

    def col(title, items, actionable, cls, cap=None):
        shown = items if cap is None else items[:cap]
        body = "".join(card(i, actionable) for i in shown) or "<p class='muted'>none</p>"
        if cap is not None and len(items) > cap:
            body += (f"<p class='muted'>&hellip; and {len(items) - cap} more "
                     f"(informational; every row is in the ledger)</p>")
        return (f"<div class='rec-col {cls}'><h4>{title} "
                f"<span class='cnt'>{len(items)}</span></h4>{body}</div>")

    return ("<section class='stage wide'><div class='s-head'><span class='s-num'>R</span>"
            "<h3>Two-layer review &mdash; rules &times; LLM</h3></div>"
            "<p class='s-note'>Agreements are informational; suggestions and conflicts "
            "can be resolved (accept keeps the LLM's call, reject keeps the rules'). "
            "<b>Unverified</b> is the fourth outcome and the one worth reading: a rule "
            "value no second layer confirmed, or a field NEITHER layer could answer. "
            "Those rows used to be dropped from this page entirely, which made a field "
            "with no proposer at all look clean.</p>"
            "<div class='rec-cols'>"
            + col("Agreements", agree, False, "agree", cap=10)
            + col("Suggestions", sugg, True, "sugg")
            + col("Conflicts", conflict, True, "conf")
            + col("Unverified", unverified, False, "unver", cap=20)
            + "</div></section>")


# ---------- metrics aggregation + comparison ----------
def _agg(Rs):
    s = lambda k, f: sum(R[k][f] for R in Rs)
    tp, pred, gold = s("rel", "tp"), s("rel", "pred"), s("rel", "gold")

    def iv(field, verdict):
        return sum(1 for R in Rs if (R.get("interviewee") or {}).get(field) == verdict)

    return {
        "clus_recall": s("cluster", "exact") / max(1, s("cluster", "gold")),
        "over_merges": s("cluster", "over_merges"),
        "splits": s("cluster", "splits"),
        "rel_prec": tp / max(1, pred), "rel_recall": tp / max(1, gold),
        "gender_recall": s("gender", "correct") / max(1, s("gender", "total")),
        "leaks": s("replace", "leaks"),
        # The rows interviewee-only surrogate generation actually runs on. The table
        # used to show "Privacy leaks" alone, computed over PEOPLE only
        # (`eval.py` iterates `gold["people"]`), so a kept village, a misattributed
        # phone number and a wrong gender for the subject were all invisible here.
        "place_leaks": s("locations", "rep_leaks"),
        "own_acc": s("owner", "correct") / max(1, s("owner", "total")),
        "own_wrong": s("owner", "wrong"),
        "own_missing": s("owner", "missing"),
        "iv_gender": f"{iv('gender', 'correct')}/{len(Rs)}"
                     + (f" WRONG {iv('gender', 'wrong')}" if iv('gender', 'wrong') else ""),
        "iv_identity": f"{iv('identity', 'correct')}/{len(Rs)}"
                       + (f" WRONG {iv('identity', 'wrong')}" if iv('identity', 'wrong') else ""),
        "blocking": sum(len(R["second_line"]["blocking"]) for R in Rs),
    }


def _cmp_table(rules, llm):
    def pct(x):
        return f"{x * 100:.1f}%"
    rows = [
        ("Clustering recall", pct(rules["clus_recall"]), pct(llm["clus_recall"])),
        ("Over-merges", rules["over_merges"], llm["over_merges"]),
        ("Splits (unmerged)", rules["splits"], llm["splits"]),
        ("Relation precision", pct(rules["rel_prec"]), pct(llm["rel_prec"])),
        ("Relation recall", pct(rules["rel_recall"]), pct(llm["rel_recall"])),
        ("Privacy leaks &mdash; people", rules["leaks"], llm["leaks"]),
        ("Privacy leaks &mdash; places", rules["place_leaks"], llm["place_leaks"]),
        ("Ownership accuracy", pct(rules["own_acc"]), pct(llm["own_acc"])),
        ("&nbsp;&nbsp;wrong owner", rules["own_wrong"], llm["own_wrong"]),
        ("&nbsp;&nbsp;unresolved owner", rules["own_missing"], llm["own_missing"]),
        ("Interviewee identity", rules["iv_identity"], llm["iv_identity"]),
        ("Interviewee gender", rules["iv_gender"], llm["iv_gender"]),
        ("Blocking fields (need a human)", rules["blocking"], llm["blocking"]),
    ]
    # labels carry deliberate entities (&mdash;, &nbsp;) so they are NOT escaped
    trs = "".join(
        f"<tr><td>{m}</td><td>{r}</td><td class='hl'>{l}</td></tr>"
        for m, r, l in rows)
    return ("<table class='cmp'><thead><tr><th>metric</th><th>rules only</th>"
            "<th>rules + LLM</th></tr></thead><tbody>" + trs + "</tbody></table>")


_EXTRA_CSS = """
.cmp{border-collapse:collapse;width:100%;max-width:560px;margin:6px 0 30px;font-size:14px;}
.cmp th,.cmp td{border-bottom:1px solid var(--line);padding:8px 12px;text-align:left;}
.cmp th{font-size:12px;color:var(--muted);font-weight:600;}
.cmp td.hl{font-weight:600;}
.rec-cols{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;}
.rec-col h4{font-size:13px;margin:0 0 10px;font-weight:600;display:flex;gap:8px;align-items:center;}
.rec-col .cnt{font-size:11px;color:var(--muted);background:var(--panel);border-radius:20px;padding:1px 8px;}
.rec-col.agree h4{color:#2f6b3b;} .rec-col.sugg h4{color:#6A4310;}
.rec-col.conf h4{color:#7C2222;} .rec-col.unver h4{color:#4A4A55;}
.rec-card{background:#fff;border:1px solid var(--line);border-radius:10px;padding:10px 12px;margin-bottom:8px;}
.rec-col.agree .rec-card{border-left:3px solid #4a9d5b;}
.rec-col.sugg .rec-card{border-left:3px solid #c99a4a;}
.rec-col.conf .rec-card{border-left:3px solid #c0574f;}
.rec-col.unver .rec-card{border-left:3px solid #9aa0aa;}
.rec-t{font-weight:600;font-size:14px;} .rec-k{font-weight:400;font-size:11px;color:var(--faint);}
.rec-d{font-size:12.5px;color:var(--muted);margin-top:2px;}
.rec-act{margin-top:8px;display:flex;gap:6px;}
.rec-act button{font:inherit;font-size:12px;border:1px solid var(--line);background:#fff;
  border-radius:6px;padding:3px 12px;cursor:pointer;color:var(--ink);}
.rec-act .acc:hover{background:#eaf3ec;border-color:#4a9d5b;}
.rec-act .rej:hover{background:#fbecea;border-color:#c0574f;}
.rec-card.accepted{background:#f2f8f3;} .rec-card.accepted .acc{background:#4a9d5b;color:#fff;border-color:#4a9d5b;}
.rec-card.rejected{background:#fbf2f1;} .rec-card.rejected .rej{background:#c0574f;color:#fff;border-color:#c0574f;}
#resolve-bar{position:sticky;bottom:16px;background:#fff;border:1px solid var(--line);
  border-radius:12px;padding:12px 18px;display:flex;gap:16px;align-items:center;
  justify-content:flex-end;font-size:13px;color:var(--muted);margin:26px 0 0;
  box-shadow:0 4px 18px rgba(20,22,26,0.06);}
#resolve-bar button{font:inherit;font-size:13px;border:1px solid var(--accent);color:var(--accent);
  background:#fff;border-radius:8px;padding:6px 16px;cursor:pointer;}
#resolve-bar button:hover{background:var(--accentbg);}
"""

_JS = """<script>
  document.querySelectorAll('.tab').forEach(function(b){
    b.addEventListener('click',function(){
      document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
      document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
      b.classList.add('active');
      document.getElementById(b.dataset.tab).classList.add('active');
      window.scrollTo({top:0});
    });
  });
  const DECISIONS={};
  function refresh(){
    const cards=document.querySelectorAll('.rec-col.sugg .rec-card, .rec-col.conf .rec-card');
    document.getElementById('rcount').textContent=Object.keys(DECISIONS).length+' / '+cards.length+' resolved';
  }
  document.querySelectorAll('.rec-card .acc, .rec-card .rej').forEach(function(btn){
    btn.addEventListener('click',function(){
      const card=btn.closest('.rec-card'); const choice=btn.classList.contains('acc')?'accept':'reject';
      DECISIONS[card.dataset.id]=choice;
      card.classList.remove('accepted','rejected');
      card.classList.add(choice==='accept'?'accepted':'rejected');
      refresh();
    });
  });
  document.getElementById('export-btn').addEventListener('click',function(){
    const blob=new Blob([JSON.stringify(DECISIONS,null,2)],{type:'application/json'});
    const a=document.createElement('a'); a.href=URL.createObjectURL(blob);
    a.download='resolutions.json'; a.click();
  });
  refresh();
</script>"""


def build():
    kg_eval = _load_eval()          # imported with KG_USE_LLM=1 -> _LLM is set

    tabs, panels = [], []
    llm_R, rules_R = [], []
    for idx, tid in enumerate(all_tids()):
        print(f"  {tid}: pipeline (rules+LLM) ...", flush=True)
        case = load_case(tid, trace=True)          # LLM on via env
        print(f"  {tid}: metrics ...", flush=True)
        m_llm = kg_eval.evaluate_one(tid)          # with LLM (merge adjudicator)
        # RULES ONLY -- and the env var has to go with it. This module sets
        # KG_USE_LLM=1 at import, and `run_pipeline` re-acquires `default_client()`
        # whenever it is handed `llm=None` with that variable set. So clearing
        # `kg_eval._LLM` alone did NOT disable the LLM: the "rules only" column was a
        # second rules+LLM run, which is why every row of the comparison table below
        # used to show two identical numbers.
        kg_eval._LLM, saved = None, kg_eval._LLM
        os.environ.pop("KG_USE_LLM", None)
        try:
            m_rules = kg_eval.evaluate_one(tid)
        finally:
            os.environ["KG_USE_LLM"] = "1"
            kg_eval._LLM = saved
        llm_R.append(m_llm)
        rules_R.append(m_rules)

        agree, sugg, conflict, unverified = reconcile_items(case)
        active = " active" if idx == 0 else ""
        num = tid.split("_")[1]
        tabs.append(f"<button class='tab{active}' data-tab='p-{tid}'>Interview {num}"
                    f"<span class='t-sub'>{escape(TITLES.get(tid, ''))}</span></button>")
        panels.append(
            f"<div class='panel{active}' id='p-{tid}'>"
            f"<h2 style='font-size:18px;font-weight:600;margin:0 0 18px'>Interview {num} "
            f"&mdash; {escape(TITLES.get(tid, ''))}</h2>"
            f"{render.transcript_panel(case, m_llm)}"
            f"<div class='stages'>{_rec_section(tid, agree, sugg, conflict, unverified)}</div></div>")

    cmp = _cmp_table(_agg(rules_R), _agg(llm_R))
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Two-layer report &middot; rules &times; LLM</title>
<style>{render.CSS}{_EXTRA_CSS}</style></head>
<body><div class="wrap">
  <h1>Rules &times; LLM &mdash; two-layer report</h1>
  <p class="lede">Both the deterministic ruleset and the local LLM layer, run on all {len(tabs)} sample
     transcripts. Each tab shows the pipeline, the entity graph, and a two-layer review where
     suggestions and conflicts can be resolved. Metrics assume a perfect detector (upper bound).</p>
  <h2 style="font-size:16px;font-weight:600;margin:22px 0 8px">Performance: rules only vs rules + LLM</h2>
  {cmp}
  <div class="tabs">{''.join(tabs)}</div>
  {''.join(panels)}
  <div id="resolve-bar"><span id="rcount">0 resolved</span>
    <button id="export-btn">Export resolutions (JSON)</button></div>
</div>{_JS}</body></html>"""


def main():
    print("Building two-layer report (rules + LLM)...")
    OUT.write_text(build(), encoding="utf-8")
    print(f"Wrote {OUT}")
    try:
        if os.environ.get("KG_NO_OPEN") != "1":
            webbrowser.open(OUT.as_uri())
    except Exception:
        print("(open the file above in a browser)")


if __name__ == "__main__":
    main()
