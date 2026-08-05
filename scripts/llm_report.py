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
def reconcile_items(case):
    info = case["info"]
    agree, sugg, conflict = [], [], []

    for m in info.get("coref_merges", []):
        agree.append({"kind": "merge", "text": f"{m['merged']} → {m['kept']}",
                      "detail": "coref + LLM agree: same person, merged"})
    for f in info.get("coref_flags", []):
        conflict.append({"kind": "merge", "text": f"{f['a']} ↔ {f['b']}",
                         "detail": f["note"]})

    for e in case["entities"]:
        nm = e.sorted_mentions[0] if e.sorted_mentions else e.entity_id
        a = e.attributes
        if a.get("gender_confirmed"):
            agree.append({"kind": "gender", "text": nm,
                          "detail": f"gender {a.get('gender')} — rule & LLM agree"})
        if a.get("suggested_gender"):
            sugg.append({"kind": "gender", "text": nm,
                         "detail": f"LLM suggests gender {a['suggested_gender']} "
                                   f"(rule left it unset)"})
        if a.get("suggested_role"):
            sugg.append({"kind": "role", "text": nm,
                         "detail": f"LLM role: {a['suggested_role']}"})
        if a.get("candidate_public_figure"):
            sugg.append({"kind": "public", "text": nm,
                         "detail": f"LLM: maybe a public figure "
                                   f"({a['candidate_public_figure']}) — keep unredacted?"})
        if a.get("suggested_subtype"):
            sugg.append({"kind": "subtype", "text": nm,
                         "detail": f"LLM: likely {a['suggested_subtype']} "
                                   f"(rule left the subtype unset)"})
        if a.get("suggested_type"):
            d = f"LLM location type: {a['suggested_type']}"
            if a.get("suggested_parent"):
                d += f", in {a['suggested_parent']}"
            sugg.append({"kind": "location", "text": nm, "detail": d})
        if a.get("suggested_value") is not None:
            if e.category == "AGE":
                sugg.append({"kind": "age", "text": nm,
                             "detail": f"LLM age: {a['suggested_value']} "
                                       f"(rule could not parse it)"})
            elif e.category == "DATE_ANCHOR":
                d = f"LLM anchor date: {a['suggested_value']}"
                if a.get("suggested_event"):
                    d += f" ({a['suggested_event']})"
                sugg.append({"kind": "anchor", "text": nm, "detail": d})
            else:
                sugg.append({"kind": "date", "text": nm,
                             "detail": f"LLM date: {a['suggested_value']} "
                                       f"(rule could not resolve it)"})
        if a.get("suggested_kind"):
            sugg.append({"kind": "identifier", "text": nm,
                         "detail": f"LLM: this identifier looks like a "
                                   f"{a['suggested_kind']} (rule flagged it)"})
        if a.get("suggested_relation"):
            sr = a["suggested_relation"]
            sugg.append({"kind": "relation", "text": nm,
                         "detail": f"LLM: maybe '{sr['detail']}' with {sr['with']} "
                                   f"(proposed but not verifiable locally)"})
        for line in (e.review_reason or "").split("; "):
            if "conflicts with rule" in line:
                conflict.append({"kind": "gender", "text": nm, "detail": line})
    return agree, sugg, conflict


_KIND = 0


def _rec_section(tid, agree, sugg, conflict):
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

    def col(title, items, actionable, cls):
        body = "".join(card(i, actionable) for i in items) or "<p class='muted'>none</p>"
        return (f"<div class='rec-col {cls}'><h4>{title} "
                f"<span class='cnt'>{len(items)}</span></h4>{body}</div>")

    return ("<section class='stage wide'><div class='s-head'><span class='s-num'>R</span>"
            "<h3>Two-layer review &mdash; rules &times; LLM</h3></div>"
            "<p class='s-note'>Agreements are informational; suggestions and conflicts "
            "can be resolved (accept keeps the LLM's call, reject keeps the rules').</p>"
            "<div class='rec-cols'>"
            + col("Agreements", agree, False, "agree")
            + col("Suggestions", sugg, True, "sugg")
            + col("Conflicts", conflict, True, "conf")
            + "</div></section>")


# ---------- metrics aggregation + comparison ----------
def _agg(Rs):
    s = lambda k, f: sum(R[k][f] for R in Rs)
    tp, pred, gold = s("rel", "tp"), s("rel", "pred"), s("rel", "gold")
    return {
        "clus_recall": s("cluster", "exact") / max(1, s("cluster", "gold")),
        "over_merges": s("cluster", "over_merges"),
        "splits": s("cluster", "splits"),
        "rel_prec": tp / max(1, pred), "rel_recall": tp / max(1, gold),
        "gender_recall": s("gender", "correct") / max(1, s("gender", "total")),
        "leaks": s("replace", "leaks"),
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
        ("Privacy leaks", rules["leaks"], llm["leaks"]),
    ]
    trs = "".join(
        f"<tr><td>{escape(m)}</td><td>{r}</td><td class='hl'>{l}</td></tr>"
        for m, r, l in rows)
    return ("<table class='cmp'><thead><tr><th>metric</th><th>rules only</th>"
            "<th>rules + LLM</th></tr></thead><tbody>" + trs + "</tbody></table>")


_EXTRA_CSS = """
.cmp{border-collapse:collapse;width:100%;max-width:560px;margin:6px 0 30px;font-size:14px;}
.cmp th,.cmp td{border-bottom:1px solid var(--line);padding:8px 12px;text-align:left;}
.cmp th{font-size:12px;color:var(--muted);font-weight:600;}
.cmp td.hl{font-weight:600;}
.rec-cols{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;}
.rec-col h4{font-size:13px;margin:0 0 10px;font-weight:600;display:flex;gap:8px;align-items:center;}
.rec-col .cnt{font-size:11px;color:var(--muted);background:var(--panel);border-radius:20px;padding:1px 8px;}
.rec-col.agree h4{color:#2f6b3b;} .rec-col.sugg h4{color:#6A4310;} .rec-col.conf h4{color:#7C2222;}
.rec-card{background:#fff;border:1px solid var(--line);border-radius:10px;padding:10px 12px;margin-bottom:8px;}
.rec-col.agree .rec-card{border-left:3px solid #4a9d5b;}
.rec-col.sugg .rec-card{border-left:3px solid #c99a4a;}
.rec-col.conf .rec-card{border-left:3px solid #c0574f;}
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
        kg_eval._LLM, saved = None, kg_eval._LLM
        m_rules = kg_eval.evaluate_one(tid)        # rules only
        kg_eval._LLM = saved
        llm_R.append(m_llm)
        rules_R.append(m_rules)

        agree, sugg, conflict = reconcile_items(case)
        active = " active" if idx == 0 else ""
        num = tid.split("_")[1]
        tabs.append(f"<button class='tab{active}' data-tab='p-{tid}'>Interview {num}"
                    f"<span class='t-sub'>{escape(TITLES.get(tid, ''))}</span></button>")
        panels.append(
            f"<div class='panel{active}' id='p-{tid}'>"
            f"<h2 style='font-size:18px;font-weight:600;margin:0 0 18px'>Interview {num} "
            f"&mdash; {escape(TITLES.get(tid, ''))}</h2>"
            f"{render.transcript_panel(case, m_llm)}"
            f"<div class='stages'>{_rec_section(tid, agree, sugg, conflict)}</div></div>")

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
