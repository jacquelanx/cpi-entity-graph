"""
Build ONE self-contained HTML dashboard covering all sample transcripts.
Run this dashboard with the following command:
./venv/bin/python3 scripts/dashboard.py
(Might take a minute to load)
"""

from __future__ import annotations
import importlib.util
import os
import sys
import webbrowser
from html import escape
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

from demo_utils import all_tids, load_case, REPO_ROOT
import render

OUT = REPO_ROOT / "tests" / "pipeline_report.html"

# short labels for the tabs
TITLES = {
    "interview_001": "Gulf Coast Vietnamese shrimping family",
    "interview_002": "Appalachian coal-mining family",
    "interview_003": "Chicago steel-mill family (short demo)",
}


"""Import scripts/eval.py so we can reuse its per-transcript scorer."""
def _load_eval():
    spec = importlib.util.spec_from_file_location("kg_eval", SCRIPTS / "eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build():
    kg_eval = _load_eval()
    tabs, panels = [], []
    tot_leaks = tot_over = 0
    for idx, tid in enumerate(all_tids()):
        print(f"  processing {tid} ...", flush=True)
        case = load_case(tid, trace=True)
        metrics = kg_eval.evaluate_one(tid)
        tot_leaks += metrics["replace"]["leaks"]
        tot_over += metrics["cluster"]["over_merges"]
        active = " active" if idx == 0 else ""
        num = tid.split("_")[1]
        tabs.append(
            f"<button class='tab{active}' data-tab='p-{tid}'>Interview {num}"
            f"<span class='t-sub'>{escape(TITLES.get(tid, ''))}</span></button>")
        panels.append(
            f"<div class='panel{active}' id='p-{tid}'>"
            f"<h2 style='font-size:18px;font-weight:600;margin:0 0 18px'>"
            f"Interview {num} &mdash; {escape(TITLES.get(tid, ''))}</h2>"
            f"{render.transcript_panel(case, metrics)}</div>")

    js = """<script>
      document.querySelectorAll('.tab').forEach(function(btn){
        btn.addEventListener('click', function(){
          document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
          document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
          btn.classList.add('active');
          document.getElementById(btn.dataset.tab).classList.add('active');
          window.scrollTo({top:0, behavior:'instant'});
        });
      });
    </script>"""

    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>De-identification pipeline &middot; report</title>
<style>{render.CSS}</style></head>
<body><div class="wrap">
  <h1>De-identification pipeline &mdash; walkthrough &amp; metrics</h1>
  <p class="lede"><b>Rules only.</b> Each transcript runs independently through all twelve stages,
     with the LLM second line switched OFF &mdash; this is the deterministic baseline. Run
     <code>scripts/llm_report.py</code> for the two-layer view. Every stage shows the resolved
     value <i>and</i> the decision behind it; stage 11 is the exhaustive ledger and stage 12 is the
     artifact the surrogate-generation stage receives. Metrics assume a perfect detector, so they
     are an upper bound on this graph stage; multiply by the detector's recall for end-to-end
     numbers.</p>
  <div class="tabs">{''.join(tabs)}</div>
  {''.join(panels)}
  <p class="foot">Across all {len(tabs)} transcripts the pipeline produced <b>{tot_leaks} privacy leaks</b> and
     <b>{tot_over} over-merges</b> (rules-only view). Uncertain cases (nickname aliases, shared first
     names, coref suggestions) are flagged for review rather than acted on silently.</p>
</div>{js}</body></html>"""


def main():
    print("Building dashboard (pipeline + coref)...")
    OUT.write_text(build(), encoding="utf-8")
    print(f"Wrote {OUT}")
    try:
        if os.environ.get("KG_NO_OPEN") != "1":
            webbrowser.open(OUT.as_uri())
    except Exception:
        print("(open the file above in a browser)")


if __name__ == "__main__":
    main()