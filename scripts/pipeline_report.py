"""
Build the RULES-ONLY HTML dashboard covering all sample transcripts.

PURPOSE
    One self-contained HTML file with a tab per sample transcript, each showing the
    full twelve-stage walkthrough plus its metrics -- with the LLM second line
    switched OFF. This is the deterministic baseline.

    ./venv/bin/python3 scripts/pipeline_report.py
    (Might take a minute to load)

FIT
    A runnable entry point over `demo/cases.py` (to run the pipeline),
    `demo/render/` (to build the HTML) and `evaluation/scoring.py` (for the
    metrics tiles). `scripts/llm_report.py` is the two-layer counterpart and
    renders the identical stages with the second line on.

HOW
    Everything -- CSS, markup and the tab-switching JavaScript -- is inlined into a
    single file, so the report can be moved or emailed without losing assets.
"""

from __future__ import annotations
import os
import sys
import webbrowser
from html import escape
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS.parent))

from demo.cases import all_tids, load_case, REPO_ROOT
from demo import render

OUT = REPO_ROOT / "reports" / "pipeline_report.html"

# short labels for the tabs
TITLES = {
    "interview_001": "Gulf Coast Vietnamese shrimping family",
    "interview_002": "Appalachian coal-mining family",
    "interview_003": "Chicago steel-mill family (short demo)",
}


def _load_eval():
    """The per-transcript scorer, imported lazily.

    `evaluation.config` reads `KG_USE_LLM` and configures logging AT IMPORT TIME,
    so importing it must happen after any environment change -- which is why this
    is a function rather than a module-level import. (This script leaves the flag
    unset; `scripts/llm_report.py` sets it, and shares the pattern.)
    """
    from evaluation import scoring
    return scoring


def build():
    """Render the whole dashboard as one HTML string.

    Loops the sample transcripts, running the pipeline and the scorer for each,
    and accumulates two headline totals -- privacy leaks and over-merges -- for the
    footer. The first tab is marked active so the page opens on a transcript
    rather than on nothing.
    """
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
    """Build the report, write it to `reports/`, and open it in a browser.

    Set `KG_NO_OPEN=1` to skip the browser (useful in CI or over SSH); a failure to
    open is caught and reported rather than raised, since the file is already
    written by then.
    """
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