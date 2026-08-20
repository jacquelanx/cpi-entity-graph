"""
Assembling the stage panels into one transcript walkthrough:
the metrics tiles at the top, the stepper, and `transcript_panel`.
"""

from __future__ import annotations

from html import escape
from .primitives import _pct
from .stages_cluster import stage_cluster, stage_coref, stage_detect, stage_relations
from .stages_graph import stage_artifact, stage_graph, stage_ledger
from .stages_people import stage_interviewee, stage_people
from .stages_world import stage_dates_ages, stage_identifiers, stage_places


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
