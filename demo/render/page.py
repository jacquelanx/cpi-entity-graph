"""
Assembling the stage panels into one transcript walkthrough.

PURPOSE
    `transcript_panel` is the one function the report scripts call: it produces the
    metrics tiles, the stage stepper, and every stage panel in pipeline order, as
    one HTML string.

FIT
    The top of `demo/render/` -- it imports every `stages_*` module and is imported
    by `scripts/pipeline_report.py` and `scripts/llm_report.py` (via the package
    `__init__`).

HOW
    `_STEPS` and the call order inside `transcript_panel` are kept in the same
    order as the stages in `graph/pipeline.py`, so the page reads as a walkthrough
    of what actually happened.
"""

from __future__ import annotations

from html import escape
from evaluation.metrics import _acc
from .primitives import _pct
from .stages_cluster import stage_cluster, stage_coref, stage_detect, stage_relations
from .stages_graph import stage_artifact, stage_graph, stage_ledger
from .stages_people import stage_interviewee, stage_people
from .stages_world import stage_dates_ages, stage_identifiers, stage_places


# ---------------------------------------------------------------- metrics
def metrics_grid(R) -> str:
    """The row of headline metric tiles, from an `evaluation.scoring` result dict.

    Tile ORDER is editorial: the first two are the numbers that decide whether the
    next stage may run at all -- total privacy LEAKS (summed across people, places,
    dates and ages, since a leak is a leak wherever it happens) and the count of
    BLOCKING fields awaiting a human. Quality metrics follow.

    Colour classes carry the same judgment: zero leaks is green, any leak is red,
    and blocking fields are amber rather than red because they are the system
    working as designed.
    """
    c, r, g, rp = R["cluster"], R["rel"], R["gender"], R["replace"]
    d, a, l = R["dates"], R["ages"], R["locations"]
    o, sl = R.get("owner") or {}, R.get("second_line") or {}
    iv = R.get("interviewee") or {}
    dr, ar = R.get("date_redaction") or {}, R.get("age_redaction") or {}
    st, lp = R.get("person_subtype") or {}, R.get("loc_parent") or {}
    gn, sn = R.get("given_name") or {}, R.get("surname") or {}
    shifter = [R.get(k) or {} for k in ("shiftable", "approximate", "stated_with")]

    leaks = rp["leaks"] + l.get("rep_leaks", 0) + dr.get("leaks", 0) + ar.get("leaks", 0)
    n_blk = len(sl.get("blocking") or [])

    def tile(val, lab, sub="", cls=""):
        """One metric tile: a big value, a label, an optional sub-line, a colour class."""
        sub = f"<div class='m-sub'>{sub}</div>" if sub else ""
        return (f"<div class='m-tile'><div class='m-val {cls}'>{val}</div>"
                f"<div class='m-lab'>{lab}</div>{sub}</div>")

    # The name parts belong in this count: they are the two fields a surrogate
    # identity for the person being de-identified is actually minted from.
    _IV_FIELDS = ("identity", "given_name", "surname", "gender", "ethnicity", "dob")
    iv_ok = sum(1 for k in _IV_FIELDS if iv.get(k) == "correct")
    iv_tot = sum(1 for k in _IV_FIELDS if iv.get(k) and iv[k] != "n/a")

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
             "identity &middot; name &middot; gender &middot; ethnicity &middot; DOB"),
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
        # The three tiles below exist because the console report grew them and this
        # page is the front door the README points at. Leaving them off would have
        # kept showing the rosy half of the picture: place TYPING was previously
        # scored as "is there any type at all" and is now scored against the gold
        # type, the LOCATED_IN hierarchy was only ever counted, and person subtype
        # was in gold from the start with nothing reading it.
        tile(_pct(st.get("accuracy")), "person subtype",
             f"{st.get('wrong', 0)} wrong &middot; {st.get('missing', 0)} missing",
             "bad" if st.get("wrong") else ""),
        tile(_pct(_acc((l.get("type_ok") or 0) + (l.get("off_type_ok") or 0),
                       l.get("in_gaz", 0) + l.get("off_gaz", 0))),
             "place type accuracy",
             f"{(l.get('type_wrong') or 0) + (l.get('off_type_wrong') or 0)} wrong "
             f"&middot; vs gold type, not merely present"),
        tile(_pct(lp.get("recall")), "location hierarchy",
             f"{lp.get('tp', 0)}/{lp.get('scorable', 0)} parents &middot; "
             f"P {_pct(lp.get('precision'))}"),
        # The two things the NEXT stage is built out of, and neither had a number
        # before: a surrogate NAME is minted from the two name parts, and the date
        # shifter is driven by shiftable + approximate + the age/date pairing.
        tile(_pct(_acc(gn.get("correct", 0) + sn.get("correct", 0),
                       gn.get("total", 0) + sn.get("total", 0))),
             "name parts",
             f"{gn.get('wrong', 0) + sn.get('wrong', 0)} in the wrong half &middot; "
             "given / surname"),
        tile(_pct(_acc(sum(x.get("correct", 0) for x in shifter),
                       sum(x.get("total", 0) for x in shifter))),
             "date-shifter inputs",
             f"{sum(x.get('wrong', 0) for x in shifter)} wrong &middot; "
             "shiftable / approximate / stated_with"),
    ]
    return "<div class='metrics'>" + "".join(tiles) + "</div>"


# ---------- assembly ----------
_STEPS = ["Detect", "Cluster", "Coref", "Relations", "Interviewee", "People",
          "Places", "Dates & ages", "Identifiers", "Graph", "Ledger", "Artifact"]


def _stepper() -> str:
    """The numbered "Detect -> Cluster -> ..." breadcrumb above the stage panels."""
    chips = [f"<span class='st'><b>{i:02d}</b>{escape(s)}</span>"
             for i, s in enumerate(_STEPS, 1)]
    return "<div class='stepper'>" + "<span class='sep'>&rarr;</span>".join(chips) + "</div>"


def transcript_panel(case, metrics=None) -> str:
    """The complete walkthrough for one transcript, as one HTML string.

    `case` is a `demo.cases.load_case` dict; `metrics` is an optional
    `evaluation.scoring` result, which adds the tiles at the top. The stage calls
    below run in the same order as `graph/pipeline.py` and as `_STEPS`.
    """
    head = metrics_grid(metrics) if metrics else ""
    stages = (stage_detect(case) + stage_cluster(case) + stage_coref(case)
              + stage_relations(case) + stage_interviewee(case) + stage_people(case)
              + stage_places(case) + stage_dates_ages(case) + stage_identifiers(case)
              + stage_graph(case) + stage_ledger(case) + stage_artifact(case))
    return head + _stepper() + f"<div class='stages'>{stages}</div>"
