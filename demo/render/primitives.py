"""
Small shared formatting helpers and the palette/label tables.

Everything here is pure string formatting with no knowledge of a pipeline
stage; the stage modules build on it.
"""

from __future__ import annotations

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


def _chip(text, cls=""):
    return f"<span class='{('chip ' + cls).strip()}'>{text}</span>"


def _names_map(case) -> dict:
    names = {e.entity_id: _pname(e) for e in case["entities"]}
    iv = case["info"]["interviewee"]
    names[iv.entity_id] = _pname(iv) if iv.sorted_mentions else "the interviewee"
    return names
